use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};

use melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use melior::dialect::llvm::attributes::{Linkage, linkage};
use melior::dialect::llvm::r#type as llvm_ty;
use melior::dialect::llvm::{AllocaOptions, LoadStoreOptions};
use melior::dialect::{DialectRegistry, cf, func, llvm};
use melior::ir::attribute::{
    Attribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
    TypeAttribute,
};
use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{
    Block, BlockRef, Identifier, Location, Module, Region, RegionLike, Type, Value, ValueLike,
};
use melior::pass::PassManager;
use melior::utility::{
    parse_pass_pipeline, register_all_dialects, register_all_llvm_translations, register_all_passes,
};
use melior::{Context, ExecutionEngine};
use novar_diagnostics::Diagnostic;
use novar_hir::hir::{self, BinOp, ExprKind, Lit, LocalId, UnOp};
use novar_hir::{Prim, Quantization, Ty};

mod tensor;

type Span = std::ops::Range<usize>;

/// A context with every dialect, pass and LLVM translation registered.
pub fn context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let ctx = Context::new();
    ctx.append_dialect_registry(&registry);
    ctx.load_all_available_dialects();
    register_all_llvm_translations(&ctx);
    register_all_passes();
    ctx
}

fn is_unsigned(ty: &Ty) -> bool {
    matches!(ty, Ty::Prim(Prim::U8 | Prim::U16 | Prim::U32 | Prim::U64))
}

fn is_float(ty: &Ty) -> bool {
    matches!(ty, Ty::Prim(p) if p.is_float())
}

/// libc functions the generated code may call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Libc {
    Malloc,
    Memcpy,
    Memcmp,
    Write,
    Abort,
}

impl Libc {
    fn name(self) -> &'static str {
        match self {
            Libc::Malloc => "malloc",
            Libc::Memcpy => "memcpy",
            Libc::Memcmp => "memcmp",
            Libc::Write => "write",
            Libc::Abort => "abort",
        }
    }
}

/// Module-wide state shared by all functions: string constants and libc declarations.
#[derive(Default)]
struct ModuleState {
    strings: RefCell<Vec<String>>,
    libc: RefCell<BTreeSet<Libc>>,
    /// Whether the integer-to-text helper is needed.
    itoa: Cell<bool>,
}

impl ModuleState {
    fn intern(&self, text: &str) -> usize {
        let mut strings = self.strings.borrow_mut();
        match strings.iter().position(|s| s == text) {
            Some(i) => i,
            None => {
                strings.push(text.to_string());
                strings.len() - 1
            }
        }
    }

    fn need(&self, f: Libc) {
        self.libc.borrow_mut().insert(f);
    }
}

struct Cx<'c, 'p> {
    ctx: &'c Context,
    program: &'p hir::Program,
    state: &'p ModuleState,
    quant: Quantization,
}

impl<'c, 'p> Cx<'c, 'p> {
    fn i(&self, bits: u32) -> Type<'c> {
        IntegerType::new(self.ctx, bits).into()
    }

    fn ptr(&self) -> Type<'c> {
        llvm_ty::pointer(self.ctx, 0)
    }

    fn struct_def(&self, name: &str) -> Option<&'p hir::StructDef> {
        self.program.structs.iter().find(|s| s.name == name)
    }

    /// The scalar MLIR types a value of `ty` is flattened into: structs become their fields in
    /// declaration order, strings become `(ptr, len)`, unit becomes nothing.
    fn flat(&self, ty: &Ty) -> Result<Vec<Type<'c>>, String> {
        self.flat_at(ty, 0)
    }

    fn flat_at(&self, ty: &Ty, depth: usize) -> Result<Vec<Type<'c>>, String> {
        Ok(match ty {
            Ty::Unit | Ty::Never => vec![],
            Ty::Prim(p) => vec![match p {
                Prim::I8 | Prim::U8 => self.i(8),
                Prim::I16 | Prim::U16 => self.i(16),
                Prim::I32 | Prim::U32 => self.i(32),
                Prim::I64 | Prim::U64 => self.i(64),
                Prim::F32 => Type::float32(self.ctx),
                Prim::F64 => Type::float64(self.ctx),
                Prim::Bool => self.i(1),
                Prim::Str => return Ok(vec![self.ptr(), self.i(64)]),
            }],
            Ty::Struct(name) => {
                if depth > 16 {
                    return Err(format!("recursive struct `{name}`"));
                }
                let def = self
                    .struct_def(name)
                    .ok_or_else(|| format!("unknown struct `{name}`"))?;
                let mut out = Vec::new();
                for (_, fty) in &def.fields {
                    out.extend(self.flat_at(fty, depth + 1)?);
                }
                out
            }
            Ty::Tensor { .. } if tensor::tensor_info(ty).is_some() => vec![self.ptr()],
            Ty::Tensor { .. } if tensor::tensor_info(ty).is_some() => vec![self.ptr()],
            Ty::Vec(elem) if matches!(&**elem, Ty::Prim(p) if *p != Prim::Str) => {
                vec![self.ptr(), self.i(64)]
            }
            other => return Err(format!("`{other}`")),
        })
    }

    /// Size in bytes of one element of a native `Vec<elem>`.
    fn elem_bytes(&self, elem: &Ty) -> i64 {
        match elem {
            Ty::Prim(Prim::Bool | Prim::I8 | Prim::U8) => 1,
            Ty::Prim(Prim::I16 | Prim::U16) => 2,
            Ty::Prim(Prim::I32 | Prim::U32 | Prim::F32) => 4,
            _ => 8,
        }
    }
}

/// Lowers every function of `program` to a verified MLIR module.
pub fn lower<'c>(ctx: &'c Context, program: &hir::Program) -> Result<Module<'c>, Vec<Diagnostic>> {
    lower_with(ctx, program, Quantization::None)
}

/// Like [`lower`], with `f32` matrix products at the given precision.
pub fn lower_with<'c>(
    ctx: &'c Context,
    program: &hir::Program,
    quant: Quantization,
) -> Result<Module<'c>, Vec<Diagnostic>> {
    let module = Module::new(Location::unknown(ctx));
    let state = ModuleState::default();
    let cx = Cx {
        ctx,
        program,
        state: &state,
        quant,
    };
    let mut diags = Vec::new();
    for f in &program.fns {
        match lower_fn(&cx, f) {
            Ok(op) => {
                module.body().append_operation(op);
            }
            Err(d) => diags.push(d),
        }
    }
    if !diags.is_empty() {
        return Err(diags);
    }
    declare_support(&cx, &module);
    if !module.as_operation().verify() {
        return Err(vec![
            Diagnostic::error(0..0, "internal error: generated MLIR failed verification")
                .with_note(module.as_operation().to_string()),
        ]);
    }
    Ok(module)
}

/// Emits the string constants and libc declarations collected while lowering.
fn declare_support<'c>(cx: &Cx<'c, '_>, module: &Module<'c>) {
    let (ctx, loc) = (cx.ctx, Location::unknown(cx.ctx));
    if cx.state.itoa.get() {
        module.body().append_operation(build_itoa(cx));
    }
    for (i, text) in cx.state.strings.borrow().iter().enumerate() {
        // NUL-terminated so the data is also a valid C string.
        let mut bytes = text.clone();
        bytes.push('\0');
        let global = OperationBuilder::new("llvm.mlir.global", loc)
            .add_attributes(&[
                (
                    Identifier::new(ctx, "sym_name"),
                    StringAttribute::new(ctx, &format!(".str.{i}")).into(),
                ),
                (
                    Identifier::new(ctx, "global_type"),
                    TypeAttribute::new(llvm_ty::array(cx.i(8), bytes.len() as u32)).into(),
                ),
                (
                    Identifier::new(ctx, "linkage"),
                    linkage(ctx, Linkage::Internal),
                ),
                (Identifier::new(ctx, "constant"), Attribute::unit(ctx)),
                (
                    Identifier::new(ctx, "value"),
                    StringAttribute::new(ctx, &bytes).into(),
                ),
            ])
            .add_regions([Region::new()])
            .build()
            .expect("valid global");
        module.body().append_operation(global);
    }
    for f in cx.state.libc.borrow().iter() {
        let (params, ret) = match f {
            Libc::Malloc => (vec![cx.i(64)], cx.ptr()),
            Libc::Memcpy => (vec![cx.ptr(), cx.ptr(), cx.i(64)], cx.ptr()),
            Libc::Memcmp => (vec![cx.ptr(), cx.ptr(), cx.i(64)], cx.i(32)),
            Libc::Write => (vec![cx.i(32), cx.ptr(), cx.i(64)], cx.i(64)),
            Libc::Abort => (vec![], cx.i(0)),
        };
        let results = if *f == Libc::Abort { vec![] } else { vec![ret] };
        module.body().append_operation(func::func(
            ctx,
            StringAttribute::new(ctx, f.name()),
            TypeAttribute::new(FunctionType::new(ctx, &params, &results).into()),
            Region::new(),
            &[(
                Identifier::new(ctx, "sym_visibility"),
                StringAttribute::new(ctx, "private").into(),
            )],
            loc,
        ));
    }
}

/// `__novar_itoa(value: i64, signed: i1) -> (ptr, len)`: decimal text in a fresh 24-byte buffer.
fn build_itoa<'c>(cx: &Cx<'c, '_>) -> melior::ir::Operation<'c> {
    let (ctx, loc) = (cx.ctx, Location::unknown(cx.ctx));
    let (i64t, i1t, i8t, ptr) = (cx.i(64), cx.i(1), cx.i(8), cx.ptr());
    cx.state.need(Libc::Malloc);

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(i64t, loc), (i1t, loc)]));
    let digits = region.append_block(Block::new(&[(i64t, loc), (i64t, loc)]));
    let fin = region.append_block(Block::new(&[(i64t, loc)]));
    macro_rules! val {
        ($b:expr, $op:expr) => {
            Value::from($b.append_operation($op).result(0).expect("one result"))
        };
    }
    macro_rules! c64 {
        ($b:expr, $v:expr) => {
            val!(
                $b,
                arith::constant(ctx, IntegerAttribute::new(i64t, $v).into(), loc)
            )
        };
    }
    macro_rules! c8 {
        ($b:expr, $v:expr) => {
            val!(
                $b,
                arith::constant(ctx, IntegerAttribute::new(i8t, $v).into(), loc)
            )
        };
    }
    macro_rules! at {
        ($b:expr, $base:expr, $idx:expr) => {
            val!(
                $b,
                llvm::get_element_ptr_dynamic(ctx, $base, &[$idx], i8t, ptr, loc)
            )
        };
    }

    // entry: split sign and magnitude, allocate the buffer.
    let (v, signed) = (
        entry.argument(0).unwrap().into(),
        entry.argument(1).unwrap().into(),
    );
    let zero = c64!(entry, 0);
    let lt = val!(entry, arith::cmpi(ctx, CmpiPredicate::Slt, v, zero, loc));
    let neg = val!(entry, arith::andi(lt, signed, loc));
    let negated = val!(entry, arith::subi(zero, v, loc));
    let mag = val!(entry, arith::select(neg, negated, v, loc));
    let size = c64!(entry, 24);
    let buf = val!(
        entry,
        func::call(
            ctx,
            FlatSymbolRefAttribute::new(ctx, "malloc"),
            &[size],
            &[ptr],
            loc
        )
    );
    entry.append_operation(cf::br(&digits, &[size, mag], loc));

    // digits: write one digit backwards per iteration until the magnitude is exhausted.
    let (pos, m) = (
        digits.argument(0).unwrap().into(),
        digits.argument(1).unwrap().into(),
    );
    let ten = c64!(digits, 10);
    let q = val!(digits, arith::divui(m, ten, loc));
    let r = val!(digits, arith::remui(m, ten, loc));
    let d8 = val!(digits, arith::trunci(r, i8t, loc));
    let ch = val!(digits, arith::addi(d8, c8!(digits, 48), loc));
    let pos1 = val!(digits, arith::subi(pos, c64!(digits, 1), loc));
    let slot = at!(digits, buf, pos1);
    digits.append_operation(llvm::store(ctx, ch, slot, loc, LoadStoreOptions::new()));
    let zero_q = c64!(digits, 0);
    let done = val!(digits, arith::cmpi(ctx, CmpiPredicate::Eq, q, zero_q, loc));
    digits.append_operation(cf::cond_br(
        ctx,
        done,
        &fin,
        &digits,
        &[pos1],
        &[pos1, q],
        loc,
    ));

    // fin: prepend '-' for negative numbers (the byte is written unconditionally but only
    // included in the result when `neg`).
    let pos = fin.argument(0).unwrap().into();
    let prev = val!(fin, arith::subi(pos, c64!(fin, 1), loc));
    let minus_slot = at!(fin, buf, prev);
    fin.append_operation(llvm::store(
        ctx,
        c8!(fin, 45),
        minus_slot,
        loc,
        LoadStoreOptions::new(),
    ));
    let start_idx = val!(fin, arith::select(neg, prev, pos, loc));
    let start = at!(fin, buf, start_idx);
    let len = val!(fin, arith::subi(size, start_idx, loc));
    fin.append_operation(func::r#return(&[start, len], loc));

    func::func(
        ctx,
        StringAttribute::new(ctx, "__novar_itoa"),
        TypeAttribute::new(FunctionType::new(ctx, &[i64t, i1t], &[ptr, i64t]).into()),
        region,
        &[(
            Identifier::new(ctx, "sym_visibility"),
            StringAttribute::new(ctx, "private").into(),
        )],
        loc,
    )
}

fn lower_fn<'c>(cx: &Cx<'c, '_>, f: &hir::FnDef) -> Result<melior::ir::Operation<'c>, Diagnostic> {
    let ctx = cx.ctx;
    let unsupported = |what: String| {
        Diagnostic::error(
            f.span.clone(),
            format!("the native backend does not support {what} yet"),
        )
        .with_note(format!("in function `{}`", f.name))
    };
    let loc = Location::unknown(ctx);

    let mut param_tys = Vec::new();
    let mut spans = Vec::new(); // per source parameter: how many flat values it occupies
    for p in &f.params {
        let flat = cx
            .flat(&p.ty)
            .map_err(|w| unsupported(format!("parameters of type {w}")))?;
        spans.push(flat.len());
        param_tys.extend(flat);
    }
    let ret_tys = cx
        .flat(&f.ret)
        .map_err(|w| unsupported(format!("return type {w}")))?;

    let region = Region::new();
    {
        let entry = region.append_block(Block::new(
            &param_tys.iter().map(|t| (*t, loc)).collect::<Vec<_>>(),
        ));
        let mut env: Env = HashMap::new();
        let mut at = 0;
        for (p, n) in f.params.iter().zip(&spans) {
            let vals = (at..at + n)
                .map(|i| entry.argument(i).expect("entry argument").into())
                .collect();
            env.insert(p.id, Binding::Vals(vals));
            at += n;
        }
        let mut em = Emitter {
            cx,
            region: &region,
            entry,
            func: f,
            loc,
            loops: Vec::new(),
        };
        for p in f.params.iter().filter(|p| p.mutable) {
            if let Some(Binding::Vals(vals)) = env.get(&p.id).cloned() {
                let slots = em.new_slots(&vals);
                em.store_slots(entry, &slots, &vals);
                env.insert(p.id, Binding::Slots(slots));
            }
        }
        let (end, flow) = em.block(&f.body, entry, &mut env)?;
        if let Flow::Value(v) = flow {
            let rets = if ret_tys.is_empty() { vec![] } else { v };
            end.append_operation(func::r#return(&rets, loc));
        }
    }

    // The C interface wrapper (used by the JIT) only supports scalar signatures.
    let scalar = |tys: &[Type]| tys.len() <= 1;
    let scalar_sig = f
        .params
        .iter()
        .all(|p| matches!(p.ty, Ty::Prim(q) if q != Prim::Str))
        && matches!(f.ret, Ty::Unit | Ty::Prim(_))
        && f.ret != Ty::Prim(Prim::Str)
        && scalar(&ret_tys);
    let attrs: Vec<(Identifier, Attribute)> = if scalar_sig {
        vec![(
            Identifier::new(ctx, "llvm.emit_c_interface"),
            Attribute::unit(ctx),
        )]
    } else {
        vec![]
    };

    let fn_ty = FunctionType::new(ctx, &param_tys, &ret_tys);
    Ok(func::func(
        ctx,
        StringAttribute::new(ctx, &f.name),
        TypeAttribute::new(fn_ty.into()),
        region,
        &attrs,
        loc,
    ))
}

/// What evaluating an expression left behind.
enum Flow<'c, 'b> {
    /// Fell through, producing the flattened value (empty for unit).
    Value(Vec<Value<'c, 'b>>),
    /// Control left via `return`; the current block is already terminated.
    Diverged,
}

/// How a local is stored: immutable locals are SSA values, `let mut` locals live in stack slots.
#[derive(Clone)]
enum Binding<'c, 'b> {
    Vals(Vec<Value<'c, 'b>>),
    Slots(Vec<(Value<'c, 'b>, Type<'c>)>),
}

type Env<'c, 'b> = HashMap<LocalId, Binding<'c, 'b>>;
type Out<'c, 'b> = Result<(BlockRef<'c, 'b>, Flow<'c, 'b>), Diagnostic>;

struct Emitter<'c, 'b, 'p> {
    cx: &'p Cx<'c, 'p>,
    region: &'b Region<'c>,
    entry: BlockRef<'c, 'b>,
    func: &'p hir::FnDef,
    loc: Location<'c>,
    /// Enclosing loops, innermost last: where `continue` and `break` jump to.
    loops: Vec<(BlockRef<'c, 'b>, BlockRef<'c, 'b>)>,
}

impl<'c, 'b, 'p> Emitter<'c, 'b, 'p> {
    fn unsupported(&self, what: &str, span: &Span) -> Diagnostic {
        Diagnostic::error(
            span.clone(),
            format!("the native backend does not support {what} yet"),
        )
        .with_note(format!("in function `{}`", self.func.name))
    }

    fn flat(&self, ty: &Ty, span: &Span) -> Result<Vec<Type<'c>>, Diagnostic> {
        self.cx
            .flat(ty)
            .map_err(|w| self.unsupported(&format!("values of type {w}"), span))
    }

    fn block(
        &mut self,
        b: &hir::Block,
        mut cur: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        for s in &b.stmts {
            match s {
                hir::Stmt::Let { local, init } => {
                    let (next, flow) = self.expr(init, cur, env)?;
                    cur = next;
                    match flow {
                        Flow::Diverged => return Ok((cur, Flow::Diverged)),
                        Flow::Value(v) if local.mutable => {
                            let slots = self.new_slots(&v);
                            self.store_slots(cur, &slots, &v);
                            env.insert(local.id, Binding::Slots(slots));
                        }
                        Flow::Value(v) => {
                            env.insert(local.id, Binding::Vals(v));
                        }
                    }
                }
                hir::Stmt::Expr(e) => {
                    let (next, flow) = self.expr(e, cur, env)?;
                    cur = next;
                    if matches!(flow, Flow::Diverged) {
                        return Ok((cur, Flow::Diverged));
                    }
                }
            }
        }
        match &b.tail {
            Some(t) => self.expr(t, cur, env),
            None => Ok((cur, Flow::Value(vec![]))),
        }
    }

    fn push(&self, blk: BlockRef<'c, 'b>, op: melior::ir::Operation<'c>) -> Value<'c, 'b> {
        blk.append_operation(op)
            .result(0)
            .expect("one result")
            .into()
    }

    fn int_const(&self, blk: BlockRef<'c, 'b>, ty: Type<'c>, v: i64) -> Value<'c, 'b> {
        self.push(
            blk,
            arith::constant(self.cx.ctx, IntegerAttribute::new(ty, v).into(), self.loc),
        )
    }

    fn new_block(&self, args: &[Type<'c>]) -> BlockRef<'c, 'b> {
        self.region.append_block(Block::new(
            &args.iter().map(|t| (*t, self.loc)).collect::<Vec<_>>(),
        ))
    }

    fn call_libc(
        &self,
        blk: BlockRef<'c, 'b>,
        f: Libc,
        args: &[Value<'c, 'b>],
        ret: Type<'c>,
    ) -> Value<'c, 'b> {
        self.cx.state.need(f);
        let op = func::call(
            self.cx.ctx,
            FlatSymbolRefAttribute::new(self.cx.ctx, f.name()),
            args,
            &[ret],
            self.loc,
        );
        self.push(blk, op)
    }

    /// `(ptr, len)` of the interned constant string `text`.
    fn str_const(&self, blk: BlockRef<'c, 'b>, text: &str) -> Vec<Value<'c, 'b>> {
        let id = self.cx.state.intern(text);
        let addr = OperationBuilder::new("llvm.mlir.addressof", self.loc)
            .add_attributes(&[(
                Identifier::new(self.cx.ctx, "global_name"),
                FlatSymbolRefAttribute::new(self.cx.ctx, &format!(".str.{id}")).into(),
            )])
            .add_results(&[self.cx.ptr()])
            .build()
            .expect("valid addressof");
        let ptr = self.push(blk, addr);
        let len = self.int_const(blk, self.cx.i(64), text.len() as i64);
        vec![ptr, len]
    }

    /// Allocates one stack slot per value, at the top of the entry block (never inside loops).
    fn new_slots(&self, vals: &[Value<'c, 'b>]) -> Vec<(Value<'c, 'b>, Type<'c>)> {
        let ctx = self.cx.ctx;
        vals.iter()
            .map(|v| {
                let ty = v.r#type();
                let one = self.entry.insert_operation(
                    0,
                    arith::constant(
                        ctx,
                        IntegerAttribute::new(self.cx.i(64), 1).into(),
                        self.loc,
                    ),
                );
                let one: Value = one.result(0).expect("constant").into();
                let alloca = self.entry.insert_operation(
                    1,
                    llvm::alloca(
                        ctx,
                        one,
                        self.cx.ptr(),
                        self.loc,
                        AllocaOptions::new().elem_type(Some(TypeAttribute::new(ty))),
                    ),
                );
                (alloca.result(0).expect("alloca").into(), ty)
            })
            .collect()
    }

    fn store_slots(
        &self,
        blk: BlockRef<'c, 'b>,
        slots: &[(Value<'c, 'b>, Type<'c>)],
        vals: &[Value<'c, 'b>],
    ) {
        for ((slot, _), v) in slots.iter().zip(vals) {
            blk.append_operation(llvm::store(
                self.cx.ctx,
                *v,
                *slot,
                self.loc,
                LoadStoreOptions::new(),
            ));
        }
    }

    fn widen_index(&self, blk: BlockRef<'c, 'b>, v: Value<'c, 'b>, ty: &Ty) -> Value<'c, 'b> {
        let i64_ty = self.cx.i(64);
        if v.r#type() == i64_ty {
            v
        } else if is_unsigned(ty) {
            self.push(blk, arith::extui(v, i64_ty, self.loc))
        } else {
            self.push(blk, arith::extsi(v, i64_ty, self.loc))
        }
    }

    fn elem_ptr(
        &self,
        blk: BlockRef<'c, 'b>,
        base: Value<'c, 'b>,
        idx: Value<'c, 'b>,
        elem: Type<'c>,
    ) -> Value<'c, 'b> {
        self.push(
            blk,
            llvm::get_element_ptr_dynamic(self.cx.ctx, base, &[idx], elem, self.cx.ptr(), self.loc),
        )
    }

    /// Terminates `blk` with a diagnostic message on stderr followed by `abort()`.
    fn trap(&self, blk: BlockRef<'c, 'b>, message: &str) {
        let msg = self.str_const(blk, message);
        let fd = self.int_const(blk, self.cx.i(32), 2);
        self.call_libc(blk, Libc::Write, &[fd, msg[0], msg[1]], self.cx.i(64));
        self.cx.state.need(Libc::Abort);
        blk.append_operation(func::call(
            self.cx.ctx,
            FlatSymbolRefAttribute::new(self.cx.ctx, "abort"),
            &[],
            &[],
            self.loc,
        ));
        blk.append_operation(llvm::unreachable(self.loc));
    }

    fn assign(
        &mut self,
        e: &hir::Expr,
        place: &hir::Place,
        value: &hir::Expr,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let ctx = self.cx.ctx;
        // Order matches the interpreter: the value, then the index expressions left to right.
        let (mut cur, flow) = self.expr(value, blk, env)?;
        let Flow::Value(v) = flow else {
            return Ok((cur, flow));
        };
        let mut indices = Vec::new();
        for pr in &place.projs {
            if let hir::Proj::Index(ix) = pr {
                let (next, xf) = self.expr(ix, cur, env)?;
                cur = next;
                let Flow::Value(x) = xf else {
                    return Ok((cur, xf));
                };
                indices.push(self.widen_index(cur, x[0], &ix.ty));
            }
        }
        let Some(Binding::Slots(slots)) = env.get(&place.local).cloned() else {
            return Err(self.unsupported("assignment to a binding without storage", &e.span));
        };

        /// Where the store lands: a run of the local's slots, or one vector element.
        enum Target<'c, 'b> {
            Slots(usize, usize),
            Elem(Value<'c, 'b>),
        }
        let mut target = Target::Slots(0, slots.len());
        let mut ty = place.root_ty.clone();
        let mut next_index = indices.iter();
        for pr in &place.projs {
            let Target::Slots(start, _) = target else {
                return Err(self.unsupported("assignment through nested vectors", &e.span));
            };
            match pr {
                hir::Proj::Field { name, struct_name } => {
                    let def = self
                        .cx
                        .struct_def(struct_name)
                        .ok_or_else(|| self.unsupported("unknown structs", &e.span))?;
                    let mut offset = 0;
                    let mut found = None;
                    for (fname, fty) in &def.fields {
                        let n = self.flat(fty, &e.span)?.len();
                        if fname == name {
                            found = Some((n, fty.clone()));
                            break;
                        }
                        offset += n;
                    }
                    let (n, fty) =
                        found.ok_or_else(|| self.unsupported("unknown fields", &e.span))?;
                    target = Target::Slots(start + offset, start + offset + n);
                    ty = fty;
                }
                hir::Proj::Index(_) => {
                    let Ty::Vec(elem_ty) = ty.clone() else {
                        return Err(self.unsupported("indexing non-vectors", &e.span));
                    };
                    self.flat(&ty, &e.span)?;
                    let elem = self.flat(&elem_ty, &e.span)?[0];
                    let load = |this: &Self, i: usize, t: Type<'c>| {
                        let (slot, _) = slots[start + i];
                        this.push(
                            cur,
                            llvm::load(ctx, slot, t, this.loc, LoadStoreOptions::new()),
                        )
                    };
                    let ptr = load(self, 0, self.cx.ptr());
                    let len = load(self, 1, self.cx.i(64));
                    let idx = *next_index.next().expect("one index per index step");
                    let in_bounds = self.push(
                        cur,
                        arith::cmpi(ctx, CmpiPredicate::Ult, idx, len, self.loc),
                    );
                    let ok_blk = self.new_block(&[]);
                    let fail = self.new_block(&[]);
                    cur.append_operation(cf::cond_br(
                        ctx,
                        in_bounds,
                        &ok_blk,
                        &fail,
                        &[],
                        &[],
                        self.loc,
                    ));
                    self.trap(fail, "index out of bounds\n");
                    cur = ok_blk;
                    target = Target::Elem(self.elem_ptr(cur, ptr, idx, elem));
                    ty = *elem_ty;
                }
            }
        }
        match target {
            Target::Slots(from, to) => self.store_slots(cur, &slots[from..to], &v),
            Target::Elem(ptr) => {
                cur.append_operation(llvm::store(
                    ctx,
                    v[0],
                    ptr,
                    self.loc,
                    LoadStoreOptions::new(),
                ));
            }
        }
        Ok((cur, Flow::Value(vec![])))
    }

    fn while_loop(
        &mut self,
        cond: &hir::Expr,
        body: &hir::Block,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let header = self.new_block(&[]);
        let body_blk = self.new_block(&[]);
        let exit = self.new_block(&[]);
        blk.append_operation(cf::br(&header, &[], self.loc));
        let (hend, cflow) = self.expr(cond, header, env)?;
        let Flow::Value(c) = cflow else {
            return Ok((hend, cflow));
        };
        hend.append_operation(cf::cond_br(
            self.cx.ctx,
            c[0],
            &body_blk,
            &exit,
            &[],
            &[],
            self.loc,
        ));
        self.loops.push((header, exit));
        let result = self.block(body, body_blk, env);
        self.loops.pop();
        let (bend, bflow) = result?;
        if let Flow::Value(_) = bflow {
            bend.append_operation(cf::br(&header, &[], self.loc));
        }
        Ok((exit, Flow::Value(vec![])))
    }

    fn for_loop(
        &mut self,
        var: &hir::Local,
        iter: &hir::ForIter,
        body: &hir::Block,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let ctx = self.cx.ctx;
        match iter {
            hir::ForIter::Range {
                start,
                end,
                inclusive,
            } => {
                let (cur, sf) = self.expr(start, blk, env)?;
                let Flow::Value(s) = sf else {
                    return Ok((cur, sf));
                };
                let (cur, ef) = self.expr(end, cur, env)?;
                let Flow::Value(en) = ef else {
                    return Ok((cur, ef));
                };
                let (s, en) = (s[0], en[0]);
                let unsigned = is_unsigned(&var.ty);
                let int_ty = s.r#type();

                let header = self.new_block(&[int_ty]);
                let body_blk = self.new_block(&[]);
                let exit = self.new_block(&[]);
                cur.append_operation(cf::br(&header, &[s], self.loc));

                let i: Value = header.argument(0).expect("loop variable").into();
                let pred = match (inclusive, unsigned) {
                    (false, false) => CmpiPredicate::Slt,
                    (false, true) => CmpiPredicate::Ult,
                    (true, false) => CmpiPredicate::Sle,
                    (true, true) => CmpiPredicate::Ule,
                };
                let go = self.push(header, arith::cmpi(ctx, pred, i, en, self.loc));
                header.append_operation(cf::cond_br(ctx, go, &body_blk, &exit, &[], &[], self.loc));

                env.insert(var.id, Binding::Vals(vec![i]));
                // Both the end of the body and `continue` arrive here to advance the counter.
                let advance = self.new_block(&[]);
                self.loops.push((advance, exit));
                let result = self.block(body, body_blk, env);
                self.loops.pop();
                let (bend, bflow) = result?;
                if let Flow::Value(_) = bflow {
                    bend.append_operation(cf::br(&advance, &[], self.loc));
                }
                let latch = if *inclusive {
                    // Stop at the end bound before incrementing so `..=MAX` cannot wrap.
                    let step = self.new_block(&[]);
                    let last = self.push(
                        advance,
                        arith::cmpi(ctx, CmpiPredicate::Eq, i, en, self.loc),
                    );
                    advance.append_operation(cf::cond_br(
                        ctx,
                        last,
                        &exit,
                        &step,
                        &[],
                        &[],
                        self.loc,
                    ));
                    step
                } else {
                    advance
                };
                let one = self.int_const(latch, int_ty, 1);
                let next = self.push(latch, arith::addi(i, one, self.loc));
                latch.append_operation(cf::br(&header, &[next], self.loc));
                Ok((exit, Flow::Value(vec![])))
            }
            hir::ForIter::Vec(v) => {
                let Ty::Vec(elem_ty) = &v.ty else {
                    return Err(self.unsupported("iterating non-vectors", &v.span));
                };
                self.flat(&v.ty, &v.span)?;
                let elem = self.flat(elem_ty, &v.span)?[0];
                let (cur, vf) = self.expr(v, blk, env)?;
                let Flow::Value(vec) = vf else {
                    return Ok((cur, vf));
                };
                let (ptr, len) = (vec[0], vec[1]);
                let i64_ty = self.cx.i(64);

                let header = self.new_block(&[i64_ty]);
                let body_blk = self.new_block(&[]);
                let exit = self.new_block(&[]);
                let zero = self.int_const(cur, i64_ty, 0);
                cur.append_operation(cf::br(&header, &[zero], self.loc));

                let idx: Value = header.argument(0).expect("index").into();
                let go = self.push(
                    header,
                    arith::cmpi(ctx, CmpiPredicate::Ult, idx, len, self.loc),
                );
                header.append_operation(cf::cond_br(ctx, go, &body_blk, &exit, &[], &[], self.loc));

                let at = self.elem_ptr(body_blk, ptr, idx, elem);
                let item = self.push(
                    body_blk,
                    llvm::load(ctx, at, elem, self.loc, LoadStoreOptions::new()),
                );
                env.insert(var.id, Binding::Vals(vec![item]));
                let advance = self.new_block(&[]);
                self.loops.push((advance, exit));
                let result = self.block(body, body_blk, env);
                self.loops.pop();
                let (bend, bflow) = result?;
                if let Flow::Value(_) = bflow {
                    bend.append_operation(cf::br(&advance, &[], self.loc));
                }
                let one = self.int_const(advance, i64_ty, 1);
                let next = self.push(advance, arith::addi(idx, one, self.loc));
                advance.append_operation(cf::br(&header, &[next], self.loc));
                Ok((exit, Flow::Value(vec![])))
            }
        }
    }

    fn index(
        &mut self,
        e: &hir::Expr,
        base: &hir::Expr,
        index: &hir::Expr,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        if matches!(base.ty, Ty::Tensor { .. }) {
            return self.tensor_index(e, base, index, blk, env);
        }
        let ctx = self.cx.ctx;
        self.flat(&base.ty, &base.span)?;
        let elem = self.flat(&e.ty, &e.span)?[0];
        let (cur, bf) = self.expr(base, blk, env)?;
        let Flow::Value(vec) = bf else {
            return Ok((cur, bf));
        };
        let (cur, xf) = self.expr(index, cur, env)?;
        let Flow::Value(x) = xf else {
            return Ok((cur, xf));
        };
        let idx = self.widen_index(cur, x[0], &index.ty);

        // Unsigned comparison also rejects negative indices.
        let in_bounds = self.push(
            cur,
            arith::cmpi(ctx, CmpiPredicate::Ult, idx, vec[1], self.loc),
        );
        let ok_blk = self.new_block(&[]);
        let fail = self.new_block(&[]);
        cur.append_operation(cf::cond_br(
            ctx,
            in_bounds,
            &ok_blk,
            &fail,
            &[],
            &[],
            self.loc,
        ));
        self.trap(fail, "index out of bounds\n");

        let at = self.elem_ptr(ok_blk, vec[0], idx, elem);
        let v = self.push(
            ok_blk,
            llvm::load(ctx, at, elem, self.loc, LoadStoreOptions::new()),
        );
        Ok((ok_blk, Flow::Value(vec![v])))
    }

    fn array(
        &mut self,
        e: &hir::Expr,
        items: &[hir::Expr],
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        // Validates that the whole `Vec<T>` is representable (scalar elements only).
        self.flat(&e.ty, &e.span)?;
        let Ty::Vec(elem_ty) = &e.ty else {
            return Err(self.unsupported("array literals of unknown type", &e.span));
        };
        let elem = self
            .flat(elem_ty, &e.span)?
            .first()
            .copied()
            .ok_or_else(|| self.unsupported("arrays of unit", &e.span))?;
        let mut cur = blk;
        let mut vals = Vec::with_capacity(items.len());
        for it in items {
            let (next, flow) = self.expr(it, cur, env)?;
            cur = next;
            match flow {
                Flow::Diverged => return Ok((cur, Flow::Diverged)),
                Flow::Value(v) => vals.push(v[0]),
            }
        }
        let i64_ty = self.cx.i(64);
        let bytes = self.int_const(
            cur,
            i64_ty,
            self.cx.elem_bytes(elem_ty) * items.len().max(1) as i64,
        );
        let buf = self.call_libc(cur, Libc::Malloc, &[bytes], self.cx.ptr());
        for (i, v) in vals.iter().enumerate() {
            let idx = self.int_const(cur, i64_ty, i as i64);
            let at = self.elem_ptr(cur, buf, idx, elem);
            cur.append_operation(llvm::store(
                self.cx.ctx,
                *v,
                at,
                self.loc,
                LoadStoreOptions::new(),
            ));
        }
        let len = self.int_const(cur, i64_ty, items.len() as i64);
        Ok((cur, Flow::Value(vec![buf, len])))
    }

    fn expr(&mut self, e: &hir::Expr, blk: BlockRef<'c, 'b>, env: &mut Env<'c, 'b>) -> Out<'c, 'b> {
        let ctx = self.cx.ctx;
        let ok = |blk, v: Vec<Value<'c, 'b>>| Ok((blk, Flow::Value(v)));
        match &e.kind {
            ExprKind::Lit(Lit::Str(text)) => {
                let v = self.str_const(blk, text);
                ok(blk, v)
            }
            ExprKind::Lit(l) => {
                let tys = self.flat(&e.ty, &e.span)?;
                let Some(&ty) = tys.first() else {
                    return ok(blk, vec![]);
                };
                let v = match l {
                    Lit::Int(n) if is_float(&e.ty) => self.push(
                        blk,
                        arith::constant(
                            ctx,
                            FloatAttribute::new(ctx, ty, *n as f64).into(),
                            self.loc,
                        ),
                    ),
                    Lit::Int(n) => self.int_const(blk, ty, *n as i64),
                    Lit::Float(x) => self.push(
                        blk,
                        arith::constant(ctx, FloatAttribute::new(ctx, ty, *x).into(), self.loc),
                    ),
                    Lit::Bool(b) => self.int_const(blk, ty, i64::from(*b)),
                    Lit::Str(_) => unreachable!("handled above"),
                };
                ok(blk, vec![v])
            }
            ExprKind::Local(id) => {
                let v = match env.get(id) {
                    Some(Binding::Vals(v)) => v.clone(),
                    Some(Binding::Slots(slots)) => slots
                        .iter()
                        .map(|(slot, ty)| {
                            self.push(
                                blk,
                                llvm::load(ctx, *slot, *ty, self.loc, LoadStoreOptions::new()),
                            )
                        })
                        .collect(),
                    None => vec![],
                };
                ok(blk, v)
            }
            // Globals are pure, so each use re-evaluates the initializer in place.
            ExprKind::Global(name) => {
                let g = self
                    .cx
                    .program
                    .globals
                    .iter()
                    .find(|g| &g.name == name)
                    .ok_or_else(|| {
                        self.unsupported(&format!("unknown global `{name}`"), &e.span)
                    })?;
                self.expr(&g.init, blk, &mut HashMap::new())
            }
            ExprKind::StructLit { name, fields } => {
                let def = self
                    .cx
                    .struct_def(name)
                    .ok_or_else(|| self.unsupported("unknown structs", &e.span))?;
                let mut cur = blk;
                let mut by_name: HashMap<&str, Vec<Value<'c, 'b>>> = HashMap::new();
                for (fname, fe) in fields {
                    let (next, flow) = self.expr(fe, cur, env)?;
                    cur = next;
                    match flow {
                        Flow::Diverged => return Ok((cur, Flow::Diverged)),
                        Flow::Value(v) => {
                            by_name.insert(fname.as_str(), v);
                        }
                    }
                }
                let flat = def
                    .fields
                    .iter()
                    .flat_map(|(n, _)| by_name.remove(n.as_str()).unwrap_or_default())
                    .collect();
                ok(cur, flat)
            }
            ExprKind::Field { base, name } => {
                let Ty::Struct(sname) = &base.ty else {
                    return Err(self.unsupported("field access on non-structs", &e.span));
                };
                let def = self
                    .cx
                    .struct_def(sname)
                    .ok_or_else(|| self.unsupported("unknown structs", &e.span))?;
                let (cur, flow) = self.expr(base, blk, env)?;
                let Flow::Value(vals) = flow else {
                    return Ok((cur, flow));
                };
                let mut offset = 0;
                for (fname, fty) in &def.fields {
                    let n = self.flat(fty, &e.span)?.len();
                    if fname == name {
                        return ok(cur, vals[offset..offset + n].to_vec());
                    }
                    offset += n;
                }
                Err(self.unsupported(&format!("unknown field `{name}`"), &e.span))
            }
            ExprKind::Call { callee, args } => self.call(e, callee, args, blk, env),
            ExprKind::Unary { op, expr } => {
                let (cur, flow) = self.expr(expr, blk, env)?;
                let Flow::Value(v) = flow else {
                    return Ok((cur, flow));
                };
                let x = v[0];
                let out = match op {
                    UnOp::Neg if is_float(&expr.ty) => self.push(cur, arith::negf(x, self.loc)),
                    UnOp::Neg => {
                        let zero = self.int_const(cur, x.r#type(), 0);
                        self.push(cur, arith::subi(zero, x, self.loc))
                    }
                    UnOp::Not => {
                        let one = self.int_const(cur, x.r#type(), 1);
                        self.push(cur, arith::xori(x, one, self.loc))
                    }
                };
                ok(cur, vec![out])
            }
            ExprKind::Binary {
                op: BinOp::And | BinOp::Or,
                ..
            } => self.short_circuit(e, blk, env),
            ExprKind::Binary { op, lhs, rhs } => {
                let (cur, lf) = self.expr(lhs, blk, env)?;
                let Flow::Value(l) = lf else {
                    return Ok((cur, lf));
                };
                let (cur, rf) = self.expr(rhs, cur, env)?;
                let Flow::Value(r) = rf else {
                    return Ok((cur, rf));
                };
                if lhs.ty == Ty::str() {
                    return self.string_op(e, *op, &l, &r, cur);
                }
                let v = self.arith(*op, &lhs.ty, l[0], r[0], cur);
                ok(cur, vec![v])
            }
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => self.if_expr(e, cond, then, otherwise.as_deref(), blk, env),
            ExprKind::Block(b) => self.block(b, blk, env),
            ExprKind::Assign { place, value } => self.assign(e, place, value, blk, env),
            ExprKind::While { cond, body } => self.while_loop(cond, body, blk, env),
            ExprKind::For { var, iter, body } => self.for_loop(var, iter, body, blk, env),
            ExprKind::Index { base, index } => self.index(e, base, index, blk, env),
            ExprKind::Array(items) => self.array(e, items, blk, env),
            ExprKind::Return(v) => {
                let (cur, vals) = match v {
                    Some(v) => {
                        let (cur, flow) = self.expr(v, blk, env)?;
                        match flow {
                            Flow::Diverged => return Ok((cur, Flow::Diverged)),
                            Flow::Value(v) => (cur, v),
                        }
                    }
                    None => (blk, vec![]),
                };
                cur.append_operation(func::r#return(&vals, self.loc));
                Ok((cur, Flow::Diverged))
            }
            ExprKind::Break | ExprKind::Continue => {
                let Some(&(next, exit)) = self.loops.last() else {
                    return Err(self.unsupported("`break` outside a loop", &e.span));
                };
                let target = if matches!(e.kind, ExprKind::Break) {
                    exit
                } else {
                    next
                };
                blk.append_operation(cf::br(&target, &[], self.loc));
                Ok((blk, Flow::Diverged))
            }
            ExprKind::Closure { .. } => Err(self.unsupported("closures", &e.span)),
            ExprKind::FString(parts) => self.fstring(parts, blk, env),
            ExprKind::Fn(_) | ExprKind::Extern(_) => {
                Err(self.unsupported("function values and std items", &e.span))
            }
            ExprKind::Error => Err(self.unsupported("code with errors", &e.span)),
        }
    }

    fn call(
        &mut self,
        e: &hir::Expr,
        callee: &hir::Expr,
        args: &[hir::Expr],
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        // Methods on strings: `s.len()` and `s.is_empty()`.
        if let ExprKind::Field { base, name } = &callee.kind {
            let is_seq = base.ty == Ty::str() || matches!(base.ty, Ty::Vec(_));
            if is_seq && args.is_empty() && (name == "len" || name == "is_empty") {
                let (cur, flow) = self.expr(base, blk, env)?;
                let Flow::Value(s) = flow else {
                    return Ok((cur, flow));
                };
                let v = if name == "len" {
                    s[1]
                } else {
                    let zero = self.int_const(cur, self.cx.i(64), 0);
                    self.push(
                        cur,
                        arith::cmpi(self.cx.ctx, CmpiPredicate::Eq, s[1], zero, self.loc),
                    )
                };
                return Ok((cur, Flow::Value(vec![v])));
            }
            return Err(self.unsupported(&format!("the method `{name}`"), &e.span));
        }

        let mut cur = blk;
        let mut vals = Vec::new();
        for a in args {
            let (next, flow) = self.expr(a, cur, env)?;
            cur = next;
            match flow {
                Flow::Diverged => return Ok((cur, Flow::Diverged)),
                Flow::Value(v) => vals.extend(v),
            }
        }

        match &callee.kind {
            ExprKind::Fn(name) => {
                let ret = self.flat(&e.ty, &e.span)?;
                let op = func::call(
                    self.cx.ctx,
                    FlatSymbolRefAttribute::new(self.cx.ctx, name),
                    &vals,
                    &ret,
                    self.loc,
                );
                let results = cur.append_operation(op);
                let out = (0..ret.len())
                    .map(|i| results.result(i).expect("call result").into())
                    .collect();
                Ok((cur, Flow::Value(out)))
            }
            ExprKind::Extern(path)
                if path.join("::") == "std::io::print" || path.join("::") == "std::io::println" =>
            {
                let fd = self.int_const(cur, self.cx.i(32), 1);
                self.call_libc(cur, Libc::Write, &[fd, vals[0], vals[1]], self.cx.i(64));
                if path.last().is_some_and(|l| l == "println") {
                    let nl = self.str_const(cur, "\n");
                    self.call_libc(cur, Libc::Write, &[fd, nl[0], nl[1]], self.cx.i(64));
                }
                Ok((cur, Flow::Value(vec![])))
            }
            ExprKind::Extern(path)
                if path.starts_with(&["std".to_string(), "tensor".to_string()]) =>
            {
                let name = path.last().cloned().unwrap_or_default();
                self.tensor_call(&name, e, args, &vals, cur)
            }
            ExprKind::Extern(path) if path.join("::") == "std::iter::count" => {
                Ok((cur, Flow::Value(vec![vals[1]])))
            }
            ExprKind::Extern(path) => {
                Err(self.unsupported(&format!("the std item `{}`", path.join("::")), &callee.span))
            }
            _ => Err(self.unsupported("calls to anything but Novar functions", &e.span)),
        }
    }

    /// Text for one interpolated value as `(ptr, len)`.
    fn to_text(
        &self,
        blk: BlockRef<'c, 'b>,
        ty: &Ty,
        v: &[Value<'c, 'b>],
        span: &Span,
    ) -> Result<(Value<'c, 'b>, Value<'c, 'b>), Diagnostic> {
        let ctx = self.cx.ctx;
        match ty {
            Ty::Prim(Prim::Str) => Ok((v[0], v[1])),
            Ty::Prim(Prim::Bool) => {
                let (t, f) = (self.str_const(blk, "true"), self.str_const(blk, "false"));
                let ptr = self.push(
                    blk,
                    OperationBuilder::new("llvm.select", self.loc)
                        .add_operands(&[v[0], t[0], f[0]])
                        .add_results(&[self.cx.ptr()])
                        .build()
                        .expect("valid select"),
                );
                let len = self.push(blk, arith::select(v[0], t[1], f[1], self.loc));
                Ok((ptr, len))
            }
            Ty::Prim(p) if p.is_int() => {
                let i64_ty = self.cx.i(64);
                let signed = !is_unsigned(ty);
                let wide = if v[0].r#type() == i64_ty {
                    v[0]
                } else if signed {
                    self.push(blk, arith::extsi(v[0], i64_ty, self.loc))
                } else {
                    self.push(blk, arith::extui(v[0], i64_ty, self.loc))
                };
                let flag = self.int_const(blk, self.cx.i(1), i64::from(signed));
                self.cx.state.itoa.set(true);
                let call = func::call(
                    ctx,
                    FlatSymbolRefAttribute::new(ctx, "__novar_itoa"),
                    &[wide, flag],
                    &[self.cx.ptr(), i64_ty],
                    self.loc,
                );
                let r = blk.append_operation(call);
                Ok((
                    r.result(0).expect("ptr").into(),
                    r.result(1).expect("len").into(),
                ))
            }
            other => Err(self.unsupported(&format!("`{other}` values in f-strings"), span)),
        }
    }

    fn fstring(
        &mut self,
        parts: &[hir::FPart],
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let mut cur = blk;
        let mut chunks: Vec<(Value<'c, 'b>, Value<'c, 'b>)> = Vec::new();
        for part in parts {
            match part {
                hir::FPart::Text(t) if t.is_empty() => {}
                hir::FPart::Text(t) => {
                    let c = self.str_const(cur, t);
                    chunks.push((c[0], c[1]));
                }
                hir::FPart::Expr(x) => {
                    let (next, flow) = self.expr(x, cur, env)?;
                    cur = next;
                    match flow {
                        Flow::Diverged => return Ok((cur, Flow::Diverged)),
                        Flow::Value(v) => chunks.push(self.to_text(cur, &x.ty, &v, &x.span)?),
                    }
                }
            }
        }
        match chunks.as_slice() {
            [] => {
                let c = self.str_const(cur, "");
                Ok((cur, Flow::Value(c)))
            }
            [(p, l)] => Ok((cur, Flow::Value(vec![*p, *l]))),
            _ => {
                let mut total = chunks[0].1;
                for (_, len) in &chunks[1..] {
                    total = self.push(cur, arith::addi(total, *len, self.loc));
                }
                let buf = self.call_libc(cur, Libc::Malloc, &[total], self.cx.ptr());
                let mut offset = self.int_const(cur, self.cx.i(64), 0);
                for (ptr, len) in &chunks {
                    let dst = self.push(
                        cur,
                        llvm::get_element_ptr_dynamic(
                            self.cx.ctx,
                            buf,
                            &[offset],
                            self.cx.i(8),
                            self.cx.ptr(),
                            self.loc,
                        ),
                    );
                    self.call_libc(cur, Libc::Memcpy, &[dst, *ptr, *len], self.cx.ptr());
                    offset = self.push(cur, arith::addi(offset, *len, self.loc));
                }
                Ok((cur, Flow::Value(vec![buf, total])))
            }
        }
    }

    /// `+` concatenates (allocating, never freed), `==`/`!=` compare contents.
    fn string_op(
        &mut self,
        e: &hir::Expr,
        op: BinOp,
        l: &[Value<'c, 'b>],
        r: &[Value<'c, 'b>],
        cur: BlockRef<'c, 'b>,
    ) -> Out<'c, 'b> {
        let (ctx, i64_ty) = (self.cx.ctx, self.cx.i(64));
        match op {
            BinOp::Add => {
                let total = self.push(cur, arith::addi(l[1], r[1], self.loc));
                let buf = self.call_libc(cur, Libc::Malloc, &[total], self.cx.ptr());
                self.call_libc(cur, Libc::Memcpy, &[buf, l[0], l[1]], self.cx.ptr());
                let tail = self.push(
                    cur,
                    llvm::get_element_ptr_dynamic(
                        ctx,
                        buf,
                        &[l[1]],
                        self.cx.i(8),
                        self.cx.ptr(),
                        self.loc,
                    ),
                );
                self.call_libc(cur, Libc::Memcpy, &[tail, r[0], r[1]], self.cx.ptr());
                Ok((cur, Flow::Value(vec![buf, total])))
            }
            BinOp::Eq | BinOp::NotEq => {
                let same_len = self.push(
                    cur,
                    arith::cmpi(ctx, CmpiPredicate::Eq, l[1], r[1], self.loc),
                );
                let cmp_blk = self.new_block(&[]);
                let join = self.new_block(&[self.cx.i(1)]);
                let no = self.int_const(cur, self.cx.i(1), 0);
                cur.append_operation(cf::cond_br(
                    ctx,
                    same_len,
                    &cmp_blk,
                    &join,
                    &[],
                    &[no],
                    self.loc,
                ));
                let diff =
                    self.call_libc(cmp_blk, Libc::Memcmp, &[l[0], r[0], l[1]], self.cx.i(32));
                let zero = self.int_const(cmp_blk, self.cx.i(32), 0);
                let eq = self.push(
                    cmp_blk,
                    arith::cmpi(ctx, CmpiPredicate::Eq, diff, zero, self.loc),
                );
                cmp_blk.append_operation(cf::br(&join, &[eq], self.loc));
                let mut v: Value = join.argument(0).expect("join argument").into();
                if op == BinOp::NotEq {
                    let one = self.int_const(join, self.cx.i(1), 1);
                    v = self.push(join, arith::xori(v, one, self.loc));
                }
                Ok((join, Flow::Value(vec![v])))
            }
            _ => {
                let _ = i64_ty;
                Err(self.unsupported(&format!("the operator {op:?} on strings"), &e.span))
            }
        }
    }

    fn arith(
        &self,
        op: BinOp,
        operand_ty: &Ty,
        l: Value<'c, 'b>,
        r: Value<'c, 'b>,
        blk: BlockRef<'c, 'b>,
    ) -> Value<'c, 'b> {
        let loc = self.loc;
        let (ctx, unsigned) = (self.cx.ctx, is_unsigned(operand_ty));
        let op = if is_float(operand_ty) {
            match op {
                BinOp::Add => arith::addf(l, r, loc),
                BinOp::Sub => arith::subf(l, r, loc),
                BinOp::Mul => arith::mulf(l, r, loc),
                BinOp::Div => arith::divf(l, r, loc),
                BinOp::Rem => arith::remf(l, r, loc),
                BinOp::Eq => arith::cmpf(ctx, CmpfPredicate::Oeq, l, r, loc),
                BinOp::NotEq => arith::cmpf(ctx, CmpfPredicate::Une, l, r, loc),
                BinOp::Lt => arith::cmpf(ctx, CmpfPredicate::Olt, l, r, loc),
                BinOp::LtEq => arith::cmpf(ctx, CmpfPredicate::Ole, l, r, loc),
                BinOp::Gt => arith::cmpf(ctx, CmpfPredicate::Ogt, l, r, loc),
                BinOp::GtEq => arith::cmpf(ctx, CmpfPredicate::Oge, l, r, loc),
                BinOp::And | BinOp::Or => unreachable!("lowered as control flow"),
            }
        } else {
            let cmp = |s, u| arith::cmpi(ctx, if unsigned { u } else { s }, l, r, loc);
            match op {
                BinOp::Add => arith::addi(l, r, loc),
                BinOp::Sub => arith::subi(l, r, loc),
                BinOp::Mul => arith::muli(l, r, loc),
                BinOp::Div if unsigned => arith::divui(l, r, loc),
                BinOp::Div => arith::divsi(l, r, loc),
                BinOp::Rem if unsigned => arith::remui(l, r, loc),
                BinOp::Rem => arith::remsi(l, r, loc),
                BinOp::Eq => arith::cmpi(ctx, CmpiPredicate::Eq, l, r, loc),
                BinOp::NotEq => arith::cmpi(ctx, CmpiPredicate::Ne, l, r, loc),
                BinOp::Lt => cmp(CmpiPredicate::Slt, CmpiPredicate::Ult),
                BinOp::LtEq => cmp(CmpiPredicate::Sle, CmpiPredicate::Ule),
                BinOp::Gt => cmp(CmpiPredicate::Sgt, CmpiPredicate::Ugt),
                BinOp::GtEq => cmp(CmpiPredicate::Sge, CmpiPredicate::Uge),
                BinOp::And | BinOp::Or => unreachable!("lowered as control flow"),
            }
        };
        self.push(blk, op)
    }

    /// `a && b` / `a || b` without evaluating `b` when `a` decides the result.
    fn short_circuit(
        &mut self,
        e: &hir::Expr,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let ExprKind::Binary { op, lhs, rhs } = &e.kind else {
            unreachable!()
        };
        let bool_ty = self.cx.i(1);
        let (cur, lf) = self.expr(lhs, blk, env)?;
        let Flow::Value(l) = lf else {
            return Ok((cur, lf));
        };

        let rhs_blk = self.new_block(&[]);
        let join = self.new_block(&[bool_ty]);
        let short = self.int_const(cur, bool_ty, i64::from(*op == BinOp::Or));
        let (on_true, on_false): (&Block, &Block) = if *op == BinOp::And {
            (&rhs_blk, &join)
        } else {
            (&join, &rhs_blk)
        };
        let (true_args, false_args): (Vec<Value>, Vec<Value>) = if *op == BinOp::And {
            (vec![], vec![short])
        } else {
            (vec![short], vec![])
        };
        cur.append_operation(cf::cond_br(
            self.cx.ctx,
            l[0],
            on_true,
            on_false,
            &true_args,
            &false_args,
            self.loc,
        ));

        let (end, rf) = self.expr(rhs, rhs_blk, env)?;
        if let Flow::Value(r) = rf {
            end.append_operation(cf::br(&join, &[r[0]], self.loc));
        }
        Ok((
            join,
            Flow::Value(vec![join.argument(0).expect("join argument").into()]),
        ))
    }

    fn if_expr(
        &mut self,
        e: &hir::Expr,
        cond: &hir::Expr,
        then: &hir::Block,
        otherwise: Option<&hir::Expr>,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let (cur, cond_flow) = self.expr(cond, blk, env)?;
        let Flow::Value(c) = cond_flow else {
            return Ok((cur, cond_flow));
        };
        let result_tys = self.flat(&e.ty, &e.span)?;

        let then_blk = self.new_block(&[]);
        let else_blk = self.new_block(&[]);
        // Kept detached until we know a branch reaches it, so dead joins never enter the IR.
        let join = Block::new(
            &result_tys
                .iter()
                .map(|t| (*t, self.loc))
                .collect::<Vec<_>>(),
        );
        cur.append_operation(cf::cond_br(
            self.cx.ctx,
            c[0],
            &then_blk,
            &else_blk,
            &[],
            &[],
            self.loc,
        ));

        let mut reachable = false;
        let mut finish = |this: &Self, end: BlockRef<'c, 'b>, flow: Flow<'c, 'b>| {
            if let Flow::Value(v) = flow {
                let args = if result_tys.is_empty() { vec![] } else { v };
                end.append_operation(cf::br(&join, &args, this.loc));
                reachable = true;
            }
        };

        let (end, flow) = self.block(then, then_blk, env)?;
        finish(self, end, flow);
        match otherwise {
            Some(o) => {
                let (end, flow) = self.expr(o, else_blk, env)?;
                finish(self, end, flow);
            }
            None => finish(self, else_blk, Flow::Value(vec![])),
        }

        if !reachable {
            return Ok((then_blk, Flow::Diverged));
        }
        let join = self.region.append_block(join);
        let vals = (0..result_tys.len())
            .map(|i| join.argument(i).expect("join argument").into())
            .collect();
        Ok((join, Flow::Value(vals)))
    }
}

// ---- lowering to the LLVM dialect and JIT ---------------------------------------------------

/// Converts `func`/`arith`/`cf` to the LLVM dialect so the module can be translated or JITed.
pub fn lower_to_llvm_dialect(ctx: &Context, module: &mut Module) -> Result<(), String> {
    let pm = PassManager::new(ctx);
    parse_pass_pipeline(
        pm.as_operation_pass_manager(),
        "builtin.module(convert-cf-to-llvm,convert-math-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)",
    )
    .map_err(|e| e.to_string())?;
    pm.run(module)
        .map_err(|e| format!("lowering to the LLVM dialect failed: {e}"))
}

/// A scalar crossing the JIT boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Unit,
}

/// Compiles `program` and calls `name` through MLIR's execution engine.
pub fn jit_call(program: &hir::Program, name: &str, args: &[JitValue]) -> Result<JitValue, String> {
    jit_call_with(program, name, args, Quantization::None)
}

/// Like [`jit_call`], with `f32` matrix products at the given precision.
pub fn jit_call_with(
    program: &hir::Program,
    name: &str,
    args: &[JitValue],
    quant: Quantization,
) -> Result<JitValue, String> {
    let f = program
        .fns
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("no function `{name}`"))?;
    if f.params.len() != args.len() {
        return Err(format!(
            "`{name}` takes {} argument(s), got {}",
            f.params.len(),
            args.len()
        ));
    }
    let ctx = context();
    let mut module = lower_with(&ctx, program, quant).map_err(|d| {
        d.into_iter()
            .map(|d| d.message)
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    lower_to_llvm_dialect(&ctx, &mut module)?;
    let engine = ExecutionEngine::new(&module, 2, &[], false, false);

    // Each argument lives in its own correctly sized cell; the engine takes pointers to them.
    enum Cell {
        I8(i8),
        I16(i16),
        I32(i32),
        I64(i64),
        F32(f32),
        F64(f64),
    }
    impl Cell {
        fn ptr(&mut self) -> *mut () {
            match self {
                Cell::I8(v) => v as *mut _ as *mut (),
                Cell::I16(v) => v as *mut _ as *mut (),
                Cell::I32(v) => v as *mut _ as *mut (),
                Cell::I64(v) => v as *mut _ as *mut (),
                Cell::F32(v) => v as *mut _ as *mut (),
                Cell::F64(v) => v as *mut _ as *mut (),
            }
        }
    }
    fn cell(ty: &Ty, v: Option<JitValue>) -> Result<Cell, String> {
        let int = |v: Option<JitValue>| match v {
            Some(JitValue::Int(i)) => Ok(i),
            Some(JitValue::Bool(b)) => Ok(i64::from(b)),
            None => Ok(0),
            Some(other) => Err(format!("expected an integer, got {other:?}")),
        };
        Ok(match ty {
            Ty::Prim(Prim::Bool | Prim::I8 | Prim::U8) => Cell::I8(int(v)? as i8),
            Ty::Prim(Prim::I16 | Prim::U16) => Cell::I16(int(v)? as i16),
            Ty::Prim(Prim::I32 | Prim::U32) => Cell::I32(int(v)? as i32),
            Ty::Prim(Prim::I64 | Prim::U64) => Cell::I64(int(v)?),
            Ty::Prim(Prim::F32 | Prim::F64) => {
                let x = match v {
                    Some(JitValue::Float(x)) => x,
                    Some(JitValue::Int(i)) => i as f64,
                    None => 0.0,
                    Some(other) => return Err(format!("expected a float, got {other:?}")),
                };
                if matches!(ty, Ty::Prim(Prim::F32)) {
                    Cell::F32(x as f32)
                } else {
                    Cell::F64(x)
                }
            }
            other => return Err(format!("the JIT cannot pass values of type `{other}`")),
        })
    }

    let mut cells = f
        .params
        .iter()
        .zip(args)
        .map(|(p, a)| cell(&p.ty, Some(*a)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut result = match &f.ret {
        Ty::Unit => None,
        t => Some(cell(t, None)?),
    };
    let mut ptrs: Vec<*mut ()> = cells.iter_mut().map(Cell::ptr).collect();
    if let Some(r) = result.as_mut() {
        ptrs.push(r.ptr());
    }
    // SAFETY: every pointer refers to a live cell whose size matches the function signature.
    unsafe { engine.invoke_packed(name, &mut ptrs) }
        .map_err(|e| format!("JIT invocation failed: {e}"))?;

    let unsigned = is_unsigned(&f.ret);
    Ok(match (result, &f.ret) {
        (None, _) => JitValue::Unit,
        (Some(c), ty) => match (c, ty) {
            (Cell::I8(v), Ty::Prim(Prim::Bool)) => JitValue::Bool(v & 1 == 1),
            (Cell::I8(v), _) => JitValue::Int(if unsigned {
                i64::from(v as u8)
            } else {
                i64::from(v)
            }),
            (Cell::I16(v), _) => JitValue::Int(if unsigned {
                i64::from(v as u16)
            } else {
                i64::from(v)
            }),
            (Cell::I32(v), _) => JitValue::Int(if unsigned {
                i64::from(v as u32)
            } else {
                i64::from(v)
            }),
            (Cell::I64(v), _) => JitValue::Int(v),
            (Cell::F32(v), _) => JitValue::Float(f64::from(v)),
            (Cell::F64(v), _) => JitValue::Float(v),
        },
    })
}

/// Textual MLIR for `program`; `llvm_dialect` selects the form after lowering.
pub fn emit_mlir(program: &hir::Program, llvm_dialect: bool) -> Result<String, Vec<Diagnostic>> {
    emit_mlir_with(program, llvm_dialect, Quantization::None)
}

/// Like [`emit_mlir`], with `f32` matrix products at the given precision.
pub fn emit_mlir_with(
    program: &hir::Program,
    llvm_dialect: bool,
    quant: Quantization,
) -> Result<String, Vec<Diagnostic>> {
    let ctx = context();
    let mut module = lower_with(&ctx, program, quant)?;
    if llvm_dialect {
        lower_to_llvm_dialect(&ctx, &mut module).map_err(|e| vec![Diagnostic::error(0..0, e)])?;
    }
    Ok(module.as_operation().to_string())
}
