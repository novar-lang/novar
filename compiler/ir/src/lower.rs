use std::collections::HashMap;

use melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use melior::dialect::{DialectRegistry, cf, func};
use melior::ir::attribute::{
    Attribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
    TypeAttribute,
};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
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
use novar_hir::{Prim, Ty};

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

fn mlir_type<'c>(ctx: &'c Context, ty: &Ty) -> Result<Option<Type<'c>>, String> {
    Ok(match ty {
        Ty::Unit | Ty::Never => None,
        Ty::Prim(p) => Some(match p {
            Prim::I8 | Prim::U8 => IntegerType::new(ctx, 8).into(),
            Prim::I16 | Prim::U16 => IntegerType::new(ctx, 16).into(),
            Prim::I32 | Prim::U32 => IntegerType::new(ctx, 32).into(),
            Prim::I64 | Prim::U64 => IntegerType::new(ctx, 64).into(),
            Prim::F32 => Type::float32(ctx),
            Prim::F64 => Type::float64(ctx),
            Prim::Bool => IntegerType::new(ctx, 1).into(),
            Prim::Str => return Err("`String`".into()),
        }),
        other => return Err(format!("`{other}`")),
    })
}

fn is_unsigned(ty: &Ty) -> bool {
    matches!(ty, Ty::Prim(Prim::U8 | Prim::U16 | Prim::U32 | Prim::U64))
}

fn is_float(ty: &Ty) -> bool {
    matches!(ty, Ty::Prim(p) if p.is_float())
}

/// Lowers every function of `program` to a verified MLIR module.
pub fn lower<'c>(ctx: &'c Context, program: &hir::Program) -> Result<Module<'c>, Vec<Diagnostic>> {
    let module = Module::new(Location::unknown(ctx));
    let mut diags = Vec::new();
    for f in &program.fns {
        match lower_fn(ctx, f) {
            Ok(op) => {
                module.body().append_operation(op);
            }
            Err(d) => diags.push(d),
        }
    }
    if !diags.is_empty() {
        return Err(diags);
    }
    if !module.as_operation().verify() {
        return Err(vec![
            Diagnostic::error(0..0, "internal error: generated MLIR failed verification")
                .with_note(module.as_operation().to_string()),
        ]);
    }
    Ok(module)
}

fn lower_fn<'c>(ctx: &'c Context, f: &hir::FnDef) -> Result<melior::ir::Operation<'c>, Diagnostic> {
    let unsupported = |what: String, span: &Span| {
        Diagnostic::error(
            span.clone(),
            format!("the native backend does not support {what} yet"),
        )
        .with_note(format!("in function `{}`", f.name))
    };
    let loc = Location::unknown(ctx);

    let mut param_tys = Vec::new();
    for p in &f.params {
        let t = mlir_type(ctx, &p.ty)
            .map_err(|w| unsupported(format!("parameters of type {w}"), &f.span))?;
        param_tys.push(t.ok_or_else(|| unsupported("unit parameters".into(), &f.span))?);
    }
    let ret_ty =
        mlir_type(ctx, &f.ret).map_err(|w| unsupported(format!("return type {w}"), &f.span))?;

    let region = Region::new();
    {
        let entry = region.append_block(Block::new(
            &param_tys.iter().map(|t| (*t, loc)).collect::<Vec<_>>(),
        ));
        let mut env: HashMap<LocalId, Value<'c, '_>> = HashMap::new();
        for (i, p) in f.params.iter().enumerate() {
            env.insert(p.id, entry.argument(i).expect("entry argument").into());
        }
        let mut em = Emitter {
            ctx,
            region: &region,
            func: f,
            loc,
        };
        let (end, flow) = em.block(&f.body, entry, &mut env)?;
        if let Flow::Value(v) = flow {
            let rets: Vec<Value> = if ret_ty.is_some() {
                v.into_iter().collect()
            } else {
                vec![]
            };
            end.append_operation(func::r#return(&rets, loc));
        }
    }

    let fn_ty = FunctionType::new(ctx, &param_tys, &ret_ty.into_iter().collect::<Vec<_>>());
    Ok(func::func(
        ctx,
        StringAttribute::new(ctx, &f.name),
        TypeAttribute::new(fn_ty.into()),
        region,
        &[(
            Identifier::new(ctx, "llvm.emit_c_interface"),
            Attribute::unit(ctx),
        )],
        loc,
    ))
}

/// What evaluating an expression left behind.
enum Flow<'c, 'b> {
    /// Fell through, optionally producing a value (none for unit).
    Value(Option<Value<'c, 'b>>),
    /// Control left via `return`; the current block is already terminated.
    Diverged,
}

type Env<'c, 'b> = HashMap<LocalId, Value<'c, 'b>>;
type Out<'c, 'b> = Result<(BlockRef<'c, 'b>, Flow<'c, 'b>), Diagnostic>;

struct Emitter<'c, 'b, 'p> {
    ctx: &'c Context,
    region: &'b Region<'c>,
    func: &'p hir::FnDef,
    loc: Location<'c>,
}

impl<'c, 'b, 'p> Emitter<'c, 'b, 'p> {
    fn unsupported(&self, what: &str, span: &Span) -> Diagnostic {
        Diagnostic::error(
            span.clone(),
            format!("the native backend does not support {what} yet"),
        )
        .with_note(format!("in function `{}`", self.func.name))
    }

    fn ty(&self, ty: &Ty, span: &Span) -> Result<Option<Type<'c>>, Diagnostic> {
        mlir_type(self.ctx, ty).map_err(|w| self.unsupported(&format!("values of type {w}"), span))
    }

    fn block(
        &mut self,
        b: &hir::Block,
        mut cur: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        for s in &b.stmts {
            let e = match s {
                hir::Stmt::Let { local, init } => {
                    let (next, flow) = self.expr(init, cur, env)?;
                    cur = next;
                    match flow {
                        Flow::Diverged => return Ok((cur, Flow::Diverged)),
                        Flow::Value(Some(v)) => {
                            env.insert(local.id, v);
                        }
                        Flow::Value(None) => {}
                    }
                    continue;
                }
                hir::Stmt::Expr(e) => e,
            };
            let (next, flow) = self.expr(e, cur, env)?;
            cur = next;
            if matches!(flow, Flow::Diverged) {
                return Ok((cur, Flow::Diverged));
            }
        }
        match &b.tail {
            Some(t) => self.expr(t, cur, env),
            None => Ok((cur, Flow::Value(None))),
        }
    }

    fn append_value(&self, blk: BlockRef<'c, 'b>, op: melior::ir::Operation<'c>) -> Value<'c, 'b> {
        blk.append_operation(op)
            .result(0)
            .expect("one result")
            .into()
    }

    fn int_const(&self, blk: BlockRef<'c, 'b>, ty: Type<'c>, v: i64) -> Value<'c, 'b> {
        self.append_value(
            blk,
            arith::constant(self.ctx, IntegerAttribute::new(ty, v).into(), self.loc),
        )
    }

    fn new_block(&self, arg: Option<Type<'c>>) -> BlockRef<'c, 'b> {
        let args: Vec<_> = arg.into_iter().map(|t| (t, self.loc)).collect();
        self.region.append_block(Block::new(&args))
    }

    fn expr(&mut self, e: &hir::Expr, blk: BlockRef<'c, 'b>, env: &mut Env<'c, 'b>) -> Out<'c, 'b> {
        let ok = |blk, v| Ok((blk, Flow::Value(v)));
        match &e.kind {
            ExprKind::Lit(l) => {
                let Some(ty) = self.ty(&e.ty, &e.span)? else {
                    return ok(blk, None);
                };
                let v = match l {
                    Lit::Int(n) if is_float(&e.ty) => self.append_value(
                        blk,
                        arith::constant(
                            self.ctx,
                            FloatAttribute::new(self.ctx, ty, *n as f64).into(),
                            self.loc,
                        ),
                    ),
                    Lit::Int(n) => self.int_const(blk, ty, *n as i64),
                    Lit::Float(x) => self.append_value(
                        blk,
                        arith::constant(
                            self.ctx,
                            FloatAttribute::new(self.ctx, ty, *x).into(),
                            self.loc,
                        ),
                    ),
                    Lit::Bool(b) => self.int_const(blk, ty, i64::from(*b)),
                    Lit::Str(_) => return Err(self.unsupported("string literals", &e.span)),
                };
                ok(blk, Some(v))
            }
            ExprKind::Local(id) => ok(blk, env.get(id).copied()),
            ExprKind::Call { callee, args } => {
                let ExprKind::Fn(name) = &callee.kind else {
                    return Err(self.unsupported("calls to anything but Novar functions", &e.span));
                };
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
                let ret = self.ty(&e.ty, &e.span)?;
                let op = func::call(
                    self.ctx,
                    FlatSymbolRefAttribute::new(self.ctx, name),
                    &vals,
                    &ret.into_iter().collect::<Vec<_>>(),
                    self.loc,
                );
                let r = cur.append_operation(op);
                ok(cur, r.result(0).ok().map(Into::into))
            }
            ExprKind::Unary { op, expr } => {
                let (cur, flow) = self.expr(expr, blk, env)?;
                let Flow::Value(Some(v)) = flow else {
                    return Ok((cur, flow));
                };
                let out = match op {
                    UnOp::Neg if is_float(&expr.ty) => {
                        self.append_value(cur, arith::negf(v, self.loc))
                    }
                    UnOp::Neg => {
                        let zero = self.int_const(cur, v.r#type(), 0);
                        self.append_value(cur, arith::subi(zero, v, self.loc))
                    }
                    UnOp::Not => {
                        let one = self.int_const(cur, v.r#type(), 1);
                        self.append_value(cur, arith::xori(v, one, self.loc))
                    }
                };
                ok(cur, Some(out))
            }
            ExprKind::Binary {
                op: BinOp::And | BinOp::Or,
                ..
            } => self.short_circuit(e, blk, env),
            ExprKind::Binary { op, lhs, rhs } => {
                let (cur, lf) = self.expr(lhs, blk, env)?;
                let Flow::Value(Some(l)) = lf else {
                    return Ok((cur, lf));
                };
                let (cur, rf) = self.expr(rhs, cur, env)?;
                let Flow::Value(Some(r)) = rf else {
                    return Ok((cur, rf));
                };
                let v = self.arith(*op, &lhs.ty, l, r, cur);
                ok(cur, Some(v))
            }
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => self.if_expr(e, cond, then, otherwise.as_deref(), blk, env),
            ExprKind::Block(b) => self.block(b, blk, env),
            ExprKind::Return(v) => {
                let (cur, vals) = match v {
                    Some(v) => {
                        let (cur, flow) = self.expr(v, blk, env)?;
                        match flow {
                            Flow::Diverged => return Ok((cur, Flow::Diverged)),
                            Flow::Value(v) => (cur, v.into_iter().collect::<Vec<_>>()),
                        }
                    }
                    None => (blk, vec![]),
                };
                cur.append_operation(func::r#return(&vals, self.loc));
                Ok((cur, Flow::Diverged))
            }
            ExprKind::Global(_) => Err(self.unsupported("global variables", &e.span)),
            ExprKind::Closure { .. } => Err(self.unsupported("closures", &e.span)),
            ExprKind::FString(_) => Err(self.unsupported("f-strings", &e.span)),
            ExprKind::StructLit { .. } | ExprKind::Field { .. } => {
                Err(self.unsupported("structs", &e.span))
            }
            ExprKind::Fn(_) | ExprKind::Extern(_) => {
                Err(self.unsupported("function values and std items", &e.span))
            }
            ExprKind::Error => Err(self.unsupported("code with errors", &e.span)),
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
        let (ctx, unsigned) = (self.ctx, is_unsigned(operand_ty));
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
        self.append_value(blk, op)
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
        let bool_ty: Type = IntegerType::new(self.ctx, 1).into();
        let (cur, lf) = self.expr(lhs, blk, env)?;
        let Flow::Value(Some(l)) = lf else {
            return Ok((cur, lf));
        };

        let rhs_blk = self.new_block(None);
        let join = self.new_block(Some(bool_ty));
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
            self.ctx,
            l,
            on_true,
            on_false,
            &true_args,
            &false_args,
            self.loc,
        ));

        let (end, rf) = self.expr(rhs, rhs_blk, env)?;
        if let Flow::Value(Some(r)) = rf {
            end.append_operation(cf::br(&join, &[r], self.loc));
        }
        Ok((
            join,
            Flow::Value(Some(join.argument(0).expect("join argument").into())),
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
        let (cur, cf_) = self.expr(cond, blk, env)?;
        let Flow::Value(Some(c)) = cf_ else {
            return Ok((cur, cf_));
        };
        let result_ty = self.ty(&e.ty, &e.span)?;

        let then_blk = self.new_block(None);
        let else_blk = self.new_block(None);
        // Kept detached until we know a branch reaches it, so dead joins never enter the IR.
        let join = Block::new(&result_ty.iter().map(|t| (*t, self.loc)).collect::<Vec<_>>());
        cur.append_operation(cf::cond_br(
            self.ctx,
            c,
            &then_blk,
            &else_blk,
            &[],
            &[],
            self.loc,
        ));

        let mut reachable = false;
        let mut finish = |this: &Self, end: BlockRef<'c, 'b>, flow: Flow<'c, 'b>| {
            if let Flow::Value(v) = flow {
                let args: Vec<Value> = if result_ty.is_some() {
                    v.into_iter().collect()
                } else {
                    vec![]
                };
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
            None => finish(self, else_blk, Flow::Value(None)),
        }

        if !reachable {
            return Ok((then_blk, Flow::Diverged));
        }
        let join = self.region.append_block(join);
        let v = result_ty.map(|_| join.argument(0).expect("join argument").into());
        Ok((join, Flow::Value(v)))
    }
}

// ---- lowering to the LLVM dialect and JIT ---------------------------------------------------

/// Converts `func`/`arith`/`cf` to the LLVM dialect so the module can be translated or JITed.
pub fn lower_to_llvm_dialect(ctx: &Context, module: &mut Module) -> Result<(), String> {
    let pm = PassManager::new(ctx);
    parse_pass_pipeline(
        pm.as_operation_pass_manager(),
        "builtin.module(convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)",
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
    let mut module = lower(&ctx, program).map_err(|d| {
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
    let ctx = context();
    let mut module = lower(&ctx, program)?;
    if llvm_dialect {
        lower_to_llvm_dialect(&ctx, &mut module).map_err(|e| vec![Diagnostic::error(0..0, e)])?;
    }
    Ok(module.as_operation().to_string())
}
