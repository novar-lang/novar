//! Tree-walking evaluator over the typed HIR.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use novar_hir::Ty;
use novar_hir::hir::{self, BinOp, ExprKind, Lit, LocalId, UnOp};
use serde_json::Value as Json;

use crate::std_impl;

#[cfg(feature = "gpu")]
mod gpu_chain;
use crate::value::{Closure, Db, Env, StructVal, Value, from_json};

/// Where `std::tensor` operations on `f32` tensors run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// A GPU through `wgpu` (needs the `gpu` feature). Results agree with the CPU path to
    /// single-precision tolerance, not bit for bit.
    Gpu,
}

/// Runs `std::dist::map` somewhere other than the calling thread: other processes, other machines,
/// a Ray cluster. Arguments and results travel as JSON, converted with the function's static types.
pub trait Executor: Send + Sync {
    /// Calls the top-level function `function` once per item and returns the results in order.
    fn map(
        &self,
        function: &str,
        items: Vec<serde_json::Value>,
    ) -> Result<Vec<serde_json::Value>, String>;
}

/// Maximum nesting of function calls before evaluation aborts. The native stack grows on demand
/// (see `stacker`), so this is a language limit, not a limit of the thread's stack.
const MAX_DEPTH: usize = 2000;
/// Free native stack kept before each call, and the size of each extra stack segment.
const STACK_RED_ZONE: usize = 256 * 1024;
const STACK_SEGMENT: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeError {
    pub message: String,
}

impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeError {}

/// What an endpoint call can fail with; maps directly onto an HTTP response.
#[derive(Debug, Clone, PartialEq)]
pub struct EndpointError {
    pub status: u16,
    pub message: String,
}

enum Ctrl {
    Return(Value),
    Break,
    Continue,
    Fail(RuntimeError),
}

impl From<RuntimeError> for Ctrl {
    fn from(e: RuntimeError) -> Self {
        Ctrl::Fail(e)
    }
}

type Eval = Result<Value, Ctrl>;

fn fail<T>(msg: impl Into<String>) -> Result<T, Ctrl> {
    Err(Ctrl::Fail(RuntimeError::new(msg)))
}

pub struct Interp {
    fns: HashMap<String, hir::FnDef>,
    structs: HashMap<String, Vec<(String, Ty)>>,
    enums: HashMap<String, Vec<(String, Vec<Ty>)>>,
    globals: RwLock<HashMap<String, Value>>,
    dbs: Mutex<HashMap<String, Arc<Db>>>,
    out: Mutex<Box<dyn std::io::Write + Send>>,
    live: std::sync::atomic::AtomicBool,
    /// Remaining evaluation steps; `u64::MAX` means unlimited.
    fuel: std::sync::atomic::AtomicU64,
    quantization: std::sync::atomic::AtomicU8,
    executor: RwLock<Option<Arc<dyn Executor>>>,
    device: std::sync::atomic::AtomicU8,
}

impl fmt::Debug for Interp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interp")
            .field("fns", &self.fns.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

struct Ctx {
    env: Env,
    depth: usize,
}

impl Interp {
    /// Prepares `program` for execution and evaluates its global `let`s in order.
    pub fn new(program: hir::Program) -> Result<Interp, RuntimeError> {
        Self::with_live(program, false)
    }

    /// Like [`Interp::new`], choosing real services (`live`) before the globals are evaluated.
    pub fn with_live(program: hir::Program, live: bool) -> Result<Interp, RuntimeError> {
        let interp = Interp {
            structs: program
                .structs
                .iter()
                .map(|s| (s.name.clone(), s.fields.clone()))
                .collect(),
            enums: program
                .enums
                .iter()
                .map(|e| (e.name.clone(), e.variants.clone()))
                .collect(),
            fns: program
                .fns
                .into_iter()
                .map(|f| (f.name.clone(), f))
                .collect(),
            globals: RwLock::new(HashMap::new()),
            dbs: Mutex::new(HashMap::new()),
            out: Mutex::new(Box::new(std::io::stdout())),
            live: std::sync::atomic::AtomicBool::new(live),
            fuel: std::sync::atomic::AtomicU64::new(u64::MAX),
            quantization: std::sync::atomic::AtomicU8::new(0),
            executor: RwLock::new(None),
            device: std::sync::atomic::AtomicU8::new(0),
        };
        for g in &program.globals {
            let mut ctx = Ctx {
                env: Env::new(),
                depth: 0,
            };
            let v = interp.eval(&g.init, &mut ctx).map_err(|c| match c {
                Ctrl::Fail(e) => RuntimeError::new(format!("while initialising `{}`: {e}", g.name)),
                Ctrl::Return(_) => RuntimeError::new("`return` outside a function"),
                Ctrl::Break | Ctrl::Continue => {
                    RuntimeError::new("`break` or `continue` outside a loop")
                }
            })?;
            interp
                .globals
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(g.name.clone(), v);
        }
        Ok(interp)
    }

    /// Redirects `std::io::print`/`println` (stdout by default).
    pub fn set_output(&self, writer: impl std::io::Write + Send + 'static) {
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = Box::new(writer);
    }

    pub(crate) fn write_out(&self, text: &str) -> Result<(), RuntimeError> {
        use std::io::Write;
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        out.write_all(text.as_bytes())
            .and_then(|()| out.flush())
            .map_err(|e| RuntimeError::new(format!("cannot write output: {e}")))
    }

    /// Switches the AI services between offline stand-ins (the default) and the real Ollama,
    /// OpenAI and PostgreSQL backends. Must be called before the program creates its clients.
    pub fn set_live(&self, live: bool) {
        self.live.store(live, std::sync::atomic::Ordering::SeqCst);
    }

    /// Embeds `text` the way `OllamaEmbedder.embed` would: with Ollama (`OLLAMA_HOST`, model
    /// `NOVAR_EMBED_MODEL`) when live, otherwise with the offline stand-in. Used to seed stores.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, RuntimeError> {
        #[cfg(feature = "live")]
        if self.is_live() {
            let host =
                std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
            let model =
                std::env::var("NOVAR_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".into());
            return crate::services::ollama_embed(
                &host,
                &model,
                text,
                novar_hir::stdlib::EMBEDDING_DIM as usize,
            );
        }
        Ok(crate::std_impl::embed_text(text))
    }

    /// Stops execution with an error after `steps` more evaluation steps (for untrusted code, e.g.
    /// a web playground). Unlimited by default.
    pub fn set_step_limit(&self, steps: u64) {
        self.fuel.store(steps, std::sync::atomic::Ordering::Relaxed);
    }

    /// Converts request-style JSON into a value of static type `ty`.
    pub fn value_from_json(&self, ty: &Ty, json: &Json) -> Result<Value, String> {
        from_json(ty, json, &self.structs, &self.enums)
    }

    /// Runs `f32` matrix multiplications at the given precision (see [`Quantization`]).
    pub fn set_quantization(&self, q: novar_hir::Quantization) {
        let v = match q {
            novar_hir::Quantization::None => 0,
            novar_hir::Quantization::Int8 => 1,
            novar_hir::Quantization::Int8Channel => 2,
            novar_hir::Quantization::Int4Channel => 3,
        };
        self.quantization
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn quantization(&self) -> novar_hir::Quantization {
        match self.quantization.load(std::sync::atomic::Ordering::Relaxed) {
            1 => novar_hir::Quantization::Int8,
            2 => novar_hir::Quantization::Int8Channel,
            3 => novar_hir::Quantization::Int4Channel,
            _ => novar_hir::Quantization::None,
        }
    }

    /// Where `std::dist::map` runs its calls. Without one, calls run on this machine's threads.
    pub fn set_executor(&self, executor: Arc<dyn Executor>) {
        *self.executor.write().unwrap_or_else(|e| e.into_inner()) = Some(executor);
    }

    pub(crate) fn executor(&self) -> Option<Arc<dyn Executor>> {
        self.executor
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Calls `name` with JSON arguments (converted by the parameter types) and returns the result
    /// as JSON; an `Ok(v)` result becomes `v`. This is how workers run remote calls.
    pub fn call_json(&self, name: &str, args: &[Json]) -> Result<Json, String> {
        let (params, _) = self
            .signature(name)
            .ok_or_else(|| format!("no function `{name}`"))?;
        if params.len() != args.len() {
            return Err(format!(
                "`{name}` takes {} argument(s), got {}",
                params.len(),
                args.len()
            ));
        }
        let values = params
            .iter()
            .zip(args)
            .map(|((pname, ty), arg)| {
                self.value_from_json(ty, arg)
                    .map_err(|e| format!("argument `{pname}`: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        match self.call(name, values).map_err(|e| e.to_string())? {
            Value::Err(e) => Err(format!("Err({e})")),
            v => Ok(v.to_json()),
        }
    }

    /// Where tensor operations run (see [`Device`]).
    pub fn set_device(&self, device: Device) {
        let v = match device {
            Device::Cpu => 0,
            Device::Gpu => 1,
        };
        self.device.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn device(&self) -> Device {
        match self.device.load(std::sync::atomic::Ordering::Relaxed) {
            1 => Device::Gpu,
            _ => Device::Cpu,
        }
    }

    pub fn is_live(&self) -> bool {
        self.live.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The database behind `url`, opened on first use: PostgreSQL when live, otherwise an
    /// in-memory store that embedders can seed.
    pub fn db_for(&self, url: &str) -> Result<Arc<Db>, RuntimeError> {
        let mut dbs = self.dbs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(db) = dbs.get(url) {
            return Ok(db.clone());
        }
        #[cfg(feature = "live")]
        let pg = if self.is_live() && url.starts_with("postgres") {
            Some(crate::services::PgStore::connect(
                url,
                novar_hir::stdlib::EMBEDDING_DIM as usize,
            )?)
        } else {
            None
        };
        let db = Arc::new(Db {
            url: url.to_string(),
            #[cfg(feature = "live")]
            pg,
            ..Db::default()
        });
        dbs.insert(url.to_string(), db.clone());
        Ok(db)
    }

    /// Every database opened so far (one per distinct URL).
    pub fn dbs(&self) -> Vec<Arc<Db>> {
        self.dbs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect()
    }

    /// The fields of struct `name`, in declaration order.
    pub fn struct_fields(&self, name: &str) -> Option<&[(String, Ty)]> {
        self.structs.get(name).map(Vec::as_slice)
    }

    /// Parameter names / types and the return type of function `name`.
    pub fn signature(&self, name: &str) -> Option<(Vec<(String, Ty)>, Ty)> {
        self.fns.get(name).map(|f| {
            (
                f.params
                    .iter()
                    .map(|p| (p.name.clone(), p.ty.clone()))
                    .collect(),
                f.ret.clone(),
            )
        })
    }

    pub fn function_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.fns.keys().cloned().collect();
        v.sort();
        v
    }

    /// Wraps an Arrow batch as a `Table<schema>`, matching columns to the struct's fields by name
    /// (casting where needed).
    pub fn table_from_batch(
        &self,
        schema: &str,
        batch: &arrow::record_batch::RecordBatch,
    ) -> Result<Value, RuntimeError> {
        let fields = self
            .structs
            .get(schema)
            .ok_or_else(|| RuntimeError::new(format!("unknown struct `{schema}`")))?;
        let batch = crate::table::conform(batch, fields)?;
        Ok(Value::Table(Arc::new(crate::value::TableVal {
            schema: schema.to_string(),
            batch,
        })))
    }

    pub fn endpoint_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self
            .fns
            .values()
            .filter(|f| f.endpoint.is_some())
            .map(|f| f.name.clone())
            .collect();
        v.sort();
        v
    }

    /// Calls a Novar function by name.
    pub fn call(&self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let f = self
            .fns
            .get(name)
            .ok_or_else(|| RuntimeError::new(format!("no function `{name}`")))?;
        self.call_fn(f, args, 0).map_err(|c| match c {
            Ctrl::Fail(e) => e,
            Ctrl::Return(_) => RuntimeError::new("unexpected return"),
            Ctrl::Break | Ctrl::Continue => {
                RuntimeError::new("`break` or `continue` outside a loop")
            }
        })
    }

    /// Runs an `@endpoint` function on a JSON payload, returning the JSON response body.
    pub fn call_endpoint(&self, name: &str, payload: &Json) -> Result<Json, EndpointError> {
        let internal = |m: String| EndpointError {
            status: 500,
            message: m,
        };
        let f = self
            .fns
            .get(name)
            .filter(|f| f.endpoint.is_some())
            .ok_or_else(|| internal(format!("no endpoint `{name}`")))?;
        let args = match f.params.first() {
            Some(p) => {
                vec![
                    from_json(&p.ty, payload, &self.structs, &self.enums).map_err(|e| {
                        EndpointError {
                            status: 400,
                            message: e,
                        }
                    })?,
                ]
            }
            None => vec![],
        };
        match self.call(name, args) {
            Ok(Value::Ok(v)) => Ok(v.to_json()),
            Ok(Value::Err(e)) => match *e {
                Value::Status(code, msg) => Err(EndpointError {
                    status: code,
                    message: msg.to_string(),
                }),
                other => Err(internal(other.to_string())),
            },
            Ok(other) => Ok(other.to_json()),
            Err(e) => Err(internal(format!("runtime error: {e}"))),
        }
    }

    /// Applies a closure or function value (used by std higher-order functions).
    pub(crate) fn apply(&self, f: &Value, args: Vec<Value>) -> Result<Value, RuntimeError> {
        self.apply_at(f, args, 0).map_err(|c| match c {
            Ctrl::Fail(e) => e,
            Ctrl::Return(v) => RuntimeError::new(format!("stray return of {v}")),
            Ctrl::Break | Ctrl::Continue => {
                RuntimeError::new("`break` or `continue` outside a loop")
            }
        })
    }

    fn apply_at(&self, f: &Value, args: Vec<Value>, depth: usize) -> Eval {
        if let Value::Func(name) = f {
            let func = self
                .fns
                .get(&**name)
                .ok_or_else(|| Ctrl::Fail(RuntimeError::new(format!("no function `{name}`"))))?;
            return self.call_fn(func, args, depth);
        }
        let Value::Closure(c) = f else {
            return fail(format!("{} is not callable", f.type_name()));
        };
        if c.params.len() != args.len() {
            return fail(format!(
                "closure takes {} argument(s), got {}",
                c.params.len(),
                args.len()
            ));
        }
        if depth >= MAX_DEPTH {
            return fail("stack overflow: call depth limit reached");
        }
        let mut ctx = Ctx {
            env: c.env.clone(),
            depth: depth + 1,
        };
        ctx.env.extend(c.params.iter().copied().zip(args));
        stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, || {
            self.eval(&c.body, &mut ctx)
        })
    }

    fn call_fn(&self, f: &hir::FnDef, args: Vec<Value>, depth: usize) -> Eval {
        if args.len() != f.params.len() {
            return fail(format!(
                "`{}` takes {} argument(s), got {}",
                f.name,
                f.params.len(),
                args.len()
            ));
        }
        if depth >= MAX_DEPTH {
            return fail(format!(
                "stack overflow: call depth limit reached in `{}`",
                f.name
            ));
        }
        let mut ctx = Ctx {
            env: Env::new(),
            depth: depth + 1,
        };
        ctx.env.extend(f.params.iter().map(|p| p.id).zip(args));
        let result = stacker::maybe_grow(STACK_RED_ZONE, STACK_SEGMENT, || {
            self.block(&f.body, &mut ctx)
        });
        match result {
            Err(Ctrl::Return(v)) => Ok(v),
            other => other,
        }
    }

    /// Runs one loop iteration; `Ok(false)` means the body executed `break`.
    fn loop_body(&self, body: &hir::Block, ctx: &mut Ctx) -> Result<bool, Ctrl> {
        match self.block(body, ctx) {
            Ok(_) | Err(Ctrl::Continue) => Ok(true),
            Err(Ctrl::Break) => Ok(false),
            Err(other) => Err(other),
        }
    }

    fn block(&self, b: &hir::Block, ctx: &mut Ctx) -> Eval {
        for s in &b.stmts {
            match s {
                hir::Stmt::Let { local, init } => {
                    let v = self.eval(init, ctx)?;
                    ctx.env.insert(local.id, v);
                }
                hir::Stmt::Expr(e) => {
                    self.eval(e, ctx)?;
                }
            }
        }
        match &b.tail {
            Some(t) => self.eval(t, ctx),
            None => Ok(Value::Unit),
        }
    }

    /// Assignment, loops and arrays. Kept out of `eval` so its stack frame stays small: the
    /// evaluator recurses once per Novar call, and debug-build frames add up quickly.
    #[inline(never)]
    fn eval_mutation_and_loops(&self, e: &hir::Expr, ctx: &mut Ctx) -> Eval {
        match &e.kind {
            ExprKind::Assign { place, value } => {
                // Order: the value, then the place's index expressions left to right.
                let v = self.eval(value, ctx)?;
                let mut steps = Vec::with_capacity(place.projs.len());
                for pr in &place.projs {
                    steps.push(match pr {
                        hir::Proj::Field { name, .. } => Step::Field(name.as_str()),
                        hir::Proj::Index(i) => match self.eval(i, ctx)? {
                            Value::Int(i) => Step::Index(i),
                            other => {
                                return fail(format!(
                                    "index is {}, not an integer",
                                    other.type_name()
                                ));
                            }
                        },
                    });
                }
                let root = ctx.env.get(&place.local).cloned().ok_or_else(|| {
                    Ctrl::Fail(RuntimeError::new(format!(
                        "unbound variable `{}`",
                        place.name
                    )))
                })?;
                let updated = update_place(root, &steps, v)?;
                ctx.env.insert(place.local, updated);
                Ok(Value::Unit)
            }
            ExprKind::While { cond, body } => {
                loop {
                    match self.eval(cond, ctx)? {
                        Value::Bool(true) => {
                            if !self.loop_body(body, ctx)? {
                                break;
                            }
                        }
                        Value::Bool(false) => break,
                        other => {
                            return fail(format!(
                                "`while` condition is {}, not bool",
                                other.type_name()
                            ));
                        }
                    }
                }
                Ok(Value::Unit)
            }
            ExprKind::For { var, iter, body } => {
                match iter {
                    hir::ForIter::Range {
                        start,
                        end,
                        inclusive,
                    } => {
                        let (Value::Int(a), Value::Int(b)) =
                            (self.eval(start, ctx)?, self.eval(end, ctx)?)
                        else {
                            return fail("range bounds must be integers");
                        };
                        // i128 so that `..=i64::MAX` cannot overflow the counter.
                        let (mut i, last) = (i128::from(a), i128::from(b));
                        while i < last || (*inclusive && i == last) {
                            ctx.env.insert(var.id, Value::Int(i as i64));
                            if !self.loop_body(body, ctx)? {
                                break;
                            }
                            i += 1;
                        }
                    }
                    hir::ForIter::Vec(v) => {
                        let items = match self.eval(v, ctx)? {
                            Value::Vec(items) => items,
                            Value::Table(t) => {
                                for i in 0..t.rows() {
                                    ctx.env.insert(var.id, t.row(i)?);
                                    if !self.loop_body(body, ctx)? {
                                        break;
                                    }
                                }
                                return Ok(Value::Unit);
                            }
                            _ => return fail("`for` needs a Vec or a Table"),
                        };
                        // Live view: element assignments inside the loop are visible to later
                        // iterations. The length is fixed when the loop starts.
                        for i in 0..items.len() {
                            let Some(item) = items.get(i) else { break };
                            ctx.env.insert(var.id, item);
                            if !self.loop_body(body, ctx)? {
                                break;
                            }
                        }
                    }
                }
                Ok(Value::Unit)
            }
            ExprKind::Index { base, index } => {
                let (base, i) = (self.eval(base, ctx)?, self.eval(index, ctx)?);
                let Value::Int(i) = i else {
                    return fail("indexing needs an integer index");
                };
                let out_of_bounds = |len: usize| {
                    Ctrl::Fail(RuntimeError::new(format!(
                        "index out of bounds: the length is {len} but the index is {i}"
                    )))
                };
                match base {
                    Value::Vec(items) => usize::try_from(i)
                        .ok()
                        .and_then(|i| items.get(i))
                        .ok_or_else(|| out_of_bounds(items.len())),
                    Value::Table(t) => match usize::try_from(i) {
                        Ok(i) if i < t.rows() => Ok(t.row(i)?),
                        _ => Err(out_of_bounds(t.rows())),
                    },
                    Value::Tensor(t) => match (usize::try_from(i), t.shape.split_first()) {
                        (Ok(i), Some((&n, rest))) if i < n => Ok(t.element(i, rest)?),
                        (_, Some((&n, _))) => Err(out_of_bounds(n)),
                        _ => fail("cannot index a scalar tensor"),
                    },
                    _ => fail("indexing needs a Vec or a Table"),
                }
            }
            ExprKind::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(self.eval(it, ctx)?);
                }
                Ok(Value::vec(out))
            }

            _ => unreachable!("only called for mutation, loops and arrays"),
        }
    }

    fn eval(&self, e: &hir::Expr, ctx: &mut Ctx) -> Eval {
        use std::sync::atomic::Ordering::Relaxed;
        let fuel = self.fuel.load(Relaxed);
        if fuel != u64::MAX {
            if fuel == 0 {
                return fail("execution step limit exceeded");
            }
            self.fuel.store(fuel - 1, Relaxed);
        }
        match &e.kind {
            ExprKind::Error => fail("cannot run code that failed to type-check"),
            ExprKind::Assign { .. }
            | ExprKind::While { .. }
            | ExprKind::For { .. }
            | ExprKind::Index { .. }
            | ExprKind::Array(_) => self.eval_mutation_and_loops(e, ctx),
            ExprKind::Lit(l) => Ok(match l {
                Lit::Int(n) => match &e.ty {
                    Ty::Prim(p) if p.is_float() => round_to(&e.ty, Value::Float(*n as f64)),
                    _ => Value::Int(
                        i64::try_from(*n)
                            .map_err(|_| RuntimeError::new("integer literal too large"))?,
                    ),
                },
                Lit::Float(x) => round_to(&e.ty, Value::Float(*x)),
                Lit::Str(s) => Value::str(s.as_str()),
                Lit::Bool(b) => Value::Bool(*b),
            }),
            ExprKind::Local(id) => ctx
                .env
                .get(id)
                .cloned()
                .ok_or_else(|| Ctrl::Fail(RuntimeError::new("unbound local"))),
            ExprKind::Global(name) => self
                .globals
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    Ctrl::Fail(RuntimeError::new(format!(
                        "global `{name}` is not initialised yet"
                    )))
                }),
            ExprKind::Fn(name) => Ok(Value::Func(name.as_str().into())),
            ExprKind::Extern(path) => {
                fail(format!("`{}` cannot be used as a value", path.join("::")))
            }
            ExprKind::FString(parts) => {
                let mut out = String::new();
                for p in parts {
                    match p {
                        hir::FPart::Text(t) => out.push_str(t),
                        hir::FPart::Expr(x) => out.push_str(&self.eval(x, ctx)?.to_string()),
                    }
                }
                Ok(Value::str(out))
            }
            ExprKind::Field { base, name } => match self.eval(base, ctx)? {
                Value::Table(t) => Ok(t.column(name)?),
                Value::Struct(s) => s
                    .fields
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| {
                        Ctrl::Fail(RuntimeError::new(format!(
                            "{} has no field `{name}`",
                            s.name
                        )))
                    }),
                Value::Tuple(items) => name
                    .parse::<usize>()
                    .ok()
                    .and_then(|i| items.get(i).cloned())
                    .ok_or_else(|| {
                        Ctrl::Fail(RuntimeError::new(format!("tuple has no part `{name}`")))
                    }),
                other => fail(format!("{} has no field `{name}`", other.type_name())),
            },
            ExprKind::DynCall {
                trait_name,
                method,
                args,
            } => {
                let mut values = Vec::with_capacity(args.len());
                for a in args {
                    values.push(self.eval(a, ctx)?);
                }
                let type_name = match values.first() {
                    Some(Value::Struct(s)) => s.name.clone(),
                    Some(Value::Enum(e)) => e.name.clone(),
                    Some(other) => {
                        return fail(format!(
                            "{}.{method}(): a {} does not implement `{trait_name}`",
                            other.type_name(),
                            other.type_name()
                        ));
                    }
                    None => return fail("a trait method needs a receiver"),
                };
                // The type's own method, else the default one the trait provides.
                let f = self
                    .fns
                    .get(&format!("{type_name}::{method}"))
                    .or_else(|| self.fns.get(&format!("{trait_name}::{method}")))
                    .ok_or_else(|| {
                        Ctrl::Fail(RuntimeError::new(format!(
                            "`{type_name}` has no method `{method}` for the trait `{trait_name}`"
                        )))
                    })?;
                self.call_fn(f, values, ctx.depth)
            }
            ExprKind::Tuple(items) => {
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values.push(self.eval(item, ctx)?);
                }
                Ok(Value::Tuple(Arc::new(values)))
            }
            ExprKind::Call { callee, args } => self.call_expr(callee, args, &e.ty, ctx),
            ExprKind::Unary { op, expr } => match (op, self.eval(expr, ctx)?) {
                (UnOp::Neg, Value::Int(i)) => i.checked_neg().map(Value::Int).ok_or_else(overflow),
                (UnOp::Neg, Value::Float(x)) => Ok(Value::Float(-x)),
                (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                (op, v) => fail(format!("cannot apply {op:?} to {}", v.type_name())),
            },
            ExprKind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, &e.ty, ctx),
            ExprKind::Closure { params, body } => Ok(Value::Closure(Arc::new(Closure {
                params: params.iter().map(|p| p.id).collect::<Vec<LocalId>>(),
                body: (**body).clone(),
                env: ctx.env.clone(),
            }))),
            ExprKind::StructLit { name, fields } => {
                let decl = self.structs.get(name).ok_or_else(|| {
                    Ctrl::Fail(RuntimeError::new(format!("unknown struct `{name}`")))
                })?;
                let mut evaluated: HashMap<&str, Value> = HashMap::new();
                for (fname, fe) in fields {
                    evaluated.insert(fname.as_str(), self.eval(fe, ctx)?);
                }
                // Keep declaration order so JSON output is stable.
                let ordered = decl
                    .iter()
                    .map(|(n, _)| {
                        evaluated
                            .remove(n.as_str())
                            .map(|v| (n.clone(), v))
                            .ok_or_else(|| {
                                Ctrl::Fail(RuntimeError::new(format!("missing field `{n}`")))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Struct(Arc::new(StructVal {
                    name: name.clone(),
                    fields: ordered,
                })))
            }
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => match self.eval(cond, ctx)? {
                Value::Bool(true) => self.block(then, ctx),
                Value::Bool(false) => match otherwise {
                    Some(o) => self.eval(o, ctx),
                    None => Ok(Value::Unit),
                },
                other => fail(format!("`if` condition is {}, not bool", other.type_name())),
            },
            ExprKind::Block(b) => self.block(b, ctx),
            ExprKind::Return(v) => {
                let v = match v {
                    Some(v) => self.eval(v, ctx)?,
                    None => Value::Unit,
                };
                Err(Ctrl::Return(v))
            }
            ExprKind::EnumCtor {
                enum_name,
                variant,
                args,
            } => {
                let mut fields = Vec::with_capacity(args.len());
                for a in args {
                    fields.push(self.eval(a, ctx)?);
                }
                Ok(Value::Enum(Arc::new(crate::value::EnumVal {
                    name: enum_name.clone(),
                    variant: variant.clone(),
                    fields,
                })))
            }
            ExprKind::Match { scrutinee, arms } => {
                let value = self.eval(scrutinee, ctx)?;
                for arm in arms {
                    if match_pattern(&arm.pat, &value, &mut ctx.env) {
                        return self.eval(&arm.body, ctx);
                    }
                }
                fail("no `match` arm matched")
            }
            ExprKind::Try(inner) => match self.eval(inner, ctx)? {
                Value::Ok(v) => Ok(*v),
                Value::Err(e) => Err(Ctrl::Return(Value::Err(e))),
                Value::Enum(e) if e.name == "Option" => match e.fields.first() {
                    Some(v) => Ok(v.clone()),
                    None => Err(Ctrl::Return(Value::none())),
                },
                other => fail(format!(
                    "`?` needs a Result or an Option, found {}",
                    other.type_name()
                )),
            },
            ExprKind::Break => Err(Ctrl::Break),
            ExprKind::Continue => Err(Ctrl::Continue),
        }
    }

    fn call_expr(
        &self,
        callee: &hir::Expr,
        args: &[hir::Expr],
        result: &Ty,
        ctx: &mut Ctx,
    ) -> Eval {
        // Evaluate arguments left to right, before dispatch.
        let eval_args = |this: &Self, ctx: &mut Ctx| {
            args.iter()
                .map(|a| this.eval(a, ctx))
                .collect::<Result<Vec<_>, _>>()
        };

        match &callee.kind {
            ExprKind::Fn(name) => {
                let args = eval_args(self, ctx)?;
                let f = self.fns.get(name).ok_or_else(|| {
                    Ctrl::Fail(RuntimeError::new(format!("no function `{name}`")))
                })?;
                self.call_fn(f, args, ctx.depth)
            }
            ExprKind::Extern(path) => {
                #[cfg(feature = "gpu")]
                if let Some(v) = self.gpu_chain(path, args, result, ctx)? {
                    return Ok(v);
                }
                let mut args = eval_args(self, ctx)?;
                match path.as_slice() {
                    [n] if n == "Ok" || n == "Err" => {
                        let v = Box::new(args.remove(0));
                        Ok(if n == "Ok" {
                            Value::Ok(v)
                        } else {
                            Value::Err(v)
                        })
                    }
                    _ => Ok(std_impl::call_function(
                        self,
                        &path.join("::"),
                        args,
                        result,
                    )?),
                }
            }
            ExprKind::Field { base, name } => {
                let receiver = self.eval(base, ctx)?;
                let args = eval_args(self, ctx)?;
                Ok(std_impl::call_method(&receiver, name, args)?)
            }
            _ => {
                let f = self.eval(callee, ctx)?;
                let args = eval_args(self, ctx)?;
                self.apply_at(&f, args, ctx.depth)
            }
        }
    }

    #[inline(never)]
    fn binary(&self, op: BinOp, lhs: &hir::Expr, rhs: &hir::Expr, ty: &Ty, ctx: &mut Ctx) -> Eval {
        self.binary_exact(op, lhs, rhs, ctx)
            .map(|v| round_to(ty, v))
    }

    fn binary_exact(&self, op: BinOp, lhs: &hir::Expr, rhs: &hir::Expr, ctx: &mut Ctx) -> Eval {
        let l = self.eval(lhs, ctx)?;
        // `&&` and `||` short-circuit.
        match (op, &l) {
            (BinOp::And, Value::Bool(false)) => return Ok(Value::Bool(false)),
            (BinOp::Or, Value::Bool(true)) => return Ok(Value::Bool(true)),
            _ => {}
        }
        let r = self.eval(rhs, ctx)?;
        use BinOp::*;
        match (op, &l, &r) {
            (And | Or, Value::Bool(_), Value::Bool(b)) => Ok(Value::Bool(*b)),
            (Eq, a, b) => Ok(Value::Bool(a.equals(b))),
            (NotEq, a, b) => Ok(Value::Bool(!a.equals(b))),
            (Add, Value::Str(a), Value::Str(b)) => Ok(Value::str(format!("{a}{b}"))),
            (_, Value::Int(a), Value::Int(b)) => int_op(op, *a, *b),
            (_, Value::Float(a), Value::Float(b)) => float_op(op, *a, *b),
            (_, Value::Int(a), Value::Float(b)) => float_op(op, *a as f64, *b),
            (_, Value::Float(a), Value::Int(b)) => float_op(op, *a, *b as f64),
            (Lt | LtEq | Gt | GtEq, Value::Str(a), Value::Str(b)) => Ok(Value::Bool(match op {
                Lt => a < b,
                LtEq => a <= b,
                Gt => a > b,
                _ => a >= b,
            })),
            _ => fail(format!(
                "cannot apply {op:?} to {} and {}",
                l.type_name(),
                r.type_name()
            )),
        }
    }
}

/// One resolved step of an assignment target.
enum Step<'a> {
    Field(&'a str),
    Index(i64),
}

/// Returns `current` with the place described by `steps` replaced by `new`. Structs are values
/// (rebuilt on the way back up); vectors are shared, so their elements are updated in place.
fn update_place(current: Value, steps: &[Step], new: Value) -> Result<Value, Ctrl> {
    let Some((step, rest)) = steps.split_first() else {
        return Ok(new);
    };
    match (step, current) {
        (Step::Field(name), Value::Struct(s)) => {
            let mut fields = s.fields.clone();
            let slot = fields.iter_mut().find(|(n, _)| n == name).ok_or_else(|| {
                Ctrl::Fail(RuntimeError::new(format!(
                    "{} has no field `{name}`",
                    s.name
                )))
            })?;
            let old = std::mem::replace(&mut slot.1, Value::Unit);
            slot.1 = update_place(old, rest, new)?;
            Ok(Value::Struct(Arc::new(StructVal {
                name: s.name.clone(),
                fields,
            })))
        }
        (Step::Index(i), Value::Vec(items)) => {
            let out_of_bounds = || {
                Ctrl::Fail(RuntimeError::new(format!(
                    "index out of bounds: the length is {} but the index is {i}",
                    items.len()
                )))
            };
            let idx = usize::try_from(*i).map_err(|_| out_of_bounds())?;
            let old = items.get(idx).ok_or_else(out_of_bounds)?;
            let updated = update_place(old, rest, new)?;
            items.set(idx, updated);
            Ok(Value::Vec(items))
        }
        (_, other) => fail(format!("cannot assign through a {}", other.type_name())),
    }
}

/// `f32` values are stored as `f64`; rounding after every operation keeps `f32` arithmetic exactly
/// what native code computes.
fn round_to(ty: &Ty, v: Value) -> Value {
    match (ty, &v) {
        (Ty::Prim(novar_hir::Prim::F32), Value::Float(x)) => Value::Float(f64::from(*x as f32)),
        _ => v,
    }
}

fn overflow() -> Ctrl {
    Ctrl::Fail(RuntimeError::new("integer overflow"))
}

fn int_op(op: BinOp, a: i64, b: i64) -> Eval {
    use BinOp::*;
    Ok(match op {
        Add => Value::Int(a.checked_add(b).ok_or_else(overflow)?),
        Sub => Value::Int(a.checked_sub(b).ok_or_else(overflow)?),
        Mul => Value::Int(a.checked_mul(b).ok_or_else(overflow)?),
        Div | Rem if b == 0 => return fail("division by zero"),
        Div => Value::Int(a.checked_div(b).ok_or_else(overflow)?),
        Rem => Value::Int(a.checked_rem(b).ok_or_else(overflow)?),
        Lt => Value::Bool(a < b),
        LtEq => Value::Bool(a <= b),
        Gt => Value::Bool(a > b),
        GtEq => Value::Bool(a >= b),
        _ => return fail("invalid integer operation"),
    })
}

fn float_op(op: BinOp, a: f64, b: f64) -> Eval {
    use BinOp::*;
    Ok(match op {
        Add => Value::Float(a + b),
        Sub => Value::Float(a - b),
        Mul => Value::Float(a * b),
        Div => Value::Float(a / b),
        Rem => Value::Float(a % b),
        Lt => Value::Bool(a < b),
        LtEq => Value::Bool(a <= b),
        Gt => Value::Bool(a > b),
        GtEq => Value::Bool(a >= b),
        _ => return fail("invalid float operation"),
    })
}

/// Matches `value` against `pat`, binding names in `env`. Bindings made before a later part of
/// the pattern fails are left behind; the arm that follows rebinds what it uses.
fn match_pattern(pat: &hir::Pat, value: &Value, env: &mut Env) -> bool {
    match (pat, value) {
        (hir::Pat::Wild, _) => true,
        (hir::Pat::Bind(local), v) => {
            env.insert(local.id, v.clone());
            true
        }
        (hir::Pat::Tuple(parts), Value::Tuple(items)) => {
            parts.len() == items.len()
                && parts
                    .iter()
                    .zip(items.iter())
                    .all(|(p, v)| match_pattern(p, v, env))
        }
        (hir::Pat::Lit(l), v) => match (l, v) {
            (Lit::Int(a), Value::Int(b)) => i128::from(*b) == i128::from(*a),
            (Lit::Float(a), Value::Float(b)) => a == b,
            (Lit::Int(a), Value::Float(b)) => *a as f64 == *b,
            (Lit::Str(a), Value::Str(b)) => a.as_str() == &**b,
            (Lit::Bool(a), Value::Bool(b)) => a == b,
            _ => false,
        },
        (
            hir::Pat::Variant {
                enum_name: None,
                variant,
                args,
            },
            Value::Ok(v),
        ) if variant == "Ok" => args.first().is_some_and(|p| match_pattern(p, v, env)),
        (
            hir::Pat::Variant {
                enum_name: None,
                variant,
                args,
            },
            Value::Err(v),
        ) if variant == "Err" => args.first().is_some_and(|p| match_pattern(p, v, env)),
        (
            hir::Pat::Variant {
                enum_name: Some(en),
                variant,
                args,
            },
            Value::Enum(e),
        ) => {
            e.name == *en
                && e.variant == *variant
                && args.len() == e.fields.len()
                && args
                    .iter()
                    .zip(&e.fields)
                    .all(|(p, v)| match_pattern(p, v, env))
        }
        _ => false,
    }
}
