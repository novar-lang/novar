//! Tree-walking evaluator over the typed HIR.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use novar_hir::Ty;
use novar_hir::hir::{self, BinOp, ExprKind, Lit, LocalId, UnOp};
use serde_json::Value as Json;

use crate::std_impl;
use crate::value::{Closure, Db, Env, StructVal, Value, from_json};

/// Maximum nesting of function calls before evaluation aborts.
const MAX_DEPTH: usize = 128;

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
    globals: RwLock<HashMap<String, Value>>,
    dbs: Mutex<HashMap<String, Arc<Db>>>,
    out: Mutex<Box<dyn std::io::Write + Send>>,
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
        let interp = Interp {
            structs: program
                .structs
                .iter()
                .map(|s| (s.name.clone(), s.fields.clone()))
                .collect(),
            fns: program
                .fns
                .into_iter()
                .map(|f| (f.name.clone(), f))
                .collect(),
            globals: RwLock::new(HashMap::new()),
            dbs: Mutex::new(HashMap::new()),
            out: Mutex::new(Box::new(std::io::stdout())),
        };
        for g in &program.globals {
            let mut ctx = Ctx {
                env: Env::new(),
                depth: 0,
            };
            let v = interp.eval(&g.init, &mut ctx).map_err(|c| match c {
                Ctrl::Fail(e) => RuntimeError::new(format!("while initialising `{}`: {e}", g.name)),
                Ctrl::Return(_) => RuntimeError::new("`return` outside a function"),
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

    /// The in-memory database behind `url`, created on first use. Lets embedders seed data.
    pub fn db_for(&self, url: &str) -> Arc<Db> {
        let mut dbs = self.dbs.lock().unwrap_or_else(|e| e.into_inner());
        dbs.entry(url.to_string())
            .or_insert_with(|| {
                Arc::new(Db {
                    url: url.to_string(),
                    ..Db::default()
                })
            })
            .clone()
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
                    from_json(&p.ty, payload, &self.structs).map_err(|e| EndpointError {
                        status: 400,
                        message: e,
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
        })
    }

    fn apply_at(&self, f: &Value, args: Vec<Value>, depth: usize) -> Eval {
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
        self.eval(&c.body, &mut ctx)
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
        match self.block(&f.body, &mut ctx) {
            Err(Ctrl::Return(v)) => Ok(v),
            other => other,
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
                            self.block(body, ctx)?;
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
                            self.block(body, ctx)?;
                            i += 1;
                        }
                    }
                    hir::ForIter::Vec(v) => {
                        let Value::Vec(items) = self.eval(v, ctx)? else {
                            return fail("`for` needs a Vec");
                        };
                        // Live view: element assignments inside the loop are visible to later
                        // iterations. The length is fixed when the loop starts.
                        for i in 0..items.len() {
                            let Some(item) = items.get(i) else { break };
                            ctx.env.insert(var.id, item);
                            self.block(body, ctx)?;
                        }
                    }
                }
                Ok(Value::Unit)
            }
            ExprKind::Index { base, index } => {
                let (Value::Vec(items), Value::Int(i)) =
                    (self.eval(base, ctx)?, self.eval(index, ctx)?)
                else {
                    return fail("indexing needs a Vec and an integer");
                };
                usize::try_from(i)
                    .ok()
                    .and_then(|i| items.get(i))
                    .ok_or_else(|| {
                        Ctrl::Fail(RuntimeError::new(format!(
                            "index out of bounds: the length is {} but the index is {i}",
                            items.len()
                        )))
                    })
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
        match &e.kind {
            ExprKind::Error => fail("cannot run code that failed to type-check"),
            ExprKind::Assign { .. }
            | ExprKind::While { .. }
            | ExprKind::For { .. }
            | ExprKind::Index { .. }
            | ExprKind::Array(_) => self.eval_mutation_and_loops(e, ctx),
            ExprKind::Lit(l) => Ok(match l {
                Lit::Int(n) => match &e.ty {
                    Ty::Prim(p) if p.is_float() => Value::Float(*n as f64),
                    _ => Value::Int(
                        i64::try_from(*n)
                            .map_err(|_| RuntimeError::new("integer literal too large"))?,
                    ),
                },
                Lit::Float(x) => Value::Float(*x),
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
            ExprKind::Fn(name) => fail(format!("function `{name}` cannot be used as a value yet")),
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
                other => fail(format!("{} has no field `{name}`", other.type_name())),
            },
            ExprKind::Call { callee, args } => self.call_expr(callee, args, ctx),
            ExprKind::Unary { op, expr } => match (op, self.eval(expr, ctx)?) {
                (UnOp::Neg, Value::Int(i)) => i.checked_neg().map(Value::Int).ok_or_else(overflow),
                (UnOp::Neg, Value::Float(x)) => Ok(Value::Float(-x)),
                (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                (op, v) => fail(format!("cannot apply {op:?} to {}", v.type_name())),
            },
            ExprKind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, ctx),
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
        }
    }

    fn call_expr(&self, callee: &hir::Expr, args: &[hir::Expr], ctx: &mut Ctx) -> Eval {
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
                    _ => Ok(std_impl::call_function(self, &path.join("::"), args)?),
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

    fn binary(&self, op: BinOp, lhs: &hir::Expr, rhs: &hir::Expr, ctx: &mut Ctx) -> Eval {
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
