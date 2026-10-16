//! What a debugger needs from the interpreter: a call before each statement, the stack of calls
//! at that point, and the variables of each call with their values.
//!
//! A debugger implements [`Debugger`] and installs it with [`Interp::set_debugger`]. The
//! interpreter calls `at_statement` on the thread that runs the program; the debugger decides
//! whether to stop, and stopping is just not returning until the user continues.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use telvax_hir::Ty;
use telvax_hir::hir::{self, ExprKind};

use crate::Interp;
use crate::value::{Env, MapKey, Value};

/// Called by the interpreter before each statement while it is installed.
pub trait Debugger: Send + Sync {
    /// `stack` holds the calls in progress on this thread, the innermost last. The innermost
    /// frame's `offset` is where the statement about to run starts.
    fn at_statement(&self, interp: &Interp, stack: &[Frame]);
}

/// One call in progress.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The function's name as the program knows it (`util__double` for an imported one).
    pub function: String,
    /// Where the statement the call is at starts (a byte offset in the program's source map).
    pub offset: usize,
    env: Env,
}

/// A variable with its type and value.
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub ty: Ty,
    pub value: Value,
}

/// A name the function declares.
#[derive(Debug, Clone)]
pub(crate) struct LocalDecl {
    id: hir::LocalId,
    name: String,
    ty: Ty,
    /// Where the name is written; the variable exists from there on.
    at: usize,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

/// Keeps a call on the stack for as long as it lives.
pub(crate) struct CallGuard;

impl CallGuard {
    pub(crate) fn enter(function: &str) -> CallGuard {
        STACK.with(|s| {
            s.borrow_mut().push(Frame {
                function: function.to_string(),
                offset: 0,
                env: Env::new(),
            });
        });
        CallGuard
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Records where the innermost call is and lets the debugger look.
pub(crate) fn statement(interp: &Interp, debugger: &dyn Debugger, offset: usize, env: &Env) {
    let known = STACK.with(|s| {
        let mut s = s.borrow_mut();
        match s.last_mut() {
            Some(top) => {
                top.offset = offset;
                top.env = env.clone();
                true
            }
            None => false,
        }
    });
    if known {
        // The debugger may block here for as long as the user looks around.
        STACK.with(|s| debugger.at_statement(interp, &s.borrow()));
    }
}

impl Interp {
    /// Installs (or, with `None`, removes) the debugger that is called before each statement.
    pub fn set_debugger(&self, debugger: Option<Arc<dyn Debugger>>) {
        use std::sync::atomic::Ordering::Relaxed;
        self.debugging.store(debugger.is_some(), Relaxed);
        *self.debugger.write().unwrap_or_else(|e| e.into_inner()) = debugger;
    }

    /// The variables of `frame` that exist at the statement it is at, in the order declared.
    pub fn locals(&self, frame: &Frame) -> Vec<Variable> {
        let table = self.local_decls();
        let Some(decls) = table.get(&frame.function) else {
            return Vec::new();
        };
        decls
            .iter()
            .filter(|d| d.at <= frame.offset)
            .filter_map(|d| {
                frame.env.get(&d.id).map(|v| Variable {
                    name: d.name.clone(),
                    ty: d.ty.clone(),
                    value: v.clone(),
                })
            })
            .collect()
    }

    /// The global `let`s of the program, by name.
    pub fn global_variables(&self) -> Vec<Variable> {
        let globals = self.globals.read().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<Variable> = globals
            .iter()
            .map(|(name, value)| Variable {
                name: name.clone(),
                ty: Ty::Unknown,
                value: value.clone(),
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// `v` as the program would print it inside something else, knowing its type when given.
    pub fn render(&self, v: &Value, ty: Option<&Ty>) -> String {
        match ty {
            Some(ty) => self.show(v, ty, true),
            None => {
                let env = crate::show::TypeEnv {
                    structs: Some(&self.structs),
                    enums: Some(&self.enums),
                };
                crate::show::show(v, None, &env, true)
            }
        }
    }

    fn local_decls(&self) -> &HashMap<String, Vec<LocalDecl>> {
        self.local_decls.get_or_init(|| {
            self.fns
                .iter()
                .map(|(name, f)| {
                    let mut out: Vec<LocalDecl> = Vec::new();
                    for p in &f.params {
                        out.push(decl(p));
                    }
                    block(&f.body, &mut out);
                    out.sort_by_key(|d| d.at);
                    (name.clone(), out)
                })
                .collect()
        })
    }
}

/// What can be looked into in `v`: the elements of a list, the entries of a map, the fields of a
/// type, the values of a variant, the items of a tuple. Empty for a plain value.
pub fn children(v: &Value) -> Vec<(String, Value)> {
    match v {
        Value::Vec(items) => items
            .snapshot()
            .into_iter()
            .enumerate()
            .map(|(i, x)| (i.to_string(), x))
            .collect(),
        Value::Map(m) => m
            .entries()
            .into_iter()
            .map(|(k, x)| (key_text(&k), x))
            .collect(),
        Value::Struct(s) => s.fields.clone(),
        Value::Enum(e) => e
            .fields
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (i.to_string(), x))
            .collect(),
        Value::Tuple(items) => items
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (i.to_string(), x))
            .collect(),
        Value::Ok(x) | Value::Err(x) => vec![("0".to_string(), (**x).clone())],
        _ => Vec::new(),
    }
}

fn key_text(k: &MapKey) -> String {
    match k {
        MapKey::Str(s) => format!("{:?}", &**s),
        other => other.json_key(),
    }
}

fn decl(l: &hir::Local) -> LocalDecl {
    LocalDecl {
        id: l.id,
        name: l.name.clone(),
        ty: l.ty.clone(),
        at: l.span.start,
    }
}

fn block(b: &hir::Block, out: &mut Vec<LocalDecl>) {
    for s in &b.stmts {
        match s {
            hir::Stmt::Let { local, init } => {
                out.push(decl(local));
                expr(init, out);
            }
            hir::Stmt::Expr(e) => expr(e, out),
        }
    }
    if let Some(t) = &b.tail {
        expr(t, out);
    }
}

fn pattern(p: &hir::Pat, out: &mut Vec<LocalDecl>) {
    match p {
        hir::Pat::Bind(l) => out.push(decl(l)),
        hir::Pat::Variant { args, .. } | hir::Pat::Tuple(args) | hir::Pat::Or(args) => {
            args.iter().for_each(|a| pattern(a, out));
        }
        hir::Pat::List {
            before,
            rest,
            after,
        } => before
            .iter()
            .chain(rest.as_deref())
            .chain(after)
            .for_each(|a| pattern(a, out)),
        hir::Pat::Struct { fields, .. } => fields.iter().for_each(|(_, a)| pattern(a, out)),
        hir::Pat::Wild | hir::Pat::Lit(_) | hir::Pat::NegInt(_) => {}
    }
}

fn expr(e: &hir::Expr, out: &mut Vec<LocalDecl>) {
    match &e.kind {
        ExprKind::Lit(_)
        | ExprKind::Local(_)
        | ExprKind::Global(_)
        | ExprKind::Fn(_)
        | ExprKind::Extern(_)
        | ExprKind::Break
        | ExprKind::Continue
        | ExprKind::Error => {}
        ExprKind::Call { callee, args } => {
            expr(callee, out);
            args.iter().for_each(|a| expr(a, out));
        }
        ExprKind::Field { base, .. } => expr(base, out),
        ExprKind::Unary { expr: x, .. } | ExprKind::Try(x) | ExprKind::Await(x) => expr(x, out),
        ExprKind::Binary { lhs, rhs, .. } => {
            expr(lhs, out);
            expr(rhs, out);
        }
        ExprKind::Closure { params, body } => {
            params.iter().for_each(|p| out.push(decl(p)));
            expr(body, out);
        }
        ExprKind::StructLit { fields, .. } => fields.iter().for_each(|(_, v)| expr(v, out)),
        ExprKind::If {
            cond,
            then,
            otherwise,
        } => {
            expr(cond, out);
            block(then, out);
            if let Some(o) = otherwise {
                expr(o, out);
            }
        }
        ExprKind::Block(b) => block(b, out),
        ExprKind::Return(v) => {
            if let Some(v) = v {
                expr(v, out);
            }
        }
        ExprKind::Tuple(items) | ExprKind::Array(items) => items.iter().for_each(|i| expr(i, out)),
        ExprKind::DynCall { args, .. } | ExprKind::EnumCtor { args, .. } => {
            args.iter().for_each(|a| expr(a, out));
        }
        ExprKind::Match { scrutinee, arms } => {
            expr(scrutinee, out);
            for arm in arms {
                pattern(&arm.pat, out);
                if let Some(g) = &arm.guard {
                    expr(g, out);
                }
                expr(&arm.body, out);
            }
        }
        ExprKind::FString(parts) => {
            for p in parts {
                p.exprs().into_iter().for_each(|x| expr(x, out));
            }
        }
        ExprKind::Assign { place, value } => {
            for pr in &place.projs {
                if let hir::Proj::Index(i) = pr {
                    expr(i, out);
                }
            }
            expr(value, out);
        }
        ExprKind::While { cond, body } => {
            expr(cond, out);
            block(body, out);
        }
        ExprKind::For { var, iter, body } => {
            out.push(decl(var));
            match iter {
                hir::ForIter::Range { start, end, .. } => {
                    expr(start, out);
                    expr(end, out);
                }
                hir::ForIter::Vec(v) => expr(v, out),
            }
            block(body, out);
        }
        ExprKind::Index { base, index } => {
            expr(base, out);
            expr(index, out);
        }
    }
}
