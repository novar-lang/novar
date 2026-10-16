//! The places in a program that `telvax test --coverage` counts: each statement, the last
//! expression of each block and the body of each `case`. The interpreter marks them as they run.

use crate::hir::{Block, Expr, ExprKind, FPart, ForIter, Program, Proj, Stmt};

/// Where a statement starts, as the interpreter reports it when the statement runs.
pub fn stmt_start(s: &Stmt) -> usize {
    match s {
        Stmt::Let { init, .. } => init.span.start,
        Stmt::Expr(e) => e.span.start,
    }
}

/// The start offset of every point in the functions for which `skip(name)` is false.
pub fn points(program: &Program, skip: impl Fn(&str) -> bool) -> Vec<usize> {
    let mut out = Vec::new();
    for f in program.fns.iter().filter(|f| !skip(&f.name)) {
        block(&f.body, &mut out);
    }
    out
}

fn block(b: &Block, out: &mut Vec<usize>) {
    for s in &b.stmts {
        out.push(stmt_start(s));
        match s {
            Stmt::Let { init, .. } => expr(init, out),
            Stmt::Expr(e) => expr(e, out),
        }
    }
    if let Some(t) = &b.tail {
        out.push(t.span.start);
        expr(t, out);
    }
}

fn expr(e: &Expr, out: &mut Vec<usize>) {
    match &e.kind {
        ExprKind::Call { callee, args } => {
            expr(callee, out);
            args.iter().for_each(|a| expr(a, out));
        }
        ExprKind::Field { base: x, .. }
        | ExprKind::Unary { expr: x, .. }
        | ExprKind::Try(x)
        | ExprKind::Await(x)
        | ExprKind::Return(Some(x)) => expr(x, out),
        ExprKind::Binary { lhs, rhs, .. }
        | ExprKind::Index {
            base: lhs,
            index: rhs,
        } => {
            expr(lhs, out);
            expr(rhs, out);
        }
        ExprKind::Closure { body, .. } => expr(body, out),
        ExprKind::StructLit { fields, .. } => fields.iter().for_each(|(_, f)| expr(f, out)),
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
        ExprKind::Tuple(items) | ExprKind::Array(items) => items.iter().for_each(|i| expr(i, out)),
        ExprKind::DynCall { args, .. } | ExprKind::EnumCtor { args, .. } => {
            args.iter().for_each(|a| expr(a, out))
        }
        ExprKind::Match { scrutinee, arms } => {
            expr(scrutinee, out);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    expr(g, out);
                }
                out.push(arm.body.span.start);
                expr(&arm.body, out);
            }
        }
        ExprKind::FString(parts) => parts
            .iter()
            .flat_map(FPart::exprs)
            .for_each(|p| expr(p, out)),
        ExprKind::Assign { place, value } => {
            for p in &place.projs {
                if let Proj::Index(i) = p {
                    expr(i, out);
                }
            }
            expr(value, out);
        }
        ExprKind::While { cond, body } => {
            expr(cond, out);
            block(body, out);
        }
        ExprKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range { start, end, .. } => {
                    expr(start, out);
                    expr(end, out);
                }
                ForIter::Vec(v) => expr(v, out),
            }
            block(body, out);
        }
        ExprKind::Lit(_)
        | ExprKind::Local(_)
        | ExprKind::Global(_)
        | ExprKind::Fn(_)
        | ExprKind::Extern(_)
        | ExprKind::Return(None)
        | ExprKind::Break
        | ExprKind::Continue
        | ExprKind::Error => {}
    }
}
