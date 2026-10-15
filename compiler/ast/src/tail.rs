//! The value of a function: its last expression, with or without `return`; through `if` / `elif` /
//! `else` and `match` it is the value of each branch. Early `return`s stay `return`s.

use crate::*;

/// Gives `f` its value: the final `return value` or final expression of the body.
pub fn finish_fn(f: &mut Fn) {
    if f.ret.is_some() {
        tailify_block(&mut f.body);
    }
}

/// Whether an expression statement can be the value of its block.
fn is_value(e: &Expr) -> bool {
    matches!(
        e.node,
        ExprKind::Call { .. }
            | ExprKind::Path(_)
            | ExprKind::Lit(_)
            | ExprKind::Binary { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Field { .. }
            | ExprKind::Index { .. }
            | ExprKind::Pipe { .. }
            | ExprKind::Tuple(_)
            | ExprKind::Array(_)
            | ExprKind::FString(_)
            | ExprKind::StructLit { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Try(_)
            | ExprKind::Await(_)
            | ExprKind::If { .. }
    )
}

/// Moves a final `return value` or final expression into the block's value, looking through a
/// final `if` / `match`.
pub fn tailify_block(b: &mut Block) {
    match b.stmts.last() {
        Some(Stmt::Expr(Spanned {
            node: ExprKind::Return(Some(_)),
            ..
        })) => {
            if let Some(Stmt::Expr(Spanned {
                node: ExprKind::Return(value),
                ..
            })) = b.stmts.pop()
            {
                b.tail = value;
            }
        }
        Some(Stmt::Expr(Spanned {
            node:
                ExprKind::If {
                    otherwise: Some(_), ..
                }
                | ExprKind::Match { .. },
            ..
        })) => {
            if let Some(Stmt::Expr(mut branching)) = b.stmts.pop() {
                tailify_branches(&mut branching);
                b.tail = Some(Box::new(branching));
            }
        }
        Some(Stmt::Expr(e)) if is_value(e) => {
            if let Some(Stmt::Expr(value)) = b.stmts.pop() {
                b.tail = Some(Box::new(value));
            }
        }
        _ => {}
    }
}

fn tailify_branches(e: &mut Expr) {
    match &mut e.node {
        ExprKind::If {
            then, otherwise, ..
        } => {
            tailify_block(then);
            if let Some(o) = otherwise {
                tailify_branches(o);
            }
        }
        ExprKind::Block(b) => tailify_block(b),
        ExprKind::Match { arms, .. } => arms.iter_mut().for_each(|a| tailify_branches(&mut a.body)),
        _ => {}
    }
}

/// Gives a `match` (or `if`) used as a value the value of each branch: the last line of every arm.
pub fn tailify_value(e: &mut Expr) {
    tailify_branches(e);
}
