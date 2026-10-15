//! Rewrites that turn the Python-style surface into the tree the rest of the compiler expects.
//!
//! * `x = value` declares `x` the first time it is assigned in a function (as a mutable `let`) and
//!   reassigns it afterwards. Names are scoped by block, like in most typed languages.
//! * A function with a return type ends with `return value`; that becomes the value of the body
//!   (and, through `if` / `elif` / `else` and `match`, the value of each branch).

use std::collections::HashSet;

use telvax_ast::*;

/// Applies both rewrites to `f`. `globals` are the names declared at the top of the file.
pub fn finish_fn(f: &mut Fn, globals: &HashSet<String>) {
    let params: HashSet<String> = f.params.iter().map(|p| p.name.node.clone()).collect();
    let mut scopes = vec![globals.clone(), params];
    block(&mut f.body, &mut scopes);
    if f.ret.is_some() {
        tailify_block(&mut f.body);
    }
}

fn declared(scopes: &[HashSet<String>], name: &str) -> bool {
    scopes.iter().any(|s| s.contains(name))
}

/// The name a plain `name = value` statement would declare, if `name` is new.
fn new_name(e: &Expr, scopes: &[HashSet<String>]) -> Option<Ident> {
    let ExprKind::Assign { target, value } = &e.node else {
        return None;
    };
    let ExprKind::Path(p) = &target.node else {
        return None;
    };
    if p.len() != 1 || declared(scopes, &p[0].node) {
        return None;
    }
    // `x += 1` is `x = x + 1` by now: it reads `x`, so it cannot declare it.
    if matches!(&value.node, ExprKind::Binary { lhs, .. } if **lhs == **target) {
        return None;
    }
    Some(p[0].clone())
}

fn block(b: &mut Block, scopes: &mut Vec<HashSet<String>>) {
    scopes.push(HashSet::new());
    for stmt in &mut b.stmts {
        match stmt {
            Stmt::Let(l) => {
                expr(&mut l.init, scopes);
                if let Some(top) = scopes.last_mut() {
                    top.insert(l.name.node.clone());
                }
            }
            Stmt::Expr(e) => {
                if let Some(name) = new_name(e, scopes) {
                    let span = e.span.clone();
                    let ExprKind::Assign { value, .. } =
                        std::mem::replace(&mut e.node, ExprKind::Error)
                    else {
                        unreachable!()
                    };
                    let mut init = *value;
                    expr(&mut init, scopes);
                    if let Some(top) = scopes.last_mut() {
                        top.insert(name.node.clone());
                    }
                    *stmt = Stmt::Let(Let {
                        mutable: true,
                        name,
                        ty: None,
                        init,
                        span,
                    });
                } else {
                    expr(e, scopes);
                }
            }
        }
    }
    if let Some(t) = &mut b.tail {
        expr(t, scopes);
    }
    scopes.pop();
}

fn binds(p: &Pattern, out: &mut HashSet<String>) {
    match &p.node {
        PatKind::Bind(n) => {
            out.insert(n.clone());
        }
        PatKind::Tuple(items) | PatKind::Or(items) => items.iter().for_each(|i| binds(i, out)),
        PatKind::Variant { args, .. } => args.iter().for_each(|i| binds(i, out)),
        PatKind::List {
            before,
            rest,
            after,
        } => before
            .iter()
            .chain(rest.as_deref())
            .chain(after)
            .for_each(|i| binds(i, out)),
        PatKind::Struct { fields, .. } => fields.iter().for_each(|(_, i)| binds(i, out)),
        PatKind::Wild | PatKind::Lit(_) | PatKind::NegLit(_) => {}
    }
}

fn expr(e: &mut Expr, scopes: &mut Vec<HashSet<String>>) {
    match &mut e.node {
        ExprKind::Lit(_) | ExprKind::Path(_) | ExprKind::Break | ExprKind::Continue => {}
        ExprKind::Error => {}
        ExprKind::Call { callee, args } => {
            expr(callee, scopes);
            args.iter_mut().for_each(|a| expr(a, scopes));
        }
        ExprKind::Field { base, .. } => expr(base, scopes),
        ExprKind::Named { value, .. } => expr(value, scopes),
        ExprKind::Unary { expr: inner, .. } | ExprKind::Try(inner) | ExprKind::Await(inner) => {
            expr(inner, scopes)
        }
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::Pipe { lhs, rhs } => {
            expr(lhs, scopes);
            expr(rhs, scopes);
        }
        ExprKind::Closure { params, body, .. } => {
            scopes.push(params.iter().map(|p| p.node.clone()).collect());
            expr(body, scopes);
            scopes.pop();
        }
        ExprKind::StructLit { fields, .. } => fields.iter_mut().for_each(|(_, v)| expr(v, scopes)),
        ExprKind::If {
            cond,
            then,
            otherwise,
        } => {
            expr(cond, scopes);
            block(then, scopes);
            if let Some(o) = otherwise {
                expr(o, scopes);
            }
        }
        ExprKind::Block(b) => block(b, scopes),
        ExprKind::Return(v) => {
            if let Some(v) = v {
                expr(v, scopes);
            }
        }
        ExprKind::Tuple(items) | ExprKind::Array(items) => {
            items.iter_mut().for_each(|i| expr(i, scopes));
        }
        ExprKind::Match { scrutinee, arms } => {
            expr(scrutinee, scopes);
            for arm in arms {
                let mut names = HashSet::new();
                binds(&arm.pat, &mut names);
                scopes.push(names);
                expr(&mut arm.body, scopes);
                scopes.pop();
            }
        }
        ExprKind::FString(parts) => {
            for p in parts {
                for x in p.exprs_mut() {
                    expr(x, scopes);
                }
            }
        }
        ExprKind::Assign { target, value } => {
            expr(target, scopes);
            expr(value, scopes);
        }
        ExprKind::While { cond, body } => {
            expr(cond, scopes);
            block(body, scopes);
        }
        ExprKind::For { var, iter, body } => {
            match iter {
                ForIter::Range { start, end, .. } => {
                    expr(start, scopes);
                    expr(end, scopes);
                }
                ForIter::Expr(x) => expr(x, scopes),
            }
            scopes.push(HashSet::from([var.node.clone()]));
            block(body, scopes);
            scopes.pop();
        }
        ExprKind::Index { base, index } => {
            expr(base, scopes);
            expr(index, scopes);
        }
    }
}

/// Moves a final `return value` into the block's value, looking through a final `if` / `match`.
pub(crate) fn tailify_block(b: &mut Block) {
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
