//! Names and defaults at the places a function or a type is used.
//!
//! * `search(v, limit: 3)` puts `3` where the parameter `limit` is, and `search(v)` uses the
//!   default of `limit` when it has one.
//! * `Query(text: "x")` fills `top_k` from its default.
//!
//! The pass knows the functions and types written in the program. For a call it cannot match to
//! one of them (a library function, a method of a value whose type is not known yet), names are
//! only labels: they are dropped and the arguments keep the order they were written in.

use std::collections::HashMap;

use telvax_ast::*;
use telvax_diagnostics::Diagnostic;

#[derive(Clone)]
struct Sig {
    params: Vec<(String, Option<Expr>)>,
}

#[derive(Default)]
struct Known {
    fns: HashMap<String, Sig>,
    /// `Type.function` for functions without `self`.
    assoc: HashMap<(String, String), Sig>,
    /// Methods (with `self`) by name; used when only one type has a method of that name.
    methods: HashMap<String, Vec<Sig>>,
    structs: HashMap<String, Vec<(String, Option<Expr>)>>,
}

fn sig_of(f: &Fn) -> (Sig, bool) {
    let has_self = f.params.first().is_some_and(|p| p.name.node == "self");
    let params = f
        .params
        .iter()
        .skip(usize::from(has_self))
        .map(|p| (p.name.node.clone(), p.default.clone()))
        .collect();
    (Sig { params }, has_self)
}

impl Known {
    fn collect(program: &Program) -> Known {
        let mut k = Known::default();
        for item in &program.items {
            match item {
                Item::Fn(f) => {
                    k.fns.insert(f.name.node.clone(), sig_of(f).0);
                }
                Item::Struct(s) => {
                    k.structs.insert(
                        s.name.node.clone(),
                        s.fields
                            .iter()
                            .map(|f| (f.name.node.clone(), f.default.clone()))
                            .collect(),
                    );
                }
                Item::Impl(i) => {
                    for f in &i.fns {
                        let (sig, has_self) = sig_of(f);
                        if has_self {
                            k.methods.entry(f.name.node.clone()).or_default().push(sig);
                        } else {
                            k.assoc
                                .insert((i.ty.node.clone(), f.name.node.clone()), sig);
                        }
                    }
                }
                Item::Trait(t) => {
                    for f in &t.fns {
                        let (sig, has_self) = sig_of(f);
                        if has_self {
                            k.methods.entry(f.name.node.clone()).or_default().push(sig);
                        }
                    }
                }
                _ => {}
            }
        }
        k
    }

    /// The parameters a call to `callee` has, when the program says.
    fn signature(&self, callee: &Expr) -> Option<&Sig> {
        match &callee.node {
            ExprKind::Path(p) if p.len() == 1 => self.fns.get(&p[0].node),
            ExprKind::Path(p) if p.len() == 2 => {
                self.assoc.get(&(p[0].node.clone(), p[1].node.clone()))
            }
            ExprKind::Field { name, .. } => match self.methods.get(&name.node) {
                Some(all) if all.len() == 1 => all.first(),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Applies names and defaults to every call and construction in `program`.
pub fn resolve(program: &mut Program) -> Vec<Diagnostic> {
    let known = Known::collect(program);
    let mut diags = Vec::new();
    let mut visit = |e: &mut Expr| rewrite(e, &known, &mut diags);
    for item in &mut program.items {
        match item {
            Item::Let(l) => walk(&mut l.init, &mut visit),
            Item::Fn(f) => walk_fn(f, &mut visit),
            Item::Impl(i) => i.fns.iter_mut().for_each(|f| walk_fn(f, &mut visit)),
            Item::Trait(t) => t.fns.iter_mut().for_each(|f| walk_fn(f, &mut visit)),
            _ => {}
        }
    }
    diags
}

fn walk_fn(f: &mut Fn, visit: &mut impl FnMut(&mut Expr)) {
    walk_block(&mut f.body, visit);
}

fn rewrite(e: &mut Expr, known: &Known, diags: &mut Vec<Diagnostic>) {
    let span = e.span.clone();
    match &mut e.node {
        ExprKind::Call { callee, args } => {
            let named = args
                .iter()
                .any(|a| matches!(a.node, ExprKind::Named { .. }));
            let sig = known.signature(callee).cloned();
            match sig {
                Some(sig) if named || args.len() < sig.params.len() => {
                    match place(&sig, args.clone(), &span, diags) {
                        Some(placed) => *args = placed,
                        // Already reported; keep the arguments in the order written so the
                        // checker does not pile a second error on top.
                        None => strip_names(args, &span),
                    }
                }
                _ if named => strip_names(args, &span),
                _ => {}
            }
        }
        ExprKind::StructLit { path, fields } => {
            let Some(name) = path.last() else { return };
            if let Some(declared) = known.structs.get(&name.node) {
                for (field, default) in declared {
                    if !fields.iter().any(|(n, _)| n.node == *field)
                        && let Some(d) = default
                    {
                        fields.push((
                            Ident {
                                node: field.clone(),
                                span: span.clone(),
                            },
                            d.clone(),
                        ));
                    }
                }
            }
        }
        _ => {}
    }
}

/// Not a function the program defines: the names are labels, and the order stays as written.
fn strip_names(args: &mut [Expr], span: &Span) {
    for a in args.iter_mut() {
        if let ExprKind::Named { value, .. } = &mut a.node {
            let inner = std::mem::replace(&mut **value, error_expr(span));
            *a = inner;
        }
    }
}

fn error_expr(span: &Span) -> Expr {
    Spanned {
        node: ExprKind::Error,
        span: span.clone(),
    }
}

/// Arranges `args` (positional first, then `name: value`) in the order of `sig`, filling in
/// defaults. Returns `None` after reporting a problem.
fn place(
    sig: &Sig,
    args: Vec<Expr>,
    span: &Span,
    diags: &mut Vec<Diagnostic>,
) -> Option<Vec<Expr>> {
    let mut slots: Vec<Option<Expr>> = vec![None; sig.params.len()];
    let mut positional = 0;
    let mut extra = Vec::new();
    for a in args {
        match a.node {
            ExprKind::Named { name, value } => {
                let Some(i) = sig.params.iter().position(|(p, _)| *p == name.node) else {
                    let known: Vec<_> = sig.params.iter().map(|(p, _)| format!("`{p}`")).collect();
                    diags.push(
                        Diagnostic::error(
                            name.span.clone(),
                            format!("no parameter `{}`", name.node),
                        )
                        .with_note(format!("the parameters are {}", known.join(", "))),
                    );
                    return None;
                };
                if slots[i].is_some() {
                    diags.push(Diagnostic::error(
                        name.span.clone(),
                        format!("`{}` is given twice", name.node),
                    ));
                    return None;
                }
                slots[i] = Some(*value);
            }
            _ => {
                if positional < slots.len() && slots[positional].is_none() {
                    slots[positional] = Some(a);
                } else {
                    extra.push(a);
                }
                positional += 1;
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..slots.len() {
        match slots[i].take() {
            Some(v) => out.push(v),
            None => match &sig.params[i].1 {
                Some(d) => {
                    let mut d = d.clone();
                    d.span = span.clone();
                    out.push(d);
                }
                None => {
                    // A gap before an argument that was given is a mistake; a missing last
                    // argument is left to the checker, which counts them.
                    if slots[i + 1..].iter().any(Option::is_some) {
                        diags.push(Diagnostic::error(
                            span.clone(),
                            format!("missing the argument `{}`", sig.params[i].0),
                        ));
                        return None;
                    }
                    break;
                }
            },
        }
    }
    out.extend(extra);
    Some(out)
}

// ---- a bottom-up walk over expressions --------------------------------------------------

fn walk_block(b: &mut Block, f: &mut impl FnMut(&mut Expr)) {
    for s in &mut b.stmts {
        match s {
            Stmt::Let(l) => walk(&mut l.init, f),
            Stmt::Expr(e) => walk(e, f),
        }
    }
    if let Some(t) = &mut b.tail {
        walk(t, f);
    }
}

fn walk(e: &mut Expr, f: &mut impl FnMut(&mut Expr)) {
    match &mut e.node {
        ExprKind::Lit(_)
        | ExprKind::Path(_)
        | ExprKind::Break
        | ExprKind::Continue
        | ExprKind::Error => {}
        ExprKind::Call { callee, args } => {
            walk(callee, f);
            args.iter_mut().for_each(|a| walk(a, f));
        }
        ExprKind::Named { value, .. } => walk(value, f),
        ExprKind::Field { base, .. } => walk(base, f),
        ExprKind::Unary { expr, .. } | ExprKind::Try(expr) => walk(expr, f),
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::Pipe { lhs, rhs } => {
            walk(lhs, f);
            walk(rhs, f);
        }
        ExprKind::Closure { body, .. } => walk(body, f),
        ExprKind::StructLit { fields, .. } => fields.iter_mut().for_each(|(_, v)| walk(v, f)),
        ExprKind::If {
            cond,
            then,
            otherwise,
        } => {
            walk(cond, f);
            walk_block(then, f);
            if let Some(o) = otherwise {
                walk(o, f);
            }
        }
        ExprKind::Block(b) => walk_block(b, f),
        ExprKind::Return(v) => {
            if let Some(v) = v {
                walk(v, f);
            }
        }
        ExprKind::Tuple(items) | ExprKind::Array(items) => {
            items.iter_mut().for_each(|i| walk(i, f))
        }
        ExprKind::Match { scrutinee, arms } => {
            walk(scrutinee, f);
            arms.iter_mut().for_each(|a| walk(&mut a.body, f));
        }
        ExprKind::FString(parts) => {
            for p in parts {
                if let FPart::Expr(x) = p {
                    walk(x, f);
                }
            }
        }
        ExprKind::Assign { target, value } => {
            walk(target, f);
            walk(value, f);
        }
        ExprKind::While { cond, body } => {
            walk(cond, f);
            walk_block(body, f);
        }
        ExprKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range { start, end, .. } => {
                    walk(start, f);
                    walk(end, f);
                }
                ForIter::Expr(x) => walk(x, f),
            }
            walk_block(body, f);
        }
        ExprKind::Index { base, index } => {
            walk(base, f);
            walk(index, f);
        }
    }
    f(e);
}
