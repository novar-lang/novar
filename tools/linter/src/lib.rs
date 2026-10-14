//! Lints for code that is declared but never used.
//!
//! * a local variable or parameter that is never read (names starting with `_` are exempt);
//! * a private function (a name starting with `_`) that nothing calls (`main` and endpoints are
//!   exempt, and so is every public function).

use std::collections::{HashMap, HashSet};

use telvax_diagnostics::Diagnostic;
use telvax_hir::hir::{self, Expr, ExprKind, ForIter, Local, LocalId, Pat, Proj, Stmt};

pub fn lint(program: &hir::Program) -> Vec<Diagnostic> {
    let mut l = Linter::default();
    for g in &program.globals {
        l.expr(&g.init);
    }
    for f in &program.fns {
        l.fn_names.insert(f.name.clone(), f.name_span.clone());
        for p in &f.params {
            l.declare(p);
        }
        l.block(&f.body);
    }

    let mut out = Vec::new();
    for (id, local) in &l.locals {
        if local.name.starts_with('_') {
            continue;
        }
        if !l.read.contains(id) {
            out.push(
                Diagnostic::warning(
                    local.span.clone(),
                    format!("unused variable `{}`", local.name),
                )
                .with_note("prefix it with `_` if that is intentional"),
            );
        }
    }
    for f in &program.fns {
        let exempt = f.is_pub || f.name == "main" || f.endpoint.is_some();
        if !exempt && !l.called.contains(&f.name) {
            out.push(Diagnostic::warning(
                f.name_span.clone(),
                format!("function `{}` is never used", f.name.replace("::", ".")),
            ));
        }
    }
    out.sort_by_key(|d| d.span.start);
    out
}

#[derive(Default)]
struct Linter {
    locals: HashMap<LocalId, Local>,
    read: HashSet<LocalId>,
    called: HashSet<String>,
    fn_names: HashMap<String, std::ops::Range<usize>>,
}

impl Linter {
    fn declare(&mut self, l: &Local) {
        self.locals.insert(l.id, l.clone());
    }

    fn block(&mut self, b: &hir::Block) {
        for s in &b.stmts {
            match s {
                Stmt::Let { local, init } => {
                    self.expr(init);
                    self.declare(local);
                }
                Stmt::Expr(e) => self.expr(e),
            }
        }
        if let Some(t) = &b.tail {
            self.expr(t);
        }
    }

    fn pattern(&mut self, p: &Pat) {
        match p {
            Pat::Bind(l) => self.declare(l),
            Pat::Variant { args, .. } | Pat::Tuple(args) => {
                args.iter().for_each(|a| self.pattern(a))
            }
            Pat::Wild | Pat::Lit(_) => {}
        }
    }

    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Local(id) => {
                self.read.insert(*id);
            }
            ExprKind::Fn(name) => {
                self.called.insert(name.clone());
            }
            ExprKind::Lit(_)
            | ExprKind::Global(_)
            | ExprKind::Extern(_)
            | ExprKind::Error
            | ExprKind::Break
            | ExprKind::Continue => {}
            ExprKind::Call { callee, args } => {
                self.expr(callee);
                args.iter().for_each(|a| self.expr(a));
            }
            ExprKind::Field { base, .. } => self.expr(base),
            ExprKind::Unary { expr, .. } | ExprKind::Try(expr) => self.expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Closure { params, body } => {
                params.iter().for_each(|p| self.declare(p));
                self.expr(body);
            }
            ExprKind::StructLit { fields, .. } => fields.iter().for_each(|(_, v)| self.expr(v)),
            ExprKind::EnumCtor { args, .. }
            | ExprKind::Tuple(args)
            | ExprKind::DynCall { args, .. } => {
                args.iter().for_each(|a| self.expr(a));
            }
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => {
                self.expr(cond);
                self.block(then);
                if let Some(o) = otherwise {
                    self.expr(o);
                }
            }
            ExprKind::Block(b) => self.block(b),
            ExprKind::Return(v) => {
                if let Some(v) = v {
                    self.expr(v);
                }
            }
            ExprKind::FString(parts) => {
                for p in parts {
                    if let hir::FPart::Expr(x) = p {
                        self.expr(x);
                    }
                }
            }
            ExprKind::Assign { place, value } => {
                // `v[i] = x` and `p.f = x` read the root (it must already hold a value); a plain
                // `x = value` does not.
                if !place.projs.is_empty() {
                    self.read.insert(place.local);
                }
                for pr in &place.projs {
                    if let Proj::Index(i) = pr {
                        self.expr(i);
                    }
                }
                self.expr(value);
            }
            ExprKind::While { cond, body } => {
                self.expr(cond);
                self.block(body);
            }
            ExprKind::For { var, iter, body } => {
                self.declare(var);
                match iter {
                    ForIter::Range { start, end, .. } => {
                        self.expr(start);
                        self.expr(end);
                    }
                    ForIter::Vec(v) => self.expr(v),
                }
                self.block(body);
            }
            ExprKind::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
            ExprKind::Array(items) => items.iter().for_each(|i| self.expr(i)),
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.pattern(&arm.pat);
                    self.expr(&arm.body);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages(src: &str) -> Vec<String> {
        let (ast, errs) = telvax_parser::parse(src);
        assert!(errs.is_empty(), "{errs:?}");
        let (hir, diags) = telvax_hir::lower(&ast.unwrap());
        assert!(
            diags.is_empty(),
            "{:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        lint(&hir).into_iter().map(|d| d.message).collect()
    }

    #[test]
    fn reports_unused_variables_and_functions() {
        let m = messages(
            r#"
def _helper() -> int32:
    return 1

def used() -> int32:
    return 2

def api(unused_param: int32) -> int32:
    a = 1
    b = 2
    _quiet = 3
    c = 0
    c = c + used()
    return b

def main() -> int32:
    return api(1)
"#,
        );
        assert_eq!(
            m,
            [
                "function `_helper` is never used",
                "unused variable `unused_param`",
                "unused variable `a`",
            ]
        );
    }

    #[test]
    fn counts_uses_in_loops_matches_and_closures() {
        let m = messages(
            r#"
enum E:
    A(int32)
    B

def main() -> int32:
    total = 0
    v = [1, 2, 3]
    for x in v:
        if x == 2:
            continue
        total += x
    e = E.A(total)
    match e:
        case E.A(n):
            return n
        case E.B:
            return 0
"#,
        );
        assert!(m.is_empty(), "{m:?}");
    }

    #[test]
    fn examples_are_clean() {
        for src in [
            include_str!("../../../examples/basics/primes.tvx"),
            include_str!("../../../examples/basics/sieve.tvx"),
        ] {
            let m = messages(src);
            assert!(m.is_empty(), "{m:?}");
        }
    }
}
