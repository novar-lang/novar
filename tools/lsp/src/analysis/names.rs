//! Names found by reading the syntax tree of the document: functions, globals, types, enum
//! variants, fields and methods, each with its declaration and every place it is used.
//!
//! The typed program only knows local variables, functions and globals by position. Everything
//! else is read here. A use that depends on a type (`p.x`, `p.norm()`) takes the type of `p` from
//! the typed program; the others (`Point`, `Shape.Circle`, `Point(x: 1)`, `case Shape.Circle(r)`)
//! are read from the syntax alone, so they keep working while the types are wrong.

use std::collections::{HashMap, HashSet};

use telvax_ast as ast;
use telvax_hir::Ty;

use super::Span;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Fn(String),
    Global(String),
    /// A `type`, `enum`, `trait` or alias.
    Type(String),
    /// `(enum, variant)`
    Variant(String, String),
    /// `(type, field)`
    Field(String, String),
    /// `(type or trait, method)`
    Method(String, String),
}

impl Key {
    pub fn name(&self) -> &str {
        match self {
            Key::Fn(n) | Key::Global(n) | Key::Type(n) => n,
            Key::Variant(_, n) | Key::Field(_, n) | Key::Method(_, n) => n,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Names {
    pub decls: Vec<(Key, Span)>,
    pub uses: Vec<(Key, Span)>,
    /// `(type, trait)`: the type implements the trait.
    impls: Vec<(String, String)>,
    /// The text parsed with no syntax error.
    pub syntax_ok: bool,
    /// The uses that need a type (`p.x`) were looked up. False when the types are not known.
    pub typed: bool,
}

/// What the document declares, read before the uses so that a use can be told from a lookalike.
#[derive(Default)]
struct Env {
    types: HashSet<String>,
    variants: HashMap<String, HashSet<String>>,
    fields: HashMap<String, HashSet<String>>,
    methods: HashMap<String, HashSet<String>>,
    impls: Vec<(String, String)>,
}

impl Names {
    /// Reads `text`. `types` gives the type of the expression at each `(start, end)`, when the
    /// program is typed.
    pub fn of(text: &str, types: Option<&HashMap<(usize, usize), Ty>>) -> Names {
        let (program, errors) = telvax_parser::parse(text);
        let Some(program) = program else {
            return Names::default();
        };
        let mut env = Env::default();
        for item in &program.items {
            env.declare(item);
        }
        let mut w = Walker {
            env: &env,
            types,
            generics: Vec::new(),
            out: Names {
                syntax_ok: errors.is_empty(),
                typed: types.is_some(),
                impls: env.impls.clone(),
                ..Names::default()
            },
        };
        for item in &program.items {
            w.item(item);
        }
        let mut out = w.out;
        out.decls.sort_by_key(|(_, s)| s.start);
        out.decls.dedup();
        out.uses.sort_by_key(|(_, s)| s.start);
        out.uses.dedup();
        out
    }

    /// The key whose declaration or use contains `offset`.
    pub fn key_at(&self, offset: usize) -> Option<(&Key, &Span)> {
        self.decls
            .iter()
            .chain(&self.uses)
            .filter(|(_, s)| s.start <= offset && offset <= s.end)
            .min_by_key(|(_, s)| s.len())
            .map(|(k, s)| (k, s))
    }

    pub fn decl(&self, key: &Key) -> Option<&Span> {
        self.decls.iter().find(|(k, _)| k == key).map(|(_, s)| s)
    }

    /// The keys that must change together with `key`: a method is renamed in the trait and in
    /// every type that implements it.
    fn group(&self, key: &Key) -> Vec<Key> {
        let Key::Method(owner, name) = key else {
            return vec![key.clone()];
        };
        let mut owners = vec![owner.clone()];
        for _ in 0..2 {
            for (ty, tr) in &self.impls {
                if owners.contains(ty) && !owners.contains(tr) {
                    owners.push(tr.clone());
                }
                if owners.contains(tr) && !owners.contains(ty) {
                    owners.push(ty.clone());
                }
            }
        }
        owners
            .into_iter()
            .map(|o| Key::Method(o, name.clone()))
            .collect()
    }

    /// The declaration and the uses of `key` (and of the keys that go with it), sorted.
    pub fn spans(&self, key: &Key) -> (Option<Span>, Vec<Span>) {
        let keys = self.group(key);
        let mut decl = None;
        let mut spans = Vec::new();
        for (k, s) in &self.decls {
            if keys.contains(k) {
                if k == key {
                    decl = Some(s.clone());
                }
                spans.push(s.clone());
            }
        }
        for (k, s) in &self.uses {
            if keys.contains(k) {
                spans.push(s.clone());
            }
        }
        spans.sort_by_key(|s| s.start);
        spans.dedup();
        (decl, spans)
    }

    /// Keeps what lies outside the edit that turned a text of `old` bytes into one of `new`
    /// bytes: spans before it stay, spans after it move, the others are dropped.
    pub fn shifted(&self, edit: &Edit) -> Names {
        let keep = |v: &[(Key, Span)]| {
            v.iter()
                .filter_map(|(k, s)| edit.map(s).map(|s| (k.clone(), s)))
                .collect()
        };
        Names {
            decls: keep(&self.decls),
            uses: keep(&self.uses),
            impls: self.impls.clone(),
            syntax_ok: false,
            typed: false,
        }
    }
}

/// The part of a text that changed between two versions.
pub struct Edit {
    /// Bytes at the start that are the same.
    prefix: usize,
    /// End of the changed part in the old text, and the amount the text after it moved.
    old_end: usize,
    new_end: usize,
}

impl Edit {
    pub fn between(old: &str, new: &str) -> Edit {
        let prefix = old
            .bytes()
            .zip(new.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let max = old.len().min(new.len()) - prefix;
        let suffix = old
            .bytes()
            .rev()
            .zip(new.bytes().rev())
            .take(max)
            .take_while(|(a, b)| a == b)
            .count();
        Edit {
            prefix,
            old_end: old.len() - suffix,
            new_end: new.len() - suffix,
        }
    }

    /// Where `span` is after the edit, or `None` when the edit touched it.
    pub fn map(&self, span: &Span) -> Option<Span> {
        if span.end <= self.prefix {
            Some(span.clone())
        } else if span.start >= self.old_end {
            let s = span.start - self.old_end + self.new_end;
            Some(s..s + span.len())
        } else {
            None
        }
    }
}

impl Env {
    fn declare(&mut self, item: &ast::Item) {
        match item {
            ast::Item::Struct(s) => {
                self.types.insert(s.name.node.clone());
                self.fields.insert(
                    s.name.node.clone(),
                    s.fields.iter().map(|f| f.name.node.clone()).collect(),
                );
            }
            ast::Item::Enum(e) => {
                self.types.insert(e.name.node.clone());
                self.variants.insert(
                    e.name.node.clone(),
                    e.variants.iter().map(|v| v.name.node.clone()).collect(),
                );
            }
            ast::Item::Alias(a) => {
                self.types.insert(a.name.node.clone());
            }
            ast::Item::Trait(t) => {
                self.types.insert(t.name.node.clone());
                self.methods
                    .entry(t.name.node.clone())
                    .or_default()
                    .extend(t.fns.iter().map(|f| f.name.node.clone()));
            }
            ast::Item::Impl(i) => {
                self.methods
                    .entry(i.ty.node.clone())
                    .or_default()
                    .extend(i.fns.iter().map(|f| f.name.node.clone()));
                if let Some(t) = &i.trait_name {
                    self.impls.push((i.ty.node.clone(), t.node.clone()));
                }
            }
            ast::Item::Import(_) | ast::Item::Let(_) | ast::Item::Fn(_) => {}
        }
    }
}

struct Walker<'a> {
    env: &'a Env,
    types: Option<&'a HashMap<(usize, usize), Ty>>,
    /// The type parameters in scope, with the traits that limit each.
    generics: Vec<(String, Vec<String>)>,
    out: Names,
}

impl Walker<'_> {
    fn decl(&mut self, key: Key, span: &Span) {
        self.out.decls.push((key, span.clone()));
    }

    fn used(&mut self, key: Key, span: &Span) {
        self.out.uses.push((key, span.clone()));
    }

    fn item(&mut self, item: &ast::Item) {
        match item {
            ast::Item::Import(_) => {}
            ast::Item::Let(l) => {
                self.decl(Key::Global(l.name.node.clone()), &l.name.span);
                self.let_(l);
            }
            ast::Item::Struct(s) => {
                self.decl(Key::Type(s.name.node.clone()), &s.name.span);
                let mark = self.push_generics(&s.generics);
                for f in &s.fields {
                    self.decl(
                        Key::Field(s.name.node.clone(), f.name.node.clone()),
                        &f.name.span,
                    );
                    self.ty(&f.ty);
                    if let Some(d) = &f.default {
                        self.expr(d);
                    }
                }
                self.generics.truncate(mark);
            }
            ast::Item::Enum(e) => {
                self.decl(Key::Type(e.name.node.clone()), &e.name.span);
                let mark = self.push_generics(&e.generics);
                for v in &e.variants {
                    self.decl(
                        Key::Variant(e.name.node.clone(), v.name.node.clone()),
                        &v.name.span,
                    );
                    v.fields.iter().for_each(|t| self.ty(t));
                }
                self.generics.truncate(mark);
            }
            ast::Item::Alias(a) => {
                self.decl(Key::Type(a.name.node.clone()), &a.name.span);
                let mark = self.push_generics(&a.generics);
                self.ty(&a.ty);
                self.generics.truncate(mark);
            }
            ast::Item::Trait(t) => {
                self.decl(Key::Type(t.name.node.clone()), &t.name.span);
                for f in &t.fns {
                    self.decl(
                        Key::Method(t.name.node.clone(), f.name.node.clone()),
                        &f.name.span,
                    );
                    self.function(f);
                }
            }
            ast::Item::Impl(i) => {
                let mark = self.push_generics(&i.generics);
                self.type_name(&i.ty);
                i.ty_args.iter().for_each(|t| self.ty(t));
                if let Some(t) = &i.trait_name {
                    self.type_name(t);
                }
                for f in &i.fns {
                    self.decl(
                        Key::Method(i.ty.node.clone(), f.name.node.clone()),
                        &f.name.span,
                    );
                    self.function(f);
                }
                self.generics.truncate(mark);
            }
            ast::Item::Fn(f) => {
                self.decl(Key::Fn(f.name.node.clone()), &f.name.span);
                self.function(f);
            }
        }
    }

    fn push_generics(&mut self, params: &[ast::GenericParam]) -> usize {
        let mark = self.generics.len();
        for p in params {
            for b in &p.bounds {
                self.type_name(b);
            }
            let bounds = p.bounds.iter().map(|b| b.node.clone()).collect();
            self.generics.push((p.name.node.clone(), bounds));
        }
        mark
    }

    fn function(&mut self, f: &ast::Fn) {
        let mark = self.push_generics(&f.generics);
        for a in &f.attrs {
            a.args.iter().for_each(|(_, v)| self.expr(v));
        }
        for p in &f.params {
            self.ty(&p.ty);
            if let Some(d) = &p.default {
                self.expr(d);
            }
        }
        if let Some(r) = &f.ret {
            self.ty(r);
        }
        self.block(&f.body);
        self.generics.truncate(mark);
    }

    /// A bare type name written in `impl T`, `impl .. for T` or `[X: T]`.
    fn type_name(&mut self, name: &ast::Ident) {
        if self.env.types.contains(&name.node) {
            self.used(Key::Type(name.node.clone()), &name.span);
        }
    }

    fn ty(&mut self, t: &ast::Type) {
        if let [only] = t.path.as_slice() {
            self.type_name(only);
        }
        for a in &t.args {
            match a {
                ast::TypeArg::Type(t) => self.ty(t),
                ast::TypeArg::Row(fields) => fields.iter().for_each(|f| self.ty(&f.ty)),
                ast::TypeArg::Shape(_) => {}
            }
        }
    }

    fn let_(&mut self, l: &ast::Let) {
        if let Some(t) = &l.ty {
            self.ty(t);
        }
        self.expr(&l.init);
    }

    fn block(&mut self, b: &ast::Block) {
        for s in &b.stmts {
            match s {
                ast::Stmt::Let(l) => self.let_(l),
                ast::Stmt::Expr(e) => self.expr(e),
            }
        }
        if let Some(t) = &b.tail {
            self.expr(t);
        }
    }

    /// `Shape`, `Shape.Circle`, `Point.new`: the type, and the variant or method after it.
    fn path(&mut self, path: &[ast::Ident]) {
        let Some(head) = path.first() else { return };
        if !self.env.types.contains(&head.node) {
            return;
        }
        self.used(Key::Type(head.node.clone()), &head.span);
        let Some(next) = path.get(1) else { return };
        if self
            .env
            .variants
            .get(&head.node)
            .is_some_and(|v| v.contains(&next.node))
        {
            self.used(
                Key::Variant(head.node.clone(), next.node.clone()),
                &next.span,
            );
        } else if self
            .env
            .methods
            .get(&head.node)
            .is_some_and(|m| m.contains(&next.node))
        {
            self.used(
                Key::Method(head.node.clone(), next.node.clone()),
                &next.span,
            );
        }
    }

    /// The types a value can have for the purpose of looking up a field or a method.
    fn owners(&self, e: &ast::Expr) -> Vec<String> {
        let Some(ty) = self.types.and_then(|t| t.get(&(e.span.start, e.span.end))) else {
            return Vec::new();
        };
        match ty {
            Ty::Struct(n) | Ty::Enum(n) | Ty::Trait(n) | Ty::Named(n) | Ty::Table(n) => {
                vec![n.clone()]
            }
            Ty::Param(p) => self
                .generics
                .iter()
                .rev()
                .find(|(n, _)| n == p)
                .map(|(_, bounds)| bounds.clone())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn member(&mut self, base: &ast::Expr, name: &ast::Ident) {
        for owner in self.owners(base) {
            if self
                .env
                .fields
                .get(&owner)
                .is_some_and(|f| f.contains(&name.node))
            {
                self.used(Key::Field(owner, name.node.clone()), &name.span);
                return;
            }
            let has = |o: &str| {
                self.env
                    .methods
                    .get(o)
                    .is_some_and(|m| m.contains(&name.node))
            };
            if has(&owner) {
                self.used(Key::Method(owner, name.node.clone()), &name.span);
                return;
            }
            let via = self
                .env
                .impls
                .iter()
                .find(|(t, tr)| *t == owner && has(tr))
                .map(|(_, tr)| tr.clone());
            if let Some(tr) = via {
                self.used(Key::Method(tr, name.node.clone()), &name.span);
                return;
            }
        }
    }

    fn expr(&mut self, e: &ast::Expr) {
        use ast::ExprKind as K;
        match &e.node {
            K::Lit(_) | K::Break | K::Continue | K::Error => {}
            K::Path(p) => self.path(p),
            K::Call { callee, args } => {
                self.expr(callee);
                args.iter().for_each(|a| self.expr(a));
            }
            K::Named { value, .. } => self.expr(value),
            K::Field { base, name } => {
                self.expr(base);
                self.member(base, name);
            }
            K::Unary { expr, .. } | K::Try(expr) | K::Await(expr) => self.expr(expr),
            K::Binary { lhs, rhs, .. } | K::Pipe { lhs, rhs } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            K::Closure { types, body, .. } => {
                types.iter().flatten().for_each(|t| self.ty(t));
                self.expr(body);
            }
            K::StructLit { path, fields } => {
                self.path(path);
                if let [only] = path.as_slice() {
                    for (name, _) in fields {
                        if self
                            .env
                            .fields
                            .get(&only.node)
                            .is_some_and(|f| f.contains(&name.node))
                        {
                            self.used(Key::Field(only.node.clone(), name.node.clone()), &name.span);
                        }
                    }
                }
                fields.iter().for_each(|(_, v)| self.expr(v));
            }
            K::If {
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
            K::Block(b) => self.block(b),
            K::Return(v) => {
                if let Some(v) = v {
                    self.expr(v);
                }
            }
            K::Tuple(items) | K::Array(items) => items.iter().for_each(|i| self.expr(i)),
            K::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.pattern(&arm.pat);
                    if let Some(g) = &arm.guard {
                        self.expr(g);
                    }
                    self.expr(&arm.body);
                }
            }
            K::FString(parts) => parts
                .iter()
                .flat_map(|p| p.exprs())
                .for_each(|x| self.expr(x)),
            K::Assign { target, value } => {
                self.expr(target);
                self.expr(value);
            }
            K::While { cond, body } => {
                self.expr(cond);
                self.block(body);
            }
            K::For { iter, body, .. } => {
                match iter {
                    ast::ForIter::Range { start, end, .. } => {
                        self.expr(start);
                        self.expr(end);
                    }
                    ast::ForIter::Expr(x) => self.expr(x),
                }
                self.block(body);
            }
            K::Index { base, index } => {
                self.expr(base);
                self.expr(index);
            }
        }
    }

    fn pattern(&mut self, p: &ast::Pattern) {
        use ast::PatKind as P;
        match &p.node {
            P::Wild | P::Bind(_) | P::Lit(_) | P::NegLit(_) => {}
            P::Tuple(items) | P::Or(items) => items.iter().for_each(|i| self.pattern(i)),
            P::List {
                before,
                rest,
                after,
            } => before
                .iter()
                .chain(rest.as_deref())
                .chain(after)
                .for_each(|i| self.pattern(i)),
            P::Struct { path, fields } => {
                self.path(path);
                if let [only] = path.as_slice() {
                    for (name, _) in fields {
                        if self
                            .env
                            .fields
                            .get(&only.node)
                            .is_some_and(|f| f.contains(&name.node))
                        {
                            self.used(Key::Field(only.node.clone(), name.node.clone()), &name.span);
                        }
                    }
                }
                fields.iter().for_each(|(_, v)| self.pattern(v));
            }
            P::Variant { path, args } => {
                self.path(path);
                args.iter().for_each(|a| self.pattern(a));
            }
        }
    }
}
