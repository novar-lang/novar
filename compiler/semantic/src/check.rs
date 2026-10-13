//! Name resolution and type checking: AST -> typed HIR.

use std::collections::{HashMap, HashSet};

use novar_ast as ast;
use novar_diagnostics::Diagnostic;

use crate::hir::{self, Endpoint, HttpMethod, LocalId};
use crate::ty::{Dim, Prim, Ty};

type Span = std::ops::Range<usize>;

#[derive(Default)]
pub(crate) struct Checker {
    diags: Vec<Diagnostic>,
    imports: HashSet<String>,
    struct_fields: HashMap<String, Vec<(String, Ty)>>,
    fn_sigs: HashMap<String, (Vec<Ty>, Ty)>,
    globals: HashMap<String, Ty>,
    scopes: Vec<HashMap<String, (LocalId, Ty)>>,
    next_local: u32,
    ret: Ty,
    endpoints: HashSet<(HttpMethod, String)>,
}

pub(crate) fn lower(program: &ast::Program) -> (hir::Program, Vec<Diagnostic>) {
    let mut c = Checker::default();
    let out = c.run(program);
    (out, c.diags)
}

impl Checker {
    fn err(&mut self, span: Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::error(span, msg));
    }

    fn run(&mut self, program: &ast::Program) -> hir::Program {
        let mut out = hir::Program::default();

        for item in &program.items {
            if let ast::Item::Import(i) = item {
                let base: Vec<String> = i.path.iter().map(|s| s.node.clone()).collect();
                if i.group.is_empty() {
                    if let Some(last) = base.last() {
                        self.imports.insert(last.clone());
                    }
                    out.imports.push(base);
                } else {
                    for name in &i.group {
                        self.imports.insert(name.node.clone());
                        let mut full = base.clone();
                        full.push(name.node.clone());
                        out.imports.push(full);
                    }
                }
            }
        }

        // Declare struct names first so fields and signatures may refer to any struct.
        let mut declared = HashSet::new();
        for item in &program.items {
            if let ast::Item::Struct(s) = item {
                if !declared.insert(s.name.node.clone()) {
                    self.err(
                        s.name.span.clone(),
                        format!("struct `{}` is defined twice", s.name.node),
                    );
                } else {
                    self.struct_fields.insert(s.name.node.clone(), Vec::new());
                }
            }
        }
        let mut done = HashSet::new();
        for item in &program.items {
            if let ast::Item::Struct(s) = item {
                if !done.insert(s.name.node.clone()) {
                    continue;
                }
                let mut fields: Vec<(String, Ty)> = Vec::new();
                for f in &s.fields {
                    if fields.iter().any(|(n, _)| *n == f.name.node) {
                        self.err(
                            f.name.span.clone(),
                            format!("field `{}` is declared twice", f.name.node),
                        );
                        continue;
                    }
                    let ty = self.resolve_type(&f.ty);
                    fields.push((f.name.node.clone(), ty));
                }
                self.struct_fields
                    .insert(s.name.node.clone(), fields.clone());
                out.structs.push(hir::StructDef {
                    name: s.name.node.clone(),
                    is_pub: s.is_pub,
                    fields,
                    span: s.span.clone(),
                });
            }
        }

        // Function signatures.
        let mut sig_done = HashSet::new();
        for item in &program.items {
            if let ast::Item::Fn(f) = item {
                if !sig_done.insert(f.name.node.clone()) {
                    self.err(
                        f.name.span.clone(),
                        format!("function `{}` is defined twice", f.name.node),
                    );
                    continue;
                }
                let params = f.params.iter().map(|p| self.resolve_type(&p.ty)).collect();
                let ret = f.ret.as_ref().map_or(Ty::Unit, |t| self.resolve_type(t));
                self.fn_sigs.insert(f.name.node.clone(), (params, ret));
            }
        }

        // Globals, in source order.
        for item in &program.items {
            if let ast::Item::Let(l) = item {
                let annotated = l.ty.as_ref().map(|t| self.resolve_type(t));
                self.ret = Ty::Unknown;
                self.scopes.clear();
                let init = self.expr(&l.init, annotated.as_ref());
                let init = match &annotated {
                    Some(t) => self.coerce(init, t),
                    None => init,
                };
                let ty = annotated.unwrap_or_else(|| init.ty.clone());
                if self
                    .globals
                    .insert(l.name.node.clone(), ty.clone())
                    .is_some()
                {
                    self.err(
                        l.name.span.clone(),
                        format!("global `{}` is defined twice", l.name.node),
                    );
                }
                out.globals.push(hir::Global {
                    name: l.name.node.clone(),
                    ty,
                    init,
                    span: l.span.clone(),
                });
            }
        }

        let mut lowered = HashSet::new();
        for item in &program.items {
            if let ast::Item::Fn(f) = item {
                if lowered.insert(f.name.node.clone()) {
                    out.fns.push(self.lower_fn(f));
                }
            }
        }
        out
    }

    // ---- types -------------------------------------------------------------------------

    fn resolve_type(&mut self, t: &ast::Type) -> Ty {
        let names: Vec<&str> = t.path.iter().map(|i| i.node.as_str()).collect();
        if names.len() > 1 {
            if names[0] == "std" || self.imports.contains(names[0]) {
                return Ty::Named(names.join("::"));
            }
            self.err(
                t.span.clone(),
                format!("cannot find type `{}`", names.join("::")),
            );
            return Ty::Unknown;
        }
        let name = names[0];
        let type_args: Vec<&ast::Type> = t
            .args
            .iter()
            .filter_map(|a| match a {
                ast::TypeArg::Type(t) => Some(t),
                ast::TypeArg::Shape(_) => None,
            })
            .collect();
        let no_args = |this: &mut Self| {
            if !t.args.is_empty() {
                this.err(
                    t.span.clone(),
                    format!("type `{name}` takes no type arguments"),
                );
            }
        };

        if let Some(p) = Prim::from_name(name) {
            no_args(self);
            return Ty::Prim(p);
        }
        match name {
            "Tensor" => self.resolve_tensor(t),
            "Vec" | "Vector" => match type_args.as_slice() {
                [] if t.args.is_empty() => Ty::Vec(Box::new(Ty::Unknown)),
                [inner] if t.args.len() == 1 => Ty::Vec(Box::new(self.resolve_type(inner))),
                _ => {
                    self.err(
                        t.span.clone(),
                        format!("`{name}` takes at most one type argument"),
                    );
                    Ty::Vec(Box::new(Ty::Unknown))
                }
            },
            "Result" => match type_args.as_slice() {
                [ok, er] if t.args.len() == 2 => Ty::Result(
                    Box::new(self.resolve_type(ok)),
                    Box::new(self.resolve_type(er)),
                ),
                _ => {
                    self.err(
                        t.span.clone(),
                        "`Result` takes two type arguments: `Result<T, E>`",
                    );
                    Ty::Unknown
                }
            },
            _ if self.struct_fields.contains_key(name) => {
                no_args(self);
                Ty::Struct(name.to_string())
            }
            _ if self.imports.contains(name) => Ty::Named(name.to_string()),
            _ => {
                self.err(
                    t.span.clone(),
                    format!("cannot find type `{name}` in this scope"),
                );
                Ty::Unknown
            }
        }
    }

    fn resolve_tensor(&mut self, t: &ast::Type) -> Ty {
        let (elem, shape) = match t.args.as_slice() {
            [ast::TypeArg::Type(e), ast::TypeArg::Shape(s)] => (e, s),
            _ => {
                self.err(
                    t.span.clone(),
                    "`Tensor` expects an element type and a shape: `Tensor<f32, [N, M]>`",
                );
                return Ty::Unknown;
            }
        };
        let elem_ty = self.resolve_type(elem);
        if !matches!(&elem_ty, Ty::Prim(p) if p.is_numeric() || *p == Prim::Bool)
            && elem_ty != Ty::Unknown
        {
            self.err(
                elem.span.clone(),
                format!("tensor element type must be a numeric primitive, found `{elem_ty}`"),
            );
        }
        let mut dims = Vec::new();
        for d in shape {
            match d {
                ast::Dim::Lit(0) => {
                    self.err(
                        t.span.clone(),
                        "tensor dimensions must be greater than zero",
                    );
                    dims.push(Dim::Lit(0));
                }
                ast::Dim::Lit(n) => dims.push(Dim::Lit(*n)),
                ast::Dim::Named(n) => dims.push(Dim::Named(n.node.clone())),
            }
        }
        if dims.is_empty() {
            self.err(
                t.span.clone(),
                "tensor shape must have at least one dimension",
            );
        }
        Ty::Tensor {
            elem: Box::new(elem_ty),
            shape: dims,
        }
    }

    // ---- scopes ------------------------------------------------------------------------

    fn declare(&mut self, name: &ast::Ident, ty: Ty) -> hir::Local {
        let id = LocalId(self.next_local);
        self.next_local += 1;
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.node.clone(), (id, ty.clone()));
        }
        hir::Local {
            id,
            name: name.node.clone(),
            ty,
        }
    }

    fn lookup_local(&self, name: &str) -> Option<(LocalId, Ty)> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }

    // ---- functions ---------------------------------------------------------------------

    fn lower_fn(&mut self, f: &ast::Fn) -> hir::FnDef {
        let (param_tys, ret) = self.fn_sigs[&f.name.node].clone();
        self.scopes = vec![HashMap::new()];
        self.ret = ret.clone();
        let mut params = Vec::new();
        for (p, ty) in f.params.iter().zip(param_tys) {
            if params.iter().any(|q: &hir::Local| q.name == p.name.node) {
                self.err(
                    p.name.span.clone(),
                    format!("parameter `{}` is declared twice", p.name.node),
                );
            }
            params.push(self.declare(&p.name, ty));
        }

        let (body, body_ty) = self.block(&f.body, Some(&ret));
        if !ret.accepts(&body_ty) {
            let span = body
                .tail
                .as_ref()
                .map_or(f.body.span.clone(), |t| t.span.clone());
            let mut d = Diagnostic::error(
                span,
                format!("mismatched types: expected `{ret}`, found `{body_ty}`"),
            );
            if body.tail.is_none() {
                d = d.with_note(
                    "the function body has no final expression and does not end in `return`",
                );
            }
            self.diags.push(d);
        }

        let endpoint = self.endpoint(f, &params, &ret);
        self.scopes.clear();
        hir::FnDef {
            name: f.name.node.clone(),
            is_pub: f.is_pub,
            params,
            ret,
            body,
            endpoint,
            span: f.span.clone(),
        }
    }

    fn endpoint(&mut self, f: &ast::Fn, params: &[hir::Local], ret: &Ty) -> Option<Endpoint> {
        let mut found = None;
        for attr in &f.attrs {
            if attr.name.node != "endpoint" {
                self.err(
                    attr.name.span.clone(),
                    format!("unknown attribute `@{}`", attr.name.node),
                );
                continue;
            }
            if found.is_some() {
                self.err(
                    attr.span.clone(),
                    "a function can have only one `@endpoint`",
                );
                continue;
            }
            let mut path = None;
            let mut method = None;
            for (key, value) in &attr.args {
                let text = match &value.node {
                    ast::ExprKind::Lit(ast::Lit::Str(s)) => s.clone(),
                    _ => {
                        self.err(
                            value.span.clone(),
                            format!("`{}` must be a string literal", key.node),
                        );
                        continue;
                    }
                };
                match key.node.as_str() {
                    "path" if path.is_none() => path = Some((text, value.span.clone())),
                    "method" if method.is_none() => method = Some((text, value.span.clone())),
                    "path" | "method" => {
                        self.err(key.span.clone(), format!("`{}` given twice", key.node))
                    }
                    other => self.err(
                        key.span.clone(),
                        format!("unknown `@endpoint` argument `{other}`"),
                    ),
                }
            }
            let Some((path, path_span)) = path else {
                self.err(attr.span.clone(), "`@endpoint` requires a `path` argument");
                continue;
            };
            if !path.starts_with('/') {
                self.err(path_span, "endpoint path must start with `/`");
            }
            let method = match method {
                None => HttpMethod::Get,
                Some((m, span)) => match m.to_ascii_uppercase().as_str() {
                    "GET" => HttpMethod::Get,
                    "POST" => HttpMethod::Post,
                    "PUT" => HttpMethod::Put,
                    "PATCH" => HttpMethod::Patch,
                    "DELETE" => HttpMethod::Delete,
                    _ => {
                        self.err(span, format!("unsupported HTTP method `{m}`"));
                        HttpMethod::Get
                    }
                },
            };
            if !self.endpoints.insert((method, path.clone())) {
                self.err(
                    attr.span.clone(),
                    format!("duplicate endpoint {method:?} `{path}`"),
                );
            }

            let wants_body = matches!(
                method,
                HttpMethod::Post | HttpMethod::Put | HttpMethod::Patch
            );
            match params {
                [] if !wants_body => {}
                [p] if matches!(p.ty, Ty::Struct(_) | Ty::Unknown) => {}
                _ => self.err(
                    f.name.span.clone(),
                    "an endpoint takes at most one parameter, which must be a struct (the request payload)",
                ),
            }
            if wants_body && params.is_empty() {
                self.err(
                    f.name.span.clone(),
                    format!("{method:?} endpoints need a payload parameter"),
                );
            }
            if !matches!(ret, Ty::Result(..) | Ty::Unknown) {
                self.err(
                    f.ret
                        .as_ref()
                        .map_or(f.name.span.clone(), |t| t.span.clone()),
                    "an endpoint must return `Result<T, E>`",
                );
            }
            found = Some(Endpoint { path, method });
        }
        found
    }

    // ---- blocks and statements --------------------------------------------------------

    fn block(&mut self, b: &ast::Block, expected: Option<&Ty>) -> (hir::Block, Ty) {
        self.scopes.push(HashMap::new());
        let mut stmts = Vec::new();
        let mut diverges = false;
        for s in &b.stmts {
            match s {
                ast::Stmt::Let(l) => {
                    let annotated = l.ty.as_ref().map(|t| self.resolve_type(t));
                    let init = self.expr(&l.init, annotated.as_ref());
                    let init = match &annotated {
                        Some(t) => self.coerce(init, t),
                        None => init,
                    };
                    let ty = annotated.unwrap_or_else(|| init.ty.clone());
                    // The binding is only visible after its initializer.
                    let local = self.declare(&l.name, ty);
                    stmts.push(hir::Stmt::Let { local, init });
                }
                ast::Stmt::Expr(e) => {
                    let e = self.expr(e, None);
                    diverges |= e.ty == Ty::Never;
                    stmts.push(hir::Stmt::Expr(e));
                }
            }
        }
        let tail = b.tail.as_ref().map(|e| Box::new(self.expr(e, expected)));
        self.scopes.pop();
        let ty = match &tail {
            Some(t) => t.ty.clone(),
            None if diverges => Ty::Never,
            None => Ty::Unit,
        };
        (
            hir::Block {
                stmts,
                tail,
                ty: ty.clone(),
            },
            ty,
        )
    }

    // ---- expressions -------------------------------------------------------------------

    /// Reports a mismatch if `e` cannot be used as `expected`.
    fn coerce(&mut self, e: hir::Expr, expected: &Ty) -> hir::Expr {
        if !expected.accepts(&e.ty) {
            self.err(
                e.span.clone(),
                format!("mismatched types: expected `{expected}`, found `{}`", e.ty),
            );
        }
        e
    }

    fn mk(kind: hir::ExprKind, ty: Ty, span: &Span) -> hir::Expr {
        hir::Expr {
            kind,
            ty,
            span: span.clone(),
        }
    }

    fn expr(&mut self, e: &ast::Expr, expected: Option<&Ty>) -> hir::Expr {
        use ast::ExprKind as K;
        let span = &e.span;
        match &e.node {
            K::Error => Self::mk(hir::ExprKind::Error, Ty::Unknown, span),
            K::Lit(l) => {
                let ty = match l {
                    ast::Lit::Int(_) => match expected {
                        Some(Ty::Prim(p)) if p.is_numeric() => Ty::Prim(*p),
                        _ => Ty::Prim(Prim::I32),
                    },
                    ast::Lit::Float(_) => match expected {
                        Some(Ty::Prim(p)) if p.is_float() => Ty::Prim(*p),
                        _ => Ty::Prim(Prim::F64),
                    },
                    ast::Lit::Str(_) => Ty::str(),
                    ast::Lit::Bool(_) => Ty::bool(),
                };
                Self::mk(hir::ExprKind::Lit(l.clone()), ty, span)
            }
            K::Path(p) => self.path(p, span),
            K::FString(parts) => {
                let parts = parts
                    .iter()
                    .map(|p| match p {
                        ast::FPart::Text(t) => hir::FPart::Text(t.clone()),
                        ast::FPart::Expr(e) => hir::FPart::Expr(self.expr(e, None)),
                    })
                    .collect();
                Self::mk(hir::ExprKind::FString(parts), Ty::str(), span)
            }
            K::Field { base, name } => {
                let base = self.expr(base, None);
                let ty = self.field_ty(&base, name);
                Self::mk(
                    hir::ExprKind::Field {
                        base: Box::new(base),
                        name: name.node.clone(),
                    },
                    ty,
                    span,
                )
            }
            K::Call { callee, args } => self.call(callee, args, expected, span),
            K::Pipe { lhs, rhs } => self.pipe(lhs, rhs, expected, span),
            K::Unary { op, expr } => {
                let inner = self.expr(expr, expected);
                let ok = match (op, &inner.ty) {
                    (_, Ty::Unknown) => true,
                    (ast::UnOp::Neg, Ty::Prim(p)) => p.is_numeric(),
                    (ast::UnOp::Not, Ty::Prim(Prim::Bool)) => true,
                    _ => false,
                };
                if !ok {
                    self.err(
                        span.clone(),
                        format!("cannot apply `{op:?}` to `{}`", inner.ty),
                    );
                }
                let ty = inner.ty.clone();
                Self::mk(
                    hir::ExprKind::Unary {
                        op: *op,
                        expr: Box::new(inner),
                    },
                    ty,
                    span,
                )
            }
            K::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs, span),
            K::Closure { params, body } => {
                let want: Option<(&Vec<Ty>, &Ty)> = match expected {
                    Some(Ty::Fn(p, r)) if p.len() == params.len() => Some((p, r)),
                    _ => None,
                };
                self.scopes.push(HashMap::new());
                let saved = std::mem::take(&mut self.ret);
                let locals: Vec<_> = params
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let ty = want.map_or(Ty::Unknown, |(ps, _)| ps[i].clone());
                        self.declare(p, ty)
                    })
                    .collect();
                let body = self.expr(body, want.map(|(_, r)| r));
                self.ret = saved;
                self.scopes.pop();
                let ty = Ty::Fn(
                    locals.iter().map(|l| l.ty.clone()).collect(),
                    Box::new(body.ty.clone()),
                );
                Self::mk(
                    hir::ExprKind::Closure {
                        params: locals,
                        body: Box::new(body),
                    },
                    ty,
                    span,
                )
            }
            K::StructLit { path, fields } => self.struct_lit(path, fields, span),
            K::If {
                cond,
                then,
                otherwise,
            } => {
                let cond = self.expr(cond, Some(&Ty::bool()));
                let cond = self.coerce(cond, &Ty::bool());
                let (then, then_ty) = self.block(then, expected);
                let (otherwise, ty) = match otherwise {
                    None => (None, Ty::Unit),
                    Some(o) => {
                        let o = self.expr(o, expected);
                        let ty = if then_ty.accepts(&o.ty) {
                            if then_ty == Ty::Never {
                                o.ty.clone()
                            } else {
                                then_ty.clone()
                            }
                        } else {
                            self.diags.push(Diagnostic::error(
                                o.span.clone(),
                                format!(
                                    "`if` and `else` have incompatible types: `{then_ty}` and `{}`",
                                    o.ty
                                ),
                            ));
                            Ty::Unknown
                        };
                        (Some(Box::new(o)), ty)
                    }
                };
                Self::mk(
                    hir::ExprKind::If {
                        cond: Box::new(cond),
                        then,
                        otherwise,
                    },
                    ty,
                    span,
                )
            }
            K::Block(b) => {
                let (block, ty) = self.block(b, expected);
                Self::mk(hir::ExprKind::Block(block), ty, span)
            }
            K::Return(value) => {
                let ret = self.ret.clone();
                let value = match value {
                    Some(v) => {
                        let v = self.expr(v, Some(&ret));
                        Some(Box::new(self.coerce(v, &ret)))
                    }
                    None => {
                        if !ret.accepts(&Ty::Unit) {
                            self.err(
                                span.clone(),
                                format!("`return` needs a value of type `{ret}`"),
                            );
                        }
                        None
                    }
                };
                Self::mk(hir::ExprKind::Return(value), Ty::Never, span)
            }
        }
    }

    fn path(&mut self, p: &[ast::Ident], span: &Span) -> hir::Expr {
        let names: Vec<String> = p.iter().map(|i| i.node.clone()).collect();
        let first = names[0].as_str();
        if names.len() == 1 {
            if let Some((id, ty)) = self.lookup_local(first) {
                return Self::mk(hir::ExprKind::Local(id), ty, span);
            }
            if let Some(ty) = self.globals.get(first) {
                return Self::mk(hir::ExprKind::Global(first.to_string()), ty.clone(), span);
            }
            if let Some((params, ret)) = self.fn_sigs.get(first) {
                let ty = Ty::Fn(params.clone(), Box::new(ret.clone()));
                return Self::mk(hir::ExprKind::Fn(first.to_string()), ty, span);
            }
            if self.struct_fields.contains_key(first) {
                self.err(
                    span.clone(),
                    format!(
                        "`{first}` is a struct, not a value; construct it with `{first} {{ .. }}`"
                    ),
                );
                return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
            }
        }
        if first == "std" || self.imports.contains(first) || matches!(first, "Ok" | "Err") {
            return Self::mk(hir::ExprKind::Extern(names), Ty::Unknown, span);
        }
        self.err(
            span.clone(),
            format!("cannot find `{}` in this scope", names.join("::")),
        );
        Self::mk(hir::ExprKind::Error, Ty::Unknown, span)
    }

    fn field_ty(&mut self, base: &hir::Expr, name: &ast::Ident) -> Ty {
        let n = name.node.as_str();
        match &base.ty {
            Ty::Struct(s) => {
                let found = self
                    .struct_fields
                    .get(s)
                    .and_then(|f| f.iter().find(|(fname, _)| fname == n));
                match found {
                    Some((_, ty)) => ty.clone(),
                    None => {
                        let ty = base.ty.clone();
                        self.err(name.span.clone(), format!("no field `{n}` on type `{ty}`"));
                        Ty::Unknown
                    }
                }
            }
            Ty::Prim(Prim::Str) => match n {
                "is_empty" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "len" => Ty::Fn(vec![], Box::new(Ty::Prim(Prim::U64))),
                "trim" | "to_lowercase" | "to_uppercase" => Ty::Fn(vec![], Box::new(Ty::str())),
                _ => Ty::Unknown,
            },
            Ty::Vec(_) => match n {
                "is_empty" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "len" => Ty::Fn(vec![], Box::new(Ty::Prim(Prim::U64))),
                _ => Ty::Unknown,
            },
            _ => Ty::Unknown,
        }
    }

    fn call(
        &mut self,
        callee: &ast::Expr,
        args: &[ast::Expr],
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        // `Ok(x)` / `Err(x)` build a `Result`, typed from context when available.
        if let ast::ExprKind::Path(p) = &callee.node {
            if p.len() == 1
                && self.lookup_local(&p[0].node).is_none()
                && !self.fn_sigs.contains_key(&p[0].node)
            {
                let ctor = p[0].node.as_str();
                if (ctor == "Ok" || ctor == "Err") && args.len() == 1 {
                    return self.result_ctor(ctor == "Ok", callee, &args[0], expected, span);
                }
                if ctor == "Ok" || ctor == "Err" {
                    self.err(span.clone(), format!("`{ctor}` takes exactly one argument"));
                }
            }
        }

        let callee = self.expr(callee, None);
        match callee.ty.clone() {
            Ty::Fn(params, ret) => {
                if params.len() != args.len() {
                    self.err(
                        span.clone(),
                        format!(
                            "expected {} argument(s), found {}",
                            params.len(),
                            args.len()
                        ),
                    );
                }
                let args = args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let want = params.get(i);
                        let a = self.expr(a, want);
                        match want {
                            Some(w) => self.coerce(a, w),
                            None => a,
                        }
                    })
                    .collect();
                Self::mk(
                    hir::ExprKind::Call {
                        callee: Box::new(callee),
                        args,
                    },
                    *ret,
                    span,
                )
            }
            other => {
                if other != Ty::Unknown {
                    self.err(
                        callee.span.clone(),
                        format!("expected a function, found `{other}`"),
                    );
                }
                let args = args.iter().map(|a| self.expr(a, None)).collect();
                Self::mk(
                    hir::ExprKind::Call {
                        callee: Box::new(callee),
                        args,
                    },
                    Ty::Unknown,
                    span,
                )
            }
        }
    }

    fn result_ctor(
        &mut self,
        is_ok: bool,
        callee: &ast::Expr,
        arg: &ast::Expr,
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        let (want_ok, want_err) = match expected {
            Some(Ty::Result(o, e)) => ((**o).clone(), (**e).clone()),
            _ => (Ty::Unknown, Ty::Unknown),
        };
        let want = if is_ok { &want_ok } else { &want_err };
        let arg = self.expr(arg, Some(want));
        let arg = self.coerce(arg, want);
        let ty = if is_ok {
            Ty::Result(Box::new(arg.ty.clone()), Box::new(want_err))
        } else {
            Ty::Result(Box::new(want_ok), Box::new(arg.ty.clone()))
        };
        // Keep the declared type when the argument is less specific.
        let ty = match expected {
            Some(e @ Ty::Result(..)) if e.accepts(&ty) => e.clone(),
            _ => ty,
        };
        let callee_name = match &callee.node {
            ast::ExprKind::Path(p) => p.iter().map(|i| i.node.clone()).collect(),
            _ => vec![],
        };
        let callee = Self::mk(
            hir::ExprKind::Extern(callee_name),
            Ty::Unknown,
            &callee.span,
        );
        Self::mk(
            hir::ExprKind::Call {
                callee: Box::new(callee),
                args: vec![arg],
            },
            ty,
            span,
        )
    }

    /// `lhs |> f(a, b)` is checked as `f(lhs, a, b)`; `lhs |> f` as `f(lhs)`.
    fn pipe(
        &mut self,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        use ast::ExprKind as K;
        let call = match &rhs.node {
            K::Call { callee, args } => {
                let mut all = Vec::with_capacity(args.len() + 1);
                all.push(lhs.clone());
                all.extend(args.iter().cloned());
                K::Call {
                    callee: callee.clone(),
                    args: all,
                }
            }
            K::Path(_) | K::Field { .. } | K::Closure { .. } => K::Call {
                callee: Box::new(rhs.clone()),
                args: vec![lhs.clone()],
            },
            _ => {
                self.diags.push(
                    Diagnostic::error(
                        rhs.span.clone(),
                        "the right side of `|>` must be a function or a call",
                    )
                    .with_note("write `x |> f()` or `x |> f`"),
                );
                let _ = self.expr(lhs, None);
                return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
            }
        };
        let desugared = ast::Spanned {
            node: call,
            span: span.clone(),
        };
        self.expr(&desugared, expected)
    }

    fn binary(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
        span: &Span,
    ) -> hir::Expr {
        use ast::BinOp::*;
        let l = self.expr(lhs, None);
        let r = self.expr(rhs, Some(&l.ty));
        // Literals adopt the other side's numeric type: `x + 1` with `x: f32`.
        let l = if matches!(lhs.node, ast::ExprKind::Lit(_)) && l.ty != r.ty && r.ty != Ty::Unknown
        {
            self.expr(lhs, Some(&r.ty))
        } else {
            l
        };
        let compatible = l.ty.accepts(&r.ty) && r.ty.accepts(&l.ty);
        if !compatible {
            self.diags.push(
                Diagnostic::error(
                    span.clone(),
                    format!("mismatched operand types: `{}` and `{}`", l.ty, r.ty),
                )
                .with_label(l.span.clone(), format!("this is `{}`", l.ty))
                .with_label(r.span.clone(), format!("this is `{}`", r.ty)),
            );
        }
        let is_num = |t: &Ty| matches!(t, Ty::Prim(p) if p.is_numeric()) || *t == Ty::Unknown;
        let ty = match op {
            Add | Sub | Mul | Div | Rem => {
                let string_concat = op == Add && l.ty == Ty::str();
                if compatible && !is_num(&l.ty) && !string_concat {
                    self.err(
                        span.clone(),
                        format!("cannot apply arithmetic to `{}`", l.ty),
                    );
                }
                if l.ty == Ty::Unknown {
                    r.ty.clone()
                } else {
                    l.ty.clone()
                }
            }
            Lt | LtEq | Gt | GtEq => {
                if compatible && !is_num(&l.ty) && l.ty != Ty::str() {
                    self.err(
                        span.clone(),
                        format!("cannot order values of type `{}`", l.ty),
                    );
                }
                Ty::bool()
            }
            Eq | NotEq => Ty::bool(),
            And | Or => {
                for side in [&l, &r] {
                    if !Ty::bool().accepts(&side.ty) {
                        self.err(
                            side.span.clone(),
                            format!("expected `bool`, found `{}`", side.ty),
                        );
                    }
                }
                Ty::bool()
            }
        };
        Self::mk(
            hir::ExprKind::Binary {
                op,
                lhs: Box::new(l),
                rhs: Box::new(r),
            },
            ty,
            span,
        )
    }

    fn struct_lit(
        &mut self,
        path: &[ast::Ident],
        fields: &[(ast::Ident, ast::Expr)],
        span: &Span,
    ) -> hir::Expr {
        let name = path
            .iter()
            .map(|i| i.node.as_str())
            .collect::<Vec<_>>()
            .join("::");
        let Some(decl) = self.struct_fields.get(&name).cloned() else {
            self.err(span.clone(), format!("cannot find struct `{name}`"));
            for (_, e) in fields {
                let _ = self.expr(e, None);
            }
            return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
        };
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (fname, value) in fields {
            let want = decl
                .iter()
                .find(|(n, _)| *n == fname.node)
                .map(|(_, t)| t.clone());
            if want.is_none() {
                self.err(
                    fname.span.clone(),
                    format!("struct `{name}` has no field `{}`", fname.node),
                );
            }
            if !seen.insert(fname.node.clone()) {
                self.err(
                    fname.span.clone(),
                    format!("field `{}` given twice", fname.node),
                );
            }
            let v = self.expr(value, want.as_ref());
            let v = match &want {
                Some(w) => self.coerce(v, w),
                None => v,
            };
            out.push((fname.node.clone(), v));
        }
        let missing: Vec<_> = decl
            .iter()
            .filter(|(n, _)| !seen.contains(n))
            .map(|(n, _)| format!("`{n}`"))
            .collect();
        if !missing.is_empty() {
            self.err(
                span.clone(),
                format!("missing field(s) in `{name}`: {}", missing.join(", ")),
            );
        }
        Self::mk(
            hir::ExprKind::StructLit {
                name: name.clone(),
                fields: out,
            },
            Ty::Struct(name),
            span,
        )
    }
}
