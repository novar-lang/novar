//! Name resolution and type checking: AST -> typed HIR.

use std::collections::{HashMap, HashSet};

use telvax_ast as ast;
use telvax_diagnostics::Diagnostic;

use crate::hir::{self, Endpoint, HttpMethod, LocalId};
use crate::stdlib::{self, Sig};
use crate::ty::{Dim, Prim, Ty};

type Span = std::ops::Range<usize>;

/// A variant (or `Ok` / `Err`) with its enum name and payload types, for exhaustiveness checks.
type Constructor = (Option<String>, String, Vec<Ty>);

/// A generic function (or method) instance waiting to have its body checked.
struct Pending {
    f: ast::Fn,
    impl_ty: Option<String>,
    subst: HashMap<String, Ty>,
}

#[derive(Default)]
pub(crate) struct Checker {
    diags: Vec<Diagnostic>,
    /// Imported name -> its full path.
    imports: HashMap<String, Vec<String>>,
    struct_fields: HashMap<String, Vec<(String, Ty)>>,
    /// Enum name -> its variants and their payload types.
    enums: HashMap<String, Vec<(String, Vec<Ty>)>>,
    /// Row types written inline (`Table<{ a: i64 }>`), turned into structs named by their text.
    anon_structs: Vec<hir::StructDef>,
    /// While checking the functions of an `impl`, the type it is for (what `Self` means).
    impl_ty: Option<String>,
    /// Declared traits: the names of their methods and whether each is required (no body).
    traits: HashMap<String, Vec<(String, bool)>>,
    /// `impl Trait for Type` pairs.
    impls: HashSet<(String, String)>,
    impl_spans: Vec<(String, String, Span)>,
    /// For a method defined in `impl Trait for Type`, the trait it belongs to.
    impl_of_fn: HashMap<String, String>,
    /// Generic functions, structs, enums and `impl` blocks, instantiated for the types that use them.
    generic_fns: HashMap<String, ast::Fn>,
    generic_structs: HashMap<String, ast::Struct>,
    generic_enums: HashMap<String, ast::Enum>,
    generic_impls: HashMap<String, Vec<ast::Impl>>,
    /// Type parameters in scope and the types they stand for.
    type_params: HashMap<String, Ty>,
    /// For an instance of a generic type (`Stack<i64>`): the generic it came from and its arguments.
    instance_args: HashMap<String, (String, Vec<Ty>)>,
    /// Instances whose bodies still have to be checked.
    pending: Vec<Pending>,
    anon_enums: Vec<hir::EnumDef>,
    /// Functions of `impl` blocks, by their full name (`Point::norm`), with the type they belong to.
    methods: HashMap<String, String>,
    fn_sigs: HashMap<String, (Vec<Ty>, Ty)>,
    globals: HashMap<String, Ty>,
    scopes: Vec<HashMap<String, (LocalId, Ty, bool)>>,
    next_local: u32,
    ret: Ty,
    /// How many loops enclose the expression being checked (reset inside closures).
    loop_depth: u32,
    /// For each closure being checked, how many scopes existed before it: a variable found in a
    /// scope below that floor was captured by value.
    closure_floors: Vec<usize>,
    endpoints: HashSet<(HttpMethod, String)>,
    /// Bindings of literal type variables (`Ty::IntVar`/`Ty::FloatVar`), indexed by id.
    subst: Vec<Option<Ty>>,
    /// Set once every struct's fields are known (table schemas are validated from then on).
    structs_resolved: bool,
}

pub(crate) fn lower(program: &ast::Program) -> (hir::Program, Vec<Diagnostic>) {
    let mut c = Checker::default();
    let out = c.run(program);
    (out, c.diags)
}

/// The std module whose functions are available as methods on a value of type `ty`.
fn method_module(ty: &Ty, name: &str) -> Option<&'static str> {
    match ty {
        Ty::Tensor { .. }
            if matches!(
                name,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "scale"
                    | "relu"
                    | "sigmoid"
                    | "softmax"
                    | "sum"
                    | "mean"
                    | "argmax"
                    | "dot"
                    | "matmul"
                    | "transpose"
                    | "reshape"
                    | "to_vec"
            ) =>
        {
            Some("tensor")
        }
        Ty::Table(_) if name == "sql" => Some("table"),
        Ty::Vec(_)
            if matches!(
                name,
                "map"
                    | "filter"
                    | "take"
                    | "count"
                    | "any"
                    | "all"
                    | "find"
                    | "for_each"
                    | "fold"
                    | "enumerate"
                    | "zip"
                    | "sum"
                    | "min"
                    | "max"
                    | "skip"
            ) =>
        {
            Some("iter")
        }
        Ty::Vec(_) if name == "join" => Some("string"),
        _ => None,
    }
}

/// Names usable without `std::`: the ones nearly every program needs.
fn prelude(name: &str) -> Option<Vec<String>> {
    let path: &[&str] = match name {
        // `print` ends the line, as in Python. `std.io.print` writes without one.
        "print" => &["std", "io", "println"],
        "assert" => &["std", "test", "assert"],
        "assert_eq" => &["std", "test", "assert_eq"],
        "read_csv" => &["std", "table", "read_csv"],
        "read_parquet" => &["std", "table", "read_parquet"],
        _ => return None,
    };
    Some(path.iter().map(|s| (*s).to_string()).collect())
}

/// The candidate closest to `name` by edit distance (at most 2, and well under the name's length).
fn closest<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let limit = (name.chars().count() / 3).clamp(1, 2);
    candidates
        .into_iter()
        .filter(|c| *c != name && !c.starts_with('_'))
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, _)| *d <= limit)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

/// Edit distance where swapping two neighbouring letters counts once (`widht` -> `width`).
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

impl Checker {
    /// Like `err`, with a "did you mean" note when `hint` names a close match.
    fn err_hint(&mut self, span: Span, msg: impl Into<String>, hint: Option<&str>) {
        let mut d = Diagnostic::error(span, msg);
        if let Some(h) = hint {
            d = d.with_note(format!("did you mean `{h}`?"));
        }
        self.diags.push(d);
    }

    fn err(&mut self, span: Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::error(span, msg));
    }

    fn run(&mut self, program: &ast::Program) -> hir::Program {
        let mut out = hir::Program::default();

        for item in &program.items {
            if let ast::Item::Import(i) = item {
                let base: Vec<String> = i.path.iter().map(|s| s.node.clone()).collect();
                let mut bring_in = |this: &mut Self, name: &ast::Ident, full: Vec<String>| {
                    let joined = full.join("::");
                    let is_std = full.first().is_some_and(|f| f == "std");
                    let known = stdlib::function(&joined).is_some()
                        || stdlib::is_type(&joined)
                        || stdlib::is_module(&joined);
                    if is_std && !known {
                        this.err(name.span.clone(), format!("unresolved import `{joined}`"));
                    }
                    this.imports.insert(name.node.clone(), full.clone());
                    out.imports.push(full);
                };
                if i.group.is_empty() {
                    if let Some(last) = i.path.last() {
                        bring_in(self, last, base);
                    }
                } else {
                    for name in &i.group {
                        let mut full = base.clone();
                        full.push(name.node.clone());
                        bring_in(self, name, full);
                    }
                }
            }
        }

        // Declare struct names first so fields and signatures may refer to any struct.
        let mut declared = HashSet::new();
        // Generic definitions are kept as written and instantiated where they are used.
        for item in &program.items {
            match item {
                ast::Item::Struct(s) if !s.generics.is_empty() => {
                    if !declared.insert(s.name.node.clone()) {
                        self.err(
                            s.name.span.clone(),
                            format!("type `{}` is defined twice", s.name.node),
                        );
                    }
                    self.generic_structs.insert(s.name.node.clone(), s.clone());
                }
                ast::Item::Enum(en) if !en.generics.is_empty() => {
                    if !declared.insert(en.name.node.clone()) {
                        self.err(
                            en.name.span.clone(),
                            format!("`{}` is defined twice", en.name.node),
                        );
                    }
                    self.generic_enums.insert(en.name.node.clone(), en.clone());
                }
                _ => {}
            }
        }
        for item in &program.items {
            if let ast::Item::Struct(s) = item
                && s.generics.is_empty()
            {
                if !declared.insert(s.name.node.clone()) {
                    self.err(
                        s.name.span.clone(),
                        format!("type `{}` is defined twice", s.name.node),
                    );
                } else {
                    self.struct_fields.insert(s.name.node.clone(), Vec::new());
                }
            }
        }
        for item in &program.items {
            if let ast::Item::Enum(en) = item
                && en.generics.is_empty()
            {
                if !declared.insert(en.name.node.clone()) {
                    self.err(
                        en.name.span.clone(),
                        format!("`{}` is defined twice", en.name.node),
                    );
                } else {
                    self.enums.insert(en.name.node.clone(), Vec::new());
                }
            }
        }
        for item in &program.items {
            if let ast::Item::Trait(t) = item {
                if !declared.insert(t.name.node.clone()) {
                    self.err(
                        t.name.span.clone(),
                        format!("`{}` is defined twice", t.name.node),
                    );
                } else {
                    let methods = t
                        .fns
                        .iter()
                        .map(|f| (f.name.node.clone(), f.required))
                        .collect();
                    self.traits.insert(t.name.node.clone(), methods);
                }
            }
        }
        let mut done = HashSet::new();
        for item in &program.items {
            if let ast::Item::Struct(s) = item
                && s.generics.is_empty()
            {
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
                    name_span: s.name.span.clone(),
                    is_pub: s.is_pub,
                    fields,
                    span: s.span.clone(),
                });
            }
        }

        let mut enum_done = HashSet::new();
        for item in &program.items {
            let ast::Item::Enum(en) = item else { continue };
            if !en.generics.is_empty() {
                continue;
            }
            if !enum_done.insert(en.name.node.clone()) {
                continue;
            }
            let mut variants: Vec<(String, Vec<Ty>)> = Vec::new();
            for v in &en.variants {
                if variants.iter().any(|(n, _)| *n == v.name.node) {
                    self.err(
                        v.name.span.clone(),
                        format!("variant `{}` is declared twice", v.name.node),
                    );
                    continue;
                }
                let tys = v.fields.iter().map(|t| self.resolve_type(t)).collect();
                variants.push((v.name.node.clone(), tys));
            }
            self.enums.insert(en.name.node.clone(), variants.clone());
            out.enums.push(hir::EnumDef {
                name: en.name.node.clone(),
                name_span: en.name.span.clone(),
                is_pub: en.is_pub,
                variants,
                span: en.span.clone(),
            });
        }

        self.structs_resolved = true;

        // Every function: the free ones, and the methods of `impl` blocks under the name
        // `Type::method`.
        let mut fns: Vec<(ast::Fn, Option<String>)> = Vec::new();
        for item in &program.items {
            match item {
                ast::Item::Fn(f) if !f.generics.is_empty() => {
                    if self
                        .generic_fns
                        .insert(f.name.node.clone(), f.clone())
                        .is_some()
                    {
                        self.err(
                            f.name.span.clone(),
                            format!("function `{}` is defined twice", f.name.node),
                        );
                    }
                }
                ast::Item::Impl(imp) if !imp.generics.is_empty() => {
                    if self.generic_structs.contains_key(&imp.ty.node)
                        || self.generic_enums.contains_key(&imp.ty.node)
                    {
                        self.generic_impls
                            .entry(imp.ty.node.clone())
                            .or_default()
                            .push(imp.clone());
                    } else {
                        self.err(
                            imp.ty.span.clone(),
                            format!("cannot find the generic type or enum `{}`", imp.ty.node),
                        );
                    }
                }
                ast::Item::Fn(f) => fns.push((f.clone(), None)),
                ast::Item::Trait(t) => {
                    for f in &t.fns {
                        let mut f = f.clone();
                        f.name.node = format!("{}::{}", t.name.node, f.name.node);
                        self.methods
                            .insert(f.name.node.clone(), t.name.node.clone());
                        fns.push((f, Some(t.name.node.clone())));
                    }
                }
                ast::Item::Impl(imp) => {
                    let ty = imp.ty.node.clone();
                    if !self.struct_fields.contains_key(&ty) && !self.enums.contains_key(&ty) {
                        self.err(
                            imp.ty.span.clone(),
                            format!("cannot find the type or enum `{ty}` to extend"),
                        );
                        continue;
                    }
                    if let Some(tr) = &imp.trait_name {
                        if self.traits.contains_key(&tr.node) {
                            self.impls.insert((tr.node.clone(), ty.clone()));
                            self.impl_spans
                                .push((tr.node.clone(), ty.clone(), imp.span.clone()));
                        } else {
                            self.err(
                                tr.span.clone(),
                                format!("cannot find the trait `{}`", tr.node),
                            );
                            continue;
                        }
                    }
                    for f in &imp.fns {
                        let mut f = f.clone();
                        f.name.node = format!("{ty}::{}", f.name.node);
                        if let Some(tr) = &imp.trait_name {
                            self.impl_of_fn.insert(f.name.node.clone(), tr.node.clone());
                        }
                        self.methods.insert(f.name.node.clone(), ty.clone());
                        fns.push((f, Some(ty.clone())));
                    }
                }
                _ => {}
            }
        }

        // Function signatures.
        let mut sig_done = HashSet::new();
        for (f, impl_ty) in &fns {
            if !sig_done.insert(f.name.node.clone()) {
                self.err(
                    f.name.span.clone(),
                    format!("function `{}` is defined twice", f.name.node),
                );
                continue;
            }
            let in_trait = impl_ty
                .as_ref()
                .is_some_and(|t| self.traits.contains_key(t));
            if f.required && !in_trait {
                self.err(
                    f.name.span.clone(),
                    format!("function `{}` needs a body", f.name.node),
                );
            }
            self.impl_ty = impl_ty.clone();
            let params = f.params.iter().map(|p| self.resolve_type(&p.ty)).collect();
            let ret = f.ret.as_ref().map_or(Ty::Unit, |t| self.resolve_type(t));
            self.impl_ty = None;
            self.fn_sigs.insert(f.name.node.clone(), (params, ret));
        }
        self.check_trait_impls(&fns);

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
                let mut init = init;
                self.finalize_expr(&mut init);
                let ty = self.finalize_ty(&ty);
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
        for (f, impl_ty) in &fns {
            if f.required {
                continue;
            }
            if lowered.insert(f.name.node.clone()) {
                self.impl_ty = impl_ty.clone();
                out.fns.push(self.lower_fn(f));
                self.impl_ty = None;
            }
        }
        // Bodies of the instances created along the way (which may create more).
        while let Some(p) = self.pending.pop() {
            self.type_params = p.subst;
            self.impl_ty = p.impl_ty;
            out.fns.push(self.lower_fn(&p.f));
            self.type_params.clear();
            self.impl_ty = None;
        }
        out.structs.append(&mut self.anon_structs);
        out.enums.append(&mut self.anon_enums);
        out
    }

    // ---- types -------------------------------------------------------------------------

    fn resolve_type(&mut self, t: &ast::Type) -> Ty {
        let names: Vec<&str> = t.path.iter().map(|i| i.node.as_str()).collect();
        if names.len() > 1 {
            if names[0] == "std" || self.imports.contains_key(names[0]) {
                return Ty::Named(names.join("::"));
            }
            self.err(
                t.span.clone(),
                format!("cannot find type `{}`", names.join(".")),
            );
            return Ty::Unknown;
        }
        let name = names[0];
        if let Some(t) = self.type_params.get(name) {
            return t.clone();
        }
        let type_args: Vec<&ast::Type> = t
            .args
            .iter()
            .filter_map(|a| match a {
                ast::TypeArg::Type(t) => Some(t),
                ast::TypeArg::Shape(_) | ast::TypeArg::Row(_) => None,
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
            "Table" if matches!(t.args.as_slice(), [ast::TypeArg::Row(_)]) => {
                let ast::TypeArg::Row(fields) = &t.args[0] else {
                    unreachable!("matched above")
                };
                self.row_type(fields, &t.span)
            }
            "Table" => match type_args.as_slice() {
                [inner] if t.args.len() == 1 => match self.resolve_type(inner) {
                    Ty::Struct(sname) => {
                        self.check_table_schema(&sname, &t.span);
                        Ty::Table(sname)
                    }
                    Ty::Unknown => Ty::Unknown,
                    other => {
                        self.err(
                            inner.span.clone(),
                            format!("a table's schema must be a type, found `{other}`"),
                        );
                        Ty::Unknown
                    }
                },
                _ => {
                    self.err(
                        t.span.clone(),
                        "`Table` takes one type argument: `Table<Row>`",
                    );
                    Ty::Unknown
                }
            },
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
            "Self" => match self.impl_ty.clone() {
                Some(t) if self.traits.contains_key(&t) => Ty::Trait(t),
                Some(t) if self.enums.contains_key(&t) => Ty::Enum(t),
                Some(t)
                    if self
                        .instance_args
                        .get(&t)
                        .is_some_and(|(base, _)| self.generic_enums.contains_key(base)) =>
                {
                    Ty::Enum(t)
                }
                Some(t) => Ty::Struct(t),
                None => {
                    self.err(t.span.clone(), "`Self` can only be used inside a `type`");
                    Ty::Unknown
                }
            },
            "(tuple)" => Ty::Tuple(type_args.iter().map(|a| self.resolve_type(a)).collect()),
            "Option" => match type_args.as_slice() {
                [inner] if t.args.len() == 1 => Ty::Option(Box::new(self.resolve_type(inner))),
                _ => {
                    self.err(t.span.clone(), "`Option` takes one type argument: `T?`");
                    Ty::Unknown
                }
            },
            "Map" => match type_args.as_slice() {
                [k, v] if t.args.len() == 2 => {
                    let key = self.resolve_type(k);
                    if !matches!(&key, Ty::Prim(p) if p.is_int() || matches!(p, Prim::Str | Prim::Bool))
                        && !matches!(key, Ty::Unknown)
                    {
                        self.err(
                            k.span.clone(),
                            format!("`Map` keys must be a Text, an Int or a Bool, found `{key}`"),
                        );
                    }
                    Ty::Map(Box::new(key), Box::new(self.resolve_type(v)))
                }
                _ => {
                    self.err(
                        t.span.clone(),
                        "`Map` takes two type arguments: `Map[K, V]`",
                    );
                    Ty::Unknown
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
            _ if self.generic_structs.contains_key(name)
                || self.generic_enums.contains_key(name) =>
            {
                let args: Vec<Ty> = type_args.iter().map(|a| self.resolve_type(a)).collect();
                self.instantiate_adt(name, args, &t.span)
            }
            _ if self.struct_fields.contains_key(name) => {
                no_args(self);
                Ty::Struct(name.to_string())
            }
            _ if self.enums.contains_key(name) => {
                no_args(self);
                Ty::Enum(name.to_string())
            }
            _ if self.traits.contains_key(name) => {
                no_args(self);
                Ty::Trait(name.to_string())
            }
            _ if self.imports.contains_key(name) => Ty::Named(name.to_string()),
            _ => {
                let mut known: Vec<&str> = vec![
                    "Int", "Float", "Text", "Bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32",
                    "u64", "f32", "f64", "List", "Map", "Tensor", "Table", "Result",
                ];
                known.extend(self.struct_fields.keys().map(String::as_str));
                known.extend(self.enums.keys().map(String::as_str));
                known.extend(self.imports.keys().map(String::as_str));
                // The spellings other languages use for the types Telvax capitalizes.
                let hint = match name {
                    "int" | "integer" => Some("Int".to_string()),
                    "float" | "double" => Some("Float".to_string()),
                    "str" | "string" | "String" => Some("Text".to_string()),
                    "bool" | "boolean" => Some("Bool".to_string()),
                    "list" | "array" => Some("List".to_string()),
                    "dict" | "map" | "hashmap" => Some("Map".to_string()),
                    _ => closest(name, known).map(str::to_string),
                };
                self.err_hint(
                    t.span.clone(),
                    format!("cannot find type `{name}` in this scope"),
                    hint.as_deref(),
                );
                Ty::Unknown
            }
        }
    }

    /// `Table<{ region: String, revenue: f64 }>`: the row is a struct named by its own text, so
    /// two tables written with the same columns share one type.
    fn row_type(&mut self, fields: &[ast::Field], span: &Span) -> Ty {
        if fields.is_empty() {
            self.err(span.clone(), "a row type needs at least one column");
            return Ty::Unknown;
        }
        let mut columns: Vec<(String, Ty)> = Vec::new();
        for f in fields {
            if columns.iter().any(|(n, _)| *n == f.name.node) {
                self.err(
                    f.name.span.clone(),
                    format!("column `{}` is declared twice", f.name.node),
                );
                continue;
            }
            let ty = self.resolve_type(&f.ty);
            columns.push((f.name.node.clone(), ty));
        }
        let name = format!(
            "{{ {} }}",
            columns
                .iter()
                .map(|(n, t)| format!("{n}: {t}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if !self.struct_fields.contains_key(&name) {
            self.struct_fields.insert(name.clone(), columns.clone());
            self.anon_structs.push(hir::StructDef {
                name: name.clone(),
                name_span: span.clone(),
                is_pub: false,
                fields: columns,
                span: span.clone(),
            });
        }
        self.check_table_schema(&name, span);
        Ty::Table(name)
    }

    /// Every column of a table must be a primitive (number, bool or string).
    fn check_table_schema(&mut self, sname: &str, span: &Span) {
        let Some(fields) = self.struct_fields.get(sname).cloned() else {
            return;
        };
        if self.structs_resolved && fields.is_empty() {
            self.err(
                span.clone(),
                format!("`{sname}` has no fields, so it cannot be a table schema"),
            );
        }
        for (name, ty) in fields {
            if !matches!(ty, Ty::Prim(_) | Ty::Unknown) {
                self.err(
                    span.clone(),
                    format!("column `{name}` of `{sname}` has type `{ty}`; table columns must be numbers, bools or strings"),
                );
            }
        }
    }

    fn resolve_tensor(&mut self, t: &ast::Type) -> Ty {
        let (elem, shape) = match t.args.as_slice() {
            [ast::TypeArg::Type(e), ast::TypeArg::Shape(s)] => (e, s),
            _ => {
                self.err(
                    t.span.clone(),
                    "`Tensor` expects an element type and a shape: `Tensor[f32, N, M]`",
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

    // ---- literal type variables ----------------------------------------------------------

    fn fresh(&mut self, float: bool) -> Ty {
        self.subst.push(None);
        let id = (self.subst.len() - 1) as u32;
        if float {
            Ty::FloatVar(id)
        } else {
            Ty::IntVar(id)
        }
    }

    /// Replaces bound variables (recursively) by what they stand for.
    fn resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::IntVar(v) | Ty::FloatVar(v) => match &self.subst[*v as usize] {
                Some(t) => self.resolve(t),
                None => ty.clone(),
            },
            Ty::Vec(t) => Ty::Vec(Box::new(self.resolve(t))),
            Ty::Result(a, b) => Ty::Result(Box::new(self.resolve(a)), Box::new(self.resolve(b))),
            Ty::Option(t) => Ty::Option(Box::new(self.resolve(t))),
            Ty::Tuple(items) => Ty::Tuple(items.iter().map(|t| self.resolve(t)).collect()),
            Ty::Map(k, v) => Ty::Map(Box::new(self.resolve(k)), Box::new(self.resolve(v))),
            Ty::Fn(ps, r) => Ty::Fn(
                ps.iter().map(|p| self.resolve(p)).collect(),
                Box::new(self.resolve(r)),
            ),
            other => other.clone(),
        }
    }

    fn finalize_pat(&mut self, p: &mut hir::Pat) {
        match p {
            hir::Pat::Bind(l) => l.ty = self.finalize_ty(&l.ty),
            hir::Pat::Variant { args, .. } => {
                for a in args {
                    self.finalize_pat(a);
                }
            }
            hir::Pat::Tuple(items) => items.iter_mut().for_each(|a| self.finalize_pat(a)),
            hir::Pat::Wild | hir::Pat::Lit(_) => {}
        }
    }

    /// Like `resolve`, but also pins still-free literal variables to their defaults.
    fn finalize_ty(&mut self, ty: &Ty) -> Ty {
        match self.resolve(ty) {
            Ty::IntVar(v) => {
                self.subst[v as usize] = Some(Ty::Prim(Prim::I32));
                Ty::Prim(Prim::I32)
            }
            Ty::FloatVar(v) => {
                self.subst[v as usize] = Some(Ty::Prim(Prim::F64));
                Ty::Prim(Prim::F64)
            }
            Ty::Vec(t) => Ty::Vec(Box::new(self.finalize_ty(&t))),
            Ty::Option(t) => Ty::Option(Box::new(self.finalize_ty(&t))),
            Ty::Tuple(items) => Ty::Tuple(items.iter().map(|t| self.finalize_ty(t)).collect()),
            Ty::Map(k, v) => Ty::Map(
                Box::new(self.finalize_ty(&k)),
                Box::new(self.finalize_ty(&v)),
            ),
            Ty::Result(a, b) => Ty::Result(
                Box::new(self.finalize_ty(&a)),
                Box::new(self.finalize_ty(&b)),
            ),
            Ty::Fn(ps, r) => {
                let ps = ps.iter().map(|p| self.finalize_ty(p)).collect();
                Ty::Fn(ps, Box::new(self.finalize_ty(&r)))
            }
            other => other,
        }
    }

    /// Whether `actual` can be used where `expected` is required, binding literal variables.
    fn unify(&mut self, expected: &Ty, actual: &Ty) -> bool {
        let (a, b) = (self.resolve(expected), self.resolve(actual));
        match (&a, &b) {
            (Ty::IntVar(x), Ty::IntVar(y)) | (Ty::FloatVar(x), Ty::FloatVar(y)) => {
                if x != y {
                    self.subst[*x as usize] = Some(b.clone());
                }
                true
            }
            (Ty::IntVar(x), Ty::Prim(p)) | (Ty::Prim(p), Ty::IntVar(x)) if p.is_int() => {
                self.subst[*x as usize] = Some(Ty::Prim(*p));
                true
            }
            (Ty::FloatVar(x), Ty::Prim(p)) | (Ty::Prim(p), Ty::FloatVar(x)) if p.is_float() => {
                self.subst[*x as usize] = Some(Ty::Prim(*p));
                true
            }
            (Ty::Vec(x), Ty::Vec(y)) => self.unify(x, y),
            (Ty::Option(x), Ty::Option(y)) => self.unify(x, y),
            // A value of a type that implements the trait can be used where the trait is expected.
            (Ty::Trait(t), Ty::Struct(s) | Ty::Enum(s)) => {
                self.impls.contains(&(t.clone(), s.clone()))
            }
            (Ty::Tuple(xs), Ty::Tuple(ys)) if xs.len() == ys.len() => {
                let mut ok = true;
                for (x, y) in xs.iter().zip(ys) {
                    ok &= self.unify(x, y);
                }
                ok
            }
            (Ty::Map(k1, v1), Ty::Map(k2, v2)) => {
                let ok = self.unify(k1, k2);
                self.unify(v1, v2) && ok
            }
            (Ty::Result(a1, b1), Ty::Result(a2, b2)) => {
                let ok = self.unify(a1, a2);
                self.unify(b1, b2) && ok
            }
            (Ty::Fn(p1, r1), Ty::Fn(p2, r2)) if p1.len() == p2.len() => {
                let mut ok = true;
                for (x, y) in p1.iter().zip(p2) {
                    ok &= self.unify(x, y);
                }
                self.unify(r1, r2) && ok
            }
            _ => a.accepts(&b),
        }
    }

    /// Rewrites every type in `e` to its final form, defaulting unconstrained literals.
    fn finalize_expr(&mut self, e: &mut hir::Expr) {
        e.ty = self.finalize_ty(&e.ty);
        use hir::ExprKind as K;
        match &mut e.kind {
            K::Lit(_) | K::Local(_) | K::Global(_) | K::Fn(_) | K::Extern(_) | K::Error => {}
            K::Call { callee, args } => {
                self.finalize_expr(callee);
                args.iter_mut().for_each(|a| self.finalize_expr(a));
            }
            K::Field { base, .. } => self.finalize_expr(base),
            K::Unary { expr, .. } => self.finalize_expr(expr),
            K::Binary { lhs, rhs, .. } => {
                self.finalize_expr(lhs);
                self.finalize_expr(rhs);
            }
            K::Closure { params, body } => {
                for p in params {
                    p.ty = self.finalize_ty(&p.ty);
                }
                self.finalize_expr(body);
            }
            K::StructLit { fields, .. } => {
                fields.iter_mut().for_each(|(_, v)| self.finalize_expr(v))
            }
            K::If {
                cond,
                then,
                otherwise,
            } => {
                self.finalize_expr(cond);
                self.finalize_block(then);
                if let Some(o) = otherwise {
                    self.finalize_expr(o);
                }
            }
            K::Block(b) => self.finalize_block(b),
            K::Return(v) => {
                if let Some(v) = v {
                    self.finalize_expr(v);
                }
            }
            K::Break | K::Continue => {}
            K::Try(inner) => self.finalize_expr(inner),
            K::Tuple(items) => items.iter_mut().for_each(|i| self.finalize_expr(i)),
            K::DynCall { args, .. } => args.iter_mut().for_each(|a| self.finalize_expr(a)),
            K::EnumCtor { args, .. } => {
                for a in args {
                    self.finalize_expr(a);
                }
            }
            K::Match { scrutinee, arms } => {
                self.finalize_expr(scrutinee);
                for arm in arms {
                    self.finalize_pat(&mut arm.pat);
                    self.finalize_expr(&mut arm.body);
                }
            }
            K::FString(parts) => {
                for p in parts {
                    if let hir::FPart::Expr(x) = p {
                        self.finalize_expr(x);
                    }
                }
            }
            K::Assign { place, value } => {
                place.ty = self.finalize_ty(&place.ty);
                place.root_ty = self.finalize_ty(&place.root_ty);
                for pr in &mut place.projs {
                    if let hir::Proj::Index(i) = pr {
                        self.finalize_expr(i);
                    }
                }
                self.finalize_expr(value);
            }
            K::While { cond, body } => {
                self.finalize_expr(cond);
                self.finalize_block(body);
            }
            K::For { var, iter, body } => {
                var.ty = self.finalize_ty(&var.ty);
                match iter {
                    hir::ForIter::Range { start, end, .. } => {
                        self.finalize_expr(start);
                        self.finalize_expr(end);
                    }
                    hir::ForIter::Vec(v) => self.finalize_expr(v),
                }
                self.finalize_block(body);
            }
            K::Index { base, index } => {
                self.finalize_expr(base);
                self.finalize_expr(index);
            }
            K::Array(items) => items.iter_mut().for_each(|i| self.finalize_expr(i)),
        }
    }

    fn finalize_block(&mut self, b: &mut hir::Block) {
        b.ty = self.finalize_ty(&b.ty);
        for s in &mut b.stmts {
            match s {
                hir::Stmt::Let { local, init } => {
                    local.ty = self.finalize_ty(&local.ty);
                    self.finalize_expr(init);
                }
                hir::Stmt::Expr(e) => self.finalize_expr(e),
            }
        }
        if let Some(t) = &mut b.tail {
            self.finalize_expr(t);
        }
    }

    // ---- scopes ------------------------------------------------------------------------

    fn declare(&mut self, name: &ast::Ident, ty: Ty) -> hir::Local {
        self.declare_mut(name, ty, false)
    }

    fn declare_mut(&mut self, name: &ast::Ident, ty: Ty, mutable: bool) -> hir::Local {
        let id = LocalId(self.next_local);
        self.next_local += 1;
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.node.clone(), (id, ty.clone(), mutable));
        }
        hir::Local {
            id,
            name: name.node.clone(),
            ty,
            mutable,
            span: name.span.clone(),
        }
    }

    fn lookup_local(&self, name: &str) -> Option<(LocalId, Ty)> {
        self.lookup_local_full(name).map(|(id, ty, _)| (id, ty))
    }

    fn lookup_local_full(&self, name: &str) -> Option<(LocalId, Ty, bool)> {
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
            params.push(self.declare_mut(&p.name, ty, p.mutable));
        }

        let (mut body, body_ty) = self.block(&f.body, Some(&ret));
        if !self.unify(&ret, &body_ty) {
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

        self.finalize_block(&mut body);
        let endpoint = self.endpoint(f, &params, &ret);
        self.scopes.clear();
        hir::FnDef {
            name: f.name.node.clone(),
            name_span: f.name.span.clone(),
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
                    "an endpoint takes at most one parameter, which must be a `type` (the request payload)",
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
                    let local = self.declare_mut(&l.name, ty, l.mutable);
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
        if !self.unify(expected, &e.ty) {
            let (want, found) = (self.resolve(expected), self.resolve(&e.ty));
            self.err(
                e.span.clone(),
                format!("mismatched types: expected `{want}`, found `{found}`"),
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
            K::Named { name, .. } => {
                self.err(
                    span.clone(),
                    format!(
                        "`{}: ...` can only be written in the arguments of a call",
                        name.node
                    ),
                );
                Self::mk(hir::ExprKind::Error, Ty::Unknown, span)
            }
            K::Assign { target, value } => self.assign(target, value, span),
            K::While { cond, body } => {
                let cond = self.expr(cond, Some(&Ty::bool()));
                let cond = self.coerce(cond, &Ty::bool());
                self.loop_depth += 1;
                let (body, _) = self.block(body, None);
                self.loop_depth -= 1;
                Self::mk(
                    hir::ExprKind::While {
                        cond: Box::new(cond),
                        body,
                    },
                    Ty::Unit,
                    span,
                )
            }
            K::For { var, iter, body } => self.for_loop(var, iter, body, span),
            K::Index { base, index } => {
                let base = self.expr(base, None);
                let index = self.expr(index, None);
                let index_ok = matches!(self.resolve(&index.ty), Ty::Prim(p) if p.is_int())
                    || matches!(
                        self.resolve(&index.ty),
                        Ty::IntVar(_) | Ty::Unknown | Ty::Never
                    );
                if !index_ok {
                    self.err(
                        index.span.clone(),
                        format!("indices must be integers, found `{}`", index.ty),
                    );
                }
                let ty = match &base.ty {
                    Ty::Vec(t) => (**t).clone(),
                    Ty::Table(s) => Ty::Struct(s.clone()),
                    // Indexing peels the leading dimension: a row of a matrix is a vector.
                    Ty::Tensor { elem, shape } if shape.len() == 1 => (**elem).clone(),
                    Ty::Tensor { elem, shape } => Ty::Tensor {
                        elem: elem.clone(),
                        shape: shape[1..].to_vec(),
                    },
                    Ty::Unknown | Ty::Never => Ty::Unknown,
                    other => {
                        let other = other.clone();
                        self.err(
                            base.span.clone(),
                            format!("cannot index into a value of type `{other}`"),
                        );
                        Ty::Unknown
                    }
                };
                Self::mk(
                    hir::ExprKind::Index {
                        base: Box::new(base),
                        index: Box::new(index),
                    },
                    ty,
                    span,
                )
            }
            K::Array(items) => {
                let want = match expected {
                    Some(Ty::Vec(t)) => Some((**t).clone()),
                    _ => None,
                };
                let mut out: Vec<hir::Expr> = Vec::with_capacity(items.len());
                for it in items {
                    let hint = want.clone().or_else(|| out.first().map(|f| f.ty.clone()));
                    let it = self.expr(it, hint.as_ref());
                    let it = match &hint {
                        Some(h) => self.coerce(it, h),
                        None => it,
                    };
                    out.push(it);
                }
                let elem = want
                    .or_else(|| out.first().map(|f| f.ty.clone()))
                    .unwrap_or(Ty::Unknown);
                Self::mk(hir::ExprKind::Array(out), Ty::Vec(Box::new(elem)), span)
            }
            K::Lit(l) => {
                let want = expected.map(|t| self.resolve(t));
                let ty = match l {
                    ast::Lit::Int(_) => match want {
                        Some(Ty::Prim(p)) if p.is_numeric() => Ty::Prim(p),
                        Some(v @ (Ty::IntVar(_) | Ty::FloatVar(_))) => v,
                        _ => self.fresh(false),
                    },
                    ast::Lit::Float(_) => match want {
                        Some(Ty::Prim(p)) if p.is_float() => Ty::Prim(p),
                        Some(v @ Ty::FloatVar(_)) => v,
                        _ => self.fresh(true),
                    },
                    ast::Lit::Str(_) => Ty::str(),
                    ast::Lit::Bool(_) => Ty::bool(),
                };
                Self::mk(hir::ExprKind::Lit(l.clone()), ty, span)
            }
            K::Path(p) if self.is_option_ctor(p, "None") => {
                let inner = match expected {
                    Some(Ty::Option(t)) => (**t).clone(),
                    _ => Ty::Unknown,
                };
                Self::mk(
                    hir::ExprKind::EnumCtor {
                        enum_name: "Option".into(),
                        variant: "None".into(),
                        args: vec![],
                    },
                    Ty::Option(Box::new(inner)),
                    span,
                )
            }
            K::Path(p) => self.path(p, expected, span),
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
                let shown = self.resolve(&inner.ty);
                let ok = match (op, &shown) {
                    (_, Ty::Unknown) => true,
                    (ast::UnOp::Neg, Ty::IntVar(_) | Ty::FloatVar(_)) => true,
                    (ast::UnOp::Neg, Ty::Prim(p)) => p.is_numeric(),
                    (ast::UnOp::Not, Ty::Prim(Prim::Bool)) => true,
                    _ => false,
                };
                if !ok {
                    self.err(span.clone(), format!("cannot apply `{op:?}` to `{shown}`"));
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
            K::Closure {
                params,
                types,
                body,
            } => {
                let want: Option<(&Vec<Ty>, &Ty)> = match expected {
                    Some(Ty::Fn(p, r)) if p.len() == params.len() => Some((p, r)),
                    _ => None,
                };
                self.closure_floors.push(self.scopes.len());
                self.scopes.push(HashMap::new());
                let saved = std::mem::take(&mut self.ret);
                let saved_depth = std::mem::take(&mut self.loop_depth);
                let locals: Vec<_> = params
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let ty = match &types[i] {
                            Some(annotation) => self.resolve_type(annotation),
                            None => want.map_or(Ty::Unknown, |(ps, _)| ps[i].clone()),
                        };
                        self.declare(p, ty)
                    })
                    .collect();
                let body = self.expr(body, want.map(|(_, r)| r));
                self.ret = saved;
                self.loop_depth = saved_depth;
                self.scopes.pop();
                self.closure_floors.pop();
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
            K::StructLit { path, fields } => self.struct_lit(path, fields, expected, span),
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
                        let ty = if self.unify(&then_ty, &o.ty) {
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
            K::Match { scrutinee, arms } => self.match_expr(scrutinee, arms, expected, span),
            K::Tuple(items) => {
                let hints: Vec<Option<Ty>> = match expected.map(|e| self.resolve(e)) {
                    Some(Ty::Tuple(ts)) if ts.len() == items.len() => {
                        ts.into_iter().map(Some).collect()
                    }
                    _ => vec![None; items.len()],
                };
                let parts: Vec<hir::Expr> = items
                    .iter()
                    .zip(&hints)
                    .map(|(item, hint)| self.expr(item, hint.as_ref()))
                    .collect();
                let ty = Ty::Tuple(parts.iter().map(|p| p.ty.clone()).collect());
                Self::mk(hir::ExprKind::Tuple(parts), ty, span)
            }
            K::Try(inner) => {
                let value = self.expr(inner, None);
                let operand = self.resolve(&value.ty);
                let ret = self.resolve(&self.ret.clone());
                let out = match &operand {
                    Ty::Result(ok, err) => {
                        match &ret {
                            Ty::Result(_, ret_err) => {
                                if !ret_err.accepts(err) {
                                    self.err(
                                        span.clone(),
                                        format!(
                                            "`?` cannot turn the error `{err}` into `{ret_err}`, the error type of this function"
                                        ),
                                    );
                                }
                            }
                            Ty::Unknown | Ty::Never => {}
                            other => self.err(
                                span.clone(),
                                format!(
                                    "`?` can only be used in a function that returns `Result`, this one returns `{other}`"
                                ),
                            ),
                        }
                        (**ok).clone()
                    }
                    Ty::Option(inner_ty) => {
                        if !matches!(ret, Ty::Option(_) | Ty::Unknown | Ty::Never) {
                            self.err(
                                span.clone(),
                                format!(
                                    "`?` on an `Option` can only be used in a function that returns `Option`, this one returns `{ret}`"
                                ),
                            );
                        }
                        (**inner_ty).clone()
                    }
                    Ty::Unknown | Ty::Never => Ty::Unknown,
                    other => {
                        self.err(
                            value.span.clone(),
                            format!("`?` needs a `Result` or an `Option`, found `{other}`"),
                        );
                        Ty::Unknown
                    }
                };
                Self::mk(hir::ExprKind::Try(Box::new(value)), out, span)
            }
            K::Break | K::Continue => {
                let (word, kind) = if matches!(e.node, K::Break) {
                    ("break", hir::ExprKind::Break)
                } else {
                    ("continue", hir::ExprKind::Continue)
                };
                if self.loop_depth == 0 {
                    self.err(span.clone(), format!("`{word}` outside of a loop"));
                }
                Self::mk(kind, Ty::Never, span)
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

    /// Resolves an assignment target (`x`, `x.f`, `x[i]`, and combinations) to a [`hir::Place`].
    fn place(&mut self, target: &ast::Expr) -> Option<hir::Place> {
        match &target.node {
            ast::ExprKind::Path(p) if p.len() == 1 => {
                let name = p[0].node.clone();
                let Some((id, ty, mutable)) = self.lookup_local_full(&name) else {
                    if self.globals.contains_key(&name) {
                        self.err(
                            target.span.clone(),
                            format!("cannot assign to the global `{name}`"),
                        );
                    } else {
                        self.diags.push(
                            Diagnostic::error(
                                target.span.clone(),
                                format!("cannot find `{name}` in this scope"),
                            )
                            .with_note(format!(
                                "declare it first with `let {name} = ...`, or `var {name} = ...` if it changes"
                            )),
                        );
                    }
                    return None;
                };
                let captured = self.closure_floors.last().is_some_and(|floor| {
                    self.scopes
                        .iter()
                        .rposition(|s| s.contains_key(&name))
                        .is_some_and(|i| i < *floor)
                });
                // `Vec` and `Map` are shared, so changing their elements from a closure is seen
                // outside; any other captured value is a copy.
                if captured && !matches!(self.resolve(&ty), Ty::Vec(_) | Ty::Map(..) | Ty::Unknown)
                {
                    self.diags.push(
                        Diagnostic::error(
                            target.span.clone(),
                            format!("cannot assign to `{name}` from inside a closure"),
                        )
                        .with_note(
                            "closures capture variables by value, so the change would be lost; \
                             return the new value from the closure instead",
                        ),
                    );
                }
                if !mutable {
                    self.diags.push(
                        Diagnostic::error(
                            target.span.clone(),
                            format!("cannot assign to the immutable variable `{name}`"),
                        )
                        .with_note(format!(
                            "declare it with `var {name}` instead of `let {name}` (loop variables \
                             and pattern names cannot change: copy them into a `var`)"
                        )),
                    );
                }
                Some(hir::Place {
                    local: id,
                    name,
                    projs: vec![],
                    ty: ty.clone(),
                    root_ty: ty,
                    span: target.span.clone(),
                })
            }
            ast::ExprKind::Field { base, name } => {
                let mut place = self.place(base)?;
                match self.resolve(&place.ty) {
                    Ty::Struct(sname) => {
                        let found = self
                            .struct_fields
                            .get(&sname)
                            .and_then(|f| f.iter().find(|(n, _)| *n == name.node))
                            .map(|(_, t)| t.clone());
                        match found {
                            Some(t) => {
                                place.projs.push(hir::Proj::Field {
                                    name: name.node.clone(),
                                    struct_name: sname,
                                });
                                place.ty = t;
                                place.span = target.span.clone();
                                Some(place)
                            }
                            None => {
                                self.err(
                                    name.span.clone(),
                                    format!("no field `{}` on type `{sname}`", name.node),
                                );
                                None
                            }
                        }
                    }
                    other => {
                        self.err(
                            target.span.clone(),
                            format!("cannot assign to a field of `{other}`"),
                        );
                        None
                    }
                }
            }
            ast::ExprKind::Index { base, index } => {
                let mut place = self.place(base)?;
                let index = self.expr(index, None);
                let shown = self.resolve(&index.ty);
                if !(matches!(shown, Ty::Prim(p) if p.is_int())
                    || matches!(shown, Ty::IntVar(_) | Ty::Unknown | Ty::Never))
                {
                    self.err(
                        index.span.clone(),
                        format!("indices must be integers, found `{shown}`"),
                    );
                }
                match self.resolve(&place.ty) {
                    Ty::Vec(elem) => {
                        place.projs.push(hir::Proj::Index(Box::new(index)));
                        place.ty = *elem;
                        place.span = target.span.clone();
                        Some(place)
                    }
                    other => {
                        self.err(
                            target.span.clone(),
                            format!("cannot index into a value of type `{other}`"),
                        );
                        None
                    }
                }
            }
            _ => {
                self.err(
                    target.span.clone(),
                    "the left side of `=` must be a variable, a field or an index",
                );
                None
            }
        }
    }

    fn assign(&mut self, target: &ast::Expr, value: &ast::Expr, span: &Span) -> hir::Expr {
        let Some(place) = self.place(target) else {
            let _ = self.expr(value, None);
            return Self::mk(hir::ExprKind::Error, Ty::Unit, span);
        };
        let ty = place.ty.clone();
        let v = self.expr(value, Some(&ty));
        let v = self.coerce(v, &ty);
        Self::mk(
            hir::ExprKind::Assign {
                place,
                value: Box::new(v),
            },
            Ty::Unit,
            span,
        )
    }

    fn for_loop(
        &mut self,
        var: &ast::Ident,
        iter: &ast::ForIter,
        body: &ast::Block,
        span: &Span,
    ) -> hir::Expr {
        let is_int = |t: &Ty| {
            matches!(t, Ty::Prim(p) if p.is_int())
                || matches!(t, Ty::IntVar(_) | Ty::Unknown | Ty::Never)
        };
        let (iter, var_ty) = match iter {
            ast::ForIter::Range {
                start,
                end,
                inclusive,
            } => {
                let mut s = self.expr(start, None);
                let e = self.expr(end, Some(&s.ty));
                // An untyped literal start adopts the type of the end bound.
                if matches!(start.node, ast::ExprKind::Lit(_))
                    && s.ty != e.ty
                    && e.ty != Ty::Unknown
                {
                    s = self.expr(start, Some(&e.ty));
                }
                for b in [&s, &e] {
                    if !is_int(&self.resolve(&b.ty)) {
                        self.err(
                            b.span.clone(),
                            format!("range bounds must be integers, found `{}`", b.ty),
                        );
                    }
                }
                if !self.unify(&s.ty, &e.ty) {
                    self.err(
                        e.span.clone(),
                        format!(
                            "range bounds have different types: `{}` and `{}`",
                            s.ty, e.ty
                        ),
                    );
                }
                let ty = if s.ty == Ty::Unknown {
                    e.ty.clone()
                } else {
                    s.ty.clone()
                };
                (
                    hir::ForIter::Range {
                        start: Box::new(s),
                        end: Box::new(e),
                        inclusive: *inclusive,
                    },
                    ty,
                )
            }
            ast::ForIter::Expr(v) => {
                let v = self.expr(v, None);
                let ty = match &v.ty {
                    Ty::Vec(t) => (**t).clone(),
                    Ty::Table(s) => Ty::Struct(s.clone()),
                    Ty::Unknown | Ty::Never => Ty::Unknown,
                    other => {
                        let other = other.clone();
                        self.err(
                            v.span.clone(),
                            format!("cannot iterate over a value of type `{other}`"),
                        );
                        Ty::Unknown
                    }
                };
                (hir::ForIter::Vec(Box::new(v)), ty)
            }
        };
        self.scopes.push(HashMap::new());
        let var = self.declare(var, var_ty);
        self.loop_depth += 1;
        let (body, _) = self.block(body, None);
        self.loop_depth -= 1;
        self.scopes.pop();
        Self::mk(hir::ExprKind::For { var, iter, body }, Ty::Unit, span)
    }

    // ---- generics: functions, structs, enums and impls instantiated per use ------------------

    /// Whether `ty` mentions a type parameter (also through the arguments of a generic type).
    fn has_param(&self, ty: &Ty) -> bool {
        match ty {
            Ty::Param(_) => true,
            Ty::Vec(t) | Ty::Option(t) => self.has_param(t),
            Ty::Result(a, b) | Ty::Map(a, b) => self.has_param(a) || self.has_param(b),
            Ty::Tuple(items) => items.iter().any(|t| self.has_param(t)),
            Ty::Fn(ps, r) => ps.iter().any(|p| self.has_param(p)) || self.has_param(r),
            Ty::Tensor { elem, .. } => self.has_param(elem),
            Ty::Struct(k) | Ty::Enum(k) => self
                .instance_args
                .get(k)
                .is_some_and(|(_, args)| args.iter().any(|a| self.has_param(a))),
            _ => false,
        }
    }

    /// Replaces the type parameters in `ty` by what `bind` says (`Unknown` when unbound).
    fn subst_params(&mut self, ty: &Ty, bind: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::Param(p) => bind.get(p).cloned().unwrap_or(Ty::Unknown),
            Ty::Vec(t) => Ty::Vec(Box::new(self.subst_params(t, bind))),
            Ty::Option(t) => Ty::Option(Box::new(self.subst_params(t, bind))),
            Ty::Result(a, b) => Ty::Result(
                Box::new(self.subst_params(a, bind)),
                Box::new(self.subst_params(b, bind)),
            ),
            Ty::Map(a, b) => Ty::Map(
                Box::new(self.subst_params(a, bind)),
                Box::new(self.subst_params(b, bind)),
            ),
            Ty::Tuple(items) => {
                Ty::Tuple(items.iter().map(|t| self.subst_params(t, bind)).collect())
            }
            Ty::Fn(ps, r) => Ty::Fn(
                ps.iter().map(|p| self.subst_params(p, bind)).collect(),
                Box::new(self.subst_params(r, bind)),
            ),
            Ty::Tensor { elem, shape } => Ty::Tensor {
                elem: Box::new(self.subst_params(elem, bind)),
                shape: shape.clone(),
            },
            Ty::Struct(k) | Ty::Enum(k) if self.has_param(ty) => {
                let (base, args) = self.instance_args[k].clone();
                let args: Vec<Ty> = args.iter().map(|a| self.subst_params(a, bind)).collect();
                self.instantiate_adt(&base, args, &(0..0))
            }
            other => other.clone(),
        }
    }

    /// Binds the type parameters in `declared` by matching it against `actual`.
    fn infer_params(&self, declared: &Ty, actual: &Ty, bind: &mut HashMap<String, Ty>) {
        let actual = self.resolve(actual);
        match (declared, &actual) {
            (Ty::Param(p), a) => {
                if !matches!(a, Ty::Unknown | Ty::Never) {
                    bind.entry(p.clone()).or_insert_with(|| a.clone());
                }
            }
            (Ty::Vec(d), Ty::Vec(a)) | (Ty::Option(d), Ty::Option(a)) => {
                self.infer_params(d, a, bind);
            }
            (Ty::Result(d1, d2), Ty::Result(a1, a2)) | (Ty::Map(d1, d2), Ty::Map(a1, a2)) => {
                self.infer_params(d1, a1, bind);
                self.infer_params(d2, a2, bind);
            }
            (Ty::Tuple(ds), Ty::Tuple(a)) if ds.len() == a.len() => {
                for (d, a) in ds.iter().zip(a) {
                    self.infer_params(d, a, bind);
                }
            }
            (Ty::Fn(dp, dr), Ty::Fn(ap, ar)) if dp.len() == ap.len() => {
                for (d, a) in dp.iter().zip(ap) {
                    self.infer_params(d, a, bind);
                }
                self.infer_params(dr, ar, bind);
            }
            (Ty::Tensor { elem: d, .. }, Ty::Tensor { elem: a, .. }) => {
                self.infer_params(d, a, bind)
            }
            (Ty::Struct(dk) | Ty::Enum(dk), Ty::Struct(ak) | Ty::Enum(ak)) => {
                if let (Some((db, dargs)), Some((ab, aargs))) =
                    (self.instance_args.get(dk), self.instance_args.get(ak))
                    && db == ab
                    && dargs.len() == aargs.len()
                {
                    for (d, a) in dargs.iter().zip(aargs) {
                        self.infer_params(d, a, bind);
                    }
                }
            }
            _ => {}
        }
    }

    /// Reports a type argument that does not implement a trait its parameter is bounded by.
    fn check_bounds(&mut self, param: &ast::GenericParam, ty: &Ty, span: &Span) {
        for bound in &param.bounds {
            if !self.traits.contains_key(&bound.node) {
                self.err(
                    bound.span.clone(),
                    format!("cannot find the trait `{}`", bound.node),
                );
                continue;
            }
            let ok = match self.resolve(ty) {
                Ty::Struct(n) | Ty::Enum(n) => self.impls.contains(&(bound.node.clone(), n)),
                Ty::Trait(t) => t == bound.node,
                Ty::Unknown | Ty::Never | Ty::Param(_) => true,
                _ => false,
            };
            if !ok {
                self.err(
                    span.clone(),
                    format!(
                        "`{ty}` does not implement `{}`, which `{}` requires",
                        bound.node, param.name.node
                    ),
                );
            }
        }
    }

    /// `Stack<i64>`: the generic struct or enum `name` for these arguments, created the first time.
    /// With a type parameter among the arguments it only names the type (used to infer them).
    fn instantiate_adt(&mut self, name: &str, args: Vec<Ty>, span: &Span) -> Ty {
        let is_struct = self.generic_structs.contains_key(name);
        let generics = if is_struct {
            self.generic_structs[name].generics.clone()
        } else {
            self.generic_enums[name].generics.clone()
        };
        if args.len() != generics.len() {
            self.err(
                span.clone(),
                format!(
                    "`{name}` takes {} type argument(s), found {}",
                    generics.len(),
                    args.len()
                ),
            );
            return Ty::Unknown;
        }
        let args: Vec<Ty> = args.iter().map(|a| self.finalize_ty(a)).collect();
        let key = format!(
            "{name}<{}>",
            args.iter()
                .map(Ty::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.instance_args
            .entry(key.clone())
            .or_insert((name.to_string(), args.clone()));
        let ty = if is_struct {
            Ty::Struct(key.clone())
        } else {
            Ty::Enum(key.clone())
        };
        if args.iter().any(|a| self.has_param(a))
            || self.struct_fields.contains_key(&key)
            || self.enums.contains_key(&key)
        {
            return ty;
        }
        for (g, a) in generics.iter().zip(&args) {
            self.check_bounds(g, a, span);
        }
        let bind: HashMap<String, Ty> = generics
            .iter()
            .map(|g| g.name.node.clone())
            .zip(args.iter().cloned())
            .collect();
        let saved_subst = std::mem::replace(&mut self.type_params, bind);
        let saved_impl = self.impl_ty.take();
        if is_struct {
            self.struct_fields.insert(key.clone(), Vec::new());
            let decl = self.generic_structs[name].clone();
            let mut fields = Vec::new();
            for f in &decl.fields {
                let t = self.resolve_type(&f.ty);
                fields.push((f.name.node.clone(), t));
            }
            self.struct_fields.insert(key.clone(), fields.clone());
            self.anon_structs.push(hir::StructDef {
                name: key.clone(),
                name_span: decl.name.span.clone(),
                is_pub: decl.is_pub,
                fields,
                span: decl.span.clone(),
            });
        } else {
            self.enums.insert(key.clone(), Vec::new());
            let decl = self.generic_enums[name].clone();
            let mut variants = Vec::new();
            for v in &decl.variants {
                let tys: Vec<Ty> = v.fields.iter().map(|t| self.resolve_type(t)).collect();
                variants.push((v.name.node.clone(), tys));
            }
            self.enums.insert(key.clone(), variants.clone());
            self.anon_enums.push(hir::EnumDef {
                name: key.clone(),
                name_span: decl.name.span.clone(),
                is_pub: decl.is_pub,
                variants,
                span: decl.span.clone(),
            });
        }
        self.type_params = saved_subst;
        self.impl_ty = saved_impl;
        self.materialize_impls(name, &key, &args);
        ty
    }

    /// The methods of every `impl<T> Name<T>` for one instance: their signatures now, their bodies
    /// when the pending instances are checked.
    fn materialize_impls(&mut self, name: &str, key: &str, args: &[Ty]) {
        let impls = self.generic_impls.get(name).cloned().unwrap_or_default();
        for imp in impls {
            let mut bind: HashMap<String, Ty> = HashMap::new();
            for (written, actual) in imp.ty_args.iter().zip(args) {
                if written.path.len() == 1
                    && imp
                        .generics
                        .iter()
                        .any(|g| g.name.node == written.path[0].node)
                {
                    bind.insert(written.path[0].node.clone(), actual.clone());
                }
            }
            if let Some(tr) = &imp.trait_name
                && self.traits.contains_key(&tr.node)
            {
                self.impls.insert((tr.node.clone(), key.to_string()));
            }
            for f in &imp.fns {
                if !f.generics.is_empty() {
                    self.err(
                        f.name.span.clone(),
                        "methods with their own type parameters are not supported yet",
                    );
                    continue;
                }
                let full = format!("{key}::{}", f.name.node);
                self.methods.insert(full.clone(), key.to_string());
                if let Some(tr) = &imp.trait_name {
                    self.impl_of_fn.insert(full.clone(), tr.node.clone());
                }
                let saved_subst = std::mem::replace(&mut self.type_params, bind.clone());
                let saved_impl = self.impl_ty.replace(key.to_string());
                let params: Vec<Ty> = f.params.iter().map(|p| self.resolve_type(&p.ty)).collect();
                let ret = f.ret.as_ref().map_or(Ty::Unit, |t| self.resolve_type(t));
                self.type_params = saved_subst;
                self.impl_ty = saved_impl;
                self.fn_sigs.insert(full.clone(), (params, ret));
                let mut renamed = f.clone();
                renamed.name.node = full;
                self.pending.push(Pending {
                    f: renamed,
                    impl_ty: Some(key.to_string()),
                    subst: bind.clone(),
                });
            }
        }
    }

    /// A call of a generic function, or of an associated function of a generic type
    /// (`Stack::new()`): infer the type arguments, create the instance, call it.
    fn generic_call(
        &mut self,
        name: &str,
        type_name: Option<&str>,
        args: &[ast::Expr],
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        let bad = |this: &mut Self, msg: String| {
            this.err(span.clone(), msg);
            Self::mk(hir::ExprKind::Error, Ty::Unknown, span)
        };
        // The declaration and the type parameters in play.
        let (decl, generics, owner) = match type_name {
            Some(t) => {
                let found = self.generic_impls.get(t).and_then(|imps| {
                    imps.iter().find_map(|imp| {
                        imp.fns
                            .iter()
                            .find(|f| f.name.node == name)
                            .map(|f| (f.clone(), imp.clone()))
                    })
                });
                let Some((f, imp)) = found else {
                    return bad(self, format!("`{t}` has no associated function `{name}`"));
                };
                if !f.generics.is_empty() {
                    return bad(
                        self,
                        "methods with their own type parameters are not supported yet".into(),
                    );
                }
                (f, imp.generics.clone(), Some(imp))
            }
            None => {
                let f = self.generic_fns[name].clone();
                let g = f.generics.clone();
                (f, g, None)
            }
        };
        // The signature with type parameters left as `Ty::Param`.
        let param_types: HashMap<String, Ty> = generics
            .iter()
            .map(|g| (g.name.node.clone(), Ty::Param(g.name.node.clone())))
            .collect();
        let saved_subst = std::mem::replace(&mut self.type_params, param_types);
        let saved_impl = self.impl_ty.take();
        if let (Some(imp), Some(t)) = (&owner, type_name) {
            let pattern: Vec<Ty> = imp.ty_args.iter().map(|a| self.resolve_type(a)).collect();
            if let Ty::Struct(k) | Ty::Enum(k) = self.instantiate_adt(t, pattern, span) {
                self.impl_ty = Some(k);
            }
        }
        let decl_params: Vec<Ty> = decl
            .params
            .iter()
            .map(|p| self.resolve_type(&p.ty))
            .collect();
        let decl_ret = decl.ret.as_ref().map_or(Ty::Unit, |t| self.resolve_type(t));
        self.type_params = saved_subst;
        self.impl_ty = saved_impl;
        if decl_params.len() != args.len() {
            return bad(
                self,
                format!(
                    "expected {} argument(s), found {}",
                    decl_params.len(),
                    args.len()
                ),
            );
        }
        // Arguments first (closures wait until the types they need are known).
        let mut bind: HashMap<String, Ty> = HashMap::new();
        let mut checked: Vec<Option<hir::Expr>> = Vec::new();
        for (a, declared) in args.iter().zip(&decl_params) {
            if matches!(a.node, ast::ExprKind::Closure { .. }) {
                checked.push(None);
            } else {
                let e = self.expr(a, None);
                self.infer_params(declared, &e.ty, &mut bind);
                checked.push(Some(e));
            }
        }
        if let Some(want) = expected {
            let want = self.resolve(want);
            self.infer_params(&decl_ret, &want, &mut bind);
        }
        for (i, a) in args.iter().enumerate() {
            if checked[i].is_none() {
                let hint = self.subst_params(&decl_params[i], &bind);
                let e = self.expr(a, Some(&hint));
                self.infer_params(&decl_params[i], &e.ty, &mut bind);
                checked[i] = Some(e);
            }
        }
        for g in &generics {
            if !bind.contains_key(&g.name.node) {
                return bad(
                    self,
                    format!(
                        "cannot infer the type parameter `{}` of `{name}`: add a type annotation",
                        g.name.node
                    ),
                );
            }
        }
        let bind: HashMap<String, Ty> = bind
            .into_iter()
            .map(|(k, v)| {
                let v = self.finalize_ty(&v);
                (k, v)
            })
            .collect();
        for g in &generics {
            let ty = bind[&g.name.node].clone();
            self.check_bounds(g, &ty, span);
        }
        let params: Vec<Ty> = decl_params
            .iter()
            .map(|p| self.subst_params(p, &bind))
            .collect();
        let ret = self.subst_params(&decl_ret, &bind);
        // The instance to call.
        let full = match (&owner, type_name) {
            (Some(imp), Some(t)) => {
                let type_args: Vec<Ty> = imp
                    .ty_args
                    .iter()
                    .map(|a| {
                        if a.path.len() == 1 {
                            bind.get(&a.path[0].node).cloned().unwrap_or(Ty::Unknown)
                        } else {
                            Ty::Unknown
                        }
                    })
                    .collect();
                match self.instantiate_adt(t, type_args, span) {
                    Ty::Struct(k) | Ty::Enum(k) => format!("{k}::{name}"),
                    _ => return Self::mk(hir::ExprKind::Error, Ty::Unknown, span),
                }
            }
            _ => {
                let shown: Vec<String> = generics
                    .iter()
                    .map(|g| bind[&g.name.node].to_string())
                    .collect();
                let key = format!("{name}<{}>", shown.join(", "));
                if !self.fn_sigs.contains_key(&key) {
                    self.fn_sigs
                        .insert(key.clone(), (params.clone(), ret.clone()));
                    let mut renamed = decl.clone();
                    renamed.name.node = key.clone();
                    renamed.generics = Vec::new();
                    self.pending.push(Pending {
                        f: renamed,
                        impl_ty: None,
                        subst: bind.clone(),
                    });
                }
                key
            }
        };
        let args: Vec<hir::Expr> = checked
            .into_iter()
            .zip(&params)
            .map(|(e, p)| {
                let e = e.expect("every argument was checked");
                self.coerce(e, p)
            })
            .collect();
        let callee = Self::mk(
            hir::ExprKind::Fn(full),
            Ty::Fn(params, Box::new(ret.clone())),
            span,
        );
        Self::mk(
            hir::ExprKind::Call {
                callee: Box::new(callee),
                args,
            },
            ret,
            span,
        )
    }

    /// The full name (`Point::norm`) of the method `name` of a struct or enum type, if there is one.
    fn method_of(&self, ty: &Ty, name: &str) -> Option<String> {
        let type_name = match self.resolve(ty) {
            Ty::Struct(n) | Ty::Enum(n) => n,
            _ => return None,
        };
        let full = format!("{type_name}::{name}");
        if self.methods.contains_key(&full) {
            return Some(full);
        }
        // A default method of a trait the type implements.
        let mut candidates: Vec<&String> = self
            .impls
            .iter()
            .filter(|(_, t)| *t == type_name)
            .map(|(tr, _)| tr)
            .collect();
        candidates.sort();
        candidates.into_iter().find_map(|tr| {
            let declares = self.traits[tr]
                .iter()
                .any(|(m, required)| m == name && !required);
            declares.then(|| format!("{tr}::{name}"))
        })
    }

    /// For a value whose type is a trait: the trait, when it declares a method called `name`.
    fn trait_method(&self, ty: &Ty, name: &str) -> Option<String> {
        match self.resolve(ty) {
            Ty::Trait(t)
                if self
                    .traits
                    .get(&t)
                    .is_some_and(|ms| ms.iter().any(|(m, _)| m == name)) =>
            {
                Some(t)
            }
            _ => None,
        }
    }

    /// `receiver.method(args)` where the receiver's type is a trait: chosen when the program runs.
    fn dyn_call(
        &mut self,
        trait_name: &str,
        method: &str,
        receiver: hir::Expr,
        args: &[ast::Expr],
        span: &Span,
    ) -> hir::Expr {
        let full = format!("{trait_name}::{method}");
        let (params, ret) = self.fn_sigs[&full].clone();
        if params.len() != args.len() + 1 {
            self.err(
                span.clone(),
                format!(
                    "expected {} argument(s), found {}",
                    params.len().saturating_sub(1),
                    args.len()
                ),
            );
        }
        let mut checked = vec![receiver];
        for (i, a) in args.iter().enumerate() {
            let want = params.get(i + 1);
            let a = self.expr(a, want);
            checked.push(match want {
                Some(w) => self.coerce(a, w),
                None => a,
            });
        }
        Self::mk(
            hir::ExprKind::DynCall {
                trait_name: trait_name.to_string(),
                method: method.to_string(),
                args: checked,
            },
            ret,
            span,
        )
    }

    /// `ty` with the trait's `Self` replaced by the implementing type.
    fn with_self(ty: &Ty, trait_name: &str, concrete: &Ty) -> Ty {
        let go = |t: &Ty| Self::with_self(t, trait_name, concrete);
        match ty {
            Ty::Trait(n) if n == trait_name => concrete.clone(),
            Ty::Vec(t) => Ty::Vec(Box::new(go(t))),
            Ty::Option(t) => Ty::Option(Box::new(go(t))),
            Ty::Result(a, b) => Ty::Result(Box::new(go(a)), Box::new(go(b))),
            Ty::Map(k, v) => Ty::Map(Box::new(go(k)), Box::new(go(v))),
            Ty::Tuple(items) => Ty::Tuple(items.iter().map(go).collect()),
            Ty::Fn(p, r) => Ty::Fn(p.iter().map(go).collect(), Box::new(go(r))),
            other => other.clone(),
        }
    }

    /// Every `impl Trait for Type` must define the required methods with the trait's signatures
    /// and may only add methods that the trait declares.
    fn check_trait_impls(&mut self, fns: &[(ast::Fn, Option<String>)]) {
        let impls = self.impl_spans.clone();
        for (trait_name, ty, span) in impls {
            let declared = self.traits[&trait_name].clone();
            let concrete = if self.enums.contains_key(&ty) {
                Ty::Enum(ty.clone())
            } else {
                Ty::Struct(ty.clone())
            };
            for (method, required) in &declared {
                let (trait_key, impl_key) =
                    (format!("{trait_name}::{method}"), format!("{ty}::{method}"));
                match (
                    self.fn_sigs.get(&impl_key).cloned(),
                    self.fn_sigs.get(&trait_key).cloned(),
                ) {
                    (None, _) if *required => self.err(
                        span.clone(),
                        format!("`type {ty}({trait_name})` is missing the method `{method}`"),
                    ),
                    (Some((got_params, got_ret)), Some((want_params, want_ret))) => {
                        let want_params: Vec<Ty> = want_params
                            .iter()
                            .map(|p| Self::with_self(p, &trait_name, &concrete))
                            .collect();
                        let want_ret = Self::with_self(&want_ret, &trait_name, &concrete);
                        let same = got_params.len() == want_params.len()
                            && got_params
                                .iter()
                                .zip(&want_params)
                                .all(|(g, w)| g.accepts(w) && w.accepts(g))
                            && got_ret.accepts(&want_ret)
                            && want_ret.accepts(&got_ret);
                        if !same {
                            let shown = |ps: &[Ty], r: &Ty| {
                                let ps: Vec<String> = ps.iter().map(Ty::to_string).collect();
                                format!("({}) -> {r}", ps.join(", "))
                            };
                            let at = fns
                                .iter()
                                .find(|(f, _)| f.name.node == impl_key)
                                .map_or(span.clone(), |(f, _)| f.name.span.clone());
                            self.err(
                                at,
                                format!(
                                    "`{ty}.{method}` has the signature {} but `{trait_name}` declares {}",
                                    shown(&got_params, &got_ret),
                                    shown(&want_params, &want_ret)
                                ),
                            );
                        }
                    }
                    _ => {}
                }
            }
            for (f, impl_ty) in fns {
                if impl_ty.as_deref() == Some(ty.as_str())
                    && let Some(m) = f.name.node.strip_prefix(&format!("{ty}::"))
                    && !declared.iter().any(|(d, _)| d == m)
                    && self.impls.contains(&(trait_name.clone(), ty.clone()))
                    && self
                        .impl_of_fn
                        .get(&f.name.node)
                        .is_some_and(|t| *t == trait_name)
                {
                    self.err(
                        f.name.span.clone(),
                        format!("`{m}` is not a method of the trait `{trait_name}`"),
                    );
                }
            }
        }
    }

    /// Checks `receiver.method(args)` against the signature of `full` (receiver first).
    fn method_call(
        &mut self,
        full: &str,
        receiver: hir::Expr,
        args: &[ast::Expr],
        span: &Span,
    ) -> hir::Expr {
        let (params, ret) = self.fn_sigs[full].clone();
        let takes_self = params.first().is_some_and(|p| {
            matches!(
                (p, self.methods.get(full)),
                (Ty::Struct(n) | Ty::Enum(n) | Ty::Trait(n), Some(t)) if n == t
            )
        });
        let name = full.rsplit("::").next().unwrap_or(full).to_string();
        if !takes_self {
            self.err(
                span.clone(),
                format!("`{name}` has no `self` parameter, call it as `{full}(..)`"),
            );
            return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
        }
        if params.len() != args.len() + 1 {
            self.err(
                span.clone(),
                format!(
                    "expected {} argument(s), found {}",
                    params.len() - 1,
                    args.len()
                ),
            );
        }
        let mut checked = vec![receiver];
        for (i, a) in args.iter().enumerate() {
            let want = params.get(i + 1);
            let a = self.expr(a, want);
            checked.push(match want {
                Some(w) => self.coerce(a, w),
                None => a,
            });
        }
        let callee = Self::mk(
            hir::ExprKind::Fn(full.to_string()),
            Ty::Fn(params, Box::new(ret.clone())),
            span,
        );
        Self::mk(
            hir::ExprKind::Call {
                callee: Box::new(callee),
                args: checked,
            },
            ret,
            span,
        )
    }

    /// `Enum::Variant` or `Enum::Variant(args..)`.
    fn enum_ctor(
        &mut self,
        enum_name: &str,
        variant: &ast::Ident,
        args: &[ast::Expr],
        span: &Span,
    ) -> hir::Expr {
        let ty = Ty::Enum(enum_name.to_string());
        if !self.enums.contains_key(enum_name) {
            // The type arguments could not be inferred (already reported).
            for a in args {
                let _ = self.expr(a, None);
            }
            return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
        }
        let Some(fields) = self.enums[enum_name]
            .iter()
            .find(|(n, _)| *n == variant.node)
            .map(|(_, f)| f.clone())
        else {
            self.err(
                variant.span.clone(),
                format!("enum `{enum_name}` has no variant `{}`", variant.node),
            );
            return Self::mk(hir::ExprKind::Error, ty, span);
        };
        if fields.len() != args.len() {
            self.err(
                span.clone(),
                format!(
                    "`{enum_name}.{}` takes {} value(s), found {}",
                    variant.node,
                    fields.len(),
                    args.len()
                ),
            );
        }
        let args = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let want = fields.get(i);
                let a = self.expr(a, want);
                match want {
                    Some(w) => self.coerce(a, w),
                    None => a,
                }
            })
            .collect();
        Self::mk(
            hir::ExprKind::EnumCtor {
                enum_name: enum_name.to_string(),
                variant: variant.node.clone(),
                args,
            },
            ty,
            span,
        )
    }

    fn match_expr(
        &mut self,
        scrutinee: &ast::Expr,
        arms: &[ast::Arm],
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        let scrut = self.expr(scrutinee, None);
        let sty = self.resolve(&scrut.ty);
        if arms.is_empty() {
            self.err(span.clone(), "a `match` needs at least one arm");
        }
        let expected = expected.cloned();
        let mut ty = Ty::Never;
        let mut out = Vec::new();
        for arm in arms {
            self.scopes.push(HashMap::new());
            let pat = self.pattern(&arm.pat, &sty);
            let want = expected
                .clone()
                .or_else(|| (ty != Ty::Never).then(|| ty.clone()));
            let body = self.expr(&arm.body, want.as_ref());
            self.scopes.pop();
            if self.unify(&ty, &body.ty) {
                if ty == Ty::Never {
                    ty = body.ty.clone();
                }
            } else {
                self.err(
                    body.span.clone(),
                    format!(
                        "`match` arms have incompatible types: `{}` and `{}`",
                        self.resolve(&ty),
                        self.resolve(&body.ty)
                    ),
                );
                ty = Ty::Unknown;
            }
            out.push(hir::Arm { pat, body });
        }
        self.check_exhaustive(&sty, &out, span);
        Self::mk(
            hir::ExprKind::Match {
                scrutinee: Box::new(scrut),
                arms: out,
            },
            ty,
            span,
        )
    }

    fn pattern(&mut self, p: &ast::Pattern, ty: &Ty) -> hir::Pat {
        use ast::PatKind as P;
        let ty = self.resolve(ty);
        match &p.node {
            P::Wild => hir::Pat::Wild,
            P::Tuple(items) => {
                let parts: Vec<Ty> = match &ty {
                    Ty::Tuple(ts) if ts.len() == items.len() => ts.clone(),
                    Ty::Unknown => vec![Ty::Unknown; items.len()],
                    other => {
                        self.err(
                            p.span.clone(),
                            format!(
                                "a pattern with {} parts cannot match `{other}`",
                                items.len()
                            ),
                        );
                        vec![Ty::Unknown; items.len()]
                    }
                };
                hir::Pat::Tuple(
                    items
                        .iter()
                        .zip(&parts)
                        .map(|(item, t)| self.pattern(item, t))
                        .collect(),
                )
            }
            P::Bind(name) if name == "None" => {
                if !matches!(ty, Ty::Option(_) | Ty::Unknown) {
                    self.err(
                        p.span.clone(),
                        format!("`None` patterns need an `Option`, found `{ty}`"),
                    );
                }
                hir::Pat::Variant {
                    enum_name: Some("Option".into()),
                    variant: "None".into(),
                    args: vec![],
                }
            }
            P::Bind(name) => {
                let id = ast::Ident {
                    node: name.clone(),
                    span: p.span.clone(),
                };
                hir::Pat::Bind(self.declare(&id, ty))
            }
            P::Lit(l) => {
                let lit_ty = match l {
                    ast::Lit::Int(_) => self.fresh(false),
                    ast::Lit::Float(_) => self.fresh(true),
                    ast::Lit::Str(_) => Ty::str(),
                    ast::Lit::Bool(_) => Ty::bool(),
                };
                if !self.unify(&ty, &lit_ty) {
                    self.err(
                        p.span.clone(),
                        format!("this pattern does not match a value of type `{ty}`"),
                    );
                }
                hir::Pat::Lit(l.clone())
            }
            P::Variant { path, args } => self.variant_pattern(p, path, args, &ty),
        }
    }

    fn variant_pattern(
        &mut self,
        p: &ast::Pattern,
        path: &[ast::Ident],
        args: &[ast::Pattern],
        ty: &Ty,
    ) -> hir::Pat {
        let bad = |this: &mut Self, msg: String| {
            this.err(p.span.clone(), msg);
            hir::Pat::Wild
        };
        let (enum_name, variant, field_tys) = match path {
            [one] if one.node == "Some" => {
                let payload = match ty {
                    Ty::Option(t) => Some((**t).clone()),
                    Ty::Unknown => Some(Ty::Unknown),
                    _ => None,
                };
                let Some(payload) = payload else {
                    return bad(
                        self,
                        format!("`Some` patterns need an `Option`, found `{ty}`"),
                    );
                };
                (
                    Some("Option".to_string()),
                    "Some".to_string(),
                    vec![payload],
                )
            }
            [one] if matches!(one.node.as_str(), "Ok" | "Err") => {
                let payload = match ty {
                    Ty::Result(o, e) => Some(if one.node == "Ok" {
                        (**o).clone()
                    } else {
                        (**e).clone()
                    }),
                    Ty::Unknown => Some(Ty::Unknown),
                    _ => None,
                };
                let Some(payload) = payload else {
                    return bad(
                        self,
                        format!("`{}` patterns need a `Result`, found `{ty}`", one.node),
                    );
                };
                (None, one.node.clone(), vec![payload])
            }
            [en, v] if self.generic_enums.contains_key(&en.node) => {
                let key = match &ty {
                    Ty::Enum(k)
                        if self
                            .instance_args
                            .get(k)
                            .is_some_and(|(b, _)| *b == en.node) =>
                    {
                        k.clone()
                    }
                    _ => {
                        return bad(
                            self,
                            format!(
                                "pattern of type `{}` cannot match a value of type `{ty}`",
                                en.node
                            ),
                        );
                    }
                };
                let Some(fields) = self.enums[&key]
                    .iter()
                    .find(|(n, _)| *n == v.node)
                    .map(|(_, f)| f.clone())
                else {
                    return bad(
                        self,
                        format!("enum `{}` has no variant `{}`", en.node, v.node),
                    );
                };
                (Some(key), v.node.clone(), fields)
            }
            [en, v] if self.enums.contains_key(&en.node) => {
                let enum_ty = Ty::Enum(en.node.clone());
                if !ty.accepts(&enum_ty) {
                    return bad(
                        self,
                        format!(
                            "pattern of type `{}` cannot match a value of type `{ty}`",
                            en.node
                        ),
                    );
                }
                let Some(fields) = self.enums[&en.node]
                    .iter()
                    .find(|(n, _)| *n == v.node)
                    .map(|(_, f)| f.clone())
                else {
                    return bad(
                        self,
                        format!("enum `{}` has no variant `{}`", en.node, v.node),
                    );
                };
                (Some(en.node.clone()), v.node.clone(), fields)
            }
            _ => {
                let name = path
                    .iter()
                    .map(|i| i.node.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                return bad(self, format!("`{name}` is not a variant of a known enum"));
            }
        };
        if field_tys.len() != args.len() {
            self.err(
                p.span.clone(),
                format!(
                    "`{variant}` has {} value(s), the pattern has {}",
                    field_tys.len(),
                    args.len()
                ),
            );
        }
        let args = args
            .iter()
            .enumerate()
            .map(|(i, a)| self.pattern(a, field_tys.get(i).unwrap_or(&Ty::Unknown)))
            .collect();
        hir::Pat::Variant {
            enum_name,
            variant,
            args,
        }
    }

    /// Reports a `match` that can fall through.
    fn check_exhaustive(&mut self, sty: &Ty, arms: &[hir::Arm], span: &Span) {
        let rows: Vec<Vec<Option<&hir::Pat>>> = arms.iter().map(|a| vec![Some(&a.pat)]).collect();
        if self.exhaustive(&rows, std::slice::from_ref(sty)) {
            return;
        }
        let missing = match self.constructors_of(sty) {
            Some(ctors) => {
                let names: Vec<String> = ctors
                    .iter()
                    .filter(|(en, v, fields)| {
                        let sub = self.specialize(&rows, en.as_deref(), v, fields.len());
                        !self.exhaustive(&sub, fields)
                    })
                    .map(|(en, v, _)| match en.as_deref() {
                        Some("Option") | None => format!("`{v}`"),
                        Some(name) => format!("`{name}.{v}`"),
                    })
                    .collect();
                format!(": missing {}", names.join(", "))
            }
            None => ": add a `_` arm".to_string(),
        };
        self.err(span.clone(), format!("non-exhaustive `match`{missing}"));
    }

    /// The constructors a value of `ty` can have, when `match` has to cover each of them.
    fn constructors_of(&self, ty: &Ty) -> Option<Vec<Constructor>> {
        match self.resolve(ty) {
            Ty::Enum(name) => self.enums.get(&name).map(|vs| {
                vs.iter()
                    .map(|(v, f)| (Some(name.clone()), v.clone(), f.clone()))
                    .collect()
            }),
            Ty::Result(o, e) => Some(vec![
                (None, "Ok".to_string(), vec![*o]),
                (None, "Err".to_string(), vec![*e]),
            ]),
            Ty::Tuple(parts) => Some(vec![(
                Some("(tuple)".to_string()),
                "(tuple)".to_string(),
                parts,
            )]),
            Ty::Option(t) => Some(vec![
                (Some("Option".to_string()), "Some".to_string(), vec![*t]),
                (Some("Option".to_string()), "None".to_string(), vec![]),
            ]),
            _ => None,
        }
    }

    /// Rows whose first column can match `variant`, with that column replaced by its fields.
    fn specialize<'p>(
        &self,
        rows: &[Vec<Option<&'p hir::Pat>>],
        enum_name: Option<&str>,
        variant: &str,
        arity: usize,
    ) -> Vec<Vec<Option<&'p hir::Pat>>> {
        let mut out = Vec::new();
        for row in rows {
            let rest = &row[1..];
            match row[0] {
                None | Some(hir::Pat::Wild | hir::Pat::Bind(_)) => {
                    let mut r = vec![None; arity];
                    r.extend_from_slice(rest);
                    out.push(r);
                }
                Some(hir::Pat::Tuple(args)) if variant == "(tuple)" => {
                    let mut r: Vec<Option<&hir::Pat>> = args.iter().map(Some).collect();
                    r.resize(arity, None);
                    r.extend_from_slice(rest);
                    out.push(r);
                }
                Some(hir::Pat::Variant {
                    enum_name: en,
                    variant: v,
                    args,
                }) if v == variant && en.as_deref() == enum_name => {
                    let mut r: Vec<Option<&hir::Pat>> = args.iter().map(Some).collect();
                    r.resize(arity, None);
                    r.extend_from_slice(rest);
                    out.push(r);
                }
                _ => {}
            }
        }
        out
    }

    /// The classic usefulness check: does every value of `tys` match some row?
    fn exhaustive(&self, rows: &[Vec<Option<&hir::Pat>>], tys: &[Ty]) -> bool {
        let Some((first, rest_tys)) = tys.split_first() else {
            return !rows.is_empty();
        };
        let constructors = self.constructors_of(first);
        if let Some(ctors) = constructors {
            return ctors.iter().all(|(en, v, fields)| {
                let sub = self.specialize(rows, en.as_deref(), v, fields.len());
                let mut next = fields.clone();
                next.extend_from_slice(rest_tys);
                self.exhaustive(&sub, &next)
            });
        }
        if matches!(self.resolve(first), Ty::Prim(Prim::Bool)) {
            return [true, false].iter().all(|b| {
                let sub: Vec<Vec<Option<&hir::Pat>>> = rows
                    .iter()
                    .filter(|r| match r[0] {
                        None | Some(hir::Pat::Wild | hir::Pat::Bind(_)) => true,
                        Some(hir::Pat::Lit(ast::Lit::Bool(x))) => x == b,
                        _ => false,
                    })
                    .map(|r| r[1..].to_vec())
                    .collect();
                self.exhaustive(&sub, rest_tys)
            });
        }
        // Numbers, strings and unknown types: only a catch-all covers every value.
        let sub: Vec<Vec<Option<&hir::Pat>>> = rows
            .iter()
            .filter(|r| matches!(r[0], None | Some(hir::Pat::Wild | hir::Pat::Bind(_))))
            .map(|r| r[1..].to_vec())
            .collect();
        self.exhaustive(&sub, rest_tys)
    }

    fn path(&mut self, p: &[ast::Ident], expected: Option<&Ty>, span: &Span) -> hir::Expr {
        let mut names: Vec<String> = p.iter().map(|i| i.node.clone()).collect();
        // `Self::new` inside an `impl` is `Type::new`.
        if names.len() == 2
            && names[0] == "Self"
            && let Some(t) = &self.impl_ty
        {
            names[0] = t.clone();
        }
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
            if self.generic_fns.contains_key(first) {
                self.err(
                    span.clone(),
                    format!("the generic function `{first}` has to be called, it cannot be used as a value"),
                );
                return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
            }
            if self.struct_fields.contains_key(first) {
                self.err(
                    span.clone(),
                    format!("`{first}` is a type, not a value; construct it with `{first}(field: value)`"),
                );
                return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
            }
        }
        if matches!(first, "Ok" | "Err") && names.len() == 1 {
            return Self::mk(hir::ExprKind::Extern(names), Ty::Unknown, span);
        }
        if names.len() == 2 {
            let is_variant = self
                .enums
                .get(first)
                .is_some_and(|vs| vs.iter().any(|(v, _)| *v == names[1]));
            let full = format!("{first}::{}", names[1]);
            if !is_variant && let Some((params, ret)) = self.fn_sigs.get(&full) {
                let ty = Ty::Fn(params.clone(), Box::new(ret.clone()));
                return Self::mk(hir::ExprKind::Fn(full), ty, span);
            }
            if self.enums.contains_key(first) {
                return self.enum_ctor(first, &p[1], &[], span);
            }
            if self.generic_enums.contains_key(first) {
                let key = self.generic_enum_instance(first, &p[1], &[], expected, span);
                return self.enum_ctor(&key, &p[1], &[], span);
            }
        }
        let resolved = if first == "std" {
            Some(names.clone())
        } else if names.len() == 1
            && let Some(full) = prelude(first)
        {
            Some(full)
        } else {
            self.imports.get(first).map(|base| {
                let mut full = base.clone();
                full.extend(names[1..].iter().cloned());
                full
            })
        };
        if let Some(full) = resolved {
            if full[0] == "std" {
                let joined = full.join("::");
                if stdlib::function(&joined).is_none() {
                    let what = if stdlib::is_type(&joined) || stdlib::is_module(&joined) {
                        format!("`{}` is not a value", names.join("."))
                    } else {
                        format!(
                            "cannot find `{}` in the standard library",
                            joined.replace("::", ".")
                        )
                    };
                    self.err(span.clone(), what);
                    return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
                }
            }
            return Self::mk(hir::ExprKind::Extern(full), Ty::Unknown, span);
        }
        let hint = if names.len() == 1 {
            let mut known: Vec<&str> = self
                .scopes
                .iter()
                .flat_map(|s| s.keys().map(String::as_str))
                .collect();
            known.extend(self.globals.keys().map(String::as_str));
            known.extend(self.fn_sigs.keys().map(String::as_str));
            // Spellings from other languages for the constants Telvax writes with a capital.
            let constant = match first {
                "true" => Some("True"),
                "false" => Some("False"),
                "none" | "null" | "nil" | "nothing" => Some("None"),
                _ => None,
            };
            constant
                .map(str::to_string)
                .or_else(|| closest(first, known).map(str::to_string))
        } else {
            None
        };
        self.err_hint(
            span.clone(),
            format!("cannot find `{}` in this scope", names.join(".")),
            hint.as_deref(),
        );
        Self::mk(hir::ExprKind::Error, Ty::Unknown, span)
    }

    /// Reports a method that the runtime does not have on a built-in type.
    fn no_method(&mut self, name: &ast::Ident, ty: &str, methods: &[&str]) -> Ty {
        let hint = closest(&name.node, methods.iter().copied()).map(str::to_string);
        self.err_hint(
            name.span.clone(),
            format!("no method `{}` on type `{ty}`", name.node),
            hint.as_deref(),
        );
        Ty::Unknown
    }

    fn field_ty(&mut self, base: &hir::Expr, name: &ast::Ident) -> Ty {
        let n = name.node.as_str();
        // Methods declared by a trait are called, not read.
        if self.trait_method(&base.ty, n).is_some() {
            return Ty::Unknown;
        }
        if let Ty::Trait(t) = self.resolve(&base.ty) {
            self.err(
                name.span.clone(),
                format!("the trait `{t}` has no method `{n}`"),
            );
            return Ty::Unknown;
        }
        // Methods of the user's own types are called, not read.
        if self.method_of(&base.ty, n).is_some() {
            return Ty::Unknown;
        }
        // Methods that `call` turns into std functions are not looked up here.
        if method_module(&self.resolve(&base.ty), n).is_some() {
            return Ty::Unknown;
        }
        // `pair.0`, `pair.1`: the parts of a tuple.
        if let Ty::Tuple(parts) = self.resolve(&base.ty) {
            return match n.parse::<usize>().ok().and_then(|i| parts.get(i)) {
                Some(t) => t.clone(),
                None => {
                    let ty = base.ty.clone();
                    self.err(
                        name.span.clone(),
                        format!("`{ty}` has no part `{n}` (parts are numbered from 0)"),
                    );
                    Ty::Unknown
                }
            };
        }
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
                        let hint = self.struct_fields.get(s).and_then(|f| {
                            closest(n, f.iter().map(|(name, _)| name.as_str())).map(str::to_string)
                        });
                        self.err_hint(
                            name.span.clone(),
                            format!("no field `{n}` on type `{ty}`"),
                            hint.as_deref(),
                        );
                        Ty::Unknown
                    }
                }
            }
            Ty::Prim(Prim::Str) => match n {
                "is_empty" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "len" => Ty::Fn(vec![], Box::new(Ty::Prim(Prim::U64))),
                "trim" | "to_lowercase" | "to_uppercase" => Ty::Fn(vec![], Box::new(Ty::str())),
                "contains" | "starts_with" | "ends_with" => {
                    Ty::Fn(vec![Ty::str()], Box::new(Ty::bool()))
                }
                "replace" => Ty::Fn(vec![Ty::str(), Ty::str()], Box::new(Ty::str())),
                "split" => Ty::Fn(vec![Ty::str()], Box::new(Ty::Vec(Box::new(Ty::str())))),
                "find" => Ty::Fn(
                    vec![Ty::str()],
                    Box::new(Ty::Option(Box::new(Ty::Prim(Prim::U64)))),
                ),
                "to_int" => Ty::Fn(vec![], Box::new(Ty::Option(Box::new(Ty::Prim(Prim::I64))))),
                "to_float" => Ty::Fn(vec![], Box::new(Ty::Option(Box::new(Ty::Prim(Prim::F64))))),
                "chars" | "lines" => Ty::Fn(vec![], Box::new(Ty::Vec(Box::new(Ty::str())))),
                _ => self.no_method(
                    name,
                    "Text",
                    &[
                        "is_empty",
                        "len",
                        "trim",
                        "to_lowercase",
                        "to_uppercase",
                        "contains",
                        "starts_with",
                        "ends_with",
                        "replace",
                        "split",
                        "find",
                        "to_int",
                        "to_float",
                        "chars",
                        "lines",
                    ],
                ),
            },
            Ty::Vec(elem) => match n {
                "is_empty" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "len" => Ty::Fn(vec![], Box::new(Ty::Prim(Prim::U64))),
                "push" => Ty::Fn(vec![(**elem).clone()], Box::new(Ty::Unit)),
                "contains" => Ty::Fn(vec![(**elem).clone()], Box::new(Ty::bool())),
                "first" | "last" | "pop" => Ty::Fn(vec![], Box::new(Ty::Option(elem.clone()))),
                "get" => {
                    let index = self.fresh(false);
                    Ty::Fn(vec![index], Box::new(Ty::Option(elem.clone())))
                }
                "sort" | "reverse" => Ty::Fn(vec![], Box::new(Ty::Unit)),
                _ => {
                    let shown = base.ty.to_string();
                    self.no_method(
                        name,
                        &shown,
                        &[
                            "is_empty", "len", "push", "contains", "first", "last", "pop", "get",
                            "sort", "reverse",
                        ],
                    )
                }
            },
            Ty::Option(inner) => match n {
                "is_some" | "is_none" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "unwrap" => Ty::Fn(vec![], inner.clone()),
                "unwrap_or" => Ty::Fn(vec![(**inner).clone()], inner.clone()),
                _ => {
                    let shown = base.ty.to_string();
                    self.no_method(name, &shown, &["is_some", "is_none", "unwrap", "unwrap_or"])
                }
            },
            Ty::Map(k, v) => match n {
                "insert" => Ty::Fn(vec![(**k).clone(), (**v).clone()], Box::new(Ty::Unit)),
                "get" | "remove" => Ty::Fn(vec![(**k).clone()], Box::new(Ty::Option(v.clone()))),
                "contains_key" => Ty::Fn(vec![(**k).clone()], Box::new(Ty::bool())),
                "len" => Ty::Fn(vec![], Box::new(Ty::Prim(Prim::U64))),
                "is_empty" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "keys" => Ty::Fn(vec![], Box::new(Ty::Vec(k.clone()))),
                "entries" => Ty::Fn(
                    vec![],
                    Box::new(Ty::Vec(Box::new(Ty::Tuple(vec![
                        (**k).clone(),
                        (**v).clone(),
                    ])))),
                ),
                "values" => Ty::Fn(vec![], Box::new(Ty::Vec(v.clone()))),
                _ => {
                    let shown = base.ty.to_string();
                    self.no_method(
                        name,
                        &shown,
                        &[
                            "insert",
                            "get",
                            "remove",
                            "contains_key",
                            "len",
                            "is_empty",
                            "keys",
                            "values",
                            "entries",
                        ],
                    )
                }
            },
            Ty::Table(s) => match n {
                // A method call: `call` turns it into `std::table::sql(table, ..)`.
                "sql" => Ty::Unknown,
                "is_empty" => Ty::Fn(vec![], Box::new(Ty::bool())),
                "len" => Ty::Fn(vec![], Box::new(Ty::Prim(Prim::U64))),
                _ => {
                    let column = self
                        .struct_fields
                        .get(s)
                        .and_then(|f| f.iter().find(|(fname, _)| fname == n))
                        .map(|(_, t)| t.clone());
                    match column {
                        // Numeric columns are tensors that share the table's Arrow buffer.
                        Some(Ty::Prim(p)) if p.is_numeric() => Ty::Tensor {
                            elem: Box::new(Ty::Prim(p)),
                            shape: vec![Dim::Named("N".into())],
                        },
                        Some(Ty::Prim(p)) => Ty::Vec(Box::new(Ty::Prim(p))),
                        Some(other) => other,
                        None => {
                            let ty = base.ty.clone();
                            self.err(name.span.clone(), format!("no column `{n}` on type `{ty}`"));
                            Ty::Unknown
                        }
                    }
                }
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
        if let ast::ExprKind::Path(p) = &callee.node {
            if p.len() == 1
                && self.lookup_local(&p[0].node).is_none()
                && !self.fn_sigs.contains_key(&p[0].node)
                && self.generic_fns.contains_key(&p[0].node)
            {
                return self.generic_call(&p[0].node.clone(), None, args, expected, span);
            }
            if p.len() == 2
                && (self.generic_structs.contains_key(&p[0].node)
                    || self.generic_enums.contains_key(&p[0].node))
                && self.generic_impls.get(&p[0].node).is_some_and(|imps| {
                    imps.iter()
                        .any(|i| i.fns.iter().any(|f| f.name.node == p[1].node))
                })
            {
                return self.generic_call(
                    &p[1].node.clone(),
                    Some(&p[0].node.clone()),
                    args,
                    expected,
                    span,
                );
            }
        }
        if let ast::ExprKind::Path(p) = &callee.node
            && p.len() == 2
        {
            let first = if p[0].node == "Self" {
                self.impl_ty.clone().unwrap_or_default()
            } else {
                p[0].node.clone()
            };
            let is_variant = self
                .enums
                .get(&first)
                .is_some_and(|vs| vs.iter().any(|(v, _)| *v == p[1].node));
            if is_variant {
                return self.enum_ctor(&first, &p[1], args, span);
            }
            if self.generic_enums.contains_key(&first)
                && self.generic_enums[&first]
                    .variants
                    .iter()
                    .any(|v| v.name.node == p[1].node)
            {
                let key = self.generic_enum_instance(&first, &p[1], args, expected, span);
                return self.enum_ctor(&key, &p[1], args, span);
            }
        }
        if let ast::ExprKind::Path(p) = &callee.node
            && self.is_option_ctor(p, "Some")
        {
            if args.len() != 1 {
                self.err(span.clone(), "`Some` takes exactly one argument");
            }
            if let Some(arg) = args.first() {
                return self.some_ctor(arg, expected, span);
            }
        }
        // `Ok(x)` / `Err(x)` build a `Result`, typed from context when available.
        if let ast::ExprKind::Path(p) = &callee.node
            && p.len() == 1
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

        let callee = self.expr(callee, None);
        // A method called on a trait-typed value is dispatched on the actual type at run time.
        if let hir::ExprKind::Field { base, name } = &callee.kind
            && let Some(trait_name) = self.trait_method(&base.ty, name)
        {
            let receiver = (**base).clone();
            return self.dyn_call(&trait_name, name, receiver, args, span);
        }
        // `p.norm()` where `norm` is a method of `p`'s type: `Point::norm(p)`.
        if let hir::ExprKind::Field { base, name } = &callee.kind
            && let Some(full) = self.method_of(&base.ty, name)
        {
            let receiver = (**base).clone();
            return self.method_call(&full, receiver, args, span);
        }
        // `t.relu()`, `x.matmul(w)`, `sales.sql("..")`: a method on a tensor or table is the std
        // function of the same name with the receiver as its first argument.
        if let hir::ExprKind::Field { base, name } = &callee.kind
            && let Some(module) = method_module(&self.resolve(&base.ty), name)
        {
            let path = ["std", module, name.as_str()].map(String::from).to_vec();
            if let Some(sig) = stdlib::function(&path.join("::")) {
                let receiver = (**base).clone();
                let func = Self::mk(hir::ExprKind::Extern(path), Ty::Unknown, &callee.span);
                return self.std_call_with(func, sig, Some(receiver), args, expected, span);
            }
        }
        if let Some(sig) = self.std_sig(&callee) {
            return self.std_call(callee, sig, args, expected, span);
        }
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

    /// Typing rule for `callee` when it names a std function or a method of a std type.
    fn std_sig(&mut self, callee: &hir::Expr) -> Option<Sig> {
        match &callee.kind {
            hir::ExprKind::Extern(path) if path.first().is_some_and(|p| p == "std") => {
                stdlib::function(&path.join("::"))
            }
            hir::ExprKind::Field { base, name } => {
                let Ty::Named(ty) = &base.ty else { return None };
                match stdlib::method(ty, name) {
                    Some(sig) => Some(sig),
                    None => {
                        if stdlib::is_std_type(ty) {
                            self.err(
                                callee.span.clone(),
                                format!("no method `{name}` on type `{ty}`"),
                            );
                        }
                        None
                    }
                }
            }
            _ => None,
        }
    }

    fn std_call(
        &mut self,
        callee: hir::Expr,
        sig: Sig,
        args: &[ast::Expr],
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        self.std_call_with(callee, sig, None, args, expected, span)
    }

    /// Like `std_call`; `receiver` is an already checked first argument (`x.f(a)` is `f(x, a)`).
    fn std_call_with(
        &mut self,
        callee: hir::Expr,
        sig: Sig,
        receiver: Option<hir::Expr>,
        args: &[ast::Expr],
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        let mut checked: Vec<hir::Expr> = Vec::with_capacity(args.len() + 1);
        let mut tys: Vec<Ty> = Vec::with_capacity(args.len() + 1);
        if let Some(r) = receiver {
            tys.push(r.ty.clone());
            checked.push(r);
        }
        for a in args {
            let hint = (sig.hint)(checked.len(), &tys);
            let a = self.expr(a, hint.as_ref());
            tys.push(a.ty.clone());
            checked.push(a);
        }
        // Distributed calls ship the function by name, so only top-level functions qualify.
        if let hir::ExprKind::Extern(path) = &callee.kind
            && path.join("::") == "std::dist::map"
            && let Some(f) = checked.get(1)
            && !matches!(f.kind, hir::ExprKind::Fn(_))
        {
            self.diags.push(
                Diagnostic::error(
                    f.span.clone(),
                    "`std.dist.map` needs the name of a top-level function",
                )
                .with_note("closures capture local variables and cannot be sent to other machines"),
            );
        }
        let tys: Vec<Ty> = tys.iter().map(|t| self.resolve(t)).collect();
        let mut ty = match (sig.check)(&tys) {
            Ok(t) => t,
            Err(msg) => {
                self.err(span.clone(), msg);
                Ty::Unknown
            }
        };
        if let Some(from_expected) = sig.from_expected {
            let want = expected.map(|e| self.resolve(e));
            ty = match want {
                // The context knows nothing about the type; the runtime checks the real shape.
                Some(Ty::Unknown | Ty::Never) => Ty::Unknown,
                other => match from_expected(&tys, &other.unwrap_or(Ty::Unknown)) {
                    Ok(t) => t,
                    Err(msg) => {
                        self.err(span.clone(), msg);
                        Ty::Unknown
                    }
                },
            };
        }
        Self::mk(
            hir::ExprKind::Call {
                callee: Box::new(callee),
                args: checked,
            },
            ty,
            span,
        )
    }

    /// Whether `p` is the bare name `ctor` (`Some` / `None`) and nothing in scope shadows it.
    fn is_option_ctor(&self, p: &[ast::Ident], ctor: &str) -> bool {
        p.len() == 1
            && p[0].node == ctor
            && self.lookup_local(ctor).is_none()
            && !self.fn_sigs.contains_key(ctor)
            && !self.globals.contains_key(ctor)
    }

    /// `Some(value)`, typed from the expected `Option<T>` when there is one.
    fn some_ctor(&mut self, arg: &ast::Expr, expected: Option<&Ty>, span: &Span) -> hir::Expr {
        let want = match expected {
            Some(Ty::Option(t)) => (**t).clone(),
            _ => Ty::Unknown,
        };
        let arg = self.expr(arg, Some(&want));
        let arg = self.coerce(arg, &want);
        let ty = match expected {
            Some(e @ Ty::Option(_)) if e.accepts(&Ty::Option(Box::new(arg.ty.clone()))) => {
                e.clone()
            }
            _ => Ty::Option(Box::new(arg.ty.clone())),
        };
        Self::mk(
            hir::ExprKind::EnumCtor {
                enum_name: "Option".into(),
                variant: "Some".into(),
                args: vec![arg],
            },
            ty,
            span,
        )
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

    /// `a + b` and friends when at least one side is a tensor.
    fn tensor_operator(
        &mut self,
        op: ast::BinOp,
        l: hir::Expr,
        r: hir::Expr,
        span: &Span,
    ) -> hir::Expr {
        use ast::BinOp::{Add, Div, Mul, Sub};
        let is_tensor = |t: &Ty| matches!(t, Ty::Tensor { .. });
        let (lt, rt) = (self.resolve(&l.ty), self.resolve(&r.ty));
        let is_number = |t: &Ty| {
            matches!(t, Ty::Prim(p) if p.is_numeric())
                || matches!(t, Ty::IntVar(_) | Ty::FloatVar(_))
        };
        let (name, args) = match (op, is_tensor(&lt), is_tensor(&rt)) {
            (Add, true, true) => ("add", vec![l, r]),
            (Sub, true, true) => ("sub", vec![l, r]),
            (Mul, true, true) => ("mul", vec![l, r]),
            (Div, true, true) => ("div", vec![l, r]),
            (Mul, true, false) if is_number(&rt) => ("scale", vec![l, r]),
            (Mul, false, true) if is_number(&lt) => ("scale", vec![r, l]),
            _ => {
                let hint = match op {
                    Div => {
                        "to divide a tensor by a number, multiply by its reciprocal: `t * (1.0 / k)`"
                    }
                    _ => {
                        "operators on tensors need two tensors of the same shape, or `*` with a number"
                    }
                };
                self.diags.push(
                    Diagnostic::error(
                        span.clone(),
                        format!("cannot apply this operator to `{lt}` and `{rt}`"),
                    )
                    .with_note(hint),
                );
                return Self::mk(hir::ExprKind::Error, Ty::Unknown, span);
            }
        };
        let path: Vec<String> = ["std", "tensor", name].map(String::from).to_vec();
        let sig = stdlib::function(&path.join("::")).expect("tensor function exists");
        let tys: Vec<Ty> = args.iter().map(|a| self.resolve(&a.ty)).collect();
        let ty = match (sig.check)(&tys) {
            Ok(t) => t,
            Err(msg) => {
                self.err(span.clone(), msg);
                Ty::Unknown
            }
        };
        let callee = Self::mk(hir::ExprKind::Extern(path), Ty::Unknown, span);
        Self::mk(
            hir::ExprKind::Call {
                callee: Box::new(callee),
                args,
            },
            ty,
            span,
        )
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
        // Operators on tensors are the elementwise std functions: `a + b`, `a - b`, `a * b`,
        // `a / b` for tensors of the same shape, and `a * k` / `k * a` to scale by a number.
        if matches!(op, Add | Sub | Mul | Div) {
            let (lt, rt) = (self.resolve(&l.ty), self.resolve(&r.ty));
            let is_tensor = |t: &Ty| matches!(t, Ty::Tensor { .. });
            if is_tensor(&lt) || is_tensor(&rt) {
                return self.tensor_operator(op, l, r, span);
            }
        }
        let compatible = self.unify(&l.ty, &r.ty);
        if !compatible {
            let (lt, rt) = (self.resolve(&l.ty), self.resolve(&r.ty));
            self.diags.push(
                Diagnostic::error(
                    span.clone(),
                    format!("mismatched operand types: `{lt}` and `{rt}`"),
                )
                .with_label(l.span.clone(), format!("this is `{lt}`"))
                .with_label(r.span.clone(), format!("this is `{rt}`")),
            );
        }
        let is_num = |t: &Ty| {
            matches!(t, Ty::Prim(p) if p.is_numeric())
                || matches!(t, Ty::IntVar(_) | Ty::FloatVar(_) | Ty::Unknown)
        };
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
                            format!("expected `Bool`, found `{}`", side.ty),
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

    /// Which instance of the generic struct `base` a literal builds: the expected type, the
    /// enclosing `impl`, or what its field values say.
    fn generic_struct_instance(
        &mut self,
        base: &str,
        fields: &[(ast::Ident, ast::Expr)],
        expected: Option<&Ty>,
        span: &Span,
    ) -> String {
        let of_base =
            |this: &Self, k: &str| this.instance_args.get(k).is_some_and(|(b, _)| b == base);
        if let Some(Ty::Struct(k)) = expected.map(|t| self.resolve(t))
            && of_base(self, &k)
        {
            return k;
        }
        if let Some(k) = self.impl_ty.clone()
            && of_base(self, &k)
        {
            return k;
        }
        let decl = self.generic_structs[base].clone();
        let param_types: HashMap<String, Ty> = decl
            .generics
            .iter()
            .map(|g| (g.name.node.clone(), Ty::Param(g.name.node.clone())))
            .collect();
        let saved = std::mem::replace(&mut self.type_params, param_types);
        let declared: Vec<(String, Ty)> = decl
            .fields
            .iter()
            .map(|f| (f.name.node.clone(), self.resolve_type(&f.ty)))
            .collect();
        self.type_params = saved;
        let mut bind = HashMap::new();
        let keep = self.diags.len();
        for (fname, value) in fields {
            if let Some((_, d)) = declared.iter().find(|(n, _)| *n == fname.node) {
                let e = self.expr(value, None);
                self.infer_params(d, &e.ty, &mut bind);
            }
        }
        self.diags.truncate(keep);
        let mut args = Vec::new();
        for g in &decl.generics {
            match bind.get(&g.name.node) {
                Some(t) => args.push(t.clone()),
                None => {
                    self.err(
                        span.clone(),
                        format!(
                            "cannot infer the type parameter `{}` of `{base}`: add a type annotation",
                            g.name.node
                        ),
                    );
                    return base.to_string();
                }
            }
        }
        match self.instantiate_adt(base, args, span) {
            Ty::Struct(k) => k,
            _ => base.to_string(),
        }
    }

    /// Which instance of the generic enum `base` a variant builds: the expected type, the
    /// enclosing `impl`, or what the variant's values say.
    fn generic_enum_instance(
        &mut self,
        base: &str,
        variant: &ast::Ident,
        args: &[ast::Expr],
        expected: Option<&Ty>,
        span: &Span,
    ) -> String {
        let of_base =
            |this: &Self, k: &str| this.instance_args.get(k).is_some_and(|(b, _)| b == base);
        if let Some(Ty::Enum(k)) = expected.map(|t| self.resolve(t))
            && of_base(self, &k)
        {
            return k;
        }
        if let Some(k) = self.impl_ty.clone()
            && of_base(self, &k)
        {
            return k;
        }
        let decl = self.generic_enums[base].clone();
        let param_types: HashMap<String, Ty> = decl
            .generics
            .iter()
            .map(|g| (g.name.node.clone(), Ty::Param(g.name.node.clone())))
            .collect();
        let saved = std::mem::replace(&mut self.type_params, param_types);
        let declared: Vec<Ty> = decl
            .variants
            .iter()
            .find(|v| v.name.node == variant.node)
            .map(|v| v.fields.iter().map(|t| self.resolve_type(t)).collect())
            .unwrap_or_default();
        self.type_params = saved;
        let mut bind = HashMap::new();
        let keep = self.diags.len();
        for (a, d) in args.iter().zip(&declared) {
            let e = self.expr(a, None);
            self.infer_params(d, &e.ty, &mut bind);
        }
        self.diags.truncate(keep);
        let mut targs = Vec::new();
        for g in &decl.generics {
            match bind.get(&g.name.node) {
                Some(t) => targs.push(t.clone()),
                None => {
                    self.err(
                        span.clone(),
                        format!(
                            "cannot infer the type parameter `{}` of `{base}`: add a type annotation",
                            g.name.node
                        ),
                    );
                    return base.to_string();
                }
            }
        }
        match self.instantiate_adt(base, targs, span) {
            Ty::Enum(k) => k,
            _ => base.to_string(),
        }
    }

    fn struct_lit(
        &mut self,
        path: &[ast::Ident],
        fields: &[(ast::Ident, ast::Expr)],
        expected: Option<&Ty>,
        span: &Span,
    ) -> hir::Expr {
        let mut name = path
            .iter()
            .map(|i| i.node.as_str())
            .collect::<Vec<_>>()
            .join("::");
        if name == "Self"
            && let Some(t) = &self.impl_ty
        {
            name = t.clone();
        }
        if self.generic_structs.contains_key(&name) {
            name = self.generic_struct_instance(&name, fields, expected, span);
        }
        let Some(decl) = self.struct_fields.get(&name).cloned() else {
            if !self.generic_structs.contains_key(&name) {
                self.err(span.clone(), format!("cannot find type `{name}`"));
            }
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
                    format!("type `{name}` has no field `{}`", fname.node),
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
