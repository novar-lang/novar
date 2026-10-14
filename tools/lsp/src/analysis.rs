//! Whole-file analysis behind every language feature: parse, type check, and index the typed
//! HIR by source position.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use novar_diagnostics::Diagnostic;
use novar_hir::hir::{self, ExprKind, ForIter, LocalId, Proj};
use novar_hir::stdlib::{self, ItemKind};
use novar_hir::{Prim, Ty};

use crate::position::LineIndex;

pub type Span = Range<usize>;

#[derive(Debug, Clone)]
enum Target {
    Local(LocalId),
    Fn(String),
    Global(String),
    Struct(String),
    Other,
}

#[derive(Debug, Clone)]
struct Node {
    span: Span,
    ty: Ty,
    target: Target,
}

#[derive(Debug, Clone)]
pub struct LocalInfo {
    pub name: String,
    pub ty: Ty,
    pub mutable: bool,
    pub span: Span,
}

/// Everything known about one version of a document.
#[derive(Debug)]
pub struct Snapshot {
    pub text: String,
    pub index: LineIndex,
    pub diagnostics: Vec<Diagnostic>,
    pub hir: Option<hir::Program>,
    /// All files of the program (this document is the first); resolves names across modules.
    pub map: novar_modules::SourceMap,
    nodes: Vec<Node>,
    locals: HashMap<LocalId, LocalInfo>,
    /// The program came from an earlier version of the text because the current one could not
    /// be analysed (typically mid-edit). Types are still useful for completion, positions are not.
    pub stale: bool,
}

/// A name and every span where it appears in the document (see [`Snapshot::occurrences`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occurrences {
    pub name: String,
    pub spans: Vec<Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Global,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub span: Span,
    pub name_span: Span,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    Keyword,
    Function,
    Struct,
    Variable,
    Field,
    Method,
    Module,
    Type,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
}

const KEYWORDS: &[&str] = &[
    "fn", "let", "mut", "struct", "pub", "if", "else", "for", "in", "while", "return", "break",
    "continue", "import", "true", "false",
];
const PRIMS: &[&str] = &[
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64", "bool", "String",
];

pub fn signature(f: &hir::FnDef) -> String {
    let params: Vec<String> = f
        .params
        .iter()
        .map(|p| {
            format!(
                "{}{}: {}",
                if p.mutable { "mut " } else { "" },
                p.name,
                p.ty
            )
        })
        .collect();
    let ret = if f.ret == Ty::Unit {
        String::new()
    } else {
        format!(" -> {}", f.ret)
    };
    format!(
        "{}fn {}({}){ret}",
        if f.is_pub { "pub " } else { "" },
        f.name,
        params.join(", ")
    )
}

impl Snapshot {
    /// Like [`Snapshot::analyze`], but when `text` cannot be analysed at all the types of `previous`
    /// are kept for completion.
    pub fn analyze_after(
        text: String,
        path: Option<&Path>,
        previous: Option<&Snapshot>,
    ) -> Snapshot {
        let mut snap = Snapshot::analyze_in(text, path);
        if snap.hir.is_none()
            && let Some(prev) = previous.filter(|p| p.hir.is_some())
        {
            snap.hir = prev.hir.clone();
            snap.nodes = prev.nodes.clone();
            snap.locals = prev.locals.clone();
            snap.map = prev.map.clone();
            snap.stale = true;
        }
        snap
    }

    /// Analyses a standalone buffer (no other modules).
    pub fn analyze(text: String) -> Snapshot {
        Snapshot::analyze_in(text, None)
    }

    /// Analyses the document at `path`, loading the modules it imports and the dependencies of its
    /// project. Only diagnostics located in this document are kept.
    pub fn analyze_in(text: String, path: Option<&Path>) -> Snapshot {
        let index = LineIndex::new(&text);
        #[cfg(feature = "project")]
        let deps = path
            .and_then(|p| novar_pkg::Project::discover(p).ok().flatten())
            .and_then(|project| project.dependency_packages().ok())
            .unwrap_or_default();
        #[cfg(not(feature = "project"))]
        let deps = Vec::new();
        let name = path.map_or("<buffer>".to_string(), |p| p.display().to_string());
        let loaded = novar_modules::load_text(&name, &text, path.and_then(Path::parent), &deps);
        let mut diagnostics = loaded.diagnostics.clone();
        let mut hir = None;
        if loaded.entry_parsed {
            let (h, d) = novar_hir::lower(&loaded.program);
            diagnostics.extend(d);
            hir = Some(h);
        }
        // Problems inside other modules are reported when those files are opened.
        diagnostics.retain(|d| d.span.start <= text.len());
        diagnostics.sort_by_key(|d| d.span.start);
        let mut snap = Snapshot {
            text,
            index,
            diagnostics,
            hir: None,
            map: loaded.map,
            nodes: Vec::new(),
            locals: HashMap::new(),
            stale: false,
        };
        if let Some(h) = &hir {
            let mut c = Collector::default();
            c.program(h);
            snap.nodes = c.nodes;
            snap.locals = c.locals;
        }
        snap.hir = hir;
        snap
    }

    /// Where `offset` is defined, possibly in another file: `(file name, byte range in that file)`.
    pub fn definition_location(&self, offset: usize) -> Option<(String, Span)> {
        let span = self.definition(offset)?;
        let (file, local) = self.map.locate(span.start)?;
        Some((file.name.clone(), local..local + span.len()))
    }

    /// The smallest indexed expression containing `offset`.
    fn node_at(&self, offset: usize) -> Option<&Node> {
        self.nodes
            .iter()
            .filter(|n| n.span.start <= offset && offset <= n.span.end && n.span.start < n.span.end)
            .min_by_key(|n| n.span.end - n.span.start)
    }

    fn program(&self) -> Option<&hir::Program> {
        self.hir.as_ref()
    }

    pub fn hover(&self, offset: usize) -> Option<(String, Span)> {
        self.hover_raw(offset)
            .map(|(text, span)| (self.map.demangle(&text), span))
    }

    fn hover_raw(&self, offset: usize) -> Option<(String, Span)> {
        if self.stale {
            return None;
        }
        let prog = self.program()?;
        // Definition names first: they are not expressions.
        for f in &prog.fns {
            if f.name_span.contains(&offset) {
                return Some((
                    format!("```novar\n{}\n```", signature(f)),
                    f.name_span.clone(),
                ));
            }
        }
        for s in &prog.structs {
            if s.name_span.contains(&offset) {
                let fields: Vec<String> = s
                    .fields
                    .iter()
                    .map(|(n, t)| format!("    {n}: {t},"))
                    .collect();
                return Some((
                    format!(
                        "```novar\nstruct {} {{\n{}\n}}\n```",
                        s.name,
                        fields.join("\n")
                    ),
                    s.name_span.clone(),
                ));
            }
        }
        if let Some(l) = self.locals.values().find(|l| l.span.contains(&offset)) {
            return Some((
                format!(
                    "```novar\nlet {}{}: {}\n```",
                    if l.mutable { "mut " } else { "" },
                    l.name,
                    l.ty
                ),
                l.span.clone(),
            ));
        }
        let node = self.node_at(offset)?;
        let text = match &node.target {
            Target::Local(id) => {
                let l = self.locals.get(id)?;
                format!(
                    "```novar\nlet {}{}: {}\n```",
                    if l.mutable { "mut " } else { "" },
                    l.name,
                    node.ty
                )
            }
            Target::Fn(name) => {
                let f = prog.fns.iter().find(|f| &f.name == name)?;
                format!("```novar\n{}\n```", signature(f))
            }
            Target::Global(name) => format!("```novar\nlet {name}: {}\n```", node.ty),
            Target::Struct(name) => format!("```novar\nstruct {name}\n```"),
            Target::Other => format!("```novar\n{}\n```", node.ty),
        };
        Some((text, node.span.clone()))
    }

    pub fn definition(&self, offset: usize) -> Option<Span> {
        if self.stale {
            return None;
        }
        let prog = self.program()?;
        let node = self.node_at(offset)?;
        match &node.target {
            Target::Local(id) => self.locals.get(id).map(|l| l.span.clone()),
            Target::Fn(name) => prog
                .fns
                .iter()
                .find(|f| &f.name == name)
                .map(|f| f.name_span.clone()),
            Target::Struct(name) => prog
                .structs
                .iter()
                .find(|s| &s.name == name)
                .map(|s| s.name_span.clone()),
            Target::Global(name) => prog
                .globals
                .iter()
                .find(|g| &g.name == name)
                .map(|g| g.span.clone()),
            Target::Other => None,
        }
    }

    /// The name under `offset` and every place it appears in this document (its definition
    /// included), for a local variable or a function. `Err` says why a symbol cannot be handled.
    pub fn occurrences(&self, offset: usize) -> Result<Occurrences, String> {
        if self.stale {
            return Err("the document has errors that prevent analysing it".into());
        }
        let prog = self.program().ok_or("the document could not be analysed")?;
        let target = self
            .locals
            .iter()
            .find(|(_, l)| l.span.start <= offset && offset <= l.span.end)
            .map(|(id, _)| Target::Local(*id))
            .or_else(|| {
                prog.fns
                    .iter()
                    .find(|f| f.name_span.start <= offset && offset <= f.name_span.end)
                    .map(|f| Target::Fn(f.name.clone()))
            })
            .or_else(|| {
                prog.structs
                    .iter()
                    .find(|s| s.name_span.start <= offset && offset <= s.name_span.end)
                    .map(|s| Target::Struct(s.name.clone()))
            })
            .or_else(|| self.node_at(offset).map(|n| n.target.clone()))
            .ok_or("there is no symbol here")?;
        let (name, decl) = match &target {
            Target::Local(id) => {
                let l = self.locals.get(id).ok_or("unknown variable")?;
                (l.name.clone(), l.span.clone())
            }
            Target::Fn(name) => {
                let f = prog
                    .fns
                    .iter()
                    .find(|f| &f.name == name)
                    .ok_or("unknown function")?;
                (f.name.clone(), f.name_span.clone())
            }
            Target::Global(_) | Target::Struct(_) => {
                return Err(
                    "only variables and functions can be renamed or searched for now".into(),
                );
            }
            Target::Other => return Err("there is no symbol here".into()),
        };
        let in_document =
            |s: &Span| s.end <= self.text.len() && self.text.get(s.clone()) == Some(name.as_str());
        if !in_document(&decl) {
            return Err(format!("`{name}` is defined in another file"));
        }
        let same = |t: &Target| match (t, &target) {
            (Target::Local(a), Target::Local(b)) => a == b,
            (Target::Fn(a), Target::Fn(b)) => a == b,
            _ => false,
        };
        let mut spans: Vec<Span> = std::iter::once(decl)
            .chain(
                self.nodes
                    .iter()
                    .filter(|n| same(&n.target))
                    .map(|n| n.span.clone()),
            )
            .filter(in_document)
            .collect();
        spans.sort_by_key(|s| s.start);
        spans.dedup();
        Ok(Occurrences { name, spans })
    }

    /// The edits that rename the symbol under `offset` to `new_name`. The result is checked: the
    /// renamed program must analyse with no more errors than before, which catches a new name
    /// that collides with another one in scope.
    pub fn rename(&self, offset: usize, new_name: &str) -> Result<Vec<(Span, String)>, String> {
        let valid = new_name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && new_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid || KEYWORDS.contains(&new_name) || PRIMS.contains(&new_name) {
            return Err(format!("`{new_name}` is not a valid name"));
        }
        let found = self.occurrences(offset)?;
        if found.name == "main" {
            return Err("`main` is the entry point and cannot be renamed".into());
        }
        let mut renamed = self.text.clone();
        for span in found.spans.iter().rev() {
            renamed.replace_range(span.clone(), new_name);
        }
        let errors = |s: &Snapshot| s.diagnostics.iter().filter(|d| d.is_error()).count();
        let after = Snapshot::analyze(renamed);
        if errors(&after) > errors(self) {
            return Err(format!("`{new_name}` collides with another name in scope"));
        }
        Ok(found
            .spans
            .into_iter()
            .map(|s| (s, new_name.to_string()))
            .collect())
    }

    pub fn symbols(&self) -> Vec<Symbol> {
        let Some(prog) = self.program().filter(|_| !self.stale) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for s in &prog.structs {
            out.push(Symbol {
                name: s.name.clone(),
                kind: SymbolKind::Struct,
                span: s.span.clone(),
                name_span: s.name_span.clone(),
                detail: format!("{} field(s)", s.fields.len()),
            });
        }
        for f in &prog.fns {
            out.push(Symbol {
                name: f.name.clone(),
                kind: SymbolKind::Function,
                span: f.span.clone(),
                name_span: f.name_span.clone(),
                detail: signature(f),
            });
        }
        for g in &prog.globals {
            out.push(Symbol {
                name: g.name.clone(),
                kind: SymbolKind::Global,
                span: g.span.clone(),
                name_span: g.span.clone(),
                detail: g.ty.to_string(),
            });
        }
        out.retain(|s| s.span.end <= self.text.len());
        out.sort_by_key(|s| s.span.start);
        out
    }

    /// Inferred types for `let` bindings and `for` variables written without an annotation.
    pub fn inlay_hints(&self) -> Vec<(usize, String)> {
        let mut hints = Vec::new();
        if self.stale {
            return hints;
        }
        for l in self.locals.values() {
            if l.ty == Ty::Unknown || l.span.is_empty() {
                continue;
            }
            let after = self.text[l.span.end.min(self.text.len())..].trim_start();
            // Annotated bindings and function parameters (`name: T`) already show their type.
            if after.starts_with(':') {
                continue;
            }
            let before = self.text[..l.span.start].trim_end();
            if before.ends_with("let") || before.ends_with("mut") || before.ends_with("for") {
                hints.push((l.span.end, format!(": {}", l.ty)));
            }
        }
        hints.sort();
        hints
    }

    // ---- completion -----------------------------------------------------------------------

    pub fn completions(&self, offset: usize) -> Vec<Completion> {
        let offset = offset.min(self.text.len());
        let before = &self.text[..offset];
        let word_start = before
            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map_or(0, |i| {
                i + before[i..].chars().next().map_or(1, char::len_utf8)
            });
        let prefix = &before[word_start..];
        let head = &before[..word_start];

        let mut items = if let Some(receiver) = head.strip_suffix('.') {
            self.member_completions(receiver, offset)
        } else if let Some(path) = head.strip_suffix("::") {
            self.path_completions(path)
        } else if head.ends_with('@') {
            vec![item(
                "endpoint",
                CompletionKind::Function,
                Some("HTTP endpoint attribute".into()),
            )]
        } else {
            self.general_completions(offset)
        };
        items.retain(|c| c.label.starts_with(prefix));
        items.sort_by(|a, b| a.label.cmp(&b.label));
        items.dedup_by(|a, b| a.label == b.label && a.kind == b.kind);
        items
    }

    fn general_completions(&self, offset: usize) -> Vec<Completion> {
        let mut out: Vec<Completion> = KEYWORDS
            .iter()
            .map(|k| item(k, CompletionKind::Keyword, None))
            .collect();
        out.extend(PRIMS.iter().map(|p| item(p, CompletionKind::Type, None)));
        for t in ["Vec", "Tensor", "Table", "Result"] {
            out.push(item(t, CompletionKind::Type, None));
        }
        out.push(item(
            "std",
            CompletionKind::Module,
            Some("standard library".into()),
        ));
        if let Some(prog) = self.program() {
            let own = |span: &Span| span.end <= self.text.len();
            for s in prog.structs.iter().filter(|s| own(&s.name_span)) {
                out.push(item(&s.name, CompletionKind::Struct, None));
            }
            for f in prog.fns.iter().filter(|f| own(&f.name_span)) {
                out.push(item(&f.name, CompletionKind::Function, Some(signature(f))));
            }
            for g in prog.globals.iter().filter(|g| own(&g.span)) {
                out.push(item(
                    &g.name,
                    CompletionKind::Variable,
                    Some(g.ty.to_string()),
                ));
            }
            for imp in &prog.imports {
                if let Some(last) = imp.last() {
                    out.push(item(last, CompletionKind::Module, Some(imp.join("::"))));
                }
            }
            // Items imported from other modules, under the names the user wrote.
            for (mangled, shown) in &self.map.display_names {
                let Some(name) = shown.rsplit("::").next() else {
                    continue;
                };
                if let Some(f) = prog.fns.iter().find(|f| &f.name == mangled) {
                    out.push(item(
                        name,
                        CompletionKind::Function,
                        Some(self.map.demangle(&signature(f))),
                    ));
                } else if prog.structs.iter().any(|s| &s.name == mangled) {
                    out.push(item(name, CompletionKind::Struct, Some(shown.clone())));
                }
            }
        }
        out.extend(
            self.visible_locals(offset)
                .into_iter()
                .map(|l| item(&l.name, CompletionKind::Variable, Some(l.ty.to_string()))),
        );
        out
    }

    fn path_completions(&self, path_text: &str) -> Vec<Completion> {
        // Take the trailing `a::b::c` run of identifiers.
        let start = path_text
            .rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
            .map_or(0, |i| {
                i + path_text[i..].chars().next().map_or(1, char::len_utf8)
            });
        let mut segments: Vec<String> = path_text[start..]
            .split("::")
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if segments.is_empty() {
            return Vec::new();
        }
        // An imported name stands for its full path.
        if let Some(prog) = self.program()
            && let Some(full) = prog.imports.iter().find(|p| p.last() == segments.first())
        {
            let mut resolved = full.clone();
            resolved.extend(segments.drain(1..));
            segments = resolved;
        }
        if segments[0] != "std" {
            return Vec::new();
        }
        stdlib::children(&segments.join("::"))
            .into_iter()
            .map(|(name, kind)| {
                let kind = match kind {
                    ItemKind::Module => CompletionKind::Module,
                    ItemKind::Function => CompletionKind::Function,
                    ItemKind::Type => CompletionKind::Type,
                };
                item(&name, kind, None)
            })
            .collect()
    }

    fn member_completions(&self, receiver: &str, offset: usize) -> Vec<Completion> {
        let Some(ty) = self.type_of_chain(receiver, offset) else {
            return Vec::new();
        };
        let methods = |names: &[&str]| -> Vec<Completion> {
            names
                .iter()
                .map(|n| item(n, CompletionKind::Method, None))
                .collect()
        };
        match &ty {
            Ty::Struct(name) => self
                .program()
                .and_then(|p| p.structs.iter().find(|s| &s.name == name))
                .map(|s| {
                    s.fields
                        .iter()
                        .map(|(n, t)| item(n, CompletionKind::Field, Some(t.to_string())))
                        .collect()
                })
                .unwrap_or_default(),
            Ty::Table(name) => {
                let mut out = methods(&["len", "is_empty"]);
                if let Some(s) = self
                    .program()
                    .and_then(|p| p.structs.iter().find(|s| &s.name == name))
                {
                    out.extend(s.fields.iter().map(|(n, t)| {
                        item(n, CompletionKind::Field, Some(format!("column: {t}")))
                    }));
                }
                out
            }
            Ty::Prim(Prim::Str) => {
                methods(&["is_empty", "len", "trim", "to_lowercase", "to_uppercase"])
            }
            Ty::Vec(_) => methods(&["is_empty", "len"]),
            Ty::Named(t) => stdlib::methods_of(t)
                .into_iter()
                .map(|n| item(n, CompletionKind::Method, None))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Locals declared in the function around `offset`, before it.
    fn visible_locals(&self, offset: usize) -> Vec<&LocalInfo> {
        let Some(prog) = self.program() else {
            return Vec::new();
        };
        let Some(f) = prog
            .fns
            .iter()
            .find(|f| f.span.start <= offset && offset <= f.span.end)
        else {
            return Vec::new();
        };
        self.locals
            .values()
            .filter(|l| {
                l.span.start >= f.span.start && l.span.start < offset && l.span.end <= f.span.end
            })
            .collect()
    }

    /// Resolves `a.b.c` / `f(x).field` from the text before a dot, using the last analysis.
    fn type_of_chain(&self, text: &str, offset: usize) -> Option<Ty> {
        let prog = self.program()?;
        let text = text.trim_end();
        // Split into segments at top-level dots, remembering calls.
        let mut segments: Vec<(String, bool)> = Vec::new();
        let mut depth = 0i32;
        let mut current = String::new();
        for c in text.chars().rev() {
            match c {
                ')' | ']' => {
                    depth += 1;
                    current.insert(0, c);
                }
                '(' | '[' => {
                    depth -= 1;
                    current.insert(0, c);
                }
                '.' if depth == 0 => {
                    segments.insert(0, (std::mem::take(&mut current), false));
                }
                c if depth > 0 || c.is_alphanumeric() || c == '_' => current.insert(0, c),
                _ => break,
            }
        }
        segments.insert(0, (current, false));
        segments.retain(|(s, _)| !s.is_empty());
        let (first, rest) = segments.split_first()?;
        let name = first.0.split(['(', '[']).next()?.to_string();
        let is_call = first.0.contains('(');
        let mut ty = if let Some(l) = self
            .visible_locals(offset)
            .into_iter()
            .filter(|l| l.name == name)
            .max_by_key(|l| l.span.start)
        {
            l.ty.clone()
        } else if let Some(g) = prog.globals.iter().find(|g| g.name == name) {
            g.ty.clone()
        } else {
            // A bare function name is not a value we can complete on; a call yields its result.
            let f = prog.fns.iter().find(|f| f.name == name)?;
            is_call.then(|| f.ret.clone())?
        };
        if first.0.contains('[') {
            ty = element_of(&ty)?;
        }
        for (seg, _) in rest {
            let field = seg.split(['(', '[']).next()?;
            ty = match &ty {
                Ty::Struct(sname) => prog
                    .structs
                    .iter()
                    .find(|s| &s.name == sname)?
                    .fields
                    .iter()
                    .find(|(n, _)| n == field)?
                    .1
                    .clone(),
                Ty::Table(sname) => {
                    let col = prog
                        .structs
                        .iter()
                        .find(|s| &s.name == sname)?
                        .fields
                        .iter()
                        .find(|(n, _)| n == field)?
                        .1
                        .clone();
                    match col {
                        Ty::Prim(p) if p.is_numeric() => Ty::Tensor {
                            elem: Box::new(Ty::Prim(p)),
                            shape: vec![novar_hir::Dim::Named("N".into())],
                        },
                        Ty::Prim(p) => Ty::Vec(Box::new(Ty::Prim(p))),
                        other => other,
                    }
                }
                _ => return None,
            };
            if seg.contains('[') {
                ty = element_of(&ty)?;
            }
        }
        Some(ty)
    }
}

fn element_of(ty: &Ty) -> Option<Ty> {
    match ty {
        Ty::Vec(t) => Some((**t).clone()),
        Ty::Table(s) => Some(Ty::Struct(s.clone())),
        Ty::Tensor { elem, shape } if shape.len() == 1 => Some((**elem).clone()),
        Ty::Tensor { elem, shape } => Some(Ty::Tensor {
            elem: elem.clone(),
            shape: shape[1..].to_vec(),
        }),
        _ => None,
    }
}

fn item(label: &str, kind: CompletionKind, detail: Option<String>) -> Completion {
    Completion {
        label: label.to_string(),
        kind,
        detail,
    }
}

// ---- HIR walk ------------------------------------------------------------------------------

#[derive(Default)]
struct Collector {
    nodes: Vec<Node>,
    locals: HashMap<LocalId, LocalInfo>,
}

impl Collector {
    fn local(&mut self, l: &hir::Local) {
        self.locals.insert(
            l.id,
            LocalInfo {
                name: l.name.clone(),
                ty: l.ty.clone(),
                mutable: l.mutable,
                span: l.span.clone(),
            },
        );
    }

    fn program(&mut self, p: &hir::Program) {
        for g in &p.globals {
            self.expr(&g.init);
        }
        for f in &p.fns {
            for param in &f.params {
                self.local(param);
            }
            self.block(&f.body);
        }
    }

    fn block(&mut self, b: &hir::Block) {
        for s in &b.stmts {
            match s {
                hir::Stmt::Let { local, init } => {
                    self.local(local);
                    self.expr(init);
                }
                hir::Stmt::Expr(e) => self.expr(e),
            }
        }
        if let Some(t) = &b.tail {
            self.expr(t);
        }
    }

    fn expr(&mut self, e: &hir::Expr) {
        let target = match &e.kind {
            ExprKind::Local(id) => Target::Local(*id),
            ExprKind::Fn(n) => Target::Fn(n.clone()),
            ExprKind::Global(n) => Target::Global(n.clone()),
            ExprKind::StructLit { name, .. } => Target::Struct(name.clone()),
            _ => Target::Other,
        };
        self.nodes.push(Node {
            span: e.span.clone(),
            ty: e.ty.clone(),
            target,
        });
        match &e.kind {
            ExprKind::Lit(_)
            | ExprKind::Local(_)
            | ExprKind::Global(_)
            | ExprKind::Fn(_)
            | ExprKind::Extern(_)
            | ExprKind::Error => {}
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
                params.iter().for_each(|p| self.local(p));
                self.expr(body);
            }
            ExprKind::StructLit { fields, .. } => fields.iter().for_each(|(_, v)| self.expr(v)),
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
            ExprKind::Break | ExprKind::Continue => {}
            ExprKind::FString(parts) => {
                for p in parts {
                    if let hir::FPart::Expr(x) = p {
                        self.expr(x);
                    }
                }
            }
            ExprKind::Assign { place, value } => {
                self.nodes.push(Node {
                    span: place.span.clone(),
                    ty: place.ty.clone(),
                    target: Target::Local(place.local),
                });
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
                self.local(var);
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
            ExprKind::EnumCtor { args, .. }
            | ExprKind::Tuple(args)
            | ExprKind::DynCall { args, .. } => {
                args.iter().for_each(|a| self.expr(a));
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.pattern(&arm.pat);
                    self.expr(&arm.body);
                }
            }
        }
    }

    fn pattern(&mut self, p: &hir::Pat) {
        match p {
            hir::Pat::Bind(l) => self.local(l),
            hir::Pat::Variant { args, .. } | hir::Pat::Tuple(args) => {
                args.iter().for_each(|a| self.pattern(a));
            }
            hir::Pat::Wild | hir::Pat::Lit(_) => {}
        }
    }
}
