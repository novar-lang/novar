//! Looking across files: which module an `import` names, which files use a function or a type
//! of another file, and what a rename changes in all of them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use telvax_ast as ast;
use telvax_modules::{Files, Package};

use crate::analysis::names::{Foreign, ItemInfo, Key, Names};
use crate::analysis::{ItemRef, Snapshot, Span, check_name};

/// The manifest that marks the root of a project.
const MANIFEST: &str = "telvax.toml";
/// How many files a search reads, so that a huge directory cannot freeze the editor.
const MAX_FILES: usize = 2000;

/// What the `import`s of a document bring in from other files of the project or its
/// dependencies. Imports that cannot be resolved are left out (the compiler reports them).
pub(crate) fn imports(
    program: &ast::Program,
    doc: Option<&Path>,
    deps: &[Package],
) -> Vec<Foreign> {
    let Some(doc) = doc else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in &program.items {
        let ast::Item::Import(imp) = item else {
            continue;
        };
        let segs: Vec<&str> = imp.path.iter().map(|s| s.node.as_str()).collect();
        let Some(first) = segs.first() else { continue };
        if *first == "std" {
            continue;
        }
        let (dep, rest) = match *first {
            "crate" => (None, &segs[1..]),
            name => match deps.iter().find(|d| d.name == name) {
                Some(d) => (Some(d), &segs[1..]),
                None => (None, &segs[..]),
            },
        };
        let (module, names): (&[&str], Vec<&ast::Ident>) = if imp.group.is_empty() {
            let Some(last) = imp.path.last().filter(|_| !rest.is_empty()) else {
                continue;
            };
            (&rest[..rest.len() - 1], vec![last])
        } else {
            (rest, imp.group.iter().collect())
        };
        let Some(file) = module_file(doc, dep, module) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let (Some(declared), _) = telvax_parser::parse(&text) else {
            continue;
        };
        let prefix = dep.map(|d| d.name.as_str());
        for (i, ident) in names.into_iter().enumerate() {
            let Some(info) = ItemInfo::of(&declared, &ident.node) else {
                continue;
            };
            let display = prefix
                .into_iter()
                .chain(module.iter().copied())
                .chain(std::iter::once(ident.node.as_str()))
                .collect::<Vec<_>>()
                .join(".");
            out.push(Foreign {
                local: imp.local_name(i, ident).node,
                name: ident.node.clone(),
                file: file.display().to_string(),
                display,
                ident: ident.span.clone(),
                info,
            });
        }
    }
    out
}

/// The file of the module `module` (`["a", "b"]` is `a/b.tvx` or `a/b/mod.tvx`) as `doc` imports
/// it: in a dependency, or in the directory of the document or one of its parents.
fn module_file(doc: &Path, dep: Option<&Package>, module: &[&str]) -> Option<PathBuf> {
    let in_dir = |dir: &Path| -> Option<PathBuf> {
        if module.is_empty() {
            return dep.map(|_| dir.join("lib.tvx")).filter(|f| f.is_file());
        }
        let mut base = dir.to_path_buf();
        base.extend(module);
        let file = base.with_extension("tvx");
        if file.is_file() {
            return Some(file);
        }
        let nested = base.join("mod.tvx");
        nested.is_file().then_some(nested)
    };
    match dep {
        Some(d) => in_dir(&d.src_dir),
        None => doc.parent()?.ancestors().take(6).find_map(in_dir),
    }
}

/// Where the item `key` is declared in the file `home`.
pub(crate) fn declaration(home: &str, key: &Key) -> Option<Span> {
    let text = std::fs::read_to_string(home).ok()?;
    let (program, errors) = telvax_parser::parse(&text);
    let names = Names::of(
        program.as_ref(),
        errors.is_empty(),
        &crate::analysis::names::Context {
            file: home,
            types: None,
            foreign: &[],
            demangle: &|n| n.to_string(),
        },
    );
    names.decl(key).cloned()
}

/// The directory of the project the document belongs to.
fn project_root(doc: &Path) -> PathBuf {
    let dir = doc.parent().unwrap_or(Path::new(".")).to_path_buf();
    dir.ancestors()
        .find(|d| d.join(MANIFEST).is_file())
        .map(Path::to_path_buf)
        .unwrap_or(dir)
}

/// Every `.tvx` file under `root`, skipping build output and hidden directories.
fn source_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if !name.starts_with('.') && name != "target" && name != "node_modules" {
                    pending.push(path);
                }
            } else if path.extension().is_some_and(|e| e == "tvx") {
                out.push(path);
            }
            if out.len() >= MAX_FILES {
                return out;
            }
        }
    }
    out.sort();
    out
}

/// Where an item appears in one file.
#[derive(Debug)]
pub struct Hit {
    pub path: PathBuf,
    pub text: String,
    pub decl: Option<Span>,
    pub spans: Vec<Span>,
    errors: usize,
    parsed: bool,
    typed: bool,
}

/// Every file of the project that declares or uses `item`. `open` holds the text of the files the
/// editor has open, which wins over what is on disk; `doc` is the open document being asked about.
pub fn find(
    item: &ItemRef,
    doc: &Snapshot,
    doc_path: &Path,
    open: &HashMap<PathBuf, String>,
) -> Vec<Hit> {
    let mut files = source_files(&project_root(doc_path));
    files.push(doc_path.to_path_buf());
    files.push(PathBuf::from(&item.home));
    files.sort();
    files.dedup();
    let name = item.key.name();
    let mut hits = Vec::new();
    for path in files {
        let is_doc = path == doc_path;
        let text = if is_doc {
            doc.text.clone()
        } else if let Some(t) = open.get(&path) {
            t.clone()
        } else if let Ok(t) = std::fs::read_to_string(&path) {
            t
        } else {
            continue;
        };
        let is_home = path.display().to_string() == item.home;
        if !is_doc && !is_home && !text.contains(name) {
            continue;
        }
        let analysed;
        let snap = if is_doc {
            doc
        } else {
            analysed = Snapshot::analyze_in(text.clone(), Some(&path));
            &analysed
        };
        let (decl, spans) = snap.item_spans(item);
        if decl.is_none() && spans.is_empty() {
            continue;
        }
        let (parsed, typed) = snap.is_exact();
        hits.push(Hit {
            path,
            text,
            decl,
            spans,
            errors: snap.error_count(),
            parsed,
            typed,
        });
    }
    hits
}

/// The texts of the files as they will be after a rename, over the files on disk.
struct Overlay<'a>(&'a HashMap<PathBuf, String>);

impl Files for Overlay<'_> {
    fn read(&self, path: &Path) -> std::io::Result<String> {
        match self.0.get(path) {
            Some(t) => Ok(t.clone()),
            None => std::fs::read_to_string(path),
        }
    }

    fn is_file(&self, path: &Path) -> bool {
        self.0.contains_key(path) || path.is_file()
    }
}

/// The edits that rename an item in one file: the text they apply to and the replacements.
#[derive(Debug)]
pub struct FileEdit {
    pub path: PathBuf,
    pub text: String,
    pub edits: Vec<(Span, String)>,
}

/// The edits, per file, that rename `item` to `new_name`. The renamed project must analyse with no
/// more errors than before, which catches a new name that collides with another one.
pub fn rename(
    item: &ItemRef,
    new_name: &str,
    doc: &Snapshot,
    doc_path: &Path,
    open: &HashMap<PathBuf, String>,
) -> Result<Vec<FileEdit>, String> {
    check_name(new_name)?;
    let name = item.key.name();
    if name == "main" {
        return Err("`main` is the entry point and cannot be renamed".into());
    }
    let hits = find(item, doc, doc_path, open);
    if hits.is_empty() {
        return Err("there is no symbol here".into());
    }
    let root = project_root(doc_path);
    let needs_types = matches!(item.key, Key::Field(..) | Key::Method(..));
    let mut edits = Vec::new();
    let mut renamed: HashMap<PathBuf, String> = HashMap::new();
    for hit in &hits {
        let shown = hit.path.display();
        if !hit.path.starts_with(&root) {
            return Err(format!("`{name}` belongs to {shown}, outside this project"));
        }
        if !hit.parsed {
            return Err(format!("fix the syntax errors in {shown} before renaming"));
        }
        if needs_types && !hit.typed {
            return Err(format!(
                "the types of {shown} are not known; fix its errors before renaming"
            ));
        }
        // A use under another name (`import a as b`) keeps that name.
        let spans: Vec<Span> = hit
            .decl
            .iter()
            .chain(&hit.spans)
            .filter(|s| hit.text.get((*s).clone()) == Some(name))
            .cloned()
            .collect();
        let mut spans = spans;
        spans.sort_by_key(|s| s.start);
        spans.dedup();
        let mut text = hit.text.clone();
        for s in spans.iter().rev() {
            text.replace_range(s.clone(), new_name);
        }
        renamed.insert(hit.path.clone(), text);
        edits.push(FileEdit {
            path: hit.path.clone(),
            text: hit.text.clone(),
            edits: spans
                .into_iter()
                .map(|s| (s, new_name.to_string()))
                .collect(),
        });
    }
    let overlay = Overlay(&renamed);
    for hit in &hits {
        let after = Snapshot::analyze_with(renamed[&hit.path].clone(), Some(&hit.path), &overlay);
        if after.error_count() > hit.errors {
            return Err(format!(
                "`{new_name}` collides with another name in scope in {}",
                hit.path.display()
            ));
        }
    }
    Ok(edits)
}
