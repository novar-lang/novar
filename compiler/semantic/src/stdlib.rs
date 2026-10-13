//! Typing rules for the Novar standard library.
//!
//! Std functions are polymorphic (`map`, `filter`, ...) and the checker has no generics yet, so
//! each item carries a small typing rule instead of a fixed signature. The interpreter in
//! `novar_runtime` implements the same set of items; keep the two in sync.

use crate::ty::{Dim, Prim, Ty};

type Check = fn(&[Ty]) -> Result<Ty, String>;
type Hint = fn(usize, &[Ty]) -> Option<Ty>;

/// Typing rule of one std function or method. For methods, `args` excludes the receiver.
#[derive(Clone, Copy)]
pub struct Sig {
    /// Validates the argument types and returns the result type.
    pub check: Check,
    /// Expected type for argument `i` given the types of the previous ones (guides closures).
    pub hint: Hint,
}

fn no_hint(_: usize, _: &[Ty]) -> Option<Ty> {
    None
}

impl std::fmt::Debug for Sig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Sig")
    }
}

const fn sig(check: Check) -> Sig {
    Sig {
        check,
        hint: no_hint,
    }
}

fn arity(args: &[Ty], n: usize) -> Result<(), String> {
    if args.len() == n {
        Ok(())
    } else {
        Err(format!("expected {n} argument(s), found {}", args.len()))
    }
}

fn want(args: &[Ty], i: usize, expected: &Ty) -> Result<(), String> {
    if expected.accepts(&args[i]) {
        Ok(())
    } else {
        Err(format!(
            "argument {} must be `{expected}`, found `{}`",
            i + 1,
            args[i]
        ))
    }
}

fn want_int(args: &[Ty], i: usize) -> Result<(), String> {
    match &args[i] {
        Ty::Prim(p) if p.is_int() => Ok(()),
        Ty::IntVar(_) | Ty::Unknown | Ty::Never => Ok(()),
        other => Err(format!(
            "argument {} must be an integer, found `{other}`",
            i + 1
        )),
    }
}

fn s() -> Ty {
    Ty::str()
}

fn vec_of(t: Ty) -> Ty {
    Ty::Vec(Box::new(t))
}

fn named(n: &str) -> Ty {
    Ty::Named(n.to_string())
}

/// Element type of a `Vec` argument; `Unknown` when the argument is not statically known.
fn elem(args: &[Ty], i: usize) -> Result<Ty, String> {
    match &args[i] {
        Ty::Vec(t) => Ok((**t).clone()),
        Ty::Unknown | Ty::Never => Ok(Ty::Unknown),
        other => Err(format!(
            "argument {} must be a `Vec`, found `{other}`",
            i + 1
        )),
    }
}

fn str_to_str(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(s())
}

fn truncate(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want(args, 0, &s())?;
    want_int(args, 1)?;
    Ok(s())
}

fn split(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want(args, 0, &s())?;
    want(args, 1, &s())?;
    Ok(vec_of(s()))
}

fn join(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    elem(args, 0)?;
    want(args, 1, &s())?;
    Ok(s())
}

fn map_hint(i: usize, prev: &[Ty]) -> Option<Ty> {
    if i == 1 {
        let t = elem(prev, 0).ok()?;
        Some(Ty::Fn(vec![t], Box::new(Ty::Unknown)))
    } else {
        None
    }
}

fn map(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    elem(args, 0)?;
    match &args[1] {
        Ty::Fn(p, r) if p.len() == 1 => Ok(vec_of((**r).clone())),
        Ty::Unknown => Ok(vec_of(Ty::Unknown)),
        other => Err(format!(
            "argument 2 must be a one-argument function, found `{other}`"
        )),
    }
}

fn filter_hint(i: usize, prev: &[Ty]) -> Option<Ty> {
    if i == 1 {
        let t = elem(prev, 0).ok()?;
        Some(Ty::Fn(vec![t], Box::new(Ty::bool())))
    } else {
        None
    }
}

fn filter(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let t = elem(args, 0)?;
    match &args[1] {
        Ty::Fn(p, r) if p.len() == 1 && Ty::bool().accepts(r) => Ok(vec_of(t)),
        Ty::Unknown => Ok(vec_of(t)),
        other => Err(format!(
            "argument 2 must be `fn(_) -> bool`, found `{other}`"
        )),
    }
}

fn take(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let t = elem(args, 0)?;
    want_int(args, 1)?;
    Ok(vec_of(t))
}

fn count(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    elem(args, 0)?;
    Ok(Ty::Prim(Prim::U64))
}

fn collect(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    Ok(vec_of(elem(args, 0)?))
}

fn print(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(Ty::Unit)
}

fn status(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(named("Status"))
}

fn new_embedder(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(named("OllamaEmbedder"))
}

fn new_llm(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(named("OpenAIClient"))
}

fn connect(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(named("PgVectorClient"))
}

/// Dimension produced by the embedding models supported in the alpha (`nomic-embed-text`).
pub const EMBEDDING_DIM: u64 = 768;

fn embedding() -> Ty {
    Ty::Tensor {
        elem: Box::new(Ty::Prim(Prim::F32)),
        shape: vec![Dim::Lit(EMBEDDING_DIM)],
    }
}

fn embed(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(embedding())
}

fn complete(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(s())
}

fn search_collection(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(named("VectorQuery"))
}

fn with_vector(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want(args, 0, &named("VectorQuery"))?;
    want(args, 1, &embedding())?;
    Ok(named("VectorQuery"))
}

fn limit(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want(args, 0, &named("VectorQuery"))?;
    want_int(args, 1)?;
    Ok(named("VectorQuery"))
}

fn execute(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &named("VectorQuery"))?;
    // Rows are objects with `id`, `text` and `score` fields.
    Ok(vec_of(Ty::Unknown))
}

fn insert(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 4)?;
    want(args, 0, &s())?;
    want(args, 1, &s())?;
    want(args, 2, &s())?;
    want(args, 3, &embedding())?;
    Ok(Ty::Unit)
}

const FUNCTIONS: &[(&str, Sig)] = &[
    ("std::text::clean_whitespace", sig(str_to_str)),
    ("std::text::trim", sig(str_to_str)),
    ("std::text::lowercase", sig(str_to_str)),
    ("std::text::uppercase", sig(str_to_str)),
    ("std::text::truncate", sig(truncate)),
    ("std::text::split", sig(split)),
    ("std::string::join", sig(join)),
    (
        "std::iter::map",
        Sig {
            check: map,
            hint: map_hint,
        },
    ),
    (
        "std::iter::filter",
        Sig {
            check: filter,
            hint: filter_hint,
        },
    ),
    ("std::iter::take", sig(take)),
    ("std::iter::count", sig(count)),
    ("std::collect", sig(collect)),
    ("std::io::print", sig(print)),
    ("std::io::println", sig(print)),
    ("std::http::Status::BadRequest", sig(status)),
    ("std::http::Status::Unauthorized", sig(status)),
    ("std::http::Status::Forbidden", sig(status)),
    ("std::http::Status::NotFound", sig(status)),
    ("std::http::Status::Unprocessable", sig(status)),
    ("std::http::Status::Internal", sig(status)),
    (
        "std::ai::embeddings::OllamaEmbedder::new",
        sig(new_embedder),
    ),
    ("std::ai::llm::OpenAIClient::new", sig(new_llm)),
    ("std::db::pgvector::PgVectorClient::connect", sig(connect)),
];

const METHODS: &[(&str, &str, Sig)] = &[
    ("OllamaEmbedder", "embed", sig(embed)),
    ("OpenAIClient", "complete", sig(complete)),
    (
        "PgVectorClient",
        "search_collection",
        sig(search_collection),
    ),
    ("PgVectorClient", "with_vector", sig(with_vector)),
    ("PgVectorClient", "limit", sig(limit)),
    ("PgVectorClient", "execute", sig(execute)),
    ("PgVectorClient", "insert", sig(insert)),
];

/// Importable std types: full path -> short name.
const TYPES: &[&str] = &[
    "std::http::Request",
    "std::http::Response",
    "std::http::Status",
    "std::ai::embeddings::OllamaEmbedder",
    "std::ai::llm::OpenAIClient",
    "std::db::pgvector::PgVectorClient",
    "std::db::pgvector::VectorQuery",
];

pub fn function(path: &str) -> Option<Sig> {
    FUNCTIONS.iter().find(|(p, _)| *p == path).map(|(_, s)| *s)
}

pub fn method(ty: &str, name: &str) -> Option<Sig> {
    METHODS
        .iter()
        .find(|(t, n, _)| *t == ty && *n == name)
        .map(|(_, _, s)| *s)
}

/// Whether `short_name` is a std type that has methods (so unknown methods are errors).
pub fn is_std_type(short_name: &str) -> bool {
    TYPES
        .iter()
        .any(|p| p.rsplit("::").next() == Some(short_name))
}

pub fn is_type(path: &str) -> bool {
    TYPES.contains(&path)
}

/// Whether `path` names a std module (a proper prefix of some item).
pub fn is_module(path: &str) -> bool {
    let prefix = format!("{path}::");
    path == "std"
        || FUNCTIONS.iter().any(|(p, _)| p.starts_with(&prefix))
        || TYPES.iter().any(|p| p.starts_with(&prefix))
}

/// Every std function path, for runtimes that must implement them all.
pub fn function_paths() -> impl Iterator<Item = &'static str> {
    FUNCTIONS.iter().map(|(p, _)| *p)
}

pub fn method_names() -> impl Iterator<Item = (&'static str, &'static str)> {
    METHODS.iter().map(|(t, n, _)| (*t, *n))
}
