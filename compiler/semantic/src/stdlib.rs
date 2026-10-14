//! Typing rules for the Telvax standard library.
//!
//! Std functions are polymorphic (`map`, `filter`, ...) and the checker has no generics yet, so
//! each item carries a small typing rule instead of a fixed signature. The interpreter in
//! `telvax_runtime` implements the same set of items; keep the two in sync.

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
    /// For functions whose result type cannot be derived from the arguments (`zeros`,
    /// `read_csv`, ...): maps the argument types and the type the context expects to the result.
    pub from_expected: Option<FromExpected>,
}

type FromExpected = fn(&[Ty], &Ty) -> Result<Ty, String>;

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
        from_expected: None,
    }
}

const fn sig_expected(check: Check, from_expected: FromExpected) -> Sig {
    Sig {
        check,
        hint: no_hint,
        from_expected: Some(from_expected),
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
            "argument {} must be a `List`, found `{other}`",
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
    let item = elem(args, 0)?;
    match &args[1] {
        Ty::Fn(p, _) if p.len() == 1 && !p[0].accepts(&item) => Err(format!(
            "the function takes `{}` but the vector holds `{item}`: mismatched element type",
            p[0]
        )),
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
        Ty::Fn(p, _) if p.len() == 1 && !p[0].accepts(&t) => Err(format!(
            "the predicate takes `{}` but the vector holds `{t}`: mismatched element type",
            p[0]
        )),
        Ty::Fn(p, r) if p.len() == 1 && Ty::bool().accepts(r) => Ok(vec_of(t)),
        Ty::Unknown => Ok(vec_of(t)),
        other => Err(format!("argument 2 must be `(_) -> Bool`, found `{other}`")),
    }
}

fn take(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let t = elem(args, 0)?;
    want_int(args, 1)?;
    Ok(vec_of(t))
}

fn bool_fn_over_vec(args: &[Ty], what: &str) -> Result<Ty, String> {
    arity(args, 2)?;
    let t = elem(args, 0)?;
    match &args[1] {
        Ty::Fn(p, _) if p.len() == 1 && !p[0].accepts(&t) => Err(format!(
            "the predicate takes `{}` but the vector holds `{t}`: mismatched element type",
            p[0]
        )),
        Ty::Fn(p, r) if p.len() == 1 && Ty::bool().accepts(r) => Ok(t),
        Ty::Unknown => Ok(t),
        other => Err(format!(
            "argument 2 of `{what}` must be `(_) -> Bool`, found `{other}`"
        )),
    }
}

fn any_all(args: &[Ty]) -> Result<Ty, String> {
    bool_fn_over_vec(args, "any / all")?;
    Ok(Ty::bool())
}

fn find(args: &[Ty]) -> Result<Ty, String> {
    let t = bool_fn_over_vec(args, "find")?;
    Ok(Ty::Option(Box::new(t)))
}

fn for_each_hint(i: usize, prev: &[Ty]) -> Option<Ty> {
    if i == 1 {
        let t = elem(prev, 0).ok()?;
        Some(Ty::Fn(vec![t], Box::new(Ty::Unknown)))
    } else {
        None
    }
}

fn for_each(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    elem(args, 0)?;
    Ok(Ty::Unit)
}

fn enumerate(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    let t = elem(args, 0)?;
    Ok(vec_of(Ty::Tuple(vec![Ty::Prim(Prim::I64), t])))
}

fn zip(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let (a, b) = (elem(args, 0)?, elem(args, 1)?);
    Ok(vec_of(Ty::Tuple(vec![a, b])))
}

fn numeric_elem(args: &[Ty], what: &str) -> Result<Ty, String> {
    arity(args, 1)?;
    let t = elem(args, 0)?;
    match &t {
        Ty::Prim(p) if p.is_numeric() => Ok(t),
        Ty::IntVar(_) | Ty::FloatVar(_) | Ty::Unknown => Ok(t),
        other => Err(format!(
            "`{what}` needs a vector of numbers, found a vector of `{other}`"
        )),
    }
}

fn sum(args: &[Ty]) -> Result<Ty, String> {
    numeric_elem(args, "sum")
}

fn min_max(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    let t = elem(args, 0)?;
    match &t {
        Ty::Prim(p) if p.is_numeric() || *p == Prim::Str => Ok(Ty::Option(Box::new(t))),
        Ty::IntVar(_) | Ty::FloatVar(_) | Ty::Unknown => Ok(Ty::Option(Box::new(t))),
        other => Err(format!(
            "`min` and `max` need numbers or strings, found `{other}`"
        )),
    }
}

fn fold_hint(i: usize, prev: &[Ty]) -> Option<Ty> {
    if i == 2 {
        let t = elem(prev, 0).ok()?;
        let acc = prev.get(1)?.clone();
        Some(Ty::Fn(vec![acc.clone(), t], Box::new(acc)))
    } else {
        None
    }
}

fn fold(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 3)?;
    let t = elem(args, 0)?;
    let acc = args[1].clone();
    match &args[2] {
        Ty::Fn(p, r) if p.len() == 2 => {
            if !p[0].accepts(&acc) || !p[1].accepts(&t) {
                return Err(format!(
                    "the function takes (`{}`, `{}`) but fold passes (`{acc}`, `{t}`)",
                    p[0], p[1]
                ));
            }
            if !acc.accepts(r) {
                return Err(format!(
                    "the function returns `{r}` but the accumulator is `{acc}`"
                ));
            }
            Ok(acc)
        }
        Ty::Unknown => Ok(acc),
        other => Err(format!(
            "argument 3 must be a function of two arguments, found `{other}`"
        )),
    }
}

fn skip(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want_int(args, 1)?;
    Ok(vec_of(elem(args, 0)?))
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

// ---- tensors -----------------------------------------------------------------------------

fn tensor_of(t: &Ty) -> Option<(Ty, Vec<Dim>)> {
    match t {
        Ty::Tensor { elem, shape } => Some(((**elem).clone(), shape.clone())),
        _ => None,
    }
}

fn is_dynamic(t: &Ty) -> bool {
    matches!(t, Ty::Unknown | Ty::Never)
}

fn want_tensor(args: &[Ty], i: usize) -> Result<Option<(Ty, Vec<Dim>)>, String> {
    if is_dynamic(&args[i]) {
        return Ok(None);
    }
    tensor_of(&args[i])
        .map(Some)
        .ok_or_else(|| format!("argument {} must be a `Tensor`, found `{}`", i + 1, args[i]))
}

fn dims_match(a: &Dim, b: &Dim) -> bool {
    match (a, b) {
        (Dim::Lit(x), Dim::Lit(y)) => x == y,
        _ => true,
    }
}

/// Dimension to keep when two compatible dims meet: the concrete one wins.
fn merge_dim(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Named(_), other) => other.clone(),
        (other, _) => other.clone(),
    }
}

fn shape_str(shape: &[Dim]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Result of an element-wise operation on two tensors of compatible shape.
fn elementwise(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let (Some((e1, s1)), Some((e2, s2))) = (want_tensor(args, 0)?, want_tensor(args, 1)?) else {
        return Ok(if is_dynamic(&args[0]) {
            args[1].clone()
        } else {
            args[0].clone()
        });
    };
    if !e1.accepts(&e2) {
        return Err(format!("tensor element types differ: `{e1}` and `{e2}`"));
    }
    if s1.len() != s2.len() || !s1.iter().zip(&s2).all(|(a, b)| dims_match(a, b)) {
        return Err(format!(
            "tensor shapes differ: {} and {}",
            shape_str(&s1),
            shape_str(&s2)
        ));
    }
    let shape = s1.iter().zip(&s2).map(|(a, b)| merge_dim(a, b)).collect();
    Ok(Ty::Tensor {
        elem: Box::new(e1),
        shape,
    })
}

fn tensor_unary(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want_tensor(args, 0)?;
    Ok(args[0].clone())
}

fn tensor_scale(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want_tensor(args, 0)?;
    match &args[1] {
        Ty::Prim(p) if p.is_float() => {}
        Ty::FloatVar(_) | Ty::IntVar(_) | Ty::Unknown | Ty::Never => {}
        other => return Err(format!("argument 2 must be a float, found `{other}`")),
    }
    Ok(args[0].clone())
}

fn tensor_reduce(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    Ok(match want_tensor(args, 0)? {
        Some((elem, _)) => elem,
        None => Ty::Unknown,
    })
}

fn tensor_argmax(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want_tensor(args, 0)?;
    Ok(Ty::Prim(Prim::U64))
}

fn tensor_dot(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let (Some((e1, s1)), Some((e2, s2))) = (want_tensor(args, 0)?, want_tensor(args, 1)?) else {
        return Ok(Ty::Unknown);
    };
    if s1.len() != 1 || s2.len() != 1 {
        return Err("`dot` needs two rank-1 tensors; use `matmul` for matrices".into());
    }
    if !dims_match(&s1[0], &s2[0]) {
        return Err(format!(
            "`dot` operands differ in length: {} and {}",
            s1[0], s2[0]
        ));
    }
    if !e1.accepts(&e2) {
        return Err(format!("tensor element types differ: `{e1}` and `{e2}`"));
    }
    Ok(e1)
}

fn tensor_matmul(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let (Some((e1, s1)), Some((e2, s2))) = (want_tensor(args, 0)?, want_tensor(args, 1)?) else {
        return Ok(Ty::Unknown);
    };
    if !e1.accepts(&e2) {
        return Err(format!("tensor element types differ: `{e1}` and `{e2}`"));
    }
    let inner_ok = |k1: &Dim, k2: &Dim| -> Result<(), String> {
        if dims_match(k1, k2) {
            Ok(())
        } else {
            Err(format!("`matmul` inner dimensions differ: {k1} and {k2}"))
        }
    };
    match (s1.as_slice(), s2.as_slice()) {
        ([m, k1], [k2, n]) => {
            inner_ok(k1, k2)?;
            Ok(Ty::Tensor {
                elem: Box::new(e1),
                shape: vec![m.clone(), n.clone()],
            })
        }
        ([m, k1], [k2]) => {
            inner_ok(k1, k2)?;
            Ok(Ty::Tensor {
                elem: Box::new(e1),
                shape: vec![m.clone()],
            })
        }
        _ => Err(format!(
            "`matmul` needs shapes [M, K] x [K, N] or [M, K] x [K], found {} and {}",
            shape_str(&s1),
            shape_str(&s2)
        )),
    }
}

fn tensor_transpose(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    match want_tensor(args, 0)? {
        None => Ok(Ty::Unknown),
        Some((elem, s)) if s.len() == 2 => Ok(Ty::Tensor {
            elem: Box::new(elem),
            shape: vec![s[1].clone(), s[0].clone()],
        }),
        Some((_, s)) => Err(format!(
            "`transpose` needs a rank-2 tensor, found {}",
            shape_str(&s)
        )),
    }
}

fn static_count(shape: &[Dim]) -> Option<u64> {
    shape.iter().try_fold(1u64, |acc, d| match d {
        Dim::Lit(n) => acc.checked_mul(*n),
        Dim::Named(_) => None,
    })
}

fn expect_tensor(expected: &Ty, what: &str) -> Result<(Ty, Vec<Dim>), String> {
    tensor_of(expected).ok_or_else(|| {
        format!("cannot infer the tensor type of `{what}`; annotate it, e.g. `let t: Tensor[f32, 3] = ...`")
    })
}

fn zeros_check(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 0)?;
    Ok(Ty::Unknown)
}

fn zeros_result(_: &[Ty], expected: &Ty) -> Result<Ty, String> {
    expect_tensor(expected, "std::tensor::zeros")?;
    Ok(expected.clone())
}

fn map_new_result(_: &[Ty], expected: &Ty) -> Result<Ty, String> {
    match expected {
        Ty::Map(..) | Ty::Unknown => Ok(expected.clone()),
        other => Err(format!(
            "`std.map.new` makes a `Map<K, V>` but the context expects `{other}`; \
             write `let m: Map<String, i64> = std::map::new();`"
        )),
    }
}

fn from_vec_check(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    elem(args, 0)?;
    Ok(Ty::Unknown)
}

fn from_vec_result(_: &[Ty], expected: &Ty) -> Result<Ty, String> {
    expect_tensor(expected, "std::tensor::from_vec")?;
    Ok(expected.clone())
}

fn reshape_check(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want_tensor(args, 0)?;
    Ok(Ty::Unknown)
}

fn reshape_result(args: &[Ty], expected: &Ty) -> Result<Ty, String> {
    let (_, to) = expect_tensor(expected, "std::tensor::reshape")?;
    if let Some((_, from)) = tensor_of(&args[0])
        && let (Some(a), Some(b)) = (static_count(&from), static_count(&to))
        && a != b
    {
        return Err(format!(
            "cannot reshape {} ({a} elements) into {} ({b} elements)",
            shape_str(&from),
            shape_str(&to)
        ));
    }
    Ok(expected.clone())
}

fn to_vec(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    Ok(match want_tensor(args, 0)? {
        Some((elem, _)) => vec_of(elem),
        None => vec_of(Ty::Unknown),
    })
}

// ---- tables ------------------------------------------------------------------------------

fn want_table(args: &[Ty], i: usize) -> Result<Option<String>, String> {
    match &args[i] {
        Ty::Table(s) => Ok(Some(s.clone())),
        t if is_dynamic(t) => Ok(None),
        other => Err(format!(
            "argument {} must be a `Table`, found `{other}`",
            i + 1
        )),
    }
}

fn expect_table(expected: &Ty, what: &str) -> Result<Ty, String> {
    match expected {
        Ty::Table(_) => Ok(expected.clone()),
        _ => Err(format!(
            "cannot infer the schema of `{what}`; annotate the result, e.g. `let t: Table[Row] = ...`"
        )),
    }
}

fn read_check(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(Ty::Unknown)
}

fn read_result(_: &[Ty], expected: &Ty) -> Result<Ty, String> {
    expect_table(expected, "a table reader")
}

fn json_stringify(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    Ok(s())
}

fn json_parse(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want(args, 0, &s())?;
    Ok(Ty::Unknown)
}

/// `std.json.parse(text)` gives a `Result[T, Text]`; `T` is the type the context asks for.
fn json_parse_result(_: &[Ty], expected: &Ty) -> Result<Ty, String> {
    match expected {
        Ty::Result(_, e) if e.accepts(&s()) => Ok(expected.clone()),
        Ty::Unknown => Ok(Ty::Unknown),
        _ => Err(
            "cannot infer the type of `std.json.parse`; annotate the result, e.g. \
                  `let q: Result[Query, Text] = std.json.parse(body)`"
                .to_string(),
        ),
    }
}

fn sql_check(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want_table(args, 0)?;
    want(args, 1, &s())?;
    Ok(Ty::Unknown)
}

fn sql_result(_: &[Ty], expected: &Ty) -> Result<Ty, String> {
    expect_table(expected, "std::table::sql")
}

fn from_rows(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    match &args[0] {
        Ty::Vec(t) => match &**t {
            Ty::Struct(s) => Ok(Ty::Table(s.clone())),
            Ty::Unknown => Ok(Ty::Unknown),
            other => Err(format!(
                "`from_rows` needs a `List` of types, found `List[{other}]`"
            )),
        },
        t if is_dynamic(t) => Ok(Ty::Unknown),
        other => Err(format!(
            "argument 1 must be a `List` of types, found `{other}`"
        )),
    }
}

fn to_rows(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    Ok(match want_table(args, 0)? {
        Some(s) => vec_of(Ty::Struct(s)),
        None => vec_of(Ty::Unknown),
    })
}

fn table_len(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 1)?;
    want_table(args, 0)?;
    Ok(Ty::Prim(Prim::U64))
}

fn write_table(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    want_table(args, 0)?;
    want(args, 1, &s())?;
    Ok(Ty::Unit)
}

fn filter_rows_hint(i: usize, prev: &[Ty]) -> Option<Ty> {
    match (i, prev.first()) {
        (1, Some(Ty::Table(s))) => Some(Ty::Fn(vec![Ty::Struct(s.clone())], Box::new(Ty::bool()))),
        _ => None,
    }
}

fn filter_rows(args: &[Ty]) -> Result<Ty, String> {
    arity(args, 2)?;
    let schema = want_table(args, 0)?;
    match &args[1] {
        Ty::Fn(p, r) if p.len() == 1 && Ty::bool().accepts(r) => {}
        Ty::Unknown => {}
        other => {
            return Err(format!(
                "argument 2 must be `(row) -> Bool`, found `{other}`"
            ));
        }
    }
    Ok(schema.map_or(Ty::Unknown, Ty::Table))
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
            from_expected: None,
        },
    ),
    (
        "std::iter::filter",
        Sig {
            check: filter,
            hint: filter_hint,
            from_expected: None,
        },
    ),
    (
        "std::dist::map",
        Sig {
            check: map,
            hint: map_hint,
            from_expected: None,
        },
    ),
    (
        "std::iter::any",
        Sig {
            check: any_all,
            hint: filter_hint,
            from_expected: None,
        },
    ),
    (
        "std::iter::all",
        Sig {
            check: any_all,
            hint: filter_hint,
            from_expected: None,
        },
    ),
    (
        "std::iter::find",
        Sig {
            check: find,
            hint: filter_hint,
            from_expected: None,
        },
    ),
    (
        "std::iter::for_each",
        Sig {
            check: for_each,
            hint: for_each_hint,
            from_expected: None,
        },
    ),
    (
        "std::iter::fold",
        Sig {
            check: fold,
            hint: fold_hint,
            from_expected: None,
        },
    ),
    ("std::iter::enumerate", sig(enumerate)),
    ("std::iter::zip", sig(zip)),
    ("std::iter::sum", sig(sum)),
    ("std::iter::min", sig(min_max)),
    ("std::iter::max", sig(min_max)),
    ("std::iter::skip", sig(skip)),
    ("std::iter::take", sig(take)),
    ("std::iter::count", sig(count)),
    ("std::collect", sig(collect)),
    ("std::map::new", sig_expected(zeros_check, map_new_result)),
    (
        "std::tensor::zeros",
        sig_expected(zeros_check, zeros_result),
    ),
    (
        "std::tensor::from_vec",
        sig_expected(from_vec_check, from_vec_result),
    ),
    (
        "std::tensor::reshape",
        sig_expected(reshape_check, reshape_result),
    ),
    ("std::tensor::to_vec", sig(to_vec)),
    ("std::tensor::add", sig(elementwise)),
    ("std::tensor::sub", sig(elementwise)),
    ("std::tensor::mul", sig(elementwise)),
    ("std::tensor::div", sig(elementwise)),
    ("std::tensor::scale", sig(tensor_scale)),
    ("std::tensor::relu", sig(tensor_unary)),
    ("std::tensor::sigmoid", sig(tensor_unary)),
    ("std::tensor::softmax", sig(tensor_unary)),
    ("std::tensor::sum", sig(tensor_reduce)),
    ("std::tensor::mean", sig(tensor_reduce)),
    ("std::tensor::argmax", sig(tensor_argmax)),
    ("std::tensor::dot", sig(tensor_dot)),
    ("std::tensor::matmul", sig(tensor_matmul)),
    ("std::tensor::transpose", sig(tensor_transpose)),
    (
        "std::table::read_csv",
        sig_expected(read_check, read_result),
    ),
    (
        "std::table::read_parquet",
        sig_expected(read_check, read_result),
    ),
    ("std::table::sql", sig_expected(sql_check, sql_result)),
    ("std::json::stringify", sig(json_stringify)),
    (
        "std::json::parse",
        sig_expected(json_parse, json_parse_result),
    ),
    ("std::table::from_rows", sig(from_rows)),
    ("std::table::to_rows", sig(to_rows)),
    ("std::table::len", sig(table_len)),
    ("std::table::write_csv", sig(write_table)),
    ("std::table::write_parquet", sig(write_table)),
    (
        "std::table::filter",
        Sig {
            check: filter_rows,
            hint: filter_rows_hint,
            from_expected: None,
        },
    ),
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

pub fn type_paths() -> impl Iterator<Item = &'static str> {
    TYPES.iter().copied()
}

/// What a name inside a std module refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ItemKind {
    Module,
    Function,
    Type,
}

/// The direct children of the std path `prefix` (e.g. `std.text`), sorted, without duplicates.
pub fn children(prefix: &str) -> Vec<(String, ItemKind)> {
    let start = format!("{prefix}::");
    let mut out: Vec<(String, ItemKind)> = Vec::new();
    let mut add = |path: &str, leaf: ItemKind| {
        let Some(rest) = path.strip_prefix(&start) else {
            return;
        };
        let (name, kind) = match rest.split_once("::") {
            Some((head, _)) => (head, ItemKind::Module),
            None => (rest, leaf),
        };
        if !out.iter().any(|(n, _)| n == name) {
            out.push((name.to_string(), kind));
        }
    };
    for p in function_paths() {
        add(p, ItemKind::Function);
    }
    for p in type_paths() {
        add(p, ItemKind::Type);
    }
    out.sort();
    out
}

/// Methods available on the std type `ty`, with their names.
pub fn methods_of(ty: &str) -> Vec<&'static str> {
    METHODS
        .iter()
        .filter(|(t, _, _)| *t == ty)
        .map(|(_, n, _)| *n)
        .collect()
}
