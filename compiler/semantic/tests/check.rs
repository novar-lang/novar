use telvax_hir::{Ty, hir, lower};

fn check(src: &str) -> (hir::Program, Vec<String>) {
    let (ast, parse_errs) = telvax_parser::parse(src);
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    let (hir, diags) = lower(&ast.unwrap());
    (hir, diags.into_iter().map(|d| d.message).collect())
}

fn errors(src: &str) -> Vec<String> {
    check(src).1
}

fn assert_clean(src: &str) {
    let errs = errors(src);
    assert!(errs.is_empty(), "unexpected errors: {errs:#?}");
}

fn assert_err(src: &str, needle: &str) {
    let errs = errors(src);
    assert!(
        errs.iter().any(|e| e.contains(needle)),
        "expected `{needle}` in {errs:#?}"
    );
}

#[test]
fn rag_example_is_clean() {
    let src = include_str!("../../../examples/rag-service.tvx");
    let (hir, errs) = check(src);
    assert!(errs.is_empty(), "{errs:#?}");

    let handler = hir.fns.iter().find(|f| f.name == "handle_rag").unwrap();
    let ep = handler.endpoint.as_ref().unwrap();
    assert_eq!(ep.path, "/v1/rag/query");
    assert_eq!(ep.method, hir::HttpMethod::Post);
    assert!(matches!(handler.params[0].ty, Ty::Struct(ref s) if s == "QueryPayload"));
    assert_eq!(hir.structs.len(), 3);
    assert_eq!(hir.globals.len(), 3);
}

#[test]
fn pipe_is_desugared_into_a_call() {
    let (hir, errs) = check(
        r#"
def inc(x: int32) -> int32:
    return x

def f(a: int32) -> int32:
    return a |> inc()
"#,
    );
    assert!(errs.is_empty(), "{errs:?}");
    let f = hir.fns.iter().find(|f| f.name == "f").unwrap();
    let tail = f.body.tail.as_ref().unwrap();
    let hir::ExprKind::Call { callee, args } = &tail.kind else {
        panic!("{:?}", tail.kind)
    };
    assert!(matches!(&callee.kind, hir::ExprKind::Fn(n) if n == "inc"));
    assert_eq!(args.len(), 1);
}

#[test]
fn pipe_type_errors_are_checked_through_the_call() {
    assert_err(
        r#"
def inc(x: int32) -> int32:
    return x

def f(s: str) -> int32:
    return s |> inc()
"#,
        "expected `int32`, found `str`",
    );
    assert_err(
        r#"
def f(a: int32) -> int32:
    return a |> 3
"#,
        "right side of `|>`",
    );
}

#[test]
fn tensor_shapes_are_enforced() {
    let ok = r#"
def take(x: Tensor[float32, [4]]):
    pass

def f(y: Tensor[float32, [4]]):
    take(y)
"#;
    assert_clean(ok);
    assert_err(
        r#"
def take(x: Tensor[float32, [4]]):
    pass

def f(y: Tensor[float32, [8]]):
    take(y)
"#,
        "expected `Tensor[float32, [4]]`, found `Tensor[float32, [8]]`",
    );
    assert_err(
        r#"
def take(x: Tensor[float32, [4]]):
    pass

def f(y: Tensor[float32, [4, 4]]):
    take(y)
"#,
        "mismatched types",
    );
    assert_err(
        r#"
def take(x: Tensor[float32, [4]]):
    pass

def f(y: Tensor[int32, [4]]):
    take(y)
"#,
        "mismatched types",
    );
    // Symbolic dimensions match any size.
    assert_clean(
        r#"
def take(x: Tensor[float32, [4]]):
    pass

def f(y: Tensor[float32, [N]]):
    take(y)
"#,
    );
}

#[test]
fn invalid_tensor_types() {
    assert_err(
        r#"
def f(x: Tensor[str, [4]]):
    pass
"#,
        "numeric primitive",
    );
    assert_err(
        r#"
def f(x: Tensor[float32]):
    pass
"#,
        "`Tensor` expects",
    );
    assert_err(
        r#"
def f(x: Tensor[float32, [0]]):
    pass
"#,
        "greater than zero",
    );
    assert_err(
        r#"
def f(x: Tensor[float32, []]):
    pass
"#,
        "at least one dimension",
    );
}

#[test]
fn name_resolution() {
    assert_err(
        r#"
def f() -> int32:
    return missing
"#,
        "cannot find `missing`",
    );
    assert_err(
        r#"
def f(x: Nope):
    pass
"#,
        "cannot find type `Nope`",
    );
    assert_err(
        r#"
class A:
    pass

def f():
    A
"#,
        "is a class, not a value",
    );
    assert_err(
        r#"
def f():
    pass

def f():
    pass
"#,
        "defined twice",
    );
    assert_err(
        r#"
def f(a: int32, a: int32):
    pass
"#,
        "declared twice",
    );
    assert_clean(
        r#"
import std.ai.llm.OpenAIClient

c = OpenAIClient.new("m")

def f():
    x = c
"#,
    );
    // A binding is not visible inside its own initializer.
    assert_err(
        r#"
def f():
    x = x
"#,
        "cannot find `x`",
    );
}

#[test]
fn literals_adapt_to_context() {
    assert_clean(
        r#"
def f(a: float32) -> float32:
    return a + 1
"#,
    );
    assert_clean(
        r#"
def f():
    x: float32 = 1.5
    y: uint8 = 3
    z: float = 2
"#,
    );
    assert_err(
        r#"
def f():
    x: int32 = 1.5
"#,
        "expected `int32`, found `{float}`",
    );
    assert_err(
        r#"
def f():
    x: int32 = "a"
"#,
        "expected `int32`, found `str`",
    );
}

#[test]
fn calls() {
    assert_err(
        r#"
def g(a: int32):
    pass

def f():
    g(1, 2)
"#,
        "expected 1 argument(s), found 2",
    );
    assert_err(
        r#"
def g(a: int32):
    pass

def f():
    g(True)
"#,
        "expected `int32`, found `bool`",
    );
    assert_err(
        r#"
def f(a: int32):
    a()
"#,
        "expected a function",
    );
}

#[test]
fn structs() {
    let decl = r#"
class P:
    x: int32
    name: str
"#;
    assert_clean(&format!(
        r#"
{decl}
def f() -> P:
    return P(x=1, name="a")
"#
    ));
    assert_err(
        &format!(
            r#"
{decl}
def f():
    P(x=1)
"#
        ),
        "missing field(s) in `P`: `name`",
    );
    assert_err(
        &format!(
            r#"
{decl}
def f():
    P(x=1, name="a", y=2)
"#
        ),
        "no field `y`",
    );
    assert_err(
        &format!(
            r#"
{decl}
def f(p: P) -> int32:
    return p.nope
"#
        ),
        "no field `nope`",
    );
    assert_err(
        &format!(
            r#"
{decl}
def f():
    P(x="s", name="a")
"#
        ),
        "expected `int32`",
    );
    assert_err(
        r#"
class A:
    x: int32
    x: int32
"#,
        "declared twice",
    );
    assert_err(
        r#"
def f():
    Q(a=1)
"#,
        "cannot find class `Q`",
    );
}

#[test]
fn returns_and_blocks() {
    assert_clean(
        r#"
def f() -> int32:
    return 1
"#,
    );
    assert_clean(
        r#"
def f(c: bool) -> int32:
    if c:
        return 1
    return 2
"#,
    );
    assert_err(
        r#"
def f() -> int32:
    x = 1
"#,
        "expected `int32`, found `None`",
    );
    assert_err(
        r#"
def f() -> int32:
    return
"#,
        "needs a value",
    );
    assert_err(
        r#"
def f() -> int32:
    return "s"
"#,
        "expected `int32`, found `str`",
    );
    assert_clean(
        r#"
def f():
    return
"#,
    );
}

#[test]
fn conditions_and_operators() {
    assert_err(
        r#"
def f():
    if 1:
        pass
"#,
        "expected `bool`, found `{integer}`",
    );
    assert_err(
        r#"
def f(c: bool) -> int32:
    if c:
        return 1
    else:
        return "s"
"#,
        "incompatible types",
    );
    assert_clean(
        r#"
def f(c: bool) -> int32:
    if c:
        return 1
    else:
        return 2
"#,
    );
    assert_err(
        r#"
def f() -> int32:
    return 1 + "a"
"#,
        "mismatched operand types",
    );
    assert_err(
        r#"
def f() -> bool:
    return True and 1
"#,
        "expected `bool`",
    );
    assert_err(
        r#"
def f() -> bool:
    return True < False
"#,
        "cannot order",
    );
    assert_clean(
        r#"
def f(a: str) -> str:
    return a + "x"
"#,
    );
    assert_err(
        r#"
def f():
    x = -"a"
"#,
        "cannot apply",
    );
}

#[test]
fn result_constructors_follow_the_declared_type() {
    assert_clean(
        r#"
def f() -> Result[int32, str]:
    return Ok(1)
"#,
    );
    assert_clean(
        r#"
def f() -> Result[int32, str]:
    return Err("bad")
"#,
    );
    assert_err(
        r#"
def f() -> Result[int32, str]:
    return Ok("x")
"#,
        "expected `int32`, found `str`",
    );
    assert_err(
        r#"
def f() -> Result[int32, str]:
    return Err(1)
"#,
        "expected `str`, found `{integer}`",
    );
    assert_err(
        r#"
def f() -> Result[int32, str]:
    return Ok(1, 2)
"#,
        "exactly one argument",
    );
}

#[test]
fn string_and_vec_methods() {
    assert_clean(
        r#"
def f(s: str) -> bool:
    return s.is_empty()
"#,
    );
    assert_err(
        r#"
def f(s: str) -> int32:
    return s.is_empty()
"#,
        "expected `int32`, found `bool`",
    );
    assert_clean(
        r#"
def f(v: list[int32]) -> uint64:
    return v.len()
"#,
    );
}

#[test]
fn endpoints() {
    let ok = r#"
class Q:
    a: int32

@endpoint(path="/x", method="POST")
def h(q: Q) -> Result[Q, str]:
    return Ok(q)
"#;
    assert_clean(ok);
    let get = r#"
@endpoint(path="/ping")
def h() -> Result[str, str]:
    return Ok("pong")
"#;
    let (hir, errs) = check(get);
    assert!(errs.is_empty(), "{errs:?}");
    assert_eq!(
        hir.fns[0].endpoint.as_ref().unwrap().method,
        hir::HttpMethod::Get
    );

    assert_err(
        r#"
@endpoint(method="GET")
def h() -> Result[int32, int32]:
    return Ok(1)
"#,
        "requires a `path`",
    );
    assert_err(
        r#"
@endpoint(path="x")
def h() -> Result[int32, int32]:
    return Ok(1)
"#,
        "must start with `/`",
    );
    assert_err(
        r#"
@endpoint(path="/x", method="FETCH")
def h() -> Result[int32, int32]:
    return Ok(1)
"#,
        "unsupported HTTP method",
    );
    assert_err(
        r#"
@endpoint(path="/x", method="POST")
def h() -> Result[int32, int32]:
    return Ok(1)
"#,
        "payload parameter",
    );
    assert_err(
        r#"
@endpoint(path="/x")
def h(a: int32) -> Result[int32, int32]:
    return Ok(1)
"#,
        "must be a class",
    );
    assert_err(
        r#"
@endpoint(path="/x")
def h() -> int32:
    return 1
"#,
        "must return `Result",
    );
    assert_err(
        r#"
@endpoint(path="/x", color="r")
def h() -> Result[int32, int32]:
    return Ok(1)
"#,
        "unknown `@endpoint` argument",
    );
    assert_err(
        r#"
@route
def h():
    pass
"#,
        "unknown attribute",
    );
    assert_err(
        r#"
@endpoint(path="/x")
def a() -> Result[int32, int32]:
    return Ok(1)

@endpoint(path="/x")
def b() -> Result[int32, int32]:
    return Ok(1)
"#,
        "duplicate endpoint",
    );
}

#[test]
fn closures_and_extern_calls_stay_lenient() {
    assert_clean(
        r#"
import std.iter.map

def f(v: list):
    r = v |> std.iter.map(lambda d: d.text) |> std.collect()
"#,
    );
    assert_clean(
        r#"
def f():
    g = lambda a, b: a + b
"#,
    );
}

#[test]
fn diagnostics_render_with_source_context() {
    let src = r#"
def f() -> int32:
    return "oops"
"#;
    let (ast, _) = telvax_parser::parse(src);
    let (_, diags) = lower(&ast.unwrap());
    let out = telvax_diagnostics::render("main.tvx", src, &diags, false);
    assert!(out.contains("main.tvx:3:12"), "{out}");
    assert!(out.contains("expected `int32`, found `str`"), "{out}");
}

// ---- standard library ----------------------------------------------------------------------

#[test]
fn std_functions_are_typed() {
    assert_clean(
        r#"
def f(s: str) -> str:
    return s |> std.text.clean_whitespace() |> std.text.truncate(10)
"#,
    );
    assert_err(
        r#"
def f(n: int32) -> str:
    return std.text.trim(n)
"#,
        "argument 1 must be `str`, found `int32`",
    );
    assert_err(
        r#"
def f(s: str) -> str:
    return std.text.truncate(s)
"#,
        "expected 2 argument(s), found 1",
    );
    assert_err(
        r#"
def f(s: str) -> str:
    return std.text.truncate(s, "x")
"#,
        "must be an integer",
    );
    assert_err(
        r#"
def f(s: str) -> int32:
    return std.text.trim(s)
"#,
        "expected `int32`, found `str`",
    );
}

#[test]
fn std_unknown_items_are_errors() {
    assert_err(
        r#"
def f():
    std.text.nope()
"#,
        "cannot find `std.text.nope`",
    );
    assert_err(
        r#"
import std.nope.Thing

def f():
    pass
"#,
        "unresolved import",
    );
    assert_err(
        r#"
import std.text

def f():
    text.nope()
"#,
        "cannot find `std.text.nope`",
    );
    assert_err(
        r#"
def f():
    x = std.text
"#,
        "is not a value",
    );
}

#[test]
fn iterators_propagate_element_types() {
    assert_clean(
        r#"
class D:
    text: str

def f(v: list[D]) -> str:
    return v |> std.iter.map(lambda d: d.text) |> std.string.join(", ")
"#,
    );
    // The closure parameter is `D`, so a missing field is caught inside it.
    assert_err(
        r#"
class D:
    text: str

def f(v: list[D]):
    v |> std.iter.map(lambda d: d.nope)
"#,
        "no field `nope`",
    );
    assert_err(
        r#"
def f(v: list[str]) -> list[int32]:
    return v |> std.iter.map(lambda s: s)
"#,
        "expected `list[int32]`, found `list[str]`",
    );
    assert_err(
        r#"
def f(n: int32):
    std.iter.map(n, lambda x: x)
"#,
        "must be a `list`",
    );
    assert_err(
        r#"
def f(v: list[int32]):
    v |> std.iter.filter(lambda x: 1)
"#,
        "must be `(_) -> bool`",
    );
    assert_clean(
        r#"
def f(v: list[int32]) -> uint64:
    return v |> std.iter.take(3) |> std.iter.count()
"#,
    );
}

#[test]
fn std_types_have_typed_methods() {
    let head = r#"
import std.ai.embeddings.OllamaEmbedder
import std.ai.llm.OpenAIClient

e = OllamaEmbedder.new("nomic-embed-text")
l = OpenAIClient.new("m")
"#;
    assert_clean(&format!(
        r#"
{head}
def f(t: str) -> Tensor[float32, [768]]:
    return t |> e.embed()
"#
    ));
    assert_err(
        &format!(
            r#"
{head}
def f(t: str) -> Tensor[float32, [512]]:
    return t |> e.embed()
"#
        ),
        "expected `Tensor[float32, [512]]`, found `Tensor[float32, [768]]`",
    );
    assert_err(
        &format!(
            r#"
{head}
def f(t: str) -> str:
    return l.nope(t)
"#
        ),
        "no method `nope` on type `OpenAIClient`",
    );
    assert_err(
        &format!(
            r#"
{head}
def f() -> str:
    return l.complete(1)
"#
        ),
        "must be `str`",
    );
}

#[test]
fn vector_query_pipeline() {
    let head = r#"
import std.db.pgvector.PgVectorClient

db = PgVectorClient.connect("postgresql://x")
"#;
    assert_clean(&format!(
        r#"
{head}
def f(v: Tensor[float32, [768]]) -> list[str]:
    return db.search_collection("kb") |> db.with_vector(v) |> db.limit(3) |> db.execute()
"#
    ));
    assert_err(
        &format!(
            r#"
{head}
def f(v: Tensor[float32, [8]]):
    db.search_collection("kb") |> db.with_vector(v)
"#
        ),
        "must be `Tensor[float32, [768]]`",
    );
    assert_err(
        &format!(
            r#"
{head}
def f():
    db.execute("kb")
"#
        ),
        "must be `VectorQuery`",
    );
}

#[test]
fn status_constructors_flow_into_endpoint_results() {
    let src = r#"
import std.http.Status

def f(ok: bool) -> Result[str, Status]:
    if ok:
        return Ok("y")
    return Err(Status.NotFound("nope"))
"#;
    assert_clean(src);
    assert_err(
        r#"
import std.http.Status

def f() -> Result[str, Status]:
    return Err(Status.Teapot("x"))
"#,
        "cannot find `std.http.Status.Teapot`",
    );
}

// ---- mutation, loops, arrays ---------------------------------------------------------------

#[test]
fn mutable_locals() {
    assert_clean(
        r#"
def f() -> int32:
    x = 1
    x = x + 1
    x += 3
    return x
"#,
    );
    assert_err(
        r#"
def f():
    x = 1
    x = "s"
"#,
        "expected `{integer}`, found `str`",
    );
    assert_err(
        r#"
g = 1

def f():
    g = 2
"#,
        "cannot assign to the global `g`",
    );
    assert_err(
        r#"
def f(p: int32):
    x = 1
    1 + 2 = 3
"#,
        "must be a variable",
    );
}

#[test]
fn while_loops() {
    assert_clean(
        r#"
def f(n: int32) -> int32:
    i = 0
    s = 0
    while i < n:
        s += i
        i += 1
    return s
"#,
    );
    assert_err(
        r#"
def f():
    while 1:
        pass
"#,
        "expected `bool`, found `{integer}`",
    );
    assert_clean(
        r#"
def f(n: int32) -> int32:
    i = 0
    while True:
        if i > n:
            return i
        i += 1
    return 0
"#,
    );
}

#[test]
fn for_loops() {
    assert_clean(
        r#"
def f(n: int32) -> int32:
    s = 0
    for i in range(n):
        s += i
    return s
"#,
    );
    assert_clean(
        r#"
def f(n: uint8) -> uint8:
    s: uint8 = 0
    for i in range(1, n + 1):
        s += i
    return s
"#,
    );
    assert_clean(
        r#"
def f(v: list[float]) -> float:
    s = 0.0
    for x in v:
        s += x
    return s
"#,
    );
    assert_err(
        r#"
def f():
    for i in range(1.5):
        pass
"#,
        "range bounds",
    );
    assert_err(
        r#"
def f(a: int32, b: int):
    for i in range(a, b):
        pass
"#,
        "different types",
    );
    assert_err(
        r#"
def f(s: str):
    for c in s:
        pass
"#,
        "cannot iterate over",
    );
    assert_err(
        r#"
def f():
    for i in range(3):
        i = 1
"#,
        "immutable variable `i`",
    );
    assert_err(
        r#"
def f():
    for i in range(3):
        pass
    y = i
"#,
        "cannot find `i`",
    );
}

#[test]
fn arrays_and_indexing() {
    assert_clean(
        r#"
def f() -> list[int32]:
    return [1, 2, 3]
"#,
    );
    assert_clean(
        r#"
def f() -> list[float32]:
    return [1, 2.5]
"#,
    );
    assert_clean(
        r#"
def f() -> list[str]:
    return []
"#,
    );
    assert_err(
        r#"
def f():
    v = [1, "a"]
"#,
        "expected `{integer}`, found `str`",
    );
    assert_err(
        r#"
def f() -> list[str]:
    return [1]
"#,
        "expected `str`, found `{integer}`",
    );
    assert_clean(
        r#"
def f(v: list[int32], i: uint8) -> int32:
    return v[i] + [4, 5][1]
"#,
    );
    assert_err(
        r#"
def f(v: list[int32]) -> int32:
    return v["a"]
"#,
        "indices must be integers",
    );
    assert_err(
        r#"
def f(s: str) -> int32:
    return s[0]
"#,
        "cannot index into",
    );
    assert_err(
        r#"
def f(v: list[int32]) -> str:
    return v[0]
"#,
        "expected `str`, found `int32`",
    );
    assert_clean(
        r#"
def f(v: list[list[int32]]) -> int32:
    return v[0][1]
"#,
    );
}

// ---- literal type inference ----------------------------------------------------------------

#[test]
fn literals_are_pinned_by_their_uses() {
    // `s` starts as a bare literal and becomes u8 through its uses.
    assert_clean(
        r#"
def f(n: uint8) -> uint8:
    s = 0
    for i in range(1, n + 1):
        s += i
    return s
"#,
    );
    assert_clean(
        r#"
def f(n: int) -> int:
    i = 0
    s = 0
    while i < n:
        s += i
        i += 1
    return s
"#,
    );
    assert_clean(
        r#"
def f(x: float32) -> float32:
    a = 0.5
    a = a * x
    return a
"#,
    );
    assert_clean(
        r#"
def take(a: int):
    pass

def f():
    x = 5
    take(x)
"#,
    );
    assert_clean(
        r#"
def f() -> list[uint8]:
    v = [1, 2, 3]
    return v
"#,
    );
    assert_clean(
        r#"
def f(c: bool) -> int:
    a = 1
    b = 2
    if c:
        return a
    else:
        return b + 1000000000000
"#,
    );
    assert_clean(
        r#"
def f() -> int32:
    a = 1
    b = a
    c = b
    return c
"#,
    );
}

#[test]
fn conflicting_literal_uses_are_still_errors() {
    assert_err(
        r#"
def f(a: uint8, b: int):
    x = 1
    y = a + x
    z = b + x
"#,
        "mismatched operand types",
    );
    assert_err(
        r#"
def f():
    x = 1
    s: str = x
"#,
        "expected `str`, found `{integer}`",
    );
    assert_err(
        r#"
def f():
    x = 1.5
    y: int32 = x
"#,
        "expected `int32`, found `{float}`",
    );
    assert_err(
        r#"
def f(a: float):
    x = 1
    y = a + x
"#,
        "mismatched operand types",
    );
    assert_err(
        r#"
def f() -> bool:
    x = 1
    return x
"#,
        "expected `bool`",
    );
}

#[test]
fn unconstrained_literals_default_and_leave_no_variables_behind() {
    let (hir, errs) = check(
        r#"
def f() -> int32:
    a = 1
    b = 2.5
    v = [3, 4]
    return a + v[0]
"#,
    );
    assert!(errs.is_empty(), "{errs:?}");
    let dump = format!("{hir:?}");
    assert!(
        !dump.contains("IntVar") && !dump.contains("FloatVar"),
        "{dump}"
    );

    let (hir, errs) = check(
        r#"
def f():
    a = 1
    b = a + 2
    c = 1.5 * 2
    for i in range(10):
        d = i
"#,
    );
    assert!(errs.is_empty(), "{errs:?}");
    let dump = format!("{hir:?}");
    assert!(
        !dump.contains("IntVar") && !dump.contains("FloatVar"),
        "{dump}"
    );
}

#[test]
fn assignment_to_fields_and_elements() {
    let decl = r#"
class P:
    x: int32
    name: str
    tags: list[int32]

class W:
    p: P
    n: uint8
"#;
    assert_clean(&format!(
        r#"
{decl}
def f() -> int32:
    p = P(x=1, name="a", tags=[1, 2])
    p.x = 5
    p.x += 1
    p.tags[1] = p.x
    p.name = p.name + "!"
    return p.x
"#
    ));
    assert_clean(&format!(
        r#"
{decl}
def f(w: W) -> uint8:
    w.n = 3
    w.p.x = 1
    w.p.tags[0] += 2
    return w.n
"#
    ));
    assert_clean(
        r#"
def fill(v: list[int32], x: int32):
    v[0] = x
    v[1] = x * 2
"#,
    );
    assert_clean(
        r#"
def f():
    grid = [[1, 2], [3, 4]]
    grid[1][0] = 9
"#,
    );
    assert_err(
        &format!(
            r#"
{decl}
def f():
    p = P(x=1, name="a", tags=[])
    p.x = "s"
"#
        ),
        "expected `int32`, found `str`",
    );
    assert_err(
        &format!(
            r#"
{decl}
def f():
    p = P(x=1, name="a", tags=[])
    p.nope = 1
"#
        ),
        "no field `nope`",
    );
    assert_err(
        r#"
def f():
    x = 1
    x.y = 2
"#,
        "cannot assign to a field of",
    );
    assert_err(
        r#"
def f():
    x = 1
    x[0] = 2
"#,
        "cannot index into",
    );
    assert_err(
        r#"
def f():
    v = [1, 2]
    v["a"] = 1
"#,
        "indices must be integers",
    );
    assert_err(
        r#"
def f():
    v = [1, 2]
    v[0] = "s"
"#,
        "expected `{integer}`, found `str`",
    );
    assert_err(
        r#"
def f():
    v = [1, 2]
    f()[0] = 1
"#,
        "must be a variable, a field or an index",
    );
}

// ---- tensor operations ---------------------------------------------------------------------

#[test]
fn tensor_shapes_flow_through_operations() {
    let ok = r#"
def f(a: Tensor[float32, [2, 3]], b: Tensor[float32, [3, 4]], v: Tensor[float32, [3]]) -> Tensor[float32, [2, 4]]:
    return a |> std.tensor.matmul(b)

def g(a: Tensor[float32, [2, 3]], v: Tensor[float32, [3]]) -> Tensor[float32, [2]]:
    return a |> std.tensor.matmul(v)

def h(a: Tensor[float32, [4]], b: Tensor[float32, [4]]) -> float32:
    return std.tensor.dot(a, b)

def i(a: Tensor[float32, [4]], b: Tensor[float32, [4]]) -> Tensor[float32, [4]]:
    return std.tensor.add(a, b) |> std.tensor.relu() |> std.tensor.scale(0.5) |> std.tensor.softmax()

def t(a: Tensor[float32, [2, 3]]) -> Tensor[float32, [3, 2]]:
    return std.tensor.transpose(a)

def s(a: Tensor[float32, [2, 3]]) -> float32:
    return std.tensor.sum(a) + std.tensor.mean(a)

def am(a: Tensor[float32, [5]]) -> uint64:
    return std.tensor.argmax(a)

def sym(a: Tensor[float32, [N]], b: Tensor[float32, [8]]) -> Tensor[float32, [8]]:
    return std.tensor.mul(a, b)
"#;
    assert_clean(ok);
}

#[test]
fn tensor_shape_mismatches_are_compile_errors() {
    assert_err(
        r#"
def f(a: Tensor[float32, [2, 3]], b: Tensor[float32, [4, 5]]):
    std.tensor.matmul(a, b)
"#,
        "inner dimensions differ: 3 and 4",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [4]], b: Tensor[float32, [5]]):
    std.tensor.add(a, b)
"#,
        "tensor shapes differ: [4] and [5]",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [4]], b: Tensor[float, [4]]):
    std.tensor.add(a, b)
"#,
        "element types differ",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [4]]) -> Tensor[float32, [2]]:
    return std.tensor.softmax(a)
"#,
        "expected `Tensor[float32, [2]]`",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [2, 2]], b: Tensor[float32, [2, 2]]):
    std.tensor.dot(a, b)
"#,
        "rank-1",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [4]]):
    std.tensor.transpose(a)
"#,
        "rank-2",
    );
    assert_err(
        r#"
def f(a: int32):
    std.tensor.relu(a)
"#,
        "must be a `Tensor`",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [2]]):
    std.tensor.scale(a, "x")
"#,
        "must be a float",
    );
}

#[test]
fn tensor_constructors_need_a_known_result_type() {
    assert_clean(
        r#"
def f() -> Tensor[float32, [2, 3]]:
    return std.tensor.zeros()
"#,
    );
    assert_clean(
        r#"
def f(v: list[float32]) -> Tensor[float32, [3]]:
    return std.tensor.from_vec(v)
"#,
    );
    assert_clean(
        r#"
def f(a: Tensor[float32, [2, 3]]) -> Tensor[float32, [6]]:
    return std.tensor.reshape(a)
"#,
    );
    assert_err(
        r#"
def f():
    t = std.tensor.zeros()
"#,
        "annotate it",
    );
    assert_err(
        r#"
def f(a: Tensor[float32, [2, 3]]) -> Tensor[float32, [5]]:
    return std.tensor.reshape(a)
"#,
        "cannot reshape [2, 3] (6 elements) into [5] (5 elements)",
    );
    assert_clean(
        r#"
def f(a: Tensor[float32, [4]]) -> list[float32]:
    return std.tensor.to_vec(a)
"#,
    );
}

// ---- tables --------------------------------------------------------------------------------

const ROW: &str = r#"
class Row:
    id: int
    score: float32
    name: str
    ok: bool
"#;

#[test]
fn tables_have_typed_columns() {
    assert_clean(&format!(
        r#"
{ROW}
def f(t: Table[Row]) -> Tensor[float32, [N]]:
    return t.score

def g(t: Table[Row]) -> list[str]:
    return t.name

def h(t: Table[Row]) -> uint64:
    return t.len()

def i(t: Table[Row]) -> Row:
    return t[0]

def j(t: Table[Row]) -> float32:
    s = 0.0
    for r in t:
        s += r.score
    return s

def k(t: Table[Row]) -> Tensor[int, [N]]:
    return t.id
"#
    ));
    assert_err(
        &format!(
            r#"
{ROW}
def f(t: Table[Row]):
    t.nope
"#
        ),
        "no column `nope`",
    );
    assert_err(
        &format!(
            r#"
{ROW}
def f(t: Table[Row]) -> list[str]:
    return t.score
"#
        ),
        "expected `list[str]`",
    );
}

#[test]
fn table_schemas_must_be_flat_structs() {
    assert_err(
        r#"
class R:
    xs: list[int32]

def f(t: Table[R]):
    pass
"#,
        "table columns must be",
    );
    assert_err(
        r#"
def f(t: Table[int32]):
    pass
"#,
        "must be a class",
    );
    assert_err(
        r#"
def f(t: Table):
    pass
"#,
        "takes one type argument",
    );
    assert_err(
        r#"
class E:
    pass

def f(t: Table[E]):
    pass
"#,
        "no fields",
    );
}

#[test]
fn table_functions() {
    assert_clean(&format!(
        r#"
{ROW}
def a() -> Table[Row]:
    return std.table.read_csv("x.csv")

def b() -> Table[Row]:
    return std.table.read_parquet("x.parquet")

def c(t: Table[Row]) -> Table[Row]:
    return t |> std.table.sql("SELECT * FROM t WHERE score > 1")

def d(rows: list[Row]) -> Table[Row]:
    return std.table.from_rows(rows)

def e(t: Table[Row]) -> list[Row]:
    return std.table.to_rows(t)

def f(t: Table[Row]) -> Table[Row]:
    return t |> std.table.filter(lambda r: r.score > 0.5)

def g(t: Table[Row]):
    std.table.write_csv(t, "o.csv")
    std.table.write_parquet(t, "o.parquet")

def h(t: Table[Row]) -> uint64:
    return std.table.len(t)
"#
    ));
    assert_err(
        &format!(
            r#"
{ROW}
def f():
    t = std.table.read_csv("x")
"#
        ),
        "annotate the result",
    );
    assert_err(
        &format!(
            r#"
{ROW}
def f(t: Table[Row]) -> Table[Row]:
    return std.table.filter(t, lambda r: r.nope)
"#
        ),
        "no field `nope`",
    );
    assert_err(
        &format!(
            r#"
{ROW}
def f(t: Table[Row]):
    std.table.sql(t, 1)
"#
        ),
        "must be `str`",
    );
}

#[test]
fn tensors_can_be_indexed() {
    assert_clean(
        r#"
def f(t: Tensor[float32, [4]]) -> float32:
    return t[0]
"#,
    );
    assert_clean(
        r#"
def f(t: Tensor[float32, [2, 3]]) -> Tensor[float32, [3]]:
    return t[1]
"#,
    );
    assert_clean(
        r#"
def f(t: Tensor[float32, [2, 3]]) -> float32:
    return t[1][2]
"#,
    );
    assert_err(
        r#"
def f(t: Tensor[float32, [4]]) -> int32:
    return t[0]
"#,
        "expected `int32`, found `float32`",
    );
}

#[test]
fn std_registry_can_be_browsed() {
    use telvax_hir::stdlib::{ItemKind, children, methods_of};
    let top = children("std");
    assert!(top.contains(&("text".to_string(), ItemKind::Module)));
    assert!(top.contains(&("collect".to_string(), ItemKind::Function)));
    let text = children("std::text");
    assert!(
        text.iter()
            .any(|(n, k)| n == "trim" && *k == ItemKind::Function)
    );
    let http = children("std::http");
    assert!(
        http.contains(&("Status".to_string(), ItemKind::Type))
            || http.contains(&("Status".to_string(), ItemKind::Module))
    );
    assert!(methods_of("PgVectorClient").contains(&"with_vector"));
}

#[test]
fn distributed_map_needs_a_named_function() {
    assert_clean(
        r#"
def sq(x: int) -> int:
    return x * x

def f(v: list[int]) -> list[int]:
    return std.dist.map(v, sq)
"#,
    );
    assert_clean(
        r#"
def sq(x: int) -> int:
    return x * x

def f(v: list[int]) -> list[int]:
    return v |> std.dist.map(sq)
"#,
    );
    assert_err(
        r#"
def f(v: list[int]) -> list[int]:
    return std.dist.map(v, lambda x: x * 2)
"#,
        "needs the name of a top-level function",
    );
    assert_err(
        r#"
def sq(x: int) -> int:
    return x

def f(v: list[str]):
    std.dist.map(v, sq)
"#,
        "mismatched",
    );
    assert_clean(
        r#"
def d(x: int32) -> int32:
    return x

def f(v: list[int32]) -> list[int32]:
    return std.iter.map(v, d)
"#,
    );
}

// ---- enums and match -------------------------------------------------------------------------

const SHAPES: &str = r#"
enum Shape:
    Circle(float)
    Rect(float, float)
    Empty

def area(s: Shape) -> float:
    match s:
        case Shape.Circle(r):
            return r * r * 3.0
        case Shape.Rect(w, h):
            return w * h
        case Shape.Empty:
            return 0.0

def make() -> Shape:
    return Shape.Rect(2.0, 3.0)

def nothing() -> Shape:
    return Shape.Empty
"#;

#[test]
fn enums_and_exhaustive_match_are_clean() {
    assert_clean(SHAPES);
    let (hir, _) = check(SHAPES);
    assert_eq!(hir.enums.len(), 1);
    assert_eq!(hir.enums[0].variants[1].1.len(), 2);
}

#[test]
fn match_must_be_exhaustive() {
    assert_err(
        r#"
enum E:
    A
    B
    C

def f(e: E) -> int32:
    match e:
        case E.A:
            return 1
        case E.B:
            return 2
"#,
        "missing `E.C`",
    );
    assert_err(
        r#"
def f(n: int32) -> int32:
    match n:
        case 0:
            return 1
        case 1:
            return 2
"#,
        "add a `_` arm",
    );
    assert_err(
        r#"
def f(b: bool) -> int32:
    match b:
        case True:
            return 1
"#,
        "non-exhaustive",
    );
    assert_clean(
        r#"
def f(b: bool) -> int32:
    match b:
        case True:
            return 1
        case False:
            return 0
"#,
    );
    assert_clean(
        r#"
def f(n: int32) -> int32:
    match n:
        case 0:
            return 1
        case other:
            return other
"#,
    );
    assert_clean(
        r#"
enum E:
    A(bool)
    B

def f(e: E) -> int32:
    match e:
        case E.A(True):
            return 1
        case E.A(False):
            return 2
        case E.B:
            return 3
"#,
    );
    assert_err(
        r#"
enum E:
    A(bool)
    B

def f(e: E) -> int32:
    match e:
        case E.A(True):
            return 1
        case E.B:
            return 3
"#,
        "non-exhaustive",
    );
}

#[test]
fn match_on_results() {
    assert_clean(
        r#"
def f(r: Result[int32, str]) -> int32:
    match r:
        case Ok(v):
            return v
        case Err(_):
            return 0
"#,
    );
    assert_err(
        r#"
def f(r: Result[int32, str]) -> int32:
    match r:
        case Ok(v):
            return v
"#,
        "non-exhaustive",
    );
}

#[test]
fn enum_type_errors() {
    assert_err(
        r#"
enum E:
    A(int32)

def f() -> E:
    return E.A
"#,
        "takes 1 value",
    );
    assert_err(
        r#"
enum E:
    A

def f() -> E:
    return E.B
"#,
        "no variant `B`",
    );
    assert_err(
        r#"
enum E:
    A(int32)

def f() -> E:
    return E.A(True)
"#,
        "mismatched types",
    );
    assert_err(
        r#"
enum E:
    A

enum F:
    B

def f(e: E) -> int32:
    match e:
        case F.B:
            return 1
        case E.A:
            return 2
"#,
        "cannot match",
    );
    assert_err(
        r#"
def f(n: int32) -> int32:
    match n:
        case 0:
            return 1
        case _:
            return "x"
"#,
        "incompatible types",
    );
    assert_err(
        r#"
def f(n: int32) -> int32:
    match n:
        case "a":
            return 1
        case _:
            return 2
"#,
        "does not match",
    );
}

// ---- friendlier errors -----------------------------------------------------------------------

fn full(src: &str) -> Vec<(String, Vec<String>)> {
    let (ast, perrs) = telvax_parser::parse(src);
    assert!(perrs.is_empty(), "parse errors: {perrs:?}");
    lower(&ast.unwrap())
        .1
        .into_iter()
        .map(|d| (d.message, d.notes))
        .collect()
}

#[test]
fn misspelled_names_get_a_suggestion() {
    let e = full(
        r#"
def main() -> int32:
    total = 5
    return totl + 1
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `total`")),
        "{e:?}"
    );
    let e = full(
        r#"
def helper() -> int32:
    return 1

def main() -> int32:
    return helpr()
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `helper`")),
        "{e:?}"
    );
    let e = full(
        r#"
def main() -> int32:
    s: flot = 1.0
    return 1
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `float`")),
        "{e:?}"
    );
    let e = full(
        r#"
class P:
    width: int32

def f(p: P) -> int32:
    return p.widht
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `width`")),
        "{e:?}"
    );
    // Nothing close: no suggestion.
    let e = full(
        r#"
def main() -> int32:
    return zzzzzz
"#,
    );
    assert!(e[0].1.is_empty(), "{e:?}");
}

#[test]
fn unknown_methods_on_builtin_types_are_compile_errors() {
    let e = full(
        r#"
def f(v: list[int32]) -> uint64:
    return v.lenn()
"#,
    );
    assert!(
        e[0].0.contains("no method `lenn` on type `list[int32]`"),
        "{e:?}"
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `len`")),
        "{e:?}"
    );
    assert_err(
        r#"
def f(s: str) -> str:
    return s.shout()
"#,
        "no method `shout` on type `str`",
    );
    assert_clean(
        r#"
def f(v: list[int32], s: str) -> uint64:
    if s.is_empty():
        return v.len()
    else:
        return 0
"#,
    );
}

#[test]
fn a_missing_colon_is_reported_where_it_belongs() {
    let src = "def main() -> int32\n    return 1\n";
    let (_, errs) = telvax_parser::parse(src);
    let first = &errs[0];
    assert!(first.message.contains("expected `:`"), "{}", first.message);
    // The error points at the end of the line, where the colon is missing.
    assert_eq!(&src[first.span.start..first.span.start], "");
    assert_eq!(first.span.start, "def main() -> int32".len());
}

// ---- Option and Map ----------------------------------------------------------------------------

#[test]
fn option_and_map_are_typed() {
    assert_clean(
        r#"
def f(o: int32 | None) -> int32:
    match o:
        case Some(n):
            return n
        case None:
            return 0
"#,
    );
    assert_clean(
        r#"
def f() -> int32 | None:
    return Some(3)

def g() -> str | None:
    return None
"#,
    );
    assert_clean(
        r#"
def f(v: list[int]) -> int:
    return v.first().unwrap_or(0)
"#,
    );
    assert_clean(
        r#"
def f() -> dict[str, int]:
    m: dict[str, int] = std.map.new()
    m.insert("a", 1)
    return m
"#,
    );
    assert_clean(
        r#"
def f(m: dict[int, str], k: int) -> str:
    match m.get(k):
        case Some(s):
            return s
        case None:
            return ""
"#,
    );
}

#[test]
fn option_and_map_mistakes_are_reported() {
    assert_err(
        r#"
def f(o: int32 | None) -> int32:
    match o:
        case Some(n):
            return n
"#,
        "missing `None`",
    );
    assert_err(
        r#"
def f(o: int32 | None) -> int32:
    match o:
        case None:
            return 0
"#,
        "missing `Some`",
    );
    assert_err(
        r#"
def f() -> int32 | None:
    return Some("x")
"#,
        "mismatched types",
    );
    assert_err(
        r#"
def f(m: dict[float, int]) -> uint64:
    return m.len()
"#,
        "keys must be",
    );
    assert_err(
        r#"
def f(m: dict[str, int]) -> int:
    m.insert(1, 2)
    return 0
"#,
        "mismatched types",
    );
    assert_err(
        r#"
def f(o: int32 | None) -> int32:
    return o.unwarp()
"#,
        "no method `unwarp`",
    );
    assert_err(
        r#"
def f() -> dict[str, int]:
    return std.map.new(1)
"#,
        "expected 0 argument",
    );
    assert_err(
        r#"
def f(n: int32) -> int32:
    match n:
        case Some(x):
            return x
        case _:
            return 0
"#,
        "need an `Option`",
    );
}

#[test]
fn closures_cannot_assign_to_captured_variables() {
    assert_err(
        r#"
def f() -> int:
    n = 0
    def inc(d: int):
        n += d
    inc(1)
    return n
"#,
        "cannot assign to `n` from inside a closure",
    );
    // A closure's own variables and shared containers are fine.
    assert_clean(
        r#"
def f() -> int:
    def inc(d: int):
        t = 0
        t += d
        return t
    return inc(1)
"#,
    );
    assert_clean(
        r#"
def f(v: list[int]) -> int:
    def set(x: int):
        v[0] = x
    set(3)
    return v[0]
"#,
    );
}

// ---- friendlier syntax -------------------------------------------------------------------------

#[test]
fn prelude_names_work_without_std() {
    assert_clean(
        r#"
def main():
    print("hi")
    std.io.print("a")
"#,
    );
    // A function or variable of the same name wins over the prelude.
    assert_clean(
        r#"
def print(s: str) -> int32:
    return 1

def main() -> int32:
    return print("x")
"#,
    );
    assert_clean(
        r#"
class R:
    a: int

def f() -> Table[R]:
    t: Table[R] = read_csv("x.csv")
    return t
"#,
    );
}

#[test]
fn tensor_methods_and_operators_are_checked() {
    assert_clean(
        r#"
def f(x: Tensor[float32, [2, 3]], w: Tensor[float32, [3, 4]]) -> Tensor[float32, [2, 4]]:
    return x.matmul(w).relu().softmax()
"#,
    );
    assert_clean(
        r#"
def f(a: Tensor[float32, [4]], b: Tensor[float32, [4]]) -> Tensor[float32, [4]]:
    return a * 2.0 + b - a / b
"#,
    );
    assert_clean(
        r#"
def f(a: Tensor[float32, [4]]) -> float32:
    return a.sum()
"#,
    );
    // The shape rules are the ones of the function forms.
    assert_err(
        r#"
def f(a: Tensor[float32, [4]], b: Tensor[float32, [5]]) -> Tensor[float32, [4]]:
    return a + b
"#,
        "shape",
    );
    assert_err(
        r#"
def f(x: Tensor[float32, [2, 3]], w: Tensor[float32, [4, 2]]) -> Tensor[float32, [2, 2]]:
    return x.matmul(w)
"#,
        "inner dimensions",
    );
    let e = full(
        r#"
def f(a: Tensor[float32, [4]]) -> Tensor[float32, [4]]:
    return a + 1.0
"#,
    );
    assert!(e[0].0.contains("cannot apply this operator"), "{e:?}");
    assert!(e[0].1.iter().any(|n| n.contains("two tensors")), "{e:?}");
    let e = full(
        r#"
def f(a: Tensor[float32, [4]]) -> Tensor[float32, [4]]:
    return a / 2.0
"#,
    );
    assert!(e[0].1.iter().any(|n| n.contains("reciprocal")), "{e:?}");
}

#[test]
fn inline_row_types_replace_throwaway_structs() {
    let src = r#"
class Sale:
    id: int
    units: int

def f(t: Table[Sale]) -> Table[{units: int}]:
    return t.sql("SELECT units FROM t")

def g(t: Table[{units: int}]) -> uint64:
    return t.len()

def h(t: Table[Sale]) -> uint64:
    return g(f(t))
"#;
    assert_clean(src);
    let (hir, _) = check(src);
    // Both mentions of the same columns are one struct.
    assert_eq!(
        hir.structs
            .iter()
            .filter(|s| s.name.starts_with('{'))
            .count(),
        1
    );
    assert_err(
        r#"
def f(t: Table[{a: int, a: int}]):
    pass
"#,
        "declared twice",
    );
    assert_err(
        r#"
def f(t: Table[{a: list[int]}]):
    pass
"#,
        "column",
    );
    assert_err(
        r#"
def f(t: Table[{}]):
    pass
"#,
        "at least one column",
    );
    assert_err(
        r#"
class Sale:
    id: int

def f(t: Table[Sale]) -> Table[{id: str}]:
    return t.sql("SELECT id FROM t")

def g(t: Table[Sale]) -> Table[Sale]:
    return f(t)
"#,
        "mismatched types",
    );
}

#[test]
fn question_mark_is_checked_against_the_function_result() {
    assert_clean(
        r#"
def f(r: Result[int32, str]) -> Result[int32, str]:
    x = r?
    return Ok(x + 1)
"#,
    );
    assert_clean(
        r#"
def f(o: int32 | None) -> int32 | None:
    x = o?
    return Some(x + 1)
"#,
    );
    assert_err(
        r#"
def f(r: Result[int32, str]) -> int32:
    return r?
"#,
        "can only be used in a function that returns `Result`",
    );
    assert_err(
        r#"
def f(r: Result[int32, int]) -> Result[int32, str]:
    x = r?
    return Ok(x)
"#,
        "cannot turn the error",
    );
    assert_err(
        r#"
def f(o: int32 | None) -> Result[int32, str]:
    x = o?
    return Ok(x)
"#,
        "returns `Option`",
    );
    assert_err(
        r#"
def f(n: int32) -> Result[int32, str]:
    x = n?
    return Ok(x)
"#,
        "needs a `Result` or an `Option`",
    );
}

#[test]
fn tuples_are_typed() {
    assert_clean(
        r#"
def f() -> (int, str):
    return (1, "a")

def g() -> int:
    n, _ = f()
    return n + f().0
"#,
    );
    assert_clean(
        r#"
def f(p: (bool, int)) -> int:
    match p:
        case (True, n):
            return n
        case (False, _):
            return 0
"#,
    );
    assert_clean(
        r#"
def f(v: list[(int, int)]) -> int:
    t = 0
    for a, b in v:
        t += a * b
    return t
"#,
    );
    assert_err(
        r#"
def f(p: (int, str)) -> int:
    return p.2
"#,
        "has no part `2`",
    );
    assert_err(
        r#"
def f() -> (int, str):
    return (1, 2)
"#,
        "mismatched types",
    );
    assert_err(
        r#"
def f(p: (bool, int)) -> int:
    match p:
        case (True, n):
            return n
"#,
        "non-exhaustive",
    );
    assert_err(
        r#"
def f(p: (int, int)) -> int:
    a, b = p
    return a + c
"#,
        "cannot find `c`",
    );
}

#[test]
fn iterator_methods_are_typed() {
    assert_clean(
        r#"
def f(v: list[int]) -> list[str]:
    return v.map(lambda x: f"{x}").filter(lambda s: s.len() > 0)
"#,
    );
    assert_clean(
        r#"
def f(v: list[int]) -> int:
    return v.fold(0, lambda acc, x: acc + x)
"#,
    );
    assert_clean(
        r#"
def f(v: list[int]) -> list[(int, int)]:
    return v.enumerate()
"#,
    );
    assert_clean(
        r#"
def f(v: list[int]) -> int | None:
    return v.max()
"#,
    );
    assert_err(
        r#"
def f(v: list[str]) -> int:
    return v.sum()
"#,
        "needs a vector of numbers",
    );
    assert_err(
        r#"
def f(v: list[int]) -> bool:
    return v.any(lambda x: x + 1)
"#,
        "bool",
    );
}

#[test]
fn impl_blocks_are_checked() {
    let src = r#"
class P:
    x: int

    def get(self) -> int:
        return self.x

    def make(x: int) -> P:
        return P(x=x)

    def add(self, n: int) -> P:
        return Self(x=self.x + n)

def f() -> int:
    return P.make(1).add(2).get()
"#;
    assert_clean(src);
    let (hir, _) = check(src);
    assert!(
        hir.fns.iter().any(|f| f.name == "P::get"),
        "methods are functions named Type::name"
    );
    assert_err(
        r#"
class P:
    x: int

    def get(self) -> int:
        return self.y
"#,
        "no field `y`",
    );
    assert_err(
        r#"
extend Nothing:
    def f():
        pass
"#,
        "cannot find the class or enum `Nothing`",
    );
    assert_err(
        r#"
class P:
    x: int

    def make() -> P:
        return P(x=1)

def f(p: P) -> P:
    return p.make()
"#,
        "has no `self` parameter",
    );
    assert_err(
        r#"
class P:
    x: int

    def get(self) -> int:
        return self.x

def f(p: P) -> int:
    return p.get(1)
"#,
        "expected 0 argument",
    );
    assert_err(
        r#"
def f() -> Self:
    return 1
"#,
        "only be used inside a class",
    );
    assert_err(
        r#"
class P:
    x: int

    def a(self):
        pass

    def a(self):
        pass
"#,
        "defined twice",
    );
}

#[test]
fn traits_are_checked() {
    let base = r#"
trait Shape:
    def area(self) -> float: ...

    def name(self) -> str:
        return "shape"

class Sq:
    s: float
"#;
    assert_clean(&format!(
        r#"
{base}
extend Sq(Shape):
    def area(self) -> float:
        return self.s

def total(v: list[Shape]) -> float:
    return v.map(lambda x: x.area()).sum()

def make() -> list[Shape]:
    return [Sq(s=1.0)]
"#
    ));
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    def name(self) -> str:
        return "sq"
"#
        ),
        "is missing the method `area`",
    );
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    def area(self) -> int:
        return 1
"#
        ),
        "has the signature",
    );
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    def area(self) -> float:
        return 1.0

    def extra(self):
        pass
"#
        ),
        "is not a method of the trait `Shape`",
    );
    assert_err(
        r#"
class S(Missing):
    def f(self):
        pass
"#,
        "cannot find the trait `Missing`",
    );
    assert_err(
        &format!(
            r#"
{base}
class Other:
    x: int

def f() -> Shape:
    return Other(x=1)
"#
        ),
        "mismatched types",
    );
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    def area(self) -> float:
        return 1.0

def f(s: Shape) -> float:
    return s.volume()
"#
        ),
        "has no method `volume`",
    );
    assert_err("def f() -> int32: ...\n", "needs a body");
}

#[test]
fn generics_are_checked_and_instantiated() {
    let base = r#"
trait Named:
    def name(self) -> str: ...

class Dog(Named):
    n: str

    def name(self) -> str:
        return self.n

def id[T](x: T) -> T:
    return x

def greet[T: Named](x: T) -> str:
    return x.name()

class Boxed[T]:
    v: T

enum Maybe[T]:
    Nothing
    Just(T)
"#;
    assert_clean(&format!(
        r#"
{base}
def a() -> str:
    return greet(Dog(n="r"))

def b() -> str:
    m: Maybe[str] = Maybe.Just(id("x"))
    match m:
        case Maybe.Just(s):
            return s
        case Maybe.Nothing:
            return ""

def c() -> Boxed[str]:
    return Boxed(v="s")
"#
    ));
    assert_err(
        &format!(
            r#"
{base}
def a() -> int32:
    return greet(3)
"#
        ),
        "does not implement `Named`",
    );
    assert_err(
        &format!(
            r#"
{base}
def a() -> int32:
    f = id
    return 0
"#
        ),
        "has to be called",
    );
    assert_err(
        &format!(
            r#"
{base}
def a() -> int32:
    m = Maybe.Nothing
    return 0
"#
        ),
        "cannot infer the type parameter `T`",
    );
    assert_err(
        &format!(
            r#"
{base}
def a(b: Boxed[int32, int32]) -> int32:
    return 0
"#
        ),
        "takes 1 type argument",
    );
}

#[test]
fn lowercase_constants_get_a_hint() {
    let e = {
        let (ast, _) = telvax_parser::parse("def f() -> bool:\n    return true\n");
        let (_, diags) = lower(&ast.unwrap());
        diags
    };
    assert!(
        e.iter()
            .any(|d| d.notes.iter().any(|n| n.contains("did you mean `True`"))),
        "{e:?}"
    );
}
