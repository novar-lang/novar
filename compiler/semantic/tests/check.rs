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
fn inc(x: i32) -> i32:
    x

fn f(a: i32) -> i32:
    a |> inc()
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
fn inc(x: i32) -> i32:
    x

fn f(s: Text) -> i32:
    s |> inc()
"#,
        "expected `i32`, found `Text`",
    );
    assert_err(
        r#"
fn f(a: i32) -> i32:
    a |> 3
"#,
        "right side of `|>`",
    );
}

#[test]
fn tensor_shapes_are_enforced() {
    let ok = r#"
fn take(x: Tensor[f32, 4]):
    pass

fn f(y: Tensor[f32, 4]):
    take(y)
"#;
    assert_clean(ok);
    assert_err(
        r#"
fn take(x: Tensor[f32, 4]):
    pass

fn f(y: Tensor[f32, 8]):
    take(y)
"#,
        "expected `Tensor[f32, 4]`, found `Tensor[f32, 8]`",
    );
    assert_err(
        r#"
fn take(x: Tensor[f32, 4]):
    pass

fn f(y: Tensor[f32, 4, 4]):
    take(y)
"#,
        "mismatched types",
    );
    assert_err(
        r#"
fn take(x: Tensor[f32, 4]):
    pass

fn f(y: Tensor[i32, 4]):
    take(y)
"#,
        "mismatched types",
    );
    // Symbolic dimensions match any size.
    assert_clean(
        r#"
fn take(x: Tensor[f32, 4]):
    pass

fn f(y: Tensor[f32, N]):
    take(y)
"#,
    );
}

#[test]
fn invalid_tensor_types() {
    assert_err(
        r#"
fn f(x: Tensor[Text, 4]):
    pass
"#,
        "numeric primitive",
    );
    assert_err(
        r#"
fn f(x: Tensor[f32]):
    pass
"#,
        "`Tensor` expects",
    );
    assert_err(
        r#"
fn f(x: Tensor[f32, 0]):
    pass
"#,
        "greater than zero",
    );
}

#[test]
fn name_resolution() {
    assert_err(
        r#"
fn f() -> i32:
    missing
"#,
        "cannot find `missing`",
    );
    assert_err(
        r#"
fn f(x: Nope):
    pass
"#,
        "cannot find type `Nope`",
    );
    assert_err(
        r#"
type A:
    pass

fn f():
    A
"#,
        "is a type, not a value",
    );
    assert_err(
        r#"
fn f():
    pass

fn f():
    pass
"#,
        "defined twice",
    );
    assert_err(
        r#"
fn f(a: i32, a: i32):
    pass
"#,
        "declared twice",
    );
    assert_clean(
        r#"
import std.ai.llm.OpenAIClient

let c = OpenAIClient.new("m")

fn f():
    let x = c
"#,
    );
    // A binding is not visible inside its own initializer.
    assert_err(
        r#"
fn f():
    let x = x
"#,
        "cannot find `x`",
    );
}

#[test]
fn literals_adapt_to_context() {
    assert_clean(
        r#"
fn f(a: f32) -> f32:
    a + 1
"#,
    );
    assert_clean(
        r#"
fn f():
    let x: f32 = 1.5
    let y: u8 = 3
    let z: Float = 2
"#,
    );
    assert_err(
        r#"
fn f():
    let x: i32 = 1.5
"#,
        "expected `i32`, found `{float}`",
    );
    assert_err(
        r#"
fn f():
    let x: i32 = "a"
"#,
        "expected `i32`, found `Text`",
    );
}

#[test]
fn calls() {
    assert_err(
        r#"
fn g(a: i32):
    pass

fn f():
    g(1, 2)
"#,
        "expected 1 argument(s), found 2",
    );
    assert_err(
        r#"
fn g(a: i32):
    pass

fn f():
    g(True)
"#,
        "expected `i32`, found `Bool`",
    );
    assert_err(
        r#"
fn f(a: i32):
    a()
"#,
        "expected a function",
    );
}

#[test]
fn structs() {
    let decl = r#"
type P:
    x: i32
    name: Text
"#;
    assert_clean(&format!(
        r#"
{decl}
fn f() -> P:
    P(x: 1, name: "a")
"#
    ));
    assert_err(
        &format!(
            r#"
{decl}
fn f():
    P(x: 1)
"#
        ),
        "missing field(s) in `P`: `name`",
    );
    assert_err(
        &format!(
            r#"
{decl}
fn f():
    P(x: 1, name: "a", y: 2)
"#
        ),
        "no field `y`",
    );
    assert_err(
        &format!(
            r#"
{decl}
fn f(p: P) -> i32:
    p.nope
"#
        ),
        "no field `nope`",
    );
    assert_err(
        &format!(
            r#"
{decl}
fn f():
    P(x: "s", name: "a")
"#
        ),
        "expected `i32`",
    );
    assert_err(
        r#"
type A:
    x: i32
    x: i32
"#,
        "declared twice",
    );
    assert_err(
        r#"
fn f():
    Q(a: 1)
"#,
        "cannot find type `Q`",
    );
}

#[test]
fn returns_and_blocks() {
    assert_clean(
        r#"
fn f() -> i32:
    1
"#,
    );
    assert_clean(
        r#"
fn f(c: Bool) -> i32:
    if c:
        return 1
    2
"#,
    );
    assert_err(
        r#"
fn f() -> i32:
    let x = 1
"#,
        "expected `i32`, found `None`",
    );
    assert_err(
        r#"
fn f() -> i32:
    return
"#,
        "needs a value",
    );
    assert_err(
        r#"
fn f() -> i32:
    "s"
"#,
        "expected `i32`, found `Text`",
    );
    assert_clean(
        r#"
fn f():
    return
"#,
    );
}

#[test]
fn conditions_and_operators() {
    assert_err(
        r#"
fn f():
    if 1:
        pass
"#,
        "expected `Bool`, found `{integer}`",
    );
    assert_err(
        r#"
fn f(c: Bool) -> i32:
    if c:
        1
    else:
        "s"
"#,
        "incompatible types",
    );
    assert_clean(
        r#"
fn f(c: Bool) -> i32:
    if c:
        1
    else:
        2
"#,
    );
    assert_err(
        r#"
fn f() -> i32:
    1 + "a"
"#,
        "mismatched operand types",
    );
    assert_err(
        r#"
fn f() -> Bool:
    True and 1
"#,
        "expected `Bool`",
    );
    assert_err(
        r#"
fn f() -> Bool:
    True < False
"#,
        "cannot order",
    );
    assert_clean(
        r#"
fn f(a: Text) -> Text:
    a + "x"
"#,
    );
    assert_err(
        r#"
fn f():
    let x = -"a"
"#,
        "cannot apply",
    );
}

#[test]
fn result_constructors_follow_the_declared_type() {
    assert_clean(
        r#"
fn f() -> Result[i32, Text]:
    Ok(1)
"#,
    );
    assert_clean(
        r#"
fn f() -> Result[i32, Text]:
    Err("bad")
"#,
    );
    assert_err(
        r#"
fn f() -> Result[i32, Text]:
    Ok("x")
"#,
        "expected `i32`, found `Text`",
    );
    assert_err(
        r#"
fn f() -> Result[i32, Text]:
    Err(1)
"#,
        "expected `Text`, found `{integer}`",
    );
    assert_err(
        r#"
fn f() -> Result[i32, Text]:
    Ok(1, 2)
"#,
        "exactly one argument",
    );
}

#[test]
fn string_and_vec_methods() {
    assert_clean(
        r#"
fn f(s: Text) -> Bool:
    s.is_empty()
"#,
    );
    assert_err(
        r#"
fn f(s: Text) -> i32:
    s.is_empty()
"#,
        "expected `i32`, found `Bool`",
    );
    assert_clean(
        r#"
fn f(v: List[i32]) -> u64:
    v.len()
"#,
    );
}

#[test]
fn endpoints() {
    let ok = r#"
type Q:
    a: i32

@endpoint(POST, "/x")
fn h(q: Q) -> Result[Q, Text]:
    Ok(q)
"#;
    assert_clean(ok);
    let get = r#"
@endpoint(path: "/ping")
fn h() -> Result[Text, Text]:
    Ok("pong")
"#;
    let (hir, errs) = check(get);
    assert!(errs.is_empty(), "{errs:?}");
    assert_eq!(
        hir.fns[0].endpoint.as_ref().unwrap().method,
        hir::HttpMethod::Get
    );

    assert_err(
        r#"
@endpoint(method: "GET")
fn h() -> Result[i32, i32]:
    Ok(1)
"#,
        "requires a `path`",
    );
    assert_err(
        r#"
@endpoint(path: "x")
fn h() -> Result[i32, i32]:
    Ok(1)
"#,
        "must start with `/`",
    );
    assert_err(
        r#"
@endpoint(FETCH, "/x")
fn h() -> Result[i32, i32]:
    Ok(1)
"#,
        "unsupported HTTP method",
    );
    assert_err(
        r#"
@endpoint(POST, "/x")
fn h() -> Result[i32, i32]:
    Ok(1)
"#,
        "payload parameter",
    );
    assert_err(
        r#"
@endpoint(path: "/x")
fn h(a: i32) -> Result[i32, i32]:
    Ok(1)
"#,
        "must be a `type`",
    );
    assert_err(
        r#"
@endpoint(path: "/x")
fn h() -> i32:
    1
"#,
        "must return `Result",
    );
    assert_err(
        r#"
@endpoint(path: "/x", color: "r")
fn h() -> Result[i32, i32]:
    Ok(1)
"#,
        "unknown `@endpoint` argument",
    );
    assert_err(
        r#"
@route
fn h():
    pass
"#,
        "unknown attribute",
    );
    assert_err(
        r#"
@endpoint(path: "/x")
fn a() -> Result[i32, i32]:
    Ok(1)

@endpoint(path: "/x")
fn b() -> Result[i32, i32]:
    Ok(1)
"#,
        "duplicate endpoint",
    );
}

#[test]
fn closures_and_extern_calls_stay_lenient() {
    assert_clean(
        r#"
import std.iter.map

fn f(v: List):
    let r = v |> std.iter.map(d => d.text) |> std.collect()
"#,
    );
    assert_clean(
        r#"
fn f():
    let g = (a, b) => a + b
"#,
    );
}

#[test]
fn diagnostics_render_with_source_context() {
    let src = r#"
fn f() -> i32:
    "oops"
"#;
    let (ast, _) = telvax_parser::parse(src);
    let (_, diags) = lower(&ast.unwrap());
    let out = telvax_diagnostics::render("main.tvx", src, &diags, false);
    assert!(out.contains("main.tvx:3:5"), "{out}");
    assert!(out.contains("expected `i32`, found `Text`"), "{out}");
}

// ---- standard library ----------------------------------------------------------------------

#[test]
fn std_functions_are_typed() {
    assert_clean(
        r#"
fn f(s: Text) -> Text:
    s |> std.text.clean_whitespace() |> std.text.truncate(10)
"#,
    );
    assert_err(
        r#"
fn f(n: i32) -> Text:
    std.text.trim(n)
"#,
        "argument 1 must be `Text`, found `i32`",
    );
    assert_err(
        r#"
fn f(s: Text) -> Text:
    std.text.truncate(s)
"#,
        "expected 2 argument(s), found 1",
    );
    assert_err(
        r#"
fn f(s: Text) -> Text:
    std.text.truncate(s, "x")
"#,
        "must be an integer",
    );
    assert_err(
        r#"
fn f(s: Text) -> i32:
    std.text.trim(s)
"#,
        "expected `i32`, found `Text`",
    );
}

#[test]
fn std_unknown_items_are_errors() {
    assert_err(
        r#"
fn f():
    std.text.nope()
"#,
        "cannot find `std.text.nope`",
    );
    assert_err(
        r#"
import std.nope.Thing

fn f():
    pass
"#,
        "unresolved import",
    );
    assert_err(
        r#"
import std.text

fn f():
    text.nope()
"#,
        "cannot find `std.text.nope`",
    );
    assert_err(
        r#"
fn f():
    let x = std.text
"#,
        "is not a value",
    );
}

#[test]
fn iterators_propagate_element_types() {
    assert_clean(
        r#"
type D:
    text: Text

fn f(v: List[D]) -> Text:
    v |> std.iter.map(d => d.text) |> std.string.join(", ")
"#,
    );
    // The closure parameter is `D`, so a missing field is caught inside it.
    assert_err(
        r#"
type D:
    text: Text

fn f(v: List[D]):
    v |> std.iter.map(d => d.nope)
"#,
        "no field `nope`",
    );
    assert_err(
        r#"
fn f(v: List[Text]) -> List[i32]:
    v |> std.iter.map(s => s)
"#,
        "expected `List[i32]`, found `List[Text]`",
    );
    assert_err(
        r#"
fn f(n: i32):
    std.iter.map(n, x => x)
"#,
        "must be a `List`",
    );
    assert_err(
        r#"
fn f(v: List[i32]):
    v |> std.iter.filter(x => 1)
"#,
        "must be `(_) -> Bool`",
    );
    assert_clean(
        r#"
fn f(v: List[i32]) -> u64:
    v |> std.iter.take(3) |> std.iter.count()
"#,
    );
}

#[test]
fn std_types_have_typed_methods() {
    let head = r#"
import std.ai.embeddings.OllamaEmbedder
import std.ai.llm.OpenAIClient

let e = OllamaEmbedder.new("nomic-embed-text")
let l = OpenAIClient.new("m")
"#;
    assert_clean(&format!(
        r#"
{head}
fn f(t: Text) -> Tensor[f32, 768]:
    t |> e.embed()
"#
    ));
    assert_err(
        &format!(
            r#"
{head}
fn f(t: Text) -> Tensor[f32, 512]:
    t |> e.embed()
"#
        ),
        "expected `Tensor[f32, 512]`, found `Tensor[f32, 768]`",
    );
    assert_err(
        &format!(
            r#"
{head}
fn f(t: Text) -> Text:
    l.nope(t)
"#
        ),
        "no method `nope` on type `OpenAIClient`",
    );
    assert_err(
        &format!(
            r#"
{head}
fn f() -> Text:
    l.complete(1)
"#
        ),
        "must be `Text`",
    );
}

#[test]
fn vector_query_pipeline() {
    let head = r#"
import std.db.pgvector.PgVectorClient

let db = PgVectorClient.connect("postgresql://x")
"#;
    assert_clean(&format!(
        r#"
{head}
fn f(v: Tensor[f32, 768]) -> List[Text]:
    db.search_collection("kb") |> db.with_vector(v) |> db.limit(3) |> db.execute()
"#
    ));
    assert_err(
        &format!(
            r#"
{head}
fn f(v: Tensor[f32, 8]):
    db.search_collection("kb") |> db.with_vector(v)
"#
        ),
        "must be `Tensor[f32, 768]`",
    );
    assert_err(
        &format!(
            r#"
{head}
fn f():
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

fn f(ok: Bool) -> Result[Text, Status]:
    if ok:
        return Ok("y")
    Err(Status.NotFound("nope"))
"#;
    assert_clean(src);
    assert_err(
        r#"
import std.http.Status

fn f() -> Result[Text, Status]:
    Err(Status.Teapot("x"))
"#,
        "cannot find `std.http.Status.Teapot`",
    );
}

// ---- mutation, loops, arrays ---------------------------------------------------------------

#[test]
fn mutable_locals() {
    assert_clean(
        r#"
fn f() -> i32:
    var x = 1
    x = x + 1
    x += 3
    x
"#,
    );
    assert_err(
        r#"
fn f():
    var x = 1
    x = "s"
"#,
        "expected `{integer}`, found `Text`",
    );
    assert_err(
        r#"
let g = 1

fn f():
    g = 2
"#,
        "cannot assign to the global `g`",
    );
    assert_err(
        r#"
fn f(p: i32):
    let x = 1
    1 + 2 = 3
"#,
        "must be a variable",
    );
}

#[test]
fn while_loops() {
    assert_clean(
        r#"
fn f(n: i32) -> i32:
    var i = 0
    var s = 0
    while i < n:
        s += i
        i += 1
    s
"#,
    );
    assert_err(
        r#"
fn f():
    while 1:
        pass
"#,
        "expected `Bool`, found `{integer}`",
    );
    assert_clean(
        r#"
fn f(n: i32) -> i32:
    var i = 0
    while True:
        if i > n:
            return i
        i += 1
    0
"#,
    );
}

#[test]
fn for_loops() {
    assert_clean(
        r#"
fn f(n: i32) -> i32:
    var s = 0
    for i in range(n):
        s += i
    s
"#,
    );
    assert_clean(
        r#"
fn f(n: u8) -> u8:
    var s: u8 = 0
    for i in range(1, n + 1):
        s += i
    s
"#,
    );
    assert_clean(
        r#"
fn f(v: List[Float]) -> Float:
    var s = 0.0
    for x in v:
        s += x
    s
"#,
    );
    assert_err(
        r#"
fn f():
    for i in range(1.5):
        pass
"#,
        "range bounds",
    );
    assert_err(
        r#"
fn f(a: i32, b: Int):
    for i in range(a, b):
        pass
"#,
        "different types",
    );
    assert_err(
        r#"
fn f(s: Text):
    for c in s:
        pass
"#,
        "cannot iterate over",
    );
    assert_err(
        r#"
fn f():
    for i in range(3):
        i = 1
"#,
        "immutable variable `i`",
    );
    assert_err(
        r#"
fn f():
    for i in range(3):
        pass
    let y = i
"#,
        "cannot find `i`",
    );
}

#[test]
fn arrays_and_indexing() {
    assert_clean(
        r#"
fn f() -> List[i32]:
    [1, 2, 3]
"#,
    );
    assert_clean(
        r#"
fn f() -> List[f32]:
    [1, 2.5]
"#,
    );
    assert_clean(
        r#"
fn f() -> List[Text]:
    []
"#,
    );
    assert_err(
        r#"
fn f():
    let v = [1, "a"]
"#,
        "expected `{integer}`, found `Text`",
    );
    assert_err(
        r#"
fn f() -> List[Text]:
    [1]
"#,
        "expected `Text`, found `{integer}`",
    );
    assert_clean(
        r#"
fn f(v: List[i32], i: u8) -> i32:
    v[i] + [4, 5][1]
"#,
    );
    assert_err(
        r#"
fn f(v: List[i32]) -> i32:
    v["a"]
"#,
        "indices must be integers",
    );
    assert_err(
        r#"
fn f(s: Text) -> i32:
    s[0]
"#,
        "cannot index into",
    );
    assert_err(
        r#"
fn f(v: List[i32]) -> Text:
    v[0]
"#,
        "expected `Text`, found `i32`",
    );
    assert_clean(
        r#"
fn f(v: List[List[i32]]) -> i32:
    v[0][1]
"#,
    );
}

// ---- literal type inference ----------------------------------------------------------------

#[test]
fn literals_are_pinned_by_their_uses() {
    // `s` starts as a bare literal and becomes u8 through its uses.
    assert_clean(
        r#"
fn f(n: u8) -> u8:
    var s = 0
    for i in range(1, n + 1):
        s += i
    s
"#,
    );
    assert_clean(
        r#"
fn f(n: Int) -> Int:
    var i = 0
    var s = 0
    while i < n:
        s += i
        i += 1
    s
"#,
    );
    assert_clean(
        r#"
fn f(x: f32) -> f32:
    var a = 0.5
    a = a * x
    a
"#,
    );
    assert_clean(
        r#"
fn take(a: Int):
    pass

fn f():
    let x = 5
    take(x)
"#,
    );
    assert_clean(
        r#"
fn f() -> List[u8]:
    let v = [1, 2, 3]
    v
"#,
    );
    assert_clean(
        r#"
fn f(c: Bool) -> Int:
    let a = 1
    let b = 2
    if c:
        a
    else:
        b + 1000000000000
"#,
    );
    assert_clean(
        r#"
fn f() -> i32:
    let a = 1
    let b = a
    let c = b
    c
"#,
    );
}

#[test]
fn conflicting_literal_uses_are_still_errors() {
    assert_err(
        r#"
fn f(a: u8, b: Int):
    let x = 1
    let y = a + x
    let z = b + x
"#,
        "mismatched operand types",
    );
    assert_err(
        r#"
fn f():
    let x = 1
    let s: Text = x
"#,
        "expected `Text`, found `{integer}`",
    );
    assert_err(
        r#"
fn f():
    let x = 1.5
    let y: i32 = x
"#,
        "expected `i32`, found `{float}`",
    );
    assert_err(
        r#"
fn f(a: Float):
    let x = 1
    let y = a + x
"#,
        "mismatched operand types",
    );
    assert_err(
        r#"
fn f() -> Bool:
    let x = 1
    x
"#,
        "expected `Bool`",
    );
}

#[test]
fn unconstrained_literals_default_and_leave_no_variables_behind() {
    let (hir, errs) = check(
        r#"
fn f() -> i32:
    let a = 1
    let b = 2.5
    let v = [3, 4]
    a + v[0]
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
fn f():
    let a = 1
    let b = a + 2
    let c = 1.5 * 2
    for i in range(10):
        let d = i
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
type P:
    x: i32
    name: Text
    tags: List[i32]

type W:
    p: P
    n: u8
"#;
    assert_clean(&format!(
        r#"
{decl}
fn f() -> i32:
    var p = P(x: 1, name: "a", tags: [1, 2])
    p.x = 5
    p.x += 1
    p.tags[1] = p.x
    p.name = p.name + "!"
    p.x
"#
    ));
    assert_clean(&format!(
        r#"
{decl}
fn f(w: W) -> u8:
    w.n = 3
    w.p.x = 1
    w.p.tags[0] += 2
    w.n
"#
    ));
    assert_clean(
        r#"
fn fill(v: List[i32], x: i32):
    v[0] = x
    v[1] = x * 2
"#,
    );
    assert_clean(
        r#"
fn f():
    var grid = [[1, 2], [3, 4]]
    grid[1][0] = 9
"#,
    );
    assert_err(
        &format!(
            r#"
{decl}
fn f():
    var p = P(x: 1, name: "a", tags: [])
    p.x = "s"
"#
        ),
        "expected `i32`, found `Text`",
    );
    assert_err(
        &format!(
            r#"
{decl}
fn f():
    var p = P(x: 1, name: "a", tags: [])
    p.nope = 1
"#
        ),
        "no field `nope`",
    );
    assert_err(
        r#"
fn f():
    var x = 1
    x.y = 2
"#,
        "cannot assign to a field of",
    );
    assert_err(
        r#"
fn f():
    var x = 1
    x[0] = 2
"#,
        "cannot index into",
    );
    assert_err(
        r#"
fn f():
    var v = [1, 2]
    v["a"] = 1
"#,
        "indices must be integers",
    );
    assert_err(
        r#"
fn f():
    var v = [1, 2]
    v[0] = "s"
"#,
        "expected `{integer}`, found `Text`",
    );
    assert_err(
        r#"
fn f():
    let v = [1, 2]
    f()[0] = 1
"#,
        "must be a variable, a field or an index",
    );
}

// ---- tensor operations ---------------------------------------------------------------------

#[test]
fn tensor_shapes_flow_through_operations() {
    let ok = r#"
fn f(a: Tensor[f32, 2, 3], b: Tensor[f32, 3, 4], v: Tensor[f32, 3]) -> Tensor[f32, 2, 4]:
    a |> std.tensor.matmul(b)

fn g(a: Tensor[f32, 2, 3], v: Tensor[f32, 3]) -> Tensor[f32, 2]:
    a |> std.tensor.matmul(v)

fn h(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
    std.tensor.dot(a, b)

fn i(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
    std.tensor.add(a, b) |> std.tensor.relu() |> std.tensor.scale(0.5) |> std.tensor.softmax()

fn t(a: Tensor[f32, 2, 3]) -> Tensor[f32, 3, 2]:
    std.tensor.transpose(a)

fn s(a: Tensor[f32, 2, 3]) -> f32:
    std.tensor.sum(a) + std.tensor.mean(a)

fn am(a: Tensor[f32, 5]) -> u64:
    std.tensor.argmax(a)

fn sym(a: Tensor[f32, N], b: Tensor[f32, 8]) -> Tensor[f32, 8]:
    std.tensor.mul(a, b)
"#;
    assert_clean(ok);
}

#[test]
fn tensor_shape_mismatches_are_compile_errors() {
    assert_err(
        r#"
fn f(a: Tensor[f32, 2, 3], b: Tensor[f32, 4, 5]):
    std.tensor.matmul(a, b)
"#,
        "inner dimensions differ: 3 and 4",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 4], b: Tensor[f32, 5]):
    std.tensor.add(a, b)
"#,
        "tensor shapes differ: [4] and [5]",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 4], b: Tensor[Float, 4]):
    std.tensor.add(a, b)
"#,
        "element types differ",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 4]) -> Tensor[f32, 2]:
    std.tensor.softmax(a)
"#,
        "expected `Tensor[f32, 2]`",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 2, 2], b: Tensor[f32, 2, 2]):
    std.tensor.dot(a, b)
"#,
        "rank-1",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 4]):
    std.tensor.transpose(a)
"#,
        "rank-2",
    );
    assert_err(
        r#"
fn f(a: i32):
    std.tensor.relu(a)
"#,
        "must be a `Tensor`",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 2]):
    std.tensor.scale(a, "x")
"#,
        "must be a float",
    );
}

#[test]
fn tensor_constructors_need_a_known_result_type() {
    assert_clean(
        r#"
fn f() -> Tensor[f32, 2, 3]:
    std.tensor.zeros()
"#,
    );
    assert_clean(
        r#"
fn f(v: List[f32]) -> Tensor[f32, 3]:
    std.tensor.from_vec(v)
"#,
    );
    assert_clean(
        r#"
fn f(a: Tensor[f32, 2, 3]) -> Tensor[f32, 6]:
    std.tensor.reshape(a)
"#,
    );
    assert_err(
        r#"
fn f():
    let t = std.tensor.zeros()
"#,
        "annotate it",
    );
    assert_err(
        r#"
fn f(a: Tensor[f32, 2, 3]) -> Tensor[f32, 5]:
    std.tensor.reshape(a)
"#,
        "cannot reshape [2, 3] (6 elements) into [5] (5 elements)",
    );
    assert_clean(
        r#"
fn f(a: Tensor[f32, 4]) -> List[f32]:
    std.tensor.to_vec(a)
"#,
    );
}

// ---- tables --------------------------------------------------------------------------------

const ROW: &str = r#"
type Row:
    id: Int
    score: f32
    name: Text
    ok: Bool
"#;

#[test]
fn tables_have_typed_columns() {
    assert_clean(&format!(
        r#"
{ROW}
fn f(t: Table[Row]) -> Tensor[f32, N]:
    t.score

fn g(t: Table[Row]) -> List[Text]:
    t.name

fn h(t: Table[Row]) -> u64:
    t.len()

fn i(t: Table[Row]) -> Row:
    t[0]

fn j(t: Table[Row]) -> f32:
    var s = 0.0
    for r in t:
        s += r.score
    s

fn k(t: Table[Row]) -> Tensor[Int, N]:
    t.id
"#
    ));
    assert_err(
        &format!(
            r#"
{ROW}
fn f(t: Table[Row]):
    t.nope
"#
        ),
        "no column `nope`",
    );
    assert_err(
        &format!(
            r#"
{ROW}
fn f(t: Table[Row]) -> List[Text]:
    t.score
"#
        ),
        "expected `List[Text]`",
    );
}

#[test]
fn table_schemas_must_be_flat_structs() {
    assert_err(
        r#"
type R:
    xs: List[i32]

fn f(t: Table[R]):
    pass
"#,
        "table columns must be",
    );
    assert_err(
        r#"
fn f(t: Table[i32]):
    pass
"#,
        "must be a type",
    );
    assert_err(
        r#"
fn f(t: Table):
    pass
"#,
        "takes one type argument",
    );
    assert_err(
        r#"
type E:
    pass

fn f(t: Table[E]):
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
fn a() -> Table[Row]:
    std.table.read_csv("x.csv")

fn b() -> Table[Row]:
    std.table.read_parquet("x.parquet")

fn c(t: Table[Row]) -> Table[Row]:
    t |> std.table.sql("SELECT * FROM t WHERE score > 1")

fn d(rows: List[Row]) -> Table[Row]:
    std.table.from_rows(rows)

fn e(t: Table[Row]) -> List[Row]:
    std.table.to_rows(t)

fn f(t: Table[Row]) -> Table[Row]:
    t |> std.table.filter(r => r.score > 0.5)

fn g(t: Table[Row]):
    std.table.write_csv(t, "o.csv")
    std.table.write_parquet(t, "o.parquet")

fn h(t: Table[Row]) -> u64:
    std.table.len(t)
"#
    ));
    assert_err(
        &format!(
            r#"
{ROW}
fn f():
    let t = std.table.read_csv("x")
"#
        ),
        "annotate the result",
    );
    assert_err(
        &format!(
            r#"
{ROW}
fn f(t: Table[Row]) -> Table[Row]:
    std.table.filter(t, r => r.nope)
"#
        ),
        "no field `nope`",
    );
    assert_err(
        &format!(
            r#"
{ROW}
fn f(t: Table[Row]):
    std.table.sql(t, 1)
"#
        ),
        "must be `Text`",
    );
}

#[test]
fn tensors_can_be_indexed() {
    assert_clean(
        r#"
fn f(t: Tensor[f32, 4]) -> f32:
    t[0]
"#,
    );
    assert_clean(
        r#"
fn f(t: Tensor[f32, 2, 3]) -> Tensor[f32, 3]:
    t[1]
"#,
    );
    assert_clean(
        r#"
fn f(t: Tensor[f32, 2, 3]) -> f32:
    t[1][2]
"#,
    );
    assert_err(
        r#"
fn f(t: Tensor[f32, 4]) -> i32:
    t[0]
"#,
        "expected `i32`, found `f32`",
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
fn sq(x: Int) -> Int:
    x * x

fn f(v: List[Int]) -> List[Int]:
    std.dist.map(v, sq)
"#,
    );
    assert_clean(
        r#"
fn sq(x: Int) -> Int:
    x * x

fn f(v: List[Int]) -> List[Int]:
    v |> std.dist.map(sq)
"#,
    );
    assert_err(
        r#"
fn f(v: List[Int]) -> List[Int]:
    std.dist.map(v, x => x * 2)
"#,
        "needs the name of a top-level function",
    );
    assert_err(
        r#"
fn sq(x: Int) -> Int:
    x

fn f(v: List[Text]):
    std.dist.map(v, sq)
"#,
        "mismatched",
    );
    assert_clean(
        r#"
fn d(x: i32) -> i32:
    x

fn f(v: List[i32]) -> List[i32]:
    std.iter.map(v, d)
"#,
    );
}

// ---- enums and match -------------------------------------------------------------------------

const SHAPES: &str = r#"
enum Shape:
    Circle(Float)
    Rect(Float, Float)
    Empty

fn area(s: Shape) -> Float:
    match s:
        case Shape.Circle(r):
            r * r * 3.0
        case Shape.Rect(w, h):
            w * h
        case Shape.Empty:
            0.0

fn make() -> Shape:
    Shape.Rect(2.0, 3.0)

fn nothing() -> Shape:
    Shape.Empty
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

fn f(e: E) -> i32:
    match e:
        case E.A:
            1
        case E.B:
            2
"#,
        "missing `E.C`",
    );
    assert_err(
        r#"
fn f(n: i32) -> i32:
    match n:
        case 0:
            1
        case 1:
            2
"#,
        "add a `_` arm",
    );
    assert_err(
        r#"
fn f(b: Bool) -> i32:
    match b:
        case True:
            1
"#,
        "non-exhaustive",
    );
    assert_clean(
        r#"
fn f(b: Bool) -> i32:
    match b:
        case True:
            1
        case False:
            0
"#,
    );
    assert_clean(
        r#"
fn f(n: i32) -> i32:
    match n:
        case 0:
            1
        case other:
            other
"#,
    );
    assert_clean(
        r#"
enum E:
    A(Bool)
    B

fn f(e: E) -> i32:
    match e:
        case E.A(True):
            1
        case E.A(False):
            2
        case E.B:
            3
"#,
    );
    assert_err(
        r#"
enum E:
    A(Bool)
    B

fn f(e: E) -> i32:
    match e:
        case E.A(True):
            1
        case E.B:
            3
"#,
        "non-exhaustive",
    );
}

#[test]
fn match_on_results() {
    assert_clean(
        r#"
fn f(r: Result[i32, Text]) -> i32:
    match r:
        case Ok(v):
            v
        case Err(_):
            0
"#,
    );
    assert_err(
        r#"
fn f(r: Result[i32, Text]) -> i32:
    match r:
        case Ok(v):
            v
"#,
        "non-exhaustive",
    );
}

#[test]
fn enum_type_errors() {
    assert_err(
        r#"
enum E:
    A(i32)

fn f() -> E:
    E.A
"#,
        "takes 1 value",
    );
    assert_err(
        r#"
enum E:
    A

fn f() -> E:
    E.B
"#,
        "no variant `B`",
    );
    assert_err(
        r#"
enum E:
    A(i32)

fn f() -> E:
    E.A(True)
"#,
        "mismatched types",
    );
    assert_err(
        r#"
enum E:
    A

enum F:
    B

fn f(e: E) -> i32:
    match e:
        case F.B:
            1
        case E.A:
            2
"#,
        "cannot match",
    );
    assert_err(
        r#"
fn f(n: i32) -> i32:
    match n:
        case 0:
            1
        case _:
            "x"
"#,
        "incompatible types",
    );
    assert_err(
        r#"
fn f(n: i32) -> i32:
    match n:
        case "a":
            1
        case _:
            2
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
fn main() -> i32:
    let total = 5
    totl + 1
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `total`")),
        "{e:?}"
    );
    let e = full(
        r#"
fn helper() -> i32:
    1

fn main() -> i32:
    helpr()
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `helper`")),
        "{e:?}"
    );
    let e = full(
        r#"
fn main() -> i32:
    let s: Flot = 1.0
    1
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `Float`")),
        "{e:?}"
    );
    let e = full(
        r#"
type P:
    width: i32

fn f(p: P) -> i32:
    p.widht
"#,
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `width`")),
        "{e:?}"
    );
    // Nothing close: no suggestion.
    let e = full(
        r#"
fn main() -> i32:
    zzzzzz
"#,
    );
    assert!(e[0].1.is_empty(), "{e:?}");
}

#[test]
fn unknown_methods_on_builtin_types_are_compile_errors() {
    let e = full(
        r#"
fn f(v: List[i32]) -> u64:
    v.lenn()
"#,
    );
    assert!(
        e[0].0.contains("no method `lenn` on type `List[i32]`"),
        "{e:?}"
    );
    assert!(
        e[0].1.iter().any(|n| n.contains("did you mean `len`")),
        "{e:?}"
    );
    assert_err(
        r#"
fn f(s: Text) -> Text:
    s.shout()
"#,
        "no method `shout` on type `Text`",
    );
    assert_clean(
        r#"
fn f(v: List[i32], s: Text) -> u64:
    if s.is_empty():
        v.len()
    else:
        0
"#,
    );
}

#[test]
fn a_missing_colon_is_reported_where_it_belongs() {
    let src = "fn main() -> i32\n    1\n";
    let (_, errs) = telvax_parser::parse(src);
    let first = &errs[0];
    assert!(first.message.contains("expected `:`"), "{}", first.message);
    // The error points at the end of the line, where the colon is missing.
    assert_eq!(&src[first.span.start..first.span.start], "");
    assert_eq!(first.span.start, "fn main() -> i32".len());
}

// ---- Option and Map ----------------------------------------------------------------------------

#[test]
fn option_and_map_are_typed() {
    assert_clean(
        r#"
fn f(o: i32?) -> i32:
    match o:
        case Some(n):
            n
        case None:
            0
"#,
    );
    assert_clean(
        r#"
fn f() -> i32?:
    Some(3)

fn g() -> Text?:
    None
"#,
    );
    assert_clean(
        r#"
fn f(v: List[Int]) -> Int:
    v.first().unwrap_or(0)
"#,
    );
    assert_clean(
        r#"
fn f() -> Map[Text, Int]:
    let m: Map[Text, Int] = std.map.new()
    m.insert("a", 1)
    m
"#,
    );
    assert_clean(
        r#"
fn f(m: Map[Int, Text], k: Int) -> Text:
    match m.get(k):
        case Some(s):
            s
        case None:
            ""
"#,
    );
}

#[test]
fn option_and_map_mistakes_are_reported() {
    assert_err(
        r#"
fn f(o: i32?) -> i32:
    match o:
        case Some(n):
            n
"#,
        "missing `None`",
    );
    assert_err(
        r#"
fn f(o: i32?) -> i32:
    match o:
        case None:
            0
"#,
        "missing `Some`",
    );
    assert_err(
        r#"
fn f() -> i32?:
    Some("x")
"#,
        "mismatched types",
    );
    assert_err(
        r#"
fn f(m: Map[Float, Int]) -> u64:
    m.len()
"#,
        "keys must be",
    );
    assert_err(
        r#"
fn f(m: Map[Text, Int]) -> Int:
    m.insert(1, 2)
    0
"#,
        "mismatched types",
    );
    assert_err(
        r#"
fn f(o: i32?) -> i32:
    o.unwarp()
"#,
        "no method `unwarp`",
    );
    assert_err(
        r#"
fn f() -> Map[Text, Int]:
    std.map.new(1)
"#,
        "expected 0 argument",
    );
    assert_err(
        r#"
fn f(n: i32) -> i32:
    match n:
        case Some(x):
            x
        case _:
            0
"#,
        "need an `Option`",
    );
}

#[test]
fn closures_cannot_assign_to_captured_variables() {
    assert_err(
        r#"
fn f() -> Int:
    var n = 0
    fn inc(d: Int):
        n += d
    inc(1)
    n
"#,
        "cannot assign to `n` from inside a closure",
    );
    // A closure's own variables and shared containers are fine.
    assert_clean(
        r#"
fn f() -> Int:
    fn inc(d: Int):
        var t = 0
        t += d
        t
    inc(1)
"#,
    );
    assert_clean(
        r#"
fn f(v: List[Int]) -> Int:
    fn set(x: Int):
        v[0] = x
    set(3)
    v[0]
"#,
    );
}

// ---- friendlier syntax -------------------------------------------------------------------------

#[test]
fn prelude_names_work_without_std() {
    assert_clean(
        r#"
fn main():
    print("hi")
    std.io.print("a")
"#,
    );
    // A function or variable of the same name wins over the prelude.
    assert_clean(
        r#"
fn print(s: Text) -> i32:
    1

fn main() -> i32:
    print("x")
"#,
    );
    assert_clean(
        r#"
type R:
    a: Int

fn f() -> Table[R]:
    let t: Table[R] = read_csv("x.csv")
    t
"#,
    );
}

#[test]
fn tensor_methods_and_operators_are_checked() {
    assert_clean(
        r#"
fn f(x: Tensor[f32, 2, 3], w: Tensor[f32, 3, 4]) -> Tensor[f32, 2, 4]:
    x.matmul(w).relu().softmax()
"#,
    );
    assert_clean(
        r#"
fn f(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
    a * 2.0 + b - a / b
"#,
    );
    assert_clean(
        r#"
fn f(a: Tensor[f32, 4]) -> f32:
    a.sum()
"#,
    );
    // The shape rules are the ones of the function forms.
    assert_err(
        r#"
fn f(a: Tensor[f32, 4], b: Tensor[f32, 5]) -> Tensor[f32, 4]:
    a + b
"#,
        "shape",
    );
    assert_err(
        r#"
fn f(x: Tensor[f32, 2, 3], w: Tensor[f32, 4, 2]) -> Tensor[f32, 2, 2]:
    x.matmul(w)
"#,
        "inner dimensions",
    );
    let e = full(
        r#"
fn f(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
    a + 1.0
"#,
    );
    assert!(e[0].0.contains("cannot apply this operator"), "{e:?}");
    assert!(e[0].1.iter().any(|n| n.contains("two tensors")), "{e:?}");
    let e = full(
        r#"
fn f(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
    a / 2.0
"#,
    );
    assert!(e[0].1.iter().any(|n| n.contains("reciprocal")), "{e:?}");
}

#[test]
fn inline_row_types_replace_throwaway_structs() {
    let src = r#"
type Sale:
    id: Int
    units: Int

fn f(t: Table[Sale]) -> Table[{units: Int}]:
    t.sql("SELECT units FROM t")

fn g(t: Table[{units: Int}]) -> u64:
    t.len()

fn h(t: Table[Sale]) -> u64:
    g(f(t))
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
fn f(t: Table[{a: Int, a: Int}]):
    pass
"#,
        "declared twice",
    );
    assert_err(
        r#"
fn f(t: Table[{a: List[Int]}]):
    pass
"#,
        "column",
    );
    assert_err(
        r#"
fn f(t: Table[{}]):
    pass
"#,
        "at least one column",
    );
    assert_err(
        r#"
type Sale:
    id: Int

fn f(t: Table[Sale]) -> Table[{id: Text}]:
    t.sql("SELECT id FROM t")

fn g(t: Table[Sale]) -> Table[Sale]:
    f(t)
"#,
        "mismatched types",
    );
}

#[test]
fn question_mark_is_checked_against_the_function_result() {
    assert_clean(
        r#"
fn f(r: Result[i32, Text]) -> Result[i32, Text]:
    let x = r?
    Ok(x + 1)
"#,
    );
    assert_clean(
        r#"
fn f(o: i32?) -> i32?:
    let x = o?
    Some(x + 1)
"#,
    );
    assert_err(
        r#"
fn f(r: Result[i32, Text]) -> i32:
    r?
"#,
        "can only be used in a function that returns `Result`",
    );
    assert_err(
        r#"
fn f(r: Result[i32, Int]) -> Result[i32, Text]:
    let x = r?
    Ok(x)
"#,
        "cannot turn the error",
    );
    assert_err(
        r#"
fn f(o: i32?) -> Result[i32, Text]:
    let x = o?
    Ok(x)
"#,
        "returns `Option`",
    );
    assert_err(
        r#"
fn f(n: i32) -> Result[i32, Text]:
    let x = n?
    Ok(x)
"#,
        "needs a `Result` or an `Option`",
    );
}

#[test]
fn tuples_are_typed() {
    assert_clean(
        r#"
fn f() -> (Int, Text):
    (1, "a")

fn g() -> Int:
    let n, _ = f()
    n + f().0
"#,
    );
    assert_clean(
        r#"
fn f(p: (Bool, Int)) -> Int:
    match p:
        case (True, n):
            n
        case (False, _):
            0
"#,
    );
    assert_clean(
        r#"
fn f(v: List[(Int, Int)]) -> Int:
    var t = 0
    for a, b in v:
        t += a * b
    t
"#,
    );
    assert_err(
        r#"
fn f(p: (Int, Text)) -> Int:
    p.2
"#,
        "has no part `2`",
    );
    assert_err(
        r#"
fn f() -> (Int, Text):
    (1, 2)
"#,
        "mismatched types",
    );
    assert_err(
        r#"
fn f(p: (Bool, Int)) -> Int:
    match p:
        case (True, n):
            n
"#,
        "non-exhaustive",
    );
    assert_err(
        r#"
fn f(p: (Int, Int)) -> Int:
    let a, b = p
    a + c
"#,
        "cannot find `c`",
    );
}

#[test]
fn iterator_methods_are_typed() {
    assert_clean(
        r#"
fn f(v: List[Int]) -> List[Text]:
    v.map(x => f"{x}").filter(s => s.len() > 0)
"#,
    );
    assert_clean(
        r#"
fn f(v: List[Int]) -> Int:
    v.fold(0, (acc, x) => acc + x)
"#,
    );
    assert_clean(
        r#"
fn f(v: List[Int]) -> List[(Int, Int)]:
    v.enumerate()
"#,
    );
    assert_clean(
        r#"
fn f(v: List[Int]) -> Int?:
    v.max()
"#,
    );
    assert_err(
        r#"
fn f(v: List[Text]) -> Int:
    v.sum()
"#,
        "needs a vector of numbers",
    );
    assert_err(
        r#"
fn f(v: List[Int]) -> Bool:
    v.any(x => x + 1)
"#,
        "Bool",
    );
}

#[test]
fn impl_blocks_are_checked() {
    let src = r#"
type P:
    x: Int

    fn get(self) -> Int:
        self.x

    fn make(x: Int) -> P:
        P(x: x)

    fn add(self, n: Int) -> P:
        Self(x: self.x + n)

fn f() -> Int:
    P.make(1).add(2).get()
"#;
    assert_clean(src);
    let (hir, _) = check(src);
    assert!(
        hir.fns.iter().any(|f| f.name == "P::get"),
        "methods are functions named Type::name"
    );
    assert_err(
        r#"
type P:
    x: Int

    fn get(self) -> Int:
        self.y
"#,
        "no field `y`",
    );
    assert_err(
        r#"
extend Nothing:
    fn f():
        pass
"#,
        "cannot find the type or enum `Nothing`",
    );
    assert_err(
        r#"
type P:
    x: Int

    fn make() -> P:
        P(x: 1)

fn f(p: P) -> P:
    p.make()
"#,
        "has no `self` parameter",
    );
    assert_err(
        r#"
type P:
    x: Int

    fn get(self) -> Int:
        self.x

fn f(p: P) -> Int:
    p.get(1)
"#,
        "expected 0 argument",
    );
    assert_err(
        r#"
fn f() -> Self:
    1
"#,
        "only be used inside a `type`",
    );
    assert_err(
        r#"
type P:
    x: Int

    fn a(self):
        pass

    fn a(self):
        pass
"#,
        "defined twice",
    );
}

#[test]
fn traits_are_checked() {
    let base = r#"
trait Shape:
    fn area(self) -> Float: ...

    fn name(self) -> Text:
        "shape"

type Sq:
    s: Float
"#;
    assert_clean(&format!(
        r#"
{base}
extend Sq(Shape):
    fn area(self) -> Float:
        self.s

fn total(v: List[Shape]) -> Float:
    v.map(x => x.area()).sum()

fn make() -> List[Shape]:
    [Sq(s: 1.0)]
"#
    ));
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    fn name(self) -> Text:
        "sq"
"#
        ),
        "is missing the method `area`",
    );
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    fn area(self) -> Int:
        1
"#
        ),
        "has the signature",
    );
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    fn area(self) -> Float:
        1.0

    fn extra(self):
        pass
"#
        ),
        "is not a method of the trait `Shape`",
    );
    assert_err(
        r#"
type S(Missing):
    fn f(self):
        pass
"#,
        "cannot find the trait `Missing`",
    );
    assert_err(
        &format!(
            r#"
{base}
type Other:
    x: Int

fn f() -> Shape:
    Other(x: 1)
"#
        ),
        "mismatched types",
    );
    assert_err(
        &format!(
            r#"
{base}
extend Sq(Shape):
    fn area(self) -> Float:
        1.0

fn f(s: Shape) -> Float:
    s.volume()
"#
        ),
        "has no method `volume`",
    );
    assert_err("fn f() -> i32: ...\n", "needs a body");
}

#[test]
fn generics_are_checked_and_instantiated() {
    let base = r#"
trait Named:
    fn name(self) -> Text: ...

type Dog(Named):
    n: Text

    fn name(self) -> Text:
        self.n

fn id[T](x: T) -> T:
    x

fn greet[T: Named](x: T) -> Text:
    x.name()

type Boxed[T]:
    v: T

enum Maybe[T]:
    Nothing
    Just(T)
"#;
    assert_clean(&format!(
        r#"
{base}
fn a() -> Text:
    greet(Dog(n: "r"))

fn b() -> Text:
    let m: Maybe[Text] = Maybe.Just(id("x"))
    match m:
        case Maybe.Just(s):
            s
        case Maybe.Nothing:
            ""

fn c() -> Boxed[Text]:
    Boxed(v: "s")
"#
    ));
    assert_err(
        &format!(
            r#"
{base}
fn a() -> i32:
    greet(3)
"#
        ),
        "does not implement `Named`",
    );
    assert_err(
        &format!(
            r#"
{base}
fn a() -> i32:
    let f = id
    0
"#
        ),
        "has to be called",
    );
    assert_err(
        &format!(
            r#"
{base}
fn a() -> i32:
    let m = Maybe.Nothing
    0
"#
        ),
        "cannot infer the type parameter `T`",
    );
    assert_err(
        &format!(
            r#"
{base}
fn a(b: Boxed[i32, i32]) -> i32:
    0
"#
        ),
        "takes 1 type argument",
    );
}

#[test]
fn lowercase_constants_get_a_hint() {
    let e = {
        let (ast, _) = telvax_parser::parse(
            r#"
fn f() -> Bool:
    true
"#,
        );
        let (_, diags) = lower(&ast.unwrap());
        diags
    };
    assert!(
        e.iter()
            .any(|d| d.notes.iter().any(|n| n.contains("did you mean `True`"))),
        "{e:?}"
    );
}

#[test]
fn let_is_fixed_and_var_can_change() {
    assert_clean(
        r#"
fn f() -> i32:
    var x = 1
    x = x + 1
    x += 2
    x
"#,
    );
    assert_err(
        r#"
fn f():
    let x = 1
    x = 2
"#,
        "cannot assign to the immutable variable `x`",
    );
    // A plain assignment never declares a name.
    assert_err(
        r#"
fn f():
    y = 2
"#,
        "cannot find `y`",
    );
}

#[test]
fn named_arguments_and_defaults_are_checked() {
    let decls = r#"
type Query:
    text: Text
    top_k: i32 = 5

fn search(q: Query, limit: i32 = 10, rerank: Bool = False) -> i32:
    limit

type Box:
    n: i32

    fn scaled(self, by: i32 = 2, add: i32 = 0) -> i32:
        self.n * by + add
"#;
    assert_clean(&format!(
        "{decls}\nfn main() -> i32:\n    let q = Query(text: \"x\")\n    let b = Box(n: 1)\n    \
         search(q) + search(q, rerank: True) + search(rerank: True, q: q, limit: 3) + b.scaled() \
         + b.scaled(add: 1) + b.scaled(3)\n"
    ));
    for (call, needle) in [
        ("search(q, speed: 1)", "no parameter `speed`"),
        ("search(q, limit: 1, limit: 2)", "`limit` is given twice"),
        ("search(limit: 1)", "missing the argument `q`"),
        ("search(q, limit: \"3\")", "expected `i32`"),
    ] {
        assert_err(
            &format!("{decls}\nfn main() -> i32:\n    let q = Query(text: \"x\")\n    {call}\n"),
            needle,
        );
    }
    // A missing field without a default is still an error.
    assert_err(
        &format!("{decls}\nfn main():\n    let q = Query(top_k: 1)\n"),
        "text",
    );
}

#[test]
fn names_on_calls_the_program_does_not_define_are_labels() {
    assert_clean(
        r#"
fn main() -> Text:
    std.text.truncate("hello", n: 2)
"#,
    );
}

#[test]
fn optional_types_and_inline_tensor_shapes() {
    assert_clean(
        r#"
fn f(o: i32?) -> i32:
    match o:
        case Some(n):
            n
        case None:
            0

fn g(x: Tensor[f32, 2, 3], w: Tensor[f32, 3, 4]) -> Tensor[f32, 2, 4]:
    x.matmul(w)
"#,
    );
    assert_err(
        r#"
fn g(x: Tensor[f32, 2, 3], w: Tensor[f32, 4, 4]) -> Tensor[f32, 2, 4]:
    x.matmul(w)
"#,
        "inner dimensions differ",
    );
    // The spellings other languages use are corrected.
    let e = errors("fn f(x: int):\n    pass\n");
    assert!(
        e.iter().any(|m| m.contains("cannot find type `int`")),
        "{e:?}"
    );
}

#[test]
fn generic_call_takes_the_expected_type_for_literals() {
    assert_clean(
        r#"
fn id[T](x: T) -> T:
    x

fn first[T](xs: List[T]) -> T?:
    xs.first()

fn f() -> Int:
    let a: Int = id(7)
    let b: Int? = first([1, 2])
    let c: Float = id(2.5)
    a
"#,
    );
    assert_err(
        r#"
fn id[T](x: T) -> T:
    x

fn f():
    let a: Text = id(7)
"#,
        "mismatched types",
    );
}

#[test]
fn one_mistake_is_reported_once() {
    let count = |src: &str| errors(src).len();
    assert_eq!(
        count("fn f(x: Int?) -> Int:\n    x + 1\n"),
        1,
        "operand mismatch returned"
    );
    assert_eq!(
        count("fn k(a: i32, b: i64) -> i64:\n    a + b\n"),
        1,
        "operand sizes"
    );
    assert_eq!(
        count("fn h() -> Result[Int, Text]:\n    Err(3)\n"),
        1,
        "wrong error type in Err"
    );
    assert_eq!(
        count("fn h() -> Result[Int, Text]:\n    Ok(\"a\")\n"),
        1,
        "wrong value type in Ok"
    );
}

#[test]
fn a_function_without_arrow_returns_its_final_expression() {
    assert_clean(
        r#"
fn dbl(x: Int):
    x * 2

fn quad(x: Int):
    dbl(dbl(x))

fn name(n: Int):
    if n > 0:
        "pos"
    else:
        "neg"

fn early(n: Int):
    if n > 100:
        return 100
    n

fn fact(n: Int):
    if n <= 1:
        1
    else:
        n * fact(n - 1)

fn pair(a: Int):
    (a, "x")

fn show(x: Int):
    print(f"{x}")

fn main() -> i32:
    let a: Int = quad(2)
    let s: Text = name(3)
    let e: Int = early(5)
    let f: Int = fact(4)
    let q: Int = pair(1).0
    show(a)
    0
"#,
    );
    let (hir, _) = check("fn dbl(x: Int):\n    x * 2\n");
    assert_eq!(hir.fns[0].ret, Ty::Prim(telvax_hir::Prim::I64));
    assert_err(
        "fn dbl(x: Int):\n    x * 2\n\nfn f():\n    let t: Text = dbl(1)\n",
        "expected `Text`, found `Int`",
    );
}

#[test]
fn main_tests_and_methods_keep_returning_nothing() {
    let (hir, errs) = check(
        r#"
fn main():
    1 + 1

fn test_it():
    1 + 1

type P:
    x: Int

    fn get(self):
        self.x
"#,
    );
    assert!(errs.is_empty(), "{errs:?}");
    for f in &hir.fns {
        assert_eq!(f.ret, Ty::Unit, "{}", f.name);
    }
}

#[test]
fn a_generic_call_waits_for_the_use_of_its_literals() {
    assert_clean(
        r#"
fn id[T](x: T) -> T:
    x

fn twice[T](x: T) -> (T, T):
    (x, x)

fn main() -> i32:
    let a = id(3)
    let n: Int = a
    let b = id(2.5)
    let f: f32 = b
    let t = twice(1)
    let u: (i64, i64) = t
    let free = id(7)
    let s = id("x")
    0
"#,
    );
    // Without a use, the literal still gets its default.
    let (hir, errs) = check("fn id[T](x: T) -> T:\n    x\n\nfn f():\n    let a = id(3)\n");
    assert!(errs.is_empty(), "{errs:?}");
    assert!(hir.fns.iter().any(|f| f.name == "id<i32>"));
    assert!(!hir.fns.iter().any(|f| f.name == "id<Int>"));
    // Different uses make different instances.
    let (hir, errs) = check(
        "fn id[T](x: T) -> T:\n    x\n\nfn f():\n    let a: Int = id(3)\n    let b: u8 = id(3)\n",
    );
    assert!(errs.is_empty(), "{errs:?}");
    assert!(hir.fns.iter().any(|f| f.name == "id<Int>"));
    assert!(hir.fns.iter().any(|f| f.name == "id<u8>"));
    // The trait bound is still checked once the type is known.
    assert_err(
        r#"
trait Shape:
    fn area(self) -> Float: ...

fn big[T: Shape](a: T) -> T:
    a

fn f():
    let a = big(3)
"#,
        "Shape",
    );
}
