use novar_hir::{Ty, hir, lower};

fn check(src: &str) -> (hir::Program, Vec<String>) {
    let (ast, parse_errs) = novar_parser::parse(src);
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
    let src = include_str!("../../../examples/rag-service.nv");
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
    let (hir, errs) = check("fn inc(x: i32) -> i32 { x } fn f(a: i32) -> i32 { a |> inc() }");
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
        "fn inc(x: i32) -> i32 { x } fn f(s: String) -> i32 { s |> inc() }",
        "expected `i32`, found `String`",
    );
    assert_err("fn f(a: i32) -> i32 { a |> 3 }", "right side of `|>`");
}

#[test]
fn tensor_shapes_are_enforced() {
    let ok = "fn take(x: Tensor<f32, [4]>) {} fn f(y: Tensor<f32, [4]>) { take(y); }";
    assert_clean(ok);
    assert_err(
        "fn take(x: Tensor<f32, [4]>) {} fn f(y: Tensor<f32, [8]>) { take(y); }",
        "expected `Tensor<f32, [4]>`, found `Tensor<f32, [8]>`",
    );
    assert_err(
        "fn take(x: Tensor<f32, [4]>) {} fn f(y: Tensor<f32, [4, 4]>) { take(y); }",
        "mismatched types",
    );
    assert_err(
        "fn take(x: Tensor<f32, [4]>) {} fn f(y: Tensor<i32, [4]>) { take(y); }",
        "mismatched types",
    );
    // Symbolic dimensions match any size.
    assert_clean("fn take(x: Tensor<f32, [4]>) {} fn f(y: Tensor<f32, [N]>) { take(y); }");
}

#[test]
fn invalid_tensor_types() {
    assert_err("fn f(x: Tensor<String, [4]>) {}", "numeric primitive");
    assert_err("fn f(x: Tensor<f32>) {}", "`Tensor` expects");
    assert_err("fn f(x: Tensor<f32, [0]>) {}", "greater than zero");
    assert_err("fn f(x: Tensor<f32, []>) {}", "at least one dimension");
}

#[test]
fn name_resolution() {
    assert_err("fn f() -> i32 { missing }", "cannot find `missing`");
    assert_err("fn f(x: Nope) {}", "cannot find type `Nope`");
    assert_err("struct A {} fn f() { A; }", "is a struct, not a value");
    assert_err("fn f() {} fn f() {}", "defined twice");
    assert_err("fn f(a: i32, a: i32) {}", "declared twice");
    assert_clean(
        "import std::ai::llm::OpenAIClient; let c = OpenAIClient::new(\"m\"); fn f() { let x = c; }",
    );
    // A binding is not visible inside its own initializer.
    assert_err("fn f() { let x = x; }", "cannot find `x`");
}

#[test]
fn literals_adapt_to_context() {
    assert_clean("fn f(a: f32) -> f32 { a + 1 }");
    assert_clean("fn f() { let x: f32 = 1.5; let y: u8 = 3; let z: f64 = 2; }");
    assert_err(
        "fn f() { let x: i32 = 1.5; }",
        "expected `i32`, found `{float}`",
    );
    assert_err(
        "fn f() { let x: i32 = \"a\"; }",
        "expected `i32`, found `String`",
    );
}

#[test]
fn calls() {
    assert_err(
        "fn g(a: i32) {} fn f() { g(1, 2); }",
        "expected 1 argument(s), found 2",
    );
    assert_err(
        "fn g(a: i32) {} fn f() { g(true); }",
        "expected `i32`, found `bool`",
    );
    assert_err("fn f(a: i32) { a(); }", "expected a function");
}

#[test]
fn structs() {
    let decl = "struct P { x: i32, name: String } ";
    assert_clean(&format!(
        "{decl}fn f() -> P {{ P {{ x: 1, name: \"a\" }} }}"
    ));
    assert_err(
        &format!("{decl}fn f() {{ P {{ x: 1 }}; }}"),
        "missing field(s) in `P`: `name`",
    );
    assert_err(
        &format!("{decl}fn f() {{ P {{ x: 1, name: \"a\", y: 2 }}; }}"),
        "no field `y`",
    );
    assert_err(
        &format!("{decl}fn f(p: P) -> i32 {{ p.nope }}"),
        "no field `nope`",
    );
    assert_err(
        &format!("{decl}fn f() {{ P {{ x: \"s\", name: \"a\" }}; }}"),
        "expected `i32`",
    );
    assert_err("struct A { x: i32, x: i32 }", "declared twice");
    assert_err("fn f() { Q { a: 1 }; }", "cannot find struct `Q`");
}

#[test]
fn returns_and_blocks() {
    assert_clean("fn f() -> i32 { return 1; }");
    assert_clean("fn f(c: bool) -> i32 { if c { return 1; } 2 }");
    assert_err("fn f() -> i32 { let x = 1; }", "expected `i32`, found `()`");
    assert_err("fn f() -> i32 { return; }", "needs a value");
    assert_err(
        "fn f() -> i32 { return \"s\"; }",
        "expected `i32`, found `String`",
    );
    assert_clean("fn f() { return; }");
}

#[test]
fn conditions_and_operators() {
    assert_err("fn f() { if 1 { } }", "expected `bool`, found `{integer}`");
    assert_err(
        "fn f(c: bool) -> i32 { if c { 1 } else { \"s\" } }",
        "incompatible types",
    );
    assert_clean("fn f(c: bool) -> i32 { if c { 1 } else { 2 } }");
    assert_err("fn f() -> i32 { 1 + \"a\" }", "mismatched operand types");
    assert_err("fn f() -> bool { true && 1 }", "expected `bool`");
    assert_err("fn f() -> bool { true < false }", "cannot order");
    assert_clean("fn f(a: String) -> String { a + \"x\" }");
    assert_err("fn f() { let x = -\"a\"; }", "cannot apply");
}

#[test]
fn result_constructors_follow_the_declared_type() {
    assert_clean("fn f() -> Result<i32, String> { Ok(1) }");
    assert_clean("fn f() -> Result<i32, String> { Err(\"bad\") }");
    assert_err(
        "fn f() -> Result<i32, String> { Ok(\"x\") }",
        "expected `i32`, found `String`",
    );
    assert_err(
        "fn f() -> Result<i32, String> { Err(1) }",
        "expected `String`, found `{integer}`",
    );
    assert_err(
        "fn f() -> Result<i32, String> { Ok(1, 2) }",
        "exactly one argument",
    );
}

#[test]
fn string_and_vec_methods() {
    assert_clean("fn f(s: String) -> bool { s.is_empty() }");
    assert_err(
        "fn f(s: String) -> i32 { s.is_empty() }",
        "expected `i32`, found `bool`",
    );
    assert_clean("fn f(v: Vec<i32>) -> u64 { v.len() }");
}

#[test]
fn endpoints() {
    let ok = "struct Q { a: i32 } @endpoint(path = \"/x\", method = \"POST\") \
              fn h(q: Q) -> Result<Q, String> { Ok(q) }";
    assert_clean(ok);
    let get = "@endpoint(path = \"/ping\") fn h() -> Result<String, String> { Ok(\"pong\") }";
    let (hir, errs) = check(get);
    assert!(errs.is_empty(), "{errs:?}");
    assert_eq!(
        hir.fns[0].endpoint.as_ref().unwrap().method,
        hir::HttpMethod::Get
    );

    assert_err(
        "@endpoint(method = \"GET\") fn h() -> Result<i32, i32> { Ok(1) }",
        "requires a `path`",
    );
    assert_err(
        "@endpoint(path = \"x\") fn h() -> Result<i32, i32> { Ok(1) }",
        "must start with `/`",
    );
    assert_err(
        "@endpoint(path = \"/x\", method = \"FETCH\") fn h() -> Result<i32, i32> { Ok(1) }",
        "unsupported HTTP method",
    );
    assert_err(
        "@endpoint(path = \"/x\", method = \"POST\") fn h() -> Result<i32, i32> { Ok(1) }",
        "payload parameter",
    );
    assert_err(
        "@endpoint(path = \"/x\") fn h(a: i32) -> Result<i32, i32> { Ok(1) }",
        "must be a struct",
    );
    assert_err(
        "@endpoint(path = \"/x\") fn h() -> i32 { 1 }",
        "must return `Result",
    );
    assert_err(
        "@endpoint(path = \"/x\", color = \"r\") fn h() -> Result<i32, i32> { Ok(1) }",
        "unknown `@endpoint` argument",
    );
    assert_err("@route fn h() {}", "unknown attribute");
    assert_err(
        "@endpoint(path = \"/x\") fn a() -> Result<i32, i32> { Ok(1) } \
         @endpoint(path = \"/x\") fn b() -> Result<i32, i32> { Ok(1) }",
        "duplicate endpoint",
    );
}

#[test]
fn closures_and_extern_calls_stay_lenient() {
    assert_clean(
        "import std::iter::map; fn f(v: Vec<String>) { let r = v |> std::iter::map(|d| d.text) |> std::collect(); }",
    );
    assert_clean("fn f() { let g = |a, b| a + b; }");
}

#[test]
fn diagnostics_render_with_source_context() {
    let src = "fn f() -> i32 {\n    \"oops\"\n}\n";
    let (ast, _) = novar_parser::parse(src);
    let (_, diags) = lower(&ast.unwrap());
    let out = novar_diagnostics::render("main.nv", src, &diags, false);
    assert!(out.contains("main.nv:2:5"), "{out}");
    assert!(out.contains("expected `i32`, found `String`"), "{out}");
}

// ---- standard library ----------------------------------------------------------------------

#[test]
fn std_functions_are_typed() {
    assert_clean(
        "fn f(s: String) -> String { s |> std::text::clean_whitespace() |> std::text::truncate(10) }",
    );
    assert_err(
        "fn f(n: i32) -> String { std::text::trim(n) }",
        "argument 1 must be `String`, found `i32`",
    );
    assert_err(
        "fn f(s: String) -> String { std::text::truncate(s) }",
        "expected 2 argument(s), found 1",
    );
    assert_err(
        "fn f(s: String) -> String { std::text::truncate(s, \"x\") }",
        "must be an integer",
    );
    assert_err(
        "fn f(s: String) -> i32 { std::text::trim(s) }",
        "expected `i32`, found `String`",
    );
}

#[test]
fn std_unknown_items_are_errors() {
    assert_err(
        "fn f() { std::text::nope(); }",
        "cannot find `std::text::nope`",
    );
    assert_err("import std::nope::Thing; fn f() {}", "unresolved import");
    assert_err(
        "import std::text; fn f() { text::nope(); }",
        "cannot find `std::text::nope`",
    );
    assert_err("fn f() { let x = std::text; }", "is not a value");
}

#[test]
fn iterators_propagate_element_types() {
    assert_clean(
        "struct D { text: String } \
         fn f(v: Vec<D>) -> String { v |> std::iter::map(|d| d.text) |> std::string::join(\", \") }",
    );
    // The closure parameter is `D`, so a missing field is caught inside it.
    assert_err(
        "struct D { text: String } fn f(v: Vec<D>) { v |> std::iter::map(|d| d.nope); }",
        "no field `nope`",
    );
    assert_err(
        "fn f(v: Vec<String>) -> Vec<i32> { v |> std::iter::map(|s| s) }",
        "expected `Vec<i32>`, found `Vec<String>`",
    );
    assert_err(
        "fn f(n: i32) { std::iter::map(n, |x| x); }",
        "must be a `Vec`",
    );
    assert_err(
        "fn f(v: Vec<i32>) { v |> std::iter::filter(|x| 1); }",
        "must be `fn(_) -> bool`",
    );
    assert_clean("fn f(v: Vec<i32>) -> u64 { v |> std::iter::take(3) |> std::iter::count() }");
}

#[test]
fn std_types_have_typed_methods() {
    let head = "import std::ai::embeddings::OllamaEmbedder; import std::ai::llm::OpenAIClient; \
                let e = OllamaEmbedder::new(\"nomic-embed-text\"); let l = OpenAIClient::new(\"m\"); ";
    assert_clean(&format!(
        "{head} fn f(t: String) -> Tensor<f32, [768]> {{ t |> e.embed() }}"
    ));
    assert_err(
        &format!("{head} fn f(t: String) -> Tensor<f32, [512]> {{ t |> e.embed() }}"),
        "expected `Tensor<f32, [512]>`, found `Tensor<f32, [768]>`",
    );
    assert_err(
        &format!("{head} fn f(t: String) -> String {{ l.nope(t) }}"),
        "no method `nope` on type `OpenAIClient`",
    );
    assert_err(
        &format!("{head} fn f() -> String {{ l.complete(1) }}"),
        "must be `String`",
    );
}

#[test]
fn vector_query_pipeline() {
    let head = "import std::db::pgvector::PgVectorClient; \
                let db = PgVectorClient::connect(\"postgresql://x\"); ";
    assert_clean(&format!(
        "{head} fn f(v: Tensor<f32, [768]>) -> Vec<String> {{ \
         db.search_collection(\"kb\") |> db.with_vector(v) |> db.limit(3) |> db.execute() }}"
    ));
    assert_err(
        &format!(
            "{head} fn f(v: Tensor<f32, [8]>) {{ db.search_collection(\"kb\") |> db.with_vector(v); }}"
        ),
        "must be `Tensor<f32, [768]>`",
    );
    assert_err(
        &format!("{head} fn f() {{ db.execute(\"kb\"); }}"),
        "must be `VectorQuery`",
    );
}

#[test]
fn status_constructors_flow_into_endpoint_results() {
    let src = "import std::http::Status; \
               fn f(ok: bool) -> Result<String, Status> { \
                   if ok { return Ok(\"y\"); } Err(Status::NotFound(\"nope\")) }";
    assert_clean(src);
    assert_err(
        "import std::http::Status; fn f() -> Result<String, Status> { Err(Status::Teapot(\"x\")) }",
        "cannot find `std::http::Status::Teapot`",
    );
}

// ---- mutation, loops, arrays ---------------------------------------------------------------

#[test]
fn mutable_locals() {
    assert_clean("fn f() -> i32 { let mut x = 1; x = x + 1; x += 3; x }");
    assert_err(
        "fn f() { let x = 1; x = 2; }",
        "cannot assign to the immutable variable `x`",
    );
    assert_err("fn f(a: i32) { a = 2; }", "immutable variable `a`");
    assert_err(
        "fn f() { let mut x = 1; x = \"s\"; }",
        "expected `{integer}`, found `String`",
    );
    assert_err(
        "let g = 1; fn f() { g = 2; }",
        "cannot assign to the global `g`",
    );
    assert_err("fn f() { y = 2; }", "cannot find `y`");
    assert_err(
        "fn f(p: i32) { let mut x = 1; 1 + 2 = 3; }",
        "must be a variable",
    );
    assert_err(
        "struct P { a: i32 } fn f(p: P) { p.a = 1; }",
        "immutable variable `p`",
    );
}

#[test]
fn while_loops() {
    assert_clean(
        "fn f(n: i32) -> i32 { let mut i = 0; let mut s = 0; while i < n { s += i; i += 1; } s }",
    );
    assert_err(
        "fn f() { while 1 { } }",
        "expected `bool`, found `{integer}`",
    );
    assert_clean(
        "fn f(n: i32) -> i32 { let mut i = 0; while true { if i > n { return i; } i += 1; } 0 }",
    );
}

#[test]
fn for_loops() {
    assert_clean("fn f(n: i32) -> i32 { let mut s = 0; for i in 0..n { s += i; } s }");
    assert_clean("fn f(n: u8) -> u8 { let mut s: u8 = 0; for i in 1..=n { s += i; } s }");
    assert_clean("fn f(v: Vec<f64>) -> f64 { let mut s = 0.0; for x in v { s += x; } s }");
    assert_err("fn f() { for i in 0..1.5 { } }", "range bounds");
    assert_err(
        "fn f(a: i32, b: i64) { for i in a..b { } }",
        "different types",
    );
    assert_err("fn f(s: String) { for c in s { } }", "cannot iterate over");
    assert_err(
        "fn f() { for i in 0..3 { i = 1; } }",
        "immutable variable `i`",
    );
    assert_err("fn f() { for i in 0..3 { } let y = i; }", "cannot find `i`");
}

#[test]
fn arrays_and_indexing() {
    assert_clean("fn f() -> Vec<i32> { [1, 2, 3] }");
    assert_clean("fn f() -> Vec<f32> { [1, 2.5] }");
    assert_clean("fn f() -> Vec<String> { [] }");
    assert_err(
        "fn f() { let v = [1, \"a\"]; }",
        "expected `{integer}`, found `String`",
    );
    assert_err(
        "fn f() -> Vec<String> { [1] }",
        "expected `String`, found `{integer}`",
    );
    assert_clean("fn f(v: Vec<i32>, i: u8) -> i32 { v[i] + [4, 5][1] }");
    assert_err(
        "fn f(v: Vec<i32>) -> i32 { v[\"a\"] }",
        "indices must be integers",
    );
    assert_err("fn f(s: String) -> i32 { s[0] }", "cannot index into");
    assert_err(
        "fn f(v: Vec<i32>) -> String { v[0] }",
        "expected `String`, found `i32`",
    );
    assert_clean("fn f(v: Vec<Vec<i32>>) -> i32 { v[0][1] }");
}

// ---- literal type inference ----------------------------------------------------------------

#[test]
fn literals_are_pinned_by_their_uses() {
    // `s` starts as a bare literal and becomes u8 through its uses.
    assert_clean("fn f(n: u8) -> u8 { let mut s = 0; for i in 1..=n { s += i; } s }");
    assert_clean(
        "fn f(n: i64) -> i64 { let mut i = 0; let mut s = 0; while i < n { s += i; i += 1; } s }",
    );
    assert_clean("fn f(x: f32) -> f32 { let mut a = 0.5; a = a * x; a }");
    assert_clean("fn take(a: i64) {} fn f() { let x = 5; take(x); }");
    assert_clean("fn f() -> Vec<u8> { let v = [1, 2, 3]; v }");
    assert_clean(
        "fn f(c: bool) -> i64 { let a = 1; let b = 2; if c { a } else { b + 1_000_000_000_000 } }",
    );
    assert_clean("fn f() -> i32 { let a = 1; let b = a; let c = b; c }");
}

#[test]
fn conflicting_literal_uses_are_still_errors() {
    assert_err(
        "fn f(a: u8, b: i64) { let x = 1; let y = a + x; let z = b + x; }",
        "mismatched operand types",
    );
    assert_err(
        "fn f() { let x = 1; let s: String = x; }",
        "expected `String`, found `{integer}`",
    );
    assert_err(
        "fn f() { let x = 1.5; let y: i32 = x; }",
        "expected `i32`, found `{float}`",
    );
    assert_err(
        "fn f(a: f64) { let x = 1; let y = a + x; }",
        "mismatched operand types",
    );
    assert_err("fn f() -> bool { let x = 1; x }", "expected `bool`");
}

#[test]
fn unconstrained_literals_default_and_leave_no_variables_behind() {
    let (hir, errs) = check("fn f() -> i32 { let a = 1; let b = 2.5; let v = [3, 4]; a + v[0] }");
    assert!(errs.is_empty(), "{errs:?}");
    let dump = format!("{hir:?}");
    assert!(
        !dump.contains("IntVar") && !dump.contains("FloatVar"),
        "{dump}"
    );

    let (hir, errs) = check(
        "fn f() { let a = 1; let b = a + 2; let c = 1.5 * 2; for i in 0..10 { let d = i; } }",
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
    let decl = "struct P { x: i32, name: String, tags: Vec<i32> } struct W { p: P, n: u8 } ";
    assert_clean(&format!(
        "{decl}fn f() -> i32 {{ let mut p = P {{ x: 1, name: \"a\", tags: [1, 2] }}; p.x = 5; p.x += 1; p.tags[1] = p.x; p.name = p.name + \"!\"; p.x }}"
    ));
    assert_clean(&format!(
        "{decl}fn f(mut w: W) -> u8 {{ w.n = 3; w.p.x = 1; w.p.tags[0] += 2; w.n }}"
    ));
    assert_clean("fn fill(mut v: Vec<i32>, x: i32) { v[0] = x; v[1] = x * 2; }");
    assert_clean("fn f() { let mut grid = [[1, 2], [3, 4]]; grid[1][0] = 9; }");
    assert_err(
        &format!(
            "{decl}fn f() {{ let mut p = P {{ x: 1, name: \"a\", tags: [] }}; p.x = \"s\"; }}"
        ),
        "expected `i32`, found `String`",
    );
    assert_err(
        &format!("{decl}fn f() {{ let mut p = P {{ x: 1, name: \"a\", tags: [] }}; p.nope = 1; }}"),
        "no field `nope`",
    );
    assert_err(
        &format!("{decl}fn f() {{ let p = P {{ x: 1, name: \"a\", tags: [] }}; p.tags[0] = 1; }}"),
        "immutable variable `p`",
    );
    assert_err(
        "fn f() { let mut x = 1; x.y = 2; }",
        "cannot assign to a field of",
    );
    assert_err("fn f() { let mut x = 1; x[0] = 2; }", "cannot index into");
    assert_err("fn f(v: Vec<i32>) { v[0] = 1; }", "immutable variable `v`");
    assert_err(
        "fn f() { let mut v = [1, 2]; v[\"a\"] = 1; }",
        "indices must be integers",
    );
    assert_err(
        "fn f() { let mut v = [1, 2]; v[0] = \"s\"; }",
        "expected `{integer}`, found `String`",
    );
    assert_err(
        "fn f() { let mut v = [1, 2]; f()[0] = 1; }",
        "must be a variable, a field or an index",
    );
}

// ---- tensor operations ---------------------------------------------------------------------

#[test]
fn tensor_shapes_flow_through_operations() {
    let ok = "
        fn f(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [3, 4]>, v: Tensor<f32, [3]>) -> Tensor<f32, [2, 4]> {
            a |> std::tensor::matmul(b)
        }
        fn g(a: Tensor<f32, [2, 3]>, v: Tensor<f32, [3]>) -> Tensor<f32, [2]> { a |> std::tensor::matmul(v) }
        fn h(a: Tensor<f32, [4]>, b: Tensor<f32, [4]>) -> f32 { std::tensor::dot(a, b) }
        fn i(a: Tensor<f32, [4]>, b: Tensor<f32, [4]>) -> Tensor<f32, [4]> {
            std::tensor::add(a, b) |> std::tensor::relu() |> std::tensor::scale(0.5) |> std::tensor::softmax()
        }
        fn t(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [3, 2]> { std::tensor::transpose(a) }
        fn s(a: Tensor<f32, [2, 3]>) -> f32 { std::tensor::sum(a) + std::tensor::mean(a) }
        fn am(a: Tensor<f32, [5]>) -> u64 { std::tensor::argmax(a) }
        fn sym(a: Tensor<f32, [N]>, b: Tensor<f32, [8]>) -> Tensor<f32, [8]> { std::tensor::mul(a, b) }
    ";
    assert_clean(ok);
}

#[test]
fn tensor_shape_mismatches_are_compile_errors() {
    assert_err(
        "fn f(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [4, 5]>) { std::tensor::matmul(a, b); }",
        "inner dimensions differ: 3 and 4",
    );
    assert_err(
        "fn f(a: Tensor<f32, [4]>, b: Tensor<f32, [5]>) { std::tensor::add(a, b); }",
        "tensor shapes differ: [4] and [5]",
    );
    assert_err(
        "fn f(a: Tensor<f32, [4]>, b: Tensor<f64, [4]>) { std::tensor::add(a, b); }",
        "element types differ",
    );
    assert_err(
        "fn f(a: Tensor<f32, [4]>) -> Tensor<f32, [2]> { std::tensor::softmax(a) }",
        "expected `Tensor<f32, [2]>`",
    );
    assert_err(
        "fn f(a: Tensor<f32, [2, 2]>, b: Tensor<f32, [2, 2]>) { std::tensor::dot(a, b); }",
        "rank-1",
    );
    assert_err(
        "fn f(a: Tensor<f32, [4]>) { std::tensor::transpose(a); }",
        "rank-2",
    );
    assert_err(
        "fn f(a: i32) { std::tensor::relu(a); }",
        "must be a `Tensor`",
    );
    assert_err(
        "fn f(a: Tensor<f32, [2]>) { std::tensor::scale(a, \"x\"); }",
        "must be a float",
    );
}

#[test]
fn tensor_constructors_need_a_known_result_type() {
    assert_clean("fn f() -> Tensor<f32, [2, 3]> { std::tensor::zeros() }");
    assert_clean("fn f(v: Vec<f32>) -> Tensor<f32, [3]> { std::tensor::from_vec(v) }");
    assert_clean("fn f(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [6]> { std::tensor::reshape(a) }");
    assert_err("fn f() { let t = std::tensor::zeros(); }", "annotate it");
    assert_err(
        "fn f(a: Tensor<f32, [2, 3]>) -> Tensor<f32, [5]> { std::tensor::reshape(a) }",
        "cannot reshape [2, 3] (6 elements) into [5] (5 elements)",
    );
    assert_clean("fn f(a: Tensor<f32, [4]>) -> Vec<f32> { std::tensor::to_vec(a) }");
}

// ---- tables --------------------------------------------------------------------------------

const ROW: &str = "struct Row { id: i64, score: f32, name: String, ok: bool } ";

#[test]
fn tables_have_typed_columns() {
    assert_clean(&format!(
        "{ROW}
        fn f(t: Table<Row>) -> Tensor<f32, [N]> {{ t.score }}
        fn g(t: Table<Row>) -> Vec<String> {{ t.name }}
        fn h(t: Table<Row>) -> u64 {{ t.len() }}
        fn i(t: Table<Row>) -> Row {{ t[0] }}
        fn j(t: Table<Row>) -> f32 {{ let mut s = 0.0; for r in t {{ s += r.score; }} s }}
        fn k(t: Table<Row>) -> Tensor<i64, [N]> {{ t.id }}"
    ));
    assert_err(
        &format!("{ROW}fn f(t: Table<Row>) {{ t.nope; }}"),
        "no column `nope`",
    );
    assert_err(
        &format!("{ROW}fn f(t: Table<Row>) -> Vec<String> {{ t.score }}"),
        "expected `Vec<String>`",
    );
}

#[test]
fn table_schemas_must_be_flat_structs() {
    assert_err(
        "struct R { xs: Vec<i32> } fn f(t: Table<R>) {}",
        "table columns must be",
    );
    assert_err("fn f(t: Table<i32>) {}", "must be a struct");
    assert_err("fn f(t: Table) {}", "takes one type argument");
    assert_err("struct E {} fn f(t: Table<E>) {}", "no fields");
}

#[test]
fn table_functions() {
    assert_clean(&format!(
        "{ROW}
        fn a() -> Table<Row> {{ std::table::read_csv(\"x.csv\") }}
        fn b() -> Table<Row> {{ std::table::read_parquet(\"x.parquet\") }}
        fn c(t: Table<Row>) -> Table<Row> {{ t |> std::table::sql(\"SELECT * FROM t WHERE score > 1\") }}
        fn d(rows: Vec<Row>) -> Table<Row> {{ std::table::from_rows(rows) }}
        fn e(t: Table<Row>) -> Vec<Row> {{ std::table::to_rows(t) }}
        fn f(t: Table<Row>) -> Table<Row> {{ t |> std::table::filter(|r| r.score > 0.5) }}
        fn g(t: Table<Row>) {{ std::table::write_csv(t, \"o.csv\"); std::table::write_parquet(t, \"o.parquet\"); }}
        fn h(t: Table<Row>) -> u64 {{ std::table::len(t) }}"
    ));
    assert_err(
        &format!("{ROW}fn f() {{ let t = std::table::read_csv(\"x\"); }}"),
        "annotate the result",
    );
    assert_err(
        &format!("{ROW}fn f(t: Table<Row>) -> Table<Row> {{ std::table::filter(t, |r| r.nope) }}"),
        "no field `nope`",
    );
    assert_err(
        &format!("{ROW}fn f(t: Table<Row>) {{ std::table::sql(t, 1); }}"),
        "must be `String`",
    );
}

#[test]
fn tensors_can_be_indexed() {
    assert_clean("fn f(t: Tensor<f32, [4]>) -> f32 { t[0] }");
    assert_clean("fn f(t: Tensor<f32, [2, 3]>) -> Tensor<f32, [3]> { t[1] }");
    assert_clean("fn f(t: Tensor<f32, [2, 3]>) -> f32 { t[1][2] }");
    assert_err(
        "fn f(t: Tensor<f32, [4]>) -> i32 { t[0] }",
        "expected `i32`, found `f32`",
    );
}

#[test]
fn std_registry_can_be_browsed() {
    use novar_hir::stdlib::{ItemKind, children, methods_of};
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
