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
    assert_clean("import std::db::Client; let c = Client::new(); fn f() { let x = c; }");
    // A binding is not visible inside its own initializer.
    assert_err("fn f() { let x = x; }", "cannot find `x`");
}

#[test]
fn literals_adapt_to_context() {
    assert_clean("fn f(a: f32) -> f32 { a + 1 }");
    assert_clean("fn f() { let x: f32 = 1.5; let y: u8 = 3; let z: f64 = 2; }");
    assert_err(
        "fn f() { let x: i32 = 1.5; }",
        "expected `i32`, found `f64`",
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
    assert_err("fn f() { if 1 { } }", "expected `bool`, found `i32`");
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
        "expected `String`, found `i32`",
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
