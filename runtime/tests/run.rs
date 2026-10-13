use novar_runtime::{Interp, Value, embed_text};
use serde_json::json;

fn load(src: &str) -> Interp {
    let (ast, perrs) = novar_parser::parse(src);
    assert!(perrs.is_empty(), "parse: {perrs:?}");
    let (hir, diags) = novar_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "check: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    Interp::new(hir).expect("globals should initialise")
}

fn run(src: &str, f: &str, args: Vec<Value>) -> Value {
    load(src)
        .call(f, args)
        .unwrap_or_else(|e| panic!("runtime error: {e}"))
}

fn int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected int, got {other:?}"),
    }
}

fn text(v: Value) -> String {
    match v {
        Value::Str(s) => s.to_string(),
        other => panic!("expected string, got {other:?}"),
    }
}

#[test]
fn arithmetic_and_recursion() {
    let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } fib(n - 1) + fib(n - 2) }";
    assert_eq!(int(run(src, "fib", vec![Value::Int(15)])), 610);
    assert_eq!(
        int(run("fn f() -> i32 { 2 + 3 * 4 - 10 / 5 % 3 }", "f", vec![])),
        12
    );
    assert_eq!(int(run("fn f() -> i32 { -(3) }", "f", vec![])), -3);
}

#[test]
fn floats_and_literal_adaptation() {
    let v = run(
        "fn f(a: f32) -> f32 { a * 2 + 0.5 }",
        "f",
        vec![Value::Float(1.25)],
    );
    assert!(
        matches!(v, Value::Float(x) if (x - 3.0).abs() < 1e-9),
        "{v:?}"
    );
}

#[test]
fn let_blocks_and_early_return() {
    let src = "fn f(x: i32) -> i32 { let y = x + 1; if y > 5 { return 100; } let z = { let w = y * 2; w + 1 }; z }";
    assert_eq!(int(run(src, "f", vec![Value::Int(1)])), 5);
    assert_eq!(int(run(src, "f", vec![Value::Int(9)])), 100);
}

#[test]
fn pipes_and_strings() {
    let src = r#"
        fn shout(s: String) -> String { s + "!" }
        fn f(s: String) -> String {
            s |> std::text::clean_whitespace() |> std::text::truncate(11) |> shout() |> std::text::uppercase()
        }
    "#;
    assert_eq!(
        text(run(src, "f", vec![Value::str("  hello    big   world  ")])),
        "HELLO BIG W!"
    );
}

#[test]
fn fstrings() {
    let src = r#"fn f(n: i32, s: String) -> String { f"{s}: {n + 1} \"ok\"" }"#;
    assert_eq!(
        text(run(src, "f", vec![Value::Int(1), Value::str("n")])),
        "n: 2 \"ok\""
    );
}

#[test]
fn iterators_and_closures() {
    let src = r#"
        fn f(v: Vec<i32>, k: i32) -> String {
            v |> std::iter::filter(|x| x > 1) |> std::iter::map(|x| f"{x * k}") |> std::string::join("-")
        }
    "#;
    let v = Value::vec(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(text(run(src, "f", vec![v, Value::Int(10)])), "20-30");
}

#[test]
fn structs_and_fields() {
    let src = "struct P { x: i32, name: String } fn f() -> String { let p = P { name: \"a\", x: 4 }; f\"{p.name}{p.x}\" }";
    assert_eq!(text(run(src, "f", vec![])), "a4");
}

#[test]
fn short_circuit_avoids_errors() {
    let src = "fn f(d: i32) -> bool { d != 0 && 10 / d > 1 }";
    assert!(matches!(
        run(src, "f", vec![Value::Int(0)]),
        Value::Bool(false)
    ));
    assert!(matches!(
        run(src, "f", vec![Value::Int(2)]),
        Value::Bool(true)
    ));
}

#[test]
fn runtime_errors_are_reported_not_panics() {
    let div = load("fn f(d: i32) -> i32 { 1 / d }");
    assert!(
        div.call("f", vec![Value::Int(0)])
            .unwrap_err()
            .message
            .contains("division by zero")
    );
    let ovf = load("fn f(a: i64) -> i64 { a * a }");
    assert!(
        ovf.call("f", vec![Value::Int(i64::MAX)])
            .unwrap_err()
            .message
            .contains("overflow")
    );
    let rec = load("fn f(n: i32) -> i32 { f(n + 1) }");
    assert!(
        rec.call("f", vec![Value::Int(0)])
            .unwrap_err()
            .message
            .contains("stack overflow")
    );
    assert!(div.call("nope", vec![]).is_err());
}

#[test]
fn globals_are_evaluated_once_in_order() {
    let src = "let a = 2; let b = a * 21; fn f() -> i32 { b }";
    assert_eq!(int(run(src, "f", vec![])), 42);
}

#[test]
fn endpoint_results_and_errors() {
    let src = r#"
        import std::http::Status;
        struct In { n: i32 }
        struct Out { double: i32 }
        @endpoint(path = "/d", method = "POST")
        pub fn d(p: In) -> Result<Out, Status> {
            if p.n < 0 { return Err(Status::BadRequest("negative")); }
            Ok(Out { double: p.n * 2 })
        }
    "#;
    let i = load(src);
    assert_eq!(i.endpoint_names(), ["d"]);
    assert_eq!(
        i.call_endpoint("d", &json!({"n": 4})).unwrap(),
        json!({"double": 8})
    );
    let e = i.call_endpoint("d", &json!({"n": -1})).unwrap_err();
    assert_eq!((e.status, e.message.as_str()), (400, "negative"));
    assert_eq!(
        i.call_endpoint("d", &json!({"n": "x"})).unwrap_err().status,
        400
    );
    assert_eq!(i.call_endpoint("nope", &json!({})).unwrap_err().status, 500);
}

#[test]
fn tensors_round_trip_through_json() {
    let src = r#"
        struct T { v: Tensor<f32, [2, 2]> }
        @endpoint(path = "/t", method = "POST")
        fn t(p: T) -> Result<T, String> { Ok(p) }
    "#;
    let i = load(src);
    let body = json!({"v": [[1.0, 2.0], [3.0, 4.0]]});
    assert_eq!(i.call_endpoint("t", &body).unwrap(), body);
    assert_eq!(
        i.call_endpoint("t", &json!({"v": [[1.0], [3.0, 4.0]]}))
            .unwrap_err()
            .status,
        400
    );
}

#[test]
fn embeddings_are_deterministic_and_similarity_aware() {
    let a = embed_text("tensors have a static shape");
    let b = embed_text("a tensor has static shapes and tensors");
    let c = embed_text("completely unrelated banana bread");
    assert_eq!(a.len(), 768);
    assert_eq!(a, embed_text("tensors have a static shape"));
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
    assert!((dot(&a, &a) - 1.0).abs() < 1e-4);
    assert!(dot(&a, &b) > dot(&a, &c));
}

#[test]
fn rag_example_end_to_end() {
    let src = include_str!("../../examples/rag-service.nv");
    let i = load(src);
    let db = i.db_for("postgresql://localhost:5432/vectors");
    for (id, text) in [
        (
            "t1",
            "Tensors in Novar carry a static shape checked at compile time",
        ),
        (
            "p1",
            "The pipeline operator composes functions left to right",
        ),
        ("b1", "Banana bread needs ripe bananas"),
    ] {
        db.insert("knowledge_base", id, text, embed_text(text));
    }

    let out = i
        .call_endpoint(
            "handle_rag",
            &json!({"query": "how are tensor shapes checked", "top_k": 2}),
        )
        .unwrap();
    let ctx = out["retrieved_context"].as_array().unwrap();
    assert_eq!(ctx.len(), 2);
    assert!(
        ctx[0].as_str().unwrap().starts_with("Tensors in Novar"),
        "{ctx:?}"
    );
    assert_eq!(out["query"], "how are tensor shapes checked");
    assert!(
        out["answer"]
            .as_str()
            .unwrap()
            .contains("how are tensor shapes checked"),
        "{out}"
    );

    let e = i
        .call_endpoint("handle_rag", &json!({"query": "", "top_k": 1}))
        .unwrap_err();
    assert_eq!(
        (e.status, e.message.as_str()),
        (400, "Query string cannot be empty.")
    );
}
