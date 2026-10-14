use novar_runtime::{Interp, Value, embed_text};
use serde_json::json;
use std::sync::Arc;

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

fn float(v: Value) -> f64 {
    match v {
        Value::Float(x) => x,
        other => panic!("expected float, got {other:?}"),
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
    let db = i.db_for("postgresql://localhost:5432/vectors").unwrap();
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
        db.insert("knowledge_base", id, text, embed_text(text))
            .unwrap();
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

#[test]
fn print_goes_to_the_configured_output() {
    use std::sync::{Arc, Mutex};
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Sink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let buf = Arc::new(Mutex::new(Vec::new()));
    let i = load(r#"fn main() -> i32 { std::io::print("a"); std::io::println("b"); 0 }"#);
    i.set_output(Sink(buf.clone()));
    i.call("main", vec![]).unwrap();
    assert_eq!(
        String::from_utf8(buf.lock().unwrap().clone()).unwrap(),
        "ab\n"
    );
}

#[test]
fn mutation_and_while() {
    let src =
        "fn sum(n: i32) -> i32 { let mut i = 0; let mut s = 0; while i < n { s += i; i += 1; } s }";
    assert_eq!(int(run(src, "sum", vec![Value::Int(10)])), 45);
    assert_eq!(int(run(src, "sum", vec![Value::Int(0)])), 0);
    let early =
        "fn f() -> i32 { let mut i = 0; while true { if i == 7 { return i * 2; } i += 1; } 0 }";
    assert_eq!(int(run(early, "f", vec![])), 14);
    let strs = "fn f() -> String { let mut s = \"\"; let mut i = 0; while i < 3 { s = s + f\"{i},\"; i += 1; } s }";
    assert_eq!(text(run(strs, "f", vec![])), "0,1,2,");
}

#[test]
fn for_loops_over_ranges_and_vectors() {
    let src = r#"
        fn tri(n: i32) -> i32 { let mut s = 0; for i in 0..n { s += i; } s }
        fn incl(n: i32) -> i32 { let mut s = 0; for i in 1..=n { s += i; } s }
        fn dot(a: Vec<i32>, b: Vec<i32>) -> i32 {
            let mut s = 0;
            for i in 0..std::iter::count(a) { s += a[i] * b[i]; }
            s
        }
        fn total(v: Vec<f64>) -> f64 { let mut s = 0.0; for x in v { s += x; } s }
        fn find(v: Vec<i32>, t: i32) -> i32 {
            let mut idx = 0;
            for x in v { if x == t { return idx; } idx += 1; }
            0 - 1
        }
        fn nested(n: i32) -> i32 { let mut c = 0; for i in 0..n { for j in 0..i { c += j; } } c }
        fn empty(n: i32) -> i32 { let mut c = 0; for i in 5..n { c += 1; } c }
    "#;
    let ints = |v: &[i64]| Value::vec(v.iter().map(|&i| Value::Int(i)).collect());
    assert_eq!(int(run(src, "tri", vec![Value::Int(5)])), 10);
    assert_eq!(int(run(src, "incl", vec![Value::Int(5)])), 15);
    assert_eq!(
        int(run(src, "dot", vec![ints(&[1, 2, 3]), ints(&[4, 5, 6])])),
        32
    );
    assert_eq!(
        int(run(src, "find", vec![ints(&[5, 6, 7]), Value::Int(7)])),
        2
    );
    assert_eq!(
        int(run(src, "find", vec![ints(&[5, 6, 7]), Value::Int(9)])),
        -1
    );
    assert_eq!(int(run(src, "nested", vec![Value::Int(4)])), 4);
    assert_eq!(int(run(src, "empty", vec![Value::Int(3)])), 0);
    let floats = Value::vec(vec![Value::Float(1.5), Value::Float(2.0)]);
    assert!(matches!(run(src, "total", vec![floats]), Value::Float(x) if x == 3.5));
}

#[test]
fn arrays_indexing_and_bounds_errors() {
    let src = "fn f(i: i32) -> i32 { let v = [10, 20, 30]; v[i] } fn g() -> i32 { [[1, 2], [3, 4]][1][0] }";
    assert_eq!(int(run(src, "f", vec![Value::Int(2)])), 30);
    assert_eq!(int(run(src, "g", vec![])), 3);
    let i = load(src);
    let e = i.call("f", vec![Value::Int(3)]).unwrap_err();
    assert!(
        e.message
            .contains("index out of bounds: the length is 3 but the index is 3"),
        "{e}"
    );
    assert!(
        i.call("f", vec![Value::Int(-1)])
            .unwrap_err()
            .message
            .contains("out of bounds")
    );
}

#[test]
fn inclusive_range_does_not_overflow() {
    let src = "fn f() -> i64 { let hi: i64 = 9223372036854775807; let mut c: i64 = 0; for i in hi - 1..=hi { c += 1; } c }";
    assert_eq!(int(run(src, "f", vec![])), 2);
}

#[test]
fn assignment_to_vector_elements_and_struct_fields() {
    let src = r#"
        struct P { x: i32, name: String, tags: Vec<i32> }
        struct W { p: P, n: i32 }
        fn elems() -> i32 { let mut v = [1, 2, 3]; v[0] = 10; v[2] += v[0]; v[1] *= 5; v[0] + v[1] + v[2] }
        fn fields() -> String {
            let mut p = P { x: 1, name: "a", tags: [7, 8] };
            p.x = 5; p.x += 1; p.name = p.name + "!"; p.tags[1] = p.x;
            f"{p.name}{p.x}{p.tags[0]}{p.tags[1]}"
        }
        fn nested() -> i32 {
            let mut w = W { p: P { x: 1, name: "n", tags: [0, 0] }, n: 2 };
            w.p.x = 40; w.p.tags[0] = 2; w.n = 7;
            w.p.x + w.p.tags[0] + w.n
        }
        fn grid() -> i32 { let mut g = [[1, 2], [3, 4]]; g[1][0] = 9; g[0][1] += g[1][0]; g[1][0] * 100 + g[0][1] }
        fn fill(mut v: Vec<i32>, x: i32) { v[0] = x; v[1] = x * 2; }
        fn via_param() -> i32 { let v = [0, 0]; fill(v, 4); v[0] + v[1] }
        fn value_semantics() -> i32 {
            let mut a = P { x: 1, name: "a", tags: [] };
            let b = a;
            a.x = 99;
            b.x
        }
        fn reference_semantics() -> i32 { let mut a = [1, 2]; let b = a; a[0] = 50; b[0] }
        fn loop_writes() -> i32 {
            let mut v = [0, 0, 0, 0];
            for i in 0..4 { v[i] = i * i; }
            let mut s = 0;
            for x in v { s += x; }
            s
        }
        fn mut_param_scalar(mut n: i32) -> i32 { n += 1; n * 2 }
    "#;
    assert_eq!(int(run(src, "elems", vec![])), 10 + 10 + 13);
    assert_eq!(
        text(run(src, "fields", vec![])),
        "a!6".to_string() + "7" + "6"
    );
    assert_eq!(int(run(src, "nested", vec![])), 49);
    assert_eq!(int(run(src, "grid", vec![])), 900 + 11);
    assert_eq!(
        int(run(src, "via_param", vec![])),
        12,
        "Vec is passed by reference"
    );
    assert_eq!(
        int(run(src, "value_semantics", vec![])),
        1,
        "structs are copied"
    );
    assert_eq!(
        int(run(src, "reference_semantics", vec![])),
        50,
        "Vec copies share elements"
    );
    assert_eq!(int(run(src, "loop_writes", vec![])), 14);
    assert_eq!(int(run(src, "mut_param_scalar", vec![Value::Int(3)])), 8);
}

#[test]
fn assignment_bounds_errors() {
    let i = load(
        "fn f(k: i32) { let mut v = [1, 2]; v[k] = 5; } fn g() { let mut v = [[1], [2]]; v[1][3] = 0; }",
    );
    assert!(
        i.call("f", vec![Value::Int(2)])
            .unwrap_err()
            .message
            .contains("out of bounds: the length is 2 but the index is 2")
    );
    assert!(i.call("f", vec![Value::Int(-1)]).is_err());
    assert!(
        i.call("g", vec![])
            .unwrap_err()
            .message
            .contains("out of bounds")
    );
    assert!(i.call("f", vec![Value::Int(1)]).is_ok());
}

#[test]
fn a_step_limit_stops_runaway_programs() {
    let i = load(
        "fn spin() -> i32 { let mut n = 0; while true { n += 1; } n } fn quick() -> i32 { 1 + 1 }",
    );
    i.set_step_limit(10_000);
    let e = i.call("spin", vec![]).unwrap_err();
    assert!(e.message.contains("step limit"), "{e}");
    i.set_step_limit(10_000);
    assert_eq!(
        int(i.call("quick", vec![]).unwrap()),
        2,
        "a fresh budget runs normal code"
    );
}

#[test]
fn deep_recursion_works_on_a_small_thread_stack_and_runaway_recursion_is_an_error() {
    let src = "fn sum(n: i64) -> i64 { if n == 0 { 0 } else { n + sum(n - 1) } } fn forever(n: i64) -> i64 { forever(n + 1) }";
    let i = std::sync::Arc::new(load(src));
    let worker = i.clone();
    // A 256 KiB thread would overflow after a few hundred native frames without stack growth.
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let deep = worker.call("sum", vec![Value::Int(1500)]).unwrap();
            let runaway = worker.call("forever", vec![Value::Int(0)]).unwrap_err();
            (int(deep), runaway.message)
        })
        .unwrap();
    let (deep, runaway) = handle.join().expect("no native stack overflow");
    assert_eq!(deep, 1500 * 1501 / 2);
    assert!(runaway.contains("stack overflow"), "{runaway}");
}

#[test]
fn top_level_functions_are_values() {
    let src = r#"
        fn double(x: i64) -> i64 { x * 2 }
        fn apply_all(v: Vec<i64>) -> Vec<i64> { std::iter::map(v, double) }
        fn total(v: Vec<i64>) -> i64 { let mut s = 0; for x in apply_all(v) { s += x; } s }
    "#;
    let v = Value::vec((1..=4).map(Value::Int).collect());
    assert_eq!(int(run(src, "total", vec![v])), 20);
}

#[test]
fn dist_map_spreads_work_over_threads_and_keeps_order() {
    let src = r#"
        fn square(x: i64) -> i64 { x * x }
        fn all(v: Vec<i64>) -> Vec<i64> { std::dist::map(v, square) }
        fn fails(x: i64) -> i64 { 10 / x }
        fn bad(v: Vec<i64>) -> Vec<i64> { std::dist::map(v, fails) }
    "#;
    let i = load(src);
    let input = Value::vec((0..200).map(Value::Int).collect());
    let Value::Vec(out) = i.call("all", vec![input]).unwrap() else {
        panic!()
    };
    let squares: Vec<i64> = out.snapshot().iter().map(|v| int(v.clone())).collect();
    assert_eq!(squares, (0..200).map(|x| x * x).collect::<Vec<_>>());
    // An error in any call fails the whole map.
    let e = i
        .call("bad", vec![Value::vec(vec![Value::Int(5), Value::Int(0)])])
        .unwrap_err();
    assert!(e.message.contains("division by zero"), "{e}");
}

#[test]
fn dist_map_can_use_a_custom_executor() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counting(AtomicUsize);
    impl novar_runtime::Executor for Counting {
        fn map(
            &self,
            function: &str,
            items: Vec<serde_json::Value>,
        ) -> Result<Vec<serde_json::Value>, String> {
            self.0.fetch_add(items.len(), Ordering::SeqCst);
            Ok(items
                .iter()
                .map(|j| serde_json::json!(j.as_i64().unwrap() + 1000 + function.len() as i64))
                .collect())
        }
    }
    let i = load(
        "fn work(x: i64) -> i64 { x } fn all(v: Vec<i64>) -> Vec<i64> { std::dist::map(v, work) }",
    );
    let exec = std::sync::Arc::new(Counting(AtomicUsize::new(0)));
    i.set_executor(exec.clone());
    let Value::Vec(out) = i
        .call("all", vec![Value::vec(vec![Value::Int(1), Value::Int(2)])])
        .unwrap()
    else {
        panic!()
    };
    let got: Vec<i64> = out.snapshot().into_iter().map(int).collect();
    assert_eq!(got, [1005, 1006], "the executor decided the results");
    assert_eq!(exec.0.load(Ordering::SeqCst), 2);
}

#[test]
fn call_json_converts_by_the_parameter_types() {
    let i = load(
        "struct P { x: i64, y: i64 } fn add(p: P, k: i64) -> i64 { p.x + p.y + k } fn boom() -> Result<i64, String> { Err(\"no\") }",
    );
    assert_eq!(
        i.call_json("add", &[json!({"x": 1, "y": 2}), json!(10)])
            .unwrap(),
        json!(13)
    );
    assert!(
        i.call_json("add", &[json!({"x": 1}), json!(1)])
            .unwrap_err()
            .contains("argument `p`")
    );
    assert!(
        i.call_json("add", &[json!(1)])
            .unwrap_err()
            .contains("takes 2")
    );
    assert!(i.call_json("boom", &[]).unwrap_err().contains("Err(no)"));
}

#[test]
fn break_and_continue_in_every_kind_of_loop() {
    let src = "
        fn range_break(n: i64) -> i64 { let mut s = 0; for i in 0..n { if i == 5 { break; } s += i; } s }
        fn range_continue(n: i64) -> i64 { let mut s = 0; for i in 0..n { if i % 2 == 0 { continue; } s += i; } s }
        fn while_loop() -> i64 { let mut i = 0; let mut s = 0; while true { i += 1; if i > 10 { break; } if i % 3 == 0 { continue; } s += i; } s }
        fn nested() -> i64 {
            let mut c = 0;
            for a in 0..5 { for b in 0..5 { if b > a { break; } c += 1; } }
            c
        }
        fn return_inside(n: i64) -> i64 { for i in 0..n { if i == 3 { return i * 100; } } 0 - 1 }
    ";
    assert_eq!(int(run(src, "range_break", vec![Value::Int(100)])), 10);
    assert_eq!(int(run(src, "range_continue", vec![Value::Int(10)])), 25);
    assert_eq!(int(run(src, "while_loop", vec![])), 37);
    assert_eq!(int(run(src, "nested", vec![])), 15);
    assert_eq!(int(run(src, "return_inside", vec![Value::Int(10)])), 300);
}

#[test]
fn enums_and_match() {
    let src = "
        enum Shape { Circle(f64), Rect(f64, f64), Empty }
        fn area(s: Shape) -> f64 {
            match s { Shape::Circle(r) => r * r * 3.0, Shape::Rect(w, h) => w * h, Shape::Empty => 0.0 }
        }
        fn total() -> f64 { area(Shape::Circle(2.0)) + area(Shape::Rect(2.0, 3.5)) + area(Shape::Empty) }
        fn describe(n: i64) -> String { match n { 0 => \"zero\", 1 => \"one\", _ => \"many\" } }
        fn safe_div(a: i64, b: i64) -> Result<i64, String> {
            if b == 0 { return Err(\"divide by zero\"); }
            Ok(a / b)
        }
        fn show(a: i64, b: i64) -> String {
            match safe_div(a, b) { Ok(v) => f\"ok {v}\", Err(e) => f\"error: {e}\" }
        }
        fn same(a: Shape, b: Shape) -> bool { a == b }
        fn pick(flag: bool) -> Shape { if flag { Shape::Rect(1.0, 2.0) } else { Shape::Empty } }
        enum Op { Add(i64), Neg, Twice(Shape) }
        fn nested(o: Op) -> f64 {
            match o { Op::Add(n) => 1.0, Op::Neg => 2.0, Op::Twice(Shape::Circle(r)) => r, Op::Twice(_) => 9.0 }
        }
    ";
    assert_eq!(float(run(src, "total", vec![])), 12.0 + 7.0);
    assert_eq!(text(run(src, "describe", vec![Value::Int(1)])), "one");
    assert_eq!(text(run(src, "describe", vec![Value::Int(42)])), "many");
    assert_eq!(
        text(run(src, "show", vec![Value::Int(9), Value::Int(3)])),
        "ok 3"
    );
    assert_eq!(
        text(run(src, "show", vec![Value::Int(9), Value::Int(0)])),
        "error: divide by zero"
    );
    let interp = load(src);
    let a = interp.call("pick", vec![Value::Bool(true)]).unwrap();
    let b = interp.call("pick", vec![Value::Bool(true)]).unwrap();
    let c = interp.call("pick", vec![Value::Bool(false)]).unwrap();
    assert!(matches!(
        interp.call("same", vec![a.clone(), b]),
        Ok(Value::Bool(true))
    ));
    assert!(matches!(
        interp.call("same", vec![a.clone(), c]),
        Ok(Value::Bool(false))
    ));
    assert_eq!(a.to_json(), json!({ "Rect": [1.0, 2.0] }));
    assert_eq!(a.to_string(), "Shape::Rect(1, 2)");
    // JSON in, JSON out (this is how endpoints and workers talk).
    let area = interp
        .call_json("area", &[json!({ "Circle": 3.0 })])
        .unwrap();
    assert_eq!(area, json!(27.0));
    assert_eq!(
        interp.call_json("area", &[json!("Empty")]).unwrap(),
        json!(0.0)
    );
    assert!(interp.call_json("area", &[json!("Triangle")]).is_err());
    assert_eq!(
        interp
            .call_json("nested", &[json!({ "Twice": { "Circle": 4.0 } })])
            .unwrap(),
        json!(4.0)
    );
}

#[test]
fn string_and_vec_methods() {
    let src = "
        fn has(s: String, t: String) -> bool { s.contains(t) }
        fn edges(s: String) -> bool { s.starts_with(\"ab\") && s.ends_with(\"yz\") }
        fn fix(s: String) -> String { s.replace(\"cat\", \"dog\") }
        fn words(s: String) -> u64 { s.split(\" \").len() }
        fn second(s: String) -> String { let parts = s.split(\",\"); parts[1] }
        fn collect(n: i64) -> u64 {
            let mut v = [0];
            for i in 1..n { v.push(i * i); }
            v.len()
        }
        fn shares(n: i64) -> bool { let v = [1, 2]; v.push(n); v.contains(n) }
    ";
    assert!(matches!(
        run(
            src,
            "has",
            vec![Value::str("hello world"), Value::str("o w")]
        ),
        Value::Bool(true)
    ));
    assert!(matches!(
        run(src, "has", vec![Value::str("hello"), Value::str("xyz")]),
        Value::Bool(false)
    ));
    assert!(matches!(
        run(src, "edges", vec![Value::str("ab...yz")]),
        Value::Bool(true)
    ));
    assert!(matches!(
        run(src, "edges", vec![Value::str("ab...")]),
        Value::Bool(false)
    ));
    assert_eq!(
        text(run(src, "fix", vec![Value::str("a cat and a cat")])),
        "a dog and a dog"
    );
    assert_eq!(int(run(src, "words", vec![Value::str("one two three")])), 3);
    assert_eq!(text(run(src, "second", vec![Value::str("a,b,c")])), "b");
    assert_eq!(int(run(src, "collect", vec![Value::Int(5)])), 5);
    assert!(matches!(
        run(src, "shares", vec![Value::Int(9)]),
        Value::Bool(true)
    ));
}

#[test]
fn option_values_and_matching() {
    let src = "
        fn parse(s: String) -> Option<i64> { s.to_int() }
        fn describe(s: String) -> String {
            match parse(s) { Some(n) => f\"number {n}\", None => \"not a number\" }
        }
        fn pick(flag: bool) -> Option<String> { if flag { Some(\"yes\") } else { None } }
        fn first_or(v: Vec<i64>, d: i64) -> i64 { v.first().unwrap_or(d) }
        fn last_plus(v: Vec<i64>) -> i64 { match v.last() { Some(x) => x + 1, None => 0 } }
        fn counts(v: Vec<i64>) -> bool { v.get(1).is_some() && v.get(9).is_none() }
        fn drain() -> i64 { let v = [1, 2, 3]; let mut total = 0; while v.len() > 0 { match v.pop() { Some(x) => { total += x; } None => { break; } } } total }
        fn where_is(s: String) -> Option<u64> { s.find(\"lo\") }
        fn sorted() -> Vec<i64> { let v = [3, 1, 2]; v.sort(); v }
        fn reversed() -> Vec<String> { let v = \"c b a\".split(\" \"); v.sort(); v.reverse(); v }
    ";
    assert_eq!(
        text(run(src, "describe", vec![Value::str("42")])),
        "number 42"
    );
    assert_eq!(
        text(run(src, "describe", vec![Value::str("x1")])),
        "not a number"
    );
    assert_eq!(
        run(src, "pick", vec![Value::Bool(true)]).to_json(),
        json!("yes")
    );
    assert_eq!(
        run(src, "pick", vec![Value::Bool(false)]).to_json(),
        json!(null)
    );
    assert_eq!(
        run(src, "pick", vec![Value::Bool(true)]).to_string(),
        "Some(yes)"
    );
    let ints = |xs: &[i64]| Value::vec(xs.iter().map(|x| Value::Int(*x)).collect());
    assert_eq!(int(run(src, "first_or", vec![ints(&[]), Value::Int(7)])), 7);
    assert_eq!(
        int(run(src, "first_or", vec![ints(&[5, 6]), Value::Int(7)])),
        5
    );
    assert_eq!(int(run(src, "last_plus", vec![ints(&[5, 6])])), 7);
    assert!(matches!(
        run(src, "counts", vec![ints(&[1, 2])]),
        Value::Bool(true)
    ));
    assert_eq!(int(run(src, "drain", vec![])), 6);
    assert_eq!(
        run(src, "where_is", vec![Value::str("hello")]).to_json(),
        json!(3)
    );
    assert_eq!(run(src, "sorted", vec![]).to_json(), json!([1, 2, 3]));
    assert_eq!(
        run(src, "reversed", vec![]).to_json(),
        json!(["c", "b", "a"])
    );
    let interp = load(src);
    assert_eq!(
        interp.call_json("pick", &[json!(true)]).unwrap(),
        json!("yes")
    );
    assert_eq!(interp.call_json("parse", &[json!("9")]).unwrap(), json!(9));
}

#[test]
fn maps() {
    let src = "
        fn word_counts(text: String) -> Map<String, i64> {
            let counts: Map<String, i64> = std::map::new();
            for w in text.split(\" \") {
                let n = counts.get(w).unwrap_or(0);
                counts.insert(w, n + 1);
            }
            counts
        }
        fn lookup(text: String, w: String) -> i64 {
            match word_counts(text).get(w) { Some(n) => n, None => 0 - 1 }
        }
        fn size(text: String) -> u64 { word_counts(text).len() }
        fn keys_of(text: String) -> Vec<String> { word_counts(text).keys() }
        fn ids() -> Map<i64, String> {
            let m: Map<i64, String> = std::map::new();
            m.insert(2, \"two\");
            m.insert(1, \"one\");
            m.remove(2);
            m
        }
        fn has(k: i64, m: Map<i64, String>) -> bool { m.contains_key(k) }
    ";
    let t = Value::str("a b a c a b");
    assert_eq!(int(run(src, "lookup", vec![t.clone(), Value::str("a")])), 3);
    assert_eq!(
        int(run(src, "lookup", vec![t.clone(), Value::str("z")])),
        -1
    );
    assert_eq!(int(run(src, "size", vec![t.clone()])), 3);
    assert_eq!(
        run(src, "keys_of", vec![t.clone()]).to_json(),
        json!(["a", "b", "c"])
    );
    assert_eq!(
        run(src, "word_counts", vec![t]).to_json(),
        json!({"a": 3, "b": 2, "c": 1})
    );
    assert_eq!(run(src, "ids", vec![]).to_json(), json!({"1": "one"}));
    let interp = load(src);
    assert_eq!(
        interp
            .call_json("has", &[json!(5), json!({"5": "five"})])
            .unwrap(),
        json!(true)
    );
    assert_eq!(
        interp
            .call_json("has", &[json!(6), json!({"5": "five"})])
            .unwrap(),
        json!(false)
    );
}

#[test]
fn closures_capture_variables() {
    let src = "
        fn shifted(v: Vec<i64>, k: i64) -> Vec<i64> { v |> std::iter::map(|x| x + k) |> std::collect() }
        fn named(k: i64) -> i64 { let add = |a: i64| a + k; add(2) + add(3) }
        fn counter() -> i64 { let mut n = 0; let bump = |d: i64| d + 1; n = bump(n); n = bump(n); n }
        fn nested(k: i64) -> i64 { let outer = |a: i64| { let inner = |b: i64| a + b + k; inner(1) }; outer(10) }
        fn filtered(v: Vec<i64>, min: i64) -> Vec<i64> { v |> std::iter::filter(|x| x >= min) |> std::collect() }
    ";
    let ints = |xs: &[i64]| Value::vec(xs.iter().map(|x| Value::Int(*x)).collect());
    assert_eq!(
        run(src, "shifted", vec![ints(&[1, 2]), Value::Int(10)]).to_json(),
        json!([11, 12])
    );
    assert_eq!(int(run(src, "named", vec![Value::Int(5)])), 7 + 8);
    assert_eq!(int(run(src, "counter", vec![])), 2);
    assert_eq!(int(run(src, "nested", vec![Value::Int(100)])), 111);
    assert_eq!(
        run(src, "filtered", vec![ints(&[1, 5, 9]), Value::Int(5)]).to_json(),
        json!([5, 9])
    );
}

#[test]
fn question_mark_propagates_errors_and_none() {
    let src = "
        fn parse(s: String) -> Result<i64, String> {
            match s.to_int() { Some(n) => Ok(n), None => Err(f\"not a number: {s}\") }
        }
        fn double(s: String) -> Result<i64, String> { let n = parse(s)?; Ok(n * 2) }
        fn sum(a: String, b: String) -> Result<i64, String> { Ok(parse(a)? + parse(b)?) }
        fn first_even(v: Vec<i64>) -> Option<i64> {
            let first = v.first()?;
            if first % 2 == 0 { Some(first) } else { None }
        }
        fn describe(s: String) -> String {
            match double(s) { Ok(n) => f\"ok {n}\", Err(e) => f\"error: {e}\" }
        }
    ";
    assert_eq!(text(run(src, "describe", vec![Value::str("21")])), "ok 42");
    assert_eq!(
        text(run(src, "describe", vec![Value::str("x")])),
        "error: not a number: x"
    );
    let both = run(src, "sum", vec![Value::str("2"), Value::str("3")]);
    assert_eq!(both.to_json(), json!(5));
    let bad = run(src, "sum", vec![Value::str("2"), Value::str("z")]);
    assert!(matches!(bad, Value::Err(_)), "{bad:?}");
    let ints = |xs: &[i64]| Value::vec(xs.iter().map(|x| Value::Int(*x)).collect());
    assert_eq!(
        run(src, "first_even", vec![ints(&[4, 1])]).to_json(),
        json!(4)
    );
    assert_eq!(
        run(src, "first_even", vec![ints(&[3])]).to_json(),
        json!(null)
    );
    assert_eq!(
        run(src, "first_even", vec![ints(&[])]).to_json(),
        json!(null)
    );
}

#[test]
fn if_let_and_while_let() {
    let src = "
        fn describe(v: Vec<i64>) -> String {
            if let Some(x) = v.first() { f\"first {x}\" } else { \"empty\" }
        }
        fn chained(a: Option<i64>, b: Option<i64>) -> i64 {
            if let Some(x) = a { x } else if let Some(y) = b { y * 10 } else { 0 }
        }
        fn drain(v: Vec<i64>) -> i64 {
            let mut total = 0;
            while let Some(x) = v.pop() { total += x; }
            total
        }
        fn until_big(v: Vec<i64>) -> i64 {
            let mut total = 0;
            while let Some(x) = v.pop() {
                if x > 100 { break; }
                if x % 2 == 1 { continue; }
                total += x;
            }
            total
        }
        fn set_only(r: Result<i64, String>) -> i64 {
            let mut out = 0;
            if let Ok(n) = r { out = n; }
            out
        }
    ";
    let ints = |xs: &[i64]| Value::vec(xs.iter().map(|x| Value::Int(*x)).collect());
    assert_eq!(text(run(src, "describe", vec![ints(&[7, 8])])), "first 7");
    assert_eq!(text(run(src, "describe", vec![ints(&[])])), "empty");
    let some = |n| Value::some(Value::Int(n));
    assert_eq!(int(run(src, "chained", vec![some(4), Value::none()])), 4);
    assert_eq!(int(run(src, "chained", vec![Value::none(), some(4)])), 40);
    assert_eq!(
        int(run(src, "chained", vec![Value::none(), Value::none()])),
        0
    );
    assert_eq!(int(run(src, "drain", vec![ints(&[1, 2, 3, 4])])), 10);
    assert_eq!(
        int(run(src, "until_big", vec![ints(&[2, 500, 4, 5, 6])])),
        6 + 4
    );
    assert_eq!(
        int(run(
            src,
            "set_only",
            vec![Value::Ok(Box::new(Value::Int(9)))]
        )),
        9
    );
    assert_eq!(
        int(run(
            src,
            "set_only",
            vec![Value::Err(Box::new(Value::str("x")))]
        )),
        0
    );
}

#[test]
fn tuples_and_iterator_methods() {
    let src = "
        fn swap(p: (i64, String)) -> (String, i64) { (p.1, p.0) }
        fn destructure(a: i64) -> i64 { let (x, y) = (a, a * 2); let (_, z, w) = (1, 2, 3); x + y + z + w }
        fn labelled(v: Vec<String>) -> Vec<String> {
            v.enumerate().map(|(i, s)| f\"{i}:{s}\")
        }
        fn pairs() -> Vec<(i64, String)> { [1, 2].zip([\"a\", \"b\"]) }
        fn total_len(v: Vec<String>) -> u64 { v.map(|s| s.len()).fold(0, |acc, n| acc + n) }
        fn evens(v: Vec<i64>) -> Vec<i64> { v.filter(|x| x % 2 == 0) }
        fn stats(v: Vec<i64>) -> (i64, Option<i64>, Option<i64>) { (v.sum(), v.min(), v.max()) }
        fn has_big(v: Vec<i64>) -> bool { v.any(|x| x > 10) && !v.all(|x| x > 10) }
        fn first_big(v: Vec<i64>) -> Option<i64> { v.find(|x| x > 10) }
        fn counts(text: String) -> Vec<String> {
            let m: Map<String, i64> = std::map::new();
            for w in text.split(\" \") { m.insert(w, m.get(w).unwrap_or(0) + 1); }
            let mut out = [\"\"];
            out.pop();
            for (word, n) in m.entries() { out.push(f\"{word}={n}\"); }
            out
        }
        fn joined(v: Vec<String>) -> String { v.join(\"+\") }
        fn skipped(v: Vec<i64>) -> Vec<i64> { v.skip(2) }
        fn tuple_match(p: (bool, i64)) -> String {
            match p { (true, n) => f\"yes {n}\", (false, 0) => \"zero\", (false, n) => f\"no {n}\" }
        }
    ";
    let ints = |xs: &[i64]| Value::vec(xs.iter().map(|x| Value::Int(*x)).collect());
    let strs = |xs: &[&str]| Value::vec(xs.iter().map(|x| Value::str(*x)).collect());
    let pair = Value::Tuple(Arc::new(vec![Value::Int(7), Value::str("seven")]));
    assert_eq!(run(src, "swap", vec![pair]).to_json(), json!(["seven", 7]));
    assert_eq!(
        int(run(src, "destructure", vec![Value::Int(5)])),
        5 + 10 + 2 + 3
    );
    assert_eq!(
        run(src, "labelled", vec![strs(&["a", "b"])]).to_json(),
        json!(["0:a", "1:b"])
    );
    assert_eq!(
        run(src, "pairs", vec![]).to_json(),
        json!([[1, "a"], [2, "b"]])
    );
    assert_eq!(int(run(src, "total_len", vec![strs(&["ab", "cde"])])), 5);
    assert_eq!(
        run(src, "evens", vec![ints(&[1, 2, 3, 4])]).to_json(),
        json!([2, 4])
    );
    assert_eq!(
        run(src, "stats", vec![ints(&[3, 9, 1])]).to_json(),
        json!([13, 1, 9])
    );
    assert_eq!(
        run(src, "stats", vec![ints(&[])]).to_json(),
        json!([0, null, null])
    );
    assert!(matches!(
        run(src, "has_big", vec![ints(&[1, 20])]),
        Value::Bool(true)
    ));
    assert_eq!(
        run(src, "first_big", vec![ints(&[1, 20, 30])]).to_json(),
        json!(20)
    );
    assert_eq!(
        run(src, "counts", vec![Value::str("b a b")]).to_json(),
        json!(["a=1", "b=2"])
    );
    assert_eq!(
        text(run(src, "joined", vec![strs(&["x", "y", "z"])])),
        "x+y+z"
    );
    assert_eq!(
        run(src, "skipped", vec![ints(&[1, 2, 3, 4])]).to_json(),
        json!([3, 4])
    );
    let tup = |b, n| Value::Tuple(Arc::new(vec![Value::Bool(b), Value::Int(n)]));
    assert_eq!(text(run(src, "tuple_match", vec![tup(true, 4)])), "yes 4");
    assert_eq!(text(run(src, "tuple_match", vec![tup(false, 0)])), "zero");
    assert_eq!(text(run(src, "tuple_match", vec![tup(false, 5)])), "no 5");
    // Tuples are JSON arrays on the way in and out.
    let interp = load(src);
    assert_eq!(
        interp.call_json("swap", &[json!([3, "x"])]).unwrap(),
        json!(["x", 3])
    );
}
