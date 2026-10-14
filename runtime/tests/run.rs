use serde_json::json;
use std::sync::Arc;
use telvax_runtime::{Interp, Value, embed_text};

fn load(src: &str) -> Interp {
    let (ast, perrs) = telvax_parser::parse(src);
    assert!(perrs.is_empty(), "parse: {perrs:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
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
    let src = r#"
def fib(n: int32) -> int32:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"#;
    assert_eq!(int(run(src, "fib", vec![Value::Int(15)])), 610);
    assert_eq!(
        int(run(
            r#"
def f() -> int32:
    return 2 + 3 * 4 - 10 / 5 % 3
"#,
            "f",
            vec![]
        )),
        12
    );
    assert_eq!(
        int(run(
            r#"
def f() -> int32:
    return -3
"#,
            "f",
            vec![]
        )),
        -3
    );
}

#[test]
fn floats_and_literal_adaptation() {
    let v = run(
        r#"
def f(a: float32) -> float32:
    return a * 2 + 0.5
"#,
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
    let src = r#"
def f(x: int32) -> int32:
    y = x + 1
    if y > 5:
        return 100
    w = y * 2
    return w + 1
"#;
    assert_eq!(int(run(src, "f", vec![Value::Int(1)])), 5);
    assert_eq!(int(run(src, "f", vec![Value::Int(9)])), 100);
}

#[test]
fn pipes_and_strings() {
    let src = r#"
def shout(s: str) -> str:
    return s + "!"

def f(s: str) -> str:
    return s |> std.text.clean_whitespace() |> std.text.truncate(11) |> shout() |> std.text.uppercase()
"#;
    assert_eq!(
        text(run(src, "f", vec![Value::str("  hello    big   world  ")])),
        "HELLO BIG W!"
    );
}

#[test]
fn fstrings() {
    let src = r#"
def f(n: int32, s: str) -> str:
    return f"{s}: {n + 1} \"ok\""
"#;
    assert_eq!(
        text(run(src, "f", vec![Value::Int(1), Value::str("n")])),
        "n: 2 \"ok\""
    );
}

#[test]
fn iterators_and_closures() {
    let src = r#"
def f(v: list[int32], k: int32) -> str:
    return v |> std.iter.filter(lambda x: x > 1) |> std.iter.map(lambda x: f"{x * k}") |> std.string.join("-")
"#;
    let v = Value::vec(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(text(run(src, "f", vec![v, Value::Int(10)])), "20-30");
}

#[test]
fn structs_and_fields() {
    let src = r#"
class P:
    x: int32
    name: str

def f() -> str:
    p = P(name="a", x=4)
    return f"{p.name}{p.x}"
"#;
    assert_eq!(text(run(src, "f", vec![])), "a4");
}

#[test]
fn short_circuit_avoids_errors() {
    let src = r#"
def f(d: int32) -> bool:
    return d != 0 and 10 / d > 1
"#;
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
    let div = load(
        r#"
def f(d: int32) -> int32:
    return 1 / d
"#,
    );
    assert!(
        div.call("f", vec![Value::Int(0)])
            .unwrap_err()
            .message
            .contains("division by zero")
    );
    let ovf = load(
        r#"
def f(a: int) -> int:
    return a * a
"#,
    );
    assert!(
        ovf.call("f", vec![Value::Int(i64::MAX)])
            .unwrap_err()
            .message
            .contains("overflow")
    );
    let rec = load(
        r#"
def f(n: int32) -> int32:
    return f(n + 1)
"#,
    );
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
    let src = r#"
a = 2
b = a * 21

def f() -> int32:
    return b
"#;
    assert_eq!(int(run(src, "f", vec![])), 42);
}

#[test]
fn endpoint_results_and_errors() {
    let src = r#"
import std.http.Status

class In:
    n: int32

class Out:
    double: int32

@endpoint(path="/d", method="POST")
def d(p: In) -> Result[Out, Status]:
    if p.n < 0:
        return Err(Status.BadRequest("negative"))
    return Ok(Out(double=p.n * 2))
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
class T:
    v: Tensor[float32, [2, 2]]

@endpoint(path="/t", method="POST")
def t(p: T) -> Result[T, str]:
    return Ok(p)
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
    let src = include_str!("../../examples/rag-service.tvx");
    let i = load(src);
    let db = i.db_for("postgresql://localhost:5432/vectors").unwrap();
    for (id, text) in [
        (
            "t1",
            "Tensors in Telvax carry a static shape checked at compile time",
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
        ctx[0].as_str().unwrap().starts_with("Tensors in Telvax"),
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
    let i = load(
        r#"
def main() -> int32:
    std.io.print("a")
    print("b")
    return 0
"#,
    );
    i.set_output(Sink(buf.clone()));
    i.call("main", vec![]).unwrap();
    assert_eq!(
        String::from_utf8(buf.lock().unwrap().clone()).unwrap(),
        "ab\n"
    );
}

#[test]
fn mutation_and_while() {
    let src = r#"
def sum(n: int32) -> int32:
    i = 0
    s = 0
    while i < n:
        s += i
        i += 1
    return s
"#;
    assert_eq!(int(run(src, "sum", vec![Value::Int(10)])), 45);
    assert_eq!(int(run(src, "sum", vec![Value::Int(0)])), 0);
    let early = r#"
def f() -> int32:
    i = 0
    while True:
        if i == 7:
            return i * 2
        i += 1
    return 0
"#;
    assert_eq!(int(run(early, "f", vec![])), 14);
    let strs = r#"
def f() -> str:
    s = ""
    i = 0
    while i < 3:
        s = s + f"{i},"
        i += 1
    return s
"#;
    assert_eq!(text(run(strs, "f", vec![])), "0,1,2,");
}

#[test]
fn for_loops_over_ranges_and_vectors() {
    let src = r#"
def tri(n: int32) -> int32:
    s = 0

    for i in range(n):
        s += i

    return s

def incl(n: int32) -> int32:
    s = 0
    for i in range(1, n + 1):
        s += i
    return s

def dot(a: list[int32], b: list[int32]) -> int32:
    s = 0
    for i in range(std.iter.count(a)):
        s += a[i] * b[i]
    return s

def total(v: list[float]) -> float:
    s = 0.0
    for x in v:
        s += x
    return s

def find(v: list[int32], t: int32) -> int32:
    idx = 0
    for x in v:
        if x == t:
            return idx
        idx += 1
    return 0 - 1

def nested(n: int32) -> int32:
    c = 0
    for i in range(n):
        for j in range(i):
            c += j
    return c

def empty(n: int32) -> int32:
    c = 0
    for i in range(5, n):
        c += 1
    return c
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
    let src = r#"
def f(i: int32) -> int32:
    v = [10, 20, 30]
    return v[i]

def g() -> int32:
    return [[1, 2], [3, 4]][1][0]
"#;
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
fn a_range_ending_at_the_integer_limit_does_not_overflow() {
    let src = r#"
def f() -> int:
    hi: int = 9223372036854775807
    c: int = 0
    for i in range(hi - 2, hi):
        c += 1
    return c
"#;
    assert_eq!(int(run(src, "f", vec![])), 2);
}

#[test]
fn assignment_to_vector_elements_and_struct_fields() {
    let src = r#"
class P:
    x: int32
    name: str
    tags: list[int32]

class W:
    p: P
    n: int32

def elems() -> int32:
    v = [1, 2, 3]
    v[0] = 10
    v[2] += v[0]
    v[1] *= 5
    return v[0] + v[1] + v[2]

def fields() -> str:
    p = P(x=1, name="a", tags=[7, 8])
    p.x = 5
    p.x += 1
    p.name = p.name + "!"
    p.tags[1] = p.x
    return f"{p.name}{p.x}{p.tags[0]}{p.tags[1]}"

def nested() -> int32:
    w = W(p=P(x=1, name="n", tags=[0, 0]), n=2)
    w.p.x = 40
    w.p.tags[0] = 2
    w.n = 7
    return w.p.x + w.p.tags[0] + w.n

def grid() -> int32:
    g = [[1, 2], [3, 4]]
    g[1][0] = 9
    g[0][1] += g[1][0]
    return g[1][0] * 100 + g[0][1]

def fill(v: list[int32], x: int32):
    v[0] = x
    v[1] = x * 2

def via_param() -> int32:
    v = [0, 0]
    fill(v, 4)
    return v[0] + v[1]

def value_semantics() -> int32:
    a = P(x=1, name="a", tags=[])
    b = a
    a.x = 99
    return b.x

def reference_semantics() -> int32:
    a = [1, 2]
    b = a
    a[0] = 50
    return b[0]

def loop_writes() -> int32:
    v = [0, 0, 0, 0]
    for i in range(4):
        v[i] = i * i
    s = 0
    for x in v:
        s += x
    return s

def mut_param_scalar(n: int32) -> int32:
    n += 1
    return n * 2
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
        r#"
def f(k: int32):
    v = [1, 2]
    v[k] = 5

def g():
    v = [[1], [2]]
    v[1][3] = 0
"#,
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
        r#"
def spin() -> int32:
    n = 0
    while True:
        n += 1
    return n

def quick() -> int32:
    return 1 + 1
"#,
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
    let src = r#"
def sum(n: int) -> int:
    if n == 0:
        return 0
    else:
        return n + sum(n - 1)

def forever(n: int) -> int:
    return forever(n + 1)
"#;
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
def double(x: int) -> int:
    return x * 2

def apply_all(v: list[int]) -> list[int]:
    return std.iter.map(v, double)

def total(v: list[int]) -> int:
    s = 0
    for x in apply_all(v):
        s += x
    return s
"#;
    let v = Value::vec((1..=4).map(Value::Int).collect());
    assert_eq!(int(run(src, "total", vec![v])), 20);
}

#[test]
fn dist_map_spreads_work_over_threads_and_keeps_order() {
    let src = r#"
def square(x: int) -> int:
    return x * x

def all(v: list[int]) -> list[int]:
    return std.dist.map(v, square)

def fails(x: int) -> int:
    return 10 / x

def bad(v: list[int]) -> list[int]:
    return std.dist.map(v, fails)
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
    impl telvax_runtime::Executor for Counting {
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
        r#"
def work(x: int) -> int:
    return x

def all(v: list[int]) -> list[int]:
    return std.dist.map(v, work)
"#,
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
        r#"
class P:
    x: int
    y: int

def add(p: P, k: int) -> int:
    return p.x + p.y + k

def boom() -> Result[int, str]:
    return Err("no")
"#,
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
    let src = r#"
def range_break(n: int) -> int:
    s = 0

    for i in range(n):
        if i == 5:
            break

        s += i

    return s

def range_continue(n: int) -> int:
    s = 0
    for i in range(n):
        if i % 2 == 0:
            continue
        s += i
    return s

def while_loop() -> int:
    i = 0
    s = 0
    while True:
        i += 1
        if i > 10:
            break
        if i % 3 == 0:
            continue
        s += i
    return s

def nested() -> int:
    c = 0
    for a in range(5):
        for b in range(5):
            if b > a:
                break
            c += 1
    return c

def return_inside(n: int) -> int:
    for i in range(n):
        if i == 3:
            return i * 100
    return 0 - 1
"#;
    assert_eq!(int(run(src, "range_break", vec![Value::Int(100)])), 10);
    assert_eq!(int(run(src, "range_continue", vec![Value::Int(10)])), 25);
    assert_eq!(int(run(src, "while_loop", vec![])), 37);
    assert_eq!(int(run(src, "nested", vec![])), 15);
    assert_eq!(int(run(src, "return_inside", vec![Value::Int(10)])), 300);
}

#[test]
fn enums_and_match() {
    let src = r#"
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

def total() -> float:
    return area(Shape.Circle(2.0)) + area(Shape.Rect(2.0, 3.5)) + area(Shape.Empty)

def describe(n: int) -> str:
    match n:
        case 0:
            return "zero"
        case 1:
            return "one"
        case _:
            return "many"

def safe_div(a: int, b: int) -> Result[int, str]:
    if b == 0:
        return Err("divide by zero")
    return Ok(a / b)

def show(a: int, b: int) -> str:
    match safe_div(a, b):
        case Ok(v):
            return f"ok {v}"
        case Err(e):
            return f"error: {e}"

def same(a: Shape, b: Shape) -> bool:
    return a == b

def pick(flag: bool) -> Shape:
    if flag:
        return Shape.Rect(1.0, 2.0)
    else:
        return Shape.Empty

enum Op:
    Add(int)
    Neg
    Twice(Shape)

def nested(o: Op) -> float:
    match o:
        case Op.Add(n):
            return 1.0
        case Op.Neg:
            return 2.0
        case Op.Twice(Shape.Circle(r)):
            return r
        case Op.Twice(_):
            return 9.0
"#;
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
    assert_eq!(a.to_string(), "Shape.Rect(1, 2)");
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
    let src = r#"
def has(s: str, t: str) -> bool:
    return s.contains(t)

def edges(s: str) -> bool:
    return s.starts_with("ab") and s.ends_with("yz")

def fix(s: str) -> str:
    return s.replace("cat", "dog")

def words(s: str) -> uint64:
    return s.split(" ").len()

def second(s: str) -> str:
    parts = s.split(",")
    return parts[1]

def collect(n: int) -> uint64:
    v = [0]
    for i in range(1, n):
        v.push(i * i)
    return v.len()

def shares(n: int) -> bool:
    v = [1, 2]
    v.push(n)
    return v.contains(n)
"#;
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
    let src = r#"
def parse(s: str) -> int | None:
    return s.to_int()

def describe(s: str) -> str:
    match parse(s):
        case Some(n):
            return f"number {n}"
        case None:
            return "not a number"

def pick(flag: bool) -> str | None:
    if flag:
        return Some("yes")
    else:
        return None

def first_or(v: list[int], d: int) -> int:
    return v.first().unwrap_or(d)

def last_plus(v: list[int]) -> int:
    match v.last():
        case Some(x):
            return x + 1
        case None:
            return 0

def counts(v: list[int]) -> bool:
    return v.get(1).is_some() and v.get(9).is_none()

def drain() -> int:
    v = [1, 2, 3]
    total = 0
    while v.len() > 0:
        match v.pop():
            case Some(x):
                total += x
            case None:
                break
    return total

def where_is(s: str) -> uint64 | None:
    return s.find("lo")

def sorted() -> list[int]:
    v = [3, 1, 2]
    v.sort()
    return v

def reversed() -> list[str]:
    v = "c b a".split(" ")
    v.sort()
    v.reverse()
    return v
"#;
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
    let src = r#"
def word_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = std.map.new()
    for w in text.split(" "):
        n = counts.get(w).unwrap_or(0)
        counts.insert(w, n + 1)
    return counts

def lookup(text: str, w: str) -> int:
    match word_counts(text).get(w):
        case Some(n):
            return n
        case None:
            return 0 - 1

def size(text: str) -> uint64:
    return word_counts(text).len()

def keys_of(text: str) -> list[str]:
    return word_counts(text).keys()

def ids() -> dict[int, str]:
    m: dict[int, str] = std.map.new()
    m.insert(2, "two")
    m.insert(1, "one")
    m.remove(2)
    return m

def has(k: int, m: dict[int, str]) -> bool:
    return m.contains_key(k)
"#;
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
    let src = r#"
def shifted(v: list[int], k: int) -> list[int]:
    return v |> std.iter.map(lambda x: x + k) |> std.collect()

def named(k: int) -> int:
    add = lambda a: a + k
    return add(2) + add(3)

def counter() -> int:
    n = 0
    bump = lambda d: d + 1
    n = bump(n)
    n = bump(n)
    return n

def nested(k: int) -> int:
    def outer(a: int):
        inner = lambda b: a + b + k
        return inner(1)
    return outer(10)

def filtered(v: list[int], min: int) -> list[int]:
    return v |> std.iter.filter(lambda x: x >= min) |> std.collect()
"#;
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
    let src = r#"
def parse(s: str) -> Result[int, str]:
    match s.to_int():
        case Some(n):
            return Ok(n)
        case None:
            return Err(f"not a number: {s}")

def double(s: str) -> Result[int, str]:
    n = parse(s)?
    return Ok(n * 2)

def sum(a: str, b: str) -> Result[int, str]:
    return Ok(parse(a)? + parse(b)?)

def first_even(v: list[int]) -> int | None:
    first = v.first()?
    if first % 2 == 0:
        return Some(first)
    else:
        return None

def describe(s: str) -> str:
    match double(s):
        case Ok(n):
            return f"ok {n}"
        case Err(e):
            return f"error: {e}"
"#;
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
    let src = r#"
def describe(v: list[int]) -> str:
    if v.first() is Some(x):
        return f"first {x}"
    else:
        return "empty"

def chained(a: int | None, b: int | None) -> int:
    if a is Some(x):
        return x
    else:
        if b is Some(y):
            return y * 10
        else:
            return 0

def drain(v: list[int]) -> int:
    total = 0
    while v.pop() is Some(x):
        total += x
    return total

def until_big(v: list[int]) -> int:
    total = 0
    while v.pop() is Some(x):
        if x > 100:
            break
        if x % 2 == 1:
            continue
        total += x
    return total

def set_only(r: Result[int, str]) -> int:
    out = 0
    if r is Ok(n):
        out = n
    return out
"#;
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
    let src = r#"
def swap(p: (int, str)) -> (str, int):
    return (p.1, p.0)

def destructure(a: int) -> int:
    x, y = (a, a * 2)
    _, z, w = (1, 2, 3)
    return x + y + z + w

def labelled(v: list[str]) -> list[str]:
    return v.enumerate().map(lambda (i, s): f"{i}:{s}")

def pairs() -> list[(int, str)]:
    return [1, 2].zip(["a", "b"])

def total_len(v: list[str]) -> uint64:
    return v.map(lambda s: s.len()).fold(0, lambda acc, n: acc + n)

def evens(v: list[int]) -> list[int]:
    return v.filter(lambda x: x % 2 == 0)

def stats(v: list[int]) -> (int, int | None, int | None):
    return (v.sum(), v.min(), v.max())

def has_big(v: list[int]) -> bool:
    return v.any(lambda x: x > 10) and not v.all(lambda x: x > 10)

def first_big(v: list[int]) -> int | None:
    return v.find(lambda x: x > 10)

def counts(text: str) -> list[str]:
    m: dict[str, int] = std.map.new()
    for w in text.split(" "):
        m.insert(w, m.get(w).unwrap_or(0) + 1)
    out = [""]
    out.pop()
    for word, n in m.entries():
        out.push(f"{word}={n}")
    return out

def joined(v: list[str]) -> str:
    return v.join("+")

def skipped(v: list[int]) -> list[int]:
    return v.skip(2)

def tuple_match(p: (bool, int)) -> str:
    match p:
        case (True, n):
            return f"yes {n}"
        case (False, 0):
            return "zero"
        case (False, n):
            return f"no {n}"
"#;
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

#[test]
fn impl_methods_and_associated_functions() {
    let src = r#"
class Point:
    x: float
    y: float

    def new(x: float, y: float) -> Point:
        return Point(x=x, y=y)

    def origin() -> Self:
        return Self(x=0.0, y=0.0)

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    def plus(self, other: Point) -> Point:
        return Point.new(self.x + other.x, self.y + other.y)

    def scaled(self, k: float) -> Self:
        return Self.new(self.x * k, self.y * k)

enum Shape:
    Circle(float)
    Square(float)

extend Shape:
    def area(self) -> float:
        match self:
            case Shape.Circle(r):
                return r * r * 3.0
            case Shape.Square(s):
                return s * s

    def unit_square() -> Shape:
        return Self.Square(1.0)

    def describe(self) -> str:
        return f"area {self.area()}"

def run() -> float:
    p = Point.new(3.0, 4.0)
    q = p.plus(Point.origin()).scaled(2.0)
    return q.norm2() + Shape.unit_square().area() + Shape.Circle(1.0).area()

def text() -> str:
    return Shape.Square(2.0).describe()

def chain() -> float:
    return Point.new(1.0, 1.0).plus(Point.new(2.0, 3.0)).norm2()
"#;
    assert_eq!(float(run(src, "run", vec![])), 100.0 + 1.0 + 3.0);
    assert_eq!(text(run(src, "text", vec![])), "area 4");
    assert_eq!(float(run(src, "chain", vec![])), 9.0 + 16.0);
}

#[test]
fn traits_with_dynamic_dispatch_and_dependency_injection() {
    let src = r#"
trait Shape:
    def area(self) -> float: ...

    def name(self) -> str: ...

    def describe(self) -> str:
        return f"{self.name()} with area {self.area()}"

class Circle:
    r: float

class Square:
    s: float

extend Circle(Shape):
    def area(self) -> float:
        return self.r * self.r * 3.0

    def name(self) -> str:
        return "circle"

extend Square(Shape):
    def area(self) -> float:
        return self.s * self.s

    def name(self) -> str:
        return "square"

    def describe(self) -> str:
        return "a square!"

def total(shapes: list[Shape]) -> float:
    return shapes.map(lambda s: s.area()).sum()

def labels(shapes: list[Shape]) -> list[str]:
    return shapes.map(lambda s: s.describe())

def mixed() -> list[Shape]:
    return [Circle(r=1.0), Square(s=2.0)]

def direct() -> str:
    return Circle(r=2.0).describe()

# An injected dependency: the service does not know which store it talks to.
trait Store:
    def load(self, key: str) -> str | None: ...

    def save(self, key: str, value: str): ...

class MemoryStore(Store):
    data: dict[str, str]

    def load(self, key: str) -> str | None:
        return self.data.get(key)

    def save(self, key: str, value: str):
        self.data.insert(key, value)

class Greeter:
    store: Store

    def new(store: Store) -> Greeter:
        return Greeter(store=store)

    def greet(self, name: str) -> str:
        match self.store.load(name):
            case Some(g):
                return g
            case None:
                self.store.save(name, f"hello, {name}")
                return f"first time, {name}"

def visits() -> list[str]:
    g = Greeter.new(MemoryStore(data=std.map.new()))
    return [g.greet("ana"), g.greet("ana"), g.greet("bo")]
"#;
    assert_eq!(
        float(run(src, "total", vec![run(src, "mixed", vec![])])),
        3.0 + 4.0
    );
    assert_eq!(
        run(src, "labels", vec![run(src, "mixed", vec![])]).to_json(),
        json!(["circle with area 3", "a square!"])
    );
    assert_eq!(text(run(src, "direct", vec![])), "circle with area 12");
    assert_eq!(
        run(src, "visits", vec![]).to_json(),
        json!(["first time, ana", "hello, ana", "first time, bo"])
    );
}

#[test]
fn generic_functions_structs_and_impls_are_monomorphized() {
    let src = r#"
trait Named:
    def name(self) -> str: ...

class Dog(Named):
    n: str

    def name(self) -> str:
        return self.n

def first[T](xs: list[T]) -> T:
    return xs[0]

def id[T](x: T) -> T:
    return x

def greet[T: Named](x: T) -> str:
    return f"hi {x.name()}"

class Pair[A, B]:
    a: A
    b: B

    def new(a: A, b: B) -> Pair[A, B]:
        return Pair(a=a, b=b)

    def first(self) -> A:
        return self.a

    def second(self) -> B:
        return self.b

def ints() -> int:
    xs: list[int] = [4, 5, 6]
    one: int = 1
    return first(xs) + id(one)

def texts() -> str:
    return first(["x", "y"])

def named() -> str:
    return greet(Dog(n="rex"))

def pairs() -> str:
    p = Pair.new(7, "seven")
    return f"{p.first()} {p.second()}"
"#;
    assert_eq!(run(src, "ints", vec![]).to_json(), json!(5));
    assert_eq!(text(run(src, "texts", vec![])), "x");
    assert_eq!(text(run(src, "named", vec![])), "hi rex");
    assert_eq!(text(run(src, "pairs", vec![])), "7 seven");
}

#[test]
fn generic_enums_and_methods_are_monomorphized() {
    let src = r#"
enum Tree[T]:
    Leaf
    Node(T, list[T])

extend Tree[T]:
    def size(self) -> uint64:
        match self:
            case Tree.Leaf:
                return 0
            case Tree.Node(_, rest):
                return 1 + rest.len()

class Stack[T]:
    items: list[T]

    def new() -> Stack[T]:
        return Stack(items=[])

    def count(self) -> uint64:
        return self.items.len()

def sizes() -> uint64:
    t: Tree[int] = Tree.Node(1, [2, 3])
    e: Tree[str] = Tree.Leaf
    return t.size() * 10 + e.size()

def empty() -> uint64:
    s: Stack[int] = Stack.new()
    return s.count()
"#;
    assert_eq!(run(src, "sizes", vec![]).to_json(), json!(30));
    assert_eq!(run(src, "empty", vec![]).to_json(), json!(0));
}
