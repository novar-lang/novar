use telvax_runtime::{Interp, Value};

fn run(src: &str, f: &str) -> Value {
    let (ast, perrs) = telvax_parser::parse(src);
    assert!(perrs.is_empty(), "parse: {perrs:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "check: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    Interp::new(hir)
        .expect("globals should initialise")
        .call(f, vec![])
        .unwrap_or_else(|e| panic!("runtime error: {e}"))
}

const SRC: &str = r#"
type R:
    n: Int

fn inc(x: Int) -> Int:
    x + 1

fn add(a: Int, b: Int) -> Int:
    a + b

fn scale(x: Int, k: Int = 2) -> Int:
    x * k

fn parse(s: Text) -> Result[Int, Text]:
    match s.to_int():
        case Some(n):
            Ok(n)
        case None:
            Err("bad")

fn mk(x: Int) -> R:
    R(n: x)

fn listy(x: Int) -> List[Int]:
    [x, x + 1]

fn read(s: Text) -> Result[Int, Text]:
    let a = s |> parse()?
    Ok(a |> inc |> scale(k: 3))

fn chain() -> Int:
    1 |> inc |> add(10) |> scale

fn named() -> Int:
    2 |> scale(k: 5)

fn suffixes() -> Int:
    (3 |> mk().n) + (4 |> listy()[1])

fn sum_first() -> Int:
    1 + 2 |> inc

fn good() -> Result[Int, Text]:
    read("5")

fn bad() -> Result[Int, Text]:
    read("x")

fn text() -> Text:
    "  hi " |> std.text.trim |> std.text.uppercase
"#;

fn int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("expected int, got {other:?}"),
    }
}

#[test]
fn pipes_compute_what_the_equivalent_calls_do() {
    assert_eq!(int(run(SRC, "chain")), 24); // inc 2, add 12, scale 24
    assert_eq!(int(run(SRC, "named")), 10);
    assert_eq!(int(run(SRC, "suffixes")), 3 + 5);
    assert_eq!(int(run(SRC, "sum_first")), 4);
    assert_eq!(run(SRC, "text").to_string(), "HI");
}

#[test]
fn a_question_mark_after_a_stage_returns_the_error() {
    assert_eq!(run(SRC, "good").to_string(), "Ok(18)");
    assert_eq!(run(SRC, "bad").to_string(), "Err(bad)");
}
