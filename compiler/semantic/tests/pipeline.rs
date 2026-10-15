//! `x |> stage` is the call `stage` with `x` as its first argument, and goes through the same rules
//! as any call: names, defaults, types, and a suffix such as `?` applies to the result.

use telvax_hir::lower;

fn errors(src: &str) -> Vec<String> {
    let (program, errs) = telvax_parser::parse(src);
    assert!(errs.is_empty(), "parse: {errs:?}");
    lower(&program.unwrap())
        .1
        .into_iter()
        .map(|d| d.message)
        .collect()
}

const DEFS: &str = r#"
type R:
    n: Int

fn inc(x: Int) -> Int:
    x + 1

fn add(a: Int, b: Int) -> Int:
    a + b

fn scale(x: Int, k: Int = 2) -> Int:
    x * k

fn parse(s: Text) -> Result[Int, Text]:
    Ok(1)

fn mk(x: Int) -> R:
    R(n: x)

fn listy(x: Int) -> List[Int]:
    [x]

fn id[T](x: T) -> T:
    x
"#;

fn with(body: &str) -> Vec<String> {
    errors(&format!(
        "{DEFS}\nfn f() -> Result[Int, Text]:\n{body}\n    Ok(0)\n"
    ))
}

#[test]
fn a_stage_is_a_function_or_a_call() {
    let e = with(
        "    let a: Int = 1 |> inc\n    let b: Int = 1 |> inc()\n    let c: Int = 1 |> add(2)\n    let d: Int = 1 |> inc |> inc |> add(10)\n    let e: Int = 1 |> (x => x + 1)\n    let g: Int = 1 |> id\n    let h: Int = [1, 2] |> std.iter.sum\n    let i: Text = \"  x \" |> std.text.trim\n",
    );
    assert!(e.is_empty(), "{e:?}");
}

#[test]
fn names_and_defaults_count_the_value_the_pipe_provides() {
    let e = with(
        "    let a: Int = 1 |> scale(k: 3)\n    let b: Int = 1 |> scale\n    let c: Int = 1 |> scale()\n",
    );
    assert!(e.is_empty(), "{e:?}");
    // The argument names are checked as in a call.
    let e = with("    let a = 1 |> scale(j: 3)\n");
    assert!(e.iter().any(|m| m.contains("no parameter `j`")), "{e:?}");
    let e = with("    let a = 1 |> scale(x: 3)\n");
    assert!(e.iter().any(|m| m.contains("`x` is given twice")), "{e:?}");
}

#[test]
fn a_suffix_applies_to_the_result_of_the_stage() {
    let e = with(
        "    let a: Int = \"1\" |> parse()?\n    let b: Int = 3 |> mk().n\n    let c: Int = 4 |> listy()[0]\n",
    );
    assert!(e.is_empty(), "{e:?}");
    // The `?` is held to its usual rule.
    let e = errors(&format!("{DEFS}\nfn g() -> Int:\n    \"1\" |> parse()?\n"));
    assert!(
        e.iter().any(|m| m.contains("`?` can only be used")),
        "{e:?}"
    );
}

#[test]
fn types_are_checked_like_the_call() {
    let e = with("    let a: Text = 1 |> inc\n");
    assert!(
        e.iter().any(|m| m.contains("expected `Text`, found `Int`")),
        "{e:?}"
    );
    let e = with("    let a = \"s\" |> inc\n");
    assert!(
        e.iter().any(|m| m.contains("expected `Int`, found `Text`")),
        "{e:?}"
    );
    let e = with("    let a = 1 |> inc(1, 2)\n");
    assert!(
        e.iter()
            .any(|m| m.contains("expected 1 argument(s), found 3")),
        "{e:?}"
    );
    let e = with("    let a = 1 |> add\n");
    assert!(
        e.iter()
            .any(|m| m.contains("expected 2 argument(s), found 1")),
        "{e:?}"
    );
}

#[test]
fn a_pipe_is_looser_than_every_operator() {
    // `a + b |> f` pipes the sum.
    let e = with("    let a: Int = 1 + 2 |> inc\n");
    assert!(e.is_empty(), "{e:?}");
    // `x |> f == 1` pipes into `f == 1`, which is not a function: the message says to use parentheses.
    let (program, _) =
        telvax_parser::parse(&format!("{DEFS}\nfn g() -> Bool:\n    1 |> inc == 2\n"));
    let (_, diags) = lower(&program.unwrap());
    let d = diags
        .iter()
        .find(|d| d.message.contains("right side of `|>`"))
        .expect("the pipe is reported");
    assert!(
        d.notes.iter().any(|n| n.contains("(x |> f()) == 1")),
        "{:?}",
        d.notes
    );
    let e = with("    let a: Bool = (1 |> inc) == 2\n");
    assert!(e.is_empty(), "{e:?}");
}

#[test]
fn a_right_side_that_is_not_a_function_is_an_error() {
    for stage in ["3", "\"s\"", "[1]", "(1, 2)", "1 + 2"] {
        let e = with(&format!("    let a = 1 |> {stage}\n"));
        assert!(
            e.iter().any(|m| m.contains("right side of `|>`")),
            "{stage}: {e:?}"
        );
    }
}

#[test]
fn stages_may_go_on_several_lines() {
    let e = with("    let a: Int = 1\n        |> inc\n        |> scale(k: 4)\n        |> add(1)\n");
    assert!(e.is_empty(), "{e:?}");
}
