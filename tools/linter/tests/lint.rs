//! What the linter reports: unread variables and private functions nothing calls.

use telvax_hir::lower;
use telvax_lint::lint;

fn warnings(src: &str) -> Vec<String> {
    let (ast, errs) = telvax_parser::parse(src);
    assert!(errs.is_empty(), "parse: {errs:?}");
    let (hir, diags) = lower(&ast.unwrap());
    assert!(
        diags
            .iter()
            .all(|d| d.severity != telvax_diagnostics::Severity::Error),
        "check: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    lint(&hir).into_iter().map(|d| d.message).collect()
}

fn unused(src: &str) -> Vec<String> {
    warnings(src)
        .into_iter()
        .filter_map(|m| {
            m.strip_prefix("unused variable `")
                .map(|r| r.trim_end_matches('`').to_string())
        })
        .collect()
}

#[test]
fn a_variable_that_is_never_read_is_reported() {
    assert_eq!(unused("fn main():\n    let dead = 5\n"), ["dead"]);
}

#[test]
fn a_variable_that_is_read_is_not() {
    assert!(unused("fn main():\n    let x = 1\n    print(f\"{x}\")\n").is_empty());
    assert!(unused("fn f(a: Int) -> Int:\n    a + 1\n").is_empty());
}

#[test]
fn a_name_that_starts_with_an_underscore_is_exempt() {
    assert!(unused("fn f(_a: Int):\n    let _quiet = 6\n").is_empty());
}

#[test]
fn only_assigning_is_not_reading() {
    assert_eq!(
        unused("fn main():\n    var counter = 0\n    counter = 3\n"),
        ["counter"]
    );
}

#[test]
fn parameters_loop_variables_and_pattern_names_are_checked() {
    assert_eq!(unused("fn f(a: Int, b: Int) -> Int:\n    a\n"), ["b"]);
    assert_eq!(
        unused("fn main():\n    for i in range(3):\n        pass\n"),
        ["i"]
    );
    assert_eq!(
        unused(
            "fn main():\n    match Some(1):\n        case Some(v):\n            pass\n        case None:\n            pass\n"
        ),
        ["v"]
    );
    assert!(unused("fn main():\n    for j in range(3):\n        print(f\"{j}\")\n").is_empty());
}

#[test]
fn a_variable_used_inside_a_closure_counts() {
    assert!(unused("fn f(a: Int) -> Int:\n    let g = x => x + a\n    g(1)\n").is_empty());
}

#[test]
fn a_private_function_nothing_calls_is_reported() {
    let w = warnings("fn _helper() -> Int:\n    1\n\nfn main():\n    pass\n");
    assert_eq!(w, ["function `_helper` is never used"]);
}

#[test]
fn a_private_function_that_is_called_is_not() {
    let w = warnings("fn _helper() -> Int:\n    1\n\nfn main():\n    print(f\"{_helper()}\")\n");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn public_functions_and_main_are_exempt() {
    let w = warnings("fn shown() -> Int:\n    1\n\nfn main():\n    pass\n");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn an_endpoint_is_exempt_even_when_private() {
    let src = "import std.http.Status\n\ntype In:\n    x: Int\n\ntype Out:\n    y: Int\n\n@endpoint(POST, \"/x\")\nfn _handle(i: In) -> Result[Out, Status]:\n    Ok(Out(y: i.x))\n";
    let w = warnings(src);
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn warnings_come_in_source_order() {
    let src = "fn f():\n    let a = 1\n    let b = 2\n    let c = 3\n";
    assert_eq!(unused(src), ["a", "b", "c"]);
}

#[test]
fn the_examples_have_nothing_to_report() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut stack = vec![dir];
    let mut seen = 0;
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "tvx") {
                let src = std::fs::read_to_string(&path).unwrap();
                let w = warnings(&src);
                assert!(w.is_empty(), "{}: {w:?}", path.display());
                seen += 1;
            }
        }
    }
    assert!(seen > 0);
}

fn with_level(src: &str, rule: &str, level: &str) -> Vec<telvax_diagnostics::Diagnostic> {
    let (ast, _) = telvax_parser::parse(src);
    let (hir, _) = lower(&ast.unwrap());
    let mut config = telvax_lint::Config::default();
    config.set(
        telvax_lint::Rule::parse(rule).unwrap(),
        telvax_lint::Level::parse(level).unwrap(),
    );
    telvax_lint::lint_with(&hir, &config)
}

#[test]
fn an_allowed_rule_reports_nothing() {
    let src = "fn main():\n    let dead = 5\n";
    assert!(with_level(src, "unused-variable", "allow").is_empty());
}

#[test]
fn a_denied_rule_reports_errors() {
    let src = "fn main():\n    let dead = 5\n";
    let found = with_level(src, "unused-variable", "deny");
    assert_eq!(found.len(), 1);
    assert!(found[0].is_error());
}

#[test]
fn a_level_only_changes_its_own_rule() {
    let src = "fn _helper() -> Int:\n    1\n\nfn main():\n    let dead = 5\n";
    let found = with_level(src, "unused-variable", "allow");
    assert_eq!(found.len(), 1);
    assert!(found[0].message.contains("_helper"));
    assert!(!found[0].is_error());
}

#[test]
fn unknown_rules_and_levels_say_what_to_write() {
    let e = telvax_lint::Rule::parse("unused").unwrap_err();
    assert!(
        e.contains("unused-variable") && e.contains("unreachable-case"),
        "{e}"
    );
    let e = telvax_lint::Level::parse("error").unwrap_err();
    assert!(e.contains("allow, warn or deny"), "{e}");
}
