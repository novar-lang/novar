use telvax_repl::{Outcome, Session};

/// Runs each input and returns the last outcome; every input before it must be accepted.
fn run(session: &mut Session, inputs: &[&str]) -> Outcome {
    let mut last = Outcome::default();
    for input in inputs {
        last = session.eval(input);
        assert!(
            last.error.is_none(),
            "{input}: {}",
            last.error.as_deref().unwrap_or("")
        );
    }
    last
}

fn value(session: &mut Session, input: &str) -> String {
    let out = session.eval(input);
    assert!(
        out.error.is_none(),
        "{input}: {}",
        out.error.as_deref().unwrap_or("")
    );
    out.value
        .unwrap_or_else(|| panic!("{input} has no value"))
        .text
}

#[test]
fn an_expression_shows_its_value_and_a_statement_shows_nothing() {
    let mut s = Session::new();
    assert_eq!(value(&mut s, "1 + 2"), "3");
    assert_eq!(value(&mut s, "\"abc\""), "\"abc\"");
    assert_eq!(value(&mut s, "[1, 2, 3]"), "[1, 2, 3]");
    let out = s.eval("print(\"hi\")");
    assert_eq!(out.output, "hi\n");
    assert!(out.value.is_none() && out.error.is_none());
}

#[test]
fn let_and_var_last_from_one_input_to_the_next() {
    let mut s = Session::new();
    let out = run(&mut s, &["let x = 5"]);
    assert_eq!(out.defined, ["x"]);
    assert!(out.value.is_none());
    assert_eq!(value(&mut s, "x * 2"), "10");

    run(&mut s, &["var n = 1", "n = n + 1", "n = n * 10"]);
    assert_eq!(value(&mut s, "n"), "20");
    // A `let` cannot be assigned to.
    let out = s.eval("x = 6");
    assert!(out.error.is_some(), "{out:?}");
    assert_eq!(value(&mut s, "x"), "5");
}

#[test]
fn a_name_can_be_defined_again_and_may_use_the_old_value() {
    let mut s = Session::new();
    run(&mut s, &["let x = 1", "let x = x + 41"]);
    assert_eq!(value(&mut s, "x"), "42");
    run(&mut s, &["let x = \"now text\""]);
    assert_eq!(value(&mut s, "x"), "\"now text\"");
}

#[test]
fn loops_and_values_that_change_in_place_persist() {
    let mut s = Session::new();
    run(
        &mut s,
        &[
            "var total = 0",
            "for i in range(0, 4):\n    total = total + i",
            "let xs = [1, 2]",
            "xs.push(3)",
        ],
    );
    assert_eq!(value(&mut s, "total"), "6");
    assert_eq!(value(&mut s, "xs"), "[1, 2, 3]");
}

#[test]
fn functions_are_defined_used_and_replaced() {
    let mut s = Session::new();
    let out = s.eval("fn double(n: Int) -> Int:\n    n * 2");
    assert!(out.error.is_none(), "{out:?}");
    assert_eq!(out.defined, ["double"]);
    assert_eq!(value(&mut s, "double(21)"), "42");
    run(&mut s, &["fn double(n: Int) -> Int:\n    n * 3"]);
    assert_eq!(value(&mut s, "double(21)"), "63");
    // A function can call one defined earlier or later.
    run(&mut s, &["fn quad(n: Int) -> Int:\n    double(double(n))"]);
    assert_eq!(value(&mut s, "quad(1)"), "9");
}

#[test]
fn types_enums_methods_and_matches_work_across_inputs() {
    let mut s = Session::new();
    run(
        &mut s,
        &[
            "type Point:\n    x: Int\n    y: Int",
            "extend Point:\n    fn sum(self) -> Int:\n        self.x + self.y",
            "enum Shape:\n    Circle(Float)\n    Empty",
            "let p = Point(x: 1, y: 2)",
        ],
    );
    assert_eq!(value(&mut s, "p"), "Point(x=1, y=2)");
    assert_eq!(value(&mut s, "p.x"), "1");
    assert_eq!(value(&mut s, "p.sum()"), "3");
    run(&mut s, &["let shape = Shape.Circle(2.0)"]);
    let shown = value(
        &mut s,
        "match shape:\n    case Shape.Circle(r):\n        r * 2.0\n    case Shape.Empty:\n        0.0",
    );
    assert_eq!(shown, "4.0");
}

#[test]
fn a_number_written_alone_is_an_int_so_that_functions_take_it() {
    let mut s = Session::new();
    run(
        &mut s,
        &[
            "fn double(k: Int) -> Int:\n    k * 2",
            "fn total(xs: List[Int]) -> Int:\n    xs.sum()",
            "let n = 4",
            "let xs = [1, 2, 3]",
        ],
    );
    assert_eq!(value(&mut s, "double(n)"), "8");
    assert_eq!(value(&mut s, "total(xs)"), "6");
    assert_eq!(s.variables()[0].1, "Int");
    // A type that was written stays as it was written.
    run(&mut s, &["let small: i32 = 7"]);
    assert_eq!(s.variables().last().unwrap().1, "i32");
}

#[test]
fn closures_are_kept() {
    let mut s = Session::new();
    run(&mut s, &["let inc = (n: Int) => n + 1"]);
    assert_eq!(value(&mut s, "inc(1)"), "2");
    // Without the type of `n` nothing could be written down about `twice`.
    let out = s.eval("let twice = n => n * 2");
    assert!(
        out.error
            .unwrap()
            .contains("say the type of the parameters")
    );
}

#[test]
fn errors_say_where_and_change_nothing() {
    let mut s = Session::new();
    run(&mut s, &["let x = 5"]);
    let out = s.eval("x + \"a\"");
    let message = out.error.expect("a type error");
    assert!(message.starts_with("error:"), "{message}");
    assert!(
        message.contains("1 | x + \"a\"") && message.contains('^'),
        "{message}"
    );
    // The session is as it was.
    assert_eq!(value(&mut s, "x"), "5");

    let out = s.eval("fn bad() -> Int:\n    \"text\"");
    assert!(
        out.error.unwrap().contains("error"),
        "a definition with an error is not kept"
    );
    assert!(s.definitions().is_empty());
    run(&mut s, &["fn good() -> Int:\n    1"]);
    assert_eq!(value(&mut s, "good()"), "1");

    let out = s.eval("nothing_here");
    assert!(out.error.is_some());
    let out = s.eval("let = 5");
    assert!(out.error.is_some());
    let out = s.eval("[1, 2][5]");
    assert!(out.error.unwrap().starts_with("error:"));
    assert_eq!(value(&mut s, "x"), "5");
}

#[test]
fn the_type_of_an_expression_is_found_without_running_it() {
    let mut s = Session::new();
    run(&mut s, &["let xs = [1, 2]"]);
    assert_eq!(s.type_of("xs").unwrap(), "List[Int]");
    assert_eq!(s.type_of("1.5").unwrap(), "Float");
    assert_eq!(s.type_of("print(\"never printed\")").unwrap(), "None");
    assert!(s.type_of("xs + 1").is_err());
}

#[test]
fn variables_and_reset() {
    let mut s = Session::new();
    run(
        &mut s,
        &["let a = 1", "var b = \"two\"", "fn f() -> Int:\n    1"],
    );
    let vars = s.variables();
    assert_eq!(vars[0].0, "a");
    assert_eq!(
        vars[1],
        ("b".to_string(), "Text".to_string(), "\"two\"".to_string())
    );
    assert_eq!(s.definitions(), ["fn f"]);
    s.reset();
    assert!(s.variables().is_empty() && s.definitions().is_empty());
    assert!(s.eval("a").error.is_some());
}

#[test]
fn a_whole_file_is_loaded_in_order() {
    let mut s = Session::new();
    let out = s.load(
        "type Pair:\n    a: Int\n    b: Int\n\nlet base = 10\n\nfn add(p: Pair) -> Int:\n    p.a + p.b + base\n",
    );
    // `base` is a variable of the session, so `add` cannot see it: the file is not a program.
    assert!(out.error.is_some(), "{out:?}");
    let out = s.load("type Pair:\n    a: Int\n    b: Int\n\nlet pair = Pair(a: 1, b: 2)\n");
    assert!(out.error.is_none(), "{out:?}");
    assert_eq!(value(&mut s, "pair.a + pair.b"), "3");
}

#[test]
fn a_runaway_input_stops_at_the_step_limit() {
    let mut s = Session::new();
    s.set_step_limit(Some(10_000));
    let out = s.eval("while True:\n    pass");
    assert!(out.error.is_some(), "{out:?}");
    s.set_step_limit(None);
    assert_eq!(value(&mut s, "1 + 1"), "2");
}

#[test]
fn it_knows_when_an_input_is_not_finished() {
    for unfinished in [
        "fn f() -> Int:",
        "fn f() -> Int:\n    1",
        "xs = [1,",
        "let s = \"abc",
        "type P:\n    x: Int",
        "@endpoint(GET, \"/x\")",
        "f(1,\n  2",
    ] {
        assert!(!Session::is_complete(unfinished), "{unfinished:?}");
    }
    for finished in [
        "1 + 2",
        "let x = [1, 2]",
        "fn f() -> Int:\n    1\n\n",
        "# just a comment: with a colon:",
        "print(\"a: b:\")",
        "f(1,\n  2)",
    ] {
        assert!(Session::is_complete(finished), "{finished:?}");
    }
}

#[test]
fn tab_completes_names_fields_and_methods() {
    let mut s = Session::new();
    run(
        &mut s,
        &[
            "type Point:\n    x: Int\n    y: Int",
            "let origin = Point(x: 1, y: 2)",
            "fn norm(p: Point) -> Int:\n    p.x",
        ],
    );
    let (start, found) = s.complete("ori", 3);
    assert_eq!(start, 0);
    assert!(found.iter().any(|c| c.label == "origin"), "{found:?}");
    let (_, found) = s.complete("no", 2);
    assert!(found.iter().any(|c| c.label == "norm"), "{found:?}");
    let (start, found) = s.complete("origin.", 7);
    assert_eq!(start, 7);
    let labels: Vec<&str> = found.iter().map(|c| c.label.as_str()).collect();
    assert!(labels.contains(&"x") && labels.contains(&"y"), "{labels:?}");
    let (start, found) = s.complete("print(origin.x) + ori", 21);
    assert_eq!(start, 18);
    assert!(found.iter().any(|c| c.label == "origin"));
}

#[test]
fn a_tensor_has_a_grid_for_a_notebook_and_a_plain_value_has_none() {
    let mut s = Session::new();
    run(&mut s, &["let t: Tensor[f32, 2, 3] = std.tensor.zeros()"]);
    let shown = s.eval("t").value.expect("a value");
    let html = shown.html.expect("a grid");
    assert!(
        html.contains("telvax-tensor") && html.contains("[2, 3]"),
        "{html}"
    );
    assert_eq!(html.matches("<tr>").count(), 2);
    assert!(s.eval("1 + 1").value.unwrap().html.is_none());
}

#[test]
fn hover_tells_the_type_or_the_signature_of_a_name() {
    let mut s = Session::new();
    run(
        &mut s,
        &["fn double(n: Int) -> Int:\n    n * 2", "let xs = [1, 2]"],
    );
    let on_function = s.hover("double(21)", 2).expect("a signature");
    assert!(
        on_function.contains("fn double(n: Int) -> Int"),
        "{on_function}"
    );
    let on_variable = s.hover("xs.len()", 1).expect("a type");
    assert!(on_variable.contains("List[Int]"), "{on_variable}");
    assert!(s.hover("1 +", 1).is_none());
    assert!(
        s.hover("nothing_here", 3).is_none(),
        "an unknown name has nothing to tell"
    );
}

#[test]
fn the_cursor_may_be_at_the_start_of_a_line_or_in_the_middle_of_one() {
    let mut s = Session::new();
    run(&mut s, &["let origin = 3", "let xs = [1, 2]"]);
    // Hover at the very start of a name, and on the second line of an input.
    assert!(s.hover("origin", 0).unwrap().contains("let origin: Int"));
    assert!(
        s.hover("let y = 1\norigin + y", 10)
            .unwrap()
            .contains("let origin: Int")
    );
    // Completing in the middle of a line leaves the rest of the line alone.
    let (start, found) = s.complete("print(ori + 1)", 9);
    assert_eq!(start, 6);
    assert!(found.iter().any(|c| c.label == "origin"), "{found:?}");
    // And on an empty line before the cursor.
    let (start, found) = s.complete("\nxs.", 4);
    assert_eq!(start, 4);
    assert!(found.iter().any(|c| c.label == "len"), "{found:?}");
}
