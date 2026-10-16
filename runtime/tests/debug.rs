//! The hook a debugger uses: where each statement is, the calls in progress and their variables.

use std::sync::{Arc, Mutex};

use telvax_runtime::debug::{Debugger, Frame, children};
use telvax_runtime::{Interp, Value};

const SRC: &str = "\
type Point:
    x: Int
    y: Int

fn add(a: Int, b: Int) -> Int:
    let total = a + b
    total

fn main() -> Int:
    let p = Point(x: 1, y: 2)
    let xs = [10, 20]
    var n = add(p.x, p.y)
    for i in range(0, 2):
        n = n + i
    match xs.len():
        case 2:
            n = add(n, 1)
        case _:
            n = 0
    n
";

/// One stop: the line the innermost call is at, the calls, and the variables of the innermost.
#[derive(Debug, Clone)]
struct Stop {
    line: usize,
    calls: Vec<String>,
    locals: Vec<(String, String)>,
}

struct Recorder {
    stops: Mutex<Vec<Stop>>,
    source: &'static str,
}

impl Debugger for Recorder {
    fn at_statement(&self, interp: &Interp, stack: &[Frame]) {
        let top = stack.last().unwrap();
        let line = self.source[..top.offset].matches('\n').count() + 1;
        let locals = interp
            .locals(top)
            .into_iter()
            .map(|v| {
                let shown = interp.render(&v.value, Some(&v.ty));
                (v.name, shown)
            })
            .collect();
        self.stops.lock().unwrap().push(Stop {
            line,
            calls: stack.iter().map(|f| f.function.clone()).collect(),
            locals,
        });
    }
}

fn run() -> (Interp, Vec<Stop>) {
    let (ast, errors) = telvax_parser::parse(SRC);
    assert!(errors.is_empty(), "{errors:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
    assert!(diags.is_empty(), "{diags:?}");
    let interp = Interp::new(hir).unwrap();
    let rec = Arc::new(Recorder {
        stops: Mutex::new(Vec::new()),
        source: SRC,
    });
    interp.set_debugger(Some(rec.clone()));
    let v = interp.call("main", vec![]).unwrap();
    assert!(matches!(v, Value::Int(5)), "{v:?}");
    let stops = rec.stops.lock().unwrap().clone();
    (interp, stops)
}

#[test]
fn every_statement_reports_its_line_and_the_calls_in_progress() {
    let (_, stops) = run();
    let lines: Vec<usize> = stops.iter().map(|s| s.line).collect();
    // main: p, xs, n (calls add: its two statements), the loop and its two turns, the match
    // and its arm (calls add again), then the last expression.
    assert_eq!(&lines[..5], [10, 11, 12, 6, 7]);
    assert_eq!(stops[3].calls, ["main", "add"]);
    assert_eq!(stops[2].calls, ["main"]);
    assert!(
        lines.contains(&14)
            && lines.contains(&17)
            && !lines.contains(&19)
            && lines.last() == Some(&20)
    );
}

#[test]
fn variables_exist_from_their_declaration_with_their_types_and_values() {
    let (_, stops) = run();
    // Inside `add`, at its last line: the parameters and `total`.
    let inside = stops.iter().find(|s| s.line == 7).unwrap();
    assert_eq!(
        inside.locals,
        [
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
            ("total".to_string(), "3".to_string()),
        ]
    );
    // In `main` at `var n = ...` only `p` and `xs` exist yet.
    let before_n = stops.iter().find(|s| s.line == 12).unwrap();
    let names: Vec<&str> = before_n.locals.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["p", "xs"]);
    assert_eq!(before_n.locals[0].1, "Point(x=1, y=2)");
    assert_eq!(before_n.locals[1].1, "[10, 20]");
    // The loop variable and the changed `n` show up later.
    let last = stops.last().unwrap();
    let n = last.locals.iter().find(|(name, _)| name == "n").unwrap();
    assert_eq!(n.1, "5");
}

#[test]
fn values_can_be_looked_into() {
    let point = Value::Struct(Arc::new(telvax_runtime::StructVal {
        name: "Point".into(),
        fields: vec![("x".into(), Value::Int(1)), ("y".into(), Value::Int(2))],
    }));
    let inside: Vec<(String, String)> = children(&point)
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
    assert_eq!(inside, [("x".into(), "1".into()), ("y".into(), "2".into())]);
    let list = Value::vec(vec![Value::Int(7), Value::Int(8)]);
    assert_eq!(children(&list).len(), 2);
    assert!(children(&Value::Int(3)).is_empty());
}

#[test]
fn without_a_debugger_nothing_is_recorded_and_it_can_be_removed() {
    let (ast, _) = telvax_parser::parse(SRC);
    let (hir, _) = telvax_hir::lower(&ast.unwrap());
    let interp = Interp::new(hir).unwrap();
    let rec = Arc::new(Recorder {
        stops: Mutex::new(Vec::new()),
        source: SRC,
    });
    interp.set_debugger(Some(rec.clone()));
    interp.set_debugger(None);
    interp.call("main", vec![]).unwrap();
    assert!(rec.stops.lock().unwrap().is_empty());
}
