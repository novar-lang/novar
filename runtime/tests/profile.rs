//! The profiler: time and calls per function and per line, `std` operations and memory.

use std::sync::{Mutex, MutexGuard};

use telvax_runtime::Interp;
use telvax_runtime::alloc_count::CountingAlloc;
use telvax_runtime::profile::Profiler;

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const SRC: &str = "\
fn busy(n: Int) -> Int:
    var s = 0
    for i in range(0, n):
        s = s + i * i
    s

fn fib(n: Int) -> Int:
    if n < 2:
        return n
    fib(n - 1) + fib(n - 2)

fn main() -> Int:
    let a = busy(2000)
    let w: Tensor[f32, 64, 64] = std.tensor.zeros()
    let y = std.tensor.matmul(w, w)
    a + fib(10)
";

/// The allocator's counters are global to the process, so tests that profile cannot overlap.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn load() -> Interp {
    let (ast, errors) = telvax_parser::parse(SRC);
    assert!(errors.is_empty(), "{errors:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
    assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
    Interp::new(hir).unwrap()
}

fn line_of(offset: usize) -> usize {
    SRC[..offset].matches('\n').count() + 1
}

#[test]
fn counts_calls_and_splits_self_time_from_total_time() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.call("main", vec![]).unwrap();
    let report = profiler.report();

    let get = |name: &str| report.functions.iter().find(|f| f.name == name).unwrap();
    assert_eq!(get("main").calls, 1);
    assert_eq!(get("busy").calls, 1);
    // fib(10) makes 177 calls.
    assert_eq!(get("fib").calls, 177);
    // A function's total time covers the calls it makes.
    assert!(get("main").total_time >= get("busy").total_time + get("fib").total_time);
    assert!(get("main").self_time <= get("main").total_time);
    // Recursion is not counted twice: `fib` cannot take longer than the whole run.
    assert!(get("fib").total_time <= report.wall);
    let self_sum: std::time::Duration = report.functions.iter().map(|f| f.self_time).sum();
    assert!(self_sum <= report.wall, "{self_sum:?} > {:?}", report.wall);
}

#[test]
fn counts_how_often_each_statement_ran() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.call("main", vec![]).unwrap();
    let report = profiler.report();
    let hits = |line: usize| {
        report
            .lines
            .iter()
            .filter(|l| line_of(l.offset) == line)
            .map(|l| l.hits)
            .sum::<u64>()
    };
    assert_eq!(hits(4), 2000, "the loop body");
    assert_eq!(hits(2), 1, "`var s = 0`");
    // fib(10) makes 177 calls: all test `n < 2`, 89 return `n`, 88 add the two results.
    assert_eq!(hits(8), 177, "the `if` of fib");
    assert_eq!(hits(9), 89, "`return n`");
    assert_eq!(hits(10), 88, "the last line of fib");
}

#[test]
fn times_std_operations_with_the_size_of_their_data() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.call("main", vec![]).unwrap();
    let report = profiler.report();
    let matmul = report
        .ops
        .iter()
        .find(|o| o.name == "std.tensor.matmul")
        .expect("matmul was measured");
    assert_eq!(matmul.calls, 1);
    assert_eq!(matmul.items_in, 2 * 64 * 64);
    assert_eq!(matmul.items_out, 64 * 64);
    assert!(matmul.bytes_out >= 64 * 64 * 4);
    assert!(report.ops.iter().any(|o| o.name == "std.tensor.zeros"));
}

#[test]
fn measures_the_memory_the_program_allocates() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.call("main", vec![]).unwrap();
    let report = profiler.report();
    let memory = report.memory.expect("the counting allocator is installed");
    // Two 64x64 `f32` tensors, at least.
    assert!(memory.allocated >= 2 * 64 * 64 * 4, "{memory:?}");
    assert!(memory.peak >= 64 * 64 * 4, "{memory:?}");
    assert!(memory.count > 0);
    // Most of it is allocated under `main` and the operations it calls.
    let main = report.functions.iter().find(|f| f.name == "main").unwrap();
    assert!(main.alloc_total >= 2 * 64 * 64 * 4, "{main:?}");
}

#[test]
fn writes_stacks_for_flame_graphs() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.call("main", vec![]).unwrap();
    let report = profiler.report();
    let stacks = report.collapsed();
    assert!(
        stacks.lines().any(|l| l.starts_with("main;fib;fib ")),
        "{stacks}"
    );
    assert!(stacks.contains("main;std.tensor.matmul "), "{stacks}");
    let scope = report.speedscope("demo");
    let profile = &scope["profiles"][0];
    assert_eq!(profile["unit"], "microseconds");
    assert_eq!(
        profile["samples"].as_array().unwrap().len(),
        profile["weights"].as_array().unwrap().len()
    );
    assert!(scope["shared"]["frames"].as_array().unwrap().len() >= 3);
}

#[test]
fn the_report_is_text_and_json() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.call("main", vec![]).unwrap();
    let report = profiler.report();
    let locate = |o: usize| format!("demo.tvx:{}", line_of(o));
    let text = report.render(&locate, 3);
    for needle in [
        "wall time",
        "memory    peak",
        "functions,",
        "fib",
        "std.tensor.matmul",
        "demo.tvx:",
    ] {
        assert!(text.contains(needle), "{needle} is missing from\n{text}");
    }
    assert!(
        report.to_json(&locate)["functions"]
            .as_array()
            .unwrap()
            .len()
            >= 3
    );
}

#[test]
fn a_run_without_a_profiler_records_nothing() {
    let _serial = serial();
    let interp = load();
    let profiler = Profiler::new();
    interp.set_profiler(Some(profiler.clone()));
    interp.set_profiler(None);
    interp.call("main", vec![]).unwrap();
    assert!(profiler.report().functions.is_empty());
}
