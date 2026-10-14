#![cfg(feature = "llvm")]

use std::path::PathBuf;
use std::process::Command;

use telvax_codegen::{BuildError, OptLevel, build_executable, build_object, llvm_ir, optimize_ir};

fn hir(src: &str) -> telvax_hir::hir::Program {
    let (ast, perrs) = telvax_parser::parse(src);
    assert!(perrs.is_empty(), "{perrs:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "{:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    hir
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("telvax-codegen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

const PROGRAM: &str = r#"
def fib(n: int32) -> int32:
    if n < 2:
        return n

    return fib(n - 1) + fib(n - 2)

def main() -> int32:
    return fib(10)
"#;

#[test]
fn emits_llvm_ir() {
    let ir = llvm_ir(&hir(PROGRAM)).unwrap();
    assert!(ir.contains("define i32 @fib(i32"), "{ir}");
    assert!(ir.contains("define i32 @main()"), "{ir}");
}

#[test]
fn optimiser_pipeline_runs() {
    let src = r#"
def twice(x: int32) -> int32:
    return x + x

def main() -> int32:
    return twice(twice(10)) + 2
"#;
    let ir = llvm_ir(&hir(src)).unwrap();
    let o0 = optimize_ir(&ir, OptLevel(0)).unwrap();
    let o2 = optimize_ir(&ir, OptLevel(2)).unwrap();
    assert!(o0.contains("call i32 @twice"), "{o0}");
    // At -O2 the calls are inlined and folded into a constant.
    assert!(o2.contains("ret i32 42"), "{o2}");
}

#[test]
fn builds_and_runs_an_executable() {
    let exe = scratch("fib10");
    build_executable(&hir(PROGRAM), OptLevel(2), &exe).unwrap();
    let status = Command::new(&exe).status().unwrap();
    assert_eq!(status.code(), Some(55));
}

#[test]
fn exit_code_reflects_control_flow() {
    let src = r#"
def classify(x: int32) -> int32:
    if x < 0:
        return 1

    else:
        if x == 0:
            return 2

    return 3

def main() -> int32:
    return classify(-5) * 100 / 100 + classify(0) * 10 + classify(9)
"#;
    let exe = scratch("classify");
    build_executable(&hir(src), OptLevel(0), &exe).unwrap();
    assert_eq!(Command::new(&exe).status().unwrap().code(), Some(24));
}

#[test]
fn builds_an_object_file() {
    let obj = scratch("lib.o");
    build_object(&hir(PROGRAM), OptLevel(1), &obj).unwrap();
    assert!(std::fs::metadata(&obj).unwrap().len() > 100);
}

#[test]
fn entry_point_is_validated() {
    let out = scratch("none");
    for (src, needle) in [
        (
            r#"
def f() -> int32:
    return 1
"#,
            "entry point",
        ),
        (
            r#"
def main() -> bool:
    return True
"#,
            "def main() -> int32",
        ),
        (
            r#"
def main(x: int32) -> int32:
    return x
"#,
            "def main() -> int32",
        ),
    ] {
        match build_executable(&hir(src), OptLevel(0), &out) {
            Err(BuildError::Diagnostics(d)) => assert!(d[0].message.contains(needle), "{d:?}"),
            other => panic!("expected a diagnostic for {src}, got {other:?}"),
        }
    }
}

#[test]
fn unsupported_code_reports_diagnostics() {
    match llvm_ir(&hir(r#"
def main() -> int32:
    g = lambda x: x
    return 1
"#))
    {
        Err(BuildError::Diagnostics(d)) => assert!(d[0].message.contains("closures")),
        other => panic!("{other:?}"),
    }
}

fn run_native(src: &str, name: &str) -> (Option<i32>, String) {
    let exe = scratch(name);
    build_executable(&hir(src), OptLevel(2), &exe).unwrap();
    let out = Command::new(&exe).output().unwrap();
    (out.status.code(), String::from_utf8(out.stdout).unwrap())
}

/// Runs `main` in the interpreter, capturing what it prints.
fn run_interpreted(src: &str) -> (i64, String) {
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
    let interp = telvax_runtime::Interp::new(hir(src)).unwrap();
    interp.set_output(Sink(buf.clone()));
    let telvax_runtime::Value::Int(code) = interp.call("main", vec![]).unwrap() else {
        panic!("main must return an int")
    };
    (
        code,
        String::from_utf8(buf.lock().unwrap().clone()).unwrap(),
    )
}

#[test]
fn hello_world() {
    let src = r#"
def main() -> int32:
    print("hello, world")
    return 0
"#;
    assert_eq!(
        run_native(src, "hello"),
        (Some(0), "hello, world\n".to_string())
    );
}

#[test]
fn native_and_interpreter_print_the_same() {
    let src = r#"
class Item:
    name: str
    qty: int32

greeting = "total"

def show(i: Item):
    std.io.print(i.name + ":")

def count(n: int32) -> int32:
    if n == 0:
        return 0
    std.io.print(".")
    return count(n - 1) + 1

def main() -> int32:
    show(Item(name="apple", qty=3))
    show(Item(qty=1, name="pear"))
    print("")
    print(greeting + "!")
    n = count(4)
    print("")
    if "a" + "b" == "ab":
        return n + 40
    else:
        return 1
"#;
    let (code, out) = run_interpreted(src);
    assert_eq!(
        run_native(src, "differential"),
        (Some(code as i32), out.clone())
    );
    assert_eq!(out, "apple:pear:\ntotal!\n....\n");
    assert_eq!(code, 44);
}

#[test]
fn fstrings_match_the_interpreter() {
    let src = r#"
class Order:
    id: uint32
    total: int
    paid: bool

def describe(o: Order) -> str:
    return f"order #{o.id}: total={o.total} paid={o.paid}"

def main() -> int32:
    print(describe(Order(id=7, total=-1250, paid=False)))
    print(describe(Order(paid=True, total=9000000000, id=4000000000)))
    print(f"{0} {-0} {255}")
    return 0
"#;
    let (code, out) = run_interpreted(src);
    assert_eq!(
        run_native(src, "fstrings"),
        (Some(code as i32), out.clone())
    );
    assert_eq!(
        out,
        "order #7: total=-1250 paid=False\norder #4000000000: total=9000000000 paid=True\n0 0 255\n"
    );
}

#[test]
fn loops_and_vectors_match_the_interpreter() {
    let src = r#"
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    d: int = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True

def fib(n: int32) -> int:
    a: int = 0
    b: int = 1
    for i in range(n):
        t = a + b
        a = b
        b = t
    return a

def main() -> int32:
    primes = 0
    limit: int = 100
    for n in range(limit):
        if is_prime(n):
            primes += 1
    print(f"primes below 100: {primes}")
    weights = [3, 1, 4, 1, 5, 9, 2, 6]
    total = 0
    for w in weights:
        total += w
    print(f"sum of {weights.len()} weights: {total}, third: {weights[2]}")
    print(f"fib(50) = {fib(50)}")
    line = ""
    for i in range(1, 6):
        line = line + f"{i * i} "
    print(line)
    return primes + total
"#;
    let (code, out) = run_interpreted(src);
    assert_eq!(
        out,
        "primes below 100: 25\nsum of 8 weights: 31, third: 4\nfib(50) = 12586269025\n1 4 9 16 25 \n"
    );
    assert_eq!(code, 56);
    assert_eq!(run_native(src, "loops"), (Some(56), out));
}

#[test]
fn out_of_bounds_index_aborts_with_a_message() {
    let src = r#"
def at(i: int32) -> int32:
    v = [1, 2, 3]
    return v[i]

def main() -> int32:
    print("before")
    return at(7)
"#;
    let exe = scratch("oob");
    build_executable(&hir(src), OptLevel(0), &exe).unwrap();
    let out = Command::new(&exe).output().unwrap();
    assert!(!out.status.success(), "must not exit normally");
    assert_eq!(
        out.status.code(),
        None,
        "terminated by SIGABRT, not an exit code"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("index out of bounds"));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "before\n");
}

#[test]
fn sieve_example_matches_the_interpreter() {
    let src = include_str!("../../../examples/basics/sieve.tvx");
    let (code, out) = run_interpreted(src);
    assert_eq!(out, "2 3 5 7 11 13 17 19 23 29 \n");
    assert_eq!(code, 10);
    assert_eq!(run_native(src, "sieve"), (Some(10), out));
}

#[test]
fn primes_example_matches_the_interpreter() {
    let src = include_str!("../../../examples/basics/primes.tvx");
    let (code, out) = run_interpreted(src);
    assert_eq!(run_native(src, "primes"), (Some(code as i32), out));
}

#[test]
fn out_of_bounds_assignment_aborts() {
    let src = r#"
def main() -> int32:
    v = [1, 2]
    k = 5
    v[k] = 1
    return 0
"#;
    let exe = scratch("oob_store");
    build_executable(&hir(src), OptLevel(1), &exe).unwrap();
    let out = Command::new(&exe).output().unwrap();
    assert_eq!(out.status.code(), None);
    assert!(String::from_utf8_lossy(&out.stderr).contains("index out of bounds"));
}
