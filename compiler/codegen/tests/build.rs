#![cfg(feature = "llvm")]

use std::path::PathBuf;
use std::process::Command;

use novar_codegen::{BuildError, OptLevel, build_executable, build_object, llvm_ir, optimize_ir};

fn hir(src: &str) -> novar_hir::hir::Program {
    let (ast, perrs) = novar_parser::parse(src);
    assert!(perrs.is_empty(), "{perrs:?}");
    let (hir, diags) = novar_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "{:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    hir
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("novar-codegen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

const PROGRAM: &str = "
    fn fib(n: i32) -> i32 { if n < 2 { return n; } fib(n - 1) + fib(n - 2) }
    fn main() -> i32 { fib(10) }
";

#[test]
fn emits_llvm_ir() {
    let ir = llvm_ir(&hir(PROGRAM)).unwrap();
    assert!(ir.contains("define i32 @fib(i32"), "{ir}");
    assert!(ir.contains("define i32 @main()"), "{ir}");
}

#[test]
fn optimiser_pipeline_runs() {
    let src = "
        fn twice(x: i32) -> i32 { x + x }
        fn main() -> i32 { twice(twice(10)) + 2 }
    ";
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
    let src = "
        fn classify(x: i32) -> i32 { if x < 0 { return 1; } else { if x == 0 { return 2; } } 3 }
        fn main() -> i32 { classify(-5) * 100 / 100 + classify(0) * 10 + classify(9) }
    ";
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
        ("fn f() -> i32 { 1 }", "entry point"),
        ("fn main() -> bool { true }", "fn main() -> i32"),
        ("fn main(x: i32) -> i32 { x }", "fn main() -> i32"),
    ] {
        match build_executable(&hir(src), OptLevel(0), &out) {
            Err(BuildError::Diagnostics(d)) => assert!(d[0].message.contains(needle), "{d:?}"),
            other => panic!("expected a diagnostic for {src}, got {other:?}"),
        }
    }
}

#[test]
fn unsupported_code_reports_diagnostics() {
    match llvm_ir(&hir("fn main() -> i32 { let s = \"x\"; 1 }")) {
        Err(BuildError::Diagnostics(d)) => assert!(d[0].message.contains("String")),
        other => panic!("{other:?}"),
    }
}
