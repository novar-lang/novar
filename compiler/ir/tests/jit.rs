#![cfg(feature = "llvm")]

use novar_mlir::{JitValue, emit_mlir, jit_call, lower};

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

fn jit(src: &str, f: &str, args: &[JitValue]) -> JitValue {
    jit_call(&hir(src), f, args).unwrap_or_else(|e| panic!("jit failed: {e}"))
}

#[test]
fn fibonacci() {
    let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } fib(n - 1) + fib(n - 2) }";
    assert_eq!(jit(src, "fib", &[JitValue::Int(20)]), JitValue::Int(6765));
    assert_eq!(jit(src, "fib", &[JitValue::Int(0)]), JitValue::Int(0));
}

#[test]
fn integer_arithmetic_and_widths() {
    assert_eq!(
        jit("fn f() -> i32 { 2 + 3 * 4 - 10 / 5 % 3 }", "f", &[]),
        JitValue::Int(12)
    );
    assert_eq!(
        jit(
            "fn f(a: i64, b: i64) -> i64 { a * b - 1 }",
            "f",
            &[JitValue::Int(1 << 40), JitValue::Int(3)]
        ),
        JitValue::Int((3i64 << 40) - 1)
    );
    assert_eq!(
        jit("fn f(a: i8) -> i8 { a + 1 }", "f", &[JitValue::Int(127)]),
        JitValue::Int(-128),
        "wraps on overflow"
    );
    assert_eq!(
        jit("fn f(a: i32) -> i32 { -a }", "f", &[JitValue::Int(5)]),
        JitValue::Int(-5)
    );
    assert_eq!(
        jit("fn f(a: i32) -> i32 { -7 / a }", "f", &[JitValue::Int(2)]),
        JitValue::Int(-3)
    );
}

#[test]
fn unsigned_semantics() {
    let src = "fn f(a: u32, b: u32) -> bool { a > b }";
    // 0xFFFF_FFFF is huge as u32 but -1 as i32.
    assert_eq!(
        jit(src, "f", &[JitValue::Int(0xFFFF_FFFF), JitValue::Int(1)]),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit("fn f(a: u8) -> u8 { a / 2 }", "f", &[JitValue::Int(200)]),
        JitValue::Int(100)
    );
    assert_eq!(
        jit("fn f(a: u8) -> u8 { a }", "f", &[JitValue::Int(200)]),
        JitValue::Int(200)
    );
}

#[test]
fn floats() {
    let v = jit(
        "fn f(a: f64, b: f64) -> f64 { a * b + 0.5 }",
        "f",
        &[JitValue::Float(1.5), JitValue::Float(4.0)],
    );
    assert_eq!(v, JitValue::Float(6.5));
    let v = jit(
        "fn f(a: f32) -> f32 { -a / 2 }",
        "f",
        &[JitValue::Float(3.0)],
    );
    assert_eq!(v, JitValue::Float(-1.5));
    assert_eq!(
        jit(
            "fn f(a: f64) -> bool { a < 1.0 }",
            "f",
            &[JitValue::Float(0.5)]
        ),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit(
            "fn f(a: f64) -> bool { a != a }",
            "f",
            &[JitValue::Float(f64::NAN)]
        ),
        JitValue::Bool(true)
    );
}

#[test]
fn booleans_and_short_circuit() {
    let src = "fn f(d: i32) -> bool { d != 0 && 10 / d > 1 }";
    // `10 / 0` would trap if the right-hand side were evaluated.
    assert_eq!(jit(src, "f", &[JitValue::Int(0)]), JitValue::Bool(false));
    assert_eq!(jit(src, "f", &[JitValue::Int(2)]), JitValue::Bool(true));
    let or = "fn f(d: i32) -> bool { d == 0 || 10 / d > 100 }";
    assert_eq!(jit(or, "f", &[JitValue::Int(0)]), JitValue::Bool(true));
    assert_eq!(jit(or, "f", &[JitValue::Int(5)]), JitValue::Bool(false));
    assert_eq!(
        jit("fn f(a: bool) -> bool { !a }", "f", &[JitValue::Bool(true)]),
        JitValue::Bool(false)
    );
    assert_eq!(
        jit(
            "fn f(a: bool, b: bool) -> bool { a == b }",
            "f",
            &[JitValue::Bool(true), JitValue::Bool(true)]
        ),
        JitValue::Bool(true)
    );
}

#[test]
fn control_flow_and_calls() {
    let src = "
        fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }
        fn clamp(x: i32, lo: i32, hi: i32) -> i32 {
            if x < lo { return lo; }
            if x > hi { return hi; }
            x
        }
        fn both(x: i32) -> i32 { let a = abs(x); let b = clamp(a, 0, 10); a + b }
        fn early(x: i32) -> i32 { if x > 0 { return 1; } else { return 2; } }
        fn unit_then_value(x: i32) -> i32 { if x > 0 { let y = x; } x * 2 }
    ";
    assert_eq!(jit(src, "abs", &[JitValue::Int(-4)]), JitValue::Int(4));
    assert_eq!(
        jit(
            src,
            "clamp",
            &[JitValue::Int(50), JitValue::Int(0), JitValue::Int(10)]
        ),
        JitValue::Int(10)
    );
    assert_eq!(
        jit(
            src,
            "clamp",
            &[JitValue::Int(-5), JitValue::Int(0), JitValue::Int(10)]
        ),
        JitValue::Int(0)
    );
    assert_eq!(jit(src, "both", &[JitValue::Int(-30)]), JitValue::Int(40));
    assert_eq!(jit(src, "early", &[JitValue::Int(1)]), JitValue::Int(1));
    assert_eq!(jit(src, "early", &[JitValue::Int(-1)]), JitValue::Int(2));
    assert_eq!(
        jit(src, "unit_then_value", &[JitValue::Int(3)]),
        JitValue::Int(6)
    );
}

#[test]
fn unit_functions() {
    let src = "fn noop(x: i32) { let y = x + 1; } fn f() -> i32 { noop(1); 7 }";
    assert_eq!(jit(src, "noop", &[JitValue::Int(1)]), JitValue::Unit);
    assert_eq!(jit(src, "f", &[]), JitValue::Int(7));
}

#[test]
fn unsupported_features_are_diagnosed_not_miscompiled() {
    let cases = [
        ("fn f(s: String) -> i32 { 1 }", "`String`"),
        (
            "struct P { x: i32 } fn f() -> i32 { let p = P { x: 1 }; 1 }",
            "structs",
        ),
        ("fn f() -> i32 { let g = |x| x; 1 }", "closures"),
        ("let g = 3; fn f() -> i32 { g }", "global variables"),
    ];
    for (src, needle) in cases {
        let h = hir(src);
        let ctx = novar_mlir::context();
        let errs = lower(&ctx, &h)
            .err()
            .unwrap_or_else(|| panic!("expected an error for {src}"));
        assert!(
            errs.iter().any(|d| d.message.contains(needle)),
            "{src}: {errs:?}"
        );
        assert!(
            errs.iter()
                .all(|d| d.notes.iter().any(|n| n.contains("in function")))
        );
    }
}

#[test]
fn mlir_text_has_the_expected_shape() {
    let h = hir("fn fib(n: i32) -> i32 { if n < 2 { return n; } fib(n - 1) + fib(n - 2) }");
    let high = emit_mlir(&h, false).unwrap();
    assert!(high.contains("func.func @fib"), "{high}");
    assert!(
        high.contains("cf.cond_br") && high.contains("arith.cmpi slt"),
        "{high}"
    );
    let low = emit_mlir(&h, true).unwrap();
    assert!(
        low.contains("llvm.func @fib") && !low.contains("arith."),
        "{low}"
    );
}

/// The interpreter and the native backend must agree on the shared language subset.
#[test]
fn differential_against_the_interpreter() {
    let src = "
        fn gcd(a: i64, b: i64) -> i64 { if b == 0 { return a; } gcd(b, a % b) }
        fn collatz(n: i64, steps: i64) -> i64 {
            if n == 1 { return steps; }
            if n % 2 == 0 { collatz(n / 2, steps + 1) } else { collatz(3 * n + 1, steps + 1) }
        }
        fn mix(x: f64, n: i32) -> f64 { if n <= 0 { x } else { mix(x * 1.5 - 0.25, n - 1) } }
    ";
    let h = hir(src);
    let interp = novar_runtime::Interp::new(h.clone()).unwrap();
    for (a, b) in [(12, 18), (17, 5), (1_000_000, 750), (7, 0)] {
        let want = interp
            .call(
                "gcd",
                vec![novar_runtime::Value::Int(a), novar_runtime::Value::Int(b)],
            )
            .unwrap();
        let novar_runtime::Value::Int(want) = want else {
            panic!()
        };
        assert_eq!(
            jit_call(&h, "gcd", &[JitValue::Int(a), JitValue::Int(b)]).unwrap(),
            JitValue::Int(want)
        );
    }
    for n in [1, 6, 27, 97] {
        let want = interp
            .call(
                "collatz",
                vec![novar_runtime::Value::Int(n), novar_runtime::Value::Int(0)],
            )
            .unwrap();
        let novar_runtime::Value::Int(want) = want else {
            panic!()
        };
        assert_eq!(
            jit_call(&h, "collatz", &[JitValue::Int(n), JitValue::Int(0)]).unwrap(),
            JitValue::Int(want)
        );
    }
    let want = interp
        .call(
            "mix",
            vec![
                novar_runtime::Value::Float(2.0),
                novar_runtime::Value::Int(10),
            ],
        )
        .unwrap();
    let novar_runtime::Value::Float(want) = want else {
        panic!()
    };
    assert_eq!(
        jit_call(&h, "mix", &[JitValue::Float(2.0), JitValue::Int(10)]).unwrap(),
        JitValue::Float(want)
    );
}
