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
        ("fn f() -> i32 { let g = |x| x; 1 }", "closures"),
        ("fn f(x: f64) -> String { f\"{x}\" }", "f-strings"),
        (
            "struct P { x: i32 } fn f(p: P) -> String { f\"{p}\" }",
            "f-strings",
        ),
        ("fn f(v: Vec<String>) -> i32 { 1 }", "`Vec<String>`"),
        (
            "fn f(s: String) -> String { s |> std::text::trim() }",
            "std item",
        ),
        ("fn f(s: String) -> bool { s < \"a\" }", "operator"),
        ("fn f(s: String) -> String { s.to_uppercase() }", "method"),
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
fn fstrings_format_ints_bools_and_strings() {
    let src = r#"
        fn len_of(n: i64) -> u64 { f"{n}".len() }
        fn is(n: i64, text: String) -> bool { f"{n}" == text }
        fn mixed(n: i32, ok: bool) -> bool { f"n={n}, ok={ok}!" == "n=-42, ok=true!" }
        fn mixed2(n: i32, ok: bool) -> bool { f"n={n}, ok={ok}!" == "n=-42, ok=true!" }
        fn unsigned(n: u8) -> bool { f"{n}" == "200" }
        fn only_text(a: i32) -> u64 { f"constant".len() }
        fn unsigned64(n: u64) -> u64 { f"{n}".len() }
        fn nested(a: i32) -> bool { let s = f"{a}"; f"[{s}|{s}]" == "[7|7]" }
    "#;
    assert_eq!(jit(src, "len_of", &[JitValue::Int(0)]), JitValue::Int(1));
    assert_eq!(
        jit(src, "len_of", &[JitValue::Int(-1234)]),
        JitValue::Int(5)
    );
    assert_eq!(
        jit(src, "len_of", &[JitValue::Int(i64::MAX)]),
        JitValue::Int(19)
    );
    assert_eq!(
        jit(src, "len_of", &[JitValue::Int(i64::MIN)]),
        JitValue::Int(20)
    );
    assert_eq!(
        jit(src, "mixed", &[JitValue::Int(-42), JitValue::Bool(true)]),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit(src, "mixed2", &[JitValue::Int(-42), JitValue::Bool(false)]),
        JitValue::Bool(false)
    );
    assert_eq!(
        jit(src, "unsigned", &[JitValue::Int(200)]),
        JitValue::Bool(true)
    );
    assert_eq!(jit(src, "only_text", &[JitValue::Int(0)]), JitValue::Int(8));
    assert_eq!(
        jit(src, "nested", &[JitValue::Int(7)]),
        JitValue::Bool(true)
    );
}

#[test]
fn global_constants_are_inlined() {
    let src = "let n = 3; let m = n * 4; let half = 0.5; fn f(x: i32) -> i32 { x + m } fn g(x: f64) -> f64 { x * half }";
    assert_eq!(jit(src, "f", &[JitValue::Int(1)]), JitValue::Int(13));
    assert_eq!(jit(src, "g", &[JitValue::Float(5.0)]), JitValue::Float(2.5));
}

#[test]
fn structs_are_passed_returned_and_nested() {
    let src = "
        struct P { x: i32, y: i32 }
        struct Seg { a: P, len: f64, flag: bool }
        fn mk(a: i32) -> P { P { y: a + 1, x: a } }
        fn sum(p: P) -> i32 { p.x * 10 + p.y }
        fn swap(p: P) -> P { P { x: p.y, y: p.x } }
        fn seg(a: i32) -> Seg { Seg { len: 1.5, a: mk(a), flag: a > 0 } }
        fn pick(c: bool) -> P { if c { mk(1) } else { mk(5) } }
        fn through(a: i32) -> i32 { sum(swap(mk(a))) }
        fn nested(a: i32) -> f64 { let s = seg(a); if s.flag { s.len + 3.0 } else { s.len } }
        fn inner(a: i32) -> i32 { let s = seg(a); s.a.y }
        fn joined(c: bool) -> i32 { let p = pick(c); p.x + p.y }
    ";
    assert_eq!(jit(src, "through", &[JitValue::Int(4)]), JitValue::Int(54));
    assert_eq!(jit(src, "inner", &[JitValue::Int(9)]), JitValue::Int(10));
    assert_eq!(
        jit(src, "nested", &[JitValue::Int(2)]),
        JitValue::Float(4.5)
    );
    assert_eq!(
        jit(src, "nested", &[JitValue::Int(-2)]),
        JitValue::Float(1.5)
    );
    assert_eq!(
        jit(src, "joined", &[JitValue::Bool(true)]),
        JitValue::Int(3)
    );
    assert_eq!(
        jit(src, "joined", &[JitValue::Bool(false)]),
        JitValue::Int(11)
    );
}

#[test]
fn strings_have_length_concat_and_equality() {
    let src = r#"
        fn len() -> u64 { "hello".len() }
        fn cat() -> u64 { ("ab" + "cde").len() }
        fn empty(s: i32) -> bool { ("" + "").is_empty() }
        fn eq(a: i32) -> bool { let s = "x" + "y"; s == "xy" }
        fn ne_same_len(a: i32) -> bool { ("ab" + "c") != "abd" }
        fn ne_len(a: i32) -> bool { ("ab" + "c") == "ab" }
        fn through(s: String) -> String { s + "!" }
        fn chained(a: i32) -> bool { through("hey") == "hey!" }
    "#;
    assert_eq!(jit(src, "len", &[]), JitValue::Int(5));
    assert_eq!(jit(src, "cat", &[]), JitValue::Int(5));
    assert_eq!(jit(src, "empty", &[JitValue::Int(0)]), JitValue::Bool(true));
    assert_eq!(jit(src, "eq", &[JitValue::Int(0)]), JitValue::Bool(true));
    assert_eq!(
        jit(src, "ne_same_len", &[JitValue::Int(0)]),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit(src, "ne_len", &[JitValue::Int(0)]),
        JitValue::Bool(false)
    );
    assert_eq!(
        jit(src, "chained", &[JitValue::Int(0)]),
        JitValue::Bool(true)
    );
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

// ---- mutation, loops, vectors --------------------------------------------------------------

#[test]
fn mutation_and_while_loops() {
    let src = "
        fn sum(n: i64) -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; while i < n { s += i; i += 1; } s }
        fn collatz(n: i64) -> i64 {
            let mut x = n; let mut steps: i64 = 0;
            while x != 1 { if x % 2 == 0 { x = x / 2; } else { x = 3 * x + 1; } steps += 1; }
            steps
        }
        fn early(n: i32) -> i32 { let mut i = 0; while true { if i == n { return i * 2; } i += 1; } 0 }
        fn shadow(n: i32) -> i32 { let mut a = n; let b = a; a = a + 10; a + b }
        fn text(n: i32) -> u64 { let mut s = \"\"; let mut i = 0; while i < n { s = s + f\"{i};\"; i += 1; } s.len() }
        fn strukt(n: i32) -> i32 {
            let mut p = P { x: 0, y: 1 };
            let mut i = 0;
            while i < n { p = P { x: p.y, y: p.x + p.y }; i += 1; }
            p.x
        }
        struct P { x: i32, y: i32 }
    ";
    assert_eq!(jit(src, "sum", &[JitValue::Int(100)]), JitValue::Int(4950));
    assert_eq!(jit(src, "sum", &[JitValue::Int(0)]), JitValue::Int(0));
    assert_eq!(
        jit(src, "collatz", &[JitValue::Int(27)]),
        JitValue::Int(111)
    );
    assert_eq!(jit(src, "early", &[JitValue::Int(7)]), JitValue::Int(14));
    assert_eq!(jit(src, "shadow", &[JitValue::Int(5)]), JitValue::Int(20));
    assert_eq!(jit(src, "text", &[JitValue::Int(12)]), JitValue::Int(26));
    assert_eq!(jit(src, "strukt", &[JitValue::Int(10)]), JitValue::Int(55));
}

#[test]
fn for_range_loops() {
    let src = "
        fn tri(n: i32) -> i32 { let mut s = 0; for i in 0..n { s += i; } s }
        fn incl(n: i32) -> i32 { let mut s = 0; for i in 1..=n { s += i; } s }
        fn nested(n: i32) -> i32 { let mut c = 0; for i in 0..n { for j in 0..i { c += j; } } c }
        fn empty(n: i32) -> i32 { let mut c = 0; for i in 5..n { c += 1; } c }
        fn unsigned(n: u8) -> u8 { let mut c: u8 = 0; for i in 250..=255 { c += 1; } c }
        fn at_max() -> i32 { let mut c = 0; let hi: u8 = 255; for i in hi - 2..=hi { c += 1; } c }
        fn ret_inside(n: i32) -> i32 { for i in 0..n { if i * i > 50 { return i; } } 0 - 1 }
        fn f64s(n: i32) -> f64 { let mut s = 0.0; for i in 0..n { s += 0.5; } s }
    ";
    assert_eq!(jit(src, "tri", &[JitValue::Int(10)]), JitValue::Int(45));
    assert_eq!(jit(src, "incl", &[JitValue::Int(10)]), JitValue::Int(55));
    assert_eq!(jit(src, "nested", &[JitValue::Int(5)]), JitValue::Int(10));
    assert_eq!(jit(src, "empty", &[JitValue::Int(3)]), JitValue::Int(0));
    assert_eq!(jit(src, "incl", &[JitValue::Int(0)]), JitValue::Int(0));
    assert_eq!(
        jit(src, "at_max", &[]),
        JitValue::Int(3),
        "..=MAX must terminate"
    );
    assert_eq!(
        jit(src, "ret_inside", &[JitValue::Int(100)]),
        JitValue::Int(8)
    );
    assert_eq!(
        jit(src, "ret_inside", &[JitValue::Int(3)]),
        JitValue::Int(-1)
    );
    assert_eq!(jit(src, "f64s", &[JitValue::Int(5)]), JitValue::Float(2.5));
}

#[test]
fn vectors_arrays_and_indexing() {
    let src = "
        fn at(i: i32) -> i32 { let v = [10, 20, 30]; v[i] }
        fn total() -> i32 { let v = [1, 2, 3, 4]; let mut s = 0; for x in v { s += x; } s }
        fn dot() -> f64 {
            let a = [1.5, 2.0, 3.0]; let b = [2.0, 4.0, 0.5];
            let mut s = 0.0;
            for i in 0..a.len() { s += a[i] * b[i]; }
            s
        }
        fn count() -> u64 { std::iter::count([true, false, true]) }
        fn flags() -> i32 { let f = [true, false, true]; let mut c = 0; for b in f { if b { c += 1; } } c }
        fn nonempty() -> bool { !([1, 2].is_empty()) }
        fn small(i: u8) -> i32 { let v = [1, 2, 3]; v[i] }
        fn sum_to(n: i32) -> i32 {
            let mut v = [0, 0, 0];
            let mut t = 0;
            for i in 0..n { t += [i, i, i][2]; }
            t
        }
    ";
    assert_eq!(jit(src, "at", &[JitValue::Int(1)]), JitValue::Int(20));
    assert_eq!(jit(src, "total", &[]), JitValue::Int(10));
    assert_eq!(jit(src, "dot", &[]), JitValue::Float(1.5 * 2.0 + 8.0 + 1.5));
    assert_eq!(jit(src, "count", &[]), JitValue::Int(3));
    assert_eq!(jit(src, "flags", &[]), JitValue::Int(2));
    assert_eq!(jit(src, "nonempty", &[]), JitValue::Bool(true));
    assert_eq!(jit(src, "small", &[JitValue::Int(2)]), JitValue::Int(3));
    assert_eq!(jit(src, "sum_to", &[JitValue::Int(5)]), JitValue::Int(10));
}

#[test]
fn unsupported_vector_shapes_are_diagnosed() {
    for (src, needle) in [
        (
            "fn f() -> i32 { let v = [\"a\", \"b\"]; 1 }",
            "`Vec<String>`",
        ),
        (
            "struct P { x: i32 } fn f() -> i32 { let v = [P { x: 1 }]; 1 }",
            "`Vec<P>`",
        ),
        ("fn f(v: Vec<Vec<i32>>) -> i32 { 1 }", "`Vec<Vec<i32>>`"),
    ] {
        let h = hir(src);
        let ctx = novar_mlir::context();
        let errs = lower(&ctx, &h)
            .err()
            .unwrap_or_else(|| panic!("expected an error for {src}"));
        assert!(
            errs.iter().any(|d| d.message.contains(needle)),
            "{src}: {errs:?}"
        );
    }
}

// ---- assignment to elements and fields -----------------------------------------------------

const PLACES: &str = r#"
    struct P { x: i32, name: String, tags: Vec<i32> }
    struct W { p: P, n: i32 }
    fn elems() -> i32 { let mut v = [1, 2, 3]; v[0] = 10; v[2] += v[0]; v[1] *= 5; v[0] + v[1] + v[2] }
    fn fields_ok() -> bool {
        let mut p = P { x: 1, name: "a", tags: [7, 8] };
        p.x = 5; p.x += 1; p.name = p.name + "!"; p.tags[1] = p.x;
        f"{p.name}{p.x}{p.tags[0]}{p.tags[1]}" == "a!676"
    }
    fn nested() -> i32 {
        let mut w = W { p: P { x: 1, name: "n", tags: [0, 0] }, n: 2 };
        w.p.x = 40; w.p.tags[0] = 2; w.n = 7;
        w.p.x + w.p.tags[0] + w.n
    }
    fn fill(mut v: Vec<i32>, x: i32) { v[0] = x; v[1] = x * 2; }
    fn via_param() -> i32 { let v = [0, 0]; fill(v, 4); v[0] + v[1] }
    fn value_semantics() -> i32 { let mut a = P { x: 1, name: "a", tags: [] }; let b = a; a.x = 99; b.x }
    fn reference_semantics() -> i32 { let mut a = [1, 2]; let b = a; a[0] = 50; b[0] }
    fn loop_writes() -> i32 {
        let mut v = [0, 0, 0, 0];
        for i in 0..4 { v[i] = i * i; }
        let mut s = 0;
        for x in v { s += x; }
        s
    }
    fn mut_param(mut n: i32) -> i32 { n += 1; n * 2 }
    fn bubble() -> i32 {
        let mut v = [5, 2, 9, 1, 7];
        let n = v.len();
        for i in 0..n {
            for j in 0..n - 1 - i {
                if v[j] > v[j + 1] { let t = v[j]; v[j] = v[j + 1]; v[j + 1] = t; }
            }
        }
        v[0] * 10000 + v[1] * 1000 + v[2] * 100 + v[3] * 10 + v[4]
    }
    fn oob(k: i32) -> i32 { let mut v = [1, 2]; v[k] = 5; v[0] }
"#;

#[test]
fn places_match_the_interpreter() {
    let h = hir(PLACES);
    let interp = novar_runtime::Interp::new(h.clone()).unwrap();
    for f in [
        "elems",
        "nested",
        "via_param",
        "value_semantics",
        "reference_semantics",
        "loop_writes",
        "bubble",
    ] {
        let novar_runtime::Value::Int(want) = interp.call(f, vec![]).unwrap() else {
            panic!("{f}")
        };
        assert_eq!(jit_call(&h, f, &[]).unwrap(), JitValue::Int(want), "{f}");
    }
    let novar_runtime::Value::Bool(want) = interp.call("fields_ok", vec![]).unwrap() else {
        panic!()
    };
    assert!(want);
    assert_eq!(
        jit_call(&h, "fields_ok", &[]).unwrap(),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit_call(&h, "mut_param", &[JitValue::Int(3)]).unwrap(),
        JitValue::Int(8)
    );
    assert_eq!(
        jit_call(&h, "oob", &[JitValue::Int(1)]).unwrap(),
        JitValue::Int(1)
    );
}

#[test]
fn bubble_sort_sorts() {
    assert_eq!(jit(PLACES, "bubble", &[]), JitValue::Int(12579));
    assert_eq!(jit(PLACES, "reference_semantics", &[]), JitValue::Int(50));
    assert_eq!(jit(PLACES, "value_semantics", &[]), JitValue::Int(1));
}
