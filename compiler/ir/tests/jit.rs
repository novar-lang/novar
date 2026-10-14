#![cfg(feature = "llvm")]

use telvax_mlir::{JitValue, emit_mlir, jit_call, jit_call_with, lower};

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

fn jit(src: &str, f: &str, args: &[JitValue]) -> JitValue {
    jit_call(&hir(src), f, args).unwrap_or_else(|e| panic!("jit failed: {e}"))
}

#[test]
fn fibonacci() {
    let src = r#"
def fib(n: int32) -> int32:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"#;
    assert_eq!(jit(src, "fib", &[JitValue::Int(20)]), JitValue::Int(6765));
    assert_eq!(jit(src, "fib", &[JitValue::Int(0)]), JitValue::Int(0));
}

#[test]
fn integer_arithmetic_and_widths() {
    assert_eq!(
        jit(
            r#"
def f() -> int32:
    return 2 + 3 * 4 - 10 / 5 % 3
"#,
            "f",
            &[]
        ),
        JitValue::Int(12)
    );
    assert_eq!(
        jit(
            r#"
def f(a: int, b: int) -> int:
    return a * b - 1
"#,
            "f",
            &[JitValue::Int(1 << 40), JitValue::Int(3)]
        ),
        JitValue::Int((3i64 << 40) - 1)
    );
    assert_eq!(
        jit(
            r#"
def f(a: int8) -> int8:
    return a + 1
"#,
            "f",
            &[JitValue::Int(127)]
        ),
        JitValue::Int(-128),
        "wraps on overflow"
    );
    assert_eq!(
        jit(
            r#"
def f(a: int32) -> int32:
    return -a
"#,
            "f",
            &[JitValue::Int(5)]
        ),
        JitValue::Int(-5)
    );
    assert_eq!(
        jit(
            r#"
def f(a: int32) -> int32:
    return -7 / a
"#,
            "f",
            &[JitValue::Int(2)]
        ),
        JitValue::Int(-3)
    );
}

#[test]
fn unsigned_semantics() {
    let src = r#"
def f(a: uint32, b: uint32) -> bool:
    return a > b
"#;
    // 0xFFFF_FFFF is huge as u32 but -1 as i32.
    assert_eq!(
        jit(src, "f", &[JitValue::Int(0xFFFF_FFFF), JitValue::Int(1)]),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit(
            r#"
def f(a: uint8) -> uint8:
    return a / 2
"#,
            "f",
            &[JitValue::Int(200)]
        ),
        JitValue::Int(100)
    );
    assert_eq!(
        jit(
            r#"
def f(a: uint8) -> uint8:
    return a
"#,
            "f",
            &[JitValue::Int(200)]
        ),
        JitValue::Int(200)
    );
}

#[test]
fn floats() {
    let v = jit(
        r#"
def f(a: float, b: float) -> float:
    return a * b + 0.5
"#,
        "f",
        &[JitValue::Float(1.5), JitValue::Float(4.0)],
    );
    assert_eq!(v, JitValue::Float(6.5));
    let v = jit(
        r#"
def f(a: float32) -> float32:
    return -a / 2
"#,
        "f",
        &[JitValue::Float(3.0)],
    );
    assert_eq!(v, JitValue::Float(-1.5));
    assert_eq!(
        jit(
            r#"
def f(a: float) -> bool:
    return a < 1.0
"#,
            "f",
            &[JitValue::Float(0.5)]
        ),
        JitValue::Bool(true)
    );
    assert_eq!(
        jit(
            r#"
def f(a: float) -> bool:
    return a != a
"#,
            "f",
            &[JitValue::Float(f64::NAN)]
        ),
        JitValue::Bool(true)
    );
}

#[test]
fn booleans_and_short_circuit() {
    let src = r#"
def f(d: int32) -> bool:
    return d != 0 and 10 / d > 1
"#;
    // `10 / 0` would trap if the right-hand side were evaluated.
    assert_eq!(jit(src, "f", &[JitValue::Int(0)]), JitValue::Bool(false));
    assert_eq!(jit(src, "f", &[JitValue::Int(2)]), JitValue::Bool(true));
    let or = r#"
def f(d: int32) -> bool:
    return d == 0 or 10 / d > 100
"#;
    assert_eq!(jit(or, "f", &[JitValue::Int(0)]), JitValue::Bool(true));
    assert_eq!(jit(or, "f", &[JitValue::Int(5)]), JitValue::Bool(false));
    assert_eq!(
        jit(
            r#"
def f(a: bool) -> bool:
    return not a
"#,
            "f",
            &[JitValue::Bool(true)]
        ),
        JitValue::Bool(false)
    );
    assert_eq!(
        jit(
            r#"
def f(a: bool, b: bool) -> bool:
    return a == b
"#,
            "f",
            &[JitValue::Bool(true), JitValue::Bool(true)]
        ),
        JitValue::Bool(true)
    );
}

#[test]
fn control_flow_and_calls() {
    let src = r#"
def abs(x: int32) -> int32:
    if x < 0:
        return -x

    else:
        return x

def clamp(x: int32, lo: int32, hi: int32) -> int32:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def both(x: int32) -> int32:
    a = abs(x)
    b = clamp(a, 0, 10)
    return a + b

def early(x: int32) -> int32:
    if x > 0:
        return 1
    else:
        return 2

def unit_then_value(x: int32) -> int32:
    if x > 0:
        y = x
    return x * 2
"#;
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
    let src = r#"
def noop(x: int32):
    y = x + 1

def f() -> int32:
    noop(1)
    return 7
"#;
    assert_eq!(jit(src, "noop", &[JitValue::Int(1)]), JitValue::Unit);
    assert_eq!(jit(src, "f", &[]), JitValue::Int(7));
}

#[test]
fn unsupported_features_are_diagnosed_not_miscompiled() {
    let cases = [
        (
            r#"
def f() -> int32:
    g = lambda x: x
    return 1
"#,
            "closures",
        ),
        (
            r#"
def f(x: float) -> str:
    return f"{x}"
"#,
            "f-strings",
        ),
        (
            r#"
class P:
    x: int32

def f(p: P) -> str:
    return f"{p}"
"#,
            "f-strings",
        ),
        (
            r#"
def f(v: list[str]) -> int32:
    return 1
"#,
            "`list[str]`",
        ),
        (
            r#"
def f(s: str) -> str:
    return s |> std.text.trim()
"#,
            "std item",
        ),
        (
            r#"
def f(s: str) -> bool:
    return s < "a"
"#,
            "operator",
        ),
        (
            r#"
def f(s: str) -> str:
    return s.to_uppercase()
"#,
            "method",
        ),
    ];
    for (src, needle) in cases {
        let h = hir(src);
        let ctx = telvax_mlir::context();
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
def len_of(n: int) -> uint64:
    return f"{n}".len()

def same(n: int, text: str) -> bool:
    return f"{n}" == text

def mixed(n: int32, ok: bool) -> bool:
    return f"n={n}, ok={ok}!" == "n=-42, ok=True!"

def unsigned(n: uint8) -> bool:
    return f"{n}" == "200"

def only_text(a: int32) -> uint64:
    return f"constant".len()

def unsigned64(n: uint64) -> uint64:
    return f"{n}".len()

def nested(a: int32) -> bool:
    s = f"{a}"
    return f"[{s}|{s}]" == "[7|7]"
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
        jit(src, "mixed", &[JitValue::Int(-42), JitValue::Bool(false)]),
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
    let src = r#"
n = 3
m = n * 4
half = 0.5

def f(x: int32) -> int32:
    return x + m

def g(x: float) -> float:
    return x * half
"#;
    assert_eq!(jit(src, "f", &[JitValue::Int(1)]), JitValue::Int(13));
    assert_eq!(jit(src, "g", &[JitValue::Float(5.0)]), JitValue::Float(2.5));
}

#[test]
fn structs_are_passed_returned_and_nested() {
    let src = r#"
class P:
    x: int32
    y: int32

class Seg:
    a: P
    len: float
    flag: bool

def mk(a: int32) -> P:
    return P(y=a + 1, x=a)

def sum(p: P) -> int32:
    return p.x * 10 + p.y

def swap(p: P) -> P:
    return P(x=p.y, y=p.x)

def seg(a: int32) -> Seg:
    return Seg(len=1.5, a=mk(a), flag=a > 0)

def pick(c: bool) -> P:
    if c:
        return mk(1)
    else:
        return mk(5)

def through(a: int32) -> int32:
    return sum(swap(mk(a)))

def nested(a: int32) -> float:
    s = seg(a)
    if s.flag:
        return s.len + 3.0
    else:
        return s.len

def inner(a: int32) -> int32:
    s = seg(a)
    return s.a.y

def joined(c: bool) -> int32:
    p = pick(c)
    return p.x + p.y
"#;
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
def len() -> uint64:
    return "hello".len()

def cat() -> uint64:
    return ("ab" + "cde").len()

def empty(s: int32) -> bool:
    return ("" + "").is_empty()

def eq(a: int32) -> bool:
    s = "x" + "y"
    return s == "xy"

def ne_same_len(a: int32) -> bool:
    return "ab" + "c" != "abd"

def ne_len(a: int32) -> bool:
    return "ab" + "c" == "ab"

def through(s: str) -> str:
    return s + "!"

def chained(a: int32) -> bool:
    return through("hey") == "hey!"
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
    let h = hir(r#"
def fib(n: int32) -> int32:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"#);
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
    let src = r#"
def gcd(a: int, b: int) -> int:
    if b == 0:
        return a

    return gcd(b, a % b)

def collatz(n: int, steps: int) -> int:
    if n == 1:
        return steps
    if n % 2 == 0:
        return collatz(n / 2, steps + 1)
    else:
        return collatz(3 * n + 1, steps + 1)

def mix(x: float, n: int32) -> float:
    if n <= 0:
        return x
    else:
        return mix(x * 1.5 - 0.25, n - 1)
"#;
    let h = hir(src);
    let interp = telvax_runtime::Interp::new(h.clone()).unwrap();
    for (a, b) in [(12, 18), (17, 5), (1_000_000, 750), (7, 0)] {
        let want = interp
            .call(
                "gcd",
                vec![telvax_runtime::Value::Int(a), telvax_runtime::Value::Int(b)],
            )
            .unwrap();
        let telvax_runtime::Value::Int(want) = want else {
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
                vec![telvax_runtime::Value::Int(n), telvax_runtime::Value::Int(0)],
            )
            .unwrap();
        let telvax_runtime::Value::Int(want) = want else {
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
                telvax_runtime::Value::Float(2.0),
                telvax_runtime::Value::Int(10),
            ],
        )
        .unwrap();
    let telvax_runtime::Value::Float(want) = want else {
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
    let src = r#"
def sum(n: int) -> int:
    i: int = 0

    s: int = 0

    while i < n:
        s += i

        i += 1

    return s

def collatz(n: int) -> int:
    x = n
    steps: int = 0
    while x != 1:
        if x % 2 == 0:
            x = x / 2
        else:
            x = 3 * x + 1
        steps += 1
    return steps

def early(n: int32) -> int32:
    i = 0
    while True:
        if i == n:
            return i * 2
        i += 1
    return 0

def shadow(n: int32) -> int32:
    a = n
    b = a
    a = a + 10
    return a + b

def text(n: int32) -> uint64:
    s = ""
    i = 0
    while i < n:
        s = s + f"{i};"
        i += 1
    return s.len()

def strukt(n: int32) -> int32:
    p = P(x=0, y=1)
    i = 0
    while i < n:
        p = P(x=p.y, y=p.x + p.y)
        i += 1
    return p.x

class P:
    x: int32
    y: int32
"#;
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
    let src = r#"
def tri(n: int32) -> int32:
    s = 0

    for i in range(n):
        s += i

    return s

def incl(n: int32) -> int32:
    s = 0
    for i in range(1, n + 1):
        s += i
    return s

def nested(n: int32) -> int32:
    c = 0
    for i in range(n):
        for j in range(i):
            c += j
    return c

def empty(n: int32) -> int32:
    c = 0
    for i in range(5, n):
        c += 1
    return c

def unsigned(n: uint8) -> uint8:
    c: uint8 = 0
    for i in range(250, 256):
        c += 1
    return c

def at_max() -> int32:
    c = 0
    hi: uint8 = 255
    for i in range(hi - 3, hi):
        c += 1
    return c

def ret_inside(n: int32) -> int32:
    for i in range(n):
        if i * i > 50:
            return i
    return 0 - 1

def f64s(n: int32) -> float:
    s = 0.0
    for i in range(n):
        s += 0.5
    return s
"#;
    assert_eq!(jit(src, "tri", &[JitValue::Int(10)]), JitValue::Int(45));
    assert_eq!(jit(src, "incl", &[JitValue::Int(10)]), JitValue::Int(55));
    assert_eq!(jit(src, "nested", &[JitValue::Int(5)]), JitValue::Int(10));
    assert_eq!(jit(src, "empty", &[JitValue::Int(3)]), JitValue::Int(0));
    assert_eq!(jit(src, "incl", &[JitValue::Int(0)]), JitValue::Int(0));
    assert_eq!(
        jit(src, "at_max", &[]),
        JitValue::Int(3),
        "a range that ends at the maximum must terminate"
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
    let src = r#"
def at(i: int32) -> int32:
    v = [10, 20, 30]

    return v[i]

def total() -> int32:
    v = [1, 2, 3, 4]
    s = 0
    for x in v:
        s += x
    return s

def dot() -> float:
    a = [1.5, 2.0, 3.0]
    b = [2.0, 4.0, 0.5]
    s = 0.0
    for i in range(a.len()):
        s += a[i] * b[i]
    return s

def count() -> uint64:
    return std.iter.count([True, False, True])

def flags() -> int32:
    f = [True, False, True]
    c = 0
    for b in f:
        if b:
            c += 1
    return c

def nonempty() -> bool:
    return not [1, 2].is_empty()

def small(i: uint8) -> int32:
    v = [1, 2, 3]
    return v[i]

def sum_to(n: int32) -> int32:
    v = [0, 0, 0]
    t = 0
    for i in range(n):
        t += [i, i, i][2]
    return t
"#;
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
            r#"
def f() -> int32:
    v = ["a", "b"]
    return 1
"#,
            "`list[str]`",
        ),
        (
            r#"
class P:
    x: int32

def f() -> int32:
    v = [P(x=1)]
    return 1
"#,
            "`list[P]`",
        ),
        (
            r#"
def f(v: list[list[int32]]) -> int32:
    return 1
"#,
            "`list[list[int32]]`",
        ),
    ] {
        let h = hir(src);
        let ctx = telvax_mlir::context();
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
class P:
    x: int32
    name: str
    tags: list[int32]

class W:
    p: P
    n: int32

def elems() -> int32:
    v = [1, 2, 3]
    v[0] = 10
    v[2] += v[0]
    v[1] *= 5
    return v[0] + v[1] + v[2]

def fields_ok() -> bool:
    p = P(x=1, name="a", tags=[7, 8])
    p.x = 5
    p.x += 1
    p.name = p.name + "!"
    p.tags[1] = p.x
    return f"{p.name}{p.x}{p.tags[0]}{p.tags[1]}" == "a!676"

def nested() -> int32:
    w = W(p=P(x=1, name="n", tags=[0, 0]), n=2)
    w.p.x = 40
    w.p.tags[0] = 2
    w.n = 7
    return w.p.x + w.p.tags[0] + w.n

def fill(v: list[int32], x: int32):
    v[0] = x
    v[1] = x * 2

def via_param() -> int32:
    v = [0, 0]
    fill(v, 4)
    return v[0] + v[1]

def value_semantics() -> int32:
    a = P(x=1, name="a", tags=[])
    b = a
    a.x = 99
    return b.x

def reference_semantics() -> int32:
    a = [1, 2]
    b = a
    a[0] = 50
    return b[0]

def loop_writes() -> int32:
    v = [0, 0, 0, 0]
    for i in range(4):
        v[i] = i * i
    s = 0
    for x in v:
        s += x
    return s

def mut_param(n: int32) -> int32:
    n += 1
    return n * 2

def bubble() -> int32:
    v = [5, 2, 9, 1, 7]
    n = v.len()
    for i in range(n):
        for j in range(n - 1 - i):
            if v[j] > v[j + 1]:
                t = v[j]
                v[j] = v[j + 1]
                v[j + 1] = t
    return v[0] * 10000 + v[1] * 1000 + v[2] * 100 + v[3] * 10 + v[4]

def oob(k: int32) -> int32:
    v = [1, 2]
    v[k] = 5
    return v[0]
"#;

#[test]
fn places_match_the_interpreter() {
    let h = hir(PLACES);
    let interp = telvax_runtime::Interp::new(h.clone()).unwrap();
    for f in [
        "elems",
        "nested",
        "via_param",
        "value_semantics",
        "reference_semantics",
        "loop_writes",
        "bubble",
    ] {
        let telvax_runtime::Value::Int(want) = interp.call(f, vec![]).unwrap() else {
            panic!("{f}")
        };
        assert_eq!(jit_call(&h, f, &[]).unwrap(), JitValue::Int(want), "{f}");
    }
    let telvax_runtime::Value::Bool(want) = interp.call("fields_ok", vec![]).unwrap() else {
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

// ---- native tensors and quantization ---------------------------------------------------------

const TENSORS: &str = r#"
def a23() -> Tensor[float32, [2, 3]]:
    return std.tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

def b32() -> Tensor[float32, [3, 2]]:
    return std.tensor.from_vec([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

def mm_sum() -> float32:
    return std.tensor.sum(std.tensor.matmul(a23(), b32()))

def mm_at(i: int32, j: int32) -> float32:
    return std.tensor.matmul(a23(), b32())[i][j]

def transposed() -> float32:
    return std.tensor.transpose(a23())[2][1]

def vec_mm() -> float32:
    v: Tensor[float32, [3]] = std.tensor.from_vec([1.0, 0.0, -1.0])
    return std.tensor.matmul(a23(), v)[1]

def elementwise() -> float32:
    x = a23()
    y = std.tensor.scale(x, 0.5)
    return std.tensor.sum(std.tensor.sub(std.tensor.add(x, y), std.tensor.mul(y, y)))

def divide() -> float32:
    return std.tensor.sum(std.tensor.div(a23(), std.tensor.scale(a23(), 2.0)))

def activations() -> float32:
    t: Tensor[float32, [4]] = std.tensor.from_vec([-2.0, -0.5, 0.5, 2.0])
    return std.tensor.sum(std.tensor.relu(t)) + std.tensor.sum(std.tensor.sigmoid(t))

def reductions() -> float32:
    t: Tensor[float32, [4]] = std.tensor.from_vec([3.0, 9.0, 2.0, 9.0])
    return std.tensor.mean(t) + std.tensor.dot(t, t)

def best() -> uint64:
    t: Tensor[float32, [5]] = std.tensor.from_vec([1.0, 7.0, 3.0, 7.0, 0.0])
    return std.tensor.argmax(t)

def zeros_sum() -> float32:
    z: Tensor[float32, [2, 2]] = std.tensor.zeros()
    return std.tensor.sum(z) + 1.0

def reshaped() -> float32:
    r: Tensor[float32, [3, 2]] = std.tensor.reshape(a23())
    return r[2][1]

def round_trip() -> uint64:
    return std.iter.count(std.tensor.to_vec(a23()))

def f64_math() -> float:
    t: Tensor[float, [3]] = std.tensor.from_vec([0.1, 0.2, 0.3])
    return std.tensor.dot(t, t)

def oob(i: int32) -> float32:
    return a23()[i][0]
"#;

#[test]
fn native_tensors_match_the_interpreter_bit_for_bit() {
    let h = hir(TENSORS);
    let interp = telvax_runtime::Interp::new(h.clone()).unwrap();
    for f in [
        "mm_sum",
        "transposed",
        "vec_mm",
        "elementwise",
        "divide",
        "activations",
        "reductions",
        "zeros_sum",
        "reshaped",
        "f64_math",
    ] {
        let telvax_runtime::Value::Float(want) = interp.call(f, vec![]).unwrap() else {
            panic!("{f}")
        };
        let JitValue::Float(got) = jit_call(&h, f, &[]).unwrap() else {
            panic!("{f}")
        };
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{f}: native {got} vs interpreted {want}"
        );
    }
    for f in ["best", "round_trip"] {
        let telvax_runtime::Value::Int(want) = interp.call(f, vec![]).unwrap() else {
            panic!("{f}")
        };
        assert_eq!(jit_call(&h, f, &[]).unwrap(), JitValue::Int(want), "{f}");
    }
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let telvax_runtime::Value::Float(want) = interp
            .call(
                "mm_at",
                vec![telvax_runtime::Value::Int(i), telvax_runtime::Value::Int(j)],
            )
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(
            jit_call(&h, "mm_at", &[JitValue::Int(i), JitValue::Int(j)]).unwrap(),
            JitValue::Float(want)
        );
    }
    assert_eq!(
        jit_call(&h, "mm_at", &[JitValue::Int(1), JitValue::Int(1)]).unwrap(),
        JitValue::Float(154.0)
    );
}

#[test]
fn quantized_native_matmul_matches_the_quantized_interpreter() {
    use telvax_hir::Quantization;
    let src = r#"
def a() -> Tensor[float32, [4, 8]]:
    v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x = -2.25
    for i in range(32):
        v[i] = x
        x += 0.7
        if x > 2.5:
            x -= 4.9
    return std.tensor.from_vec(v)

def b() -> Tensor[float32, [8, 3]]:
    v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x = 1.6
    for i in range(24):
        v[i] = x
        x -= 0.45
        if x < -1.7:
            x += 3.3
    return std.tensor.from_vec(v)

def checksum() -> float32:
    c = std.tensor.matmul(a(), b())
    s = 0.0
    w = 1.0
    for i in range(4):
        for j in range(3):
            s += c[i][j] * w
            w += 1.0
    return s
"#;
    let h = hir(src);
    let run = |q: Quantization| {
        let interp = telvax_runtime::Interp::new(h.clone()).unwrap();
        interp.set_quantization(q);
        let telvax_runtime::Value::Float(x) = interp.call("checksum", vec![]).unwrap() else {
            panic!()
        };
        let native = jit_call_with(&h, "checksum", &[], q).unwrap();
        (x, native)
    };
    let (exact_i, exact_n) = run(Quantization::None);
    let (quant_i, quant_n) = run(Quantization::Int8);
    assert_eq!(exact_n, JitValue::Float(exact_i), "float path agrees");
    assert_eq!(
        quant_n,
        JitValue::Float(quant_i),
        "int8 path agrees bit for bit"
    );
    assert_ne!(exact_i, quant_i, "quantization changes the numbers");
    assert!(
        ((exact_i - quant_i) / exact_i).abs() < 0.05,
        "{exact_i} vs {quant_i}"
    );

    let text = telvax_mlir::emit_mlir_with(&h, false, Quantization::Int8).unwrap();
    assert!(
        text.contains("arith.extsi") && text.contains("math.round") && text.contains("i8"),
        "the int8 kernel is in the IR"
    );
    let plain = telvax_mlir::emit_mlir_with(&h, false, Quantization::None).unwrap();
    assert!(
        !plain.contains("math.round"),
        "the float build has no quantization code"
    );
}

#[test]
fn native_tensor_indexing_is_bounds_checked_and_symbolic_shapes_are_rejected() {
    let ctx = telvax_mlir::context();
    for (src, needle) in [
        (
            r#"
def f(t: Tensor[float32, [N]]) -> float32:
    return 1.0
"#,
            "`Tensor[float32, [N]]`",
        ),
        (
            r#"
def f(t: Tensor[int32, [2]]) -> int32:
    return 1
"#,
            "`Tensor[int32, [2]]`",
        ),
        (
            r#"
def f(t: Tensor[float32, [4]]) -> Tensor[float32, [4]]:
    return std.tensor.softmax(t)
"#,
            "std::tensor::softmax",
        ),
    ] {
        let h = hir(src);
        let errs = lower(&ctx, &h)
            .err()
            .unwrap_or_else(|| panic!("expected an error for {src}"));
        assert!(
            errs.iter().any(|d| d.message.contains(needle)),
            "{src}: {errs:?}"
        );
    }
}

// ---- break and continue ----------------------------------------------------------------------

const BREAK_CONTINUE: &str = r#"
def first_square_over(limit: int32) -> int32:
    found = -1
    for i in range(1, 1000):
        if i * i > limit:
            found = i
            break
    return found

def sum_odd(n: int32) -> int32:
    s = 0
    for i in range(0, n + 1):
        if i % 2 == 0:
            continue
        s += i
    return s

def while_mix(n: int32) -> int32:
    i = 0
    s = 0
    while True:
        i += 1
        if i > n:
            break
        if i % 3 == 0:
            continue
        s += i
    return s

def nested(n: int32) -> int32:
    count = 0
    for a in range(n):
        for b in range(n):
            if b > a:
                break
            if (a + b) % 2 == 1:
                continue
            count += 1
    return count

def over_vec(n: int32) -> int32:
    v = [5, 8, 13, 21, 34]
    s = 0
    for x in v:
        if x > n:
            break
        if x == 8:
            continue
        s += x
    return s
"#;

#[test]
fn break_and_continue_match_the_interpreter() {
    let h = hir(BREAK_CONTINUE);
    let interp = telvax_runtime::Interp::new(h.clone()).unwrap();
    for (f, args) in [
        ("first_square_over", vec![50]),
        ("first_square_over", vec![5000000]),
        ("sum_odd", vec![10]),
        ("sum_odd", vec![0]),
        ("while_mix", vec![20]),
        ("nested", vec![9]),
        ("over_vec", vec![20]),
        ("over_vec", vec![100]),
    ] {
        let want = interp
            .call(
                f,
                args.iter()
                    .map(|a| telvax_runtime::Value::Int(*a))
                    .collect(),
            )
            .unwrap();
        let telvax_runtime::Value::Int(want) = want else {
            panic!("{want:?}")
        };
        let got = jit_call(
            &h,
            f,
            &args.iter().map(|a| JitValue::Int(*a)).collect::<Vec<_>>(),
        )
        .unwrap_or_else(|e| panic!("{f}{args:?}: {e}"));
        assert_eq!(got, JitValue::Int(want), "{f}{args:?}");
    }
    let sq = interp
        .call("first_square_over", vec![telvax_runtime::Value::Int(50)])
        .unwrap();
    assert!(matches!(sq, telvax_runtime::Value::Int(8)));
}

#[test]
fn break_outside_a_loop_is_an_error() {
    let (ast, _) = telvax_parser::parse(
        "def f() -> int32:\n    break\n    return 1\n\ndef g() -> int32:\n    def h():\n        continue\n    return 2\n",
    );
    let (_, diags) = telvax_hir::lower(&ast.unwrap());
    let msgs: Vec<_> = diags.iter().map(|d| d.message.as_str()).collect();
    assert!(
        msgs.iter().any(|m| m.contains("`break` outside of a loop")),
        "{msgs:?}"
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("`continue` outside of a loop")),
        "{msgs:?}"
    );
}

#[test]
fn enums_are_reported_as_unsupported_natively() {
    let h = hir(r#"
enum E:
    A
    B

def f() -> int32:
    match E.A:
        case E.A:
            return 1
        case E.B:
            return 2
"#);
    let err = jit_call(&h, "f", &[]).unwrap_err().to_string();
    assert!(err.contains("enums and `match`"), "{err}");
}

#[test]
fn per_channel_quantization_is_refused_natively() {
    let h = hir(r#"
def f(a: Tensor[float32, [2, 2]], b: Tensor[float32, [2, 2]]) -> Tensor[float32, [2, 2]]:
    return std.tensor.matmul(a, b)
"#);
    for q in [
        telvax_hir::Quantization::Int8Channel,
        telvax_hir::Quantization::Int4Channel,
    ] {
        let err = telvax_mlir::emit_mlir_with(&h, false, q).unwrap_err();
        assert!(err[0].message.contains("interpreter only"), "{err:?}");
    }
}

#[test]
fn impl_methods_compile_natively() {
    let src = r#"
class Point:
    x: int
    y: int

    def new(x: int, y: int) -> Point:
        return Point(x=x, y=y)

    def norm2(self) -> int:
        return self.x * self.x + self.y * self.y

    def plus(self, other: Point) -> Self:
        return Self.new(self.x + other.x, self.y + other.y)

def run(a: int) -> int:
    return Point.new(a, 4).plus(Point.new(0, 0)).norm2()
"#;
    assert_eq!(jit(src, "run", &[JitValue::Int(3)]), JitValue::Int(25));
    let h = hir(src);
    let interp = telvax_runtime::Interp::new(h).unwrap();
    let want = interp
        .call("run", vec![telvax_runtime::Value::Int(3)])
        .unwrap();
    assert!(matches!(want, telvax_runtime::Value::Int(25)));
}
