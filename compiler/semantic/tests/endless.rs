//! A generic function or type that keeps asking for a bigger version of itself is an error, not
//! an instantiation that never ends.

use std::sync::mpsc;
use std::time::Duration;

use telvax_hir::lower;

/// Checks `src` on another thread; a checker that does not return fails the test.
fn messages(src: &'static str) -> Vec<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (program, errs) = telvax_parser::parse(src);
        assert!(errs.is_empty(), "parse: {errs:?}");
        let (_, diags) = lower(&program.unwrap());
        let _ = tx.send(diags.into_iter().map(|d| d.message).collect::<Vec<_>>());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("the checker did not finish ({e}):\n{src}"))
}

fn assert_growing(src: &'static str) {
    let m = messages(src);
    assert!(
        m.iter().any(|m| m.contains("keep growing")),
        "expected the growing-types error in {m:#?}"
    );
}

#[test]
fn a_function_that_calls_itself_with_a_bigger_type() {
    assert_growing("fn f[T](x: T) -> Int:\n    f([x])\n\nfn main() -> i32:\n    f(1)\n    0\n");
    assert_growing("fn f[T](x: T) -> Int:\n    f((x, x))\n\nfn main() -> i32:\n    f(1)\n    0\n");
    assert_growing("fn f[T](x: T) -> Int:\n    f(Some(x))\n\nfn main() -> i32:\n    f(1)\n    0\n");
}

#[test]
fn two_functions_that_grow_each_other() {
    assert_growing(
        "fn a[T](x: T) -> Int:\n    b([x])\n\nfn b[T](x: T) -> Int:\n    a([x])\n\nfn main() -> i32:\n    a(1)\n    0\n",
    );
}

#[test]
fn a_type_that_contains_a_bigger_version_of_itself() {
    assert_growing("type L[T]:\n    next: L[List[T]]\n\nfn f(x: L[Int]) -> Int:\n    1\n");
    assert_growing("type L[T]:\n    next: L[(T, T)]\n\nfn f(x: L[Int]) -> Int:\n    1\n");
}

#[test]
fn a_method_with_type_parameters_that_grows() {
    assert_growing(
        "type P:\n    n: Int\n\n    fn go[U](self, u: U) -> Int:\n        self.go((u, u))\n\nfn main() -> i32:\n    P(n: 1).go(1)\n    0\n",
    );
}

#[test]
fn recursion_that_does_not_grow_is_fine() {
    let m = messages(
        "fn f[T](x: T, n: Int) -> Int:\n    if n == 0:\n        0\n    else:\n        f(x, n - 1)\n\nfn main() -> i32:\n    f(1, 3)\n    0\n",
    );
    assert!(m.is_empty(), "{m:?}");
    let m = messages(
        "type Node[T]:\n    value: T\n    next: List[Node[T]]\n\nfn f(n: Node[Int]) -> Int:\n    n.value\n",
    );
    assert!(m.is_empty(), "{m:?}");
}

#[test]
fn big_types_a_person_writes_are_fine() {
    let m = messages(
        "fn id[T](x: T) -> T:\n    x\n\nfn main() -> i32:\n    let m: Map[Text, List[(Int, Text, Float)]] = std.map.new()\n    let a = id(m)\n    let b = id([[(1, 2.5, 'a')], [(2, 3.5, 'b')]])\n    0\n",
    );
    assert!(m.is_empty(), "{m:?}");
}
