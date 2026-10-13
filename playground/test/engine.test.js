// Runs the real WebAssembly build under Node: `scripts/build-playground.sh --node && node --test playground/test`.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const novar = require("../pkg-node/novar_wasm.js");

const examples = path.join(__dirname, "../../examples");
const read = (f) => fs.readFileSync(path.join(examples, f), "utf8");
const call = (fn, ...args) => JSON.parse(fn(...args));

test("a valid program checks cleanly", () => {
  const r = call(novar.check, "fn f() -> i32 { 1 + 1 }");
  assert.deepStrictEqual([r.ok, r.diagnostics.length], [true, 0]);
});

test("errors come back with positions and rendered text", () => {
  const r = call(novar.check, 'fn f() -> i32 {\n    "oops"\n}\n');
  assert.strictEqual(r.ok, false);
  assert.strictEqual(r.diagnostics[0].severity, "error");
  assert.deepStrictEqual([r.diagnostics[0].startLine, r.diagnostics[0].startCol], [1, 4]);
  assert.match(r.diagnostics[0].message, /expected `i32`, found `String`/);
  assert.match(r.rendered, /playground\.nv:2:5/);
});

test("tensor shape errors are reported in the browser engine too", () => {
  const r = call(novar.check, "fn f(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [4, 5]>) { std::tensor::matmul(a, b); }");
  assert.match(r.diagnostics[0].message, /inner dimensions differ: 3 and 4/);
});

test("functions are listed with their types", () => {
  const fns = call(novar.functions, "fn add(a: i64, b: i64) -> i64 { a + b }\nfn hi() {}");
  assert.deepStrictEqual(fns[0], { name: "add", params: [{ name: "a", ty: "i64" }, { name: "b", ty: "i64" }], ret: "i64" });
  assert.strictEqual(fns[1].ret, "()");
});

test("running a function with JSON arguments", () => {
  const src = "struct P { x: i64, y: i64 }\nfn sum(p: P, k: i64) -> i64 { (p.x + p.y) * k }";
  const r = call(novar.run, src, "sum", JSON.stringify([{ x: 3, y: 4 }, 10]));
  assert.deepStrictEqual([r.ok, r.result], [true, 70]);
});

test("the primes example prints the same text as the native build", () => {
  const r = call(novar.run, read("basics/primes.nv"), "main", "[]");
  assert.strictEqual(r.ok, true);
  assert.strictEqual(r.output, "primes below 100: 25\n5 squares add up to 55, the third is 9\n");
  assert.strictEqual(r.result, 25);
});

test("the sieve example runs", () => {
  const r = call(novar.run, read("basics/sieve.nv"), "main", "[]");
  assert.strictEqual(r.output, "2 3 5 7 11 13 17 19 23 29 \n");
  assert.strictEqual(r.result, 10);
});

test("tensor operations run in the browser", () => {
  const src = "fn f(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [3, 2]>) -> Tensor<f32, [2, 2]> { std::tensor::matmul(a, b) }";
  const r = call(novar.run, src, "f", JSON.stringify([[[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]]));
  assert.deepStrictEqual(r.result, [[58, 64], [139, 154]]);
});

test("the RAG example runs with the offline model stand-ins", () => {
  const body = { query: "tensor shapes", top_k: 2 };
  const r = call(novar.run, read("rag-service.nv"), "handle_rag", JSON.stringify([body]));
  assert.strictEqual(r.ok, true, JSON.stringify(r));
  assert.strictEqual(r.result.query, "tensor shapes");
  assert.match(r.result.answer, /offline stand-in/);
});

test("runaway programs are stopped", () => {
  const r = call(novar.run, "fn spin() -> i32 { let mut n = 0; while true { n += 1; } n }", "spin", "[]");
  assert.strictEqual(r.ok, false);
  assert.match(r.error, /step limit/);
});

test("runtime errors and bad requests are reported, not thrown", () => {
  assert.match(call(novar.run, "fn f(d: i32) -> i32 { 1 / d }", "f", "[0]").error, /division by zero/);
  assert.match(call(novar.run, "fn f() {}", "g", "[]").error, /no function `g`/);
  assert.match(call(novar.run, "fn f(a: i32) {}", "f", "[]").error, /takes 1 argument/);
  assert.match(call(novar.run, "fn f(a: i32) {}", "f", "{").error, /not valid JSON/);
  assert.match(call(novar.run, "fn f(a: i32) {}", "f", '["x"]').error, /argument `a`/);
  assert.match(call(novar.run, "fn f() -> i32 { \"x\" }", "f", "[]").error, /has errors/);
});

test("output is capped", () => {
  const src = 'fn main() { let mut i = 0; while i < 100000 { std::io::println("0123456789012345678901234567890123456789"); i += 1; } }';
  const r = call(novar.run, src, "main", "[]");
  assert.ok(r.output.length <= 256 * 1024, r.output.length);
});

test("hover and completion work in the engine", () => {
  const src = "struct Point { x: f64, y: f64 }\nfn f(p: Point) -> f64 {\n    p.x\n}\n";
  assert.match(novar.hover(src, 2, 6), /f64/);
  assert.match(novar.hover(src, 1, 3), /fn f\(p: Point\) -> f64/);
  const items = call(novar.complete, src, 2, 7);
  assert.ok(Array.isArray(items));
  const std = call(novar.complete, "fn f() { std::te|", 0, 16);
  assert.ok(std.every((i) => typeof i.label === "string"));
});
