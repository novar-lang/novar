// Runs the real WebAssembly build under Node: `scripts/build-playground.sh --node && node --test playground/test`.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const telvax = require("../pkg-node/telvax_wasm.js");

const examples = path.join(__dirname, "../../examples");
const read = (f) => fs.readFileSync(path.join(examples, f), "utf8");
const call = (fn, ...args) => JSON.parse(fn(...args));

test("a valid program checks cleanly", () => {
  const r = call(telvax.check, "def f() -> int32:\n    return 1 + 1\n");
  assert.deepStrictEqual([r.ok, r.diagnostics.length], [true, 0]);
});

test("errors come back with positions and rendered text", () => {
  const r = call(telvax.check, 'def f() -> int32:\n    return "oops"\n');
  assert.strictEqual(r.ok, false);
  assert.strictEqual(r.diagnostics[0].severity, "error");
  assert.deepStrictEqual([r.diagnostics[0].startLine, r.diagnostics[0].startCol], [1, 11]);
  assert.match(r.diagnostics[0].message, /expected `int32`, found `str`/);
  assert.match(r.rendered, /playground\.tvx:2:12/);
});

test("tensor shape errors are reported in the browser engine too", () => {
  const r = call(telvax.check, "def f(a: Tensor[float32, [2, 3]], b: Tensor[float32, [4, 5]]):\n    std.tensor.matmul(a, b)\n");
  assert.match(r.diagnostics[0].message, /inner dimensions differ: 3 and 4/);
});

test("functions are listed with their types", () => {
  const fns = call(telvax.functions, "def add(a: int, b: int) -> int:\n    return a + b\n\ndef hi():\n    pass\n");
  assert.deepStrictEqual(fns[0], { name: "add", params: [{ name: "a", ty: "int" }, { name: "b", ty: "int" }], ret: "int" });
  assert.strictEqual(fns[1].ret, "None");
});

test("running a function with JSON arguments", () => {
  const src = "class P:\n    x: int\n    y: int\n\ndef sum(p: P, k: int) -> int:\n    return (p.x + p.y) * k\n";
  const r = call(telvax.run, src, "sum", JSON.stringify([{ x: 3, y: 4 }, 10]));
  assert.deepStrictEqual([r.ok, r.result], [true, 70]);
});

test("the primes example prints the same text as the native build", () => {
  const r = call(telvax.run, read("basics/primes.tvx"), "main", "[]");
  assert.strictEqual(r.ok, true);
  assert.strictEqual(r.output, "primes below 100: 25\n5 squares add up to 55, the third is 9\n");
  assert.strictEqual(r.result, 25);
});

test("the sieve example runs", () => {
  const r = call(telvax.run, read("basics/sieve.tvx"), "main", "[]");
  assert.strictEqual(r.output, "2 3 5 7 11 13 17 19 23 29 \n");
  assert.strictEqual(r.result, 10);
});

test("tensor operations run in the browser", () => {
  const src = "def f(a: Tensor[float32, [2, 3]], b: Tensor[float32, [3, 2]]) -> Tensor[float32, [2, 2]]:\n    return std.tensor.matmul(a, b)\n";
  const r = call(telvax.run, src, "f", JSON.stringify([[[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]]));
  assert.deepStrictEqual(r.result, [[58, 64], [139, 154]]);
});

test("the RAG example runs with the offline model stand-ins", () => {
  const body = { query: "tensor shapes", top_k: 2 };
  const r = call(telvax.run, read("rag-service.tvx"), "handle_rag", JSON.stringify([body]));
  assert.strictEqual(r.ok, true, JSON.stringify(r));
  assert.strictEqual(r.result.query, "tensor shapes");
  assert.match(r.result.answer, /offline stand-in/);
});

test("runaway programs are stopped", () => {
  const r = call(telvax.run, "def spin() -> int32:\n    n = 0\n    while True:\n        n += 1\n    return n\n", "spin", "[]");
  assert.strictEqual(r.ok, false);
  assert.match(r.error, /step limit/);
});

test("runtime errors and bad requests are reported, not thrown", () => {
  assert.match(call(telvax.run, "def f(d: int32) -> int32:\n    return 1 / d\n", "f", "[0]").error, /division by zero/);
  assert.match(call(telvax.run, "def f():\n    pass\n", "g", "[]").error, /no function `g`/);
  assert.match(call(telvax.run, "def f(a: int32):\n    pass\n", "f", "[]").error, /takes 1 argument/);
  assert.match(call(telvax.run, "def f(a: int32):\n    pass\n", "f", "{").error, /not valid JSON/);
  assert.match(call(telvax.run, "def f(a: int32):\n    pass\n", "f", '["x"]').error, /argument `a`/);
  assert.match(call(telvax.run, "def f() -> int32:\n    return \"x\"\n", "f", "[]").error, /has errors/);
});

test("output is capped", () => {
  const src = 'def main():\n    i = 0\n    while i < 100000:\n        print("0123456789012345678901234567890123456789")\n        i += 1\n';
  const r = call(telvax.run, src, "main", "[]");
  assert.ok(r.output.length <= 256 * 1024, r.output.length);
});

test("hover and completion work in the engine", () => {
  const src = "class Point:\n    x: float\n    y: float\n\ndef f(p: Point) -> float:\n    return p.x\n";
  assert.match(telvax.hover(src, 5, 13), /float/);
  assert.match(telvax.hover(src, 4, 5), /def f\(p: Point\) -> float/);
  const items = call(telvax.complete, src, 5, 14);
  assert.ok(Array.isArray(items));
  const std = call(telvax.complete, "def f():\n    std.te", 1, 10);
  assert.ok(std.every((i) => typeof i.label === "string"));
});
