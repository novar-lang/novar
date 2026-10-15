// Tokenises Telvax code with the same TextMate engine VS Code uses and checks the scopes.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vsctm = require("vscode-textmate");
const oniguruma = require("vscode-oniguruma");

const wasm = fs.readFileSync(path.join(__dirname, "../node_modules/vscode-oniguruma/release/onig.wasm")).buffer;
const onigLib = oniguruma.loadWASM(wasm).then(() => ({
  createOnigScanner: (patterns) => new oniguruma.OnigScanner(patterns),
  createOnigString: (s) => new oniguruma.OnigString(s),
}));

const registry = new vsctm.Registry({
  onigLib,
  loadGrammar: async (scope) => {
    if (scope !== "source.telvax") return null;
    const file = path.join(__dirname, "../syntaxes/telvax.tmLanguage.json");
    return vsctm.parseRawGrammar(fs.readFileSync(file, "utf8"), file);
  },
});

/** Returns [text, scopes] for every token of every line. */
async function tokenize(source) {
  const grammar = await registry.loadGrammar("source.telvax");
  let ruleStack = vsctm.INITIAL;
  const out = [];
  for (const line of source.split("\n")) {
    const result = grammar.tokenizeLine(line, ruleStack);
    for (const t of result.tokens) out.push([line.slice(t.startIndex, t.endIndex), t.scopes]);
    ruleStack = result.ruleStack;
  }
  return out;
}

function scopeOf(tokens, text, nth = 0) {
  const hits = tokens.filter(([t]) => t === text || t.trim() === text);
  assert.ok(hits[nth], `no token "${text}" (found ${tokens.map(([t]) => JSON.stringify(t)).join(" ")})`);
  return hits[nth][1];
}

function has(tokens, text, scope, nth = 0) {
  const scopes = scopeOf(tokens, text, nth);
  assert.ok(scopes.some((s) => s.startsWith(scope)), `"${text}" should be ${scope}, got ${scopes.join(", ")}`);
}

test("declarations and types", async () => {
  const t = await tokenize("fn norm(p: Point, t: Tensor[f32, 768]) -> Float:\n    0.0\ntype Row:\n    id: Int");
  has(t, "fn", "storage.type.function");
  has(t, "norm", "entity.name.function");
  has(t, "Point", "entity.name.type");
  has(t, "Tensor", "support.type.builtin");
  has(t, "f32", "support.type.primitive");
  has(t, "768", "constant.numeric.integer");
  has(t, "type", "storage.type.struct");
  has(t, "Row", "entity.name.type.struct");
  has(t, "->", "keyword.operator.arrow");
  has(t, "Int", "support.type.primitive");
});

test("the pipeline operator, paths and calls", async () => {
  const t = await tokenize("text |> std.text.clean_whitespace() |> embedder.embed()");
  has(t, "|>", "keyword.operator.pipe", 0);
  has(t, "|>", "keyword.operator.pipe", 1);
  has(t, "std", "support.module.std");
  has(t, "clean_whitespace", "entity.name.function.call");
  has(t, "embed", "entity.name.function.call");
});

test("decorators", async () => {
  const t = await tokenize('@endpoint(POST, "/v1/rag/query")');
  has(t, "endpoint", "entity.name.function.decorator");
  has(t, "@", "punctuation.definition.attribute");
  has(t, '"', "string.quoted.double");
});

test("f-strings highlight their interpolations as code", async () => {
  const t = await tokenize('let s = f"hi {name}, you are {age + 1}!\\n"');
  const start = t.findIndex(([x]) => x === 'f"');
  assert.ok(start >= 0, "f-string start");
  has(t, "name", "meta.interpolation");
  has(t, "age", "meta.interpolation");
  has(t, "1", "constant.numeric.integer");
  has(t, "\\n", "constant.character.escape");
  has(t, "{", "punctuation.section.interpolation.begin", 0);
});

test("single-quoted strings and f-strings", async () => {
  const t = await tokenize("a = 'it' + f'{b}'");
  has(t, "'", "string.quoted.single");
  has(t, "b", "meta.interpolation");
});

test("triple-quoted strings, short operators and slices", async () => {
  const t = await tokenize('let s = """a\nb"""\nlet n = 2 ** 3 // 2\nlet v = m ?? xs[1:2]');
  has(t, '"""', "string.quoted.triple");
  has(t, "**", "keyword.operator.arithmetic");
  has(t, "//", "keyword.operator.arithmetic");
  has(t, "??", "keyword.operator.coalesce");
  has(t, "1", "constant.numeric.integer");
});

test("f-strings take quotes and formats inside their braces", async () => {
  const t = await tokenize('let s = f"{m["a"]:>8} and {x=}"');
  has(t, "m[", "meta.interpolation");
  has(t, "a", "string.quoted.double");
  has(t, "8", "constant.numeric.integer");
  has(t, "x", "meta.interpolation");
});

test("alternatives and list patterns in a case", async () => {
  const t = await tokenize("match x:\n    case 1 | 2:\n        a\n    case [first, *rest] if first > 1:\n        b");
  has(t, "|", "keyword.operator.alternative");
  has(t, "case", "keyword.control");
  has(t, "if", "keyword.control");
});

test("numbers, constants, comments and control flow", async () => {
  const t = await tokenize("for i in range(10_000):\n    if x >= 2.5e-3 and ok or not done:\n        return True  # done");
  has(t, "for", "keyword.control");
  has(t, "in", "keyword.control");
  has(t, "10_000", "constant.numeric.integer");
  has(t, "2.5e-3", "constant.numeric.float");
  has(t, ">=", "keyword.operator.comparison");
  has(t, "and", "keyword.operator.logical");
  has(t, "or", "keyword.operator.logical");
  has(t, "not", "keyword.operator.logical");
  has(t, "True", "constant.language");
  has(t, "# done", "comment.line");
});

test("match, lambda, imports and pattern tests", async () => {
  const t = await tokenize("from std.http import Status\nmatch x:\n    case Some(n):\n        var f = a => a\nif y is None:\n    pass");
  has(t, "from", "keyword.control.import");
  has(t, "match", "keyword.control");
  has(t, "case", "keyword.control");
  has(t, "=>", "keyword.operator.arrow");
  has(t, "var", "storage.type");
  has(t, "is", "keyword.control");
  has(t, "None", "constant.language");
  has(t, "pass", "keyword.control");
});

test("the example programs tokenise without raw leftovers", async () => {
  const dir = path.join(__dirname, "../../../examples");
  const files = [path.join(dir, "rag-service.tvx"), path.join(dir, "basics/primes.tvx"), path.join(dir, "basics/sieve.tvx"), path.join(dir, "errors/parse-config.tvx")];
  for (const file of files) {
    const tokens = await tokenize(fs.readFileSync(file, "utf8"));
    assert.ok(tokens.length > 50, file);
    const unscoped = tokens.filter(([text, scopes]) => /[A-Za-z_]{2,}/.test(text) && scopes.length === 1 && scopes[0] === "source.telvax");
    // Plain variables stay unscoped; keywords and declarations must never be.
    assert.ok(!unscoped.some(([text]) => ["fn", "let", "var", "return", "import", "from", "elif"].includes(text)), file);
  }
});
