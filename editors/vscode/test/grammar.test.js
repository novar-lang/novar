// Tokenises Novar code with the same TextMate engine VS Code uses and checks the scopes.
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
    if (scope !== "source.novar") return null;
    const file = path.join(__dirname, "../syntaxes/novar.tmLanguage.json");
    return vsctm.parseRawGrammar(fs.readFileSync(file, "utf8"), file);
  },
});

/** Returns [text, scopes] for every token of every line. */
async function tokenize(source) {
  const grammar = await registry.loadGrammar("source.novar");
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
  const t = await tokenize("pub fn norm(p: Point, t: Tensor<f32, [768]>) -> f64 {}\nstruct Row { id: i64 }");
  has(t, "pub", "storage.modifier");
  has(t, "fn", "storage.type.function");
  has(t, "norm", "entity.name.function");
  has(t, "Point", "entity.name.type");
  has(t, "Tensor", "support.type.builtin");
  has(t, "f32", "support.type.primitive");
  has(t, "768", "constant.numeric.integer");
  has(t, "struct", "storage.type.struct");
  has(t, "Row", "entity.name.type.struct");
  has(t, "->", "keyword.operator.arrow");
});

test("the pipeline operator, paths and calls", async () => {
  const t = await tokenize("text |> std::text::clean_whitespace() |> embedder.embed()");
  has(t, "|>", "keyword.operator.pipe", 0);
  has(t, "|>", "keyword.operator.pipe", 1);
  has(t, "std", "support.module.std");
  has(t, "text", "entity.name.namespace", 1);
  has(t, "clean_whitespace", "entity.name.function.call");
  has(t, "embed", "entity.name.function.call");
});

test("attributes", async () => {
  const t = await tokenize('@endpoint(path = "/v1/rag/query", method = "POST")');
  has(t, "endpoint", "entity.name.function.decorator");
  has(t, "@", "punctuation.definition.attribute");
  has(t, '"', "string.quoted.double");
});

test("f-strings highlight their interpolations as code", async () => {
  const t = await tokenize('let s = f"hi {name}, you are {age + 1}!\\n";');
  has(t, "let", "storage.type");
  const start = t.findIndex(([x]) => x === 'f"');
  assert.ok(start >= 0, "f-string start");
  has(t, "name", "meta.interpolation");
  has(t, "age", "meta.interpolation");
  has(t, "1", "constant.numeric.integer");
  has(t, "\\n", "constant.character.escape");
  has(t, "{", "punctuation.section.interpolation.begin", 0);
});

test("numbers, booleans, comments and control flow", async () => {
  const t = await tokenize("for i in 1..=10_000 { if x >= 2.5e-3 && ok { return true; } } // done");
  has(t, "for", "keyword.control");
  has(t, "in", "keyword.control");
  has(t, "..=", "keyword.operator.range");
  has(t, "10_000", "constant.numeric.integer");
  has(t, "2.5e-3", "constant.numeric.float");
  has(t, ">=", "keyword.operator.comparison");
  has(t, "&&", "keyword.operator.logical");
  has(t, "true", "constant.language.boolean");
  has(t, "// done", "comment.line");
});

test("the example programs tokenise without raw leftovers", async () => {
  const dir = path.join(__dirname, "../../../examples");
  const files = [path.join(dir, "rag-service.nv"), path.join(dir, "basics/primes.nv"), path.join(dir, "basics/sieve.nv")];
  for (const file of files) {
    const tokens = await tokenize(fs.readFileSync(file, "utf8"));
    assert.ok(tokens.length > 50, file);
    const unscoped = tokens.filter(([text, scopes]) => /[A-Za-z_]{2,}/.test(text) && scopes.length === 1 && scopes[0] === "source.novar");
    // Plain variables stay unscoped; keywords and declarations must never be.
    assert.ok(!unscoped.some(([text]) => ["fn", "let", "struct", "return", "import", "pub"].includes(text)), file);
  }
});
