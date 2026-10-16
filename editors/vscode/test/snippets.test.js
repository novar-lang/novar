const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));

test("the snippets are declared and well formed", () => {
  const declared = manifest.contributes.snippets;
  assert.ok(declared.some((s) => s.language === "telvax"));
  for (const { path: file } of declared) {
    const snippets = JSON.parse(fs.readFileSync(path.join(root, file), "utf8"));
    assert.ok(Object.keys(snippets).length > 0);
    for (const [name, snippet] of Object.entries(snippets)) {
      assert.ok(snippet.prefix, `${name} has a prefix`);
      assert.ok(Array.isArray(snippet.body) && snippet.body.length > 0, `${name} has a body`);
    }
  }
});
