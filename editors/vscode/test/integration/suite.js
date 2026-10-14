// Runs inside VS Code. Plain assertions: a thrown error fails the run.
const assert = require("node:assert");
const path = require("node:path");
const vscode = require("vscode");

const ws = process.env.NOVAR_TEST_WORKSPACE;
const open = async (name) => {
  const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(path.join(ws, name)));
  await vscode.window.showTextDocument(doc);
  return doc;
};

async function until(what, fn, timeoutMs = 30000) {
  const start = Date.now();
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() - start > timeoutMs) assert.fail(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

async function run() {
  const ok = await open("ok.nv");
  assert.strictEqual(ok.languageId, "novar", "the .nv extension must map to the novar language");

  const ext = vscode.extensions.getExtension("novar-lang.novar");
  assert.ok(ext, "the extension is installed in the test host");
  await ext.activate();

  // Diagnostics come from the language server.
  const bad = await open("bad.nv");
  const diagnostics = await until("diagnostics for bad.nv", () => {
    const d = vscode.languages.getDiagnostics(bad.uri);
    return d.length ? d : null;
  });
  assert.ok(
    diagnostics.some((d) => d.message.includes("mismatched types")),
    `expected a type error, got ${JSON.stringify(diagnostics.map((d) => d.message))}`,
  );
  assert.deepStrictEqual(vscode.languages.getDiagnostics(ok.uri), [], "ok.nv has no problems");

  // Hover on `norm` in the call on line 5.
  const hovers = await until("a hover", async () => {
    const h = await vscode.commands.executeCommand(
      "vscode.executeHoverProvider",
      ok.uri,
      new vscode.Position(4, 6),
    );
    return h && h.length ? h : null;
  });
  const hoverText = hovers
    .flatMap((h) => h.contents)
    .map((c) => (typeof c === "string" ? c : c.value))
    .join("\n");
  assert.ok(hoverText.includes("fn norm(p: Point) -> f64"), hoverText);

  // Go to definition jumps to the function.
  const defs = await vscode.commands.executeCommand(
    "vscode.executeDefinitionProvider",
    ok.uri,
    new vscode.Position(4, 6),
  );
  assert.ok(defs && defs.length, "definition");
  const target = defs[0].targetRange ?? defs[0].range;
  assert.strictEqual(target.start.line, 1);

  // Completion offers the program's own names.
  const completions = await vscode.commands.executeCommand(
    "vscode.executeCompletionItemProvider",
    ok.uri,
    new vscode.Position(4, 9),
  );
  const labels = completions.items.map((i) => (typeof i.label === "string" ? i.label : i.label.label));
  assert.ok(labels.includes("norm") || labels.includes("origin"), `completions: ${labels.slice(0, 10)}`);

  // Formatting rewrites a messy file.
  const messy = await open("messy.nv");
  // The server learns about the document asynchronously, so ask until it answers.
  const edits = await until("a formatting edit", async () => {
    const e = await vscode.commands.executeCommand(
      "vscode.executeFormatDocumentProvider",
      messy.uri,
      { tabSize: 4, insertSpaces: true },
    );
    return e && e.length ? e : null;
  });
  // VS Code splits the whole-file replacement into small edits; apply them and compare the text.
  const edit = new vscode.WorkspaceEdit();
  edit.set(messy.uri, edits);
  assert.ok(await vscode.workspace.applyEdit(edit), "the edits apply");
  assert.strictEqual(messy.getText(), "fn main() -> i32 { let x = 1; x + 2 }\n");

}

module.exports = { run };
