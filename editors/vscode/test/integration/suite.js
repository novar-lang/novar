// Runs inside VS Code. Plain assertions: a thrown error fails the run.
const assert = require("node:assert");
const path = require("node:path");
const vscode = require("vscode");

const ws = process.env.TELVAX_TEST_WORKSPACE;
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
  const ok = await open("ok.tvx");
  assert.strictEqual(ok.languageId, "telvax", "the .tvx extension must map to the telvax language");

  const ext = vscode.extensions.getExtension("telvax.telvax");
  assert.ok(ext, "the extension is installed in the test host");
  await ext.activate();

  // Diagnostics come from the language server.
  const bad = await open("bad.tvx");
  const diagnostics = await until("diagnostics for bad.tvx", () => {
    const d = vscode.languages.getDiagnostics(bad.uri);
    return d.length ? d : null;
  });
  assert.ok(
    diagnostics.some((d) => d.message.includes("mismatched types")),
    `expected a type error, got ${JSON.stringify(diagnostics.map((d) => d.message))}`,
  );
  assert.deepStrictEqual(vscode.languages.getDiagnostics(ok.uri), [], "ok.tvx has no problems");

  // Hover on `norm` in the call on line 10.
  const hovers = await until("a hover", async () => {
    const h = await vscode.commands.executeCommand(
      "vscode.executeHoverProvider",
      ok.uri,
      new vscode.Position(9, 6),
    );
    return h && h.length ? h : null;
  });
  const hoverText = hovers
    .flatMap((h) => h.contents)
    .map((c) => (typeof c === "string" ? c : c.value))
    .join("\n");
  assert.ok(hoverText.includes("fn norm(p: Point) -> Float"), hoverText);

  // Go to definition jumps to the function.
  const defs = await vscode.commands.executeCommand(
    "vscode.executeDefinitionProvider",
    ok.uri,
    new vscode.Position(9, 6),
  );
  assert.ok(defs && defs.length, "definition");
  const target = defs[0].targetRange ?? defs[0].range;
  assert.strictEqual(target.start.line, 4);

  // Find all references: `origin` is declared on line 9 and used on line 10.
  const refs = await vscode.commands.executeCommand(
    "vscode.executeReferenceProvider",
    ok.uri,
    new vscode.Position(9, 11),
  );
  assert.deepStrictEqual(refs.map((r) => r.range.start.line).sort(), [8, 9]);

  // Rename the function `norm`: its definition and its call change together.
  const renamed = await vscode.commands.executeCommand(
    "vscode.executeDocumentRenameProvider",
    ok.uri,
    new vscode.Position(9, 6),
    "length",
  );
  const changes = renamed.entries().flatMap(([, edits]) => edits);
  assert.strictEqual(changes.length, 2, `rename edits: ${JSON.stringify(changes)}`);
  assert.ok(changes.every((e) => e.newText === "length"));

  // Completion offers the program's own names.
  const completions = await vscode.commands.executeCommand(
    "vscode.executeCompletionItemProvider",
    ok.uri,
    new vscode.Position(9, 6),
  );
  const labels = completions.items.map((i) => (typeof i.label === "string" ? i.label : i.label.label));
  assert.ok(labels.includes("norm") || labels.includes("origin"), `completions: ${labels.slice(0, 10)}`);

  // Formatting rewrites a messy file.
  const messy = await open("messy.tvx");
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
  assert.strictEqual(messy.getText(), "fn main() -> i32:\n    let x = 1\n    x + 2\n");

}

module.exports = { run };
