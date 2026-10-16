const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const contributes = manifest.contributes;

test("the extension contributes a debugger for .tvx files with breakpoints", () => {
  assert.deepStrictEqual(contributes.breakpoints, [{ language: "telvax" }]);
  const debuggers = contributes.debuggers;
  assert.strictEqual(debuggers.length, 1);
  const [telvax] = debuggers;
  assert.strictEqual(telvax.type, "telvax");
  assert.deepStrictEqual(telvax.languages, ["telvax"]);
  assert.ok(manifest.activationEvents.includes("onDebugResolve:telvax"));
});

test("the launch attributes are the ones `telvax dap` reads", () => {
  const launch = contributes.debuggers[0].configurationAttributes.launch;
  assert.deepStrictEqual(launch.required, ["program"]);
  assert.deepStrictEqual(Object.keys(launch.properties).sort(), [
    "allow",
    "args",
    "function",
    "noDebug",
    "program",
    "stopOnEntry",
  ]);
  assert.strictEqual(launch.properties.args.type, "array");
  assert.strictEqual(launch.properties.stopOnEntry.type, "boolean");
});

test("the initial configurations and snippets are launch configurations that name a program", () => {
  const [telvax] = contributes.debuggers;
  assert.ok(telvax.initialConfigurations.length > 0);
  for (const config of telvax.initialConfigurations) {
    assert.strictEqual(config.type, "telvax");
    assert.strictEqual(config.request, "launch");
    assert.ok(config.program, "a program");
  }
  for (const snippet of telvax.configurationSnippets) {
    assert.strictEqual(snippet.body.type, "telvax");
    assert.ok(snippet.body.program);
  }
});

test("the debug commands are declared and shown for Telvax files", () => {
  const commands = contributes.commands.map((c) => c.command);
  for (const id of ["telvax.debugFile", "telvax.runFileWithoutDebugging"]) {
    assert.ok(commands.includes(id), id);
    const entry = contributes.menus["editor/title/run"].find((m) => m.command === id);
    assert.ok(entry && entry.when.includes("telvax"), `${id} is in the run menu`);
  }
});

test("the source starts the adapter with `dap` and registers the commands", () => {
  const source = fs.readFileSync(path.join(root, "src", "extension.ts"), "utf8");
  assert.ok(source.includes('["dap"]'));
  assert.ok(source.includes('registerDebugAdapterDescriptorFactory("telvax"'));
  for (const id of ["telvax.debugFile", "telvax.runFileWithoutDebugging"]) {
    assert.ok(source.includes(`registerCommand("${id}"`), id);
  }
});
