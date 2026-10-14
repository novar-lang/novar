// Downloads a real VS Code, installs this extension in it and runs `suite.js` inside it.
//   NOVAR_BIN=/path/to/novar npm run test:vscode
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const { runTests } = require("@vscode/test-electron");

async function main() {
  const novar = process.env.NOVAR_BIN;
  if (!novar || !fs.existsSync(novar)) {
    throw new Error("set NOVAR_BIN to the path of a built `novar` executable");
  }
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "novar-vscode-ws-"));
  fs.mkdirSync(path.join(workspace, ".vscode"));
  fs.writeFileSync(
    path.join(workspace, ".vscode", "settings.json"),
    JSON.stringify({ "novar.server.path": path.resolve(novar) }),
  );
  fs.writeFileSync(
    path.join(workspace, "ok.nv"),
    "struct Point { x: f64, y: f64 }\nfn norm(p: Point) -> f64 { p.x * p.x + p.y * p.y }\nfn main() -> f64 {\n    let origin = Point { x: 1.0, y: 2.0 };\n    norm(origin)\n}\n",
  );
  fs.writeFileSync(path.join(workspace, "bad.nv"), 'fn main() -> i32 { "not a number" }\n');
  fs.writeFileSync(path.join(workspace, "messy.nv"), "fn main()->i32{let x=1;x+2}\n");

  const userData = fs.mkdtempSync(path.join(os.tmpdir(), "novar-vscode-user-"));
  try {
    await runTests({
      extensionDevelopmentPath: path.resolve(__dirname, "../.."),
      extensionTestsPath: path.resolve(__dirname, "suite.js"),
      launchArgs: [workspace, "--disable-extensions", "--user-data-dir", userData, "--disable-gpu"],
      extensionTestsEnv: { NOVAR_TEST_WORKSPACE: workspace },
    });
  } finally {
    fs.rmSync(workspace, { recursive: true, force: true });
    fs.rmSync(userData, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
