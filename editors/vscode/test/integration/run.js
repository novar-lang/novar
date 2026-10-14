// Downloads a real VS Code, installs this extension in it and runs `suite.js` inside it.
//   TELVAX_BIN=/path/to/telvax npm run test:vscode
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const { runTests } = require("@vscode/test-electron");

async function main() {
  const telvax = process.env.TELVAX_BIN;
  if (!telvax || !fs.existsSync(telvax)) {
    throw new Error("set TELVAX_BIN to the path of a built `telvax` executable");
  }
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "telvax-vscode-ws-"));
  fs.mkdirSync(path.join(workspace, ".vscode"));
  fs.writeFileSync(
    path.join(workspace, ".vscode", "settings.json"),
    JSON.stringify({ "telvax.server.path": path.resolve(telvax) }),
  );
  fs.writeFileSync(
    path.join(workspace, "ok.tvx"),
    "struct Point { x: f64, y: f64 }\nfn norm(p: Point) -> f64 { p.x * p.x + p.y * p.y }\nfn main() -> f64 {\n    let origin = Point { x: 1.0, y: 2.0 };\n    norm(origin)\n}\n",
  );
  fs.writeFileSync(path.join(workspace, "bad.tvx"), 'fn main() -> i32 { "not a number" }\n');
  fs.writeFileSync(path.join(workspace, "messy.tvx"), "fn main()->i32{let x=1;x+2}\n");

  const userData = fs.mkdtempSync(path.join(os.tmpdir(), "telvax-vscode-user-"));
  try {
    await runTests({
      extensionDevelopmentPath: path.resolve(__dirname, "../.."),
      extensionTestsPath: path.resolve(__dirname, "suite.js"),
      launchArgs: [workspace, "--disable-extensions", "--user-data-dir", userData, "--disable-gpu"],
      extensionTestsEnv: { TELVAX_TEST_WORKSPACE: workspace },
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
