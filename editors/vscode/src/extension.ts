import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  Trace,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

function createClient(): LanguageClient {
  const config = vscode.workspace.getConfiguration("telvax");
  const command = config.get<string>("server.path", "telvax");
  const args = config.get<string[]>("server.args", ["lsp"]);

  const serverOptions: ServerOptions = {
    run: { command, args },
    debug: { command, args },
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "telvax" }, { scheme: "untitled", language: "telvax" }],
    outputChannelName: "Telvax Language Server",
  };
  return new LanguageClient("telvax", "Telvax", serverOptions, clientOptions);
}

async function start(context: vscode.ExtensionContext): Promise<void> {
  client = createClient();
  const trace = vscode.workspace.getConfiguration("telvax").get<string>("trace.server", "off");
  client.setTrace(trace === "verbose" ? Trace.Verbose : trace === "messages" ? Trace.Messages : Trace.Off);
  try {
    await client.start();
  } catch (err) {
    const path = vscode.workspace.getConfiguration("telvax").get<string>("server.path", "telvax");
    vscode.window.showErrorMessage(
      `Could not start the Telvax language server ("${path} lsp"): ${err}. ` +
        `Install telvax or set "telvax.server.path".`,
    );
    client = undefined;
  }
  context.subscriptions.push({ dispose: () => void client?.stop() });
}

/** Starts `telvax dap` for each debug session. */
class AdapterFactory implements vscode.DebugAdapterDescriptorFactory {
  createDebugAdapterDescriptor(): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
    const command = vscode.workspace.getConfiguration("telvax").get<string>("server.path", "telvax");
    return new vscode.DebugAdapterExecutable(command, ["dap"]);
  }
}

/** Fills in a launch configuration for the file in the editor, so that F5 works without a launch.json. */
class ConfigurationProvider implements vscode.DebugConfigurationProvider {
  resolveDebugConfiguration(
    _folder: vscode.WorkspaceFolder | undefined,
    config: vscode.DebugConfiguration,
  ): vscode.ProviderResult<vscode.DebugConfiguration> {
    if (!config.type && !config.request && !config.name) {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== "telvax") {
        vscode.window.showErrorMessage("Open a .tvx file to debug it.");
        return undefined;
      }
      config.type = "telvax";
      config.request = "launch";
      config.name = "Debug the current Telvax file";
      config.program = "${file}";
    }
    if (!config.program) {
      vscode.window.showErrorMessage('The launch configuration needs "program": the .tvx file to run.');
      return undefined;
    }
    return config;
  }
}

/** Runs the file in the editor in the debugger (or, with `noDebug`, without stopping). */
async function launchCurrentFile(noDebug: boolean): Promise<boolean> {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== "telvax") {
    vscode.window.showErrorMessage("Open a .tvx file to debug it.");
    return false;
  }
  await editor.document.save();
  const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
  return vscode.debug.startDebugging(folder, {
    type: "telvax",
    request: "launch",
    name: "Debug the current Telvax file",
    program: editor.document.uri.fsPath,
    noDebug,
  });
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
    vscode.debug.registerDebugAdapterDescriptorFactory("telvax", new AdapterFactory()),
    vscode.debug.registerDebugConfigurationProvider("telvax", new ConfigurationProvider()),
    vscode.commands.registerCommand("telvax.debugFile", () => launchCurrentFile(false)),
    vscode.commands.registerCommand("telvax.runFileWithoutDebugging", () => launchCurrentFile(true)),
    vscode.commands.registerCommand("telvax.restartServer", async () => {
      await client?.stop();
      await start(context);
    }),
    vscode.workspace.onDidChangeConfiguration(async (e) => {
      if (e.affectsConfiguration("telvax.server")) {
        await client?.stop();
        await start(context);
      }
    }),
  );
  await start(context);
}

export async function deactivate(): Promise<void> {
  await client?.stop();
}
