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

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
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
