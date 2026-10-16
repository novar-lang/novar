# Telvax for VS Code

Language support for [Telvax](https://github.com/telvax/telvax), powered by the compiler itself:

* **Syntax highlighting** for `.tvx` files (keywords, `@endpoint`, f-strings with interpolation, the `|>` operator, tensor and table types).
* **Diagnostics** from the real parser and type checker, including tensor-shape mismatches.
* **Hover** with inferred types and function signatures.
* **Inlay hints** showing the inferred type of `let` bindings and loop variables.
* **Completion** for types, fields, table columns (also inside `t["..` and `t.sql("..`), methods of built-in and
  your own types, enum variants, `std.` paths, functions and variables, ordered by the type the code expects.
* **Go to definition**, **find references**, **rename** and **document symbols**.
* **Formatting** with the same rules as `telvax fmt`.
* **Snippets** for `fn`, `main`, `type`, `enum`, `match`, `for`, `test`, `endpoint` and `trait`.

## Setup

Install the `telvax` executable so that `telvax lsp` is on your `PATH`:

```sh
curl -fsSL https://raw.githubusercontent.com/telvax/telvax/main/scripts/install.sh | sh        # Linux, macOS
irm https://raw.githubusercontent.com/telvax/telvax/main/scripts/install.ps1 | iex             # Windows (PowerShell)
```

or point the `telvax.server.path` setting at the executable. Each GitHub release also carries the
`.vsix`, which installs with `code --install-extension telvax-<version>.vsix`.

| Setting | Default | |
|---|---|---|
| `telvax.server.path` | `telvax` | executable to start |
| `telvax.server.args` | `["lsp"]` | arguments that select language-server mode |
| `telvax.trace.server` | `off` | protocol tracing (`messages`, `verbose`) |

Command: **Telvax: Restart Language Server**.

## Publishing

A release publishes the extension by itself (the `vscode` job of `release.yml`): it packages the
`.vsix`, attaches it to the release and, when the secrets exist, publishes it.

1. Visual Studio Marketplace: create the publisher `telvax` at <https://marketplace.visualstudio.com/manage>
   (it needs an Azure DevOps organization), then create a Personal Access Token there with the scope
   *Marketplace > Manage* and *All accessible organizations*. Save it as the repository secret `VSCE_PAT`.
2. Open VSX (VSCodium, Cursor and others read from it): log in at <https://open-vsx.org> with a GitHub
   account linked to an Eclipse account, sign the publisher agreement, create a token under
   *Settings > Access Tokens*, and run `npx ovsx create-namespace telvax --pat <token>` once.
   Save the token as the repository secret `OVSX_PAT`.
3. Cut a release. A step is skipped while its secret is missing, so the two can be set up one at a time.

To publish by hand: `npx vsce publish --packagePath telvax-<version>.vsix --pat <token>` and
`npx ovsx publish telvax-<version>.vsix --pat <token>`.

## Development

```sh
npm install
npm run compile      # TypeScript -> out/
npm test             # tokenises sample code with the real TextMate engine
npx vsce package     # builds telvax-<version>.vsix
```
