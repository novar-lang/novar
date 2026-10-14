# Telvax for VS Code

Language support for [Telvax](https://github.com/telvax/telvax), powered by the compiler itself:

* **Syntax highlighting** for `.tvx` files (keywords, `@endpoint`, f-strings with interpolation, the `|>` operator, tensor and table types).
* **Diagnostics** from the real parser and type checker, including tensor-shape mismatches.
* **Hover** with inferred types and function signatures.
* **Inlay hints** showing the inferred type of `let` bindings and loop variables.
* **Completion** for fields, table columns, methods, `std::` paths, functions and variables.
* **Go to definition** and **document symbols**.

## Setup

Install the `telvax` executable so that `telvax lsp` is on your `PATH`:

```sh
curl -fsSL https://raw.githubusercontent.com/ntsation/telvax/main/scripts/install.sh | sh        # Linux, macOS
irm https://raw.githubusercontent.com/ntsation/telvax/main/scripts/install.ps1 | iex             # Windows (PowerShell)
```

or point the `telvax.server.path` setting at the executable. Each GitHub release also carries the
`.vsix`, which installs with `code --install-extension telvax-<version>.vsix`.

| Setting | Default | |
|---|---|---|
| `telvax.server.path` | `telvax` | executable to start |
| `telvax.server.args` | `["lsp"]` | arguments that select language-server mode |
| `telvax.trace.server` | `off` | protocol tracing (`messages`, `verbose`) |

Command: **Telvax: Restart Language Server**.

## Development

```sh
npm install
npm run compile      # TypeScript -> out/
npm test             # tokenises sample code with the real TextMate engine
npx vsce package     # builds telvax-0.1.0.vsix
```
