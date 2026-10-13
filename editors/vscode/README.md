# Novar for VS Code

Language support for [Novar](https://github.com/ntsation/novar), powered by the compiler itself:

* **Syntax highlighting** for `.nv` files (keywords, `@endpoint`, f-strings with interpolation, the `|>` operator, tensor and table types).
* **Diagnostics** from the real parser and type checker, including tensor-shape mismatches.
* **Hover** with inferred types and function signatures.
* **Inlay hints** showing the inferred type of `let` bindings and loop variables.
* **Completion** for fields, table columns, methods, `std::` paths, functions and variables.
* **Go to definition** and **document symbols**.

## Setup

Install the `novar` executable (`cargo install --path cli` from the repository) so that `novar lsp` is on your `PATH`,
or point the `novar.server.path` setting at it.

| Setting | Default | |
|---|---|---|
| `novar.server.path` | `novar` | executable to start |
| `novar.server.args` | `["lsp"]` | arguments that select language-server mode |
| `novar.trace.server` | `off` | protocol tracing (`messages`, `verbose`) |

Command: **Novar: Restart Language Server**.

## Development

```sh
npm install
npm run compile      # TypeScript -> out/
npm test             # tokenises sample code with the real TextMate engine
npx vsce package     # builds novar-0.1.0.vsix
```
