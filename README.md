# Novar

Novar is an open source programming language. This project is in its early stages.

## Layout

| Directory      | Purpose                                              |
|----------------|------------------------------------------------------|
| `spec/`        | Formal language specification (grammar, semantics)   |
| `compiler/`    | Lexer, parser, AST, semantic analysis, IR, codegen   |
| `runtime/`     | Runtime support (memory, panics, intrinsics)         |
| `stdlib/`      | Standard library written in Novar                    |
| `cli/`         | The `novar` command-line entry point                 |
| `tools/`       | Formatter, linter, LSP server, package manager       |
| `editors/`     | Editor integrations and syntax highlighting          |
| `examples/`    | Example programs                                     |
| `tests/`       | Unit, integration and fixture tests                  |
| `benchmarks/`  | Performance benchmarks                               |
| `docs/`        | User guide, reference, design and contributor docs   |
| `rfcs/`        | Proposals for language changes                       |
| `scripts/`     | Build and release helper scripts                     |

## License

To be decided (see `LICENSE`).
