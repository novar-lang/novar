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

## Usage

```sh
cargo build -p novar_cli            # the `novar` binary
novar check examples/rag-service.nv # parse + type-check with rendered diagnostics
novar run prog.nv fib 20            # interpret a function
novar serve examples/rag-service.nv --port 8080 --seed docs.jsonl
```

### Native backend (optional)

`novar jit` and `novar build` compile through MLIR and LLVM 22. Install LLVM
(`brew install llvm@22`), then:

```sh
. scripts/llvm-env.sh
cargo build -p novar_cli --features llvm
novar build prog.nv -o prog          # needs `fn main() -> i32`; the result is the exit code
novar jit prog.nv fib 30
```

The native subset covers integers, floats, booleans, `if`, `return` and calls.
Strings, structs and std items still run only in the interpreter (`novar run`/`serve`).

## License

Dual-licensed under MIT (`LICENSE-MIT`) or Apache 2.0 (`LICENSE-APACHE`).
