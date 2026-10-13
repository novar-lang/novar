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

The native subset covers integers, floats, booleans, `if`, `return`, calls, global
constants, structs of supported types, `let mut` with `=`/`+=`/..., `while`, `for i in a..b`
and `for x in vec`, arrays of numbers/bools (`[1, 2, 3]`, `v[i]`, `.len()`, bounds-checked),
element/field assignment (`v[i] = x`, `p.a.b += 1`, `mut` parameters), and strings (literals, `+`, `==`, `.len()`, `.is_empty()`, f-strings with integers, booleans
and strings, `std::io::print`/`println`). Closures, floats in f-strings, arrays of
strings/structs, tensors and the rest of std still run only in the interpreter (`novar run`/`serve`). Strings built with `+` are
never freed yet.

### Value semantics

Structs and strings are values: copying one and then changing the copy leaves the original
alone. A `Vec` is a reference: `let b = a;` makes both names share the same elements, so
`a[0] = 1` is visible through `b`, and a function taking `mut v: Vec<T>` can fill the caller's
vector. Untyped number literals take their type from how they are used (`let mut n = 0;`
becomes `u64` if you compare it with `v.len()`) and default to `i32` / `f64`.

## License

Dual-licensed under MIT (`LICENSE-MIT`) or Apache 2.0 (`LICENSE-APACHE`).
