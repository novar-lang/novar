# Changelog

## 0.0.2

### Language
* `break` and `continue` in `while` and `for` loops (interpreter and native backend).
* Enums with tuple and unit variants, and `match` over enums, `Result`, integers, strings and
  booleans, with exhaustiveness checking. Enums cross module boundaries and travel as JSON.
  They run in the interpreter; the native backend reports them as unsupported.

### Tools
* `novar fmt [--check]`: a token-based formatter that keeps comments and your line breaks. The
  language server formats documents too.
* `novar lint [--deny-warnings]`: unused variables, needless `mut`, unused private functions.
* `novar --version` and `--help`.

### Platforms and releases
* Windows: CI builds and tests it, releases include a `.zip`, and `scripts/install.ps1` installs
  it. Windows builds have not been run on a real machine by the author.
* Release binaries for Linux arm64 and macOS Intel; pushing a higher `version` to `main` tags and
  publishes a release automatically.
* Dependabot covers Cargo, pip, npm and GitHub Actions; the `time` advisories are fixed (Rust
  1.88 or newer is required).

### Measurements and docs
* The RAG endpoint was measured against a real Ollama and PostgreSQL+pgvector: about 1.9 ms mean
  overhead (`benchmarks/README.md`).
* `CONTRIBUTING.md`, issue templates, a CI job that builds the generated Dockerfile (not run yet).

## 0.0.1 - first public alpha

A first version for trying the language out. Expect rough edges and breaking changes.

### Works
* Front end: lexer, parser with error recovery, typed HIR, literal type inference, tensor shapes
  checked at compile time, `Table<Row>`, loops, `let mut`, arrays, f-strings, modules and `import`.
* `novar check | run | serve | jit | build | lsp | worker`.
* Interpreter on Arrow: tensors, tables, SQL (DataFusion), CSV and Parquet.
* `@endpoint` functions served over HTTP, with JSON validation; `build --serve` bundles the
  service into one executable and can write a Dockerfile.
* `--live` talks to real Ollama, OpenAI-compatible APIs and PostgreSQL + pgvector.
* Native backend (`--features llvm`, LLVM 22): ints, floats, strings, structs, static-shape `f32`
  tensors, int8 quantized `matmul`.
* `--device gpu` (`--features gpu`): `f32` tensor operations on the GPU through wgpu.
* `--emit ptx` and `--emit stablehlo` for NVIDIA and TPU toolchains (generated, **not executed on
  that hardware**).
* `nvpm` package manager, language server, VS Code extension, Wasm playground, Python bindings
  (zero-copy NumPy, Arrow, Polars, PyTorch), Ray executor, cluster workers.

### Known limits
See [`docs/design/status.md`](docs/design/status.md). Notably: native executables do not serve HTTP
yet, PTX / StableHLO / CUDA-array interchange have not run on real NVIDIA or TPU hardware, the
Dockerfile was never built, and there are no enums, `break` or `continue`.
