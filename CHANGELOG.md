# Changelog

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
