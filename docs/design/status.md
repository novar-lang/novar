# Status against the engineering specification

What exists for each part of the original specification, how it was checked, and what is still open.
"Tested" means an automated test in this repository; "measured" means a benchmark run on an Apple
M-series laptop. Things that could not be exercised on this hardware are called out.

## 1. Setup and tooling

| Item | State |
|---|---|
| Rust nightly, edition 2024 | done (`rust-toolchain.toml`) |
| `melior` / `inkwell` | done, against LLVM/MLIR **22** (the specification asks for 18+) |
| `arrow` + `datafusion` 52+ | done: Arrow 59, DataFusion 55 |
| `pyo3` 0.21+ | done: PyO3 0.29 |
| Workspace crates | all present: `novar_lexer`, `_parser`, `_ast`, `_hir` (`compiler/semantic`), `_mlir` (`compiler/ir`), `_codegen`, `_runtime`, `_server`, `_cli`; plus `_diagnostics`, `_modules`, `_pkg`, `_lsp`, `_cluster`, `_wasm`, `_bench` |

## 2. Frontend

| Item | State |
|---|---|
| `logos` lexer, `chumsky` parser with recovery | done; recovery keeps later items after an error (tested) |
| Pipeline operator | done; desugared to a call in the HIR |
| Tensor types with shape enforcement | done: shapes are static types, `matmul`/`add`/`reshape`/... check dimensions at compile time (tested) |
| Relational types | done as `Table<Row>`: the schema is a struct, columns are checked, `t.col` is a typed column. Columns are primitives; there are no joins at the type level (use SQL) |
| AST for pipelines, tensors, endpoints | done |

## 3. Execution backend and interoperability

| Item | State |
|---|---|
| Tensors and tables backed by Arrow | done in the runtime (`ArrayRef` / `RecordBatch`). Reading a table column as a tensor shares the buffer (tested by pointer equality) |
| Ingestion from CSV, Parquet, memory | done and tested (CSV and Parquet round trips, `from_rows`, JSON). Files are decoded into Arrow arrays, not memory-mapped |
| Arrow C Data Interface | done through the Arrow PyCapsule protocol (`__arrow_c_array__`, `__arrow_c_stream__`): PyArrow and Polars in and out without copies (tested) |
| PyO3 bridge, NumPy / PyTorch | done: NumPy `__array_interface__`, PyTorch through NumPy, pointer equality checked (tested with real NumPy, PyTorch, Polars) |
| `cuda_array_interface` | protocol implemented (`DeviceTensor` adopts and re-exports a GPU buffer by pointer) and tested **with stand-in objects only**. It has not run against a physical GPU, and Novar has no GPU kernels: device tensors cannot be passed to Novar functions |
| Scalar bridging overhead | measured: 0.3-0.6 microseconds per call |
| CPU / GPU / TPU targets | **CPU only.** There is no GPU or TPU lowering |

## 4. MLOps and deployment

| Item | State |
|---|---|
| `@endpoint` -> routing table, validation, JSON | done (`novar_server`); validation derives from the types, including tensor shapes (tested) |
| Embedded Axum server | done, but the handlers run in the **interpreter**; native binaries do not serve HTTP |
| `novar build --serve` | done: one binary containing the program (all modules, no source tree needed at run time) (tested) |
| Dockerfile | generated (multi-stage, distroless runtime) and its content tested; **never built** because Docker is not available here |
| Real services | Ollama, OpenAI-compatible chat, PostgreSQL+pgvector: HTTP clients tested against local stand-in servers; pgvector tested against a real PostgreSQL 17; Ollama exercised against a real local Ollama (`nomic-embed-text`). The OpenAI client was only tested against a stand-in server (no API key) |

## 5. Alpha 0.1.0 targets (`benchmarks/`)

| Target | Measured |
|---|---|
| RAG overhead < 15 ms | p99 0.88 ms (1 connection), 6.0 ms (8 connections), 2,000 documents, with the offline model stand-ins. Real model and database latency adds to this |
| Base RSS < 30 MB | 17.8 MB idle |
| High-throughput vector ingestion | CSV 19.7 M rows/s, Parquet read 88 M rows/s, 84 K embedded documents/s into the in-memory store |

## 6. Launch

| Item | State |
|---|---|
| VS Code extension with LSP | done: language server (diagnostics, hover, completion, definition, inferred-type hints, module aware) tested over real stdio; the extension compiles, packages (`.vsix`) and its TextMate grammar is tested with the real tokenizer. It has **not been run inside VS Code** |
| WebAssembly playground | done: the engine is tested as a real `.wasm` under Node (check, run, hover, step limit). The web page itself was **not exercised in a browser** |
| Dual Apache-2.0 / MIT | done |

## 7. Ecosystem

| Item | State |
|---|---|
| `nvpm` package manager | done: manifests, semver resolution with backtracking, lockfile, directory and HTTP registries, checksums, yank, `import` across files and packages (tested end to end) |
| Distributed backends | `std::dist::map` runs on local threads, on `novar worker` processes (coordinator with retries and token auth; node loss tested with workers on one machine) or on **Ray** (tested against a real local Ray cluster). Multi-machine runs were not tried |
| Quantization passes | `--quantize int8` lowers `f32` `matmul` to a symmetric per-tensor int8 kernel with i32 accumulation, in both the interpreter and the MLIR lowering; the two agree bit for bit (tested). Only `matmul`; dynamic scales, no calibration, no int4 |

## Known gaps

* The native backend supports a subset (see the README); tables, SQL, closures and most of `std` are interpreter-only.
* Strings, vectors and tensors allocated by native code are never freed.
* The interpreter is a tree walker: fine for serving glue, not for heavy numeric loops (use `novar build`).
* No GPU/TPU code generation, no formatter or linter, no `break`/`continue`, no enums.
