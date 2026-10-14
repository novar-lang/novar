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
| GPU execution (Metal / Vulkan / DX12) | done and **run on real hardware** (Apple M4 over Metal): `--device gpu` runs `f32` `add sub mul div scale relu sigmoid softmax matmul sum mean dot` as WGSL kernels through `wgpu` (`runtime` feature `gpu`). Tests compare against the CPU path within `f32` tolerance; other operations, other dtypes and int8 matmul stay on the CPU. Each call copies its inputs to the device and the result back, so it pays off for large matrices, not small ones. Operations nested in one expression (`softmax(matmul(relu(matmul(x, w1)), w2))`) keep their intermediate results on the GPU and copy only the leaves up and the final result down (tested by counting transfers); splitting the same code over `let` statements copies each intermediate through the host. Measured on the M4 for a 3-layer network (512x1024 input): 44 ms chained, 47 ms split, 9.0 s in the interpreter on the CPU. The M4 shares memory with the CPU, so avoiding the copies saves little here; discrete GPUs have not been measured |
| NVIDIA GPUs (PTX) | `novar build --emit ptx [--sm sm_80]` compiles kernels for the `std::tensor` operations a program uses (same set as above) with LLVM's NVPTX backend. The LLVM verifier accepts them and the PTX text is checked, but **it has never been executed on an NVIDIA GPU** (none available) and there is no CUDA host launcher |
| TPUs (StableHLO) | `novar build --emit stablehlo` translates straight-line tensor functions (let bindings, `std::tensor` calls, `+ - * /`, calls between such functions) to StableHLO text; functions with loops, branches or other types are reported as skipped. The output is **not run through XLA/PJRT or on a TPU** |

## 4. MLOps and deployment

| Item | State |
|---|---|
| `@endpoint` -> routing table, validation, JSON | done (`novar_server`); validation derives from the types, including tensor shapes (tested) |
| Embedded Axum server | done, but the handlers run in the **interpreter**; native binaries do not serve HTTP |
| `novar build --serve` | done: one binary containing the program (all modules, no source tree needed at run time) (tested) |
| Dockerfile | generated (multi-stage, distroless runtime) and its content tested. A CI job builds it and calls the container, but that job has **not run yet** and Docker is not available on the development machine |
| Real services | Ollama, OpenAI-compatible chat, PostgreSQL+pgvector: HTTP clients tested against local stand-in servers; pgvector tested against a real PostgreSQL 17; Ollama exercised against a real local Ollama (`nomic-embed-text`). The OpenAI client was only tested against a stand-in server (no API key) |

## 5. Alpha 0.1.0 targets (`benchmarks/`)

| Target | Measured |
|---|---|
| RAG overhead < 15 ms | with the offline stand-ins: p99 0.88 ms (1 connection), 6.0 ms (8 connections), 2,000 documents. With a real Ollama and a real PostgreSQL+pgvector (500 documents, sequential, LLM stubbed): about 1.9 ms mean overhead, ~6 ms p99, 14 MB RSS (see `benchmarks/README.md`) |
| Base RSS < 30 MB | 17.8 MB idle |
| High-throughput vector ingestion | CSV 19.7 M rows/s, Parquet read 88 M rows/s, 84 K embedded documents/s into the in-memory store |

## 6. Launch

| Item | State |
|---|---|
| VS Code extension with LSP | done: language server (diagnostics, hover, completion, definition, references, rename, formatting, inferred-type hints, module aware) tested over real stdio, and the extension is exercised **inside a real VS Code** (downloaded by `@vscode/test-electron`: language mapping, diagnostics, hover, definition, references, rename, completion, formatting; run locally on macOS and as the `vscode-host` CI job). The TextMate grammar is tested with the real tokenizer |
| WebAssembly playground | done: the engine is tested as a real `.wasm` under Node (check, run, hover, step limit) and the page itself runs in a **real Chromium** through Playwright (load, run, Ctrl+Enter, type errors, runtime errors, examples, share link; run locally and as the `playground-browser` CI job). Only Chromium has been tried |
| Dual Apache-2.0 / MIT | done |

## 7. Ecosystem

| Item | State |
|---|---|
| `nvpm` package manager | done: manifests, semver resolution with backtracking, lockfile, directory and HTTP registries, checksums, yank, `import` across files and packages (tested end to end) |
| Distributed backends | `std::dist::map` runs on local threads, on `novar worker` processes (coordinator with retries and token auth; node loss tested with workers on one machine) or on **Ray** (tested against a real local Ray cluster). Multi-machine runs were not tried |
| Quantization passes | `--quantize int8` lowers `f32` `matmul` to a symmetric per-tensor int8 kernel with i32 accumulation, in both the interpreter and the MLIR lowering; the two agree bit for bit (tested). `--quantize int8-channel` and `--quantize int4-channel` (interpreter only; the native backend refuses them) use one scale per row of the activations and per column of the weights, the latter with 4-bit weights. With columns 1000x apart in magnitude the worst relative error is 100% per-tensor, 0.75% per-channel int8 and 18% per-channel int4 (tested). Only `matmul`; scales are computed per call, so there is **no calibration** against sample data |

## Known gaps

* The full `novar` is about 51 MB because of Arrow and DataFusion; `novar-lite` (about 10 MB) leaves them out, so tables, CSV, Parquet and SQL report an error there.
* The native backend supports a subset (see the README); tables, SQL, closures and most of `std` are interpreter-only.
* Strings, vectors and tensors allocated by native code are never freed.
* The interpreter is a tree walker: fine for serving glue, not for heavy numeric loops (use `novar build`).
* Enums and `match` run in the interpreter only; the native backend reports them as unsupported.
* `novar fmt` keeps the line breaks you wrote (it fixes indentation, spacing and blank lines); it does not re-wrap long lines. `novar lint` only finds unused variables, needless `mut` and unused private functions.
* Windows builds come from CI and have not been exercised on a real Windows machine; the native backend is not built there. Each published release is now installed and run on Windows, Linux and macOS by the `verify` job of the release workflow (it first runs with the 0.0.4 release).
* PTX and StableHLO output have not been executed on NVIDIA or TPU hardware.
