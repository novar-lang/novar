# Novar

Novar is a statically typed, compiled language for data processing, ML pipelines and AI serving.
Tensor shapes are part of the type system, a pipeline operator (`|>`) composes stages, and a
function marked `@endpoint` becomes an HTTP route with validation and JSON handling generated
for it.

```novar
import std::ai::embeddings::OllamaEmbedder;
import std::http::Status;

let embedder = OllamaEmbedder::new("nomic-embed-text");

struct QueryPayload { query: String, top_k: i32 }

fn embed(text: String) -> Tensor<f32, [768]> {
    text |> std::text::clean_whitespace() |> std::text::truncate(512) |> embedder.embed()
}

@endpoint(path = "/v1/embed", method = "POST")
pub fn handle(payload: QueryPayload) -> Result<Tensor<f32, [768]>, Status> {
    if payload.query.is_empty() { return Err(Status::BadRequest("empty query")); }
    return Ok(embed(payload.query));
}
```

The full RAG service from the specification is in [`examples/rag-service.nv`](examples/rag-service.nv).

## Quick start

```sh
cargo build --release -p novar_cli          # target/release/novar
novar check examples/rag-service.nv         # parse + type-check, diagnostics point at the exact span
novar run examples/basics/primes.nv main    # interpret
novar serve examples/rag-service.nv         # HTTP server on :8080 (offline model stand-ins)
novar serve examples/rag-service.nv --live  # real Ollama / OpenAI / PostgreSQL+pgvector
novar run model.nv infer --device gpu       # tensor ops on the GPU (build with --features gpu)
novar build model.nv --emit stablehlo       # StableHLO for XLA / TPUs (--emit ptx for NVIDIA)
novar build --serve examples/rag-service.nv -o service --dockerfile   # one binary + Dockerfile
```

Nightly Rust is required (`rust-toolchain.toml` selects it). More in [`docs/guide`](docs/guide/tour.md).

## What is here

| Area | Where | Notes |
|---|---|---|
| Compiler front end | `compiler/{lexer,parser,ast,semantic}` | `logos` + `chumsky` with error recovery; typed HIR; literal type inference; tensor shapes checked statically |
| Interpreter and runtime | `runtime/` | Arrow-backed tensors and tables, CSV / Parquet / SQL (DataFusion), real Ollama / OpenAI / pgvector clients |
| Native backend | `compiler/{ir,codegen}` | MLIR (`melior`) and LLVM (`inkwell`) 22; JIT and executables; int8 quantization of `matmul`. Optional: `--features llvm` |
| HTTP serving | `server/`, `cli/` | `@endpoint` becomes an Axum route; `novar build --serve` embeds the program in one binary |
| Modules and packages | `compiler/modules`, `tools/package-manager` | `import` between files, `nvpm` with registries, a resolver and a lockfile |
| Editor support | `tools/lsp`, `editors/vscode` | diagnostics, hover, completion, go to definition, inferred-type hints |
| Web playground | `playground/` | the checker and interpreter compiled to WebAssembly |
| Accelerators | `compiler/accel/` | StableHLO and PTX generators (not run on TPU / NVIDIA hardware); GPU execution lives in `runtime` (`gpu` feature) |
| Python | `python/` | PyO3 module; zero-copy Arrow, NumPy and CUDA-array interchange; Ray executor |
| Distribution | `tools/cluster` | `novar worker` + `--workers` for `std::dist::map` |
| Benchmarks | `benchmarks/` | alpha targets: RAG overhead, memory footprint, ingestion throughput |

[`docs/design/status.md`](docs/design/status.md) maps every item of the original engineering
specification to what exists, what was verified and how, and what is still open.

## Language notes

* **Values.** Structs and strings are values; a `Vec` is a reference (copies share elements).
  Tensors are immutable values. Untyped number literals take their type from how they are used.
* **Tensors.** `Tensor<f32, [2, 3]>` carries its shape: `std::tensor::matmul` of mismatched
  shapes is a compile error. Symbolic dimensions (`[N]`) accept any size.
* **Tables.** `Table<Row>` is an Arrow table whose columns are the fields of struct `Row`;
  `t.score` is the column as a zero-copy tensor.
* **Modules.** `import util::helper;`, `import crate::geometry::{Point, norm};`,
  `import mathx::vec::dot;` (a dependency). Only `pub fn` and `pub struct` are importable.
* **Native subset.** Integers, floats, bools, strings, structs, number vectors, static-shape
  `f32`/`f64` tensors, loops and f-strings (ints, bools, strings). Closures, `Table`, SQL and most
  of `std` run in the interpreter only.

## Testing

```sh
cargo test --workspace                       # no LLVM needed
. scripts/llvm-env.sh && cargo test --workspace --features llvm
cd python && maturin develop && python -m pytest tests
scripts/build-playground.sh --node && node --test playground/test
cd editors/vscode && npm ci && npm test
```

## License

Dual-licensed under MIT (`LICENSE-MIT`) or Apache 2.0 (`LICENSE-APACHE`).
