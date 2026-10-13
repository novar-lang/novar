# Benchmarks

Alpha 0.1.0 targets and how to measure them. Always use release builds.

```sh
cargo build --release -p novar_cli -p novar_bench
./target/release/rag_latency target/release/novar --docs 2000   # add --check to fail when a target is missed
./target/release/ingest --rows 1000000
```

## RAG serving overhead and footprint

`rag_latency` bundles `examples/rag-service.nv` with `novar build --serve`, starts it, and times
`POST /v1/rag/query` over keep-alive connections. The model and the vector database are the
offline stand-ins, so the numbers are Novar's own overhead (HTTP, validation, interpreter,
retrieval over the corpus, JSON). A live Ollama / OpenAI / PostgreSQL adds their latency on top.

| Alpha target | Result (2000 documents) |
|---|---|
| p99 overhead < 15 ms | 0.83 ms with 1 connection, 5.9 ms with 8 concurrent connections |
| Base RSS < 30 MB | 17.5 MB idle, 19.2 MB after 7,000 requests |

Retrieval scans the in-memory store linearly, so latency grows with the corpus: ~0.1 ms for 3
documents, ~0.8 ms for 2,000.

## Ingestion and stream processing

`ingest` runs 1,000,000 rows (`id`, `user`, `amount`, `label`) through the Arrow-backed tables:

| Step | Throughput |
|---|---|
| CSV -> Arrow table | 19.7 M rows/s (434 MB/s) |
| Arrow table -> Parquet | 27 M rows/s |
| Parquet -> Arrow table | 88 M rows/s |
| Column read + `mean` (zero-copy) | > 1 G rows/s |
| SQL filter (DataFusion) | 120 M rows/s |
| SQL group-by (DataFusion) | 151 M rows/s |
| Vector ingestion (embed + upsert, 768-d, offline embedder) | 84 K docs/s |
| `matmul` 256x256 f32 in the interpreter | 15.7 GFLOP/s |

Measured on an Apple M-series laptop (10 cores, 16 GB). Re-run on your hardware before quoting.
