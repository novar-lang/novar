# Benchmarks

Alpha 0.1.0 targets and how to measure them. Always use release builds.

```sh
cargo build --release -p telvax_cli -p telvax_bench
./target/release/rag_latency target/release/telvax --docs 2000   # add --check to fail when a target is missed
./target/release/ingest --rows 1000000
```

## In CI

The `benchmark` job runs both programs on every pull request and fails on a regression:
`rag_latency --check` (p99 under 15 ms for one connection, under 45 ms for 8 concurrent ones, which
also queue behind each other on a 2-core runner, and RSS under 30 MB) and `benchmarks/speed.sh`
(the interpreter must run `fib(27)` in under 2 s and a 3,000,000-iteration loop in under 6 s;
a laptop does them in about 0.3 s and 0.65 s, a CI runner in 0.2 s and 1.5 s). Coverage has its own
job and a floor in `scripts/coverage-floor.txt`.

## RAG serving overhead and footprint

`rag_latency` bundles `examples/rag-service.tvx` with `telvax build --serve`, starts it, and times
`POST /v1/rag/query` over keep-alive connections. The model and the vector database are the
offline stand-ins, so the numbers are Telvax's own overhead (HTTP, validation, interpreter,
retrieval over the corpus, JSON). A live Ollama / OpenAI / PostgreSQL adds their latency on top.

| Alpha target | Result (2000 documents) |
|---|---|
| p99 overhead < 15 ms | 0.88 ms with 1 connection, 6.0 ms with 8 concurrent connections |
| Base RSS < 30 MB | 17.8 MB idle, 19.5 MB after 7,000 requests |

### With the real services

The same endpoint (`examples/rag-service.tvx --live`) against a real Ollama (`nomic-embed-text`),
a real PostgreSQL 17 with pgvector (500 documents, 768-d) and a stand-in OpenAI server, called
200 times in sequence from Python on one Apple M4 laptop. There is no API key here, so the LLM is
a local stub that answers in 0.2 ms; its cost is subtracted.

| What | mean | p99 |
|---|---|---|
| `POST /v1/rag/query` end to end | 20.4 ms | 28.8 ms |
| Ollama embedding alone (same queries) | 17.3 ms | 22.3 ms |
| pgvector query (server-side `EXPLAIN ANALYZE`) | 1.0 ms | 1.1 ms |
| LLM stub alone | 0.18 ms | 0.32 ms |
| **Telvax overhead** (end to end minus the three above) | **1.9 ms** | ~6 ms |

RSS of the serving process was 14.2 MB. The p99 overhead is noisy (it subtracts p99s measured in
separate runs), so read it as "a few milliseconds", not a precise figure. One machine, sequential
requests, no real LLM: this shows Telvax adds little on top of the services, not what a production
deployment would see.

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
