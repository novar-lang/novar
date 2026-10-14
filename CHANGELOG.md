# Changelog

## 0.0.4

### Language
* **Friendlier syntax** (all additive): `println`, `print`, `read_csv` and `read_parquet` work
  without `std::`; tensors have methods (`x.matmul(w).relu().softmax()`, `sales.price.mean()`) and
  operators (`a + b`, `a * 2.0`); tables have `.sql(...)`; a row type can be written in place
  (`Table<{ region: String, revenue: f64 }>`); `fn main()` may return nothing.
* `Option<T>` (`Some`, `None`, `match`, `unwrap`, `unwrap_or`, ...) and `Map<K, V>` (`std::map::new()`,
  `insert`, `get`, `remove`, `contains_key`, `keys`, `values`, ...), also across the Python and JSON
  boundaries. `Vec` gains `first`, `last`, `pop`, `get`, `sort`, `reverse`; `String` gains `find`,
  `to_int`, `to_float`, `chars`, `lines`.
* Closure parameters can be annotated (`|x: i64| x + 1`), and assigning to a captured variable is a
  compile error (closures capture by value).

### Tools
* The language server finds references and renames variables and functions (a rename that would
  collide with another name is refused). `novar fmt` indents the line after a trailing `=`.
* Quantization: `--quantize int8-channel` and `--quantize int4-channel` (interpreter only).

### GPU
* Nested tensor operations (`softmax(matmul(relu(matmul(x, w1)), w2))`) keep their intermediate
  results on the GPU. On the M4 a 3-layer network takes 44 ms against 9.0 s in the CPU interpreter;
  the chaining itself saves little there because memory is shared.

### Verification
* The playground page is tested in a real Chromium (Playwright), the extension inside a real VS
  Code, and every published release is installed and run on Linux, macOS and Windows by the release
  workflow. The LLVM job is blocking.
* CI also runs a build of the ~10 MB `lite` binary.


## 0.0.3

### Size
* Two builds per system: the full `novar` (about 51 MB, down from 58 with full LTO) and a new
  `novar-lite` (about 10 MB) without SQL and Parquet (tables in memory and CSV still work). `NOVAR_LITE=1` makes the install scripts
  pick it. Build it yourself with `--no-default-features --features live`.

### Language and errors
* `String`: `contains`, `starts_with`, `ends_with`, `replace`, `split`. `Vec`: `push`, `contains`.
* A method that does not exist on a built-in type is a compile error (it used to fail only when
  run), and misspelled names, types, fields and methods get a "did you mean ...?" note.
* A missing `;` is reported right after the previous token instead of listing every token that
  could follow.

### Dependencies
* `wgpu` 30 (tested on a real GPU), `lsp-server` 0.10, `vscode-languageclient` 10 and TypeScript 7
  (the extension now needs VS Code 1.91), newer GitHub Actions and maturin.

### Fixes
* The Dockerfile written by `build --serve` did not create `/out`; it builds now (CI builds it).


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
