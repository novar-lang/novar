# Changelog

## Unreleased

* **The `|>` operator follows one rule:** `x |> stage` is the call `stage` with `x` as its first argument.
  Names and defaults now count that argument (`1 |> scale(k: 3)` and `1 |> scale` used to fail with
  "missing the argument" and "expected 2 argument(s)"), and a suffix after the stage applies to the
  result of the call (`s |> parse()?`, `n |> make().field`, `n |> items()[0]`; the first two were errors).
  `|>` binds looser than every operator, and the error for `x |> f == 1` says to write `(x |> f()) == 1`.
* **A generic that grows forever is an error:** `fn f[T](x: T): f([x])` is instantiated for `T`, then
  `List[T]`, then `List[List[T]]`, and so on; the checker never finished and used all the memory. A type
  of more than 200 parts in an instance is now an error that says the types keep growing.
* **Deeply nested code no longer hangs or crashes the compiler:** parentheses inside parentheses were
  read twice per level, so twenty levels took seconds and forty never finished; they now take linear
  time. Blocks nested past 64 levels and brackets past 200 are an error that says so, and the parser and
  the checker run on a stack of their own, so a long sum or calls inside calls are checked instead of
  overflowing the stack and ending the process.
* **Methods with type parameters of their own:** `fn map[U](self, f: (T) -> U) -> Stack[U]` in a generic
  or a plain `type`, or in an `extend`. The parameters are inferred from the call and may have limits
  (`[N: Num]`). Before, they were an error ("not supported yet").
* **Type arguments in a call:** `empty[Text]()`, `pair[Int, Text](1, "a")`. It makes a generic function
  that cannot infer its types from the arguments usable, and the "cannot infer" error now says so.
* **Built-in limits for type parameters:** `T: Num`, `T: Ord` and `T: Eq` (and `T: (Num, Ord)`) without
  declaring a trait. A call with a type that does not meet a limit is reported once, at the call, instead
  of again inside the generic function's body.
* **Function types:** `(Int, Int) -> Int` is a type, so a function can be a parameter, a field, a list
  element or a return value (`fn adder(n: Int) -> (Int) -> Int: x => x + n`). `() -> ()` stands for no
  value. A field that holds a function can be called (`handler.run(5)`), and a closure parameter can
  carry its type: `(x: Int) => x + 1`.
* **Imports can be renamed:** `from util import clamp as limit`, `import util.clamp as limit`. Importing
  a module as a whole (`import util`) is an error that says to write `from util import name`, and the
  messages for the modules of your own project no longer say `crate`.
* **Syntax errors say what to write:** a `{ }` body, a parameter without a type, `: Type` for a return
  type, and `use`, `const` and `function` now get the same hint as `def` and `pub`. The extra
  "unexpected `def`" that came before the hint is gone.
* **Methods on `Result` and `Option`:** `Result` has `is_ok`, `is_err`, `unwrap`, `expect`,
  `unwrap_or`, `unwrap_or_else`, `map`, `map_err`, `and_then`, `ok` and `err`; `Option` gains `map`,
  `and_then`, `filter`, `ok_or`, `unwrap_or_else` and `expect`. They are checked: a method that does not
  exist on a `Result` is an error (it used to pass without a type).
* **Inferred return type:** a top-level function without `->` returns the type of its last expression
  (`fn double(x: Int): x * 2` returns an `Int`). `main`, tests, endpoints, generic functions and
  methods still need the `->`.
* **Generic calls and literals:** `let a = id(3)` no longer fixes `a` to `i32` at the call; a later use
  such as `let n: Int = a` decides the type, and the default applies only if nothing does.

## 0.1.0

**Breaking: a new syntax.** Telvax is now easy to read for anyone who knows Python (blocks by
indentation, no braces or semicolons) and keeps its own words and ideas. The Rust-style syntax is gone.
The design is in `docs/design/sintaxe-pythonica.md`; the checker, the interpreter and the native
backend are unchanged except where noted.

```telvax
type Query:
    text: Text
    top_k: Int = 5

fn embed(text: Text) -> Tensor[f32, 768]:
    text
        |> std.text.clean_whitespace()
        |> std.text.truncate(512)
        |> embedder.embed()

@endpoint(POST, "/v1/rag/query")
fn handle(q: Query) -> Result[Reply, Status]:
    let docs = search(embed(q.text), limit: q.top_k)
    ...
```

### Language
* **Blocks:** a line that ends with `:` opens an indented block; `#` starts a comment.
* **Functions and variables:** `fn name(a: Int) -> Int:`; the last expression is the value, `return`
  leaves early. `let x = 1` cannot change, `var x = 1` can; a plain `x = 2` never declares.
* **Types:** `Int`, `Float`, `Text`, `Bool`, `List[T]`, `Map[K, V]`, `T?` for a value that may be `None`,
  `Result[T, E]`, `Tensor[f32, 2, 3]` (the dimensions follow the element type) and `Table[Row]`; sized
  numbers keep `i8`..`i64`, `u8`..`u64`, `f32`, `f64`. Error messages use the same names.
* **Your own types:** `type Point:` with fields and methods in one block, `type Circle(Shape)` to implement
  a trait, `extend Point:`, `enum Shape:` with one variant per line, `trait Shape:` and `type Stack[T]:`.
* **Named arguments and defaults:** `Query(text: "x")`, `search(q, limit: 3)`; a field or a parameter can
  have a default (`top_k: Int = 5`). Names are checked against the function.
* **Functions as values:** `x => x + 1`, `(a, b) => a + b`; an `fn` inside a function is a closure with
  several statements.
* **Words and operators:** `and`, `or`, `not`, `in`, `elif`, `True` / `False` / `None`, `a if c else b`,
  `for i in range(n)`, `for i, x in items`, `pass`. `x is Some(v)` replaces `if let` and `while let`.
* **Modules:** `.` replaces `::`; `from util import clamp`, `import util.clamp`. A name that starts with `_`
  is private; there is no `pub`.
* `@endpoint(POST, "/path")`; `print` ends the line (`std.io.print` does not); values print as
  `True`, `None`, `Point(x=1, y=2)`.
* Kept: `|>` (a line may start with it), `?`, `Some`, `Ok`, `Err`.

### Tools
* `telvax fmt` is rebuilt for indentation and checks that the tokens are unchanged.
* `telvax lint` no longer reports "needless `mut`"; an unused private function is one whose name starts
  with `_`.
* The language server, the VS Code grammar and the playground follow the new syntax.
* Syntax errors say what to write: `def` -> `fn`, a missing `:`, `;`, `..`, `x = 1` without `let`.
* `telvax-migrate` (`tools/migrate`) converts files written in the old Rust-style syntax and keeps their
  comments: `cargo run -p telvax_migrate -- file.tvx`. It will be removed in a later release.

### Moving a program
Run `telvax-migrate` on each file. What it cannot carry over: an `if` or `match` used as a value (write a
conditional expression, or assign inside the branches), blocks used as expressions, closed ranges
(`0..=n` is `range(0, n + 1)`), and typed closure parameters.

## 0.0.5

The project is now called **Telvax** (it was Novar). Everything was renamed: the binaries are `telvax`
and `tvpm`, source files end in `.tvx`, environment variables start with `TELVAX_`, the manifest is
`telvax.toml`, and the repository is `telvax/telvax`. Releases before this one keep the old names.

### Language
* **Errors and control flow:** `expr?` passes the error of a `Result` (or the `None` of an `Option`) up
  to the caller, and `if let` / `while let` match a single pattern.
* **Tuples:** the type `(A, B)`, values, `.0` / `.1`, patterns, and destructuring in `let`, `for` and
  closure parameters.
* **Iterator methods on `Vec`:** `map`, `filter`, `any`, `all`, `find`, `enumerate`, `zip`, `sum`, `min`,
  `max`, `fold`, `for_each`, `skip`, `join`; `Map::entries`.
* **`impl` blocks, traits and generics:** methods on structs and enums, `trait` with
  `impl Trait for Type`, trait-typed values dispatched at run time, and generic functions, structs,
  enums and impls with trait bounds.
* The new syntax runs in the interpreter; the native backend does not compile it yet.

### Docs and examples
* A use-cases guide (`docs/guide/use-cases.md`) and four new runnable examples: word count, a
  shape-checked network, an HTTP service that needs no model, and error handling with `?`.

### Fixes and security
* `rustls` 0.23.46 (RUSTSEC-2026-0349).
* The table tests no longer share a scratch file, and the bundle test waits for the server to answer.

### Releases
* Every pull request merged into `main` is now released automatically with the next patch version,
  so no file needs editing. The VS Code extension is packaged and attached to each release, and
  published to the Marketplace and Open VSX when their tokens are configured.

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
  collide with another name is refused). `telvax fmt` indents the line after a trailing `=`.
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
* Two builds per system: the full `telvax` (about 51 MB, down from 58 with full LTO) and a new
  `telvax-lite` (about 10 MB) without SQL and Parquet (tables in memory and CSV still work). `TELVAX_LITE=1` makes the install scripts
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
* `telvax fmt [--check]`: a token-based formatter that keeps comments and your line breaks. The
  language server formats documents too.
* `telvax lint [--deny-warnings]`: unused variables, needless `mut`, unused private functions.
* `telvax --version` and `--help`.

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
* `telvax check | run | serve | jit | build | lsp | worker`.
* Interpreter on Arrow: tensors, tables, SQL (DataFusion), CSV and Parquet.
* `@endpoint` functions served over HTTP, with JSON validation; `build --serve` bundles the
  service into one executable and can write a Dockerfile.
* `--live` talks to real Ollama, OpenAI-compatible APIs and PostgreSQL + pgvector.
* Native backend (`--features llvm`, LLVM 22): ints, floats, strings, structs, static-shape `f32`
  tensors, int8 quantized `matmul`.
* `--device gpu` (`--features gpu`): `f32` tensor operations on the GPU through wgpu.
* `--emit ptx` and `--emit stablehlo` for NVIDIA and TPU toolchains (generated, **not executed on
  that hardware**).
* `tvpm` package manager, language server, VS Code extension, Wasm playground, Python bindings
  (zero-copy NumPy, Arrow, Polars, PyTorch), Ray executor, cluster workers.

### Known limits
See [`docs/design/status.md`](docs/design/status.md). Notably: native executables do not serve HTTP
yet, PTX / StableHLO / CUDA-array interchange have not run on real NVIDIA or TPU hardware, the
Dockerfile was never built, and there are no enums, `break` or `continue`.
