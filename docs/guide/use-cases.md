# Use cases

What Telvax is good for, with a program for each case that you can run from the repository root.
Every program below is in [`examples/`](../../examples) and is run by the commands shown.

| You want to | Look at | Runs with |
|---|---|---|
| Summarise a CSV or Parquet file | [`dataframe/sales.tvx`](../../examples/dataframe/sales.tvx) | `telvax run` |
| Check the shapes of a model before it runs | [`ml/forward.tvx`](../../examples/ml/forward.tvx) | `telvax run` |
| Serve a function over HTTP | [`service/sentiment.tvx`](../../examples/service/sentiment.tvx) | `telvax serve` |
| Build a RAG service | [`rag-service.tvx`](../../examples/rag-service.tvx) | `telvax serve` |
| Process text | [`text/word-count.tvx`](../../examples/text/word-count.tvx) | `telvax run` |
| Handle errors without exceptions | [`errors/parse-config.tvx`](../../examples/errors/parse-config.tvx) | `telvax run` |
| Write plain numeric code | [`basics/primes.tvx`](../../examples/basics/primes.tvx) | `telvax run`, or native with `telvax build` |

## 1. Data analysis on tables

A `Table<Row>` is an Arrow table whose columns are the fields of the struct `Row`. The column names
and types are checked before the program runs, and a column is read as a tensor without copying it.

```telvax
struct Sale { id: i64, region: String, units: i64, price: f64 }

fn main() {
    let sales: Table<Sale> = read_csv("examples/dataframe/sales.csv");
    println(f"mean price: {sales.price.mean()}");

    let by_region: Table<{ region: String, revenue: f64 }> =
        sales.sql("SELECT region, SUM(units * price) AS revenue FROM t GROUP BY region ORDER BY revenue DESC");
    for row in by_region { println(f"{row.region}: {row.revenue}"); }
}
```

```sh
telvax run examples/dataframe/sales.tvx main
```

Use it for reports, feature preparation and quick checks of a data file. A misspelled column
(`sales.prise`) is a compile error (`no column prise on type Table<Sale>`) instead of a failure
halfway through a run. SQL (and Parquet) need the full build; `telvax-lite` keeps tables in memory and reads CSV.

## 2. Models whose shapes are checked

Tensor shapes are part of the type, so a matrix product of mismatched shapes does not compile:

```telvax
fn layer1(x: Tensor<f32, [2, 3]>, w: Tensor<f32, [3, 4]>) -> Tensor<f32, [2, 4]> {
    x.matmul(w).relu()
}
```

```sh
telvax run examples/ml/forward.tvx main
telvax run examples/ml/forward.tvx main --device gpu      # needs a build with --features gpu
telvax run examples/ml/forward.tvx main --quantize int8   # int8 matrix products
```

Change the weights in `layer1` to `Tensor<f32, [4, 4]>` and `telvax check` reports
``matmul inner dimensions differ: 3 and 4`` at the call. Use symbolic dimensions (`Tensor<f32, [N, 3]>`) when a size is only known at run time.
Weights come from `std::tensor::from_vec` here; a real model would load them from a file or a table.

## 3. Serving a function over HTTP

`@endpoint` turns a function into a route. The parameter struct is the JSON body and is validated
for you; `Err(Status::BadRequest(...))` becomes a 400.

```telvax
@endpoint(path = "/v1/sentiment", method = "POST")
pub fn sentiment(input: Input) -> Result<Output, Status> {
    if input.text.is_empty() { return Err(Status::BadRequest("empty text")); }
    ...
}
```

```sh
telvax serve examples/service/sentiment.tvx
curl -XPOST localhost:8080/v1/sentiment -H 'content-type: application/json' -d '{"text":"great and good"}'
# {"label":"positive","score":2}
curl -XPOST localhost:8080/v1/sentiment -H 'content-type: application/json' -d '{"text":""}'
# {"error":"empty text"}   (HTTP 400)
```

To ship it, `telvax build --serve examples/service/sentiment.tvx -o service --dockerfile` produces one
binary with the program embedded, plus a Dockerfile.

## 4. Retrieval-augmented generation

[`examples/rag-service.tvx`](../../examples/rag-service.tvx) embeds a question, searches a vector
store, builds a prompt from the hits and asks a model, as one `@endpoint`. Stages are chained with
`|>`:

```telvax
let answer = prompt |> llm.complete() |> std::text::trim();
```

```sh
telvax serve examples/rag-service.tvx          # offline stand-ins for the model and the database
telvax serve examples/rag-service.tvx --live   # Ollama, OpenAI and PostgreSQL with pgvector
```

The offline mode lets you develop and test the service without any of those running. `--live`
talks to the real ones.

## 5. Text processing

`String`, `Vec` and `Map` have the methods you expect, and `Vec` has iterator methods (`map`,
`filter`, `fold`, `sum`, `enumerate`, `zip`, ...):

```telvax
fn count_words(text: String) -> Map<String, i64> {
    let counts: Map<String, i64> = std::map::new();
    for word in text.to_lowercase().split(" ") {
        let seen = counts.get(word).unwrap_or(0);
        counts.insert(word, seen + 1);
    }
    counts
}
```

```sh
telvax run examples/text/word-count.tvx main
```

## 6. Errors as values

There are no exceptions. A function that can fail returns `Result<T, E>` or `Option<T>`, `match`
must cover every case, and `?` passes an error up to the caller.

```telvax
fn port_of(line: String) -> Result<i64, String> {
    let (key, value) = split_pair(line)?;
    ...
}
```

```sh
telvax run examples/errors/parse-config.tvx main
```

## 7. Plain numeric code, interpreted or native

Integers, floats, loops and structs also work as an ordinary language. The interpreter runs
anything; the native backend (`--features llvm`) compiles the subset described in the
[README](../../README.md#language-notes).

```sh
telvax run examples/basics/primes.tvx main
telvax build examples/basics/primes.tvx -o primes     # native, needs --features llvm
```

## Beyond one file

* **Several files and packages:** `import util::helper;` and `tvpm`, see the [tour](tour.md#modules-and-packages).
* **Many machines:** `std::dist::map` with `telvax worker`, see the [tour](tour.md#distribution).
* **From Python:** the `python/` module exchanges Arrow tables and NumPy arrays without copying.
* **In the browser:** the playground runs the checker and interpreter as WebAssembly.
* **Editor:** `telvax lsp` and the VS Code extension give diagnostics, hover and completion.

## When Telvax is not the right tool yet

It is an alpha. The native backend covers a small subset, PTX and StableHLO output has not been run
on NVIDIA GPUs or TPUs, and syntax can still change. [`docs/design/status.md`](../design/status.md)
lists what was verified and how.
