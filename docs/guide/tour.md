# A tour of Novar

## Functions, types and inference

```novar
struct Point { x: f64, y: f64 }

fn norm(p: Point) -> f64 { p.x * p.x + p.y * p.y }

fn main() -> i32 {
    let p = Point { x: 3.0, y: 4.0 };
    let mut total = 0;            // an untyped literal: its type comes from the uses
    for i in 0..10 { total += i; }
    std::io::println(f"norm = {norm(p)}, total = {total}");
    total
}
```

Types: `i8..i64`, `u8..u64`, `f32`, `f64`, `bool`, `String`, `Vec<T>`, `Result<T, E>`, structs,
`Tensor<T, [dims]>` and `Table<Row>`. `let mut` and `mut` parameters allow assignment, including
`v[i] = x` and `p.a.b = x`. Control flow: `if`/`else`, `while`, `for i in a..b`, `for x in vec`,
`return`.

## Pipelines

`x |> f(a)` means `f(x, a)`, so stages read left to right:

```novar
let answer = prompt |> llm.complete() |> std::text::trim();
```

## Tensors and tables

```novar
fn forward(x: Tensor<f32, [2, 3]>, w: Tensor<f32, [3, 2]>) -> Tensor<f32, [2, 2]> {
    x |> std::tensor::matmul(w) |> std::tensor::relu()     // [2,3] x [3,2] -> [2,2], checked
}

struct Row { id: i64, score: f32 }
fn best(t: Table<Row>) -> Table<Row> {
    t |> std::table::sql("SELECT * FROM t WHERE score > 0.5 ORDER BY score DESC")
}
fn mean_score(t: Table<Row>) -> f32 { std::tensor::mean(t.score) }   // zero-copy column
```

`std::table::read_csv(path)` and `read_parquet` need the result type to be written down
(`let t: Table<Row> = ...`), as do `std::tensor::zeros()`, `from_vec` and `reshape`.

## HTTP endpoints

```novar
@endpoint(path = "/v1/score", method = "POST")
pub fn score(input: In) -> Result<Out, Status> { ... }
```

The parameter struct becomes the validated JSON body (GET/DELETE read it from the query string).
Returning `Err(Status::BadRequest("..."))` becomes an HTTP 400.

## Modules and packages

```novar
import util::clamp;            // ./util.nv
import mathx::vec::dot;        // dependency `mathx`, file vec.nv
```

```sh
nvpm new app && cd app
nvpm add mathx --registry ~/registry     # or --path ../mathx
nvpm install --registry ~/registry       # resolves, writes novar.lock, downloads
nvpm run                                  # runs src/main.nv with the dependencies
```

## Distribution

```novar
fn work(item: Job) -> Done { ... }
fn all(jobs: Vec<Job>) -> Vec<Done> { std::dist::map(jobs, work) }
```

`std::dist::map` takes the *name* of a top-level function. It runs on local threads by default;
`novar run app.nv all --workers host1:7077,host2:7077` sends the calls to machines running
`novar worker`, and `novar.ray.RayExecutor` (Python) sends them to Ray.

## Quantization

`--quantize int8` (on `run`, `serve`, `jit`, `build`) makes `f32` matrix products use symmetric
per-tensor int8 arithmetic with `i32` accumulation. The interpreter and the native backend give
identical results.

`--quantize int8-channel` and `--quantize int4-channel` (`run` and `serve`; the native backend
refuses them) give every row of the activations and every column of the weights its own scale, so
a column that is much larger or smaller than the rest no longer loses its precision. The int4 mode
stores the weights in 4 bits. Scales are computed on each call; there is no calibration step.

## Accelerators

* `--device gpu` (on `run` and `serve`, needs `cargo build --features gpu`) runs `f32` tensor operations on the GPU through `wgpu`: Metal on macOS, Vulkan on Linux, DX12 on Windows.
* `novar build file.nv --emit stablehlo` prints StableHLO for the tensor functions in the file, for XLA / PJRT.
* `novar build file.nv --emit ptx [--sm sm_80]` prints PTX kernels for the tensor operations the file uses (needs `--features llvm`).

PTX and StableHLO are generated and checked, not executed: see `docs/design/status.md`.

## Enums, `match`, `break` and `continue`

```novar
enum Shape { Circle(f64), Rect(f64, f64), Empty }

fn area(s: Shape) -> f64 {
    match s {
        Shape::Circle(r) => r * r * 3.14159,
        Shape::Rect(w, h) => w * h,
        Shape::Empty => 0.0,
    }
}

fn first_over(limit: i32) -> i32 {
    let mut found = -1;
    for i in 1..1000 {
        if i * i > limit { found = i; break; }
    }
    found
}
```

`match` must cover every case (the compiler names the missing variants) and also works on
`Result` (`Ok(v)` / `Err(e)`), integers, strings, booleans and `_`. Enums travel as JSON: a unit
variant is its name (`"Empty"`), one with values is `{"Circle": 2.0}` or `{"Rect": [2.0, 3.0]}`.
Enums run in the interpreter; the native backend does not compile them yet.

## Formatting and linting

```sh
novar fmt file.nv            # rewrite in place
novar fmt --check file.nv    # list files that would change (exit 1), for CI
novar lint file.nv           # unused variables, needless `mut`, unused private functions
```

The language server formats documents too, so VS Code can format on save. It also finds all
references to a variable or function and renames them (a rename is refused when the new name
would collide with another one in scope, or when the symbol is a struct or lives in another file).

## Strings and vectors

```novar
fn demo(s: String) -> u64 {
    let mut words = s.split(" ");           // Vec<String>
    words.push("extra");
    if s.contains("novar") && !s.starts_with("//") { words.len() } else { 0 }
}
```

String: `len`, `is_empty`, `trim`, `to_lowercase`, `to_uppercase`, `contains`, `starts_with`,
`ends_with`, `replace`, `split`. Vec: `len`, `is_empty`, `push`, `contains`. A method that does not
exist is a compile error with a "did you mean" hint. These methods run in the interpreter; the
native backend still only has `len` and `is_empty`.

## Option, Map and closures

```novar
fn demo(text: String) -> i64 {
    let counts: Map<String, i64> = std::map::new();
    for word in text.split(" ") {
        let seen = counts.get(word).unwrap_or(0);   // Option<i64>
        counts.insert(word, seen + 1);
    }
    match counts.get("novar") {
        Some(n) => n,
        None => 0,
    }
}
```

* `Option<T>` is `Some(value)` or `None`; `match` must cover both. Methods: `is_some`, `is_none`,
  `unwrap` (an error on `None`), `unwrap_or`. Several methods return one: `Vec::first`, `last`,
  `pop`, `get`, `String::find`, `to_int`, `to_float`, `Map::get`, `remove`. As JSON, `None` is
  `null` and `Some(x)` is just `x`.
* `Map<K, V>` has `String`, integer or `bool` keys. It is created with `std::map::new()` and a
  type annotation, and has `insert`, `get`, `remove`, `contains_key`, `len`, `is_empty`, `keys` and
  `values`. Like `Vec`, copies share the same entries. Keys are kept in order; JSON objects use
  string keys.
* `Vec` also has `sort` (numbers, strings, bools) and `reverse`; `String` has `chars` and `lines`.
* Closure parameters can be annotated (`|x: i64| x + 1`). A closure captures variables by value,
  so assigning to a variable from outside is a compile error; shared `Vec` and `Map` values can
  still be changed from inside one.
* These run in the interpreter; the native backend does not compile them yet.

## Tables and tensors, the short way

```novar
struct Sale { id: i64, region: String, units: i64, price: f64 }

fn main() {
    let sales: Table<Sale> = read_csv("sales.csv");
    println(f"mean price: {sales.price.mean()}");

    let by_region: Table<{ region: String, revenue: f64 }> =
        sales.sql("SELECT region, SUM(units * price) AS revenue FROM t GROUP BY region");
    for row in by_region { println(f"{row.region}: {row.revenue}"); }
}
```

* `println`, `print`, `read_csv` and `read_parquet` need no `std::` prefix.
* Methods on tensors are the `std::tensor` functions: `x.matmul(w).relu().softmax()`,
  `sales.price.mean()`. A table has `sql`. They are the same calls underneath, so
  `std::tensor::relu(x)` still works.
* Operators on tensors: `a + b`, `a - b`, `a * b`, `a / b` (same shape) and `a * k`, `k * a` to
  scale by a number.
* `Table<{ region: String, revenue: f64 }>` writes the row type in place; no throwaway `struct` for
  the result of a query. The same columns mean the same type everywhere.
* `fn main()` may return nothing.

See `examples/dataframe/`. The pandas version of the same program is about as long, but the
Novar one checks the column names and types before it runs.

## Errors, tuples and iterators

```novar
fn split_pair(line: String) -> Result<(String, String), String> {
    let parts = line.split("=");
    if parts.len() != 2 { return Err(f"bad line: {line}"); }
    Ok((parts[0].trim(), parts[1].trim()))
}

fn port_of(line: String) -> Result<i64, String> {
    let (key, value) = split_pair(line)?;          // `?` returns the Err early
    match value.to_int() { Some(n) => Ok(n), None => Err(f"not a number: {value}") }
}

fn main() {
    let xs = [3, 1, 4, 1, 5];
    let big = xs.filter(|x| x > 2).map(|x| x * 10);     // [30, 40, 50]
    for (i, x) in xs.enumerate() { println(f"{i}: {x}"); }
    let mut stack = [1, 2, 3];
    while let Some(top) = stack.pop() { println(f"{top}"); }
    if let Some(first) = xs.first() { println(f"first = {first}"); }
}
```

* `expr?` on a `Result` or `Option` returns the error (or `None`) from the current function; the
  function's return type must be a `Result` or `Option` too, and the error types must match.
* `if let` and `while let` match one pattern; they work with `break` and `continue`.
* Tuples: the type `(A, B)`, values `(1, "a")`, fields `.0` and `.1`, and destructuring in `let`,
  `for` and closure parameters.
* `Vec` methods `map`, `filter`, `any`, `all`, `find`, `enumerate`, `zip`, `sum`, `min`, `max`,
  `fold`, `for_each`, `skip` and `join`, and `Map::entries`. `min`, `max` and `find` return an
  `Option`. These run in the interpreter; the native backend does not compile them yet.

See [use cases](use-cases.md) for complete programs.
