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

The language server formats documents too, so VS Code can format on save.
