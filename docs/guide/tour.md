# A tour of Telvax

Telvax reads like Python: blocks are made by indentation, a block opens with `:`, and there are no
braces or semicolons. What you cannot write in Python is checked before the program runs: every
name has a type, written or inferred.

## Functions, types and inference

```telvax
class Point:
    x: float
    y: float

def norm(p: Point) -> float:
    return p.x * p.x + p.y * p.y

def main() -> int32:
    p = Point(x=3.0, y=4.0)
    total = 0                    # an untyped literal: its type comes from the uses
    for i in range(10):
        total += i
    print(f"norm = {norm(p)}, total = {total}")
    return total
```

* `x = value` creates a variable the first time and changes it afterwards. `x: int = 1` writes the
  type down. A variable lives until the end of the block that created it.
* Types: `int` (64 bits), `float` (64 bits), `bool`, `str`, `list[T]`, `dict[K, V]`, `T | None`,
  `Result[T, E]`, your classes, `Tensor[T, [dims]]` and `Table[Row]`. When the size matters:
  `int8`, `int16`, `int32`, `uint8` to `uint64`, `float32`, `float64`.
* A name that starts with `_` is private to its file; everything else can be imported.
* Control flow: `if` / `elif` / `else`, `while`, `for x in items`, `for i in range(n)`,
  `for i in range(a, b)`, `break`, `continue`, `pass` and `return`. `a if cond else b` is an
  expression. `and`, `or`, `not`, `in` and `not in` work as in Python.
* Comments start with `#`. Strings use `"` or `'`, and `f"..."` interpolates `{expr}`.

## Pipelines

`x |> f(a)` means `f(x, a)`, so stages read left to right. A line that starts with `|>` continues
the line before it:

```telvax
answer = prompt |> llm.complete() |> std.text.trim()

docs = query
    |> embed()
    |> store.search(3)
```

## Tensors and tables

```telvax
def forward(x: Tensor[float32, [2, 3]], w: Tensor[float32, [3, 2]]) -> Tensor[float32, [2, 2]]:
    return x |> std.tensor.matmul(w) |> std.tensor.relu()   # [2,3] x [3,2] -> [2,2], checked

class Row:
    id: int
    score: float32

def best(t: Table[Row]) -> Table[Row]:
    return t |> std.table.sql("SELECT * FROM t WHERE score > 0.5 ORDER BY score DESC")

def mean_score(t: Table[Row]) -> float32:
    return std.tensor.mean(t.score)       # a zero-copy column
```

`std.table.read_csv(path)` and `read_parquet` need the result type to be written down
(`t: Table[Row] = ...`), as do `std.tensor.zeros()`, `from_vec` and `reshape`.

## HTTP endpoints

```telvax
@endpoint(path="/v1/score", method="POST")
def score(input: In) -> Result[Out, Status]:
    ...
```

The parameter class becomes the validated JSON body (GET/DELETE read it from the query string).
Returning `Err(Status.BadRequest("..."))` becomes an HTTP 400.

## Modules and packages

```telvax
from util import clamp            # ./util.tvx
from mathx.vec import dot         # dependency `mathx`, file vec.tvx
```

`import util.clamp` brings in the same name. Names from `std` need no import: `std.text.trim(s)`.

```sh
tvpm new app && cd app
tvpm add mathx --registry ~/registry     # or --path ../mathx
tvpm install --registry ~/registry       # resolves, writes telvax.lock, downloads
tvpm run                                  # runs src/main.tvx with the dependencies
```

## Distribution

```telvax
def work(item: Job) -> Done:
    ...

def run_all(jobs: list[Job]) -> list[Done]:
    return std.dist.map(jobs, work)
```

`std.dist.map` takes the *name* of a top-level function. It runs on local threads by default;
`telvax run app.tvx run_all --workers host1:7077,host2:7077` sends the calls to machines running
`telvax worker`, and `telvax.ray.RayExecutor` (Python) sends them to Ray.

## Quantization

`--quantize int8` (on `run`, `serve`, `jit`, `build`) makes `float32` matrix products use symmetric
per-tensor int8 arithmetic with `int32` accumulation. The interpreter and the native backend give
identical results.

`--quantize int8-channel` and `--quantize int4-channel` (`run` and `serve`; the native backend
refuses them) give every row of the activations and every column of the weights its own scale, so
a column that is much larger or smaller than the rest no longer loses its precision. The int4 mode
stores the weights in 4 bits. Scales are computed on each call; there is no calibration step.

## Accelerators

* `--device gpu` (on `run` and `serve`, needs `cargo build --features gpu`) runs `float32` tensor operations on the GPU through `wgpu`: Metal on macOS, Vulkan on Linux, DX12 on Windows.
* `telvax build file.tvx --emit stablehlo` prints StableHLO for the tensor functions in the file, for XLA / PJRT.
* `telvax build file.tvx --emit ptx [--sm sm_80]` prints PTX kernels for the tensor operations the file uses (needs `--features llvm`).

PTX and StableHLO are generated and checked, not executed: see `docs/design/status.md`.

## Enums, `match`, `break` and `continue`

```telvax
enum Shape:
    Circle(float)
    Rect(float, float)
    Empty

def area(s: Shape) -> float:
    match s:
        case Shape.Circle(r):
            return r * r * 3.14159
        case Shape.Rect(w, h):
            return w * h
        case Shape.Empty:
            return 0.0

def first_over(limit: int32) -> int32:
    found = -1
    for i in range(1, 1000):
        if i * i > limit:
            found = i
            break
    return found
```

`match` must cover every case (the compiler names the missing variants) and also works on
`Result` (`Ok(v)` / `Err(e)`), integers, strings, booleans and `_`. Enums travel as JSON: a unit
variant is its name (`"Empty"`), one with values is `{"Circle": 2.0}` or `{"Rect": [2.0, 3.0]}`.
Enums run in the interpreter; the native backend does not compile them yet.

## Formatting and linting

```sh
telvax fmt file.tvx            # rewrite in place
telvax fmt --check file.tvx    # list files that would change (exit 1), for CI
telvax lint file.tvx           # unused variables and unused private functions
```

Indentation is part of the syntax, so `telvax fmt` never moves a line between blocks: it sets four
spaces per level, spaces inside a line, continuation lines and blank lines.

The language server formats documents too, so VS Code can format on save. It also finds all
references to a variable or function and renames them (a rename is refused when the new name
would collide with another one in scope, or when the symbol is a class or lives in another file).

## Strings and lists

```telvax
def demo(s: str) -> uint64:
    words = s.split(" ")                       # list[str]
    words.push("extra")
    if s.contains("telvax") and not s.starts_with("#"):
        return words.len()
    return 0
```

`str`: `len`, `is_empty`, `trim`, `to_lowercase`, `to_uppercase`, `contains`, `starts_with`,
`ends_with`, `replace`, `split`. `list`: `len`, `is_empty`, `push`, `contains`. `x in xs` is
`xs.contains(x)`. A method that does not exist is a compile error with a "did you mean" hint.
These methods run in the interpreter; the native backend still only has `len` and `is_empty`.

## Option, dict and local functions

```telvax
def demo(text: str) -> int:
    counts: dict[str, int] = std.map.new()
    for word in text.split(" "):
        seen = counts.get(word).unwrap_or(0)       # int | None
        counts.insert(word, seen + 1)
    match counts.get("telvax"):
        case Some(n):
            return n
        case None:
            return 0
```

* `T | None` is `Some(value)` or `None`; `match` must cover both. Methods: `is_some`, `is_none`,
  `unwrap` (an error on `None`), `unwrap_or`. Several methods return one: `list.first`, `last`,
  `pop`, `get`, `str.find`, `to_int`, `to_float`, `dict.get`, `remove`. As JSON, `None` is
  `null` and `Some(x)` is just `x`.
* `dict[K, V]` has `str`, integer or `bool` keys. It is created with `std.map.new()` and a
  type annotation, and has `insert`, `get`, `remove`, `contains_key`, `len`, `is_empty`, `keys` and
  `values`. Like `list`, copies share the same entries. Keys are kept in order; JSON objects use
  string keys.
* `list` also has `sort` (numbers, strings, bools) and `reverse`; `str` has `chars` and `lines`.
* `lambda x: x + 1` is a one-line function. For more, write a `def` inside the function:
  `def add(a: int): return a + k`. Both capture variables by value, so assigning to a variable
  from outside is a compile error; shared `list` and `dict` values can still be changed from
  inside one.
* These run in the interpreter; the native backend does not compile them yet.

## Tables and tensors, the short way

```telvax
class Sale:
    id: int
    region: str
    units: int
    price: float

def main():
    sales: Table[Sale] = read_csv("sales.csv")
    print(f"mean price: {sales.price.mean()}")

    by_region: Table[{region: str, revenue: float}] = sales.sql(
        "SELECT region, SUM(units * price) AS revenue FROM t GROUP BY region")
    for row in by_region:
        print(f"{row.region}: {row.revenue}")
```

* `print`, `read_csv` and `read_parquet` need no `std.` prefix. `print` ends the line;
  `std.io.print` writes without one.
* Methods on tensors are the `std.tensor` functions: `x.matmul(w).relu().softmax()`,
  `sales.price.mean()`. A table has `sql`. They are the same calls underneath, so
  `std.tensor.relu(x)` still works.
* Operators on tensors: `a + b`, `a - b`, `a * b`, `a / b` (same shape) and `a * k`, `k * a` to
  scale by a number.
* `Table[{region: str, revenue: float}]` writes the row type in place; no throwaway class for
  the result of a query. The same columns mean the same type everywhere.
* `def main():` may return nothing.

See `examples/dataframe/`. The pandas version of the same program is about as long, but the
Telvax one checks the column names and types before it runs.

## Errors, tuples and iterators

```telvax
def split_pair(line: str) -> Result[(str, str), str]:
    parts = line.split("=")
    if parts.len() != 2:
        return Err(f"bad line: {line}")
    return Ok((parts[0].trim(), parts[1].trim()))

def port_of(line: str) -> Result[int, str]:
    key, value = split_pair(line)?             # `?` returns the Err early
    match value.to_int():
        case Some(n):
            return Ok(n)
        case None:
            return Err(f"not a number: {value}")

def main():
    xs = [3, 1, 4, 1, 5]
    big = xs.filter(lambda x: x > 2).map(lambda x: x * 10)     # [30, 40, 50]
    for i, x in xs.enumerate():
        print(f"{i}: {x}")
    stack = [1, 2, 3]
    while stack.pop() is Some(top):
        print(f"{top}")
    if xs.first() is Some(first):
        print(f"first = {first}")
```

* `expr?` on a `Result` or `Option` returns the error (or `None`) from the current function; the
  function's return type must be a `Result` or `T | None` too, and the error types must match.
* `if x is Pattern:` and `while x is Pattern:` match one pattern and bind its names; they work
  with `break` and `continue`, and `else:` runs when the pattern does not match.
* Tuples: the type `(A, B)`, values `(1, "a")`, fields `.0` and `.1`, and destructuring in
  assignments (`a, b = pair`), `for` loops and `lambda (a, b): ...`.
* `list` methods `map`, `filter`, `any`, `all`, `find`, `enumerate`, `zip`, `sum`, `min`, `max`,
  `fold`, `for_each`, `skip` and `join`, and `dict.entries`. `min`, `max` and `find` return an
  `Option`. These run in the interpreter; the native backend does not compile them yet.

## Classes, methods, traits and generics

```telvax
trait Shape:
    def area(self) -> float: ...
    def describe(self) -> str:
        return f"area {self.area()}"

class Circle(Shape):
    r: float

    def area(self) -> float:
        return self.r * self.r * 3.14159

class Stack[T]:
    items: list[T]

    def push(self, x: T):
        self.items.push(x)
```

A `class` lists its fields, then its methods. `class Circle(Shape)` says the class implements the
trait; a method without a body (`: ...`) is required, one with a body is a default. `extend Circle:`
adds methods to a class declared elsewhere. Call a function that has no `self` as `Circle.new(1.0)`.
`[T]` after a name declares a type parameter; `T: Shape` limits it to types that implement `Shape`.

See [use cases](use-cases.md) for complete programs.
