# telvax (Python)

Call Telvax from Python and exchange tensors and tables without copying.

```sh
pip install maturin
cd python && maturin develop --release     # builds and installs the `telvax` module
python -m pytest tests                     # needs numpy, pyarrow (polars / torch are optional)
```

```python
import telvax, numpy as np, pyarrow as pa

prog = telvax.compile("""
type Row:
    id: Int
    score: f32

fn add(a: Int, b: Int) -> Int:
    a + b

fn double(t: Tensor[f32, 4]) -> Tensor[f32, 4]:
    std.tensor.scale(t, 2.0)

fn best(t: Table[Row]) -> Table[Row]:
    t |> std.table.sql("SELECT * FROM t WHERE score > 1.5")
""")

prog.call("add", 1, 2)                                   # 3   (~0.3 us per call)
out = prog.call("double", np.ones(4, np.float32))        # telvax.Tensor, input buffer is shared
np.asarray(out)                                          # NumPy view of Telvax's memory
prog.call("best", pa.table({"id": [1, 2], "score": pa.array([1.0, 2.0], pa.float32())}))
```

| Interface | Telvax type | Python side | Cost |
|---|---|---|---|
| Arrow PyCapsule (`__arrow_c_array__`, `__arrow_c_stream__`) | `Tensor`, `Table` | PyArrow, Polars, pandas (via Arrow) | zero-copy |
| NumPy array interface (`__array_interface__`) | `Tensor` | NumPy, PyTorch (`torch.from_numpy`) | zero-copy (C-contiguous) |
| CUDA array interface (`__cuda_array_interface__`) | `DeviceTensor` | PyTorch, CuPy, Numba | GPU pointer passing |
| PyO3 value conversion | scalars, strings, types you define, `List` | `int`, `float`, `str`, `dict`, `list` | sub-microsecond |

* Functions check argument types and tensor shapes at the boundary: a `Tensor[f32, 2, 3]`
  parameter rejects any other shape, and converts the element type (one copy) when it differs.
* `Program.endpoint(name, payload)` runs an `@endpoint` on a dict; error statuses raise
  `telvax.StatusError` with `.status`.
* `Program.set_executor(callable)` sends `std.dist.map` calls to any scheduler.
  `telvax.ray.RayExecutor(source)` runs them as Ray tasks (tested against a local Ray cluster):

  ```python
  ray.init(); program.set_executor(telvax.ray.RayExecutor(SOURCE)); program.call("process_all", items)
  ```
* `DeviceTensor` carries a GPU buffer through Telvax by pointer and exposes
  `__cuda_array_interface__` again so the next framework adopts it as is. Telvax has no GPU
  kernels yet, so device tensors cannot be passed to Telvax functions (they raise a clear error).
  The CUDA interface is tested with stand-in objects; it has not run against a physical GPU.

## Notebooks

```sh
pip install "telvax[kernel]"       # the Telvax kernel (ipykernel) and the %%telvax magic
jupyter lab                        # pick "Telvax" in the launcher or in Kernel > Change Kernel
```

Installing the package registers the kernel (`jupyter kernelspec list` shows `telvax`); if you installed
without pip's data files, `python -m telvax.kernel install [--user | --sys-prefix | --prefix DIR]` does it.

**A Telvax notebook.** Each cell is Telvax: a definition (`fn`, `type`, `enum`, `trait`, `extend`, `import`),
a variable (`let`, `var`), a statement or an expression. Definitions and variables last from one cell to the
next, the value of the last expression is shown, a table or a tensor as HTML, and what the program prints
appears as the cell's output. An error shows the line and defines nothing. Tab completes names, fields and
methods; Shift+Tab shows the type of the name at the cursor. Cells starting with `:` are commands: `:type
<expression>`, `:vars`, `:defs`, `:reset`, `:help`.

**Telvax in a Python notebook.**

```python
%load_ext telvax
```

```
%%telvax
fn double(n: Int) -> Int:
    n * 2
```

```
%%telvax
double(21)
```

The session belongs to the notebook; `%telvax_reset` forgets it and `%telvax_vars` lists the variables.

**From Python code.** `telvax.Session` is the engine of both, and of `telvax repl`:

```python
s = telvax.Session(allow="none", step_limit=1_000_000)   # what it may do, and when to stop an input
s.eval("let x = 5")                                      # {'output': '', 'value': None, 'defined': ['x'], ...}
s.eval("x * 2")["value"]                                 # {'text': '10', 'type': 'Int', 'html': None}
s.type_of("x")  # 'Int'   s.variables()   s.complete("x.", 2)   s.hover("x", 0)   s.reset()
```

Limits: a cell's output appears when it ends, not as it is printed; interrupting the kernel does not stop a
cell that is already running (restart it, or set `step_limit` in the session); a closure kept in a variable
needs the types of its parameters (`let inc = (n: Int) => n + 1`); `await` is not available.
