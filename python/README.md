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
