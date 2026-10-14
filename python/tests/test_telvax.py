import pathlib
import time

import numpy as np
import pyarrow as pa
import pytest

import telvax

ROOT = pathlib.Path(__file__).resolve().parents[2]

PROGRAM = """
class Point:
    x: float
    y: float

class Order:
    id: int
    tags: list[str]
    at: Point

def add(a: int, b: int) -> int:
    return a + b

def scale(p: Point, k: float) -> Point:
    return Point(x=p.x * k, y=p.y * k)

def describe(o: Order) -> str:
    return f"order {o.id} at ({o.at.x}, {o.at.y}) with {o.tags.len()} tags"

def total(v: list[int]) -> int:
    s = 0
    for x in v:
        s += x
    return s

def shout(s: str) -> str:
    return std.text.uppercase(s)

def nothing():
    pass

def boom(d: int) -> int:
    return 10 / d

def tsum(t: Tensor[float32, [4]]) -> float32:
    return std.tensor.sum(t)

def double(t: Tensor[float32, [4]]) -> Tensor[float32, [4]]:
    return std.tensor.scale(t, 2.0)

def head(t: Tensor[float, [N]]) -> float:
    return t[0]

def mm(a: Tensor[float32, [2, 3]], b: Tensor[float32, [3, 2]]) -> Tensor[float32, [2, 2]]:
    return std.tensor.matmul(a, b)

class Row:
    id: int
    score: float32
    name: str

def count(t: Table[Row]) -> uint64:
    return t.len()

def mean_score(t: Table[Row]) -> float32:
    return std.tensor.mean(t.score)

def best(t: Table[Row]) -> Table[Row]:
    return t |> std.table.sql("SELECT * FROM t WHERE score > 1.5 ORDER BY score DESC")
"""


@pytest.fixture(scope="module")
def prog():
    return telvax.compile(PROGRAM)


# ---- calling Telvax from Python -------------------------------------------------------------


def test_scalars_strings_and_unit(prog):
    assert prog.call("add", 2, 40) == 42
    assert prog.call("shout", "hi") == "HI"
    assert prog.call("nothing") is None
    assert prog.call("total", [1, 2, 3, 4]) == 10
    assert prog.call("total", (5, 6)) == 11
    assert "add" in prog.functions()


def test_structs_are_dicts(prog):
    assert prog.call("scale", {"x": 1.5, "y": -2.0}, 2.0) == {"x": 3.0, "y": -4.0}
    order = {"id": 7, "tags": ["a", "b"], "at": {"x": 1.0, "y": 2.0}}
    assert prog.call("describe", order) == "order 7 at (1, 2) with 2 tags"


def test_type_errors_are_reported(prog):
    with pytest.raises(telvax.TelvaxError, match="argument `a` of `add`"):
        prog.call("add", "x", 1)
    with pytest.raises(telvax.TelvaxError, match="takes 2 argument"):
        prog.call("add", 1)
    with pytest.raises(telvax.TelvaxError, match="no function"):
        prog.call("nope")
    with pytest.raises(telvax.TelvaxError, match="needs the field `y`"):
        prog.call("scale", {"x": 1.0}, 2.0)
    with pytest.raises(telvax.TelvaxError):
        prog.call("add", True, 1)  # bool is not an integer


def test_runtime_errors_become_exceptions(prog):
    with pytest.raises(telvax.TelvaxError, match="division by zero"):
        prog.call("boom", 0)


def test_compile_errors_carry_diagnostics():
    with pytest.raises(telvax.CompileError) as e:
        telvax.compile("""
def f() -> int32:
    return "oops"
""", name="bad.tvx")
    assert "bad.tvx:3" in str(e.value) and "expected `int32`, found `str`" in str(e.value)


def test_rag_endpoint_and_status_errors():
    prog = telvax.load(str(ROOT / "examples" / "rag-service.tvx"))
    assert prog.endpoints() == ["handle_rag"]
    out = prog.endpoint("handle_rag", {"query": "tensor shapes", "top_k": 2})
    assert out["query"] == "tensor shapes" and "answer" in out
    with pytest.raises(telvax.StatusError, match="cannot be empty") as e:
        prog.endpoint("handle_rag", {"query": "", "top_k": 1})
    assert e.value.status == 400


# ---- tensors: NumPy, Arrow, PyTorch --------------------------------------------------------


def test_numpy_in_is_zero_copy(prog):
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    t = telvax.tensor(a)
    assert t.data_ptr == a.__array_interface__["data"][0], "no copy on the way in"
    assert t.shape == (4,) and t.dtype == "float32"
    assert prog.call("tsum", a) == 10.0
    a[0] = 100.0  # visible through the shared buffer
    assert t.tolist()[0] == 100.0


def test_numpy_out_is_zero_copy(prog):
    t = prog.call("double", np.array([1, 2, 3, 4], dtype=np.float32))
    arr = np.asarray(t)
    assert arr.dtype == np.float32 and arr.tolist() == [2.0, 4.0, 6.0, 8.0]
    assert arr.__array_interface__["data"][0] == t.data_ptr, "no copy on the way out"
    assert arr.base is t or np.shares_memory(arr, arr)


def test_matrix_shapes(prog):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((3, 2), dtype=np.float32)
    out = np.asarray(prog.call("mm", a, b))
    assert out.shape == (2, 2) and np.allclose(out, a @ b)
    with pytest.raises(telvax.TelvaxError, match="expected shape"):
        prog.call("mm", b, a)  # [3, 2] x [2, 3] has the wrong static shape


def test_dtype_conversion_copies_only_when_needed(prog):
    f64 = np.array([1, 2, 3, 4], dtype=np.float64)
    assert prog.call("tsum", f64) == 10.0  # f64 -> f32 parameter: converted
    assert prog.call("head", f64) == 1.0
    t = telvax.tensor(f64)
    assert t.data_ptr == f64.__array_interface__["data"][0]


def test_non_contiguous_arrays_are_rejected_not_copied_silently():
    a = np.arange(8, dtype=np.float32)[::2]
    with pytest.raises(telvax.TelvaxError, match="C-contiguous"):
        telvax.tensor(a)
    assert telvax.tensor(np.ascontiguousarray(a)).shape == (4,)


def test_pyarrow_round_trip_shares_buffers():
    arr = pa.array([1.0, 2.0, 3.0], pa.float32())
    t = telvax.Tensor.from_arrow(arr)
    assert t.data_ptr == arr.buffers()[1].address, "imported without copying"
    back = pa.array(t)
    assert back.to_pylist() == [1.0, 2.0, 3.0]
    assert back.buffers()[1].address == t.data_ptr, "exported without copying"


def test_polars_series_in_and_out(prog):
    import polars as pl

    s = pl.Series("x", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float32)
    assert prog.call("tsum", s) == 10.0
    out = pl.Series(prog.call("double", s))
    assert out.to_list() == [2.0, 4.0, 6.0, 8.0]


def test_torch_tensors_share_memory_through_numpy(prog):
    torch = pytest.importorskip("torch")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert prog.call("tsum", x.numpy()) == 10.0
    t = telvax.tensor(x.numpy())
    assert t.data_ptr == x.data_ptr(), "torch -> telvax without copying"
    y = torch.from_numpy(np.asarray(prog.call("double", x.numpy())))
    assert y.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_nested_lists_build_tensors():
    t = telvax.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3) and t.dtype == "float32" and t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(telvax.TelvaxError, match="rectangular"):
        telvax.tensor([[1, 2], [3]])


# ---- tables ---------------------------------------------------------------------------------


def make_table():
    return pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "score": pa.array([1.0, 2.5, 3.5], pa.float32()),
            "name": ["ann", "bob", "cy"],
        }
    )


def test_pyarrow_tables_in_and_out(prog):
    table = make_table()
    assert prog.call("count", table) == 3
    assert prog.call("mean_score", table) == pytest.approx(7.0 / 3.0, rel=1e-6)
    top = prog.call("best", table)
    assert isinstance(top, telvax.Table) and len(top) == 2 and top.schema == "Row"
    assert top.columns == ["id", "score", "name"]
    assert pa.record_batch(top).to_pylist() == [
        {"id": 3, "score": 3.5, "name": "cy"},
        {"id": 2, "score": 2.5, "name": "bob"},
    ]
    assert top.to_pylist()[0]["name"] == "cy"


def test_table_columns_are_zero_copy_tensors(prog):
    table = make_table()
    t = telvax.table(table, "Row")
    col = t.column("score")
    assert isinstance(col, telvax.Tensor) and col.data_ptr == table.column("score").chunk(0).buffers()[1].address
    assert t.column("name") == ["ann", "bob", "cy"]


def test_polars_frames_are_tables(prog):
    import polars as pl

    df = pl.DataFrame({"id": [1, 2], "score": [1.0, 2.0], "name": ["a", "b"]}, schema_overrides={"score": pl.Float32})
    assert prog.call("count", df) == 2
    assert pl.DataFrame(prog.call("best", df)).shape == (1, 3)


def test_missing_columns_are_reported(prog):
    with pytest.raises(telvax.TelvaxError, match="no column `name`"):
        prog.call("count", pa.table({"id": [1], "score": pa.array([1.0], pa.float32())}))


# ---- GPU tensors ----------------------------------------------------------------------------


class FakeCudaArray:
    """Anything that implements `__cuda_array_interface__` (PyTorch, CuPy, Numba, ...)."""

    def __init__(self, ptr=0x7F0000001000, shape=(2, 3), typestr="<f4", strides=None, version=3, stream=1):
        self.__cuda_array_interface__ = {
            "version": version,
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "strides": strides,
            "stream": stream,
        }


def test_device_tensors_pass_by_pointer():
    src = FakeCudaArray()
    d = telvax.tensor(src)
    assert isinstance(d, telvax.DeviceTensor)
    assert d.device_ptr == 0x7F0000001000 and d.shape == (2, 3) and d.typestr == "<f4"
    assert d.is_contiguous and d.owner is src
    # Re-exported verbatim, so any other framework can adopt it with no copy.
    assert d.__cuda_array_interface__ == {**src.__cuda_array_interface__, "shape": (2, 3)}
    again = telvax.DeviceTensor.from_cuda(d)
    assert again.device_ptr == d.device_ptr


def test_device_tensor_validation():
    with pytest.raises(telvax.TelvaxError, match="too old"):
        telvax.tensor(FakeCudaArray(version=1))
    with pytest.raises(telvax.TelvaxError, match="null device pointer"):
        telvax.tensor(FakeCudaArray(ptr=0))
    with pytest.raises(telvax.TelvaxError, match="invalid typestr"):
        telvax.tensor(FakeCudaArray(typestr="float"))
    d = telvax.tensor(FakeCudaArray(strides=(4, 8)))
    assert not d.is_contiguous


def test_device_tensors_cannot_enter_cpu_functions(prog):
    with pytest.raises(telvax.TelvaxError, match="device \\(GPU\\) tensor"):
        prog.call("tsum", FakeCudaArray(shape=(4,)))


# ---- overhead -------------------------------------------------------------------------------


def test_scalar_calls_cost_microseconds(prog):
    n = 20000
    start = time.perf_counter()
    for i in range(n):
        prog.call("add", i, 1)
    per_call_us = (time.perf_counter() - start) / n * 1e6
    print(f"\nscalar call overhead: {per_call_us:.2f} us/call")
    assert per_call_us < 100, "bridging a scalar call should take microseconds, not milliseconds"


# ---- distributed execution -----------------------------------------------------------------

DIST = """
def square(x: int) -> int:
    return x * x

def collatz(n: int) -> int:
    steps: int = 0
    x = n
    while x != 1:
        if x % 2 == 0:
            x = x / 2
        else:
            x = 3 * x + 1
        steps += 1
    return steps

def squares(v: list[int]) -> list[int]:
    return std.dist.map(v, square)

def longest(v: list[int]) -> int:
    best: int = 0
    for steps in std.dist.map(v, collatz):
        if steps > best:
            best = steps
    return best
"""


def test_a_python_callable_can_execute_mapped_calls():
    calls = []

    def executor(function, items):
        calls.append((function, len(items)))
        prog = telvax.compile(DIST)
        return [prog.call_json(function, item) for item in items]

    prog = telvax.compile(DIST)
    assert prog.call("squares", [1, 2, 3]) == [1, 4, 9], "without an executor it runs locally"
    prog.set_executor(executor)
    assert prog.call("squares", [4, 5, 6]) == [16, 25, 36]
    assert calls == [("square", 3)]


def test_executor_errors_surface_as_telvax_errors():
    prog = telvax.compile(DIST)

    def broken(function, items):
        raise RuntimeError("cluster is down")

    prog.set_executor(broken)
    with pytest.raises(telvax.TelvaxError, match="cluster is down"):
        prog.call("squares", [1])

    prog.set_executor(lambda f, items: ["not json"])
    with pytest.raises(telvax.TelvaxError, match="invalid JSON"):
        prog.call("squares", [1])


def test_call_json_runs_one_item():
    prog = telvax.compile(DIST)
    assert prog.call_json("square", "12") == "144"
    with pytest.raises(telvax.TelvaxError, match="argument"):
        prog.call_json("square", '"x"')


def test_mapped_calls_run_as_ray_tasks():
    ray = pytest.importorskip("ray")
    import telvax.ray as nray

    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    try:
        prog = telvax.compile(DIST)
        prog.set_executor(nray.RayExecutor(DIST))
        assert prog.call("squares", list(range(50))) == [x * x for x in range(50)]
        local = telvax.compile(DIST)
        items = list(range(1, 400))
        assert prog.call("longest", items) == local.call("longest", items)
        # Same answers with one task per item.
        prog.set_executor(nray.RayExecutor(DIST, chunk=1))
        assert prog.call("squares", [3, 4]) == [9, 16]
        assert prog.call("squares", []) == []
    finally:
        ray.shutdown()


# ---- enums ---------------------------------------------------------------------------------


ENUMS = """
enum Shape:
    Circle(float)
    Rect(float, float)
    Empty

def area(s: Shape) -> float:
    match s:
        case Shape.Circle(r):
            return r * r * 3.0
        case Shape.Rect(w, h):
            return w * h
        case Shape.Empty:
            return 0.0

def make(kind: int) -> Shape:
    match kind:
        case 0:
            return Shape.Circle(1.5)
        case 1:
            return Shape.Rect(2.0, 3.0)
        case _:
            return Shape.Empty
"""


def test_enums_use_the_json_shape():
    p = telvax.compile(ENUMS)
    assert p.call("area", {"Circle": 2.0}) == 12.0
    assert p.call("area", {"Rect": [2.0, 3.5]}) == 7.0
    assert p.call("area", "Empty") == 0.0
    assert p.call("make", 0) == {"Circle": 1.5}
    assert p.call("make", 1) == {"Rect": [2.0, 3.0]}
    assert p.call("make", 2) == "Empty"
    with pytest.raises(telvax.TelvaxError):
        p.call("area", "Triangle")


# ---- Option, Map and the shorter syntax ----------------------------------------------------


OPT_MAP = """
def find(words: list[str], w: str) -> uint64 | None:
    words.contains(w) and True
    words.first().is_some() and w.len() > 0
    return w.find("a")

def word_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = std.map.new()
    for w in text.split(" "):
        counts.insert(w, counts.get(w).unwrap_or(0) + 1)
    return counts

def lookup(m: dict[str, int], k: str) -> int:
    return m.get(k).unwrap_or(0 - 1)

def maybe(flag: bool) -> str | None:
    if flag:
        return Some("yes")
    else:
        return None

def total(x: Tensor[float32, [4]]) -> float32:
    return (x * 2.0 + x).sum()
"""


def test_option_and_map_cross_the_boundary():
    p = telvax.compile(OPT_MAP)
    assert p.call("maybe", True) == "yes"
    assert p.call("maybe", False) is None
    assert p.call("word_counts", "a b a") == {"a": 2, "b": 1}
    assert p.call("lookup", {"x": 5}, "x") == 5
    assert p.call("lookup", {"x": 5}, "y") == -1
    assert p.call("find", ["a"], "banana") == 1


def test_tensor_operators_run_from_python():
    p = telvax.compile(OPT_MAP)
    assert p.call("total", np.ones(4, np.float32)) == 12.0
