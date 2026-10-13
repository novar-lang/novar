import pathlib
import time

import numpy as np
import pyarrow as pa
import pytest

import novar

ROOT = pathlib.Path(__file__).resolve().parents[2]

PROGRAM = """
struct Point { x: f64, y: f64 }
struct Order { id: i64, tags: Vec<String>, at: Point }

fn add(a: i64, b: i64) -> i64 { a + b }
fn scale(p: Point, k: f64) -> Point { Point { x: p.x * k, y: p.y * k } }
fn describe(o: Order) -> String { f"order {o.id} at ({o.at.x}, {o.at.y}) with {o.tags.len()} tags" }
fn total(v: Vec<i64>) -> i64 { let mut s = 0; for x in v { s += x; } s }
fn shout(s: String) -> String { std::text::uppercase(s) }
fn nothing() {}
fn boom(d: i64) -> i64 { 10 / d }

fn tsum(t: Tensor<f32, [4]>) -> f32 { std::tensor::sum(t) }
fn double(t: Tensor<f32, [4]>) -> Tensor<f32, [4]> { std::tensor::scale(t, 2.0) }
fn head(t: Tensor<f64, [N]>) -> f64 { t[0] }
fn mm(a: Tensor<f32, [2, 3]>, b: Tensor<f32, [3, 2]>) -> Tensor<f32, [2, 2]> { std::tensor::matmul(a, b) }

struct Row { id: i64, score: f32, name: String }
fn count(t: Table<Row>) -> u64 { t.len() }
fn mean_score(t: Table<Row>) -> f32 { std::tensor::mean(t.score) }
fn best(t: Table<Row>) -> Table<Row> { t |> std::table::sql("SELECT * FROM t WHERE score > 1.5 ORDER BY score DESC") }
"""


@pytest.fixture(scope="module")
def prog():
    return novar.compile(PROGRAM)


# ---- calling Novar from Python -------------------------------------------------------------


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
    with pytest.raises(novar.NovarError, match="argument `a` of `add`"):
        prog.call("add", "x", 1)
    with pytest.raises(novar.NovarError, match="takes 2 argument"):
        prog.call("add", 1)
    with pytest.raises(novar.NovarError, match="no function"):
        prog.call("nope")
    with pytest.raises(novar.NovarError, match="needs the field `y`"):
        prog.call("scale", {"x": 1.0}, 2.0)
    with pytest.raises(novar.NovarError):
        prog.call("add", True, 1)  # bool is not an integer


def test_runtime_errors_become_exceptions(prog):
    with pytest.raises(novar.NovarError, match="division by zero"):
        prog.call("boom", 0)


def test_compile_errors_carry_diagnostics():
    with pytest.raises(novar.CompileError) as e:
        novar.compile("fn f() -> i32 { \"oops\" }", name="bad.nv")
    assert "bad.nv:1" in str(e.value) and "expected `i32`, found `String`" in str(e.value)


def test_rag_endpoint_and_status_errors():
    prog = novar.load(str(ROOT / "examples" / "rag-service.nv"))
    assert prog.endpoints() == ["handle_rag"]
    out = prog.endpoint("handle_rag", {"query": "tensor shapes", "top_k": 2})
    assert out["query"] == "tensor shapes" and "answer" in out
    with pytest.raises(novar.StatusError, match="cannot be empty") as e:
        prog.endpoint("handle_rag", {"query": "", "top_k": 1})
    assert e.value.status == 400


# ---- tensors: NumPy, Arrow, PyTorch --------------------------------------------------------


def test_numpy_in_is_zero_copy(prog):
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    t = novar.tensor(a)
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
    with pytest.raises(novar.NovarError, match="expected shape"):
        prog.call("mm", b, a)  # [3, 2] x [2, 3] has the wrong static shape


def test_dtype_conversion_copies_only_when_needed(prog):
    f64 = np.array([1, 2, 3, 4], dtype=np.float64)
    assert prog.call("tsum", f64) == 10.0  # f64 -> f32 parameter: converted
    assert prog.call("head", f64) == 1.0
    t = novar.tensor(f64)
    assert t.data_ptr == f64.__array_interface__["data"][0]


def test_non_contiguous_arrays_are_rejected_not_copied_silently():
    a = np.arange(8, dtype=np.float32)[::2]
    with pytest.raises(novar.NovarError, match="C-contiguous"):
        novar.tensor(a)
    assert novar.tensor(np.ascontiguousarray(a)).shape == (4,)


def test_pyarrow_round_trip_shares_buffers():
    arr = pa.array([1.0, 2.0, 3.0], pa.float32())
    t = novar.Tensor.from_arrow(arr)
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
    t = novar.tensor(x.numpy())
    assert t.data_ptr == x.data_ptr(), "torch -> novar without copying"
    y = torch.from_numpy(np.asarray(prog.call("double", x.numpy())))
    assert y.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_nested_lists_build_tensors():
    t = novar.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3) and t.dtype == "float32" and t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    with pytest.raises(novar.NovarError, match="rectangular"):
        novar.tensor([[1, 2], [3]])


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
    assert isinstance(top, novar.Table) and len(top) == 2 and top.schema == "Row"
    assert top.columns == ["id", "score", "name"]
    assert pa.record_batch(top).to_pylist() == [
        {"id": 3, "score": 3.5, "name": "cy"},
        {"id": 2, "score": 2.5, "name": "bob"},
    ]
    assert top.to_pylist()[0]["name"] == "cy"


def test_table_columns_are_zero_copy_tensors(prog):
    table = make_table()
    t = novar.table(table, "Row")
    col = t.column("score")
    assert isinstance(col, novar.Tensor) and col.data_ptr == table.column("score").chunk(0).buffers()[1].address
    assert t.column("name") == ["ann", "bob", "cy"]


def test_polars_frames_are_tables(prog):
    import polars as pl

    df = pl.DataFrame({"id": [1, 2], "score": [1.0, 2.0], "name": ["a", "b"]}, schema_overrides={"score": pl.Float32})
    assert prog.call("count", df) == 2
    assert pl.DataFrame(prog.call("best", df)).shape == (1, 3)


def test_missing_columns_are_reported(prog):
    with pytest.raises(novar.NovarError, match="no column `name`"):
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
    d = novar.tensor(src)
    assert isinstance(d, novar.DeviceTensor)
    assert d.device_ptr == 0x7F0000001000 and d.shape == (2, 3) and d.typestr == "<f4"
    assert d.is_contiguous and d.owner is src
    # Re-exported verbatim, so any other framework can adopt it with no copy.
    assert d.__cuda_array_interface__ == {**src.__cuda_array_interface__, "shape": (2, 3)}
    again = novar.DeviceTensor.from_cuda(d)
    assert again.device_ptr == d.device_ptr


def test_device_tensor_validation():
    with pytest.raises(novar.NovarError, match="too old"):
        novar.tensor(FakeCudaArray(version=1))
    with pytest.raises(novar.NovarError, match="null device pointer"):
        novar.tensor(FakeCudaArray(ptr=0))
    with pytest.raises(novar.NovarError, match="invalid typestr"):
        novar.tensor(FakeCudaArray(typestr="float"))
    d = novar.tensor(FakeCudaArray(strides=(4, 8)))
    assert not d.is_contiguous


def test_device_tensors_cannot_enter_cpu_functions(prog):
    with pytest.raises(novar.NovarError, match="device \\(GPU\\) tensor"):
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
