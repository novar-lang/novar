//! `novar.Tensor`: an Arrow-backed tensor that exchanges memory with the Python ecosystem.

use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array, make_array,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi};
use novar_runtime::TensorVal;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};

use crate::NovarError;

/// A dense n-dimensional array (row-major) backed by an Arrow buffer.
#[pyclass(module = "novar", frozen)]
pub struct Tensor {
    pub inner: Arc<TensorVal>,
}

pub fn typestr(dt: &DataType) -> PyResult<&'static str> {
    Ok(match dt {
        DataType::Float32 => "<f4",
        DataType::Float64 => "<f8",
        DataType::Int8 => "|i1",
        DataType::Int16 => "<i2",
        DataType::Int32 => "<i4",
        DataType::Int64 => "<i8",
        DataType::UInt8 => "|u1",
        DataType::UInt16 => "<u2",
        DataType::UInt32 => "<u4",
        DataType::UInt64 => "<u8",
        other => return Err(NovarError::new_err(format!("tensors of {other} cannot be exposed as a buffer"))),
    })
}

fn dtype_name(dt: &DataType) -> String {
    match dt {
        DataType::Float32 => "float32".into(),
        DataType::Float64 => "float64".into(),
        DataType::Int8 => "int8".into(),
        DataType::Int16 => "int16".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::UInt8 => "uint8".into(),
        DataType::UInt16 => "uint16".into(),
        DataType::UInt32 => "uint32".into(),
        DataType::UInt64 => "uint64".into(),
        other => other.to_string(),
    }
}

/// Address of the first element, honouring the array's offset into its buffer.
fn data_pointer(array: &ArrayRef) -> PyResult<usize> {
    let data = array.to_data();
    let width = data.data_type().primitive_width().ok_or_else(|| NovarError::new_err("not a numeric tensor"))?;
    let buffer = data.buffers().first().ok_or_else(|| NovarError::new_err("tensor has no data buffer"))?;
    Ok(buffer.as_ptr() as usize + data.offset() * width)
}

/// The two PyCapsules of the Arrow PyCapsule interface for `array`.
pub fn export_capsules<'py>(
    py: Python<'py>,
    array: &ArrayRef,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    let (ffi_array, ffi_schema) = to_ffi(&array.to_data()).map_err(|e| NovarError::new_err(e.to_string()))?;
    let schema = PyCapsule::new_with_value_and_destructor(py, ffi_schema, c"arrow_schema", |s, _| drop(s))?;
    let array = PyCapsule::new_with_value_and_destructor(py, ffi_array, c"arrow_array", |a, _| drop(a))?;
    Ok((schema, array))
}

/// Imports the array described by `obj.__arrow_c_array__()` without copying its buffers.
pub fn import_capsules(obj: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let pair = obj.call_method0("__arrow_c_array__")?;
    let (schema_cap, array_cap): (Bound<'_, PyCapsule>, Bound<'_, PyCapsule>) = pair.extract()?;
    let schema_ptr = schema_cap.pointer_checked(Some(c"arrow_schema"))?.as_ptr() as *mut FFI_ArrowSchema;
    let array_ptr = array_cap.pointer_checked(Some(c"arrow_array"))?.as_ptr() as *mut FFI_ArrowArray;
    // SAFETY: the capsules hold live Arrow C Data Interface structs. Per the interface the consumer
    // takes ownership of the array (we leave an empty struct behind) and only borrows the schema.
    let data = unsafe {
        let array = std::ptr::replace(array_ptr, FFI_ArrowArray::empty());
        from_ffi(array, &*schema_ptr)
    }
    .map_err(|e| NovarError::new_err(format!("cannot import the Arrow array: {e}")))?;
    Ok(make_array(data))
}

fn shape_of(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    obj.extract::<Vec<usize>>()
}

macro_rules! typed_from_buffer {
    ($arr:ident, $t:ty, $buffer:expr, $n:expr) => {
        Arc::new($arr::new(ScalarBuffer::<$t>::new($buffer, 0, $n), None)) as ArrayRef
    };
}

/// Zero-copy tensor over a NumPy-style buffer (`__array_interface__`). The array is kept alive by
/// the tensor; writes through either side are visible to both.
fn from_array_interface(obj: &Bound<'_, PyAny>, shape: Option<Vec<usize>>) -> PyResult<Tensor> {
    let iface = obj.getattr("__array_interface__")?;
    let dict = iface.cast::<PyDict>().map_err(|_| NovarError::new_err("__array_interface__ is not a dict"))?;
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        dict.get_item(k)?.ok_or_else(|| NovarError::new_err(format!("__array_interface__ has no `{k}`")))
    };
    let typestr: String = get("typestr")?.extract()?;
    let array_shape = shape_of(&get("shape")?)?;
    let strides = dict.get_item("strides")?;
    let contiguous = match &strides {
        None => true,
        Some(s) if s.is_none() => true,
        Some(s) => {
            let strides: Vec<usize> = s.extract()?;
            let width: usize = typestr[2..].parse().unwrap_or(0);
            let mut expected = width;
            let mut ok = true;
            for (dim, stride) in array_shape.iter().zip(&strides).rev() {
                ok &= *stride == expected || *dim <= 1;
                expected *= dim;
            }
            ok
        }
    };
    if !contiguous {
        return Err(NovarError::new_err("only C-contiguous arrays can be shared; call `.copy()` or `np.ascontiguousarray` first"));
    }
    let (ptr, _readonly): (usize, bool) = get("data")?.extract()?;
    let n: usize = array_shape.iter().product();
    let width: usize = typestr[2..].parse().map_err(|_| NovarError::new_err(format!("bad typestr `{typestr}`")))?;
    let target_shape = match shape {
        Some(s) => {
            if s.iter().product::<usize>() != n {
                return Err(NovarError::new_err(format!("shape {s:?} does not match the {n} elements")));
            }
            s
        }
        None => array_shape,
    };
    if !typestr.starts_with(['<', '|']) {
        return Err(NovarError::new_err("big-endian arrays are not supported"));
    }

    let array: ArrayRef = if n == 0 {
        Arc::new(Float32Array::from(Vec::<f32>::new()))
    } else {
        let nn = NonNull::new(ptr as *mut u8).ok_or_else(|| NovarError::new_err("null data pointer"))?;
        if ptr % width.max(1) != 0 {
            return Err(NovarError::new_err("the array is not aligned for its element type"));
        }
        // SAFETY: `obj` owns the memory for as long as the buffer's owner (a strong reference to
        // `obj`) lives, the region is `n * width` bytes by the interface contract, and it is aligned.
        let owner = Arc::new(AssertUnwindSafe(obj.clone().unbind()));
        let buffer = unsafe { Buffer::from_custom_allocation(nn, n * width, owner) };
        match typestr.as_str() {
            "<f4" => typed_from_buffer!(Float32Array, f32, buffer, n),
            "<f8" => typed_from_buffer!(Float64Array, f64, buffer, n),
            "|i1" => typed_from_buffer!(Int8Array, i8, buffer, n),
            "<i2" => typed_from_buffer!(Int16Array, i16, buffer, n),
            "<i4" => typed_from_buffer!(Int32Array, i32, buffer, n),
            "<i8" => typed_from_buffer!(Int64Array, i64, buffer, n),
            "|u1" => typed_from_buffer!(UInt8Array, u8, buffer, n),
            "<u2" => typed_from_buffer!(UInt16Array, u16, buffer, n),
            "<u4" => typed_from_buffer!(UInt32Array, u32, buffer, n),
            "<u8" => typed_from_buffer!(UInt64Array, u64, buffer, n),
            other => return Err(NovarError::new_err(format!("unsupported array type `{other}`"))),
        }
    };
    Ok(Tensor { inner: Arc::new(TensorVal::from_array(target_shape, array).map_err(crate::runtime_err)?) })
}

fn nested_to_flat(obj: &Bound<'_, PyAny>, depth: usize, shape: &mut Vec<usize>, out: &mut Vec<f64>) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        if shape.len() == depth {
            shape.push(list.len());
        } else if shape[depth] != list.len() {
            return Err(NovarError::new_err("nested lists must be rectangular"));
        }
        for item in list.iter() {
            nested_to_flat(&item, depth + 1, shape, out)?;
        }
        return Ok(());
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        if shape.len() == depth {
            shape.push(tuple.len());
        } else if shape[depth] != tuple.len() {
            return Err(NovarError::new_err("nested sequences must be rectangular"));
        }
        for item in tuple.iter() {
            nested_to_flat(&item, depth + 1, shape, out)?;
        }
        return Ok(());
    }
    out.push(obj.extract::<f64>().map_err(|_| NovarError::new_err("tensor elements must be numbers"))?);
    Ok(())
}

#[pymethods]
impl Tensor {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape.iter().copied())
    }

    #[getter]
    fn dtype(&self) -> String {
        dtype_name(self.inner.dtype())
    }

    /// The address of the first element (for zero-copy checks and FFI).
    #[getter]
    fn data_ptr(&self) -> PyResult<usize> {
        data_pointer(&self.inner.array)
    }

    fn __len__(&self) -> usize {
        self.inner.shape.first().copied().unwrap_or(0)
    }

    /// Nested Python lists (copies the data).
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        fn build(py: Python<'_>, shape: &[usize], data: &[f64], int: bool) -> PyResult<Py<PyAny>> {
            match shape.split_first() {
                None => Ok(if int { (data[0] as i64).into_pyobject(py)?.into_any().unbind() } else { data[0].into_pyobject(py)?.into_any().unbind() }),
                Some((&n, rest)) => {
                    let chunk: usize = rest.iter().product();
                    let items = (0..n)
                        .map(|i| build(py, rest, &data[i * chunk..(i + 1) * chunk], int))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyList::new(py, items)?.into_any().unbind())
                }
            }
        }
        let int = !matches!(self.inner.dtype(), DataType::Float32 | DataType::Float64);
        build(py, &self.inner.shape, &self.inner.to_f64_vec(), int)
    }

    /// Arrow PyCapsule interface: the flat element array, without copying.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let _ = requested_schema;
        export_capsules(py, &self.inner.array)
    }

    /// NumPy array interface: `numpy.asarray(tensor)` shares this tensor's memory.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("version", 3)?;
        dict.set_item("shape", PyTuple::new(py, self.inner.shape.iter().copied())?)?;
        dict.set_item("typestr", typestr(self.inner.dtype())?)?;
        dict.set_item("data", (data_pointer(&self.inner.array)?, true))?;
        dict.set_item("strides", py.None())?;
        Ok(dict)
    }

    /// Imports any object with `__arrow_c_array__` (PyArrow, Polars, ...) without copying.
    #[staticmethod]
    #[pyo3(signature = (obj, shape = None))]
    fn from_arrow(obj: &Bound<'_, PyAny>, shape: Option<Vec<usize>>) -> PyResult<Tensor> {
        let array = import_capsules(obj)?;
        let shape = shape.unwrap_or_else(|| vec![array.len()]);
        Ok(Tensor { inner: Arc::new(TensorVal::from_array(shape, array).map_err(crate::runtime_err)?) })
    }

    /// Shares the memory of a C-contiguous NumPy array (or any `__array_interface__` object).
    #[staticmethod]
    #[pyo3(signature = (obj, shape = None))]
    fn from_buffer(obj: &Bound<'_, PyAny>, shape: Option<Vec<usize>>) -> PyResult<Tensor> {
        from_array_interface(obj, shape)
    }

    fn __repr__(&self) -> String {
        format!("<novar.Tensor shape={:?} dtype={}>", self.inner.shape, dtype_name(self.inner.dtype()))
    }
}

/// Builds a tensor from the most capable protocol the object offers: Arrow PyCapsules, then the
/// NumPy array interface and `to_numpy()` (all zero-copy when the data allows), then nested lists
/// of numbers (copied as float32).
#[pyfunction]
#[pyo3(signature = (obj, shape = None))]
pub fn tensor(py: Python<'_>, obj: &Bound<'_, PyAny>, shape: Option<Vec<usize>>) -> PyResult<Py<PyAny>> {
    if obj.hasattr("__cuda_array_interface__")? {
        return Ok(Py::new(py, crate::cuda::DeviceTensor::from_cuda(obj)?)?.into_any());
    }
    if obj.hasattr("__arrow_c_array__")? {
        return Ok(Py::new(py, Tensor::from_arrow(obj, shape)?)?.into_any());
    }
    if obj.hasattr("__array_interface__")? {
        return Ok(Py::new(py, from_array_interface(obj, shape)?)?.into_any());
    }
    // Series-like objects (Polars, pandas) hand out NumPy arrays, zero-copy when they can.
    if obj.hasattr("to_numpy")? {
        let array = obj.call_method0("to_numpy")?;
        return Ok(Py::new(py, from_array_interface(&array, shape)?)?.into_any());
    }
    let mut dims = Vec::new();
    let mut flat = Vec::new();
    nested_to_flat(obj, 0, &mut dims, &mut flat)?;
    let shape = shape.unwrap_or(dims);
    let data = flat.iter().map(|x| *x as f32).collect();
    let t = TensorVal::from_f32(vec![flat.len()], data);
    let t = TensorVal::from_array(shape, t.array.clone()).map_err(crate::runtime_err)?;
    Ok(Py::new(py, Tensor { inner: Arc::new(t) })?.into_any())
}
