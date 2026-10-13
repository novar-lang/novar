//! `novar.DeviceTensor`: a GPU buffer handed through Novar by pointer.
//!
//! Frameworks that keep tensors on the GPU (PyTorch, CuPy, Numba, JAX, ...) describe them with
//! `__cuda_array_interface__`. A `DeviceTensor` records that description and keeps the source
//! object alive, and exposes the same interface again, so another framework can adopt the buffer
//! with no host round trip and no copy. Novar itself has no GPU kernels yet: the tensor is
//! carried, validated and re-exported, not computed on.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::NovarError;

#[pyclass(module = "novar", frozen)]
pub struct DeviceTensor {
    ptr: usize,
    shape: Vec<usize>,
    typestr: String,
    strides: Option<Vec<usize>>,
    readonly: bool,
    stream: Option<usize>,
    version: u32,
    /// The object that owns the device memory.
    owner: Py<PyAny>,
}

fn width_of(typestr: &str) -> PyResult<usize> {
    let bytes = typestr.as_bytes();
    let ok = bytes.len() >= 3
        && matches!(bytes[0], b'<' | b'>' | b'|')
        && matches!(bytes[1], b'b' | b'i' | b'u' | b'f' | b'c');
    if !ok {
        return Err(NovarError::new_err(format!("invalid typestr `{typestr}` in __cuda_array_interface__")));
    }
    typestr[2..].parse().map_err(|_| NovarError::new_err(format!("invalid typestr `{typestr}` in __cuda_array_interface__")))
}

impl DeviceTensor {
    pub fn from_cuda(obj: &Bound<'_, PyAny>) -> PyResult<DeviceTensor> {
        let iface = obj.getattr("__cuda_array_interface__")?;
        let dict = iface
            .cast::<PyDict>()
            .map_err(|_| NovarError::new_err("__cuda_array_interface__ must be a dict"))?;
        let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
            dict.get_item(k)?.ok_or_else(|| NovarError::new_err(format!("__cuda_array_interface__ has no `{k}`")))
        };
        let version: u32 = get("version")?.extract()?;
        if version < 2 {
            return Err(NovarError::new_err(format!("__cuda_array_interface__ version {version} is too old (need >= 2)")));
        }
        let typestr: String = get("typestr")?.extract()?;
        let width = width_of(&typestr)?;
        let shape: Vec<usize> = get("shape")?.extract()?;
        let (ptr, readonly): (usize, bool) = get("data")?.extract()?;
        let n: usize = shape.iter().product();
        if ptr == 0 && n > 0 {
            return Err(NovarError::new_err("null device pointer for a non-empty tensor"));
        }
        let strides = match dict.get_item("strides")? {
            Some(s) if !s.is_none() => {
                let strides: Vec<usize> = s.extract()?;
                if strides.len() != shape.len() {
                    return Err(NovarError::new_err("strides and shape have different lengths"));
                }
                Some(strides)
            }
            _ => None,
        };
        let stream = match dict.get_item("stream")? {
            Some(s) if !s.is_none() => Some(s.extract::<usize>()?),
            _ => None,
        };
        let _ = width;
        Ok(DeviceTensor { ptr, shape, typestr, strides, readonly, stream, version, owner: obj.clone().unbind() })
    }
}

#[pymethods]
impl DeviceTensor {
    #[staticmethod]
    #[pyo3(name = "from_cuda")]
    fn py_from_cuda(obj: &Bound<'_, PyAny>) -> PyResult<DeviceTensor> {
        DeviceTensor::from_cuda(obj)
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape.iter().copied())
    }

    #[getter]
    fn typestr(&self) -> &str {
        &self.typestr
    }

    /// Device address of the first element.
    #[getter]
    fn device_ptr(&self) -> usize {
        self.ptr
    }

    #[getter]
    fn readonly(&self) -> bool {
        self.readonly
    }

    /// The object that owns the device memory (kept alive by this tensor).
    #[getter]
    fn owner(&self, py: Python<'_>) -> Py<PyAny> {
        self.owner.clone_ref(py)
    }

    /// Whether the elements are densely packed in C order.
    #[getter]
    fn is_contiguous(&self) -> PyResult<bool> {
        let Some(strides) = &self.strides else { return Ok(true) };
        let mut expected = width_of(&self.typestr)?;
        let mut ok = true;
        for (dim, stride) in self.shape.iter().zip(strides).rev() {
            ok &= *stride == expected || *dim <= 1;
            expected *= dim;
        }
        Ok(ok)
    }

    /// Same description the buffer was adopted with, so any consumer can take it zero-copy.
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("version", self.version)?;
        dict.set_item("shape", PyTuple::new(py, self.shape.iter().copied())?)?;
        dict.set_item("typestr", &self.typestr)?;
        dict.set_item("data", (self.ptr, self.readonly))?;
        match &self.strides {
            Some(s) => dict.set_item("strides", PyTuple::new(py, s.iter().copied())?)?,
            None => dict.set_item("strides", py.None())?,
        }
        match self.stream {
            Some(s) => dict.set_item("stream", s)?,
            None => dict.set_item("stream", py.None())?,
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("<novar.DeviceTensor shape={:?} typestr={} ptr=0x{:x}>", self.shape, self.typestr, self.ptr)
    }
}
