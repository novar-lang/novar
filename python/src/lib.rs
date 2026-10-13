//! Python bindings for Novar.
//!
//! * `novar.load(path)` / `novar.compile(source)` type-check a program and return a `Program`
//!   whose functions and `@endpoint`s can be called from Python.
//! * `Tensor` and `Table` wrap the runtime's Arrow-backed values. They speak the Arrow PyCapsule
//!   interface (`__arrow_c_array__`) and NumPy's `__array_interface__`, so PyArrow, Polars,
//!   NumPy and PyTorch (through NumPy) exchange data with Novar without copying.
//! * `DeviceTensor` carries a GPU buffer described by `__cuda_array_interface__` (PyTorch, CuPy,
//!   Numba, ...) through Novar by pointer, again without copying.
//! * Scalars cross the boundary by value; calling a Novar function costs a few microseconds.

mod convert;
mod cuda;
mod tables;
mod tensor;

use std::sync::Arc;

use novar_runtime::Interp;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use convert::{json_to_py, py_to_json, py_to_value, value_to_py};

create_exception!(novar, NovarError, PyException, "Base class of every Novar error.");
create_exception!(novar, CompileError, NovarError, "The program has syntax or type errors.");
create_exception!(
    novar,
    StatusError,
    NovarError,
    "An endpoint returned `Err(Status::..)`; `.status` holds the HTTP status code."
);

pub(crate) fn runtime_err(e: impl std::fmt::Display) -> PyErr {
    NovarError::new_err(e.to_string())
}

/// A checked Novar program ready to run.
#[pyclass(module = "novar")]
struct Program {
    interp: Arc<Interp>,
}

#[pymethods]
impl Program {
    /// Names of every function in the program.
    fn functions(&self) -> Vec<String> {
        self.interp.function_names()
    }

    /// Names of the `@endpoint` functions.
    fn endpoints(&self) -> Vec<String> {
        self.interp.endpoint_names()
    }

    /// Calls function `name` with Python arguments converted by the function's parameter types.
    #[pyo3(signature = (name, *args))]
    fn call(&self, py: Python<'_>, name: &str, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let (params, _) = self
            .interp
            .signature(name)
            .ok_or_else(|| NovarError::new_err(format!("no function `{name}`")))?;
        if params.len() != args.len() {
            return Err(NovarError::new_err(format!(
                "`{name}` takes {} argument(s), got {}",
                params.len(),
                args.len()
            )));
        }
        let mut values = Vec::with_capacity(params.len());
        for ((pname, ty), arg) in params.iter().zip(args.iter()) {
            values.push(
                py_to_value(&self.interp, &arg, ty)
                    .map_err(|e| NovarError::new_err(format!("argument `{pname}` of `{name}`: {e}")))?,
            );
        }
        let interp = self.interp.clone();
        let result = py.detach(move || interp.call(name, values));
        value_to_py(py, result.map_err(runtime_err)?)
    }

    /// Runs an `@endpoint` function on a JSON-like payload (dict / list / scalars) and returns the
    /// decoded response. Raises `StatusError` when the endpoint answers with an error status.
    fn endpoint(&self, py: Python<'_>, name: &str, payload: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let json = py_to_json(payload)?;
        let interp = self.interp.clone();
        let result = py.detach(move || interp.call_endpoint(name, &json));
        match result {
            Ok(v) => json_to_py(py, &v),
            Err(e) => {
                let err = StatusError::new_err(e.message.clone());
                err.value(py).setattr("status", e.status)?;
                Err(err)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("<novar.Program {} functions>", self.interp.function_names().len())
    }
}

fn build(source: &str, name: &str, live: bool) -> PyResult<Program> {
    let (ast, mut diags) = novar_parser::parse(source);
    let mut hir = None;
    if let Some(ast) = ast {
        let (h, d) = novar_hir::lower(&ast);
        diags.extend(d);
        hir = Some(h);
    }
    if diags.iter().any(|d| d.is_error()) {
        let errors = diags.iter().filter(|d| d.is_error()).count();
        let rendered = novar_diagnostics::render(name, source, &diags, false);
        return Err(CompileError::new_err(format!("{rendered}{errors} error(s) in `{name}`")));
    }
    let hir = hir.ok_or_else(|| CompileError::new_err("the program could not be parsed"))?;
    let interp = Interp::with_live(hir, live).map_err(runtime_err)?;
    Ok(Program { interp: Arc::new(interp) })
}

/// Type-checks Novar source code and returns a `Program`.
#[pyfunction]
#[pyo3(signature = (source, name = "<string>", live = false))]
fn compile(source: &str, name: &str, live: bool) -> PyResult<Program> {
    build(source, name, live)
}

/// Reads, type-checks and loads a `.nv` file.
#[pyfunction]
#[pyo3(signature = (path, live = false))]
fn load(path: &str, live: bool) -> PyResult<Program> {
    let source = std::fs::read_to_string(path).map_err(|e| NovarError::new_err(format!("cannot read `{path}`: {e}")))?;
    build(&source, path, live)
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn novar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Program>()?;
    m.add_class::<tensor::Tensor>()?;
    m.add_class::<tables::Table>()?;
    m.add_class::<cuda::DeviceTensor>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tables::table, m)?)?;
    m.add("NovarError", m.py().get_type::<NovarError>())?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;
    m.add("StatusError", m.py().get_type::<StatusError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let _ = PyDict::new(m.py());
    Ok(())
}
