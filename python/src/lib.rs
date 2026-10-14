//! Python bindings for Telvax.
//!
//! * `telvax.load(path)` / `telvax.compile(source)` type-check a program and return a `Program`
//!   whose functions and `@endpoint`s can be called from Python.
//! * `Tensor` and `Table` wrap the runtime's Arrow-backed values. They speak the Arrow PyCapsule
//!   interface (`__arrow_c_array__`) and NumPy's `__array_interface__`, so PyArrow, Polars,
//!   NumPy and PyTorch (through NumPy) exchange data with Telvax without copying.
//! * `DeviceTensor` carries a GPU buffer described by `__cuda_array_interface__` (PyTorch, CuPy,
//!   Numba, ...) through Telvax by pointer, again without copying.
//! * Scalars cross the boundary by value; calling a Telvax function costs a few microseconds.

mod convert;
mod cuda;
mod tables;
mod tensor;

use std::sync::Arc;

use telvax_runtime::Interp;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use convert::{json_to_py, py_to_json, py_to_value, value_to_py};

create_exception!(telvax, TelvaxError, PyException, "Base class of every Telvax error.");
create_exception!(telvax, CompileError, TelvaxError, "The program has syntax or type errors.");
create_exception!(
    telvax,
    StatusError,
    TelvaxError,
    "An endpoint returned `Err(Status::..)`; `.status` holds the HTTP status code."
);

pub(crate) fn runtime_err(e: impl std::fmt::Display) -> PyErr {
    TelvaxError::new_err(e.to_string())
}

/// Runs `std.dist.map` through a Python callable.
struct PyExecutor {
    callable: Py<PyAny>,
}

impl telvax_runtime::Executor for PyExecutor {
    fn map(&self, function: &str, items: Vec<serde_json::Value>) -> Result<Vec<serde_json::Value>, String> {
        Python::attach(|py| {
            let texts: Vec<String> = items.iter().map(|v| v.to_string()).collect();
            let out = self
                .callable
                .call1(py, (function, texts))
                .map_err(|e| format!("the executor failed: {e}"))?;
            let results: Vec<String> = out
                .extract(py)
                .map_err(|e| format!("the executor must return a list of JSON strings: {e}"))?;
            results
                .iter()
                .map(|t| serde_json::from_str(t).map_err(|e| format!("the executor returned invalid JSON: {e}")))
                .collect()
        })
    }
}

/// A checked Telvax program ready to run.
#[pyclass(module = "telvax")]
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

    /// Routes `std.dist.map` calls through `executor(function, items)`, where `items` is a list
    /// of JSON strings and the result is a list of JSON strings in the same order. This is how the
    /// work is handed to Ray (see `telvax.ray`) or any other scheduler.
    fn set_executor(&self, executor: Py<PyAny>) {
        self.interp.set_executor(Arc::new(PyExecutor { callable: executor }));
    }

    /// Calls `function` with its arguments as JSON strings and returns the result as JSON.
    /// Workers use this to run one mapped item.
    #[pyo3(signature = (function, *args))]
    fn call_json(&self, py: Python<'_>, function: &str, args: &Bound<'_, PyTuple>) -> PyResult<String> {
        let parsed = args
            .iter()
            .map(|a| {
                let text: String = a.extract()?;
                serde_json::from_str(&text).map_err(|e| TelvaxError::new_err(format!("invalid JSON argument: {e}")))
            })
            .collect::<PyResult<Vec<serde_json::Value>>>()?;
        let interp = self.interp.clone();
        let result = py.detach(move || interp.call_json(function, &parsed));
        result.map(|v| v.to_string()).map_err(TelvaxError::new_err)
    }

    /// Calls function `name` with Python arguments converted by the function's parameter types.
    #[pyo3(signature = (name, *args))]
    fn call(&self, py: Python<'_>, name: &str, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let (params, _) = self
            .interp
            .signature(name)
            .ok_or_else(|| TelvaxError::new_err(format!("no function `{name}`")))?;
        if params.len() != args.len() {
            return Err(TelvaxError::new_err(format!(
                "`{name}` takes {} argument(s), got {}",
                params.len(),
                args.len()
            )));
        }
        let mut values = Vec::with_capacity(params.len());
        for ((pname, ty), arg) in params.iter().zip(args.iter()) {
            values.push(
                py_to_value(&self.interp, &arg, ty)
                    .map_err(|e| TelvaxError::new_err(format!("argument `{pname}` of `{name}`: {e}")))?,
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
        format!("<telvax.Program {} functions>", self.interp.function_names().len())
    }
}

fn build(source: &str, name: &str, live: bool) -> PyResult<Program> {
    let (ast, mut diags) = telvax_parser::parse(source);
    let mut hir = None;
    if let Some(ast) = ast {
        let (h, d) = telvax_hir::lower(&ast);
        diags.extend(d);
        hir = Some(h);
    }
    if diags.iter().any(|d| d.is_error()) {
        let errors = diags.iter().filter(|d| d.is_error()).count();
        let rendered = telvax_diagnostics::render(name, source, &diags, false);
        return Err(CompileError::new_err(format!("{rendered}{errors} error(s) in `{name}`")));
    }
    let hir = hir.ok_or_else(|| CompileError::new_err("the program could not be parsed"))?;
    let interp = Interp::with_live(hir, live).map_err(runtime_err)?;
    Ok(Program { interp: Arc::new(interp) })
}

/// Type-checks Telvax source code and returns a `Program`.
#[pyfunction]
#[pyo3(signature = (source, name = "<string>", live = false))]
fn compile(source: &str, name: &str, live: bool) -> PyResult<Program> {
    build(source, name, live)
}

/// Reads, type-checks and loads a `.tvx` file.
#[pyfunction]
#[pyo3(signature = (path, live = false))]
fn load(path: &str, live: bool) -> PyResult<Program> {
    let source = std::fs::read_to_string(path).map_err(|e| TelvaxError::new_err(format!("cannot read `{path}`: {e}")))?;
    build(&source, path, live)
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Program>()?;
    m.add_class::<tensor::Tensor>()?;
    m.add_class::<tables::Table>()?;
    m.add_class::<cuda::DeviceTensor>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(tensor::tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tables::table, m)?)?;
    m.add("TelvaxError", m.py().get_type::<TelvaxError>())?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;
    m.add("StatusError", m.py().get_type::<StatusError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let _ = PyDict::new(m.py());
    Ok(())
}
