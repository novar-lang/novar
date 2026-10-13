//! `novar.Table`: an Arrow `RecordBatch` with a Novar struct as its schema.

use std::sync::Arc;

use arrow::array::{Array, StructArray};
use arrow::compute::concat_batches;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use novar_runtime::TableVal;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};

use crate::NovarError;
use crate::tensor::{Tensor, export_capsules, import_capsules};

#[pyclass(module = "novar", frozen)]
pub struct Table {
    pub inner: Arc<TableVal>,
}

/// A batch from anything that speaks the Arrow PyCapsule interface: a record batch
/// (`__arrow_c_array__`) or a table / reader (`__arrow_c_stream__`).
pub fn import_batch(obj: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    if obj.hasattr("__arrow_c_array__")? {
        let array = import_capsules(obj)?;
        let structs = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| NovarError::new_err("expected a struct array (a record batch)"))?;
        return Ok(RecordBatch::from(structs.clone()));
    }
    if obj.hasattr("__arrow_c_stream__")? {
        let capsule = obj.call_method1("__arrow_c_stream__", (obj.py().None(),))?;
        let capsule = capsule.cast::<PyCapsule>().map_err(|_| NovarError::new_err("__arrow_c_stream__ did not return a capsule"))?;
        let ptr = capsule.pointer_checked(Some(c"arrow_array_stream"))?.as_ptr() as *mut FFI_ArrowArrayStream;
        // SAFETY: the capsule holds a live stream struct; `from_raw` takes ownership of it.
        let reader = unsafe { ArrowArrayStreamReader::from_raw(ptr) }.map_err(|e| NovarError::new_err(e.to_string()))?;
        let schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>().map_err(|e| NovarError::new_err(e.to_string()))?;
        return concat_batches(&schema, &batches).map_err(|e| NovarError::new_err(e.to_string()));
    }
    Err(NovarError::new_err(
        "expected an object with __arrow_c_array__ or __arrow_c_stream__ (a PyArrow table, a Polars frame, ...)",
    ))
}

#[pymethods]
impl Table {
    /// Name of the Novar struct whose fields are the table's columns.
    #[getter]
    fn schema(&self) -> String {
        self.inner.schema.clone()
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.batch.schema().fields().iter().map(|f| f.name().clone()).collect()
    }

    fn __len__(&self) -> usize {
        self.inner.rows()
    }

    /// One column: numeric columns are zero-copy `Tensor`s, others are Python lists.
    fn column(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        crate::convert::value_to_py(py, self.inner.column(name).map_err(crate::runtime_err)?)
    }

    /// A list of dicts, one per row (copies the data).
    fn to_pylist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rows = (0..self.inner.rows())
            .map(|i| self.inner.row(i).map_err(crate::runtime_err).and_then(|v| crate::convert::value_to_py(py, v)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, rows)?.into_any().unbind())
    }

    /// Arrow PyCapsule interface: the table as one struct array, without copying.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let _ = requested_schema;
        let array: arrow::array::ArrayRef = Arc::new(StructArray::from(self.inner.batch.clone()));
        export_capsules(py, &array)
    }

    fn __repr__(&self) -> String {
        format!("<novar.Table {}: {} rows x {} columns>", self.inner.schema, self.inner.rows(), self.inner.batch.num_columns())
    }
}

/// Wraps a PyArrow table / record batch (or any Arrow-capsule object) as a `Table`.
#[pyfunction]
#[pyo3(signature = (obj, schema = "Row"))]
pub fn table(obj: &Bound<'_, PyAny>, schema: &str) -> PyResult<Table> {
    let batch = import_batch(obj)?;
    Ok(Table { inner: Arc::new(TableVal { schema: schema.to_string(), batch }) })
}

#[allow(dead_code)]
fn _assert_tensor_is_used(_: &Tensor) {}
