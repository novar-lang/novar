//! Conversions between Python objects and runtime values, driven by Telvax's static types.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use telvax_hir::{Dim, Prim, Ty};
use telvax_runtime::{Interp, StructVal, TableVal, TensorVal, Value};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::Value as Json;

use crate::tables::{Table, import_batch};
use crate::tensor::{Tensor, import_capsules};
use crate::{TelvaxError, StatusError, runtime_err};

fn arrow_elem(ty: &Ty) -> Option<DataType> {
    Some(match ty {
        Ty::Prim(Prim::F32) => DataType::Float32,
        Ty::Prim(Prim::F64) => DataType::Float64,
        Ty::Prim(Prim::I8) => DataType::Int8,
        Ty::Prim(Prim::I16) => DataType::Int16,
        Ty::Prim(Prim::I32) => DataType::Int32,
        Ty::Prim(Prim::I64) => DataType::Int64,
        Ty::Prim(Prim::U8) => DataType::UInt8,
        Ty::Prim(Prim::U16) => DataType::UInt16,
        Ty::Prim(Prim::U32) => DataType::UInt32,
        Ty::Prim(Prim::U64) => DataType::UInt64,
        _ => return None,
    })
}

fn tensor_to_value(obj: &Bound<'_, PyAny>, elem: &Ty, shape: &[Dim]) -> PyResult<Value> {
    let t: Arc<TensorVal> = if let Ok(t) = obj.cast::<Tensor>() {
        t.get().inner.clone()
    } else if obj.hasattr("__cuda_array_interface__")? {
        return Err(TelvaxError::new_err(
            "a device (GPU) tensor cannot be passed to a CPU function; copy it to the host first",
        ));
    } else {
        match crate::tensor::tensor(obj.py(), obj, None)?.bind(obj.py()).cast::<Tensor>() {
            Ok(t) => t.get().inner.clone(),
            Err(_) => return Err(PyTypeError::new_err("expected a tensor")),
        }
    };
    if t.shape.len() != shape.len() {
        return Err(TelvaxError::new_err(format!(
            "expected a tensor of rank {}, got shape {:?}",
            shape.len(),
            t.shape
        )));
    }
    for (got, want) in t.shape.iter().zip(shape) {
        if let Dim::Lit(n) = want {
            if *got as u64 != *n {
                return Err(TelvaxError::new_err(format!("expected shape {shape:?}, got {:?}", t.shape)));
            }
        }
    }
    // Same element type: share the buffer. Otherwise convert (a copy).
    if let Some(want) = arrow_elem(elem) {
        if t.array.data_type() != &want {
            let array: ArrayRef = cast(&t.array, &want).map_err(|e| TelvaxError::new_err(e.to_string()))?;
            return Ok(Value::Tensor(Arc::new(TensorVal::from_array(t.shape.clone(), array).map_err(runtime_err)?)));
        }
    }
    Ok(Value::Tensor(t))
}

/// Converts `obj` to a runtime value of static type `ty`.
pub fn py_to_value(interp: &Interp, obj: &Bound<'_, PyAny>, ty: &Ty) -> PyResult<Value> {
    match ty {
        Ty::Unit => Ok(Value::Unit),
        Ty::Prim(Prim::Bool) => Ok(Value::Bool(obj.extract()?)),
        Ty::Prim(Prim::Str) => Ok(Value::str(obj.extract::<String>()?)),
        Ty::Prim(p) if p.is_float() => Ok(Value::Float(obj.extract()?)),
        Ty::Prim(_) => {
            if obj.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err("expected an integer, got a bool"));
            }
            Ok(Value::Int(obj.extract()?))
        }
        Ty::Vec(inner) => {
            let items = obj
                .try_iter()
                .map_err(|_| PyTypeError::new_err("expected a sequence"))?
                .map(|item| py_to_value(interp, &item?, inner))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Value::vec(items))
        }
        Ty::Struct(name) => {
            let fields = interp
                .struct_fields(name)
                .ok_or_else(|| TelvaxError::new_err(format!("unknown class `{name}`")))?;
            let mut out = Vec::with_capacity(fields.len());
            for (fname, fty) in fields {
                let item = match obj.cast::<PyDict>() {
                    Ok(d) => d.get_item(fname)?,
                    Err(_) => obj.getattr(fname.as_str()).ok(),
                }
                .ok_or_else(|| TelvaxError::new_err(format!("`{name}` needs the field `{fname}`")))?;
                out.push((
                    fname.clone(),
                    py_to_value(interp, &item, fty).map_err(|e| TelvaxError::new_err(format!("field `{fname}`: {e}")))?,
                ));
            }
            Ok(Value::Struct(Arc::new(StructVal { name: name.clone(), fields: out })))
        }
        Ty::Tensor { elem, shape } => tensor_to_value(obj, elem, shape),
        // Same shapes as the JSON form: `"Unit"`, `{"One": value}`, `{"Many": [values..]}`.
        Ty::Enum(_) | Ty::Option(_) | Ty::Map(..) | Ty::Tuple(_) => {
            let json = py_to_json(obj)?;
            interp.value_from_json(ty, &json).map_err(TelvaxError::new_err)
        }
        Ty::Table(schema) => {
            let batch = match obj.cast::<Table>() {
                Ok(t) => t.get().inner.batch.clone(),
                Err(_) => import_batch(obj)?,
            };
            interp.table_from_batch(schema, &batch).map_err(runtime_err)
        }
        _ => infer(obj),
    }
}

/// Converts a Python object whose Telvax type is not statically known.
fn infer(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Unit)
    } else if obj.is_instance_of::<PyBool>() {
        Ok(Value::Bool(obj.extract()?))
    } else if obj.is_instance_of::<PyInt>() {
        Ok(Value::Int(obj.extract()?))
    } else if obj.is_instance_of::<PyFloat>() {
        Ok(Value::Float(obj.extract()?))
    } else if obj.is_instance_of::<PyString>() {
        Ok(Value::str(obj.extract::<String>()?))
    } else if let Ok(t) = obj.cast::<Tensor>() {
        Ok(Value::Tensor(t.get().inner.clone()))
    } else if let Ok(t) = obj.cast::<Table>() {
        Ok(Value::Table(t.get().inner.clone()))
    } else if let Ok(list) = obj.cast::<PyList>() {
        Ok(Value::vec(list.iter().map(|i| infer(&i)).collect::<PyResult<_>>()?))
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        Ok(Value::vec(tuple.iter().map(|i| infer(&i)).collect::<PyResult<_>>()?))
    } else if let Ok(d) = obj.cast::<PyDict>() {
        let mut fields = Vec::new();
        for (k, v) in d.iter() {
            fields.push((k.extract::<String>()?, infer(&v)?));
        }
        Ok(Value::Struct(Arc::new(StructVal { name: "Object".into(), fields })))
    } else if obj.hasattr("__arrow_c_array__")? {
        let array = import_capsules(obj)?;
        Ok(Value::Tensor(Arc::new(TensorVal::from_array(vec![array.len()], array).map_err(runtime_err)?)))
    } else {
        Err(PyTypeError::new_err(format!("cannot convert a `{}` to a Telvax value", obj.get_type().name()?)))
    }
}

/// Converts a runtime value to Python. `Err(Status)` results raise `StatusError`.
pub fn value_to_py(py: Python<'_>, v: Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        Value::Unit => py.None(),
        Value::Bool(b) => PyBool::new(py, b).to_owned().into_any().unbind(),
        Value::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        Value::Float(x) => x.into_pyobject(py)?.into_any().unbind(),
        Value::Str(s) => PyString::new(py, &s).into_any().unbind(),
        Value::Vec(items) => {
            let items = items.snapshot().into_iter().map(|i| value_to_py(py, i)).collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any().unbind()
        }
        Value::Struct(s) => {
            let d = PyDict::new(py);
            for (k, v) in &s.fields {
                d.set_item(k, value_to_py(py, v.clone())?)?;
            }
            d.into_any().unbind()
        }
        // A unit variant is its name; one with values is `{"Variant": value}` (one value) or
        // `{"Variant": [values..]}`, the same shape as the JSON form.
        // `None` is Python's `None`; `Some(x)` is just `x`.
        Value::Enum(e) if e.name == "Option" => match e.fields.first() {
            Some(v) => value_to_py(py, v.clone())?,
            None => py.None(),
        },
        Value::Map(m) => {
            let d = PyDict::new(py);
            for (k, v) in m.entries() {
                d.set_item(value_to_py(py, k.to_value())?, value_to_py(py, v)?)?;
            }
            d.into_any().unbind()
        }
        Value::Enum(e) => match e.fields.as_slice() {
            [] => PyString::new(py, &e.variant).into_any().unbind(),
            fields => {
                let d = PyDict::new(py);
                let payload = if let [one] = fields {
                    value_to_py(py, one.clone())?
                } else {
                    let items = fields.iter().map(|f| value_to_py(py, f.clone())).collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, items)?.into_any().unbind()
                };
                d.set_item(&e.variant, payload)?;
                d.into_any().unbind()
            }
        },
        Value::Tuple(items) => {
            let parts = items.iter().map(|i| value_to_py(py, i.clone())).collect::<PyResult<Vec<_>>>()?;
            pyo3::types::PyTuple::new(py, parts)?.into_any().unbind()
        }
        Value::Tensor(t) => Py::new(py, Tensor { inner: t })?.into_any(),
        Value::Table(t) => Py::new(py, Table { inner: t })?.into_any(),
        Value::Ok(v) => value_to_py(py, *v)?,
        Value::Err(e) => {
            return Err(match *e {
                Value::Status(code, msg) => {
                    let err = StatusError::new_err(msg.to_string());
                    err.value(py).setattr("status", code)?;
                    err
                }
                other => TelvaxError::new_err(format!("Err({other})")),
            });
        }
        Value::Status(code, msg) => {
            let d = PyDict::new(py);
            d.set_item("status", code)?;
            d.set_item("message", msg.to_string())?;
            d.into_any().unbind()
        }
        Value::Closure(_) | Value::Func(_) | Value::Native(_) => return Err(TelvaxError::new_err("this value cannot be converted to Python")),
    })
}

pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Json> {
    if obj.is_none() {
        Ok(Json::Null)
    } else if obj.is_instance_of::<PyBool>() {
        Ok(Json::Bool(obj.extract()?))
    } else if obj.is_instance_of::<PyInt>() {
        Ok(Json::from(obj.extract::<i64>()?))
    } else if obj.is_instance_of::<PyFloat>() {
        Ok(serde_json::Number::from_f64(obj.extract()?).map_or(Json::Null, Json::Number))
    } else if obj.is_instance_of::<PyString>() {
        Ok(Json::String(obj.extract()?))
    } else if let Ok(t) = obj.cast::<Tensor>() {
        Ok(Value::Tensor(t.get().inner.clone()).to_json())
    } else if let Ok(t) = obj.cast::<Table>() {
        Ok(t.get().inner.to_json())
    } else if let Ok(d) = obj.cast::<PyDict>() {
        let mut m = serde_json::Map::new();
        for (k, v) in d.iter() {
            m.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
        Ok(Json::Object(m))
    } else if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        Ok(Json::Array(obj.try_iter()?.map(|i| py_to_json(&i?)).collect::<PyResult<_>>()?))
    } else {
        Err(PyTypeError::new_err(format!("cannot convert a `{}` to JSON", obj.get_type().name()?)))
    }
}

pub fn json_to_py(py: Python<'_>, v: &Json) -> PyResult<Py<PyAny>> {
    Ok(match v {
        Json::Null => py.None(),
        Json::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        Json::Number(n) => match n.as_i64() {
            Some(i) => i.into_pyobject(py)?.into_any().unbind(),
            None => n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any().unbind(),
        },
        Json::String(s) => PyString::new(py, s).into_any().unbind(),
        Json::Array(a) => {
            let items = a.iter().map(|i| json_to_py(py, i)).collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any().unbind()
        }
        Json::Object(o) => {
            let d = PyDict::new(py);
            for (k, v) in o {
                d.set_item(k, json_to_py(py, v)?)?;
            }
            d.into_any().unbind()
        }
    })
}

#[allow(dead_code)]
fn _unused(_: &TableVal) {}
