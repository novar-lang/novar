//! Runtime values and their JSON conversion.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use novar_hir::hir::{self, LocalId};
use novar_hir::{Prim, Ty};
use serde_json::{Map, Number, Value as Json};

pub type Env = HashMap<LocalId, Value>;

#[derive(Debug, Clone)]
pub struct StructVal {
    pub name: String,
    pub fields: Vec<(String, Value)>,
}

/// Shared, mutable element storage of a `Vec`.
#[derive(Debug, Default)]
pub struct VecVal(RwLock<Vec<Value>>);

impl VecVal {
    pub fn new(items: Vec<Value>) -> Self {
        Self(RwLock::new(items))
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vec<Value>> {
        self.0.read().unwrap_or_else(|e| e.into_inner())
    }

    pub fn len(&self) -> usize {
        self.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    pub fn get(&self, i: usize) -> Option<Value> {
        self.read().get(i).cloned()
    }

    /// Replaces element `i`; `false` when out of bounds.
    pub fn set(&self, i: usize, v: Value) -> bool {
        match self.0.write().unwrap_or_else(|e| e.into_inner()).get_mut(i) {
            Some(slot) => {
                *slot = v;
                true
            }
            None => false,
        }
    }

    /// A copy of the current elements.
    pub fn snapshot(&self) -> Vec<Value> {
        self.read().clone()
    }
}

/// A dense tensor: a flat Arrow array (row-major) plus its shape. Cloning shares the buffer.
#[derive(Debug, Clone)]
pub struct TensorVal {
    pub shape: Vec<usize>,
    pub array: arrow::array::ArrayRef,
}

/// A columnar table; its columns follow the fields of the struct named `schema`.
#[derive(Debug, Clone)]
pub struct TableVal {
    pub schema: String,
    pub batch: arrow::record_batch::RecordBatch,
}

#[derive(Debug)]
pub struct Closure {
    pub params: Vec<LocalId>,
    pub body: hir::Expr,
    pub env: Env,
}

#[derive(Debug, Clone)]
pub enum Value {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    /// Vectors are reference values: copies of a `Vec` share the same elements.
    Vec(Arc<VecVal>),
    Struct(Arc<StructVal>),
    Tensor(Arc<TensorVal>),
    Table(Arc<TableVal>),
    Ok(Box<Value>),
    Err(Box<Value>),
    Status(u16, Arc<str>),
    Closure(Arc<Closure>),
    Native(Native),
}

/// Objects provided by the runtime's standard library.
#[derive(Debug, Clone)]
pub enum Native {
    Embedder { dim: usize },
    Llm { model: Arc<str> },
    Db(Arc<Db>),
    Query(Arc<Query>),
}

#[derive(Debug)]
pub struct Doc {
    pub id: String,
    pub text: String,
    pub vector: Vec<f32>,
}

/// In-memory stand-in for a pgvector database.
#[derive(Debug, Default)]
pub struct Db {
    pub url: String,
    pub collections: Mutex<HashMap<String, Vec<Doc>>>,
}

impl Db {
    pub fn insert(&self, collection: &str, id: &str, text: &str, vector: Vec<f32>) {
        let mut cols = self.collections.lock().unwrap_or_else(|e| e.into_inner());
        let docs = cols.entry(collection.to_string()).or_default();
        // Upsert by id, like a primary-key conflict would.
        docs.retain(|d| d.id != id);
        docs.push(Doc {
            id: id.to_string(),
            text: text.to_string(),
            vector,
        });
    }
}

#[derive(Debug, Clone)]
pub struct Query {
    pub db: Arc<Db>,
    pub collection: String,
    pub vector: Option<Vec<f32>>,
    pub limit: usize,
}

impl Value {
    pub fn str(s: impl Into<Arc<str>>) -> Value {
        Value::Str(s.into())
    }

    pub fn vec(items: Vec<Value>) -> Value {
        Value::Vec(Arc::new(VecVal::new(items)))
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "()",
            Value::Bool(_) => "bool",
            Value::Int(_) => "integer",
            Value::Float(_) => "float",
            Value::Str(_) => "String",
            Value::Vec(_) => "Vec",
            Value::Struct(_) => "struct",
            Value::Tensor(_) => "Tensor",
            Value::Table(_) => "Table",
            Value::Ok(_) | Value::Err(_) => "Result",
            Value::Status(..) => "Status",
            Value::Closure(_) => "function",
            Value::Native(_) => "object",
        }
    }

    /// Structural equality; closures and native objects are never equal.
    pub fn equals(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => *a as f64 == *b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Vec(a), Value::Vec(b)) => {
                let (a, b) = (a.snapshot(), b.snapshot());
                a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.equals(y))
            }
            (Value::Struct(a), Value::Struct(b)) => {
                a.name == b.name
                    && a.fields.len() == b.fields.len()
                    && a.fields
                        .iter()
                        .zip(&b.fields)
                        .all(|((n1, v1), (n2, v2))| n1 == n2 && v1.equals(v2))
            }
            (Value::Tensor(a), Value::Tensor(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => a.schema == b.schema && a.batch == b.batch,
            (Value::Ok(a), Value::Ok(b)) | (Value::Err(a), Value::Err(b)) => a.equals(b),
            (Value::Status(a, m), Value::Status(b, n)) => a == b && m == n,
            _ => false,
        }
    }

    pub fn to_json(&self) -> Json {
        match self {
            Value::Unit => Json::Null,
            Value::Bool(b) => Json::Bool(*b),
            Value::Int(i) => Json::from(*i),
            Value::Float(f) => Number::from_f64(*f).map_or(Json::Null, Json::Number),
            Value::Str(s) => Json::String(s.to_string()),
            Value::Vec(items) => Json::Array(items.snapshot().iter().map(Value::to_json).collect()),
            Value::Struct(s) => {
                let mut m = Map::new();
                for (k, v) in &s.fields {
                    m.insert(k.clone(), v.to_json());
                }
                Json::Object(m)
            }
            Value::Tensor(t) => {
                let single = matches!(t.dtype(), arrow::datatypes::DataType::Float32);
                tensor_json(&t.shape, &t.to_f64_vec(), single)
            }
            Value::Table(t) => t.to_json(),
            Value::Ok(v) => v.to_json(),
            Value::Err(v) => serde_json::json!({ "error": v.to_json() }),
            Value::Status(code, msg) => serde_json::json!({ "status": code, "message": &**msg }),
            Value::Closure(_) => Json::String("<function>".into()),
            Value::Native(_) => Json::String("<object>".into()),
        }
    }
}

fn tensor_json(shape: &[usize], data: &[f64], single: bool) -> Json {
    match shape.split_first() {
        None => {
            let x = data[0];
            // f32 values print with their shortest decimal form, not widened f64 noise.
            let x = if single {
                format!("{}", x as f32).parse().unwrap_or(x)
            } else {
                x
            };
            Number::from_f64(x).map_or(Json::Null, Json::Number)
        }
        Some((&n, rest)) => {
            let chunk: usize = rest.iter().product();
            Json::Array(
                (0..n)
                    .map(|i| tensor_json(rest, &data[i * chunk..(i + 1) * chunk], single))
                    .collect(),
            )
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => f.write_str("()"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::Str(s) => f.write_str(s),
            Value::Vec(items) => {
                f.write_str("[")?;
                for (i, v) in items.snapshot().iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
            Value::Struct(s) => {
                write!(f, "{} {{ ", s.name)?;
                for (i, (k, v)) in s.fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                f.write_str(" }")
            }
            Value::Tensor(t) => write!(f, "Tensor{:?}", t.shape),
            Value::Table(t) => write!(f, "Table[{} rows]", t.rows()),
            Value::Ok(v) => write!(f, "Ok({v})"),
            Value::Err(v) => write!(f, "Err({v})"),
            Value::Status(code, msg) => write!(f, "{code}: {msg}"),
            Value::Closure(_) => f.write_str("<function>"),
            Value::Native(_) => f.write_str("<object>"),
        }
    }
}

/// Converts request JSON into a runtime value of static type `ty`.
pub fn from_json(
    ty: &Ty,
    json: &Json,
    structs: &HashMap<String, Vec<(String, Ty)>>,
) -> Result<Value, String> {
    let bad = |what: &str| format!("expected {what}, found {json}");
    match ty {
        Ty::Prim(Prim::Bool) => json.as_bool().map(Value::Bool).ok_or_else(|| bad("bool")),
        Ty::Prim(Prim::Str) => json.as_str().map(Value::str).ok_or_else(|| bad("String")),
        Ty::Prim(p) if p.is_float() => json
            .as_f64()
            .map(Value::Float)
            .ok_or_else(|| bad("a number")),
        Ty::Prim(_) => json
            .as_i64()
            .map(Value::Int)
            .ok_or_else(|| bad("an integer")),
        Ty::Vec(inner) => {
            let items = json.as_array().ok_or_else(|| bad("an array"))?;
            items
                .iter()
                .map(|j| from_json(inner, j, structs))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::vec)
        }
        Ty::Tensor { elem, shape } => {
            let mut dims = Vec::new();
            let mut data = Vec::new();
            flatten(json, shape.len(), 0, &mut dims, &mut data)?;
            let array: arrow::array::ArrayRef = match &**elem {
                Ty::Prim(Prim::F64) => Arc::new(arrow::array::Float64Array::from(data)),
                Ty::Prim(Prim::I32) => Arc::new(arrow::array::Int32Array::from(
                    data.iter().map(|x| *x as i32).collect::<Vec<_>>(),
                )),
                Ty::Prim(Prim::I64) => Arc::new(arrow::array::Int64Array::from(
                    data.iter().map(|x| *x as i64).collect::<Vec<_>>(),
                )),
                _ => Arc::new(arrow::array::Float32Array::from(
                    data.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                )),
            };
            TensorVal::from_array(dims, array)
                .map(|t| Value::Tensor(Arc::new(t)))
                .map_err(|e| e.message)
        }
        Ty::Table(name) => {
            let fields = structs
                .get(name)
                .ok_or_else(|| format!("unknown struct `{name}`"))?;
            let items = json.as_array().ok_or_else(|| bad("an array of rows"))?;
            let rows = items
                .iter()
                .map(|j| from_json(&Ty::Struct(name.clone()), j, structs))
                .collect::<Result<Vec<_>, _>>()?;
            crate::table::rows_to_batch(fields, &rows)
                .map(|batch| {
                    Value::Table(Arc::new(TableVal {
                        schema: name.clone(),
                        batch,
                    }))
                })
                .map_err(|e| e.message)
        }
        Ty::Struct(name) => {
            let fields = structs
                .get(name)
                .ok_or_else(|| format!("unknown struct `{name}`"))?;
            let obj = json.as_object().ok_or_else(|| bad("an object"))?;
            let mut out = Vec::new();
            for (fname, fty) in fields {
                let v = obj
                    .get(fname)
                    .ok_or_else(|| format!("missing field `{fname}`"))?;
                out.push((
                    fname.clone(),
                    from_json(fty, v, structs).map_err(|e| format!("{fname}: {e}"))?,
                ));
            }
            Ok(Value::Struct(Arc::new(StructVal {
                name: name.clone(),
                fields: out,
            })))
        }
        _ => Ok(generic_from_json(json)),
    }
}

fn flatten(
    json: &Json,
    rank: usize,
    depth: usize,
    dims: &mut Vec<usize>,
    data: &mut Vec<f64>,
) -> Result<(), String> {
    if depth == rank {
        let x = json
            .as_f64()
            .ok_or_else(|| format!("expected a number, found {json}"))?;
        data.push(x);
        return Ok(());
    }
    let items = json
        .as_array()
        .ok_or_else(|| format!("expected an array, found {json}"))?;
    if dims.len() == depth {
        dims.push(items.len());
    } else if dims[depth] != items.len() {
        return Err("tensor rows have different lengths".into());
    }
    items
        .iter()
        .try_for_each(|j| flatten(j, rank, depth + 1, dims, data))
}

fn generic_from_json(json: &Json) -> Value {
    match json {
        Json::Null => Value::Unit,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => n
            .as_i64()
            .map_or_else(|| Value::Float(n.as_f64().unwrap_or(f64::NAN)), Value::Int),
        Json::String(s) => Value::str(s.as_str()),
        Json::Array(a) => Value::vec(a.iter().map(generic_from_json).collect()),
        Json::Object(o) => Value::Struct(Arc::new(StructVal {
            name: "Object".into(),
            fields: o
                .iter()
                .map(|(k, v)| (k.clone(), generic_from_json(v)))
                .collect(),
        })),
    }
}
