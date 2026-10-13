//! Runtime implementations of the std items declared in `novar_hir::stdlib`.
//!
//! External services are replaced by deterministic in-process stand-ins (hashed bag-of-words
//! embeddings, an in-memory vector store, a canned LLM) so programs run without network access.

use std::sync::Arc;

use crate::interp::{Interp, RuntimeError};
use crate::value::{Db, Native, Query, StructVal, TensorVal, Value};
use novar_hir::stdlib::EMBEDDING_DIM;

type Res = Result<Value, RuntimeError>;

fn err<T>(msg: impl Into<String>) -> Result<T, RuntimeError> {
    Err(RuntimeError::new(msg))
}

fn want_str(v: &Value) -> Result<Arc<str>, RuntimeError> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => err(format!("expected a String, found {}", other.type_name())),
    }
}

fn want_vec(v: &Value) -> Result<Arc<Vec<Value>>, RuntimeError> {
    match v {
        Value::Vec(items) => Ok(items.clone()),
        other => err(format!("expected a Vec, found {}", other.type_name())),
    }
}

fn want_usize(v: &Value) -> Result<usize, RuntimeError> {
    match v {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        other => err(format!("expected a non-negative integer, found {other}")),
    }
}

/// Deterministic embedding: hashed bag of lowercase words, L2-normalised. Texts that share words
/// get a higher cosine similarity, which is enough to exercise retrieval pipelines offline.
pub fn embed_text(text: &str) -> Vec<f32> {
    let dim = EMBEDDING_DIM as usize;
    let mut v = vec![0f32; dim];
    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in word.to_lowercase().bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
        let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
        v[(h % dim as u64) as usize] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn tensor(data: Vec<f32>) -> Value {
    Value::Tensor(Arc::new(TensorVal {
        shape: vec![data.len()],
        data,
    }))
}

fn status(code: u16, args: &[Value]) -> Res {
    Ok(Value::Status(code, want_str(&args[0])?))
}

pub(crate) fn call_function(interp: &Interp, path: &str, args: Vec<Value>) -> Res {
    match path {
        "std::text::clean_whitespace" => Ok(Value::str(
            want_str(&args[0])?
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        )),
        "std::text::trim" => Ok(Value::str(want_str(&args[0])?.trim())),
        "std::text::lowercase" => Ok(Value::str(want_str(&args[0])?.to_lowercase())),
        "std::text::uppercase" => Ok(Value::str(want_str(&args[0])?.to_uppercase())),
        "std::text::truncate" => {
            let n = want_usize(&args[1])?;
            Ok(Value::str(
                want_str(&args[0])?.chars().take(n).collect::<String>(),
            ))
        }
        "std::text::split" => {
            let s = want_str(&args[0])?;
            let sep = want_str(&args[1])?;
            if sep.is_empty() {
                return err("`split` separator must not be empty");
            }
            Ok(Value::vec(s.split(&*sep).map(Value::str).collect()))
        }
        "std::string::join" => {
            let items = want_vec(&args[0])?;
            let sep = want_str(&args[1])?;
            Ok(Value::str(
                items
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(&sep),
            ))
        }
        "std::iter::map" => {
            let items = want_vec(&args[0])?;
            let out = items
                .iter()
                .map(|v| interp.apply(&args[1], vec![v.clone()]))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::vec(out))
        }
        "std::iter::filter" => {
            let items = want_vec(&args[0])?;
            let mut out = Vec::new();
            for v in items.iter() {
                match interp.apply(&args[1], vec![v.clone()])? {
                    Value::Bool(true) => out.push(v.clone()),
                    Value::Bool(false) => {}
                    other => {
                        return err(format!("filter predicate returned {}", other.type_name()));
                    }
                }
            }
            Ok(Value::vec(out))
        }
        "std::iter::take" => {
            let n = want_usize(&args[1])?;
            Ok(Value::vec(
                want_vec(&args[0])?.iter().take(n).cloned().collect(),
            ))
        }
        "std::iter::count" => Ok(Value::Int(want_vec(&args[0])?.len() as i64)),
        "std::collect" => Ok(args[0].clone()),
        "std::http::Status::BadRequest" => status(400, &args),
        "std::http::Status::Unauthorized" => status(401, &args),
        "std::http::Status::Forbidden" => status(403, &args),
        "std::http::Status::NotFound" => status(404, &args),
        "std::http::Status::Unprocessable" => status(422, &args),
        "std::http::Status::Internal" => status(500, &args),
        "std::ai::embeddings::OllamaEmbedder::new" => {
            want_str(&args[0])?;
            Ok(Value::Native(Native::Embedder {
                dim: EMBEDDING_DIM as usize,
            }))
        }
        "std::ai::llm::OpenAIClient::new" => Ok(Value::Native(Native::Llm {
            model: want_str(&args[0])?,
        })),
        "std::db::pgvector::PgVectorClient::connect" => Ok(Value::Native(Native::Db(
            interp.db_for(&want_str(&args[0])?),
        ))),
        other => err(format!(
            "std function `{other}` is not implemented by the runtime"
        )),
    }
}

/// `receiver.name(args)`. For vector-store query builders the piped query is `args[0]`.
pub(crate) fn call_method(receiver: &Value, name: &str, args: Vec<Value>) -> Res {
    match receiver {
        Value::Str(s) => match name {
            "is_empty" => Ok(Value::Bool(s.is_empty())),
            "len" => Ok(Value::Int(s.len() as i64)),
            "trim" => Ok(Value::str(s.trim())),
            "to_lowercase" => Ok(Value::str(s.to_lowercase())),
            "to_uppercase" => Ok(Value::str(s.to_uppercase())),
            _ => err(format!("String has no method `{name}`")),
        },
        Value::Vec(v) => match name {
            "is_empty" => Ok(Value::Bool(v.is_empty())),
            "len" => Ok(Value::Int(v.len() as i64)),
            _ => err(format!("Vec has no method `{name}`")),
        },
        Value::Native(Native::Embedder { .. }) if name == "embed" => {
            Ok(tensor(embed_text(&want_str(&args[0])?)))
        }
        Value::Native(Native::Llm { model }) if name == "complete" => {
            let prompt = want_str(&args[0])?;
            let question = prompt
                .lines()
                .rev()
                .find_map(|l| l.strip_prefix("User Question: "))
                .unwrap_or_else(|| prompt.lines().last().unwrap_or(""));
            let context = prompt.lines().filter(|l| !l.is_empty()).count();
            Ok(Value::str(format!(
                "[{model} offline stand-in] answer to \"{question}\" from {context} prompt line(s)"
            )))
        }
        Value::Native(Native::Db(db)) => db_method(db, name, args),
        other => err(format!("{} has no method `{name}`", other.type_name())),
    }
}

fn db_method(db: &Arc<Db>, name: &str, args: Vec<Value>) -> Res {
    match name {
        "search_collection" => Ok(Value::Native(Native::Query(Arc::new(Query {
            db: db.clone(),
            collection: want_str(&args[0])?.to_string(),
            vector: None,
            limit: 10,
        })))),
        "with_vector" => {
            let mut q = want_query(&args[0])?;
            match &args[1] {
                Value::Tensor(t) => q.vector = Some(t.data.clone()),
                other => return err(format!("expected a Tensor, found {}", other.type_name())),
            }
            Ok(Value::Native(Native::Query(Arc::new(q))))
        }
        "limit" => {
            let mut q = want_query(&args[0])?;
            q.limit = want_usize(&args[1])?;
            Ok(Value::Native(Native::Query(Arc::new(q))))
        }
        "execute" => execute(&want_query(&args[0])?),
        "insert" => {
            let Value::Tensor(t) = &args[3] else {
                return err("insert expects a Tensor vector");
            };
            db.insert(
                &want_str(&args[0])?,
                &want_str(&args[1])?,
                &want_str(&args[2])?,
                t.data.clone(),
            );
            Ok(Value::Unit)
        }
        _ => err(format!("PgVectorClient has no method `{name}`")),
    }
}

fn want_query(v: &Value) -> Result<Query, RuntimeError> {
    match v {
        Value::Native(Native::Query(q)) => Ok((**q).clone()),
        other => err(format!(
            "expected a VectorQuery, found {}",
            other.type_name()
        )),
    }
}

fn execute(q: &Query) -> Res {
    let Some(vector) = &q.vector else {
        return err("vector query has no `with_vector(..)`");
    };
    let cols = q.db.collections.lock().unwrap_or_else(|e| e.into_inner());
    let mut scored: Vec<(f64, &crate::value::Doc)> = cols
        .get(&q.collection)
        .map(|docs| {
            docs.iter()
                .map(|d| {
                    let dot: f32 = d.vector.iter().zip(vector).map(|(a, b)| a * b).sum();
                    (f64::from(dot), d)
                })
                .collect()
        })
        .unwrap_or_default();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    let rows = scored
        .into_iter()
        .take(q.limit)
        .map(|(score, d)| {
            Value::Struct(Arc::new(StructVal {
                name: "Row".into(),
                fields: vec![
                    ("id".into(), Value::str(d.id.as_str())),
                    ("text".into(), Value::str(d.text.as_str())),
                    ("score".into(), Value::Float(score)),
                ],
            }))
        })
        .collect();
    Ok(Value::vec(rows))
}
