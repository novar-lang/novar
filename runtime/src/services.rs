//! Real backends for the std AI services: Ollama embeddings, OpenAI-compatible chat completions
//! and pgvector. Enabled with `Interp::set_live(true)` (`telvax serve --live`).

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{Value as Json, json};

use crate::interp::RuntimeError;

fn err<T>(msg: impl Into<String>) -> Result<T, RuntimeError> {
    Err(RuntimeError::new(msg))
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(120)))
        .http_status_as_error(false)
        .build()
        .into()
}

/// Sends `body` as JSON and returns the decoded JSON reply, turning HTTP errors into messages.
fn post_json(url: &str, bearer: Option<&str>, body: &Json) -> Result<Json, RuntimeError> {
    let mut req = agent().post(url);
    if let Some(key) = bearer {
        req = req.header("Authorization", &format!("Bearer {key}"));
    }
    let mut resp = req
        .send_json(body)
        .map_err(|e| RuntimeError::new(format!("request to {url} failed: {e}")))?;
    let status = resp.status().as_u16();
    let reply: Json = resp.body_mut().read_json().map_err(|e| {
        RuntimeError::new(format!(
            "{url} returned an unreadable reply (HTTP {status}): {e}"
        ))
    })?;
    if !(200..300).contains(&status) {
        let detail = reply["error"]["message"]
            .as_str()
            .or_else(|| reply["error"].as_str())
            .unwrap_or("no details");
        return err(format!("{url} answered HTTP {status}: {detail}"));
    }
    Ok(reply)
}

/// `POST {host}/api/embed` (Ollama). The reply must have exactly `dim` numbers.
pub(crate) fn ollama_embed(
    host: &str,
    model: &str,
    text: &str,
    dim: usize,
) -> Result<Vec<f32>, RuntimeError> {
    let reply = post_json(
        &format!("{}/api/embed", host.trim_end_matches('/')),
        None,
        &json!({ "model": model, "input": text }),
    )?;
    let vector = reply["embeddings"][0]
        .as_array()
        .ok_or_else(|| RuntimeError::new("Ollama reply has no `embeddings[0]`"))?;
    let out: Vec<f32> = vector
        .iter()
        .filter_map(|x| x.as_f64())
        .map(|x| x as f32)
        .collect();
    if out.len() != vector.len() {
        return err("Ollama returned non-numeric embedding values");
    }
    if out.len() != dim {
        return err(format!(
            "model `{model}` returned {} dimensions but the type requires {dim}",
            out.len()
        ));
    }
    Ok(out)
}

/// `POST {base}/chat/completions` (OpenAI and compatible servers).
pub(crate) fn openai_complete(
    base: &str,
    key: &str,
    model: &str,
    prompt: &str,
) -> Result<String, RuntimeError> {
    let reply = post_json(
        &format!("{}/chat/completions", base.trim_end_matches('/')),
        Some(key),
        &json!({ "model": model, "messages": [{ "role": "user", "content": prompt }] }),
    )?;
    reply["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| RuntimeError::new("completion reply has no `choices[0].message.content`"))
}

/// One pgvector connection; each collection is a table `(id, text, embedding vector(dim))`.
pub struct PgStore {
    client: Mutex<postgres::Client>,
    ready: Mutex<HashSet<String>>,
    dim: usize,
}

impl std::fmt::Debug for PgStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PgStore")
    }
}

/// Collection names become table names, so only plain identifiers are accepted.
fn table_name(collection: &str) -> Result<&str, RuntimeError> {
    let ok = !collection.is_empty()
        && collection.len() <= 63
        && collection
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !collection.starts_with(|c: char| c.is_ascii_digit());
    if ok {
        Ok(collection)
    } else {
        err(format!(
            "`{collection}` is not a valid collection name (letters, digits and `_` only)"
        ))
    }
}

fn vector_literal(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

impl PgStore {
    pub(crate) fn connect(url: &str, dim: usize) -> Result<Self, RuntimeError> {
        let mut client = postgres::Client::connect(url, postgres::NoTls)
            .map_err(|e| RuntimeError::new(format!("cannot connect to PostgreSQL: {e}")))?;
        client
            .batch_execute("CREATE EXTENSION IF NOT EXISTS vector")
            .map_err(|e| {
                RuntimeError::new(format!("the pgvector extension is not available: {e}"))
            })?;
        Ok(Self {
            client: Mutex::new(client),
            ready: Mutex::new(HashSet::new()),
            dim,
        })
    }

    fn ensure(&self, client: &mut postgres::Client, table: &str) -> Result<(), RuntimeError> {
        let mut ready = self.ready.lock().unwrap_or_else(|e| e.into_inner());
        if ready.contains(table) {
            return Ok(());
        }
        client
            .batch_execute(&format!(
                "CREATE TABLE IF NOT EXISTS \"{table}\" (id text PRIMARY KEY, text text NOT NULL, embedding vector({}) NOT NULL)",
                self.dim
            ))
            .map_err(|e| RuntimeError::new(format!("cannot create table `{table}`: {e}")))?;
        ready.insert(table.to_string());
        Ok(())
    }

    pub(crate) fn insert(
        &self,
        collection: &str,
        id: &str,
        text: &str,
        vector: &[f32],
    ) -> Result<(), RuntimeError> {
        let table = table_name(collection)?;
        if vector.len() != self.dim {
            return err(format!(
                "vectors must have {} dimensions, got {}",
                self.dim,
                vector.len()
            ));
        }
        let mut client = self.client.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure(&mut client, table)?;
        client
            .execute(
                &format!(
                    "INSERT INTO \"{table}\" (id, text, embedding) VALUES ($1, $2, $3::text::vector) \
                     ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding"
                ),
                &[&id, &text, &vector_literal(vector)],
            )
            .map(|_| ())
            .map_err(|e| RuntimeError::new(format!("insert failed: {e}")))
    }

    /// Nearest neighbours by cosine distance, as `(id, text, similarity)`.
    pub(crate) fn search(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(String, String, f64)>, RuntimeError> {
        let table = table_name(collection)?;
        let mut client = self.client.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure(&mut client, table)?;
        let rows = client
            .query(
                &format!(
                    "SELECT id, text, (1 - (embedding <=> $1::text::vector))::float8 AS score FROM \"{table}\" \
                     ORDER BY embedding <=> $1::text::vector, id LIMIT $2::bigint"
                ),
                &[&vector_literal(vector), &(limit as i64)],
            )
            .map_err(|e| RuntimeError::new(format!("search failed: {e}")))?;
        Ok(rows
            .iter()
            .map(|r| (r.get(0), r.get(1), r.get(2)))
            .collect())
    }
}
