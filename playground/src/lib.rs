//! The Novar playground engine: type-check and run programs inside a browser tab.
//!
//! Everything crosses the JS boundary as JSON strings so the page needs no glue code beyond
//! `JSON.parse`. Programs run in the interpreter with a step limit, with the offline model
//! stand-ins and without file or network access.

use std::io::Write;
use std::sync::{Arc, Mutex};

use novar_lsp::analysis::Snapshot;
use novar_runtime::Interp;
use serde_json::{Value as Json, json};
use wasm_bindgen::prelude::*;

/// How many evaluation steps a run may take before it is stopped.
const STEP_LIMIT: u64 = 20_000_000;
/// Longest program output kept, in bytes.
const OUTPUT_LIMIT: usize = 256 * 1024;

struct Sink(Arc<Mutex<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut out = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if out.len() < OUTPUT_LIMIT {
            let room = OUTPUT_LIMIT - out.len();
            out.extend_from_slice(&buf[..buf.len().min(room)]);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn diagnostics_json(snap: &Snapshot) -> Json {
    Json::Array(
        snap.diagnostics
            .iter()
            .map(|d| {
                let r = snap.index.range(&d.span);
                json!({
                    "severity": if d.is_error() { "error" } else { "warning" },
                    "message": d.message,
                    "notes": d.notes,
                    "startLine": r.start.line, "startCol": r.start.character,
                    "endLine": r.end.line, "endCol": r.end.character,
                })
            })
            .collect(),
    )
}

/// Type-checks `source`. Returns `{ok, diagnostics, rendered}`.
#[wasm_bindgen]
pub fn check(source: &str) -> String {
    let snap = Snapshot::analyze(source.to_string());
    let ok = !snap.diagnostics.iter().any(|d| d.is_error());
    json!({
        "ok": ok,
        "diagnostics": diagnostics_json(&snap),
        "rendered": novar_diagnostics::render("playground.nv", source, &snap.diagnostics, false),
    })
    .to_string()
}

/// The functions of `source` that can be run from the page: `[{name, params: [{name, ty}], ret}]`.
#[wasm_bindgen]
pub fn functions(source: &str) -> String {
    let snap = Snapshot::analyze(source.to_string());
    let list: Vec<Json> = snap
        .hir
        .iter()
        .flat_map(|h| h.fns.iter())
        .map(|f| {
            json!({
                "name": f.name,
                "params": f.params.iter().map(|p| json!({"name": p.name, "ty": p.ty.to_string()})).collect::<Vec<_>>(),
                "ret": f.ret.to_string(),
            })
        })
        .collect();
    Json::Array(list).to_string()
}

fn failure(message: impl Into<String>, snap: &Snapshot, output: &str) -> String {
    json!({"ok": false, "error": message.into(), "output": output, "diagnostics": diagnostics_json(snap)}).to_string()
}

/// Runs `function` of `source` with JSON `args` (an array). Returns
/// `{ok, result, output, error, diagnostics}`.
#[wasm_bindgen]
pub fn run(source: &str, function: &str, args: &str) -> String {
    let snap = Snapshot::analyze(source.to_string());
    if snap.diagnostics.iter().any(|d| d.is_error()) {
        return failure("the program has errors", &snap, "");
    }
    let Some(program) = snap.hir.clone() else {
        return failure("the program could not be analysed", &snap, "");
    };
    let args: Vec<Json> = match serde_json::from_str(args) {
        Ok(Json::Array(a)) => a,
        Ok(_) => return failure("arguments must be a JSON array", &snap, ""),
        Err(e) => return failure(format!("arguments are not valid JSON: {e}"), &snap, ""),
    };

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let output =
        || String::from_utf8_lossy(&buffer.lock().unwrap_or_else(|e| e.into_inner())).into_owned();
    let interp = match Interp::new(program) {
        Ok(i) => i,
        Err(e) => return failure(e.to_string(), &snap, &output()),
    };
    interp.set_output(Sink(buffer.clone()));
    interp.set_step_limit(STEP_LIMIT);

    let Some((params, _)) = interp.signature(function) else {
        return failure(format!("no function `{function}`"), &snap, &output());
    };
    if params.len() != args.len() {
        return failure(
            format!(
                "`{function}` takes {} argument(s), got {}",
                params.len(),
                args.len()
            ),
            &snap,
            &output(),
        );
    }
    let mut values = Vec::new();
    for ((name, ty), arg) in params.iter().zip(&args) {
        match interp.value_from_json(ty, arg) {
            Ok(v) => values.push(v),
            Err(e) => return failure(format!("argument `{name}`: {e}"), &snap, &output()),
        }
    }
    match interp.call(function, values) {
        Ok(v) => json!({"ok": true, "result": v.to_json(), "display": v.to_string(), "output": output(), "diagnostics": []}).to_string(),
        Err(e) => failure(e.to_string(), &snap, &output()),
    }
}

/// Markdown hover text at (`line`, `character`), or an empty string.
#[wasm_bindgen]
pub fn hover(source: &str, line: u32, character: u32) -> String {
    let snap = Snapshot::analyze(source.to_string());
    let offset = snap.index.offset(lsp_position(line, character));
    snap.hover(offset).map(|(text, _)| text).unwrap_or_default()
}

/// Completion labels at (`line`, `character`): `[{label, kind, detail}]`.
#[wasm_bindgen]
pub fn complete(source: &str, line: u32, character: u32) -> String {
    let snap = Snapshot::analyze(source.to_string());
    let offset = snap.index.offset(lsp_position(line, character));
    let items: Vec<Json> = snap
        .completions(offset)
        .into_iter()
        .map(|c| json!({"label": c.label, "kind": format!("{:?}", c.kind).to_lowercase(), "detail": c.detail}))
        .collect();
    Json::Array(items).to_string()
}

fn lsp_position(line: u32, character: u32) -> lsp_types_position::Position {
    lsp_types_position::Position { line, character }
}

mod lsp_types_position {
    pub use novar_lsp::position::Position;
}
