//! The Telvax playground engine: type-check and run programs inside a browser tab.
//!
//! Everything crosses the JS boundary as JSON strings so the page needs no glue code beyond
//! `JSON.parse`. Programs run in the interpreter with a step limit, with the offline model
//! stand-ins and without file or network access.

use std::cell::RefCell;
use std::io::Write;
use std::sync::{Arc, Mutex};

use serde_json::{Value as Json, json};
use telvax_lsp::analysis::Snapshot;
use telvax_repl::{Outcome, Session};
use telvax_runtime::{Caps, Interp};
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
        "rendered": telvax_diagnostics::render("playground.tvx", source, &snap.diagnostics, false),
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

// ---- the console: an interactive session that lasts as long as the page ---------------------

thread_local! {
    static CONSOLE: RefCell<Session> = RefCell::new(new_console());
}

fn new_console() -> Session {
    let mut session = Session::new();
    // The browser has no files, environment or network.
    session.set_caps(Caps::parse("none").unwrap_or(Caps::ALL));
    session.set_step_limit(Some(STEP_LIMIT));
    session
}

fn outcome_json(out: &Outcome) -> String {
    let mut output = out.output.clone();
    if output.len() > OUTPUT_LIMIT {
        let mut end = OUTPUT_LIMIT;
        while !output.is_char_boundary(end) {
            end -= 1;
        }
        output.truncate(end);
    }
    json!({
        "output": output,
        "value": out.value.as_ref().map(|v| json!({"text": v.text, "ty": v.ty})),
        "defined": out.defined,
        "notes": out.notes,
        "error": out.error,
    })
    .to_string()
}

/// Runs one input of the console. Returns `{output, value: {text, ty} | null, defined, notes, error}`.
#[wasm_bindgen]
pub fn repl_eval(input: &str) -> String {
    CONSOLE.with(|c| outcome_json(&c.borrow_mut().eval(input)))
}

/// Enters the definitions and variables of `source` (the program in the editor).
#[wasm_bindgen]
pub fn repl_load(source: &str) -> String {
    CONSOLE.with(|c| outcome_json(&c.borrow_mut().load(source)))
}

/// The type of an expression, without running it: `{ok, type}` or `{ok: false, error}`.
#[wasm_bindgen]
pub fn repl_type(expression: &str) -> String {
    CONSOLE
        .with(|c| match c.borrow_mut().type_of(expression) {
            Ok(ty) => json!({"ok": true, "type": ty}),
            Err(error) => json!({"ok": false, "error": error}),
        })
        .to_string()
}

/// The variables of the console: `[{name, ty, value}]`.
#[wasm_bindgen]
pub fn repl_vars() -> String {
    CONSOLE.with(|c| {
        let vars: Vec<Json> = c
            .borrow()
            .variables()
            .into_iter()
            .map(|(name, ty, value)| json!({"name": name, "ty": ty, "value": value}))
            .collect();
        Json::Array(vars).to_string()
    })
}

/// The functions, types and the like defined in the console.
#[wasm_bindgen]
pub fn repl_defs() -> String {
    CONSOLE.with(|c| json!(c.borrow().definitions()).to_string())
}

/// Forgets everything the console holds.
#[wasm_bindgen]
pub fn repl_reset() {
    CONSOLE.with(|c| c.borrow_mut().reset());
}

/// Whether `input` can be run as it is, or the person is in the middle of a block or a bracket.
#[wasm_bindgen]
pub fn repl_is_complete(input: &str) -> bool {
    Session::is_complete(input)
}

/// What Tab can complete the word before byte `cursor` to: `{start, items: [{label, detail}]}`.
#[wasm_bindgen]
pub fn repl_complete(input: &str, cursor: u32) -> String {
    CONSOLE.with(|c| {
        let (start, found) = c.borrow().complete(input, cursor as usize);
        let items: Vec<Json> = found
            .into_iter()
            .map(|s| json!({"label": s.label, "detail": s.detail}))
            .collect();
        json!({"start": start, "items": items}).to_string()
    })
}

fn lsp_position(line: u32, character: u32) -> lsp_types_position::Position {
    lsp_types_position::Position { line, character }
}

mod lsp_types_position {
    pub use telvax_lsp::position::Position;
}
