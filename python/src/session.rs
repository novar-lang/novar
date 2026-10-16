//! `telvax.Session`: an interactive session (the engine of `telvax repl`) for notebooks and
//! other front ends. Definitions and variables last from one `eval` to the next.

use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use telvax_repl as repl;

use crate::TelvaxError;

/// A session: `eval` runs a definition, a variable, a statement or an expression.
#[pyclass(module = "telvax", name = "Session")]
pub struct Session {
    inner: Mutex<repl::Session>,
}

impl Session {
    fn lock(&self) -> std::sync::MutexGuard<'_, repl::Session> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The byte offset of character `chars` of `text`.
fn byte_offset(text: &str, chars: usize) -> usize {
    text.char_indices()
        .nth(chars)
        .map_or(text.len(), |(i, _)| i)
}

/// The character index of byte `bytes` of `text`.
fn char_index(text: &str, bytes: usize) -> usize {
    text[..bytes.min(text.len())].chars().count()
}

fn outcome(py: Python<'_>, out: repl::Outcome) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("output", out.output)?;
    match out.value {
        Some(v) => {
            let value = PyDict::new(py);
            value.set_item("text", v.text)?;
            value.set_item("type", v.ty)?;
            value.set_item("html", v.html)?;
            dict.set_item("value", value)?;
        }
        None => dict.set_item("value", py.None())?,
    }
    dict.set_item("defined", out.defined)?;
    dict.set_item("notes", out.notes)?;
    dict.set_item("error", out.error)?;
    Ok(dict.unbind())
}

#[pymethods]
impl Session {
    /// `allow` limits what the program may do outside itself, as `telvax run --allow` does
    /// (`"fs.read,net"`, `"none"`); `step_limit` stops an input that runs for more evaluation steps.
    #[new]
    #[pyo3(signature = (allow = None, step_limit = None))]
    fn new(allow: Option<&str>, step_limit: Option<u64>) -> PyResult<Session> {
        let mut session = repl::Session::new();
        if let Some(list) = allow {
            session.set_caps(telvax_runtime::Caps::parse(list).map_err(TelvaxError::new_err)?);
        }
        session.set_step_limit(step_limit);
        Ok(Session {
            inner: Mutex::new(session),
        })
    }

    /// Runs one input. Returns a dict with `output` (what the program printed), `value` (`None`, or a
    /// dict with `text`, `type` and `html`), `defined`, `notes` and `error` (`None`, or the message).
    fn eval(&self, py: Python<'_>, code: &str) -> PyResult<Py<PyDict>> {
        let out = py.detach(|| self.lock().eval(code));
        outcome(py, out)
    }

    /// Enters a whole file: its definitions, then its variables in order. Returns what `eval` does.
    fn load(&self, py: Python<'_>, source: &str) -> PyResult<Py<PyDict>> {
        let out = py.detach(|| self.lock().load(source));
        outcome(py, out)
    }

    /// The type of an expression, without running it. Raises `TelvaxError` when it has errors.
    fn type_of(&self, py: Python<'_>, expression: &str) -> PyResult<String> {
        py.detach(|| self.lock().type_of(expression))
            .map_err(TelvaxError::new_err)
    }

    /// The variables as `(name, type, value)`.
    fn variables(&self) -> Vec<(String, String, String)> {
        self.lock().variables()
    }

    /// The functions, types and the like defined so far, as `fn name` or `type Name`.
    fn definitions(&self) -> Vec<String> {
        self.lock().definitions()
    }

    /// Forgets every definition and variable.
    fn reset(&self) {
        self.lock().reset();
    }

    /// Whether `code` can be run as it is, or is in the middle of a block, a bracket or a string.
    #[staticmethod]
    fn is_complete(code: &str) -> bool {
        repl::Session::is_complete(code)
    }

    /// What the word before character `cursor` can be completed to: `(start, [(label, detail)])`,
    /// where `start` is the character index where the word begins.
    fn complete(&self, code: &str, cursor: usize) -> (usize, Vec<(String, Option<String>)>) {
        let (start, found) = self.lock().complete(code, byte_offset(code, cursor));
        (
            char_index(code, start),
            found.into_iter().map(|s| (s.label, s.detail)).collect(),
        )
    }

    /// What is known about the name at character `cursor`, as Markdown, or `None`.
    fn hover(&self, code: &str, cursor: usize) -> Option<String> {
        self.lock().hover(code, byte_offset(code, cursor))
    }

    fn __repr__(&self) -> String {
        let session = self.lock();
        format!(
            "<telvax.Session {} definitions, {} variables>",
            session.definitions().len(),
            session.variables().len()
        )
    }
}
