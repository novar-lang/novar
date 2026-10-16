//! The Debug Adapter Protocol on a pair of streams: `Content-Length` framing and JSON messages.

use std::io::{self, BufRead, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use serde_json::{Value, json};

/// Reads the next message; `None` when the stream ends.
pub fn read(input: &mut impl BufRead) -> io::Result<Option<Value>> {
    let mut length = None;
    loop {
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(n) = line.strip_prefix("Content-Length:") {
            length = n.trim().parse::<usize>().ok();
        }
    }
    let Some(length) = length else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a message without Content-Length",
        ));
    };
    let mut body = vec![0; length];
    input.read_exact(&mut body)?;
    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Where responses and events go. Shared by the thread that answers requests and the thread
/// that runs the program, so every message is written whole.
pub struct Out {
    writer: Mutex<Box<dyn Write + Send>>,
    seq: AtomicI64,
}

impl std::fmt::Debug for Out {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Out")
    }
}

impl Out {
    pub fn new(writer: impl Write + Send + 'static) -> Out {
        Out {
            writer: Mutex::new(Box::new(writer)),
            seq: AtomicI64::new(1),
        }
    }

    fn send(&self, mut message: Value) {
        message["seq"] = json!(self.seq.fetch_add(1, Ordering::Relaxed));
        let body = message.to_string();
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let _ = write!(w, "Content-Length: {}\r\n\r\n{body}", body.len());
        let _ = w.flush();
    }

    pub fn event(&self, name: &str, body: Value) {
        self.send(json!({"type": "event", "event": name, "body": body}));
    }

    /// The answer to `request`: its body, or the message of why it failed.
    pub fn respond(&self, request: &Value, result: Result<Value, String>) {
        let mut message = json!({
            "type": "response",
            "request_seq": request["seq"],
            "command": request["command"],
        });
        match result {
            Ok(body) => {
                message["success"] = json!(true);
                message["body"] = body;
            }
            Err(text) => {
                message["success"] = json!(false);
                message["message"] = json!(text);
            }
        }
        self.send(message);
    }

    /// Text for the debug console (`category`: `stdout`, `stderr` or `console`).
    pub fn output(&self, category: &str, text: &str) {
        self.event("output", json!({"category": category, "output": text}));
    }
}

/// `std.io.print` of the program, sent to the debug console as it is written.
pub struct ConsoleWriter(pub std::sync::Arc<Out>);

impl Write for ConsoleWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.output("stdout", &String::from_utf8_lossy(buf));
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
