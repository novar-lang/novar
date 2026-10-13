//! Talks to the real `novar-lsp` binary over stdio, like an editor would.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl Client {
    fn start() -> Client {
        let mut child = Command::new(env!("CARGO_BIN_EXE_novar-lsp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Client {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn send(&mut self, msg: Value) {
        let body = msg.to_string();
        write!(self.stdin, "Content-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut len = 0;
        loop {
            let mut line = String::new();
            assert!(
                self.stdout.read_line(&mut line).unwrap() > 0,
                "server closed the pipe"
            );
            if line == "\r\n" {
                break;
            }
            if let Some(v) = line.strip_prefix("Content-Length: ") {
                len = v.trim().parse().unwrap();
            }
        }
        let mut buf = vec![0; len];
        self.stdout.read_exact(&mut buf).unwrap();
        serde_json::from_slice(&buf).unwrap()
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let msg = self.read();
            if msg["id"] == id {
                return msg;
            }
        }
    }

    fn notify(&mut self, method: &str, params: Value) {
        self.send(json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    /// The next `publishDiagnostics` for `uri`.
    fn diagnostics(&mut self, uri: &str) -> Vec<Value> {
        loop {
            let msg = self.read();
            if msg["method"] == "textDocument/publishDiagnostics" && msg["params"]["uri"] == uri {
                return msg["params"]["diagnostics"].as_array().unwrap().clone();
            }
        }
    }
}

const URI: &str = "file:///project/app.nv";

fn open(c: &mut Client, text: &str) {
    c.notify(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": URI, "languageId": "novar", "version": 1, "text": text}}),
    );
}

fn at(line: u32, character: u32) -> Value {
    json!({"textDocument": {"uri": URI}, "position": {"line": line, "character": character}})
}

fn initialize(c: &mut Client) -> Value {
    let r = c.request(
        "initialize",
        json!({"processId": null, "rootUri": null, "capabilities": {}}),
    );
    c.notify("initialized", json!({}));
    r
}

const PROGRAM: &str = "struct Point { x: f64, y: f64 }\nfn norm(p: Point) -> f64 { p.x * p.x + p.y * p.y }\nfn main() -> f64 {\n    let origin = Point { x: 1.0, y: 2.0 };\n    norm(origin)\n}\n";

#[test]
fn advertises_its_capabilities() {
    let mut c = Client::start();
    let r = initialize(&mut c);
    let caps = &r["result"]["capabilities"];
    assert_eq!(caps["hoverProvider"], true);
    assert_eq!(caps["definitionProvider"], true);
    assert_eq!(caps["inlayHintProvider"], true);
    assert_eq!(caps["textDocumentSync"], 1);
    assert!(
        caps["completionProvider"]["triggerCharacters"]
            .as_array()
            .unwrap()
            .contains(&json!("."))
    );
    c.request("shutdown", json!(null));
    c.notify("exit", json!(null));
    assert!(c.child.wait().unwrap().success());
}

#[test]
fn publishes_diagnostics_and_clears_them() {
    let mut c = Client::start();
    initialize(&mut c);
    open(&mut c, "fn f() -> i32 {\n    \"oops\"\n}\n");
    let d = c.diagnostics(URI);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0]["severity"], 1);
    assert_eq!(d[0]["source"], "novar");
    assert_eq!(d[0]["range"]["start"], json!({"line": 1, "character": 4}));
    assert!(
        d[0]["message"]
            .as_str()
            .unwrap()
            .contains("expected `i32`, found `String`")
    );

    c.notify(
        "textDocument/didChange",
        json!({"textDocument": {"uri": URI, "version": 2}, "contentChanges": [{"text": "fn f() -> i32 { 1 }\n"}]}),
    );
    assert!(
        c.diagnostics(URI).is_empty(),
        "fixing the code clears the squiggle"
    );
}

#[test]
fn answers_editor_requests() {
    let mut c = Client::start();
    initialize(&mut c);
    open(&mut c, PROGRAM);
    assert!(c.diagnostics(URI).is_empty());

    let hover = c.request("textDocument/hover", at(4, 6));
    let text = hover["result"]["contents"]["value"].as_str().unwrap();
    assert!(text.contains("fn norm(p: Point) -> f64"), "{text}");

    let def = c.request("textDocument/definition", at(4, 6));
    assert_eq!(
        def["result"]["range"]["start"],
        json!({"line": 1, "character": 3})
    );

    let symbols = c.request(
        "textDocument/documentSymbol",
        json!({"textDocument": {"uri": URI}}),
    );
    let names: Vec<_> = symbols["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["Point", "norm", "main"]);

    let hints = c.request(
        "textDocument/inlayHint",
        json!({"textDocument": {"uri": URI}, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 9, "character": 0}}}),
    );
    let hints = hints["result"].as_array().unwrap();
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0]["label"], ": Point");
    assert_eq!(hints[0]["position"], json!({"line": 3, "character": 14}));

    let items = c.request("textDocument/completion", at(4, 9));
    let labels: Vec<_> = items["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    assert!(
        labels.contains(&"origin") || labels.contains(&"norm"),
        "{labels:?}"
    );
}

#[test]
fn completes_members_while_typing() {
    let mut c = Client::start();
    initialize(&mut c);
    open(
        &mut c,
        "struct Point { x: f64, y: f64 }\nfn f(p: Point) -> f64 {\n    0.0\n}\n",
    );
    c.diagnostics(URI);
    // The user types `p.` on a new line; the half-written statement does not parse.
    c.notify(
        "textDocument/didChange",
        json!({"textDocument": {"uri": URI, "version": 2}, "contentChanges": [{"text": "struct Point { x: f64, y: f64 }\nfn f(p: Point) -> f64 {\n    p.\n    0.0\n}\n"}]}),
    );
    c.diagnostics(URI);
    let items = c.request("textDocument/completion", at(2, 6));
    let labels: Vec<_> = items["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["x", "y"]);
}

#[test]
fn unknown_requests_get_an_error_not_a_hang() {
    let mut c = Client::start();
    initialize(&mut c);
    let r = c.request("textDocument/rename", json!({}));
    assert_eq!(r["error"]["code"], -32601);
}
