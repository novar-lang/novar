//! Talks to the real `telvax-lsp` binary over stdio, like an editor would.

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
        let mut child = Command::new(env!("CARGO_BIN_EXE_telvax-lsp"))
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

const URI: &str = "file:///project/app.tvx";

fn open(c: &mut Client, text: &str) {
    c.notify(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": URI, "languageId": "telvax", "version": 1, "text": text}}),
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

const PROGRAM: &str = r#"class Point:
    x: float
    y: float

def norm(p: Point) -> float:
    return p.x * p.x + p.y * p.y

def main() -> float:
    origin = Point(x=1.0, y=2.0)
    return norm(origin)
"#;

#[test]
fn advertises_its_capabilities() {
    let mut c = Client::start();
    let r = initialize(&mut c);
    let caps = &r["result"]["capabilities"];
    assert_eq!(caps["hoverProvider"], true);
    assert_eq!(caps["definitionProvider"], true);
    assert_eq!(caps["inlayHintProvider"], true);
    assert_eq!(caps["documentFormattingProvider"], true);
    assert_eq!(caps["referencesProvider"], true);
    assert_eq!(caps["renameProvider"]["prepareProvider"], true);
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
    open(
        &mut c,
        r#"
def f() -> int32:
    return "oops"
"#,
    );
    let d = c.diagnostics(URI);
    assert_eq!(d.len(), 1);
    assert_eq!(d[0]["severity"], 1);
    assert_eq!(d[0]["source"], "telvax");
    assert_eq!(d[0]["range"]["start"], json!({"line": 2, "character": 11}));
    assert!(
        d[0]["message"]
            .as_str()
            .unwrap()
            .contains("expected `int32`, found `str`")
    );

    c.notify(
        "textDocument/didChange",
        json!({"textDocument": {"uri": URI, "version": 2}, "contentChanges": [{"text": r#"
def f() -> int32:
    return 1
"#}]}),
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

    let hover = c.request("textDocument/hover", at(9, 13));
    let text = hover["result"]["contents"]["value"].as_str().unwrap();
    assert!(text.contains("def norm(p: Point) -> float"), "{text}");

    let def = c.request("textDocument/definition", at(9, 13));
    assert_eq!(
        def["result"]["range"]["start"],
        json!({"line": 4, "character": 4})
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
        json!({"textDocument": {"uri": URI}, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 11, "character": 0}}}),
    );
    let hints = hints["result"].as_array().unwrap();
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0]["label"], ": Point");
    assert_eq!(hints[0]["position"], json!({"line": 8, "character": 10}));

    // The sample is already formatted, so there is nothing to change.
    let fmt = c.request(
        "textDocument/formatting",
        json!({"textDocument": {"uri": URI}, "options": {"tabSize": 4, "insertSpaces": true}}),
    );
    assert_eq!(fmt["result"], json!([]));

    let items = c.request("textDocument/completion", at(9, 13));
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
        r#"
class Point:
    x: float
    y: float

def f(p: Point) -> float:
    return 0.0
"#,
    );
    c.diagnostics(URI);
    // The user types `p.` on a new line; the half-written statement does not parse.
    c.notify(
        "textDocument/didChange",
        json!({"textDocument": {"uri": URI, "version": 2}, "contentChanges": [{"text": r#"
class Point:
    x: float
    y: float

def f(p: Point) -> float:
    return p.
"#}]}),
    );
    c.diagnostics(URI);
    let items = c.request("textDocument/completion", at(6, 13));
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
    let r = c.request("textDocument/codeLens", json!({}));
    assert_eq!(r["error"]["code"], -32601);
}

#[test]
fn formats_a_messy_document() {
    let mut c = Client::start();
    initialize(&mut c);
    open(&mut c, "def main() -> int32:\n  x=1\n  return x+2\n");
    let r = c.request(
        "textDocument/formatting",
        json!({"textDocument": {"uri": URI}, "options": {"tabSize": 4, "insertSpaces": true}}),
    );
    let edits = r["result"].as_array().unwrap();
    assert_eq!(edits.len(), 1);
    assert_eq!(
        edits[0]["newText"],
        "def main() -> int32:\n    x = 1\n    return x + 2\n"
    );
    c.request("shutdown", json!(null));
}

fn rename_params(line: u32, character: u32, new_name: &str) -> Value {
    json!({
        "textDocument": {"uri": URI},
        "position": {"line": line, "character": character},
        "newName": new_name,
    })
}

#[test]
fn finds_references_and_renames_variables_and_functions() {
    let mut c = Client::start();
    initialize(&mut c);
    open(&mut c, PROGRAM);
    assert!(c.diagnostics(URI).is_empty());

    // `origin` is declared on line 8 and used on line 9.
    let refs = c.request(
        "textDocument/references",
        json!({"textDocument": {"uri": URI}, "position": {"line": 9, "character": 18},
               "context": {"includeDeclaration": true}}),
    );
    let lines: Vec<_> = refs["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["range"]["start"]["line"].as_u64().unwrap())
        .collect();
    assert_eq!(lines, [8, 9]);
    let no_decl = c.request(
        "textDocument/references",
        json!({"textDocument": {"uri": URI}, "position": {"line": 9, "character": 18},
               "context": {"includeDeclaration": false}}),
    );
    assert_eq!(no_decl["result"].as_array().unwrap().len(), 1);

    // The editor asks first which text will be renamed.
    let prep = c.request("textDocument/prepareRename", at(9, 13));
    assert_eq!(prep["result"]["placeholder"], "norm");

    // Renaming the function edits its definition and its call.
    let r = c.request("textDocument/rename", rename_params(9, 13, "length"));
    let edits = r["result"]["changes"][URI].as_array().unwrap();
    assert_eq!(edits.len(), 2);
    assert!(edits.iter().all(|e| e["newText"] == "length"));
    let starts: Vec<_> = edits
        .iter()
        .map(|e| e["range"]["start"]["line"].as_u64().unwrap())
        .collect();
    assert_eq!(starts, [4, 9]);

    // Things that must be refused, with a message the editor can show.
    let keyword = c.request("textDocument/rename", rename_params(9, 13, "while"));
    assert!(
        keyword["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not a valid name"),
        "{keyword}"
    );
    let main = c.request("textDocument/rename", rename_params(7, 6, "start"));
    assert!(
        main["error"]["message"]
            .as_str()
            .unwrap()
            .contains("entry point"),
        "{main}"
    );
    // `p` is already a parameter of `norm`: renaming `origin` to it would change what the code means.
    let clash = c.request("textDocument/rename", rename_params(8, 6, "norm"));
    assert!(
        clash["error"]["message"]
            .as_str()
            .unwrap()
            .contains("collides"),
        "{clash}"
    );
    let on_struct = c.request("textDocument/prepareRename", at(0, 8));
    assert!(
        on_struct["error"]["message"]
            .as_str()
            .unwrap()
            .contains("only variables and functions"),
        "{on_struct}"
    );
    c.request("shutdown", json!(null));
}
