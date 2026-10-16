//! `telvax dap` driven like an editor would: breakpoints, stepping and looking at variables.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

const TELVAX: &str = env!("CARGO_BIN_EXE_telvax");

const SRC: &str = "\
type Point:
    x: Int
    y: Int

fn add(a: Int, b: Int) -> Int:
    let total = a + b
    total

fn main() -> Int:
    let p = Point(x: 1, y: 2)
    var n = add(p.x, p.y)
    for i in range(0, 2):
        n = n + i
    print(\"done\")
    n
";

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    seq: i64,
    /// Events that arrived while waiting for something else.
    events: VecDeque<Value>,
}

impl Client {
    fn start() -> Client {
        let mut child = Command::new(TELVAX)
            .arg("dap")
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
            seq: 0,
            events: VecDeque::new(),
        }
    }

    fn read(&mut self) -> Value {
        let mut length = 0;
        loop {
            let mut line = String::new();
            assert!(
                self.stdout.read_line(&mut line).unwrap() > 0,
                "the adapter closed the pipe"
            );
            if line == "\r\n" {
                break;
            }
            if let Some(n) = line.strip_prefix("Content-Length: ") {
                length = n.trim().parse().unwrap();
            }
        }
        let mut body = vec![0; length];
        self.stdout.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// Sends a request and returns its response; events seen meanwhile are kept.
    fn request(&mut self, command: &str, arguments: Value) -> Value {
        self.seq += 1;
        let body = json!({"seq": self.seq, "type": "request", "command": command,
                          "arguments": arguments})
        .to_string();
        write!(self.stdin, "Content-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        self.stdin.flush().unwrap();
        loop {
            let m = self.read();
            if m["type"] == "response" && m["request_seq"] == self.seq {
                return m;
            }
            self.events.push_back(m);
        }
    }

    /// The next event called `name`.
    fn event(&mut self, name: &str) -> Value {
        if let Some(i) = self.events.iter().position(|e| e["event"] == name) {
            return self.events.remove(i).unwrap();
        }
        loop {
            let m = self.read();
            if m["type"] == "event" && m["event"] == name {
                return m;
            }
            self.events.push_back(m);
        }
    }

    /// The calls of the stopped program, innermost first, as `(function, line)`.
    fn stack(&mut self) -> Vec<(String, u64)> {
        let r = self.request("stackTrace", json!({"threadId": 1}));
        assert_eq!(r["success"], true, "{r}");
        r["body"]["stackFrames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (
                    f["name"].as_str().unwrap().to_string(),
                    f["line"].as_u64().unwrap(),
                )
            })
            .collect()
    }

    fn frame_id(&mut self, index: usize) -> i64 {
        let r = self.request("stackTrace", json!({"threadId": 1}));
        r["body"]["stackFrames"][index]["id"].as_i64().unwrap()
    }

    /// The variables of a scope of a frame, as `(name, value)`.
    fn variables(&mut self, frame: i64, scope: usize) -> Vec<(String, String)> {
        let scopes = self.request("scopes", json!({"frameId": frame}));
        let reference = scopes["body"]["scopes"][scope]["variablesReference"].clone();
        self.list(reference)
    }

    fn list(&mut self, reference: Value) -> Vec<(String, String)> {
        let r = self.request("variables", json!({"variablesReference": reference}));
        assert_eq!(r["success"], true, "{r}");
        r["body"]["variables"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v["name"].as_str().unwrap().to_string(),
                    v["value"].as_str().unwrap().to_string(),
                )
            })
            .collect()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn program(name: &str) -> (std::path::PathBuf, String) {
    let dir = std::env::temp_dir().join(format!("telvax-dap-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("app.tvx");
    std::fs::write(&file, SRC).unwrap();
    let path = file.to_str().unwrap().to_string();
    (dir, path)
}

#[test]
fn stops_at_a_breakpoint_shows_the_calls_and_the_variables_and_steps() {
    let (dir, path) = program("session");
    let mut c = Client::start();
    let init = c.request("initialize", json!({"adapterID": "telvax"}));
    assert_eq!(init["body"]["supportsConfigurationDoneRequest"], true);
    let launch = c.request("launch", json!({"program": path}));
    assert_eq!(launch["success"], true, "{launch}");
    c.event("initialized");

    // Line 6 is a statement; line 99 has nothing at or after it.
    let set = c.request(
        "setBreakpoints",
        json!({"source": {"path": path}, "breakpoints": [{"line": 6}, {"line": 99}]}),
    );
    let bps = &set["body"]["breakpoints"];
    assert_eq!(
        (bps[0]["verified"].clone(), bps[0]["line"].clone()),
        (json!(true), json!(6))
    );
    assert_eq!(bps[1]["verified"], false);
    // A breakpoint on a blank line moves to the next statement (line 9 is `fn main`).
    let moved = c.request(
        "setBreakpoints",
        json!({"source": {"path": path}, "breakpoints": [{"line": 6}, {"line": 8}]}),
    );
    assert_eq!(moved["body"]["breakpoints"][1]["line"], 10);

    assert_eq!(c.request("configurationDone", json!({}))["success"], true);
    let stopped = c.event("stopped");
    assert_eq!(stopped["body"]["reason"], "breakpoint");
    // The second request replaced the first: the program stops at line 10 of `main`, then at 6.
    assert_eq!(c.stack(), [("main".to_string(), 10)]);
    c.request("continue", json!({"threadId": 1}));
    c.event("stopped");

    // Inside `add`, called from line 11 of `main`.
    assert_eq!(
        c.stack(),
        [("add".to_string(), 6), ("main".to_string(), 11)]
    );
    let add = c.frame_id(0);
    assert_eq!(
        c.variables(add, 0),
        [
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string())
        ]
    );
    // `main` has `p`, which opens up into its fields.
    let main = c.frame_id(1);
    let scopes = c.request("scopes", json!({"frameId": main}));
    let locals = scopes["body"]["scopes"][0]["variablesReference"].clone();
    let r = c.request("variables", json!({"variablesReference": locals}));
    let p = &r["body"]["variables"][0];
    assert_eq!(p["name"], "p");
    assert_eq!(p["value"], "Point(x=1, y=2)");
    assert_eq!(p["type"], "Point");
    let fields = c.list(p["variablesReference"].clone());
    assert_eq!(
        fields,
        [
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "2".to_string())
        ]
    );
    // Hovering `p.y` in the editor.
    let hover = c.request(
        "evaluate",
        json!({"expression": "p.y", "frameId": main, "context": "hover"}),
    );
    assert_eq!(hover["body"]["result"], "2");
    let bad = c.request("evaluate", json!({"expression": "1 + 1", "frameId": main}));
    assert_eq!(bad["success"], false);

    // Step over the `let`, then out of `add` to the next line of `main`.
    c.request("next", json!({"threadId": 1}));
    assert_eq!(c.event("stopped")["body"]["reason"], "step");
    assert_eq!(c.stack()[0], ("add".to_string(), 7));
    let add = c.frame_id(0);
    assert!(
        c.variables(add, 0)
            .contains(&("total".to_string(), "3".to_string()))
    );
    c.request("stepOut", json!({"threadId": 1}));
    c.event("stopped");
    assert_eq!(c.stack(), [("main".to_string(), 12)]);

    // Step into the loop body, then let it run to the end with no breakpoints.
    c.request("stepIn", json!({"threadId": 1}));
    c.event("stopped");
    assert_eq!(c.stack(), [("main".to_string(), 13)]);
    c.request(
        "setBreakpoints",
        json!({"source": {"path": path}, "breakpoints": []}),
    );
    c.request("continue", json!({"threadId": 1}));
    c.event("exited");
    c.event("terminated");
    let output = c.event("output");
    assert!(output["body"]["output"].as_str().unwrap().contains("done"));
    c.request("disconnect", json!({}));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn stop_on_entry_stops_at_the_first_statement() {
    let (dir, path) = program("entry");
    let mut c = Client::start();
    c.request("initialize", json!({}));
    c.request("launch", json!({"program": path, "stopOnEntry": true}));
    c.event("initialized");
    c.request("configurationDone", json!({}));
    assert_eq!(c.event("stopped")["body"]["reason"], "step");
    assert_eq!(c.stack(), [("main".to_string(), 10)]);
    c.request("continue", json!({"threadId": 1}));
    c.event("terminated");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn launch_reports_what_is_wrong() {
    let mut c = Client::start();
    c.request("initialize", json!({}));
    let r = c.request("launch", json!({}));
    assert_eq!(r["success"], false);
    assert!(
        r["message"].as_str().unwrap().contains("needs `program`"),
        "{r}"
    );

    let (dir, path) = program("errors");
    let r = c.request("launch", json!({"program": path, "function": "nope"}));
    assert!(
        r["message"]
            .as_str()
            .unwrap()
            .contains("has no function `nope`"),
        "{r}"
    );

    std::fs::write(&path, "fn main() -> Int:\n    \"text\"\n").unwrap();
    let r = c.request("launch", json!({"program": path}));
    assert_eq!(r["success"], false);
    assert!(r["message"].as_str().unwrap().contains("error"), "{r}");
    let r = c.request("stackTrace", json!({"threadId": 1}));
    assert_eq!(r["success"], false);
    let _ = std::fs::remove_dir_all(dir);
}
