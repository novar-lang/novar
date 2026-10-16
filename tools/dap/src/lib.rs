//! The Telvax debug adapter, for editors that speak the Debug Adapter Protocol (VS Code and
//! others). It runs a program in the interpreter and stops it at breakpoints, lets the editor
//! step, look at the calls in progress and open up the variables.
//!
//! Two threads share one state: the one that reads requests and the one that runs the program.
//! The interpreter calls the adapter before each statement on the program's thread; to stop, the
//! adapter waits there until a request to continue arrives.

mod program;
mod protocol;

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;

use serde_json::{Value as Json, json};
use telvax_runtime::debug::{Debugger, Frame, children};
use telvax_runtime::{Caps, Interp, Value};

use program::Program;
use protocol::{ConsoleWriter, Out};

/// The thread that runs the program has the deepest recursion of any Telvax tool.
const STACK_BYTES: usize = 512 * 1024 * 1024;
/// The one thread of the program, as the editor sees it.
const THREAD: i64 = 1;
/// The most variables listed under one name.
const MAX_CHILDREN: usize = 1000;

/// Serves a debug session on standard input and output.
pub fn run_stdio() -> io::Result<()> {
    run(io::BufReader::new(io::stdin()), io::stdout())
}

/// Serves a debug session until the editor disconnects.
pub fn run(mut input: impl BufRead, output: impl Write + Send + 'static) -> io::Result<()> {
    let out = Arc::new(Out::new(output));
    let mut session = Session {
        out: out.clone(),
        shared: Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            out,
        }),
        launched: None,
        refs: Vec::new(),
    };
    while let Some(request) = protocol::read(&mut input)? {
        if session.handle(&request) {
            break;
        }
    }
    Ok(())
}

// ---- state shared with the thread that runs the program -------------------------------------

#[derive(Default)]
struct State {
    /// Statements to stop at, by file as the editor named it.
    breakpoints: HashMap<String, Vec<usize>>,
    /// All of them, for the check before each statement.
    active: HashSet<usize>,
    mode: Mode,
    /// The calls in progress while the program is stopped.
    stopped: Option<Vec<Frame>>,
    /// Set by a request that lets the program go on.
    go: bool,
    /// The thread that runs the program: statements on other threads are not stopped at.
    thread: Option<ThreadId>,
}

/// When the program stops next, besides at breakpoints.
#[derive(Default, Clone)]
enum Mode {
    #[default]
    Run,
    /// At the next statement, wherever it is.
    In,
    /// At the next line of this call, or of the one that called it.
    Over {
        depth: usize,
        at: (String, usize),
    },
    /// Once this call has returned.
    Out {
        depth: usize,
    },
    Pause,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    out: Arc<Out>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

struct Hook {
    shared: Arc<Shared>,
    program: Arc<Program>,
}

impl Debugger for Hook {
    fn at_statement(&self, _: &Interp, stack: &[Frame]) {
        let Some(top) = stack.last() else { return };
        let mut st = self.shared.state();
        if st.thread != Some(std::thread::current().id()) {
            return;
        }
        let reason = if st.active.contains(&top.offset) {
            Some("breakpoint")
        } else {
            match &st.mode {
                Mode::Run => None,
                Mode::Pause => Some("pause"),
                Mode::In => Some("step"),
                Mode::Over { depth, at } => (stack.len() < *depth
                    || (stack.len() == *depth && self.line(top.offset).as_ref() != Some(at)))
                .then_some("step"),
                Mode::Out { depth } => (stack.len() < *depth).then_some("step"),
            }
        };
        let Some(reason) = reason else { return };
        st.mode = Mode::Run;
        st.stopped = Some(stack.to_vec());
        st.go = false;
        self.shared.out.event(
            "stopped",
            json!({"reason": reason, "threadId": THREAD, "allThreadsStopped": true}),
        );
        while !st.go {
            st = self.shared.wake.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.stopped = None;
    }
}

impl Hook {
    fn line(&self, offset: usize) -> Option<(String, usize)> {
        self.program.position(offset).map(|(f, l, _)| (f, l))
    }
}

// ---- the session ------------------------------------------------------------------------------

/// What `launch` set up.
struct Launched {
    interp: Arc<Interp>,
    program: Arc<Program>,
    function: String,
    args: Vec<Value>,
    stop_on_entry: bool,
}

/// What a `variablesReference` stands for, until the program goes on.
enum Ref {
    Locals(usize),
    Globals,
    Children(Value),
}

struct Session {
    out: Arc<Out>,
    shared: Arc<Shared>,
    launched: Option<Launched>,
    refs: Vec<Ref>,
}

impl Session {
    /// Answers one request. `true` when the session is over.
    fn handle(&mut self, request: &Json) -> bool {
        let command = request["command"].as_str().unwrap_or("");
        let args = &request["arguments"];
        let result = match command {
            "initialize" => Ok(json!({
                "supportsConfigurationDoneRequest": true,
                "supportsEvaluateForHovers": true,
                "supportsTerminateRequest": true,
            })),
            "launch" => self.launch(args),
            "setBreakpoints" => self.set_breakpoints(args),
            "setExceptionBreakpoints" => Ok(json!({})),
            "configurationDone" => self.start(),
            "threads" => Ok(json!({"threads": [{"id": THREAD, "name": "main"}]})),
            "stackTrace" => self.stack_trace(),
            "scopes" => self.scopes(args),
            "variables" => self.variables(args),
            "evaluate" => self.evaluate(args),
            "continue" => self.resume(|_, _| Mode::Run, json!({"allThreadsContinued": true})),
            "next" => self.resume(|depth, at| Mode::Over { depth, at }, json!({})),
            "stepIn" => self.resume(|_, _| Mode::In, json!({})),
            "stepOut" => self.resume(|depth, _| Mode::Out { depth }, json!({})),
            "pause" => {
                self.shared.state().mode = Mode::Pause;
                Ok(json!({}))
            }
            "disconnect" | "terminate" => {
                self.out.respond(request, Ok(json!({})));
                return true;
            }
            other => Err(format!("`{other}` is not supported by the Telvax debugger")),
        };
        let launched = command == "launch" && result.is_ok();
        self.out.respond(request, result);
        if launched {
            self.out.event("initialized", json!({}));
        }
        false
    }

    fn launch(&mut self, args: &Json) -> Result<Json, String> {
        let file = args["program"]
            .as_str()
            .ok_or("the launch configuration needs `program`: the .tvx file to run")?;
        let function = args["function"].as_str().unwrap_or("main").to_string();
        let caps = match args["allow"].as_str() {
            Some(list) => Caps::parse(list)?,
            None => Caps::ALL,
        };
        let (program, hir) = Program::load(file)?;
        let interp = Interp::with_caps(hir, false, caps).map_err(|e| e.to_string())?;
        if interp.signature(&function).is_none() {
            let mut names = interp.function_names();
            names.sort();
            return Err(format!(
                "`{file}` has no function `{function}`; it has: {}",
                names.join(", ")
            ));
        }
        let words: Vec<String> = args["args"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|w| w.as_str().map_or_else(|| w.to_string(), str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let values = arguments(&interp, &function, &words)?;
        interp.set_output(ConsoleWriter(self.out.clone()));
        let program = Arc::new(program);
        if !args["noDebug"].as_bool().unwrap_or(false) {
            interp.set_debugger(Some(Arc::new(Hook {
                shared: self.shared.clone(),
                program: program.clone(),
            })));
        }
        self.launched = Some(Launched {
            interp: Arc::new(interp),
            program,
            function,
            args: values,
            stop_on_entry: args["stopOnEntry"].as_bool().unwrap_or(false),
        });
        Ok(json!({}))
    }

    fn set_breakpoints(&mut self, args: &Json) -> Result<Json, String> {
        let path = args["source"]["path"].as_str().unwrap_or("").to_string();
        let lines: Vec<usize> = match args["breakpoints"].as_array() {
            Some(list) => list
                .iter()
                .filter_map(|b| b["line"].as_u64().map(|l| l as usize))
                .collect(),
            None => args["lines"]
                .as_array()
                .map(|l| {
                    l.iter()
                        .filter_map(|x| x.as_u64().map(|x| x as usize))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let mut offsets = Vec::new();
        let mut answers = Vec::new();
        for line in lines {
            let found = self
                .launched
                .as_ref()
                .and_then(|l| l.program.breakpoint(&path, line));
            match found {
                Some((at, stmts)) => {
                    offsets.extend_from_slice(stmts);
                    answers.push(json!({"verified": true, "line": at}));
                }
                None => answers.push(json!({
                    "verified": false,
                    "line": line,
                    "message": "no statement on this line or after it",
                })),
            }
        }
        let mut st = self.shared.state();
        st.breakpoints.insert(path, offsets);
        st.active = st.breakpoints.values().flatten().copied().collect();
        Ok(json!({"breakpoints": answers}))
    }

    /// `configurationDone`: the editor has sent its breakpoints, the program can start.
    fn start(&mut self) -> Result<Json, String> {
        let Some(l) = &self.launched else {
            return Err("launch the program first".into());
        };
        let (interp, shared, out) = (l.interp.clone(), self.shared.clone(), self.out.clone());
        let (function, args) = (l.function.clone(), l.args.clone());
        if l.stop_on_entry {
            shared.state().mode = Mode::In;
        }
        let spawned = std::thread::Builder::new()
            .stack_size(STACK_BYTES)
            .spawn(move || {
                shared.state().thread = Some(std::thread::current().id());
                let code = match interp.call(&function, args) {
                    Ok(Value::Unit) => 0,
                    Ok(value) => {
                        out.output("stdout", &format!("{value}\n"));
                        0
                    }
                    Err(e) => {
                        out.output("stderr", &format!("error: {e}\n"));
                        1
                    }
                };
                out.event("exited", json!({"exitCode": code}));
                out.event("terminated", json!({}));
            });
        spawned.map(|_| json!({})).map_err(|e| e.to_string())
    }

    /// Lets the program go on in `mode`, computed from the depth and line it is stopped at.
    fn resume(
        &mut self,
        mode: impl Fn(usize, (String, usize)) -> Mode,
        body: Json,
    ) -> Result<Json, String> {
        let launched = self.launched.as_ref().ok_or("the program is not running")?;
        let mut st = self.shared.state();
        let Some(frames) = &st.stopped else {
            return Err("the program is not stopped".into());
        };
        let top = frames.last().map_or(0, |f| f.offset);
        let at = launched
            .program
            .position(top)
            .map(|(f, l, _)| (f, l))
            .unwrap_or_default();
        st.mode = mode(frames.len(), at);
        st.go = true;
        drop(st);
        self.shared.wake.notify_all();
        self.refs.clear();
        Ok(body)
    }

    fn stack_trace(&mut self) -> Result<Json, String> {
        let launched = self.launched.as_ref().ok_or("the program is not running")?;
        let st = self.shared.state();
        let frames = st.stopped.as_ref().ok_or("the program is not stopped")?;
        let list: Vec<Json> = (0..frames.len())
            .rev()
            .map(|i| {
                let f = &frames[i];
                let (file, line, column) = launched
                    .program
                    .position(f.offset)
                    .unwrap_or_else(|| ("?".into(), 1, 1));
                let name = std::path::Path::new(&file)
                    .file_name()
                    .map_or(file.clone(), |n| n.to_string_lossy().into_owned());
                json!({
                    "id": i + 1,
                    "name": launched.program.map.demangle(&f.function),
                    "source": {"name": name, "path": file},
                    "line": line,
                    "column": column,
                })
            })
            .collect();
        Ok(json!({"stackFrames": list, "totalFrames": frames.len()}))
    }

    fn new_ref(&mut self, r: Ref) -> usize {
        self.refs.push(r);
        self.refs.len()
    }

    fn scopes(&mut self, args: &Json) -> Result<Json, String> {
        let frame = args["frameId"].as_u64().unwrap_or(0) as usize;
        let locals = self.new_ref(Ref::Locals(frame.saturating_sub(1)));
        let globals = self.new_ref(Ref::Globals);
        Ok(json!({"scopes": [
            {"name": "Locals", "variablesReference": locals, "expensive": false},
            {"name": "Globals", "variablesReference": globals, "expensive": true},
        ]}))
    }

    /// One variable as the editor lists it.
    fn variable(&mut self, name: &str, value: &Value, ty: Option<&telvax_hir::Ty>) -> Json {
        let interp = &self.launched.as_ref().expect("launched").interp;
        let shown = interp.render(value, ty);
        let type_name = ty.map_or_else(|| value.type_name().to_string(), ToString::to_string);
        let reference = if children(value).is_empty() {
            0
        } else {
            self.new_ref(Ref::Children(value.clone()))
        };
        json!({
            "name": name,
            "value": shown,
            "type": type_name,
            "variablesReference": reference,
        })
    }

    fn variables(&mut self, args: &Json) -> Result<Json, String> {
        let id = args["variablesReference"].as_u64().unwrap_or(0) as usize;
        let start = args["start"].as_u64().unwrap_or(0) as usize;
        let count = args["count"].as_u64().map_or(MAX_CHILDREN, |c| c as usize);
        let interp = self
            .launched
            .as_ref()
            .ok_or("the program is not running")?
            .interp
            .clone();
        let items: Vec<(String, Value, Option<telvax_hir::Ty>)> = match self
            .refs
            .get(id.wrapping_sub(1))
            .ok_or("that variable is no longer available")?
        {
            Ref::Locals(i) => {
                let st = self.shared.state();
                let frame = st
                    .stopped
                    .as_ref()
                    .and_then(|f| f.get(*i))
                    .ok_or("the program is not stopped")?;
                interp
                    .locals(frame)
                    .into_iter()
                    .map(|v| (v.name, v.value, Some(v.ty)))
                    .collect()
            }
            Ref::Globals => interp
                .global_variables()
                .into_iter()
                .map(|v| (v.name, v.value, None))
                .collect(),
            Ref::Children(v) => children(v).into_iter().map(|(k, v)| (k, v, None)).collect(),
        };
        let list: Vec<Json> = items
            .iter()
            .skip(start)
            .take(count.min(MAX_CHILDREN))
            .map(|(name, value, ty)| self.variable(name, value, ty.as_ref()))
            .collect();
        Ok(json!({"variables": list}))
    }

    /// `evaluate`: a variable, or a field or item of one (`p.x`, `xs.0`).
    fn evaluate(&mut self, args: &Json) -> Result<Json, String> {
        let expression = args["expression"].as_str().unwrap_or("").trim().to_string();
        let interp = self
            .launched
            .as_ref()
            .ok_or("the program is not running")?
            .interp
            .clone();
        let frame = args["frameId"].as_u64().map(|f| f as usize);
        let (value, ty) = {
            let st = self.shared.state();
            let frames = st.stopped.as_ref().ok_or("the program is not stopped")?;
            let frame = frame
                .and_then(|f| frames.get(f.wrapping_sub(1)))
                .or(frames.last())
                .ok_or("no call to look in")?;
            let mut path = expression.split('.');
            let first = path.next().unwrap_or("");
            let found = interp
                .locals(frame)
                .into_iter()
                .rev()
                .find(|v| v.name == first)
                .map(|v| (v.value, Some(v.ty)))
                .or_else(|| {
                    interp
                        .global_variables()
                        .into_iter()
                        .find(|v| v.name == first)
                        .map(|v| (v.value, None))
                })
                .ok_or_else(|| {
                    format!(
                        "`{expression}` cannot be evaluated: write the name of a variable, or a field or item of one such as `p.x` or `xs.0`"
                    )
                })?;
            let (mut value, mut ty) = found;
            for step in path {
                let next = children(&value)
                    .into_iter()
                    .find(|(k, _)| k == step)
                    .ok_or_else(|| format!("`{step}` is not in {expression}"))?;
                value = next.1;
                ty = None;
            }
            (value, ty)
        };
        let shown = interp.render(&value, ty.as_ref());
        let reference = if children(&value).is_empty() {
            0
        } else {
            self.new_ref(Ref::Children(value))
        };
        Ok(json!({"result": shown, "variablesReference": reference}))
    }
}

/// The values for the words given to the function, read as the types it declares (a `Text`
/// takes the word as it is, the others take it as JSON).
fn arguments(interp: &Interp, function: &str, words: &[String]) -> Result<Vec<Value>, String> {
    let Some((params, _)) = interp.signature(function) else {
        return Err(format!("no function `{function}`"));
    };
    if params.len() != words.len() {
        let list: Vec<String> = params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
        return Err(format!(
            "`{function}` takes {} argument(s) ({}), got {}; set them in `args`",
            params.len(),
            list.join(", "),
            words.len()
        ));
    }
    params
        .iter()
        .zip(words)
        .map(|((name, ty), word)| {
            let json: Json = if *ty == telvax_hir::Ty::str() {
                Json::String(word.clone())
            } else {
                serde_json::from_str(word).map_err(|_| {
                    format!("argument `{name}` of `{function}` is {ty}, and `{word}` is not a value of that type")
                })?
            };
            interp
                .value_from_json(ty, &json)
                .map_err(|e| format!("argument `{name}` of `{function}` is {ty}: {e}"))
        })
        .collect()
}
