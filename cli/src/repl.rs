//! `telvax repl`: type definitions, variables and expressions and see the result at once.

use std::cell::RefCell;
use std::io::IsTerminal;
use std::rc::Rc;

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::Validator;
use rustyline::{Context, Editor, Helper};
use telvax_repl::{Outcome, Session};

use super::{CliError, failed, take_allow};

const HELP: &str = "\
Type a definition (fn, type, enum, trait, extend, import), a variable (let, var), a statement
or an expression. A block ends with an empty line. Names last until :reset.

  :type <expression>   the type of an expression, without running it
  :vars                the variables, with their types and values
  :defs                the functions, types and the like defined so far
  :load <file.tvx>     enter the definitions and variables of a file
  :reset               forget everything
  :help                this text
  :quit                leave (also Ctrl-D)
Tab completes names, fields and methods; the up and down arrows go through earlier inputs.";

/// Tab completion for the line being typed.
struct Completion(Rc<RefCell<Session>>);

impl Completer for Completion {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, found) = self.0.borrow().complete(line, pos);
        let pairs = found
            .into_iter()
            .map(|s| Pair {
                display: match &s.detail {
                    Some(d) => format!("{}  {d}", s.label),
                    None => s.label.clone(),
                },
                replacement: s.label,
            })
            .collect();
        Ok((start, pairs))
    }
}

impl Hinter for Completion {
    type Hint = String;
}
impl Highlighter for Completion {}
impl Validator for Completion {}
impl Helper for Completion {}

/// `telvax repl [file.tvx...] [--allow list]`
pub fn run(args: &[&str]) -> Result<(), CliError> {
    let (caps, files) = take_allow(args)?;
    let session = Rc::new(RefCell::new(Session::printing()));
    session.borrow_mut().set_caps(caps);
    for file in files {
        load(&session, file)?;
    }
    let interactive = std::io::stdin().is_terminal();
    let mut editor: Editor<Completion, FileHistory> =
        Editor::new().or_else(|e| failed(e.to_string()))?;
    editor.set_helper(Some(Completion(session.clone())));
    let history = std::env::var_os("TELVAX_HISTORY")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|h| std::path::PathBuf::from(h).join(".telvax_history"))
        });
    if let Some(path) = &history {
        let _ = editor.load_history(path);
    }
    if interactive {
        println!(
            "Telvax {}   :help for the commands, :quit to leave",
            env!("CARGO_PKG_VERSION")
        );
    }

    let mut pending = String::new();
    loop {
        let prompt = match (interactive, pending.is_empty()) {
            (false, _) => "",
            (true, true) => ">>> ",
            (true, false) => "... ",
        };
        let initial = next_indent(&pending);
        match editor.readline_with_initial(prompt, (&initial, "")) {
            Ok(line) => {
                if pending.is_empty() && line.trim().is_empty() {
                    continue;
                }
                if pending.is_empty() && line.trim_start().starts_with(':') {
                    let _ = editor.add_history_entry(line.as_str());
                    if command(&session, line.trim()) {
                        break;
                    }
                    continue;
                }
                let blank = line.trim().is_empty();
                if !pending.is_empty() {
                    pending.push('\n');
                }
                pending.push_str(&line);
                // An empty line ends a block; otherwise the input ends when it is complete.
                if blank || Session::is_complete(&pending) {
                    let _ = editor.add_history_entry(pending.as_str());
                    show(&session.borrow_mut().eval(&pending));
                    pending.clear();
                }
            }
            Err(ReadlineError::Interrupted) => {
                if !pending.is_empty() {
                    println!("(input cleared)");
                }
                pending.clear();
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => return failed(e.to_string()),
        }
    }
    if let Some(path) = &history {
        let _ = editor.save_history(path);
    }
    Ok(())
}

/// The spaces to start the next line with: those of the line before, and one level more after a `:`.
fn next_indent(pending: &str) -> String {
    let Some(last) = pending.lines().rev().find(|l| !l.trim().is_empty()) else {
        return String::new();
    };
    let spaces = last.len() - last.trim_start().len();
    let more = if last.trim_end().ends_with(':') { 4 } else { 0 };
    " ".repeat(spaces + more)
}

/// Runs a `:command`. `true` when the session should end.
fn command(session: &Rc<RefCell<Session>>, line: &str) -> bool {
    let (word, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    let rest = rest.trim();
    match word {
        ":quit" | ":q" | ":exit" => return true,
        ":help" | ":h" | ":?" => println!("{HELP}"),
        ":reset" => {
            session.borrow_mut().reset();
            println!("(everything forgotten)");
        }
        ":vars" => {
            let vars = session.borrow().variables();
            if vars.is_empty() {
                println!("(no variables)");
            }
            for (name, ty, value) in vars {
                println!("{name}: {ty} = {value}");
            }
        }
        ":defs" => {
            let defs = session.borrow().definitions();
            if defs.is_empty() {
                println!("(nothing defined)");
            }
            for d in defs {
                println!("{d}");
            }
        }
        ":type" | ":t" if !rest.is_empty() => match session.borrow_mut().type_of(rest) {
            Ok(ty) => println!("{ty}"),
            Err(e) => eprintln!("{e}"),
        },
        ":type" | ":t" => eprintln!("error: :type takes an expression, as in `:type 1 + 2`"),
        ":load" if !rest.is_empty() => {
            if let Err(CliError::Failed(msg)) = load(session, rest) {
                eprintln!("error: {msg}");
            }
        }
        ":load" => eprintln!("error: :load takes a file, as in `:load app.tvx`"),
        other => eprintln!("error: no command `{other}`; :help lists them"),
    }
    false
}

fn load(session: &Rc<RefCell<Session>>, file: &str) -> Result<(), CliError> {
    let text =
        std::fs::read_to_string(file).or_else(|e| failed(format!("cannot read `{file}`: {e}")))?;
    let out = session.borrow_mut().load(&text);
    match &out.error {
        Some(e) => failed(format!("`{file}`: {e}")),
        None => {
            if !out.defined.is_empty() {
                println!("{file}: {}", out.defined.join(", "));
            }
            Ok(())
        }
    }
}

/// What an input did: its value, what it defined, what went wrong.
fn show(out: &Outcome) {
    if let Some(value) = &out.value {
        println!("{}", value.text);
    }
    for note in &out.notes {
        println!("note: {note}");
    }
    if let Some(e) = &out.error {
        eprintln!("{e}");
    }
}
