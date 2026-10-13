use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::process::ExitCode;
use std::sync::Arc;

use novar_diagnostics::Diagnostic;
use novar_runtime::{Interp, Value, embed_text};
use novar_server::{HandlerError, handler};

const USAGE: &str = "\
usage:
  novar check <file.nv>
  novar run <file.nv> <function> [args...]
  novar serve <file.nv> [--port N] [--seed docs.jsonl]
  novar jit <file.nv> <function> [args...]      (native, needs --features llvm)
  novar build <file.nv> [-o out] [--emit exe|obj|llvm-ir|mlir] [-O 0-3]   (needs --features llvm)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["check", file] => check(file).map(|_| ()),
        ["run", file, func, rest @ ..] => run(file, func, rest),
        ["serve", file, rest @ ..] => serve(file, rest),
        ["jit", file, func, rest @ ..] => native::jit(file, func, rest),
        ["build", file, rest @ ..] => native::build(file, rest),
        _ => Err(CliError::Usage),
    };
    match code {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage) => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err(CliError::Failed(msg)) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}");
            }
            ExitCode::FAILURE
        }
    }
}

enum CliError {
    Usage,
    Failed(String),
}

fn failed<T>(msg: impl Into<String>) -> Result<T, CliError> {
    Err(CliError::Failed(msg.into()))
}

/// Parses and checks `file`, printing diagnostics. Fails if any is an error.
fn check(file: &str) -> Result<novar_hir::hir::Program, CliError> {
    let src =
        std::fs::read_to_string(file).or_else(|e| failed(format!("cannot read `{file}`: {e}")))?;
    let (program, mut diags): (_, Vec<Diagnostic>) = novar_parser::parse(&src);
    // Semantic analysis also runs on recovered trees, so users see all errors at once.
    let mut hir = None;
    if let Some(program) = program {
        let (h, d) = novar_hir::lower(&program);
        diags.extend(d);
        hir = Some(h);
    }
    eprint!(
        "{}",
        novar_diagnostics::render(file, &src, &diags, std::io::stderr().is_terminal())
    );
    let errors = diags.iter().filter(|d| d.is_error()).count();
    match (errors, hir) {
        (0, Some(h)) => {
            eprintln!("`{file}` ok");
            Ok(h)
        }
        _ => failed(format!("{errors} error(s) in `{file}`")),
    }
}

#[cfg(feature = "llvm")]
mod native {
    use super::{CliError, check, failed};
    use novar_codegen::{OptLevel, build_executable, build_object, llvm_ir, optimize_ir};
    use novar_mlir::JitValue;
    use std::path::Path;

    fn report(err: novar_codegen::BuildError, file: &str) -> CliError {
        if let novar_codegen::BuildError::Diagnostics(diags) = &err {
            let src = std::fs::read_to_string(file).unwrap_or_default();
            eprint!("{}", novar_diagnostics::render(file, &src, diags, false));
        }
        CliError::Failed(err.to_string())
    }

    fn jit_arg(raw: &str) -> JitValue {
        if let Ok(i) = raw.parse::<i64>() {
            JitValue::Int(i)
        } else if let Ok(x) = raw.parse::<f64>() {
            JitValue::Float(x)
        } else {
            JitValue::Bool(raw == "true")
        }
    }

    pub fn jit(file: &str, func: &str, args: &[&str]) -> Result<(), CliError> {
        let hir = check(file)?;
        let args: Vec<_> = args.iter().map(|a| jit_arg(a)).collect();
        match novar_mlir::jit_call(&hir, func, &args) {
            Ok(JitValue::Int(i)) => println!("{i}"),
            Ok(JitValue::Float(x)) => println!("{x}"),
            Ok(JitValue::Bool(b)) => println!("{b}"),
            Ok(JitValue::Unit) => println!("()"),
            Err(e) => return failed(e),
        }
        Ok(())
    }

    pub fn build(file: &str, flags: &[&str]) -> Result<(), CliError> {
        let (mut out, mut emit, mut opt) = (None, "exe", 2u8);
        let mut it = flags.iter();
        while let Some(flag) = it.next() {
            match (*flag, it.next()) {
                ("-o", Some(p)) => out = Some(*p),
                ("--emit", Some(k)) => emit = k,
                ("-O", Some(n)) => {
                    opt = n
                        .parse()
                        .or_else(|_| failed(format!("invalid optimisation level `{n}`")))?
                }
                _ => return Err(CliError::Usage),
            }
        }
        let hir = check(file)?;
        let stem = Path::new(file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("out");
        let opt = OptLevel(opt);
        let wrote = |p: &str| eprintln!("wrote {p}");
        match emit {
            "mlir" => {
                let text = novar_mlir::emit_mlir(&hir, false)
                    .map_err(|d| report(novar_codegen::BuildError::Diagnostics(d), file))?;
                print!("{text}");
            }
            "llvm-ir" => {
                let ir = llvm_ir(&hir).map_err(|e| report(e, file))?;
                print!("{}", optimize_ir(&ir, opt).map_err(|e| report(e, file))?);
            }
            "obj" => {
                let path = out.map_or_else(|| format!("{stem}.o"), str::to_string);
                build_object(&hir, opt, Path::new(&path)).map_err(|e| report(e, file))?;
                wrote(&path);
            }
            "exe" => {
                let path = out.map_or_else(|| stem.to_string(), str::to_string);
                build_executable(&hir, opt, Path::new(&path)).map_err(|e| report(e, file))?;
                wrote(&path);
            }
            other => return failed(format!("unknown --emit kind `{other}`")),
        }
        Ok(())
    }
}

#[cfg(not(feature = "llvm"))]
mod native {
    use super::{CliError, failed};

    const MSG: &str = "this `novar` was built without the native backend; rebuild with `--features llvm` (see scripts/llvm-env.sh)";

    pub fn jit(_: &str, _: &str, _: &[&str]) -> Result<(), CliError> {
        failed(MSG)
    }

    pub fn build(_: &str, _: &[&str]) -> Result<(), CliError> {
        failed(MSG)
    }
}

fn parse_arg(raw: &str) -> Value {
    if let Ok(i) = raw.parse::<i64>() {
        Value::Int(i)
    } else if let Ok(x) = raw.parse::<f64>() {
        Value::Float(x)
    } else if let Ok(b) = raw.parse::<bool>() {
        Value::Bool(b)
    } else {
        Value::str(raw)
    }
}

fn run(file: &str, func: &str, args: &[&str]) -> Result<(), CliError> {
    let hir = check(file)?;
    let interp = Interp::new(hir).or_else(|e| failed(e.to_string()))?;
    let value = interp
        .call(func, args.iter().map(|a| parse_arg(a)).collect())
        .or_else(|e| failed(e.to_string()))?;
    println!("{value}");
    Ok(())
}

/// Loads `{"collection": .., "id": .., "text": ..}` lines into every database the program opened.
fn seed(interp: &Interp, path: &str) -> Result<usize, CliError> {
    let data =
        std::fs::read_to_string(path).or_else(|e| failed(format!("cannot read `{path}`: {e}")))?;
    let dbs = interp.dbs();
    let mut n = 0;
    for (i, line) in data
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let doc: serde_json::Value =
            serde_json::from_str(line).or_else(|e| failed(format!("{path}:{}: {e}", i + 1)))?;
        let field = |k: &str| doc[k].as_str().map(str::to_string);
        let (Some(col), Some(id), Some(text)) = (field("collection"), field("id"), field("text"))
        else {
            return failed(format!(
                "{path}:{}: expected string fields `collection`, `id`, `text`",
                i + 1
            ));
        };
        for db in &dbs {
            db.insert(&col, &id, &text, embed_text(&text));
        }
        n += 1;
    }
    Ok(n)
}

fn serve(file: &str, flags: &[&str]) -> Result<(), CliError> {
    let mut port: u16 = 8080;
    let mut seed_file = None;
    let mut it = flags.iter();
    while let Some(flag) = it.next() {
        match (*flag, it.next()) {
            ("--port", Some(p)) => {
                port = p
                    .parse()
                    .or_else(|_| failed(format!("invalid port `{p}`")))?
            }
            ("--seed", Some(f)) => seed_file = Some(*f),
            _ => return Err(CliError::Usage),
        }
    }

    let hir = check(file)?;
    let routes = novar_server::routes(&hir);
    if routes.is_empty() {
        return failed("no `@endpoint` functions found, nothing to serve");
    }
    let interp = Arc::new(Interp::new(hir).or_else(|e| failed(e.to_string()))?);
    if let Some(path) = seed_file {
        let n = seed(&interp, path)?;
        eprintln!("seeded {n} document(s) from {path}");
    }

    let mut handlers = BTreeMap::new();
    for route in &routes {
        let (interp, name) = (interp.clone(), route.name.clone());
        handlers.insert(
            route.name.clone(),
            handler(move |payload| {
                let (interp, name) = (interp.clone(), name.clone());
                async move {
                    // The evaluator is synchronous and may be CPU-heavy; keep it off the async workers.
                    tokio::task::spawn_blocking(move || interp.call_endpoint(&name, &payload))
                        .await
                        .map_err(|e| HandlerError::new(500, format!("handler panicked: {e}")))?
                        .map_err(|e| HandlerError::new(e.status, e.message))
                }
            }),
        );
    }
    let app = novar_server::router(&routes, handlers).or_else(|e| failed(e.to_string()))?;

    let rt = tokio::runtime::Runtime::new().or_else(|e| failed(e.to_string()))?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .or_else(|e| failed(format!("cannot bind 127.0.0.1:{port}: {e}")))?;
        for r in &routes {
            eprintln!("  {:?} http://127.0.0.1:{port}{}", r.method, r.path);
        }
        eprintln!("listening on http://127.0.0.1:{port} (Ctrl-C to stop)");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .or_else(|e| failed(e.to_string()))
    })
}
