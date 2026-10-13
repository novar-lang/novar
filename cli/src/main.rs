mod bundle;
mod native;
mod serve;

use std::io::IsTerminal;
use std::process::ExitCode;

use novar_diagnostics::Diagnostic;
use novar_runtime::{Interp, Value};

const USAGE: &str = "\
usage:
  novar check <file.nv>
  novar run <file.nv> <function> [args...]
  novar serve <file.nv> [--port N] [--host H] [--seed docs.jsonl] [--live]
  novar build --serve <file.nv> [-o out] [--dockerfile]   single-binary server (+ Dockerfile)
  novar jit <file.nv> <function> [args...]                 (native, needs --features llvm)
  novar build <file.nv> [-o out] [--emit exe|obj|llvm-ir|mlir] [-O 0-3]   (needs --features llvm)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match bundle::embedded() {
        Some(source) => run_bundled(&source, &args),
        None => dispatch(&args),
    };
    match result {
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

fn dispatch(args: &[String]) -> Result<(), CliError> {
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["check", file] => check(file).map(|_| ()),
        ["run", file, func, rest @ ..] => run(file, func, rest),
        ["serve", file, rest @ ..] => {
            let opts = serve::Options::parse("127.0.0.1", rest)?;
            serve::run(check(file)?, &opts)
        }
        ["build", "--serve", file, rest @ ..] => bundle::build_serve(file, rest),
        ["jit", file, func, rest @ ..] => native::jit(file, func, rest),
        ["build", file, rest @ ..] => native::build(file, rest),
        _ => Err(CliError::Usage),
    }
}

/// A binary produced by `novar build --serve`: serve the embedded program. Containers set
/// `PORT` / `HOST`; flags win over the environment.
fn run_bundled(source: &str, args: &[String]) -> Result<(), CliError> {
    let flags: Vec<&str> = args.iter().map(String::as_str).collect();
    let mut opts = serve::Options::parse(
        &std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
        &flags,
    )?;
    if !flags.contains(&"--port") {
        if let Ok(p) = std::env::var("PORT") {
            opts.port = p
                .parse()
                .or_else(|_| failed(format!("invalid PORT `{p}`")))?;
        }
    }
    opts.live |= std::env::var("NOVAR_LIVE").is_ok_and(|v| v == "1");
    serve::run(check_source("service.nv", source)?, &opts)
}

#[derive(Debug)]
pub enum CliError {
    Usage,
    Failed(String),
}

pub fn failed<T>(msg: impl Into<String>) -> Result<T, CliError> {
    Err(CliError::Failed(msg.into()))
}

/// Parses and checks `file`, printing diagnostics. Fails if any is an error.
pub fn check(file: &str) -> Result<novar_hir::hir::Program, CliError> {
    let src =
        std::fs::read_to_string(file).or_else(|e| failed(format!("cannot read `{file}`: {e}")))?;
    check_source(file, &src)
}

pub fn check_source(name: &str, src: &str) -> Result<novar_hir::hir::Program, CliError> {
    let (program, mut diags): (_, Vec<Diagnostic>) = novar_parser::parse(src);
    // Semantic analysis also runs on recovered trees, so users see all errors at once.
    let mut hir = None;
    if let Some(program) = program {
        let (h, d) = novar_hir::lower(&program);
        diags.extend(d);
        hir = Some(h);
    }
    eprint!(
        "{}",
        novar_diagnostics::render(name, src, &diags, std::io::stderr().is_terminal())
    );
    let errors = diags.iter().filter(|d| d.is_error()).count();
    match (errors, hir) {
        (0, Some(h)) => {
            eprintln!("`{name}` ok");
            Ok(h)
        }
        _ => failed(format!("{errors} error(s) in `{name}`")),
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
