mod bundle;
mod native;
mod serve;

use std::io::IsTerminal;
use std::path::Path;
use std::process::ExitCode;

use novar_diagnostics::Diagnostic;
use novar_runtime::{Interp, Value};

const USAGE: &str = "\
usage:
  novar check <file.nv>
  novar run <file.nv> <function> [args...] [--quantize int8]
  novar lsp                                                language server over stdio
  novar serve <file.nv> [--port N] [--host H] [--seed docs.jsonl] [--live] [--quantize int8]
  novar build --serve <file.nv> [-o out] [--dockerfile]   single-binary server (+ Dockerfile)
  novar jit <file.nv> <function> [args...]                 (native, needs --features llvm)
  novar build <file.nv> [-o out] [--emit exe|obj|llvm-ir|mlir] [-O 0-3]   (needs --features llvm)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match bundle::embedded() {
        Some(bundle) => run_bundled(&bundle, &args),
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
        ["lsp"] => novar_lsp::run_stdio().or_else(|e| failed(e.to_string())),
        ["jit", file, func, rest @ ..] => native::jit(file, func, rest),
        ["build", file, rest @ ..] => native::build(file, rest),
        _ => Err(CliError::Usage),
    }
}

/// A binary produced by `novar build --serve`: serve the embedded program. Containers set
/// `PORT` / `HOST`; flags win over the environment.
fn run_bundled(bundle: &bundle::Bundle, args: &[String]) -> Result<(), CliError> {
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
    let loaded = bundle.load();
    serve::run(lower_loaded(&loaded)?.0, &opts)
}

/// Splits `--quantize int8|none` out of `args`.
pub fn take_quantize<'a>(
    args: &[&'a str],
) -> Result<(novar_hir::Quantization, Vec<&'a str>), CliError> {
    let mut quant = novar_hir::Quantization::None;
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if *a == "--quantize" {
            quant = match it.next().copied() {
                Some("int8") => novar_hir::Quantization::Int8,
                Some("none") => novar_hir::Quantization::None,
                other => {
                    return failed(format!(
                        "--quantize takes `int8` or `none`, not {}",
                        other.unwrap_or("nothing")
                    ));
                }
            };
        } else {
            rest.push(*a);
        }
    }
    Ok((quant, rest))
}

#[derive(Debug)]
pub enum CliError {
    Usage,
    Failed(String),
}

pub fn failed<T>(msg: impl Into<String>) -> Result<T, CliError> {
    Err(CliError::Failed(msg.into()))
}

#[derive(Debug)]
pub struct Checked {
    pub hir: novar_hir::hir::Program,
    /// Everything needed to rebuild the program without the source tree.
    pub bundle: bundle::Bundle,
}

/// Type-checks a loaded program, printing every diagnostic against the file it belongs to.
fn lower_loaded(
    loaded: &novar_modules::Loaded,
) -> Result<(novar_hir::hir::Program, usize), CliError> {
    let (hir, diags) = novar_hir::lower(&loaded.program);
    let all: Vec<Diagnostic> = loaded.diagnostics.iter().cloned().chain(diags).collect();
    eprint!(
        "{}",
        loaded.map.render(&all, std::io::stderr().is_terminal())
    );
    let errors = all.iter().filter(|d| d.is_error()).count();
    if errors > 0 {
        let name = loaded
            .map
            .files
            .first()
            .map_or("program", |f| f.name.as_str());
        return failed(format!("{errors} error(s) in `{name}`"));
    }
    Ok((hir, all.len()))
}

/// Loads `file` with its modules and the dependencies of its project (if it has a `novar.toml`),
/// then type-checks everything. Errors from any file are printed against that file.
pub fn check_project(file: &str) -> Result<Checked, CliError> {
    let path = Path::new(file);
    let deps = match novar_pkg::Project::discover(path) {
        Ok(Some(project)) => project.dependency_packages().or_else(failed)?,
        Ok(None) => Vec::new(),
        Err(e) => return failed(e),
    };
    let loaded = novar_modules::load(path, &deps).or_else(failed)?;
    let (hir, _) = lower_loaded(&loaded)?;
    eprintln!("`{file}` ok");
    let bundle = bundle::Bundle::new(file, &loaded, &deps);
    Ok(Checked { hir, bundle })
}

/// Parses and checks `file`, printing diagnostics. Fails if any is an error.
pub fn check(file: &str) -> Result<novar_hir::hir::Program, CliError> {
    check_project(file).map(|c| c.hir)
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
    let (quant, args) = take_quantize(args)?;
    let hir = check(file)?;
    let interp = Interp::new(hir).or_else(|e| failed(e.to_string()))?;
    interp.set_quantization(quant);
    let value = interp
        .call(func, args.iter().map(|a| parse_arg(a)).collect())
        .or_else(|e| failed(e.to_string()))?;
    println!("{value}");
    Ok(())
}
