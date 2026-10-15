mod bundle;
mod native;
mod serve;

use std::io::IsTerminal;
use std::path::Path;
use std::process::ExitCode;

use telvax_diagnostics::Diagnostic;
use telvax_runtime::{Interp, Value};

const USAGE: &str = "\
usage:
  telvax check <file.tvx>
  telvax lint <file.tvx> [--deny-warnings]                    unused variables and functions
  telvax fmt [--check] <file.tvx>...                          format in place (`-` reads stdin)
  telvax test <file.tvx> [name]                             run the `test_*` functions
  telvax run <file.tvx> <function> [args...] [--quantize int8|int8-channel|int4-channel] [--device gpu]
  telvax lsp                                                language server over stdio
  telvax worker [--listen host:port] [--token T]            run calls for a coordinator
  telvax serve <file.tvx> [--port N] [--host H] [--seed docs.jsonl] [--live] [--quantize int8|int8-channel|int4-channel] [--device gpu]
  telvax build --serve <file.tvx> [-o out] [--dockerfile]   single-binary server (+ Dockerfile)
  telvax jit <file.tvx> <function> [args...]                 (native, needs --features llvm)
  telvax build <file.tvx> [-o out] [--emit exe|obj|llvm-ir|mlir|ptx|stablehlo] [-O 0-3] [--sm sm_80]
                                                    (all but stablehlo need --features llvm)";

/// Stack for the thread that does the work. Checking and running recurse once per level of a
/// program's nesting (a sum of a few hundred terms, calls inside calls), and the 8 MB of a main
/// thread is not enough for the deepest programs. Only the pages that are used are ever mapped.
const STACK_BYTES: usize = 512 * 1024 * 1024;

fn main() -> ExitCode {
    match std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(real_main)
    {
        Ok(worker) => worker.join().unwrap_or(ExitCode::FAILURE),
        Err(_) => real_main(),
    }
}

fn real_main() -> ExitCode {
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
        ["fmt", rest @ ..] => fmt(rest),
        ["lint", file, rest @ ..] => lint(file, rest),
        ["--version" | "-V" | "version"] => {
            println!("telvax {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        ["--help" | "-h" | "help"] => {
            println!("{USAGE}");
            Ok(())
        }
        ["check", file] => check(file).map(|_| ()),
        ["run", file, func, rest @ ..] => run(file, func, rest),
        ["test", file] => test(file, None),
        ["test", file, filter] => test(file, Some(filter)),
        ["serve", file, rest @ ..] => {
            let (workers, rest) = take_workers(rest);
            let opts = serve::Options::parse("127.0.0.1", &rest)?;
            let checked = check_project(file)?;
            serve::run(checked.hir, &opts, workers.map(|w| (w, checked.bundle)))
        }
        ["worker", rest @ ..] => worker(rest),
        ["build", "--serve", file, rest @ ..] => bundle::build_serve(file, rest),
        ["lsp"] => telvax_lsp::run_stdio().or_else(|e| failed(e.to_string())),
        ["jit", file, func, rest @ ..] => native::jit(file, func, rest),
        ["build", file, rest @ ..] => native::build(file, rest),
        _ => Err(CliError::Usage),
    }
}

/// A binary produced by `telvax build --serve`: serve the embedded program. Containers set
/// `PORT` / `HOST`; flags win over the environment.
/// `telvax worker [--listen host:port] [--token T]`: runs calls sent by a coordinator.
fn worker(flags: &[&str]) -> Result<(), CliError> {
    let mut listen = "127.0.0.1:7077".to_string();
    let mut token = std::env::var("TELVAX_CLUSTER_TOKEN").ok();
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match (*f, it.next()) {
            ("--listen", Some(a)) => listen = a.to_string(),
            ("--token", Some(t)) => token = Some(t.to_string()),
            _ => return Err(CliError::Usage),
        }
    }
    telvax_cluster::worker::serve(&listen, token).or_else(failed)
}

fn run_bundled(bundle: &bundle::Bundle, args: &[String]) -> Result<(), CliError> {
    let flags: Vec<&str> = args.iter().map(String::as_str).collect();
    let mut opts = serve::Options::parse(
        &std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
        &flags,
    )?;
    if !flags.contains(&"--port")
        && let Ok(p) = std::env::var("PORT")
    {
        opts.port = p
            .parse()
            .or_else(|_| failed(format!("invalid PORT `{p}`")))?;
    }
    opts.live |= std::env::var("TELVAX_LIVE").is_ok_and(|v| v == "1");
    let loaded = bundle.load();
    serve::run(lower_loaded(&loaded)?.0, &opts, None)
}

/// Splits `--workers host:port,host:port` out of `args`; `TELVAX_WORKERS` is the default.
pub fn take_workers<'a>(args: &[&'a str]) -> (Option<Vec<String>>, Vec<&'a str>) {
    let mut workers = std::env::var("TELVAX_WORKERS")
        .ok()
        .filter(|w| !w.is_empty());
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if *a == "--workers" {
            workers = it.next().map(|w| w.to_string());
        } else {
            rest.push(*a);
        }
    }
    (
        workers.map(|w| {
            w.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        }),
        rest,
    )
}

/// Sends `std.dist.map` calls to `workers`.
pub fn attach_cluster(
    interp: &Interp,
    workers: Vec<String>,
    image: &bundle::Bundle,
) -> Result<(), CliError> {
    let token = std::env::var("TELVAX_CLUSTER_TOKEN").ok();
    let exec = telvax_cluster::ClusterExecutor::new(workers, image, token).or_else(failed)?;
    interp.set_executor(std::sync::Arc::new(exec));
    Ok(())
}

/// Splits `--device cpu|gpu` out of `args`.
pub fn take_device<'a>(
    args: &[&'a str],
) -> Result<(telvax_runtime::Device, Vec<&'a str>), CliError> {
    let mut device = telvax_runtime::Device::Cpu;
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if *a == "--device" {
            device = match it.next().copied() {
                Some("gpu") => telvax_runtime::Device::Gpu,
                Some("cpu") => telvax_runtime::Device::Cpu,
                other => {
                    return failed(format!(
                        "--device takes `cpu` or `gpu`, not {}",
                        other.unwrap_or("nothing")
                    ));
                }
            };
        } else {
            rest.push(*a);
        }
    }
    Ok((device, rest))
}

/// Splits `--quantize int8|int8-channel|int4-channel|none` out of `args`.
pub fn take_quantize<'a>(
    args: &[&'a str],
) -> Result<(telvax_hir::Quantization, Vec<&'a str>), CliError> {
    let mut quant = telvax_hir::Quantization::None;
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if *a == "--quantize" {
            quant = match it.next().copied() {
                Some("int8") => telvax_hir::Quantization::Int8,
                Some("int8-channel") => telvax_hir::Quantization::Int8Channel,
                Some("int4-channel") => telvax_hir::Quantization::Int4Channel,
                Some("none") => telvax_hir::Quantization::None,
                other => {
                    return failed(format!(
                        "--quantize takes `int8`, `int8-channel`, `int4-channel` or `none`, not {}",
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
    pub hir: telvax_hir::hir::Program,
    /// Everything needed to rebuild the program without the source tree.
    pub bundle: bundle::Bundle,
}

/// Type-checks a loaded program, printing every diagnostic against the file it belongs to.
fn lower_loaded(
    loaded: &telvax_modules::Loaded,
) -> Result<(telvax_hir::hir::Program, usize), CliError> {
    let (hir, diags) = telvax_hir::lower(&loaded.program);
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

/// Loads `file` with its modules and the dependencies of its project (if it has a `telvax.toml`),
/// then type-checks everything. Errors from any file are printed against that file.
pub fn check_project(file: &str) -> Result<Checked, CliError> {
    let path = Path::new(file);
    let deps = match telvax_pkg::Project::discover(path) {
        Ok(Some(project)) => project.dependency_packages().or_else(failed)?,
        Ok(None) => Vec::new(),
        Err(e) => return failed(e),
    };
    let loaded = telvax_modules::load(path, &deps).or_else(failed)?;
    let (hir, _) = lower_loaded(&loaded)?;
    eprintln!("`{file}` ok");
    let bundle = bundle::Bundle::new(file, &loaded, &deps);
    Ok(Checked { hir, bundle })
}

/// Parses and checks `file`, printing diagnostics. Fails if any is an error.
pub fn check(file: &str) -> Result<telvax_hir::hir::Program, CliError> {
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

/// `telvax test <file> [name]`: runs every function called `test_*` that takes no arguments. A
/// test passes when it returns and fails when it stops with an error (`assert`, `assert_eq`, ...).
fn test(file: &str, filter: Option<&str>) -> Result<(), CliError> {
    let checked = check_project(file)?;
    let mut names: Vec<String> = checked
        .hir
        .fns
        .iter()
        .filter(|f| f.name.starts_with("test_") && f.params.is_empty())
        .filter(|f| filter.is_none_or(|n| f.name.contains(n)))
        .map(|f| f.name.clone())
        .collect();
    names.sort();
    if names.is_empty() {
        return failed(
            "no test functions found (a test is a function named `test_...` without arguments)",
        );
    }
    let interp = Interp::new(checked.hir).or_else(|e| failed(e.to_string()))?;
    let mut failures = 0;
    for name in &names {
        match interp.call(name, vec![]) {
            Ok(_) => println!("ok      {name}"),
            Err(e) => {
                failures += 1;
                println!("FAILED  {name}: {e}");
            }
        }
    }
    println!("{} passed, {failures} failed", names.len() - failures);
    if failures > 0 { failed("") } else { Ok(()) }
}

fn run(file: &str, func: &str, args: &[&str]) -> Result<(), CliError> {
    let (quant, args) = take_quantize(args)?;
    let (device, args) = take_device(&args)?;
    let (workers, args) = take_workers(&args);
    let checked = check_project(file)?;
    let interp = Interp::new(checked.hir).or_else(|e| failed(e.to_string()))?;
    interp.set_quantization(quant);
    interp.set_device(device);
    if let Some(workers) = workers {
        attach_cluster(&interp, workers, &checked.bundle)?;
    }
    let value = interp
        .call(func, args.iter().map(|a| parse_arg(a)).collect())
        .or_else(|e| failed(e.to_string()))?;
    // A function that returns nothing has nothing to show.
    if !matches!(value, Value::Unit) {
        println!("{value}");
    }
    Ok(())
}

/// `telvax fmt [--check] <file>...`: rewrites files in place, or only reports with `--check`.
fn fmt(args: &[&str]) -> Result<(), CliError> {
    let check = args.contains(&"--check");
    let files: Vec<&str> = args.iter().copied().filter(|a| *a != "--check").collect();
    if files.is_empty() {
        return Err(CliError::Usage);
    }
    let mut unformatted = 0;
    for file in files {
        if file == "-" {
            let mut src = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut src)
                .or_else(|e| failed(format!("cannot read stdin: {e}")))?;
            print!("{}", telvax_fmt::format(&src).or_else(failed)?);
            continue;
        }
        let src = std::fs::read_to_string(file)
            .or_else(|e| failed(format!("cannot read `{file}`: {e}")))?;
        let formatted = telvax_fmt::format(&src).or_else(|e| failed(format!("`{file}`: {e}")))?;
        if formatted == src {
            continue;
        }
        if check {
            println!("would reformat {file}");
            unformatted += 1;
        } else {
            std::fs::write(file, &formatted)
                .or_else(|e| failed(format!("cannot write `{file}`: {e}")))?;
            println!("formatted {file}");
        }
    }
    if unformatted > 0 {
        return failed(format!("{unformatted} file(s) need formatting"));
    }
    Ok(())
}

/// `telvax lint <file> [--deny-warnings]`: type-checks, then reports unused code.
fn lint(file: &str, flags: &[&str]) -> Result<(), CliError> {
    let deny = match flags {
        [] => false,
        ["--deny-warnings"] => true,
        _ => return Err(CliError::Usage),
    };
    let path = Path::new(file);
    let deps = match telvax_pkg::Project::discover(path) {
        Ok(Some(project)) => project.dependency_packages().or_else(failed)?,
        Ok(None) => Vec::new(),
        Err(e) => return failed(e),
    };
    let loaded = telvax_modules::load(path, &deps).or_else(failed)?;
    let (hir, _) = lower_loaded(&loaded)?;
    let warnings = telvax_lint::lint(&hir);
    eprint!(
        "{}",
        loaded
            .map
            .render(&warnings, std::io::stderr().is_terminal())
    );
    if warnings.is_empty() {
        eprintln!("`{file}`: no warnings");
        return Ok(());
    }
    eprintln!("`{file}`: {} warning(s)", warnings.len());
    if deny {
        Err(CliError::Failed(String::new()))
    } else {
        Ok(())
    }
}
