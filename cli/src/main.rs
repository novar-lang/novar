mod bundle;
mod check;
mod doctor;
mod native;
mod serve;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use telvax_diagnostics::Diagnostic;
use telvax_runtime::{Interp, Value};

const USAGE: &str = "\
usage:
  telvax check [--json] [file.tvx...]                       no file: the package in this directory
  telvax lint <file.tvx> [--deny-warnings]                    unused variables and functions
  telvax fmt [--check] <file.tvx>...                          format in place (`-` reads stdin)
  telvax test [file.tvx|dir] [name] [--allow list]          run the `test_*` functions (a directory: `test_*.tvx`, `*_test.tvx`)
  telvax run <file.tvx> [function] [args...] [--allow list] [--quantize int8|int8-channel|int4-channel] [--device gpu]
  telvax doctor                                             what this build can do and how it is set up
  telvax lsp                                                language server over stdio
  telvax worker [--listen host:port] [--token T]            run calls for a coordinator
  telvax serve <file.tvx> [--port N] [--host H] [--seed docs.jsonl] [--live] [--allow list] [--quantize int8|int8-channel|int4-channel] [--device gpu]
  telvax build --serve <file.tvx> [-o out] [--dockerfile]   single-binary server (+ Dockerfile)
  telvax jit <file.tvx> <function> [args...]                 (native, needs --features llvm)
  telvax build <file.tvx> [-o out] [--emit exe|obj|llvm-ir|mlir|ptx|stablehlo] [-O 0-3] [--sm sm_80]
                                                    (all but stablehlo need --features llvm)

--allow takes a list of fs.read, fs.write, fs, env, net, all, none. Without it everything is allowed.";

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
        Err(CliError::Unknown(word)) => {
            eprint!("error: unknown command `{word}`");
            if let Some(near) = doctor::closest_command(&word) {
                eprint!("; did you mean `{near}`?");
            }
            eprintln!("\nrun `telvax --help` for the commands");
            ExitCode::from(2)
        }
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
        ["doctor"] => {
            print!("{}", doctor::report());
            Ok(())
        }
        ["check", rest @ ..] => check::run(rest),
        // Without a function name, `main` runs: `telvax run app.tvx --allow fs`.
        ["run", file, rest @ ..] => match rest {
            [func, args @ ..] if !func.starts_with('-') => run(file, func, args),
            _ => run(file, "main", rest),
        },
        // Without a path, `test` looks in this directory.
        ["test", rest @ ..] => {
            let (path, rest) = match rest {
                [path, rest @ ..] if !path.starts_with('-') => (*path, rest),
                _ => (".", rest),
            };
            let (caps, rest) = take_allow(rest)?;
            match rest.as_slice() {
                [] => test(path, None, caps),
                [filter] => test(path, Some(filter), caps),
                _ => Err(CliError::Usage),
            }
        }
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
        [word, ..] if !doctor::COMMANDS.contains(word) && !word.starts_with('-') => {
            Err(CliError::Unknown((*word).to_string()))
        }
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

/// Splits `--allow fs.read,net,...` out of `args`. Without it everything is allowed.
pub fn take_allow<'a>(args: &[&'a str]) -> Result<(telvax_runtime::Caps, Vec<&'a str>), CliError> {
    let mut caps = telvax_runtime::Caps::ALL;
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if *a == "--allow" {
            let Some(list) = it.next() else {
                return failed("--allow takes a list such as `fs.read,net`, or `none`");
            };
            caps = telvax_runtime::Caps::parse(list).or_else(failed)?;
        } else {
            rest.push(*a);
        }
    }
    Ok((caps, rest))
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
    /// The first word is not a command.
    Unknown(String),
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
    let bundle = bundle::Bundle::new(file, &loaded, &deps);
    Ok(Checked { hir, bundle })
}

/// Parses and checks `file`, printing diagnostics. Fails if any is an error.
pub fn check(file: &str) -> Result<telvax_hir::hir::Program, CliError> {
    check_project(file).map(|c| c.hir)
}

/// The values for the command-line words after the function name, read as the types the function
/// declares: a `Text` parameter takes the word as it is (so `"123"` stays text), the others take it
/// as JSON (`3`, `2.5`, `true`, `[1, 2]`).
fn call_arguments(interp: &Interp, func: &str, words: &[&str]) -> Result<Vec<Value>, CliError> {
    let Some((params, _)) = interp.signature(func) else {
        return failed(format!("no function `{func}`"));
    };
    if params.len() != words.len() {
        let list: Vec<String> = params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
        return failed(format!(
            "`{func}` takes {} argument(s) ({}), got {}",
            params.len(),
            list.join(", "),
            words.len()
        ));
    }
    params
        .iter()
        .zip(words)
        .map(|((name, ty), word)| {
            let json = if *ty == telvax_hir::Ty::str() {
                serde_json::Value::String((*word).to_string())
            } else {
                serde_json::from_str(word).or_else(|_| {
                    failed(format!(
                        "argument `{name}` of `{func}` is {ty}, and `{word}` is not a value of that type (write numbers, true, false, null, [..] or {{..}} as JSON)"
                    ))
                })?
            };
            interp
                .value_from_json(ty, &json)
                .or_else(|e| failed(format!("argument `{name}` of `{func}` is {ty}: {e}")))
        })
        .collect()
}

/// `telvax test [path] [name]`: runs every function called `test_*` that takes no arguments. A
/// test passes when it returns and fails when it stops with an error (`assert`, `assert_eq`, ...).
/// A directory runs every `test_*.tvx` and `*_test.tvx` file under it.
fn test(path: &str, filter: Option<&str>, caps: telvax_runtime::Caps) -> Result<(), CliError> {
    if !Path::new(path).is_dir() {
        let (passed, failures) = test_file(path, filter, caps)?;
        println!("{passed} passed, {failures} failed");
        return if failures > 0 { failed("") } else { Ok(()) };
    }
    let mut files = Vec::new();
    collect_test_files(Path::new(path), &mut files);
    files.sort();
    if files.is_empty() {
        return failed(format!(
            "no test files in `{path}` (a test file is named `test_*.tvx` or `*_test.tvx`)"
        ));
    }
    let (mut passed, mut failures, mut ran) = (0, 0, 0);
    for file in &files {
        let file = file.to_string_lossy();
        println!("== {file}");
        match test_file(&file, filter, caps) {
            Ok((p, f)) => {
                passed += p;
                failures += f;
                ran += 1;
            }
            Err(CliError::Failed(msg)) if msg.starts_with("no test functions") => {}
            Err(CliError::Failed(msg)) => {
                failures += 1;
                if !msg.is_empty() {
                    eprintln!("error: {msg}");
                }
            }
            Err(e) => return Err(e),
        }
    }
    if ran == 0 && failures == 0 {
        return failed(
            "no test functions found (a test is a function named `test_...` without arguments)",
        );
    }
    println!("{passed} passed, {failures} failed");
    if failures > 0 { failed("") } else { Ok(()) }
}

/// The `test_*.tvx` and `*_test.tvx` files under `dir`, skipping hidden folders, `target` and
/// `node_modules`.
fn collect_test_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if !name.starts_with('.') && name != "target" && name != "node_modules" {
                collect_test_files(&path, out);
            }
        } else if let Some(stem) = name.strip_suffix(".tvx")
            && (stem.starts_with("test_") || stem.ends_with("_test"))
        {
            out.push(path);
        }
    }
}

/// Runs the tests of one file and returns how many passed and failed.
fn test_file(
    file: &str,
    filter: Option<&str>,
    caps: telvax_runtime::Caps,
) -> Result<(usize, usize), CliError> {
    let checked = check_project(file)?;
    let mut names: Vec<String> = checked
        .hir
        .fns
        .iter()
        .filter(|f| f.name.starts_with("test_"))
        .filter(|f| filter.is_none_or(|n| f.name.contains(n)))
        .map(|f| f.name.clone())
        .collect();
    names.sort();
    if names.is_empty() {
        return failed("no test functions found (a test is a function named `test_...`)");
    }
    let interp = Interp::with_caps(checked.hir, false, caps).or_else(|e| failed(e.to_string()))?;
    let mut failures = 0;
    let mut total = 0;
    for name in &names {
        let runs = match test_runs(&interp, name) {
            Ok(runs) => runs,
            Err(e) => {
                total += 1;
                failures += 1;
                println!("FAILED  {name}: {e}");
                continue;
            }
        };
        for (label, given) in runs {
            total += 1;
            match run_test(&interp, name, given) {
                Ok(()) => println!("ok      {label}"),
                Err(e) => {
                    failures += 1;
                    println!("FAILED  {label}: {e}");
                }
            }
        }
    }
    Ok((total - failures, failures))
}

/// The runs of a test: one, or one per case when there is a `cases_<name>()` function, where the
/// name is the test's without `test_`. A case gives the test's first parameters (a tuple gives
/// several); the others are fixtures.
fn test_runs(interp: &Interp, name: &str) -> Result<Vec<(String, Vec<Value>)>, String> {
    let cases = format!("cases_{}", name.strip_prefix("test_").unwrap_or(name));
    let Some((cases_params, ret)) = interp.signature(&cases) else {
        return Ok(vec![(name.to_string(), Vec::new())]);
    };
    let telvax_hir::Ty::Vec(case_ty) = &ret else {
        return Err(format!(
            "`{cases}` must return a list of cases: write `fn {cases}() -> List[...]`"
        ));
    };
    if !cases_params.is_empty() {
        return Err(format!("`{cases}` takes no arguments"));
    }
    let given = match &**case_ty {
        telvax_hir::Ty::Tuple(parts) => parts.clone(),
        one => vec![one.clone()],
    };
    let params = interp.signature(name).map(|(p, _)| p).unwrap_or_default();
    if given.len() > params.len() {
        return Err(format!(
            "a case of `{cases}` has {} value(s) but `{name}` takes {} parameter(s)",
            given.len(),
            params.len()
        ));
    }
    for ((param, ty), case) in params.iter().zip(&given) {
        if !ty.accepts(case) {
            return Err(format!(
                "`{name}` takes `{param}: {ty}` but the cases of `{cases}` give `{case}`"
            ));
        }
    }
    let Value::Vec(list) = interp
        .call(&cases, Vec::new())
        .map_err(|e| format!("`{cases}` failed: {e}"))?
    else {
        return Err(format!("`{cases}` must return a list"));
    };
    if list.is_empty() {
        return Err(format!("`{cases}` has no cases"));
    }
    Ok(list
        .snapshot()
        .into_iter()
        .map(|case| {
            let values = match (&case, given.len()) {
                (Value::Tuple(items), n) if n > 1 => items.to_vec(),
                _ => vec![case.clone()],
            };
            (format!("{name}[{case}]"), values)
        })
        .collect())
}

/// Runs one test. Each parameter of the test that a case does not give is a fixture: `fn test_x(db: Db)` gets the value of
/// `fixture_db()`, made again for every test. A `teardown_db(db)` function, if there is one, runs
/// after the test whether it passed or not.
fn run_test(interp: &Interp, name: &str, given: Vec<Value>) -> Result<(), String> {
    let params = interp.signature(name).map(|(p, _)| p).unwrap_or_default();
    let mut made: Vec<(String, Value)> = Vec::new();
    let result = params
        .iter()
        .enumerate()
        .map(|(i, (p, ty))| match given.get(i) {
            Some(value) => Ok(value.clone()),
            None => fixture(interp, p, Some(ty), &mut made, &mut Vec::new()),
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|args| {
            interp
                .call(name, args)
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
    let mut teardown = Ok(());
    for (fixture_name, value) in made.iter().rev() {
        let t = format!("teardown_{fixture_name}");
        let Some((params, _)) = interp.signature(&t) else {
            continue;
        };
        let done = if params.len() == 1 {
            interp
                .call(&t, vec![value.clone()])
                .map(|_| ())
                .map_err(|e| format!("`{t}` failed: {e}"))
        } else {
            Err(format!(
                "`{t}` takes the fixture to clean up: write `fn {t}({fixture_name}: ...)`"
            ))
        };
        if teardown.is_ok() {
            teardown = done;
        }
    }
    result.and(teardown)
}

/// The value of the fixture `name`, made once per test; `made` holds the ones already made, in order.
fn fixture(
    interp: &Interp,
    name: &str,
    wanted: Option<&telvax_hir::Ty>,
    made: &mut Vec<(String, Value)>,
    stack: &mut Vec<String>,
) -> Result<Value, String> {
    if let Some((_, v)) = made.iter().find(|(n, _)| n == name) {
        return Ok(v.clone());
    }
    if stack.iter().any(|n| n == name) {
        return Err(format!(
            "fixture `{name}` needs itself ({} -> {name})",
            stack.join(" -> ")
        ));
    }
    let function = format!("fixture_{name}");
    let Some((params, ret)) = interp.signature(&function) else {
        return Err(format!(
            "the test asks for `{name}` but there is no fixture: write `fn {function}() -> ...`"
        ));
    };
    if let Some(ty) = wanted
        && !ty.accepts(&ret)
    {
        return Err(format!(
            "the test takes `{name}: {ty}` but `{function}` returns `{ret}`"
        ));
    }
    stack.push(name.to_string());
    let args = params
        .iter()
        .map(|(p, ty)| fixture(interp, p, Some(ty), made, stack))
        .collect::<Result<Vec<_>, _>>()?;
    stack.pop();
    let value = interp
        .call(&function, args)
        .map_err(|e| format!("`{function}` failed: {e}"))?;
    made.push((name.to_string(), value.clone()));
    Ok(value)
}

fn run(file: &str, func: &str, args: &[&str]) -> Result<(), CliError> {
    let (caps, args) = take_allow(args)?;
    let (quant, args) = take_quantize(&args)?;
    let (device, args) = take_device(&args)?;
    let (workers, args) = take_workers(&args);
    let checked = check_project(file)?;
    let mut names: Vec<String> = checked.hir.fns.iter().map(|f| f.name.clone()).collect();
    names.sort();
    let interp = Interp::with_caps(checked.hir, false, caps).or_else(|e| failed(e.to_string()))?;
    if interp.signature(func).is_none() {
        return failed(format!(
            "`{file}` has no function `{func}`; it has: {}",
            names.join(", ")
        ));
    }
    interp.set_quantization(quant);
    interp.set_device(device);
    if let Some(workers) = workers {
        attach_cluster(&interp, workers, &checked.bundle)?;
    }
    let values = call_arguments(&interp, func, &args)?;
    let value = interp
        .call(func, values)
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
