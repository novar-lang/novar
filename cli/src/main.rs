use std::io::IsTerminal;
use std::process::ExitCode;

const USAGE: &str = "usage: novar check <file.nv>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, file] if cmd == "check" => check(file),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn check(file: &str) -> ExitCode {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read `{file}`: {e}");
            return ExitCode::from(2);
        }
    };

    let (program, mut diags) = novar_parser::parse(&src);
    // Semantic analysis also runs on recovered trees, so users see all errors at once.
    if let Some(program) = program {
        diags.extend(novar_hir::lower(&program).1);
    }

    let color = std::io::stderr().is_terminal();
    eprint!("{}", novar_diagnostics::render(file, &src, &diags, color));

    let errors = diags.iter().filter(|d| d.is_error()).count();
    if errors > 0 {
        eprintln!("{errors} error(s) in `{file}`");
        ExitCode::FAILURE
    } else {
        eprintln!("`{file}` ok");
        ExitCode::SUCCESS
    }
}
