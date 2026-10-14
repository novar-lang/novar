//! `telvax-migrate`: converts Telvax source from the old Rust-style syntax to the Python-style one.
//!
//! This tool exists only for the move to the new syntax. It carries a copy of the old lexer and
//! parser, reads a file with them and prints the tree in the new syntax, keeping the comments.
//!
//! ```text
//! telvax-migrate FILE...        rewrite the files in place
//! telvax-migrate --stdout FILE  print the result instead
//! telvax-migrate --batch        read sources separated by a \x01 line on stdin, write one
//!                               "STATUS\ntext" per source, also separated by \x01 lines
//! ```

#[allow(dead_code)]
mod legacy_lexer;
#[allow(dead_code)]
mod legacy_parser;
mod printer;
#[allow(dead_code)]
mod py_desugar;
#[allow(dead_code)]
mod py_lexer;
#[allow(dead_code)]
mod py_parser;

use std::io::{Read, Write};

/// Converts one source. `Ok((text, warnings))`, or the errors of the old parser.
pub fn migrate(src: &str, from_py: bool) -> Result<(String, Vec<String>), String> {
    let (program, errors) = if from_py {
        py_parser::parse(src)
    } else {
        legacy_parser::parse(src)
    };
    if !errors.is_empty() {
        let first = &errors[0];
        return Err(format!(
            "old syntax: {} at byte {}",
            first.message, first.span.start
        ));
    }
    let program = program.ok_or("old syntax: nothing parsed")?;
    let marker = if from_py { "#" } else { "//" };
    let (text, mut warnings) = printer::Printer::new(src, marker).program(&program);
    let (_, new_errors) = telvax_parser::parse(&text);
    if let Some(e) = new_errors.first() {
        warnings.push(format!(
            "new syntax does not parse: {} at byte {}",
            e.message, e.span.start
        ));
    }
    Ok((text, warnings))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let from_py = args.iter().any(|a| a == "--from-py");
    args.retain(|a| a != "--from-py");
    if args.first().map(String::as_str) == Some("--batch") {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .expect("read stdin");
        let mut out = String::new();
        for (i, src) in input.split("\u{1}\n").enumerate() {
            if i > 0 {
                out.push_str("\u{1}\n");
            }
            match migrate(src, from_py) {
                Ok((text, warnings)) if warnings.is_empty() => {
                    out.push_str("OK\n");
                    out.push_str(&text);
                }
                Ok((text, warnings)) => {
                    out.push_str("WARN ");
                    out.push_str(&warnings.join("; ").replace('\n', " "));
                    out.push('\n');
                    out.push_str(&text);
                }
                Err(e) => {
                    out.push_str("ERR ");
                    out.push_str(&e.replace('\n', " "));
                    out.push('\n');
                }
            }
        }
        std::io::stdout()
            .write_all(out.as_bytes())
            .expect("write stdout");
        return;
    }
    let to_stdout = args.first().map(String::as_str) == Some("--stdout");
    let files = if to_stdout { &args[1..] } else { &args[..] };
    let mut failed = false;
    for file in files {
        let src = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{file}: {e}");
                failed = true;
                continue;
            }
        };
        match migrate(&src, from_py) {
            Ok((text, warnings)) => {
                for w in &warnings {
                    eprintln!("{file}: warning: {w}");
                }
                if to_stdout {
                    print!("{text}");
                } else if let Err(e) = std::fs::write(file, text) {
                    eprintln!("{file}: {e}");
                    failed = true;
                }
            }
            Err(e) => {
                eprintln!("{file}: {e}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
