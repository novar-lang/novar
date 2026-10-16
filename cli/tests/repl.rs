use std::io::Write;
use std::process::{Command, Stdio};

const TELVAX: &str = env!("CARGO_BIN_EXE_telvax");

/// Runs `telvax repl` with `input` on its standard input, as a pipe does, and returns what it
/// printed and what it complained about.
fn repl(args: &[&str], input: &str) -> (String, String) {
    let mut child = Command::new(TELVAX)
        .arg("repl")
        .args(args)
        .env(
            "TELVAX_HISTORY",
            std::env::temp_dir().join(format!("telvax-history-{}", std::process::id())),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_session_keeps_definitions_and_variables_between_lines() {
    let (out, err) = repl(
        &[],
        "let x = 5\nx * 2\nvar n = 0\nfor i in range(0, 4):\n    n = n + i\n\nn\n\
         fn double(k: Int) -> Int:\n    k * 2\n\ndouble(n)\nprint(\"hi\")\n",
    );
    assert_eq!(out, "10\n6\n12\nhi\n", "{err}");
    assert!(err.is_empty(), "{err}");
}

#[test]
fn errors_go_to_standard_error_and_the_session_goes_on() {
    let (out, err) = repl(&[], "let x = 1\nx + \"a\"\nnothing\nx + 1\n");
    assert_eq!(out, "2\n");
    assert!(err.contains("mismatched operand types"), "{err}");
    assert!(err.contains("1 | x + \"a\""), "{err}");
    assert!(err.contains("nothing"), "{err}");
}

#[test]
fn commands_show_types_variables_and_definitions() {
    let (out, err) = repl(
        &[],
        "let a = [1, 2]\nvar s = \"two\"\nfn f() -> Int:\n    1\n\n:type a\n:vars\n:defs\n:type 1 +\n:nope\n:reset\n:vars\n:quit\nunreached\n",
    );
    assert!(err.contains("no command `:nope`"), "{err}");
    assert_eq!(
        out,
        "List[Int]\na: List[Int] = [1, 2]\ns: Text = \"two\"\nfn f\n(no variables)\n".replacen(
            "(no variables)\n",
            "(everything forgotten)\n(no variables)\n",
            1
        )
    );
}

#[test]
fn files_are_loaded_from_the_command_line_and_with_load() {
    let dir = std::env::temp_dir().join(format!("telvax-repl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lib = dir.join("lib.tvx");
    std::fs::write(
        &lib,
        "type Pair:\n    a: Int\n    b: Int\n\nfn sum(p: Pair) -> Int:\n    p.a + p.b\n",
    )
    .unwrap();
    let (out, err) = repl(&[lib.to_str().unwrap()], "sum(Pair(a: 1, b: 2))\n");
    assert!(out.ends_with("3\n"), "{out} {err}");
    let (out, err) = repl(
        &[],
        &format!(":load {}\nsum(Pair(a: 3, b: 4))\n", lib.display()),
    );
    assert!(out.ends_with("7\n"), "{out} {err}");
    let (_, err) = repl(&[], ":load /nonexistent/file.tvx\n");
    assert!(err.contains("cannot read"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn what_the_program_may_do_can_be_limited() {
    let (out, err) = repl(
        &["--allow", "none"],
        "std.fs.read(\"/etc/hostname\")\n1 + 1\n",
    );
    assert!(err.contains("needs the `fs.read` capability"), "{err}");
    assert_eq!(out, "2\n");
}
