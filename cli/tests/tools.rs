use std::process::Command;

const TELVAX: &str = env!("CARGO_BIN_EXE_telvax");

fn scratch(name: &str, text: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("telvax-tools-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{name}.tvx"));
    std::fs::write(&file, text).unwrap();
    file
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(TELVAX).args(args).output().unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn fmt_rewrites_in_place_and_check_reports() {
    let messy = "fn main() -> i32:\n  let x=1\n  x+2\n";
    let tidy = "fn main() -> i32:\n    let x = 1\n    x + 2\n";
    let file = scratch("messy", messy);
    let path = file.to_str().unwrap();
    let (ok, out, _) = run(&["fmt", "--check", path]);
    assert!(!ok && out.contains("would reformat"), "{out}");
    assert_eq!(std::fs::read_to_string(&file).unwrap(), messy);

    let (ok, out, err) = run(&["fmt", path]);
    assert!(ok, "{err}");
    assert!(out.contains("formatted"));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), tidy);
    let (ok, _, _) = run(&["fmt", "--check", path]);
    assert!(ok, "formatted code must pass --check");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn fmt_refuses_broken_source() {
    let file = scratch("broken", "fn main():\n    let s = \"open\n");
    let (ok, _, err) = run(&["fmt", file.to_str().unwrap()]);
    assert!(!ok && err.contains("cannot format"), "{err}");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn lint_reports_unused_code() {
    let file = scratch(
        "lintme",
        r#"
fn _helper() -> i32:
    1

fn main() -> i32:
    let unused = 2
    3
"#,
    );
    let path = file.to_str().unwrap();
    let (ok, _, err) = run(&["lint", path]);
    assert!(ok, "{err}");
    assert!(
        err.contains("unused variable `unused`") && err.contains("`_helper` is never used"),
        "{err}"
    );
    let (ok, _, _) = run(&["lint", path, "--deny-warnings"]);
    assert!(!ok);
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn test_runs_the_test_functions() {
    let src = "fn double(x: Int) -> Int:\n    x * 2\n\nfn test_double():\n    assert_eq(double(2), 4)\n\nfn test_wrong():\n    assert_eq(double(3), 7)\n";
    let file = scratch("tests", src);
    let path = file.to_str().unwrap();
    let (ok, out, err) = run(&["test", path]);
    assert!(!ok, "a failing test must fail the run");
    assert!(out.contains("ok      test_double"), "{out}");
    assert!(
        out.contains("FAILED  test_wrong") && out.contains("left is `6`, right is `7`"),
        "{out}"
    );
    assert!(out.contains("1 passed, 1 failed"), "{out}{err}");
    let (ok, out, _) = run(&["test", path, "double"]);
    assert!(ok && out.contains("1 passed, 0 failed"), "{out}");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn deeply_nested_programs_are_checked_not_crashed() {
    // Calls inside calls, within the limit on brackets, and a sum of many terms: the checker
    // recurses once per level and used to overflow the 8 MB stack of a main thread.
    let calls = format!(
        "fn f(x: Int) -> Int:\n    x\n\nfn main() -> i32:\n    let y = {}1{}\n    0\n",
        "f(".repeat(190),
        ")".repeat(190)
    );
    let sum = format!(
        "fn main() -> i32:\n    let y = {}1\n    0\n",
        "1 + ".repeat(5_000)
    );
    for (name, src) in [("calls", calls), ("sum", sum)] {
        let file = scratch(&format!("deep-{name}"), &src);
        let (ok, _, err) = run(&["check", file.to_str().unwrap()]);
        assert!(ok, "{name}: {err}");
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }
}

#[test]
fn nesting_past_the_limit_is_an_error_not_a_crash() {
    let blocks = {
        let mut s = String::from("fn f():\n");
        for level in 0..5000 {
            s.push_str(&format!("{}if True:\n", "    ".repeat(level + 1)));
        }
        s.push_str(&format!("{}pass\n", "    ".repeat(5001)));
        s
    };
    let brackets = format!("let x = {}1{}\n", "(".repeat(5000), ")".repeat(5000));
    for (name, src, needle) in [
        ("blocks", blocks, "blocks are nested too deeply"),
        ("brackets", brackets, "brackets are nested too deeply"),
    ] {
        let file = scratch(&format!("limit-{name}"), &src);
        let out = Command::new(TELVAX)
            .args(["check", file.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(1),
            "{name}: it must fail, not crash"
        );
        let shown = String::from_utf8_lossy(&out.stderr).into_owned()
            + &String::from_utf8_lossy(&out.stdout);
        assert!(shown.contains(needle), "{name}: {shown}");
        let _ = std::fs::remove_dir_all(file.parent().unwrap());
    }
}
