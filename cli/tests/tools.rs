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
    let messy = "def main() -> int32:\n  x=1\n  return x+2\n";
    let tidy = "def main() -> int32:\n    x = 1\n    return x + 2\n";
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
    let file = scratch("broken", "def main():\n    s = \"open\n");
    let (ok, _, err) = run(&["fmt", file.to_str().unwrap()]);
    assert!(!ok && err.contains("cannot format"), "{err}");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn lint_reports_unused_code() {
    let file = scratch(
        "lintme",
        r#"
def _helper() -> int32:
    return 1

def main() -> int32:
    unused = 2
    return 3
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
