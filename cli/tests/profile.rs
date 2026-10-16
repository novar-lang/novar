use std::process::Command;

const TELVAX: &str = env!("CARGO_BIN_EXE_telvax");

const SRC: &str = "\
fn busy(n: Int) -> Int:
    var s = 0
    for i in range(0, n):
        s = s + i * i
    s

fn fib(n: Int) -> Int:
    if n < 2:
        return n
    fib(n - 1) + fib(n - 2)

fn main() -> Int:
    let w: Tensor[f32, 32, 32] = std.tensor.zeros()
    let y = std.tensor.matmul(w, w)
    busy(500) + fib(8)
";

fn scratch(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("telvax-profile-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("app.tvx");
    std::fs::write(&file, SRC).unwrap();
    (dir, file)
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
fn profile_reports_functions_statements_operations_and_memory() {
    let (dir, file) = scratch("report");
    let (ok, out, err) = run(&["profile", file.to_str().unwrap(), "--top", "3"]);
    assert!(ok, "{err}");
    for needle in [
        "profile of main",
        "wall time",
        "memory    peak",
        "functions, by time",
        "statements, by time",
        "std operations, by time",
        "std.tensor.matmul",
        "app.tvx:4  s = s + i * i",
    ] {
        assert!(out.contains(needle), "{needle} is missing from\n{out}");
    }
    // fib(8) makes 67 calls.
    assert!(
        out.lines()
            .any(|l| l.contains(" 67 ") && l.trim_end().ends_with("fib")),
        "{out}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn profile_writes_json_flame_graph_and_speedscope_files() {
    let (dir, file) = scratch("files");
    let path = |n: &str| dir.join(n).to_str().unwrap().to_string();
    let (ok, _, err) = run(&[
        "profile",
        file.to_str().unwrap(),
        "main",
        "--json",
        &path("r.json"),
        "--collapsed",
        &path("c.txt"),
        "--speedscope",
        &path("s.json"),
    ]);
    assert!(ok, "{err}");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("r.json")).unwrap()).unwrap();
    assert!(json["functions"].as_array().unwrap().len() >= 3);
    assert!(json["memory"]["allocated_bytes"].as_u64().unwrap() > 0);
    let collapsed = std::fs::read_to_string(dir.join("c.txt")).unwrap();
    assert!(
        collapsed.lines().any(|l| l.starts_with("main;busy ")),
        "{collapsed}"
    );
    let scope: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("s.json")).unwrap()).unwrap();
    assert_eq!(scope["profiles"][0]["unit"], "microseconds");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn profile_says_what_is_wrong_with_the_command() {
    let (dir, file) = scratch("errors");
    let (ok, _, err) = run(&["profile", file.to_str().unwrap(), "nope"]);
    assert!(!ok && err.contains("has no function `nope`"), "{err}");
    let (ok, _, err) = run(&["profile", file.to_str().unwrap(), "--top", "many"]);
    assert!(!ok && err.contains("--top takes a number"), "{err}");
    let (ok, _, err) = run(&["profile"]);
    assert!(!ok && err.contains("usage:"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}
