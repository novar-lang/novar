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
fn test_on_a_directory_runs_every_test_file() {
    let dir = std::env::temp_dir().join(format!("telvax-tools-{}-testdir", std::process::id()));
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let good = "fn test_one():\n    assert_eq(1, 1)\n";
    std::fs::write(dir.join("test_a.tvx"), good).unwrap();
    std::fs::write(dir.join("sub").join("b_test.tvx"), good).unwrap();
    std::fs::write(
        dir.join("helper.tvx"),
        "fn test_ignored():\n    assert_eq(1, 2)\n",
    )
    .unwrap();
    let path = dir.to_str().unwrap();
    let (ok, out, err) = run(&["test", path]);
    assert!(ok, "{out}{err}");
    assert!(
        out.contains("test_a.tvx") && out.contains("b_test.tvx"),
        "{out}"
    );
    assert!(!out.contains("helper.tvx"), "{out}");
    assert!(out.contains("2 passed, 0 failed"), "{out}");
    std::fs::write(
        dir.join("test_c.tvx"),
        "fn test_bad():\n    assert_eq(1, 2)\n",
    )
    .unwrap();
    let (ok, out, _) = run(&["test", path]);
    assert!(!ok && out.contains("2 passed, 1 failed"), "{out}");
    let (ok, out, _) = run(&["test", path, "one"]);
    assert!(ok && out.contains("2 passed, 0 failed"), "{out}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn macros_quote_their_argument_and_dbg_gives_the_value_back() {
    let src = "fn pick(n: Int) -> Int:\n    if n > 0:\n        n\n    else:\n        todo!()\n\nfn test_dbg():\n    let x = dbg!(pick(4))\n    assert!(x == 4)\n\nfn test_quote():\n    let x = 1\n    assert!(x > 3)\n\nfn test_message():\n    assert!(1 > 3, \"one is small\")\n\nfn test_todo():\n    pick(0)\n\nfn test_panic():\n    panic!(\"boom\")\n";
    let file = scratch("macros", src);
    let path = file.to_str().unwrap();
    let (ok, out, err) = run(&["test", path]);
    assert!(!ok);
    assert!(out.contains("ok      test_dbg"), "{out}");
    assert!(err.contains("pick(4) = 4"), "{err}");
    assert!(out.contains("test_quote: assertion failed: x > 3"), "{out}");
    assert!(out.contains("test_message: one is small"), "{out}");
    assert!(out.contains("test_todo: not yet implemented"), "{out}");
    assert!(out.contains("test_panic: boom"), "{out}");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn an_unknown_macro_names_the_ones_that_exist() {
    let file = scratch("badmacro", "fn main() -> i32:\n    foo!(1)\n    0\n");
    let (ok, out, err) = run(&["check", file.to_str().unwrap()]);
    assert!(!ok);
    assert!(
        format!("{out}{err}").contains("there is no macro `foo!`"),
        "{out}{err}"
    );
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn allow_limits_what_a_program_may_touch() {
    let src = "import std.fs\n\nfn has(path: Text) -> Bool:\n    std.fs.exists(path)\n\nfn test_has():\n    assert(has(\"/\"))\n\nfn add(a: Int, b: Int) -> Int:\n    a + b\n";
    let file = scratch("caps", src);
    let path = file.to_str().unwrap();
    // Everything is allowed unless `--allow` says otherwise.
    let (ok, out, _) = run(&["run", path, "has", "/"]);
    assert!(ok && out.contains("True"), "{out}");
    let (ok, _, err) = run(&["run", path, "has", "/", "--allow", "none"]);
    assert!(!ok);
    assert!(
        err.contains("`std.fs.exists` needs the `fs.read` capability")
            && err.contains("--allow fs.read"),
        "{err}"
    );
    let (ok, out, _) = run(&["run", path, "has", "/", "--allow", "fs.read"]);
    assert!(ok && out.contains("True"), "{out}");
    // A function that does no I/O is not affected.
    let (ok, out, _) = run(&["run", path, "add", "1", "2", "--allow", "none"]);
    assert!(ok && out.contains('3'), "{out}");
    let (ok, out, _) = run(&["test", path, "--allow", "env"]);
    assert!(
        !ok && out.contains("FAILED  test_has") && out.contains("--allow fs.read"),
        "{out}"
    );
    let (ok, _, err) = run(&["run", path, "add", "1", "2", "--allow", "disk"]);
    assert!(!ok && err.contains("unknown capability `disk`"), "{err}");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn doctor_reports_the_build_and_an_unknown_command_is_named() {
    let (ok, out, _) = run(&["doctor"]);
    assert!(ok, "{out}");
    assert!(out.contains("syntax editions read: 2026"), "{out}");
    assert!(out.contains("native backend (build, jit)"), "{out}");
    // A key is reported as set or not, never shown.
    let key = Command::new(TELVAX)
        .arg("doctor")
        .env("OPENAI_API_KEY", "sk-secret-value")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&key.stdout);
    assert!(
        text.contains("OPENAI_API_KEY        set") && !text.contains("sk-secret-value"),
        "{text}"
    );
    let out = Command::new(TELVAX).arg("chek").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        err.contains("unknown command `chek`; did you mean `check`?"),
        "{err}"
    );
    let out = Command::new(TELVAX).arg("frobnicate").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("unknown command `frobnicate`") && !err.contains("did you mean"),
        "{err}"
    );
}

#[test]
fn run_defaults_to_main_and_reads_arguments_by_their_declared_type() {
    let src = "fn main() -> i32:\n    7\n\nfn add(a: Int, b: Int) -> Int:\n    a + b\n\nfn shout(s: Text) -> Text:\n    s\n\nfn total(xs: List[Int]) -> Int:\n    var t = 0\n    for x in xs:\n        t += x\n    t\n";
    let file = scratch("runargs", src);
    let path = file.to_str().unwrap();
    let (ok, out, err) = run(&["run", path]);
    assert!(ok && out.trim() == "7", "{out}{err}");
    // The check that comes before is silent: only `telvax check` says "ok".
    assert!(!err.contains(" ok"), "{err}");
    let (ok, out, _) = run(&["run", path, "--allow", "none"]);
    assert!(ok && out.trim() == "7", "{out}");
    // A `Text` parameter keeps the word as it is.
    let (ok, out, _) = run(&["run", path, "shout", "123"]);
    assert!(ok && out.trim() == "123", "{out}");
    let (ok, out, _) = run(&["run", path, "total", "[1,2,3]"]);
    assert!(ok && out.trim() == "6", "{out}");
    let (ok, _, err) = run(&["run", path, "add", "1", "x"]);
    assert!(
        !ok && err.contains("argument `b` of `add` is Int, and `x` is not a value"),
        "{err}"
    );
    let (ok, _, err) = run(&["run", path, "add", "1"]);
    assert!(
        !ok && err.contains("takes 2 argument(s) (a: Int, b: Int), got 1"),
        "{err}"
    );
    let (ok, _, err) = run(&["run", path, "nope"]);
    assert!(
        !ok && err.contains("has no function `nope`; it has: add, main, shout, total"),
        "{err}"
    );
    let (ok, _, err) = run(&["check", path]);
    assert!(ok && err.contains(" ok"), "{err}");
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn check_takes_many_files_a_package_and_json() {
    let good = scratch("check_good", "fn main() -> i32:\n    0\n");
    let bad = scratch(
        "check_bad",
        "fn main() -> i32:\n    let x: Int = \"a\"\n    0\n",
    );
    let (good_path, bad_path) = (good.to_str().unwrap(), bad.to_str().unwrap());
    let (ok, _, err) = run(&["check", good_path, bad_path]);
    assert!(!ok);
    assert!(
        err.contains("2 file(s) checked, 1 with errors") && err.contains("check_good.tvx` ok"),
        "{err}"
    );
    // JSON: one array on stdout, the places as line and column, and the exit code still says failed.
    let (ok, out, _) = run(&["check", "--json", bad_path]);
    assert!(!ok);
    let records: serde_json::Value = serde_json::from_str(&out).unwrap();
    let first = &records[0];
    assert_eq!(first["severity"], "error");
    assert_eq!(
        (first["line"].as_u64(), first["column"].as_u64()),
        (Some(2), Some(18))
    );
    assert_eq!(first["end_column"].as_u64(), Some(21));
    assert!(
        first["message"]
            .as_str()
            .unwrap()
            .contains("expected `Int`, found `Text`")
    );
    assert!(first["file"].as_str().unwrap().ends_with("check_bad.tvx"));
    let (ok, out, _) = run(&["check", "--json", good_path]);
    assert!(ok && out.trim() == "[]", "{out}");
    let (ok, out, _) = run(&["check", "--json", "no-such-file.tvx"]);
    assert!(
        !ok && out.contains("cannot read `no-such-file.tvx`") && out.contains("\"line\":null"),
        "{out}"
    );
    let (ok, _, err) = run(&["check", "--bogus"]);
    assert!(!ok && err.contains("no option `--bogus`"), "{err}");
    // Without a file, the package around the current directory.
    let pkg = std::env::temp_dir().join(format!("telvax-tools-{}-checkpkg", std::process::id()));
    std::fs::create_dir_all(pkg.join("src")).unwrap();
    std::fs::write(
        pkg.join("telvax.toml"),
        "[package]\nname = \"app\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(pkg.join("src/main.tvx"), "fn main() -> i32:\n    0\n").unwrap();
    let out = Command::new(TELVAX)
        .arg("check")
        .current_dir(&pkg)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.contains("main.tvx` ok"),
        "{err}"
    );
    let outside = Command::new(TELVAX)
        .arg("check")
        .current_dir(good.parent().unwrap())
        .output()
        .unwrap();
    assert!(!outside.status.success());
    assert!(String::from_utf8_lossy(&outside.stderr).contains("inside a package"));
    let _ = std::fs::remove_dir_all(&pkg);
    let _ = std::fs::remove_dir_all(good.parent().unwrap());
    let _ = std::fs::remove_dir_all(bad.parent().unwrap());
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
