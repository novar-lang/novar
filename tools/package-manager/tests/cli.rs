use std::path::PathBuf;
use std::process::{Command, Output};

const TVPM: &str = env!("CARGO_BIN_EXE_tvpm");

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tvpm-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tvpm(cwd: &std::path::Path, home: &std::path::Path, args: &[&str]) -> Output {
    Command::new(TVPM)
        .current_dir(cwd)
        .env("TVPM_HOME", home)
        .env_remove("TVPM_REGISTRY")
        .args(args)
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

#[test]
fn the_command_line_covers_the_whole_workflow() {
    let root = scratch("workflow");
    let home = root.join("home");
    let reg = root.join("registry");
    let reg_s = reg.to_str().unwrap();

    // A library, published twice.
    let o = tvpm(&root, &home, &["new", "mathx", "--lib"]);
    assert!(o.status.success(), "{}", text(&o));
    let lib = root.join("mathx");
    assert!(lib.join("telvax.toml").is_file() && lib.join("src/lib.tvx").is_file());
    std::fs::write(
        lib.join("src/lib.tvx"),
        r#"
fn double(x: i32) -> i32:
    x * 2
"#,
    )
    .unwrap();
    let o = tvpm(&lib, &home, &["publish", "--registry", reg_s]);
    assert!(
        o.status.success() && text(&o).contains("published mathx 0.1.0"),
        "{}",
        text(&o)
    );
    assert!(
        !tvpm(&lib, &home, &["publish", "--registry", reg_s])
            .status
            .success(),
        "republishing the same version fails"
    );

    // An application that uses it.
    assert!(tvpm(&root, &home, &["new", "app"]).status.success());
    let app = root.join("app");
    assert!(
        tvpm(&root, &home, &["new", "app"]).status.code() == Some(1),
        "an existing project is not overwritten"
    );
    let o = tvpm(&app, &home, &["add", "mathx", "--registry", reg_s]);
    assert!(o.status.success(), "{}", text(&o));
    let manifest = std::fs::read_to_string(app.join("telvax.toml")).unwrap();
    assert!(manifest.contains("mathx = \"^0.1.0\""), "{manifest}");

    let o = tvpm(&app, &home, &["install", "--registry", reg_s]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(
        text(&o).contains("mathx 0.1.0") && text(&o).contains("downloaded"),
        "{}",
        text(&o)
    );
    assert!(app.join("telvax.lock").is_file());
    let tree = text(&tvpm(&app, &home, &["tree"]));
    assert!(
        tree.contains("app 0.1.0") && tree.contains("└── mathx 0.1.0"),
        "{tree}"
    );

    let found = text(&tvpm(&app, &home, &["search", "math", "--registry", reg_s]));
    assert!(found.contains("mathx 0.1.0"), "{found}");

    let o = tvpm(&app, &home, &["remove", "mathx"]);
    assert!(o.status.success());
    assert!(
        !std::fs::read_to_string(app.join("telvax.toml"))
            .unwrap()
            .contains("mathx")
    );
    assert!(!tvpm(&app, &home, &["remove", "mathx"]).status.success());
}

#[test]
fn path_dependencies_and_bad_input() {
    let root = scratch("paths");
    let home = root.join("home");
    assert!(
        tvpm(&root, &home, &["new", "shared", "--lib"])
            .status
            .success()
    );
    assert!(tvpm(&root, &home, &["new", "app"]).status.success());
    let app = root.join("app");
    let o = tvpm(&app, &home, &["add", "shared", "--path", "../shared"]);
    assert!(o.status.success(), "{}", text(&o));
    let o = tvpm(&app, &home, &["lock"]);
    assert!(
        o.status.success() && text(&o).contains("shared 0.1.0"),
        "{}",
        text(&o)
    );

    let o = tvpm(&app, &home, &["add", "wrong", "--path", "../shared"]);
    assert!(
        !o.status.success() && text(&o).contains("not `wrong`"),
        "{}",
        text(&o)
    );
    let o = tvpm(&root, &home, &["new", "Bad-Name"]);
    assert!(!o.status.success() && text(&o).contains("invalid package name"));
    let o = tvpm(&root, &home, &["lock"]);
    assert!(
        !o.status.success() && text(&o).contains("no telvax.toml"),
        "{}",
        text(&o)
    );
    assert!(text(&tvpm(&root, &home, &["--help"])).contains("the Telvax package manager"));
    assert_eq!(tvpm(&root, &home, &["frobnicate"]).status.code(), Some(2));
}

#[test]
fn yank_hides_a_version_from_new_installs() {
    let root = scratch("yank");
    let home = root.join("home");
    let reg = root.join("registry");
    let reg_s = reg.to_str().unwrap();
    assert!(
        tvpm(&root, &home, &["new", "lib1", "--lib"])
            .status
            .success()
    );
    let lib = root.join("lib1");
    assert!(
        tvpm(&lib, &home, &["publish", "--registry", reg_s])
            .status
            .success()
    );
    let o = tvpm(&lib, &home, &["yank", "lib1", "0.1.0", "--registry", reg_s]);
    assert!(
        o.status.success() && text(&o).contains("yanked lib1 0.1.0"),
        "{}",
        text(&o)
    );
    assert!(tvpm(&root, &home, &["new", "app2"]).status.success());
    let app = root.join("app2");
    let o = tvpm(&app, &home, &["add", "lib1@0.1", "--registry", reg_s]);
    assert!(o.status.success());
    let o = tvpm(&app, &home, &["lock", "--registry", reg_s]);
    assert!(
        !o.status.success() && text(&o).contains("no version of `lib1`"),
        "{}",
        text(&o)
    );
    assert!(
        tvpm(
            &lib,
            &home,
            &["yank", "lib1", "0.1.0", "--undo", "--registry", reg_s]
        )
        .status
        .success()
    );
    assert!(
        tvpm(&app, &home, &["lock", "--registry", reg_s])
            .status
            .success()
    );
}
