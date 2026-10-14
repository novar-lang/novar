use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const TELVAX: &str = env!("CARGO_BIN_EXE_telvax");

const APP: &str = r#"
    struct In { name: String }
    struct Out { greeting: String }
    @endpoint(path = "/hello", method = "POST")
    pub fn hello(p: In) -> Result<Out, String> { Ok(Out { greeting: f"hello, {p.name}!" }) }
"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("telvax-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A scratch path for an executable written by `telvax build --serve` (`.exe` on Windows). Each
/// one is a full copy of the `telvax` binary (over a gigabyte in a debug build), so it is deleted
/// when the test ends, passing or not.
struct TempExe(PathBuf);

impl Drop for TempExe {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl AsRef<std::ffi::OsStr> for TempExe {
    fn as_ref(&self) -> &std::ffi::OsStr {
        self.0.as_os_str()
    }
}

impl AsRef<Path> for TempExe {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

fn scratch_exe(name: &str) -> TempExe {
    TempExe(scratch(&format!("{name}{}", std::env::consts::EXE_SUFFIX)))
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// The listener can accept a connection a moment before the server answers requests, so a failed
/// exchange is retried for a few seconds. The endpoints under test have no side effects.
fn post(port: u16, path: &str, body: &str) -> Option<(u16, String)> {
    for _ in 0..50 {
        if let Some(reply) = post_once(port, path, body) {
            return Some(reply);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

fn post_once(port: u16, path: &str, body: &str) -> Option<(u16, String)> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        s,
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut raw = String::new();
    s.read_to_string(&mut raw).ok()?;
    let status = raw.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, raw.split("\r\n\r\n").nth(1)?.to_string()))
}

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_for(port: u16) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(30) {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("server did not start");
}

#[test]
fn a_bundled_binary_serves_its_program() {
    let src = scratch("hello.tvx");
    std::fs::write(&src, APP).unwrap();
    let exe = scratch_exe("hello-service");
    let out = Command::new(TELVAX)
        .args(["build", "--serve"])
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let port = free_port();
    let child = Command::new(&exe)
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _server = Server(child);
    wait_for(port);
    let (status, body) = post(port, "/hello", r#"{"name":"world"}"#).unwrap();
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap(),
        serde_json::json!({"greeting": "hello, world!"})
    );
    assert_eq!(post(port, "/hello", r#"{"nope":1}"#).unwrap().0, 400);
}

#[test]
fn the_port_can_come_from_the_environment() {
    let src = scratch("env.tvx");
    std::fs::write(&src, APP).unwrap();
    let exe = scratch_exe("env-service");
    assert!(
        Command::new(TELVAX)
            .args(["build", "--serve"])
            .arg(&src)
            .arg("-o")
            .arg(&exe)
            .status()
            .unwrap()
            .success()
    );
    let port = free_port();
    let child = Command::new(&exe)
        .env("PORT", port.to_string())
        .env("HOST", "127.0.0.1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _server = Server(child);
    wait_for(port);
    assert_eq!(post(port, "/hello", r#"{"name":"env"}"#).unwrap().0, 200);
}

#[test]
fn programs_without_endpoints_are_rejected() {
    let src = scratch("noep.tvx");
    std::fs::write(&src, "fn main() -> i32 { 0 }").unwrap();
    let out = Command::new(TELVAX)
        .args(["build", "--serve"])
        .arg(&src)
        .arg("-o")
        .arg(scratch_exe("noep"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("nothing to serve"));
}

#[test]
fn the_dockerfile_is_multi_stage_and_names_the_program() {
    let dir = scratch("docker-app");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("app.tvx");
    std::fs::write(&src, APP).unwrap();
    let exe = scratch_exe("docker-app/app");
    let out = Command::new(TELVAX)
        .args(["build", "--serve"])
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .arg("--dockerfile")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let docker = std::fs::read_to_string(dir.join("Dockerfile")).unwrap();
    assert_eq!(docker.matches("\nFROM ").count(), 2, "{docker}");
    assert!(docker.contains("COPY app.tvx ."));
    assert!(docker.contains("mkdir -p /out && telvax build --serve app.tvx -o /out/service"));
    assert!(docker.contains("--mount=type=cache,target=/telvax/target"));
    assert!(docker.contains("COPY --from=builder /out/service /service"));
    assert!(docker.contains("USER nonroot") && docker.contains("EXPOSE 8080"));
}

#[test]
fn a_multi_file_program_is_bundled_with_all_its_modules() {
    let dir = scratch("multi-app");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("main.tvx"),
        "import util::shout;\nstruct In { name: String }\nstruct Out { greeting: String }\n\
         @endpoint(path = \"/hello\", method = \"POST\")\npub fn hello(p: In) -> Result<Out, String> { Ok(Out { greeting: shout(p.name) }) }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("util.tvx"),
        "pub fn shout(s: String) -> String { f\"{s}!\" }\n",
    )
    .unwrap();
    let exe = scratch_exe("multi-service");
    let out = Command::new(TELVAX)
        .args(["build", "--serve"])
        .arg(dir.join("main.tvx"))
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The service must not read the sources again: delete them.
    std::fs::remove_dir_all(&dir).unwrap();

    let port = free_port();
    let child = Command::new(&exe)
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _server = Server(child);
    wait_for(port);
    let (status, body) = post(port, "/hello", r#"{"name":"modules"}"#).unwrap();
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).unwrap(),
        serde_json::json!({"greeting": "modules!"})
    );
}

#[test]
fn errors_in_a_module_are_reported_against_that_file() {
    let dir = scratch("err-app");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("main.tvx"),
        "import util::bad;\nfn main() -> i32 { bad() }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("util.tvx"),
        "pub fn bad() -> i32 {\n    \"nope\"\n}\n",
    )
    .unwrap();
    let out = Command::new(TELVAX)
        .arg("check")
        .arg(dir.join("main.tvx"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("util.tvx:2:5") && err.contains("expected `i32`, found `String`"),
        "{err}"
    );
}

#[test]
fn tvpm_projects_run_against_installed_dependencies() {
    use telvax_pkg::{Project, registry};

    let root = scratch("tvpm-e2e");
    let _ = std::fs::remove_dir_all(&root);
    let (reg, lib, app, home) = (
        root.join("registry"),
        root.join("mathx"),
        root.join("app"),
        root.join("home"),
    );
    std::fs::create_dir_all(lib.join("src")).unwrap();
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::write(
        lib.join("telvax.toml"),
        "[package]\nname = \"mathx\"\nversion = \"1.3.0\"\n",
    )
    .unwrap();
    std::fs::write(
        lib.join("src/lib.tvx"),
        "pub fn cube(x: i64) -> i64 { x * x * x }\n",
    )
    .unwrap();
    registry::publish(&reg, &lib).unwrap();

    std::fs::write(
        app.join("telvax.toml"),
        "[package]\nname = \"app\"\nversion = \"0.1.0\"\n[dependencies]\nmathx = \"^1.2\"\n",
    )
    .unwrap();
    std::fs::write(
        app.join("src/main.tvx"),
        "import mathx::cube;\nfn main() -> i64 { cube(3) + 1 }\n",
    )
    .unwrap();

    // Without a lockfile the compiler asks for `tvpm install`.
    let out = Command::new(TELVAX)
        .env("TVPM_HOME", &home)
        .arg("check")
        .arg(app.join("src/main.tvx"))
        .output()
        .unwrap();
    assert!(!out.status.success() && String::from_utf8_lossy(&out.stderr).contains("tvpm install"));

    let project = Project::open(&app).unwrap();
    let r = registry::open(reg.to_str().unwrap());
    unsafe { std::env::set_var("TVPM_HOME", &home) };
    project.lock(Some(r.as_ref())).unwrap();
    project.install(Some(r.as_ref())).unwrap();

    let out = Command::new(TELVAX)
        .env("TVPM_HOME", &home)
        .args(["run"])
        .arg(app.join("src/main.tvx"))
        .arg("main")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "28");
}
