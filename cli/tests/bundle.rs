use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const NOVAR: &str = env!("CARGO_BIN_EXE_novar");

const APP: &str = r#"
    struct In { name: String }
    struct Out { greeting: String }
    @endpoint(path = "/hello", method = "POST")
    pub fn hello(p: In) -> Result<Out, String> { Ok(Out { greeting: f"hello, {p.name}!" }) }
"#;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("novar-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn post(port: u16, path: &str, body: &str) -> Option<(u16, String)> {
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
    let src = scratch("hello.nv");
    std::fs::write(&src, APP).unwrap();
    let exe = scratch("hello-service");
    let out = Command::new(NOVAR)
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
    let src = scratch("env.nv");
    std::fs::write(&src, APP).unwrap();
    let exe = scratch("env-service");
    assert!(
        Command::new(NOVAR)
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
    let src = scratch("noep.nv");
    std::fs::write(&src, "fn main() -> i32 { 0 }").unwrap();
    let out = Command::new(NOVAR)
        .args(["build", "--serve"])
        .arg(&src)
        .arg("-o")
        .arg(scratch("noep"))
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("nothing to serve"));
}

#[test]
fn the_dockerfile_is_multi_stage_and_names_the_program() {
    let dir = scratch("docker-app");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("app.nv");
    std::fs::write(&src, APP).unwrap();
    let out = Command::new(NOVAR)
        .args(["build", "--serve"])
        .arg(&src)
        .arg("-o")
        .arg(dir.join("app"))
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
    assert!(docker.contains("COPY app.nv ."));
    assert!(docker.contains("build --serve app.nv -o /out/service"));
    assert!(docker.contains("COPY --from=builder /out/service /service"));
    assert!(docker.contains("USER nonroot") && docker.contains("EXPOSE 8080"));
}
