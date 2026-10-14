use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const TELVAX: &str = env!("CARGO_BIN_EXE_telvax");

struct Proc(Child);

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn worker(port: u16) -> Proc {
    let child = Command::new(TELVAX)
        .args(["worker", "--listen", &format!("127.0.0.1:{port}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "worker did not start"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    Proc(child)
}

fn calls(port: u16) -> u64 {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let body = raw.split("\r\n\r\n").nth(1).unwrap();
    serde_json::from_str::<serde_json::Value>(body).unwrap()["calls"]
        .as_u64()
        .unwrap()
}

fn program(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("telvax-cluster-cli-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("app.tvx");
    std::fs::write(
        &path,
        r#"
def square(x: int) -> int:
    return x * x

def ones() -> list[int]:
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def all() -> int:
    s: int = 0
    for v in std.dist.map(ones(), square):
        s += v
    return s
"#,
    )
    .unwrap();
    path
}

#[test]
fn run_sends_dist_map_to_workers_started_with_telvax_worker() {
    let (pa, pb) = (free_port(), free_port());
    let (_a, _b) = (worker(pa), worker(pb));
    let file = program("run");

    let local = Command::new(TELVAX)
        .args(["run"])
        .arg(&file)
        .arg("all")
        .output()
        .unwrap();
    assert!(
        local.status.success(),
        "{}",
        String::from_utf8_lossy(&local.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&local.stdout).trim(), "385");

    let out = Command::new(TELVAX)
        .args(["run"])
        .arg(&file)
        .arg("all")
        .args(["--workers", &format!("127.0.0.1:{pa},127.0.0.1:{pb}")])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "385",
        "same answer on the cluster"
    );
    assert_eq!(
        calls(pa) + calls(pb),
        10,
        "the ten calls ran on the workers"
    );
}

#[test]
fn workers_can_require_a_token() {
    let port = free_port();
    let child = Command::new(TELVAX)
        .args([
            "worker",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--token",
            "hunter2",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _w = Proc(child);
    let start = Instant::now();
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(start.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_millis(50));
    }
    let file = program("token");
    let workers = format!("127.0.0.1:{port}");
    let denied = Command::new(TELVAX)
        .args(["run"])
        .arg(&file)
        .arg("all")
        .args(["--workers", &workers])
        .output()
        .unwrap();
    assert!(
        !denied.status.success()
            && String::from_utf8_lossy(&denied.stderr).contains("rejected the token")
    );
    let ok = Command::new(TELVAX)
        .args(["run"])
        .arg(&file)
        .arg("all")
        .args(["--workers", &workers])
        .env("TELVAX_CLUSTER_TOKEN", "hunter2")
        .output()
        .unwrap();
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&ok.stdout).trim(), "385");
}
