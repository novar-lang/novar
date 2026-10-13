use std::process::Command;

const NOVAR: &str = env!("CARGO_BIN_EXE_novar");

const SRC: &str =
    "fn layer(x: Tensor<f32, [2, 3]>, w: Tensor<f32, [3, 4]>) -> Tensor<f32, [2, 4]> {
    std::tensor::softmax(std::tensor::relu(std::tensor::matmul(x, w)))
}
";

fn emit(kind: &str) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!("novar-accel-{}-{kind}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("layer.nv");
    std::fs::write(&file, SRC).unwrap();
    let out = Command::new(NOVAR)
        .args(["build", file.to_str().unwrap(), "--emit", kind])
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn emits_stablehlo() {
    let (ok, out, err) = emit("stablehlo");
    assert!(ok, "{err}");
    assert!(out.contains("func.func @layer") && out.contains("stablehlo.dot_general"));
}

#[cfg(feature = "llvm")]
#[test]
fn emits_ptx() {
    let (ok, out, err) = emit("ptx");
    assert!(ok, "{err}");
    assert!(out.contains(".visible .entry matmul_f32(") && out.contains(".target sm_80"));
}

#[cfg(not(feature = "llvm"))]
#[test]
fn ptx_needs_llvm() {
    let (ok, _, err) = emit("ptx");
    assert!(!ok && err.contains("needs LLVM"), "{err}");
}
