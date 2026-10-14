#![cfg(feature = "gpu")]

use std::sync::Arc;

use telvax_runtime::{Device, Interp, TensorVal, Value};

fn load(src: &str, device: Device) -> Interp {
    let (ast, perrs) = telvax_parser::parse(src);
    assert!(perrs.is_empty(), "parse: {perrs:?}");
    let (hir, diags) = telvax_hir::lower(&ast.unwrap());
    assert!(
        diags.is_empty(),
        "check: {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let interp = Interp::new(hir).expect("globals should initialise");
    interp.set_device(device);
    interp
}

fn f32s(shape: &[usize], data: &[f32]) -> Value {
    Value::Tensor(Arc::new(TensorVal::from_f32(shape.to_vec(), data.to_vec())))
}

fn tensor(v: Value) -> Vec<f32> {
    match v {
        Value::Tensor(t) => t.to_f32_vec(),
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn scalar(v: Value) -> f64 {
    match v {
        Value::Float(x) => x,
        other => panic!("expected float, got {other:?}"),
    }
}

fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * seed % 29) as f32 - 14.0) / 8.0)
        .collect()
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-4 * (1.0 + x.abs()), "{x} vs {y}");
    }
}

#[test]
fn gpu_matches_cpu_on_tensor_ops() {
    let src = r#"
def mm(a: Tensor[float32, [5, 7]], b: Tensor[float32, [7, 3]]) -> Tensor[float32, [5, 3]]:
    return std.tensor.matmul(a, b)

def add(a: Tensor[float32, [5, 7]], b: Tensor[float32, [5, 7]]) -> Tensor[float32, [5, 7]]:
    return std.tensor.add(a, b)

def mul(a: Tensor[float32, [5, 7]], b: Tensor[float32, [5, 7]]) -> Tensor[float32, [5, 7]]:
    return std.tensor.mul(a, b)

def rl(a: Tensor[float32, [5, 7]]) -> Tensor[float32, [5, 7]]:
    return std.tensor.relu(a)

def sg(a: Tensor[float32, [5, 7]]) -> Tensor[float32, [5, 7]]:
    return std.tensor.sigmoid(a)

def sm(a: Tensor[float32, [5, 7]]) -> Tensor[float32, [5, 7]]:
    return std.tensor.softmax(a)

def sc(a: Tensor[float32, [5, 7]]) -> Tensor[float32, [5, 7]]:
    return std.tensor.scale(a, 2.5)

def total(a: Tensor[float32, [5, 7]]) -> float32:
    return std.tensor.sum(a)

def avg(a: Tensor[float32, [5, 7]]) -> float32:
    return std.tensor.mean(a)

def dt(a: Tensor[float32, [35]], b: Tensor[float32, [35]]) -> float32:
    return std.tensor.dot(a, b)
"#;
    let cpu = load(src, Device::Cpu);
    let gpu = load(src, Device::Gpu);
    let a = f32s(&[5, 7], &data(35, 13));
    let b = f32s(&[5, 7], &data(35, 7));
    let w = f32s(&[7, 3], &data(21, 11));
    for (f, args) in [
        ("mm", vec![a.clone(), w]),
        ("add", vec![a.clone(), b.clone()]),
        ("mul", vec![a.clone(), b.clone()]),
        ("rl", vec![a.clone()]),
        ("sg", vec![a.clone()]),
        ("sm", vec![a.clone()]),
        ("sc", vec![a.clone()]),
    ] {
        close(
            &tensor(cpu.call(f, args.clone()).unwrap()),
            &tensor(gpu.call(f, args).unwrap()),
        );
    }
    for f in ["total", "avg"] {
        let x = scalar(cpu.call(f, vec![a.clone()]).unwrap());
        let y = scalar(gpu.call(f, vec![a.clone()]).unwrap());
        assert!((x - y).abs() < 1e-3, "{f}: {x} vs {y}");
    }
    let (v, w) = (f32s(&[35], &data(35, 13)), f32s(&[35], &data(35, 7)));
    let x = scalar(cpu.call("dt", vec![v.clone(), w.clone()]).unwrap());
    let y = scalar(gpu.call("dt", vec![v, w]).unwrap());
    assert!((x - y).abs() < 1e-3, "dot: {x} vs {y}");
}

#[test]
fn gpu_matmul_handles_large_sizes() {
    let src = r#"
def mm(a: Tensor[float32, [130, 70]], b: Tensor[float32, [70, 65]]) -> Tensor[float32, [130, 65]]:
    return std.tensor.matmul(a, b)
"#;
    let a = f32s(&[130, 70], &data(130 * 70, 13));
    let b = f32s(&[70, 65], &data(70 * 65, 7));
    let x = tensor(
        load(src, Device::Cpu)
            .call("mm", vec![a.clone(), b.clone()])
            .unwrap(),
    );
    let y = tensor(load(src, Device::Gpu).call("mm", vec![a, b]).unwrap());
    close(&x, &y);
}
