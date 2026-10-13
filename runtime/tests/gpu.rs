#![cfg(feature = "gpu")]

use std::sync::Arc;

use novar_runtime::{Device, Interp, TensorVal, Value};

fn load(src: &str, device: Device) -> Interp {
    let (ast, perrs) = novar_parser::parse(src);
    assert!(perrs.is_empty(), "parse: {perrs:?}");
    let (hir, diags) = novar_hir::lower(&ast.unwrap());
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
    let src = "
        fn mm(a: Tensor<f32, [5, 7]>, b: Tensor<f32, [7, 3]>) -> Tensor<f32, [5, 3]> { std::tensor::matmul(a, b) }
        fn add(a: Tensor<f32, [5, 7]>, b: Tensor<f32, [5, 7]>) -> Tensor<f32, [5, 7]> { std::tensor::add(a, b) }
        fn mul(a: Tensor<f32, [5, 7]>, b: Tensor<f32, [5, 7]>) -> Tensor<f32, [5, 7]> { std::tensor::mul(a, b) }
        fn rl(a: Tensor<f32, [5, 7]>) -> Tensor<f32, [5, 7]> { std::tensor::relu(a) }
        fn sg(a: Tensor<f32, [5, 7]>) -> Tensor<f32, [5, 7]> { std::tensor::sigmoid(a) }
        fn sm(a: Tensor<f32, [5, 7]>) -> Tensor<f32, [5, 7]> { std::tensor::softmax(a) }
        fn sc(a: Tensor<f32, [5, 7]>) -> Tensor<f32, [5, 7]> { std::tensor::scale(a, 2.5) }
        fn total(a: Tensor<f32, [5, 7]>) -> f32 { std::tensor::sum(a) }
        fn avg(a: Tensor<f32, [5, 7]>) -> f32 { std::tensor::mean(a) }
        fn dt(a: Tensor<f32, [35]>, b: Tensor<f32, [35]>) -> f32 { std::tensor::dot(a, b) }
    ";
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
    let src = "fn mm(a: Tensor<f32, [130, 70]>, b: Tensor<f32, [70, 65]>) -> Tensor<f32, [130, 65]> { std::tensor::matmul(a, b) }";
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
