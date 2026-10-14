//! Chained tensor expressions keep their intermediate results on the GPU. This file has a single
//! test on purpose: it counts transfers on the process-wide GPU, so nothing else may run beside it.
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

fn tensor(shape: &[usize], seed: usize) -> Value {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * seed % 23) as f32 - 11.0) / 9.0)
        .collect();
    Value::Tensor(Arc::new(TensorVal::from_f32(shape.to_vec(), data)))
}

fn data(v: Value) -> Vec<f32> {
    match v {
        Value::Tensor(t) => t.to_f32_vec(),
        other => panic!("expected a tensor, got {other:?}"),
    }
}

const SRC: &str = "
    fn chained(x: Tensor<f32, [8, 16]>, w1: Tensor<f32, [16, 32]>, w2: Tensor<f32, [32, 4]>) -> Tensor<f32, [8, 4]> {
        std::tensor::softmax(std::tensor::matmul(std::tensor::relu(std::tensor::matmul(x, w1)), w2))
    }
    fn split(x: Tensor<f32, [8, 16]>, w1: Tensor<f32, [16, 32]>, w2: Tensor<f32, [32, 4]>) -> Tensor<f32, [8, 4]> {
        let h = std::tensor::relu(std::tensor::matmul(x, w1));
        std::tensor::softmax(std::tensor::matmul(h, w2))
    }
    fn scaled(a: Tensor<f32, [4, 4]>, b: Tensor<f32, [4, 4]>) -> Tensor<f32, [4, 4]> {
        std::tensor::sigmoid(std::tensor::scale(std::tensor::sub(std::tensor::add(a, b), b), 0.5))
    }
";

#[test]
fn chained_expressions_stay_on_the_gpu() {
    let gpu = telvax_runtime::gpu::global().expect("a GPU is needed for this test");
    let args = || {
        vec![
            tensor(&[8, 16], 13),
            tensor(&[16, 32], 7),
            tensor(&[32, 4], 11),
        ]
    };

    let cpu = load(SRC, Device::Cpu);
    let on_gpu = load(SRC, Device::Gpu);
    let want = data(cpu.call("chained", args()).unwrap());

    // Three leaves go up, one result comes back; the two intermediates never leave the GPU.
    let before = gpu.transfers();
    let got = data(on_gpu.call("chained", args()).unwrap());
    let after = gpu.transfers();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (3, 1),
        "uploads and downloads"
    );
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    // The same network written with a `let` in the middle copies the intermediate through the host.
    let before = gpu.transfers();
    let got = data(on_gpu.call("split", args()).unwrap());
    let after = gpu.transfers();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (4, 2),
        "split into two statements"
    );
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    // Elementwise chains, with a scalar factor.
    let (a, b) = (tensor(&[4, 4], 5), tensor(&[4, 4], 3));
    let want = data(cpu.call("scaled", vec![a.clone(), b.clone()]).unwrap());
    let before = gpu.transfers();
    let got = data(on_gpu.call("scaled", vec![a, b]).unwrap());
    let after = gpu.transfers();
    // `b` appears twice in the expression, so it is uploaded twice.
    assert_eq!((after.0 - before.0, after.1 - before.1), (3, 1));
    for (x, y) in got.iter().zip(&want) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

/// Not a pass/fail check: prints how much the chain saves on a 1024-wide network.
/// Run with `cargo test -p telvax_runtime --features gpu --test gpu_chain -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_of_chained_and_split_networks() {
    let src = "
        fn chained(x: Tensor<f32, [512, 1024]>, w1: Tensor<f32, [1024, 1024]>, w2: Tensor<f32, [1024, 1024]>, w3: Tensor<f32, [1024, 64]>) -> Tensor<f32, [512, 64]> {
            std::tensor::softmax(std::tensor::matmul(std::tensor::relu(std::tensor::matmul(std::tensor::relu(std::tensor::matmul(x, w1)), w2)), w3))
        }
        fn split(x: Tensor<f32, [512, 1024]>, w1: Tensor<f32, [1024, 1024]>, w2: Tensor<f32, [1024, 1024]>, w3: Tensor<f32, [1024, 64]>) -> Tensor<f32, [512, 64]> {
            let h1 = std::tensor::relu(std::tensor::matmul(x, w1));
            let h2 = std::tensor::relu(std::tensor::matmul(h1, w2));
            std::tensor::softmax(std::tensor::matmul(h2, w3))
        }";
    let args = || {
        vec![
            tensor(&[512, 1024], 13),
            tensor(&[1024, 1024], 7),
            tensor(&[1024, 1024], 11),
            tensor(&[1024, 64], 5),
        ]
    };
    let gpu = load(src, Device::Gpu);
    let cpu = load(src, Device::Cpu);
    let time = |interp: &Interp, f: &str| {
        let _ = interp.call(f, args()).unwrap(); // warm up (shaders compile on first use)
        let start = std::time::Instant::now();
        for _ in 0..5 {
            interp.call(f, args()).unwrap();
        }
        start.elapsed().as_secs_f64() * 1000.0 / 5.0
    };
    println!("cpu   chained: {:8.1} ms", time(&cpu, "chained"));
    println!("gpu   split:   {:8.1} ms", time(&gpu, "split"));
    println!("gpu   chained: {:8.1} ms", time(&gpu, "chained"));
}
