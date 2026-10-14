//! Tensor expressions such as `relu(matmul(x, w))` on the GPU without copying the intermediate
//! tensors back to the host: every operand that is itself a supported tensor operation stays in a
//! GPU buffer, and only the outermost result is read back.
//!
//! A chain starts only when the whole expression is known (from its types) to be `f32` tensors
//! and supported operations, so it never has to hand half-evaluated code to the CPU path.

use std::sync::Arc;

use telvax_hir::hir::{self, ExprKind};
use telvax_hir::{Prim, Quantization, Ty};

use super::{Ctrl, Ctx, Interp, RuntimeError};
use crate::gpu::{self, Elementwise, Gpu, GpuBuf};
use crate::interp::Device;
use crate::value::{TensorVal, Value};

struct Node {
    buf: GpuBuf,
    shape: Vec<usize>,
}

fn fail<T>(msg: impl Into<String>) -> Result<T, Ctrl> {
    Err(Ctrl::Fail(RuntimeError::new(msg)))
}

fn op_name(path: &[String]) -> Option<&str> {
    match path {
        [a, b, op]
            if a == "std"
                && b == "tensor"
                && matches!(
                    op.as_str(),
                    "add"
                        | "sub"
                        | "mul"
                        | "div"
                        | "scale"
                        | "relu"
                        | "sigmoid"
                        | "softmax"
                        | "matmul"
                ) =>
        {
            Some(op)
        }
        _ => None,
    }
}

fn chain_call(e: &hir::Expr) -> Option<(&str, &[hir::Expr])> {
    if let ExprKind::Call { callee, args } = &e.kind
        && let ExprKind::Extern(path) = &callee.kind
        && let Some(op) = op_name(path)
    {
        Some((op, args))
    } else {
        None
    }
}

fn is_f32_tensor(t: &Ty) -> bool {
    matches!(t, Ty::Tensor { elem, .. } if matches!(**elem, Ty::Prim(Prim::F32)))
}

/// Whether `op(args)` (producing `result`) and everything below it can run as a chain.
fn supported(op: &str, args: &[hir::Expr], result: &Ty) -> bool {
    if !is_f32_tensor(result) {
        return false;
    }
    args.iter().enumerate().all(|(i, a)| {
        if op == "scale" && i == 1 {
            return true; // the scalar factor
        }
        match chain_call(a) {
            Some((inner, inner_args)) => supported(inner, inner_args, &a.ty),
            None => is_f32_tensor(&a.ty),
        }
    })
}

impl Interp {
    /// Runs `path(args)` as a GPU chain when it is one (at least two operations nested), else
    /// returns `None` and the caller evaluates it normally.
    pub(super) fn gpu_chain(
        &self,
        path: &[String],
        args: &[hir::Expr],
        result: &Ty,
        ctx: &mut Ctx,
    ) -> Result<Option<Value>, Ctrl> {
        if self.device() != Device::Gpu || self.quantization() != Quantization::None {
            return Ok(None);
        }
        let Some(op) = op_name(path) else {
            return Ok(None);
        };
        if !args.iter().any(|a| chain_call(a).is_some()) || !supported(op, args, result) {
            return Ok(None);
        }
        let gpu = match gpu::global() {
            Ok(g) => g,
            Err(e) => return fail(e),
        };
        let node = self.gpu_op(op, args, &gpu, ctx)?;
        let data = gpu
            .download(&node.buf)
            .map_err(|e| Ctrl::Fail(RuntimeError::new(e)))?;
        Ok(Some(Value::Tensor(Arc::new(TensorVal::from_f32(
            node.shape, data,
        )))))
    }

    fn gpu_node(&self, e: &hir::Expr, gpu: &Gpu, ctx: &mut Ctx) -> Result<Node, Ctrl> {
        if let Some((op, args)) = chain_call(e) {
            return self.gpu_op(op, args, gpu, ctx);
        }
        // A leaf: evaluate it on the host once and upload it.
        match self.eval(e, ctx)? {
            Value::Tensor(t) => Ok(Node {
                buf: gpu.upload(&t.to_f32_vec()),
                shape: t.shape.clone(),
            }),
            other => fail(format!("expected a tensor, found {}", other.type_name())),
        }
    }

    fn gpu_op(&self, op: &str, args: &[hir::Expr], gpu: &Gpu, ctx: &mut Ctx) -> Result<Node, Ctrl> {
        let a = self.gpu_node(&args[0], gpu, ctx)?;
        let elementwise = |kind: Elementwise, a: Node, b: Option<&Node>, scalar: f32| Node {
            buf: gpu.elementwise_buf(kind, &a.buf, b.map(|b| &b.buf), scalar),
            shape: a.shape,
        };
        match op {
            "add" | "sub" | "mul" | "div" => {
                let b = self.gpu_node(&args[1], gpu, ctx)?;
                if a.shape != b.shape {
                    return fail(format!(
                        "std::tensor::{op}: shapes differ: {:?} and {:?}",
                        a.shape, b.shape
                    ));
                }
                let kind = match op {
                    "add" => Elementwise::Add,
                    "sub" => Elementwise::Sub,
                    "mul" => Elementwise::Mul,
                    _ => Elementwise::Div,
                };
                Ok(elementwise(kind, a, Some(&b), 0.0))
            }
            "scale" => {
                let factor = match self.eval(&args[1], ctx)? {
                    Value::Float(x) => x as f32,
                    Value::Int(i) => i as f32,
                    other => {
                        return fail(format!(
                            "`scale` needs a number, found {}",
                            other.type_name()
                        ));
                    }
                };
                Ok(elementwise(Elementwise::Scale, a, None, factor))
            }
            "relu" => Ok(elementwise(Elementwise::Relu, a, None, 0.0)),
            "sigmoid" => Ok(elementwise(Elementwise::Sigmoid, a, None, 0.0)),
            "softmax" => {
                let Some(&cols) = a.shape.last() else {
                    return fail("`softmax` needs a tensor with at least one dimension");
                };
                let rows = a.buf.len() / cols.max(1);
                Ok(Node {
                    buf: gpu.softmax_buf(&a.buf, rows, cols),
                    shape: a.shape,
                })
            }
            _ => {
                // matmul: [m, k] x [k, n] or [m, k] x [k]
                let b = self.gpu_node(&args[1], gpu, ctx)?;
                let [m, k] = a.shape[..] else {
                    return fail("`matmul` needs a matrix as its first argument");
                };
                let (k2, n, vector) = match b.shape[..] {
                    [k2, n] => (k2, n, false),
                    [k2] => (k2, 1, true),
                    _ => return fail("`matmul` needs a matrix or a vector as its second argument"),
                };
                if k != k2 {
                    return fail(format!("`matmul` inner dimensions differ: {k} and {k2}"));
                }
                Ok(Node {
                    buf: gpu.matmul_buf(&a.buf, &b.buf, m, k, n),
                    shape: if vector { vec![m] } else { vec![m, n] },
                })
            }
        }
    }
}
