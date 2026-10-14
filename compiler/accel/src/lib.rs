//! Code generation for accelerators.
//!
//! * [`stablehlo`] turns tensor functions into StableHLO text, the input of XLA / PJRT and with
//!   it of TPUs.
//! * `ptx` (feature `llvm`) turns the `std::tensor` operations a program uses into PTX kernels
//!   through LLVM's NVPTX backend.
//!
//! Neither output is executed by this repository's tests: there is no TPU or NVIDIA GPU on the
//! development machine. They are checked structurally (StableHLO text, the LLVM verifier and the
//! PTX header and entry points). Execution on a real GPU in this repository goes through
//! `telvax_runtime`'s `gpu` feature instead.

mod stablehlo;
pub use stablehlo::{StableHlo, stablehlo};

#[cfg(feature = "llvm")]
mod ptx;
#[cfg(feature = "llvm")]
pub use ptx::{Ptx, ptx, used_ops};

/// Names of the `std::tensor` operations a program calls, in first-use order.
pub fn tensor_ops(program: &telvax_hir::hir::Program) -> Vec<String> {
    let mut out = Vec::new();
    for f in &program.fns {
        walk_block(&f.body, &mut |e| {
            if let telvax_hir::hir::ExprKind::Extern(path) = &e.kind
                && path.len() == 3
                && path[0] == "std"
                && path[1] == "tensor"
                && !out.contains(&path[2])
            {
                out.push(path[2].clone());
            }
        });
    }
    out
}

fn walk_block(b: &telvax_hir::hir::Block, f: &mut impl FnMut(&telvax_hir::hir::Expr)) {
    use telvax_hir::hir::Stmt;
    for s in &b.stmts {
        match s {
            Stmt::Let { init, .. } => walk(init, f),
            Stmt::Expr(e) => walk(e, f),
        }
    }
    if let Some(t) = &b.tail {
        walk(t, f);
    }
}

fn walk(e: &telvax_hir::hir::Expr, f: &mut impl FnMut(&telvax_hir::hir::Expr)) {
    use telvax_hir::hir::ExprKind::*;
    f(e);
    match &e.kind {
        Call { callee, args } => {
            walk(callee, f);
            args.iter().for_each(|a| walk(a, f));
        }
        Unary { expr, .. } => walk(expr, f),
        Binary { lhs, rhs, .. } => {
            walk(lhs, f);
            walk(rhs, f);
        }
        Block(b) => walk_block(b, f),
        If {
            cond,
            then,
            otherwise,
        } => {
            walk(cond, f);
            walk_block(then, f);
            if let Some(o) = otherwise {
                walk(o, f);
            }
        }
        Return(Some(x)) => walk(x, f),
        Field { base, .. } => walk(base, f),
        _ => {}
    }
}
