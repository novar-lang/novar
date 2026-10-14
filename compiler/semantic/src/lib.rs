//! High-Level IR: name resolution, type checking and tensor-shape checking.

mod check;
pub mod hir;
pub mod stdlib;
pub mod ty;

pub use ty::{Dim, Prim, Ty};

/// Numeric precision used for `std::tensor::matmul` on `f32` tensors.
///
/// `Int8` is symmetric per-tensor dynamic quantization: both operands are scaled by
/// `max|x| / 127`, rounded to `i8` (ties away from zero), multiplied with `i32` accumulation and
/// scaled back. The interpreter and the native backend implement exactly the same arithmetic, so
/// their results agree bit for bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quantization {
    #[default]
    None,
    Int8,
    /// Int8 with one scale per row of the left operand and one per column of the right one
    /// (`--quantize int8-channel`). More accurate than [`Quantization::Int8`] when columns differ
    /// in magnitude. Interpreter only.
    Int8Channel,
    /// Like `Int8Channel`, but the weights (the right operand) use 4 bits: values in `-7..=7`
    /// (`--quantize int4-channel`). The activations stay 8-bit. Interpreter only.
    Int4Channel,
}

use novar_diagnostics::Diagnostic;

/// Checks a parsed program, returning its typed HIR and every diagnostic found.
/// The HIR is always produced so later tooling can work on partially valid code.
pub fn lower(program: &novar_ast::Program) -> (hir::Program, Vec<Diagnostic>) {
    check::lower(program)
}
