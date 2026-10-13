//! High-Level IR: name resolution, type checking and tensor-shape checking.

mod check;
pub mod hir;
pub mod stdlib;
pub mod ty;

pub use ty::{Dim, Prim, Ty};

use novar_diagnostics::Diagnostic;

/// Checks a parsed program, returning its typed HIR and every diagnostic found.
/// The HIR is always produced so later tooling can work on partially valid code.
pub fn lower(program: &novar_ast::Program) -> (hir::Program, Vec<Diagnostic>) {
    check::lower(program)
}
