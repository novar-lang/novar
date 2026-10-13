//! Lowering of the typed HIR to MLIR (`func`, `arith` and `cf` dialects), plus a JIT for it.
//!
//! The native backend covers the numeric core of Novar: integers, floats and booleans, local
//! bindings, arithmetic and comparisons, `if`/`else`, early `return` and calls between
//! functions. Anything else is reported as a diagnostic instead of being miscompiled.
//!
//! Integer arithmetic wraps on overflow, and division by zero is undefined behaviour, unlike the
//! interpreter which checks both. Build with `--features llvm` (see `scripts/llvm-env.sh`).

#[cfg(feature = "llvm")]
mod lower;

#[cfg(feature = "llvm")]
pub use lower::*;

/// Whether this build includes the MLIR/LLVM backend.
pub const ENABLED: bool = cfg!(feature = "llvm");
