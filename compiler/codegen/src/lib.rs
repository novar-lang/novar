//! Native code generation: MLIR (LLVM dialect) -> LLVM IR -> optimised object -> executable.
//!
//! The pipeline is `novar_mlir::lower` (HIR to `func`/`arith`/`cf`), conversion to the LLVM
//! dialect, translation to textual LLVM IR, then `inkwell` parses, verifies and optimises it and
//! asks the host target machine for an object file. Linking is delegated to the system `cc`.
//!
//! Executables need an entry point `fn main() -> i32` whose value becomes the exit code.
//! Build with `--features llvm` (see `scripts/llvm-env.sh`).

#[cfg(feature = "llvm")]
mod build;

#[cfg(feature = "llvm")]
pub use build::*;

/// Whether this build includes the LLVM backend.
pub const ENABLED: bool = cfg!(feature = "llvm");
