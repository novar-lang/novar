//! The Telvax language server.
//!
//! Every feature runs on a [`analysis::Snapshot`]: the file is parsed and type-checked on each
//! change, and the typed HIR is indexed by source position. Diagnostics, hover, go to
//! definition, completion, document symbols and inlay hints (inferred types) are answered from
//! that index, so what the editor shows is exactly what the compiler decided.

pub mod analysis;
pub mod position;
pub mod server;

pub use server::run_stdio;
