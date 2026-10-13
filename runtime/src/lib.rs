//! Novar runtime: a tree-walking interpreter over the typed HIR plus the std implementations.

#[cfg(feature = "gpu")]
pub mod gpu;
mod interp;
#[cfg(feature = "live")]
mod services;
mod std_impl;
mod table;
mod tensor;
mod value;

pub use interp::{Device, EndpointError, Executor, Interp, RuntimeError};
pub use std_impl::embed_text;
pub use value::{Db, Native, Query, StructVal, TableVal, TensorVal, Value, VecVal};
