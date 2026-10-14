//! Telvax runtime: a tree-walking interpreter over the typed HIR plus the std implementations.

// wgpu's types nest deeply enough that proving `Send` for the GPU handle overflows the default
// limit (a warning today, a hard error in a future compiler).
#![cfg_attr(feature = "gpu", recursion_limit = "256")]

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
pub use value::{
    Db, EnumVal, MapKey, MapVal, Native, Query, StructVal, TableVal, TensorVal, Value, VecVal,
};
