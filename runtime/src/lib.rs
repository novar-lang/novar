//! Novar runtime: a tree-walking interpreter over the typed HIR plus the std implementations.

mod interp;
mod std_impl;
mod value;

pub use interp::{EndpointError, Interp, RuntimeError};
pub use std_impl::embed_text;
pub use value::{Db, Native, Query, StructVal, TensorVal, Value};
