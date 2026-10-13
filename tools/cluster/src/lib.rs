//! Distributed execution for Novar.
//!
//! `std::dist::map(items, f)` calls the top-level function `f` on every item. By default the calls
//! run on the local machine's threads; with a [`ClusterExecutor`] they run on remote
//! [`worker`]s instead, started with `novar worker`. Calls travel as JSON, converted with the
//! function's static types, so any value that can cross an HTTP endpoint can be mapped.

mod coordinator;
pub mod worker;

pub use coordinator::ClusterExecutor;
pub use worker::{WorkerHandle, spawn};
