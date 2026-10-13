//! nvpm, the Novar package manager.
//!
//! * `manifest`: `novar.toml` (package metadata, registry and path dependencies)
//! * `resolver`: picks one version per package, newest first, backing up on conflicts
//! * `lock`: `novar.lock`, so every machine builds the same graph
//! * `registry`: directory and HTTP registries, publishing, checksums
//! * `project`: discovering a project, installing packages, handing dependencies to the compiler

pub mod lock;
pub mod manifest;
pub mod project;
pub mod registry;
pub mod resolver;

pub use lock::Lock;
pub use manifest::Manifest;
pub use project::Project;
