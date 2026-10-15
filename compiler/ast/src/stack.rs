//! Running recursive work with room to spare.
//!
//! The parser and the checker recurse once per level of a program's nesting. A text nested a few
//! hundred levels deep needs more stack than the 2 MB of a test thread or the 8 MB of a main
//! thread, and a stack overflow ends the whole process instead of reporting an error. Work done
//! through [`with_big_stack`] gets a stack of its own; only the pages it touches are ever mapped.

/// Size of the stack the work runs on.
pub const STACK_BYTES: usize = 256 * 1024 * 1024;

/// Runs `work` on a thread with [`STACK_BYTES`] of stack and returns its result. A panic inside
/// `work` is raised again in the caller. Where threads do not exist (WebAssembly), or cannot be
/// started, `work` runs on the current one.
pub fn with_big_stack<R: Send>(work: impl FnOnce() -> R + Send) -> R {
    #[cfg(target_arch = "wasm32")]
    {
        work()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::Mutex;

        let slot = Mutex::new(Some(work));
        let take = || {
            let work = slot.lock().ok().and_then(|mut w| w.take());
            work.map(|w| w())
        };
        let result = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(STACK_BYTES)
                .spawn_scoped(scope, take)
                .map(|worker| worker.join())
        });
        match result {
            Ok(Ok(Some(done))) => done,
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            // No thread could be started (or the work was already taken): run it here.
            _ => take().expect("the work runs once"),
        }
    }
}
