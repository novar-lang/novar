//! A global allocator that counts what a program allocates, for `telvax profile`.
//!
//! A binary that wants memory figures installs it:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: telvax_runtime::alloc_count::CountingAlloc = telvax_runtime::alloc_count::CountingAlloc;
//! ```
//!
//! It counts only while a profiler is running; otherwise it adds one flag check to each
//! allocation. The counts are for the whole process, so with several threads a call also counts
//! what the other threads allocate meanwhile.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Bytes allocated, and allocations made, while counting.
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static COUNT: AtomicU64 = AtomicU64::new(0);
/// Bytes alive now (of what was allocated while counting), and the most there has been.
static CURRENT: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Allocations and the memory they held during a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memory {
    /// The most bytes alive at once beyond what was alive when counting started.
    pub peak: u64,
    /// Bytes allocated in all, freed or not.
    pub allocated: u64,
    /// How many allocations.
    pub count: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CountingAlloc;

fn add(bytes: u64) {
    ALLOCATED.fetch_add(bytes, Relaxed);
    COUNT.fetch_add(1, Relaxed);
    let now = CURRENT.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(now, Relaxed);
}

fn grow(bytes: u64) {
    ALLOCATED.fetch_add(bytes, Relaxed);
    let now = CURRENT.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(now, Relaxed);
}

fn sub(bytes: u64) {
    let mut now = CURRENT.load(Relaxed);
    while let Err(seen) =
        CURRENT.compare_exchange_weak(now, now.saturating_sub(bytes), Relaxed, Relaxed)
    {
        now = seen;
    }
}

// SAFETY: every call is passed to `System` unchanged; the counters only observe the sizes.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() && ENABLED.load(Relaxed) {
            add(layout.size() as u64);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() && ENABLED.load(Relaxed) {
            add(layout.size() as u64);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        if ENABLED.load(Relaxed) {
            sub(layout.size() as u64);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() && ENABLED.load(Relaxed) {
            if new_size >= layout.size() {
                grow((new_size - layout.size()) as u64);
            } else {
                sub((layout.size() - new_size) as u64);
            }
        }
        p
    }
}

/// Starts counting. `None` when the counting allocator is not the program's allocator.
pub(crate) fn start() -> Option<Memory> {
    let before = COUNT.load(Relaxed);
    ENABLED.store(true, Relaxed);
    drop(std::hint::black_box(Box::new(0u64)));
    if COUNT.load(Relaxed) == before {
        ENABLED.store(false, Relaxed);
        return None;
    }
    PEAK.store(CURRENT.load(Relaxed), Relaxed);
    // The fields hold the starting point: the bytes alive, the total and the count.
    Some(Memory {
        peak: CURRENT.load(Relaxed),
        allocated: ALLOCATED.load(Relaxed),
        count: COUNT.load(Relaxed),
    })
}

/// What happened since `start` (the value `start` returned).
pub(crate) fn since(start: Memory) -> Memory {
    Memory {
        peak: PEAK.load(Relaxed).saturating_sub(start.peak),
        allocated: ALLOCATED.load(Relaxed).saturating_sub(start.allocated),
        count: COUNT.load(Relaxed).saturating_sub(start.count),
    }
}

/// Bytes allocated so far while counting (0 when not counting).
pub(crate) fn allocated() -> u64 {
    ALLOCATED.load(Relaxed)
}
