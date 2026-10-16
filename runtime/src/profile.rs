//! Profiling of a program run by the interpreter: time and calls per function and per line, the
//! time and size of each `std` operation, and the memory the program allocates.
//!
//! A [`Profiler`] is installed with [`Interp::set_profiler`]. The interpreter then reports each
//! call, each statement and each `std` operation. Times are wall-clock, measured by the
//! interpreter around what the program does, so they show where the *program* spends its time.
//! The time of a function's own statements is its *self* time; its *total* time adds the calls
//! it makes.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::Interp;
use crate::value::Value;

pub use crate::alloc_count::Memory;

/// Time, calls and allocation of one function.
#[derive(Debug, Clone, Default)]
struct FnStat {
    calls: u64,
    self_ns: u64,
    total_ns: u64,
    alloc_self: u64,
    alloc_total: u64,
}

/// How often a statement started and how long it took before the next one of its call started
/// (the time of calls it made is not counted).
#[derive(Debug, Clone, Default)]
struct LineStat {
    hits: u64,
    self_ns: u64,
}

#[derive(Debug, Clone, Default)]
struct OpStat {
    calls: u64,
    total_ns: u64,
    items_in: u64,
    items_out: u64,
    bytes_out: u64,
}

#[derive(Debug, Default)]
struct Data {
    fns: HashMap<String, FnStat>,
    lines: HashMap<usize, LineStat>,
    ops: HashMap<String, OpStat>,
    /// `main;parse;std.text.split` -> the nanoseconds spent in the last frame itself.
    stacks: HashMap<String, u64>,
}

/// Collects what the interpreter reports. Create one, install it, run the program, then call
/// [`Profiler::report`].
#[derive(Debug)]
pub struct Profiler {
    data: Mutex<Data>,
    started: Instant,
    memory_at_start: Option<Memory>,
}

impl Profiler {
    /// Starts measuring now (and counting allocations, when the program was built with the
    /// counting allocator).
    pub fn new() -> Arc<Profiler> {
        let memory_at_start = crate::alloc_count::start();
        Arc::new(Profiler {
            data: Mutex::new(Data::default()),
            started: Instant::now(),
            memory_at_start,
        })
    }

    fn data(&self) -> std::sync::MutexGuard<'_, Data> {
        self.data.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Everything measured since [`Profiler::new`].
    pub fn report(&self) -> Report {
        let wall = self.started.elapsed();
        let memory = self.memory_at_start.map(crate::alloc_count::since);
        let data = self.data();
        let mut functions: Vec<FnRow> = data
            .fns
            .iter()
            .map(|(name, s)| FnRow {
                name: name.clone(),
                calls: s.calls,
                self_time: Duration::from_nanos(s.self_ns),
                total_time: Duration::from_nanos(s.total_ns),
                alloc_self: s.alloc_self,
                alloc_total: s.alloc_total,
            })
            .collect();
        functions.sort_by(|a, b| b.self_time.cmp(&a.self_time).then(a.name.cmp(&b.name)));
        let mut lines: Vec<LineRow> = data
            .lines
            .iter()
            .map(|(offset, s)| LineRow {
                offset: *offset,
                hits: s.hits,
                self_time: Duration::from_nanos(s.self_ns),
            })
            .collect();
        lines.sort_by(|a, b| b.self_time.cmp(&a.self_time).then(a.offset.cmp(&b.offset)));
        let mut ops: Vec<OpRow> = data
            .ops
            .iter()
            .map(|(name, s)| OpRow {
                name: name.clone(),
                calls: s.calls,
                total_time: Duration::from_nanos(s.total_ns),
                items_in: s.items_in,
                items_out: s.items_out,
                bytes_out: s.bytes_out,
            })
            .collect();
        ops.sort_by(|a, b| b.total_time.cmp(&a.total_time).then(a.name.cmp(&b.name)));
        let mut stacks: Vec<(String, u64)> =
            data.stacks.iter().map(|(k, v)| (k.clone(), *v)).collect();
        stacks.sort();
        Report {
            wall,
            functions,
            lines,
            ops,
            memory,
            stacks,
        }
    }
}

/// One function in a [`Report`].
#[derive(Debug, Clone)]
pub struct FnRow {
    pub name: String,
    pub calls: u64,
    pub self_time: Duration,
    pub total_time: Duration,
    /// Bytes allocated by the function itself, and by it and the calls it made.
    pub alloc_self: u64,
    pub alloc_total: u64,
}

/// One statement in a [`Report`], by the byte offset where it starts.
#[derive(Debug, Clone)]
pub struct LineRow {
    pub offset: usize,
    pub hits: u64,
    pub self_time: Duration,
}

/// One `std` function in a [`Report`].
#[derive(Debug, Clone)]
pub struct OpRow {
    pub name: String,
    pub calls: u64,
    pub total_time: Duration,
    /// Rows, elements, items or bytes of the arguments that are tables, tensors, lists or text.
    pub items_in: u64,
    /// The same for the results.
    pub items_out: u64,
    /// Bytes held by the results that are tables or tensors, or the length of a text.
    pub bytes_out: u64,
}

/// What a run measured.
#[derive(Debug, Clone)]
pub struct Report {
    pub wall: Duration,
    pub functions: Vec<FnRow>,
    pub lines: Vec<LineRow>,
    pub ops: Vec<OpRow>,
    /// `None` when the program was not built with the counting allocator.
    pub memory: Option<Memory>,
    stacks: Vec<(String, u64)>,
}

impl Report {
    /// Call stacks with the time spent in their last frame, one per line (`a;b;c 1234`, in
    /// microseconds), the input of `flamegraph.pl` and `inferno`.
    pub fn collapsed(&self) -> String {
        let mut out = String::new();
        for (stack, ns) in &self.stacks {
            let micros = ns / 1000;
            if micros > 0 {
                out.push_str(&format!("{stack} {micros}\n"));
            }
        }
        out
    }

    /// The same as a file for <https://www.speedscope.app>.
    pub fn speedscope(&self, name: &str) -> serde_json::Value {
        let mut frames: Vec<String> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut samples = Vec::new();
        let mut weights = Vec::new();
        let mut total = 0u64;
        for (stack, ns) in &self.stacks {
            let micros = ns / 1000;
            if micros == 0 {
                continue;
            }
            let ids: Vec<usize> = stack
                .split(';')
                .map(|f| {
                    *index.entry(f.to_string()).or_insert_with(|| {
                        frames.push(f.to_string());
                        frames.len() - 1
                    })
                })
                .collect();
            samples.push(ids);
            weights.push(micros);
            total += micros;
        }
        serde_json::json!({
            "$schema": "https://www.speedscope.app/file-format-schema.json",
            "name": name,
            "exporter": "telvax profile",
            "activeProfileIndex": 0,
            "shared": {"frames": frames.iter().map(|f| serde_json::json!({"name": f})).collect::<Vec<_>>()},
            "profiles": [{
                "type": "sampled",
                "name": name,
                "unit": "microseconds",
                "startValue": 0,
                "endValue": total,
                "samples": samples,
                "weights": weights,
            }],
        })
    }

    /// The whole report as JSON. `locate` says where an offset is (`file.tvx:12`).
    pub fn to_json(&self, locate: &dyn Fn(usize) -> String) -> serde_json::Value {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        serde_json::json!({
            "wall_ms": ms(self.wall),
            "functions": self.functions.iter().map(|f| serde_json::json!({
                "name": f.name, "calls": f.calls,
                "self_ms": ms(f.self_time), "total_ms": ms(f.total_time),
                "alloc_self_bytes": f.alloc_self, "alloc_total_bytes": f.alloc_total,
            })).collect::<Vec<_>>(),
            "lines": self.lines.iter().map(|l| serde_json::json!({
                "at": locate(l.offset), "hits": l.hits, "self_ms": ms(l.self_time),
            })).collect::<Vec<_>>(),
            "operations": self.ops.iter().map(|o| serde_json::json!({
                "name": o.name, "calls": o.calls, "total_ms": ms(o.total_time),
                "items_in": o.items_in, "items_out": o.items_out, "bytes_out": o.bytes_out,
            })).collect::<Vec<_>>(),
            "memory": self.memory.map(|m| serde_json::json!({
                "peak_bytes": m.peak, "allocated_bytes": m.allocated, "allocations": m.count,
            })),
        })
    }

    /// The report as tables for a terminal. `locate` says where a statement is, with its text;
    /// `top` limits the lines shown.
    pub fn render(&self, locate: &dyn Fn(usize) -> String, top: usize) -> String {
        use std::fmt::Write;
        let ms = |d: Duration| format!("{:.2}", d.as_secs_f64() * 1e3);
        let mut out = String::new();
        let _ = writeln!(out, "wall time {} ms", ms(self.wall));
        if let Some(m) = self.memory {
            let _ = writeln!(
                out,
                "memory    peak {}, {} allocated in {} allocations",
                bytes(m.peak),
                bytes(m.allocated),
                m.count
            );
        } else {
            let _ = writeln!(out, "memory    not measured (no counting allocator)");
        }
        let total_self: Duration = self.functions.iter().map(|f| f.self_time).sum();
        if !self.functions.is_empty() {
            let _ = writeln!(out, "\nfunctions, by time in the function itself");
            let _ = writeln!(
                out,
                "  {:>7}  {:>10}  {:>10}  {:>6}  {:>10}  function",
                "calls", "self ms", "total ms", "self%", "alloc self"
            );
            for f in &self.functions {
                let pct = if total_self.is_zero() {
                    0.0
                } else {
                    100.0 * f.self_time.as_secs_f64() / total_self.as_secs_f64()
                };
                let _ = writeln!(
                    out,
                    "  {:>7}  {:>10}  {:>10}  {:>5.1}%  {:>10}  {}",
                    f.calls,
                    ms(f.self_time),
                    ms(f.total_time),
                    pct,
                    if self.memory.is_some() {
                        bytes(f.alloc_self)
                    } else {
                        "-".into()
                    },
                    f.name
                );
            }
        }
        if !self.lines.is_empty() {
            let _ = writeln!(out, "\nstatements, by time until the next one starts");
            let _ = writeln!(out, "  {:>9}  {:>10}  where", "hits", "self ms");
            for l in self.lines.iter().take(top) {
                let _ = writeln!(
                    out,
                    "  {:>9}  {:>10}  {}",
                    l.hits,
                    ms(l.self_time),
                    locate(l.offset)
                );
            }
        }
        if !self.ops.is_empty() {
            let _ = writeln!(out, "\nstd operations, by time");
            let _ = writeln!(
                out,
                "  {:>7}  {:>10}  {:>10}  {:>10}  {:>10}  operation",
                "calls", "total ms", "in", "out", "bytes out"
            );
            for o in &self.ops {
                let _ = writeln!(
                    out,
                    "  {:>7}  {:>10}  {:>10}  {:>10}  {:>10}  {}",
                    o.calls,
                    ms(o.total_time),
                    o.items_in,
                    o.items_out,
                    bytes(o.bytes_out),
                    o.name
                );
            }
        }
        out
    }
}

/// `n` bytes as `1.5 MiB`.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

// ---- what the interpreter reports ---------------------------------------------------------

/// A call (or a `std` operation) in progress on this thread.
struct Frame {
    name: String,
    /// A `std` operation: shown in stacks and in the operations table, not as a function.
    op: bool,
    start: Instant,
    alloc_start: u64,
    /// Time and allocation of the calls made from here that have returned.
    child_ns: u64,
    child_alloc: u64,
    /// The statement that is running: where it starts, when, and `child_ns` then.
    stmt: Option<(usize, Instant, u64)>,
}

/// What one thread has measured and not yet handed over.
#[derive(Default)]
struct Local {
    frames: Vec<Frame>,
    data: Data,
}

thread_local! {
    static LOCAL: RefCell<Local> = RefCell::new(Local::default());
}

fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// Ends the statement that is running in the innermost frame.
fn finish_statement(frame: &mut Frame, data: &mut Data, now: Instant) {
    if let Some((offset, began, child_mark)) = frame.stmt.take() {
        let spent = nanos(now - began).saturating_sub(frame.child_ns - child_mark);
        data.lines.entry(offset).or_default().self_ns += spent;
    }
}

/// A statement starts in the innermost call.
pub(crate) fn statement(offset: usize) {
    let now = Instant::now();
    LOCAL.with(|l| {
        let mut l = l.borrow_mut();
        let Local { frames, data } = &mut *l;
        let Some(top) = frames.last_mut() else { return };
        finish_statement(top, data, now);
        data.lines.entry(offset).or_default().hits += 1;
        top.stmt = Some((offset, now, top.child_ns));
    });
}

/// Measures a call until it is dropped.
pub(crate) struct Probe {
    profiler: Arc<Profiler>,
}

impl Probe {
    pub(crate) fn enter(profiler: Arc<Profiler>, name: &str, op: bool) -> Probe {
        let alloc_start = crate::alloc_count::allocated();
        LOCAL.with(|l| {
            l.borrow_mut().frames.push(Frame {
                name: name.to_string(),
                op,
                start: Instant::now(),
                alloc_start,
                child_ns: 0,
                child_alloc: 0,
                stmt: None,
            });
        });
        Probe { profiler }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        let now = Instant::now();
        let alloc_now = crate::alloc_count::allocated();
        let outermost = LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            let Local { frames, data } = &mut *l;
            let Some(mut frame) = frames.pop() else {
                return false;
            };
            finish_statement(&mut frame, data, now);
            let total = nanos(now - frame.start);
            let self_ns = total.saturating_sub(frame.child_ns);
            let alloc_total = alloc_now.saturating_sub(frame.alloc_start);
            let alloc_self = alloc_total.saturating_sub(frame.child_alloc);
            let mut stack: Vec<&str> = frames.iter().map(|f| f.name.as_str()).collect();
            stack.push(&frame.name);
            *data.stacks.entry(stack.join(";")).or_default() += self_ns;
            if !frame.op {
                // A recursive call is already counted by the call that is still running.
                let recursive = frames.iter().any(|f| !f.op && f.name == frame.name);
                let stat = data.fns.entry(frame.name.clone()).or_default();
                stat.calls += 1;
                stat.self_ns += self_ns;
                stat.alloc_self += alloc_self;
                if !recursive {
                    stat.total_ns += total;
                    stat.alloc_total += alloc_total;
                }
            }
            if let Some(parent) = frames.last_mut() {
                parent.child_ns += total;
                parent.child_alloc += alloc_total;
            }
            frames.is_empty()
        });
        if outermost {
            // Hand over what this thread measured while no call is running.
            let taken = LOCAL.with(|l| std::mem::take(&mut l.borrow_mut().data));
            self.profiler.merge(taken);
        }
    }
}

impl Profiler {
    fn merge(&self, from: Data) {
        let mut data = self.data();
        for (k, s) in from.fns {
            let to = data.fns.entry(k).or_default();
            to.calls += s.calls;
            to.self_ns += s.self_ns;
            to.total_ns += s.total_ns;
            to.alloc_self += s.alloc_self;
            to.alloc_total += s.alloc_total;
        }
        for (k, s) in from.lines {
            let to = data.lines.entry(k).or_default();
            to.hits += s.hits;
            to.self_ns += s.self_ns;
        }
        for (k, ns) in from.stacks {
            *data.stacks.entry(k).or_default() += ns;
        }
    }

    /// A `std` function ran: how long, and how big its inputs and results were.
    pub(crate) fn operation(&self, name: &str, took: Duration, args: &[Value], result: &Value) {
        let mut data = self.data();
        let op = data.ops.entry(name.to_string()).or_default();
        op.calls += 1;
        op.total_ns += nanos(took);
        op.items_in += args.iter().map(|a| size(a).0).sum::<u64>();
        let (items, bytes) = size(result);
        op.items_out += items;
        op.bytes_out += bytes;
    }
}

/// How many rows, elements or items a value holds, and the bytes behind it.
fn size(v: &Value) -> (u64, u64) {
    match v {
        Value::Table(t) => (
            t.batch.num_rows() as u64,
            t.batch.get_array_memory_size() as u64,
        ),
        Value::Tensor(t) => (
            t.shape.iter().product::<usize>() as u64,
            t.array.get_array_memory_size() as u64,
        ),
        Value::Vec(items) => (items.len() as u64, 0),
        Value::Map(m) => (m.len() as u64, 0),
        Value::Str(s) => (s.len() as u64, s.len() as u64),
        Value::Ok(x) => size(x),
        _ => (0, 0),
    }
}

impl Interp {
    /// Installs (or, with `None`, removes) the profiler that the interpreter reports to.
    pub fn set_profiler(&self, profiler: Option<Arc<Profiler>>) {
        use std::sync::atomic::Ordering::Relaxed;
        self.profiling.store(profiler.is_some(), Relaxed);
        *self.profiler.write().unwrap_or_else(|e| e.into_inner()) = profiler;
    }

    pub(crate) fn current_profiler(&self) -> Option<Arc<Profiler>> {
        self.profiler
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}
