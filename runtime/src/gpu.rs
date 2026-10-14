//! Tensor kernels on the GPU, through `wgpu` (Metal on macOS, Vulkan on Linux, DirectX on Windows).
//!
//! Each operation uploads its operands, runs a WGSL compute shader and reads the result back. The
//! arithmetic is `f32` on the device, so results differ from the CPU path (which accumulates in
//! `f64`) in the last bits; they agree to ordinary single-precision tolerances.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use wgpu::util::DeviceExt;

const WORKGROUP: u32 = 256;
/// Largest dispatch along one axis the specification guarantees.
const MAX_GROUPS_X: u32 = 65_535;

const ELEMENTWISE: &str = r#"
struct Params { n: u32, op: u32, scalar: f32, stride: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.stride + gid.x;
    if (i >= p.n) { return; }
    let x = a[i];
    var r: f32;
    switch (p.op) {
        case 0u: { r = x + b[i]; }
        case 1u: { r = x - b[i]; }
        case 2u: { r = x * b[i]; }
        case 3u: { r = x / b[i]; }
        case 4u: { r = x * p.scalar; }
        case 5u: { r = max(x, 0.0); }
        case 6u: { r = 1.0 / (1.0 + exp(-x)); }
        default: { r = x; }
    }
    out[i] = r;
}
"#;

const MATMUL: &str = r#"
struct Dims { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

const T: u32 = 16u;
var<workgroup> ta: array<f32, 256>;
var<workgroup> tb: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;
    let tiles = (d.k + T - 1u) / T;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let ak = t * T + lid.x;
        let bk = t * T + lid.y;
        ta[lid.y * T + lid.x] = select(0.0, a[row * d.k + ak], row < d.m && ak < d.k);
        tb[lid.y * T + lid.x] = select(0.0, b[bk * d.n + col], bk < d.k && col < d.n);
        workgroupBarrier();
        for (var i = 0u; i < T; i = i + 1u) {
            acc = acc + ta[lid.y * T + i] * tb[i * T + lid.x];
        }
        workgroupBarrier();
    }
    if (row < d.m && col < d.n) {
        c[row * d.n + col] = acc;
    }
}
"#;

const SOFTMAX: &str = r#"
struct Dims { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> d: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= d.rows) { return; }
    let base = r * d.cols;
    var m = a[base];
    for (var j = 1u; j < d.cols; j = j + 1u) { m = max(m, a[base + j]); }
    var s = 0.0;
    for (var j = 0u; j < d.cols; j = j + 1u) {
        let e = exp(a[base + j] - m);
        out[base + j] = e;
        s = s + e;
    }
    for (var j = 0u; j < d.cols; j = j + 1u) { out[base + j] = out[base + j] / s; }
}
"#;

const REDUCE: &str = r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> part: array<f32, 256>;

// out[0] = sum(a[i] * b[i]); callers pass b = a for a plain sum of squares, or use the ones kernel
// below for a plain sum.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0.0;
    for (var i = lid.x; i < p.n; i = i + 256u) { acc = acc + a[i] * b[i]; }
    part[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { part[lid.x] = part[lid.x] + part[lid.x + s]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { out[0] = part[0]; }
}
"#;

#[derive(Debug, Clone, Copy)]
pub enum Elementwise {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Scale = 4,
    Relu = 5,
    Sigmoid = 6,
}

/// `f32` data that lives on the GPU. Operations on `GpuBuf`s chain without copying through the
/// host; only [`Gpu::download`] brings a result back.
pub struct GpuBuf {
    buffer: wgpu::Buffer,
    len: usize,
}

impl GpuBuf {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::fmt::Debug for GpuBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuBuf({} floats)", self.len)
    }
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
    name: String,
    uploads: AtomicU64,
    downloads: AtomicU64,
}

// SAFETY: on native targets wgpu's device, queue and pipelines are `Send + Sync`; the compiler
// cannot prove it here because evaluating the auto traits of wgpu's backend types overflows.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gpu({})", self.name)
    }
}

static GLOBAL: OnceLock<Result<Arc<Gpu>, String>> = OnceLock::new();

/// The process-wide GPU, initialised on first use.
pub fn global() -> Result<Arc<Gpu>, String> {
    GLOBAL.get_or_init(|| Gpu::new().map(Arc::new)).clone()
}

fn bytes<T: bytemuck::NoUninit>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("no GPU adapter is available: {e}"))?;
            let info = adapter.get_info();
            let limits = adapter.limits();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("novar"),
                    required_limits: limits,
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("cannot open the GPU: {e}"))?;
            Ok(Gpu {
                device,
                queue,
                pipelines: Mutex::new(HashMap::new()),
                name: format!("{} ({:?})", info.name, info.backend),
                uploads: AtomicU64::new(0),
                downloads: AtomicU64::new(0),
            })
        })
    }

    /// Name and backend of the adapter, e.g. `Apple M4 (Metal)`.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn pipeline(&self, key: &'static str, source: &'static str) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry(key)
            .or_insert_with(|| {
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(key),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                Arc::new(
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(key),
                            layout: None,
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                )
            })
            .clone()
    }

    fn storage(&self, data: &[f32]) -> wgpu::Buffer {
        // Zero-length buffers are not allowed; pad to one element.
        let padded;
        let slice = if data.is_empty() {
            padded = [0.0f32];
            &padded[..]
        } else {
            data
        };
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes(slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    fn output(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Records one compute pass over `groups` workgroups and submits it. The result stays in the
    /// buffers; later passes on the same queue see it in order.
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        groups: (u32, u32, u32),
    ) {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Copies `data` to the GPU.
    pub fn upload(&self, data: &[f32]) -> GpuBuf {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        GpuBuf {
            buffer: self.storage(data),
            len: data.len(),
        }
    }

    /// Reads a GPU buffer back (this waits for every pass that writes it).
    pub fn download(&self, buf: &GpuBuf) -> Result<Vec<f32>, String> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        let size = (buf.len.max(1) * 4) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&buf.buffer, 0, &readback, 0, size);
        self.queue.submit([enc.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e:?}"))?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("cannot read the GPU result: {e}"))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| format!("cannot read the GPU result: {e}"))?;
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec();
        drop(data);
        readback.unmap();
        Ok(out)
    }

    /// `(uploads, downloads)` since the GPU was opened: how many times data crossed the bus.
    pub fn transfers(&self) -> (u64, u64) {
        (
            self.uploads.load(Ordering::Relaxed),
            self.downloads.load(Ordering::Relaxed),
        )
    }

    /// Elementwise `op` on device buffers. `b` is ignored by the unary operations and `Scale`.
    pub fn elementwise_buf(
        &self,
        op: Elementwise,
        a: &GpuBuf,
        b: Option<&GpuBuf>,
        scalar: f32,
    ) -> GpuBuf {
        let n = a.len;
        let out = self.output(n);
        let groups_total = (n as u32).div_ceil(WORKGROUP).max(1);
        let gx = groups_total.min(MAX_GROUPS_X);
        let gy = groups_total.div_ceil(gx);
        let params = self.uniform(&[n as u32, op as u32, scalar.to_bits(), gx * WORKGROUP]);
        let pipeline = self.pipeline("elementwise", ELEMENTWISE);
        let second = b.map_or(&a.buffer, |b| &b.buffer);
        self.dispatch(&pipeline, &[&a.buffer, second, &out, &params], (gx, gy, 1));
        GpuBuf {
            buffer: out,
            len: n,
        }
    }

    /// `[m, k] x [k, n]` row-major, on device buffers.
    pub fn matmul_buf(&self, a: &GpuBuf, b: &GpuBuf, m: usize, k: usize, n: usize) -> GpuBuf {
        let out = self.output(m * n);
        let dims = self.uniform(&[m as u32, k as u32, n as u32, 0]);
        let pipeline = self.pipeline("matmul", MATMUL);
        let groups = (
            (n as u32).div_ceil(16).max(1),
            (m as u32).div_ceil(16).max(1),
            1,
        );
        self.dispatch(&pipeline, &[&a.buffer, &b.buffer, &out, &dims], groups);
        GpuBuf {
            buffer: out,
            len: m * n,
        }
    }

    /// Softmax over the last axis of a `[rows, cols]` matrix, on a device buffer.
    pub fn softmax_buf(&self, a: &GpuBuf, rows: usize, cols: usize) -> GpuBuf {
        let out = self.output(rows * cols);
        let dims = self.uniform(&[rows as u32, cols as u32, 0, 0]);
        let pipeline = self.pipeline("softmax", SOFTMAX);
        self.dispatch(
            &pipeline,
            &[&a.buffer, &out, &dims],
            ((rows as u32).div_ceil(64).max(1), 1, 1),
        );
        GpuBuf {
            buffer: out,
            len: rows * cols,
        }
    }

    /// `sum(a[i] * b[i])` on device buffers, read back as one number.
    pub fn dot_buf(&self, a: &GpuBuf, b: &GpuBuf) -> Result<f32, String> {
        let out = self.output(1);
        let params = self.uniform(&[a.len as u32, 0, 0, 0]);
        let pipeline = self.pipeline("reduce", REDUCE);
        self.dispatch(&pipeline, &[&a.buffer, &b.buffer, &out, &params], (1, 1, 1));
        Ok(self.download(&GpuBuf {
            buffer: out,
            len: 1,
        })?[0])
    }

    pub fn elementwise(
        &self,
        op: Elementwise,
        a: &[f32],
        b: &[f32],
        scalar: f32,
    ) -> Result<Vec<f32>, String> {
        let pa = self.upload(a);
        let pb = (!b.is_empty()).then(|| self.upload(b));
        self.download(&self.elementwise_buf(op, &pa, pb.as_ref(), scalar))
    }

    /// `[m, k] x [k, n]` row-major.
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let (pa, pb) = (self.upload(a), self.upload(b));
        self.download(&self.matmul_buf(&pa, &pb, m, k, n))
    }

    /// Softmax over the last axis of a `[rows, cols]` matrix.
    pub fn softmax(&self, a: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        let pa = self.upload(a);
        self.download(&self.softmax_buf(&pa, rows, cols))
    }

    /// `sum(a[i] * b[i])`; pass the same slice twice for a sum of squares, or use [`Gpu::sum`].
    pub fn dot(&self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        let (pa, pb) = (self.upload(a), self.upload(b));
        self.dot_buf(&pa, &pb)
    }

    pub fn sum(&self, a: &[f32]) -> Result<f32, String> {
        let ones = vec![1.0f32; a.len()];
        self.dot(a, &ones)
    }
}
