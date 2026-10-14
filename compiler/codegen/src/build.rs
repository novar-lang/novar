use std::ffi::CStr;
use std::fmt;
use std::path::Path;
use std::process::Command;

use inkwell::OptimizationLevel;
use inkwell::context::Context as LlvmContext;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use telvax_diagnostics::Diagnostic;
use telvax_hir::hir;
use telvax_hir::{Quantization, Ty};

#[derive(Debug)]
pub enum BuildError {
    /// Source-level problems (e.g. language features the backend cannot compile yet).
    Diagnostics(Vec<Diagnostic>),
    /// Toolchain or environment failures.
    Tool(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::Diagnostics(d) => {
                write!(
                    f,
                    "{}",
                    d.iter()
                        .map(|d| d.message.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            }
            BuildError::Tool(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for BuildError {}

fn tool<T, E: fmt::Display>(r: Result<T, E>) -> Result<T, BuildError> {
    r.map_err(|e| BuildError::Tool(e.to_string()))
}

/// Textual LLVM IR for every function in `program`.
pub fn llvm_ir(program: &hir::Program) -> Result<String, BuildError> {
    llvm_ir_with(program, Quantization::None)
}

/// Like [`llvm_ir`], with `f32` matrix products at the given precision.
pub fn llvm_ir_with(program: &hir::Program, quant: Quantization) -> Result<String, BuildError> {
    let ctx = telvax_mlir::context();
    let mut module =
        telvax_mlir::lower_with(&ctx, program, quant).map_err(BuildError::Diagnostics)?;
    tool(telvax_mlir::lower_to_llvm_dialect(&ctx, &mut module))?;
    // SAFETY: `module` is a live LLVM-dialect module; the returned buffer is malloc'ed and owned by us.
    unsafe {
        let raw = mlir_sys::mlirTranslateModuleToLLVMIRToString(module.as_operation().to_raw());
        if raw.is_null() {
            return Err(BuildError::Tool("translation to LLVM IR failed".into()));
        }
        let text = CStr::from_ptr(raw).to_string_lossy().into_owned();
        libc::free(raw.cast());
        Ok(text)
    }
}

/// Optimisation level, `0` (none) to `3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptLevel(pub u8);

impl OptLevel {
    fn pipeline(self) -> &'static str {
        match self.0 {
            0 => "default<O0>",
            1 => "default<O1>",
            2 => "default<O2>",
            _ => "default<O3>",
        }
    }

    fn codegen(self) -> OptimizationLevel {
        match self.0 {
            0 => OptimizationLevel::None,
            1 => OptimizationLevel::Less,
            2 => OptimizationLevel::Default,
            _ => OptimizationLevel::Aggressive,
        }
    }
}

/// Parses `ir`, runs the optimisation pipeline and returns the optimised IR text.
pub fn optimize_ir(ir: &str, opt: OptLevel) -> Result<String, BuildError> {
    let (llvm, tm) = parse_and_optimize(ir, opt)?;
    drop(tm);
    Ok(llvm.print_to_string().to_string())
}

fn parse_and_optimize<'c>(
    ir: &str,
    opt: OptLevel,
) -> Result<(inkwell::module::Module<'c>, TargetMachine), BuildError> {
    // The context must outlive the module; leaking one per build keeps the API simple and
    // builds are one-shot compiler invocations.
    let context: &'c LlvmContext = Box::leak(Box::new(LlvmContext::create()));
    tool(Target::initialize_native(&InitializationConfig::default()))?;
    let triple = TargetMachine::get_default_triple();
    let target = tool(Target::from_triple(&triple))?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let tm = target
        .create_target_machine(
            &triple,
            cpu.to_str().unwrap_or("generic"),
            features.to_str().unwrap_or(""),
            opt.codegen(),
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| BuildError::Tool("cannot create a target machine for the host".into()))?;

    let buffer = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), "telvax");
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|e| BuildError::Tool(format!("invalid LLVM IR: {e}")))?;
    tool(module.verify().map_err(|e| e.to_string()))?;
    module.set_triple(&triple);
    module.set_data_layout(&tm.get_target_data().get_data_layout());
    tool(module.run_passes(opt.pipeline(), &tm, PassBuilderOptions::create()))?;
    Ok((module, tm))
}

/// Compiles `program` to a native object file at `path`.
pub fn build_object(program: &hir::Program, opt: OptLevel, path: &Path) -> Result<(), BuildError> {
    build_object_with(program, opt, Quantization::None, path)
}

/// Like [`build_object`], with `f32` matrix products at the given precision.
pub fn build_object_with(
    program: &hir::Program,
    opt: OptLevel,
    quant: Quantization,
    path: &Path,
) -> Result<(), BuildError> {
    let ir = llvm_ir_with(program, quant)?;
    let (module, tm) = parse_and_optimize(&ir, opt)?;
    tool(tm.write_to_file(&module, FileType::Object, path))
}

/// Checks that `program` has a usable `def main() -> int32`.
pub fn check_entry_point(program: &hir::Program) -> Result<(), BuildError> {
    let diag = |span: std::ops::Range<usize>, msg: &str| {
        BuildError::Diagnostics(vec![Diagnostic::error(span, msg)])
    };
    match program.fns.iter().find(|f| f.name == "main") {
        None => Err(diag(
            0..0,
            "an executable needs an entry point: `def main() -> int32`",
        )),
        Some(f) if !f.params.is_empty() || f.ret != Ty::Prim(telvax_hir::Prim::I32) => Err(diag(
            f.span.clone(),
            "`main` must have the signature `def main() -> int32`",
        )),
        Some(_) => Ok(()),
    }
}

/// Compiles and links `program` into the executable `out`. `main`'s result is the exit code.
pub fn build_executable(
    program: &hir::Program,
    opt: OptLevel,
    out: &Path,
) -> Result<(), BuildError> {
    build_executable_with(program, opt, Quantization::None, out)
}

/// Like [`build_executable`], with `f32` matrix products at the given precision.
pub fn build_executable_with(
    program: &hir::Program,
    opt: OptLevel,
    quant: Quantization,
    out: &Path,
) -> Result<(), BuildError> {
    check_entry_point(program)?;
    let obj = out.with_extension("o");
    build_object_with(program, opt, quant, &obj)?;
    let status = Command::new("cc").arg(&obj).arg("-o").arg(out).status();
    let _ = std::fs::remove_file(&obj);
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(BuildError::Tool(format!("linker `cc` failed with {s}"))),
        Err(e) => Err(BuildError::Tool(format!(
            "cannot run the system linker `cc`: {e}"
        ))),
    }
}
