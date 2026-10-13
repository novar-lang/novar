//! Native backend commands (`jit`, `build`), available with the `llvm` feature.

#[cfg(feature = "llvm")]
mod imp {
    use crate::{CliError, check, failed};
    use novar_codegen::{
        OptLevel, build_executable_with, build_object_with, llvm_ir_with, optimize_ir,
    };
    use novar_mlir::JitValue;
    use std::path::Path;

    fn report(err: novar_codegen::BuildError, file: &str) -> CliError {
        if let novar_codegen::BuildError::Diagnostics(diags) = &err {
            let src = std::fs::read_to_string(file).unwrap_or_default();
            eprint!("{}", novar_diagnostics::render(file, &src, diags, false));
        }
        CliError::Failed(err.to_string())
    }

    fn jit_arg(raw: &str) -> JitValue {
        if let Ok(i) = raw.parse::<i64>() {
            JitValue::Int(i)
        } else if let Ok(x) = raw.parse::<f64>() {
            JitValue::Float(x)
        } else {
            JitValue::Bool(raw == "true")
        }
    }

    pub fn jit(file: &str, func: &str, args: &[&str]) -> Result<(), CliError> {
        let (quant, args) = crate::take_quantize(args)?;
        let hir = check(file)?;
        let args: Vec<_> = args.iter().map(|a| jit_arg(a)).collect();
        match novar_mlir::jit_call_with(&hir, func, &args, quant) {
            Ok(JitValue::Int(i)) => println!("{i}"),
            Ok(JitValue::Float(x)) => println!("{x}"),
            Ok(JitValue::Bool(b)) => println!("{b}"),
            Ok(JitValue::Unit) => println!("()"),
            Err(e) => return failed(e),
        }
        Ok(())
    }

    pub fn build(file: &str, flags: &[&str]) -> Result<(), CliError> {
        let (quant, flags) = crate::take_quantize(flags)?;
        let (mut out, mut emit, mut opt) = (None, "exe", 2u8);
        let mut it = flags.iter();
        while let Some(flag) = it.next() {
            match (*flag, it.next()) {
                ("-o", Some(p)) => out = Some(*p),
                ("--emit", Some(k)) => emit = k,
                ("-O", Some(n)) => {
                    opt = n
                        .parse()
                        .or_else(|_| failed(format!("invalid optimisation level `{n}`")))?
                }
                _ => return Err(CliError::Usage),
            }
        }
        let hir = check(file)?;
        let stem = Path::new(file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("out");
        let opt = OptLevel(opt);
        let wrote = |p: &str| eprintln!("wrote {p}");
        match emit {
            "mlir" => {
                let text = novar_mlir::emit_mlir_with(&hir, false, quant)
                    .map_err(|d| report(novar_codegen::BuildError::Diagnostics(d), file))?;
                print!("{text}");
            }
            "llvm-ir" => {
                let ir = llvm_ir_with(&hir, quant).map_err(|e| report(e, file))?;
                print!("{}", optimize_ir(&ir, opt).map_err(|e| report(e, file))?);
            }
            "obj" => {
                let path = out.map_or_else(|| format!("{stem}.o"), str::to_string);
                build_object_with(&hir, opt, quant, Path::new(&path))
                    .map_err(|e| report(e, file))?;
                wrote(&path);
            }
            "exe" => {
                let path = out.map_or_else(|| stem.to_string(), str::to_string);
                build_executable_with(&hir, opt, quant, Path::new(&path))
                    .map_err(|e| report(e, file))?;
                wrote(&path);
            }
            other => return failed(format!("unknown --emit kind `{other}`")),
        }
        Ok(())
    }
}

#[cfg(not(feature = "llvm"))]
mod imp {
    use crate::{CliError, failed};

    const MSG: &str = "this `novar` was built without the native backend; rebuild with `--features llvm` (see scripts/llvm-env.sh)";

    pub fn jit(_: &str, _: &str, _: &[&str]) -> Result<(), CliError> {
        failed(MSG)
    }

    pub fn build(_: &str, _: &[&str]) -> Result<(), CliError> {
        failed(MSG)
    }
}

pub use imp::jit;

use crate::{CliError, failed};

/// `novar build`; `--emit stablehlo|ptx` are handled here, the rest by the native backend.
pub fn build(file: &str, flags: &[&str]) -> Result<(), CliError> {
    let kind = flags
        .iter()
        .position(|f| *f == "--emit")
        .and_then(|i| flags.get(i + 1).copied());
    match kind {
        Some("stablehlo") => stablehlo(file),
        Some("ptx") => ptx(file, flags),
        _ => imp::build(file, flags),
    }
}

fn stablehlo(file: &str) -> Result<(), crate::CliError> {
    let checked = crate::check_project(file)?;
    let out = novar_accel::stablehlo(&checked.hir).map_err(|d| {
        CliError::Failed(
            d.iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })?;
    print!("{}", out.text);
    for (name, why) in &out.skipped {
        eprintln!("skipped `{name}`: {why}");
    }
    if out.emitted.is_empty() {
        return failed("no function could be translated to StableHLO");
    }
    Ok(())
}

#[cfg(feature = "llvm")]
fn ptx(file: &str, flags: &[&str]) -> Result<(), CliError> {
    let sm = flags
        .iter()
        .position(|f| *f == "--sm")
        .and_then(|i| flags.get(i + 1).copied())
        .unwrap_or("sm_80");
    let checked = crate::check_project(file)?;
    let out = novar_accel::ptx(&checked.hir, sm).or_else(failed)?;
    print!("{}", out.text);
    for op in &out.unsupported {
        eprintln!("no PTX kernel for `std::tensor::{op}`");
    }
    Ok(())
}

#[cfg(not(feature = "llvm"))]
fn ptx(_: &str, _: &[&str]) -> Result<(), CliError> {
    failed("PTX generation needs LLVM; rebuild with `--features llvm` (see scripts/llvm-env.sh)")
}
