//! Native backend commands (`jit`, `build`), available with the `llvm` feature.

#[cfg(feature = "llvm")]
mod imp {
    use crate::{CliError, check, failed};
    use novar_codegen::{OptLevel, build_executable, build_object, llvm_ir, optimize_ir};
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
        let hir = check(file)?;
        let args: Vec<_> = args.iter().map(|a| jit_arg(a)).collect();
        match novar_mlir::jit_call(&hir, func, &args) {
            Ok(JitValue::Int(i)) => println!("{i}"),
            Ok(JitValue::Float(x)) => println!("{x}"),
            Ok(JitValue::Bool(b)) => println!("{b}"),
            Ok(JitValue::Unit) => println!("()"),
            Err(e) => return failed(e),
        }
        Ok(())
    }

    pub fn build(file: &str, flags: &[&str]) -> Result<(), CliError> {
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
                let text = novar_mlir::emit_mlir(&hir, false)
                    .map_err(|d| report(novar_codegen::BuildError::Diagnostics(d), file))?;
                print!("{text}");
            }
            "llvm-ir" => {
                let ir = llvm_ir(&hir).map_err(|e| report(e, file))?;
                print!("{}", optimize_ir(&ir, opt).map_err(|e| report(e, file))?);
            }
            "obj" => {
                let path = out.map_or_else(|| format!("{stem}.o"), str::to_string);
                build_object(&hir, opt, Path::new(&path)).map_err(|e| report(e, file))?;
                wrote(&path);
            }
            "exe" => {
                let path = out.map_or_else(|| stem.to_string(), str::to_string);
                build_executable(&hir, opt, Path::new(&path)).map_err(|e| report(e, file))?;
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

pub use imp::{build, jit};
