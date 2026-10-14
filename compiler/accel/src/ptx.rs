//! PTX kernels for the `std.tensor` operations, built as LLVM IR for the `nvptx64` target and
//! compiled by LLVM's NVPTX backend. One thread handles one output element (or one row for
//! `softmax`); reductions accumulate with a global atomic add. The kernels are meant to be
//! launched through the CUDA driver API (`cuModuleLoadData` + `cuLaunchKernel`).

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};

/// PTX text plus the entry points it contains.
#[derive(Debug)]
pub struct Ptx {
    pub text: String,
    pub kernels: Vec<String>,
    /// Operations used by the program that have no kernel.
    pub unsupported: Vec<String>,
}

const PRELUDE: &str = r#"target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare float @llvm.nvvm.ex2.approx.f(float)
declare float @llvm.maxnum.f32(float, float)
"#;

/// `%i` = global thread index, then a bounds check against `%n`.
fn header(name: &str, params: &str, n: &str) -> String {
    format!(
        "define ptx_kernel void @{name}({params}) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cta = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %dim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %cta, %dim
  %i = add i32 %base, %tid
  %ok = icmp slt i32 %i, {n}
  br i1 %ok, label %body, label %done
"
    )
}

const FOOTER: &str = "done:\n  ret void\n}\n";

fn binary(name: &str, op: &str) -> String {
    header(
        &format!("{name}_f32"),
        "ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %n",
        "%n",
    ) + &format!(
        "body:
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %i
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %i
  %po = getelementptr float, ptr addrspace(1) %out, i32 %i
  %x = load float, ptr addrspace(1) %pa
  %y = load float, ptr addrspace(1) %pb
  %r = {op} float %x, %y
  store float %r, ptr addrspace(1) %po
  br label %done
"
    ) + FOOTER
}

fn unary(name: &str, extra: &str, params: &str, compute: &str) -> String {
    header(
        &format!("{name}_f32"),
        &format!("ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n{extra}{params}"),
        "%n",
    ) + &format!(
        "body:
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %i
  %po = getelementptr float, ptr addrspace(1) %out, i32 %i
  %x = load float, ptr addrspace(1) %pa
{compute}
  store float %r, ptr addrspace(1) %po
  br label %done
"
    ) + FOOTER
}

fn matmul() -> String {
    // out[row, col] = sum_k a[row, k] * b[k, col]; one thread per output element.
    header(
        "matmul_f32",
        "ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %m, i32 %k, i32 %n",
        "%total",
    )
    .replace("  %ok = ", "  %total = mul i32 %m, %n\n  %ok = ")
        + "body:
  %row = sdiv i32 %i, %n
  %col = srem i32 %i, %n
  %arow = mul i32 %row, %k
  br label %loop
loop:
  %j = phi i32 [ 0, %body ], [ %jn, %loop ]
  %acc = phi float [ 0.0, %body ], [ %accn, %loop ]
  %ai = add i32 %arow, %j
  %bj = mul i32 %j, %n
  %bi = add i32 %bj, %col
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %ai
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %bi
  %x = load float, ptr addrspace(1) %pa
  %y = load float, ptr addrspace(1) %pb
  %prod = fmul float %x, %y
  %accn = fadd float %acc, %prod
  %jn = add i32 %j, 1
  %more = icmp slt i32 %jn, %k
  br i1 %more, label %loop, label %store
store:
  %po = getelementptr float, ptr addrspace(1) %out, i32 %i
  store float %accn, ptr addrspace(1) %po
  br label %done
" + FOOTER
}

fn softmax() -> String {
    // One thread per row: max, exp(x - max) and the normalising sum.
    header(
        "softmax_f32",
        "ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %rows, i32 %cols",
        "%rows",
    ) + "body:
  %off = mul i32 %i, %cols
  br label %maxloop
maxloop:
  %mj = phi i32 [ 0, %body ], [ %mjn, %maxloop ]
  %mx = phi float [ 0xFFF0000000000000, %body ], [ %mxn, %maxloop ]
  %mi = add i32 %off, %mj
  %mp = getelementptr float, ptr addrspace(1) %a, i32 %mi
  %mv = load float, ptr addrspace(1) %mp
  %mxn = call float @llvm.maxnum.f32(float %mx, float %mv)
  %mjn = add i32 %mj, 1
  %mmore = icmp slt i32 %mjn, %cols
  br i1 %mmore, label %maxloop, label %sumloop
sumloop:
  %sj = phi i32 [ 0, %maxloop ], [ %sjn, %sumloop ]
  %sa = phi float [ 0.0, %maxloop ], [ %san, %sumloop ]
  %si = add i32 %off, %sj
  %sp = getelementptr float, ptr addrspace(1) %a, i32 %si
  %sv = load float, ptr addrspace(1) %sp
  %sd = fsub float %sv, %mxn
  %sl = fmul float %sd, 0x3FF7154760000000
  %se = call float @llvm.nvvm.ex2.approx.f(float %sl)
  %op = getelementptr float, ptr addrspace(1) %out, i32 %si
  store float %se, ptr addrspace(1) %op
  %san = fadd float %sa, %se
  %sjn = add i32 %sj, 1
  %smore = icmp slt i32 %sjn, %cols
  br i1 %smore, label %sumloop, label %normloop
normloop:
  %nj = phi i32 [ 0, %sumloop ], [ %njn, %normloop ]
  %ni = add i32 %off, %nj
  %np = getelementptr float, ptr addrspace(1) %out, i32 %ni
  %nv = load float, ptr addrspace(1) %np
  %nr = fdiv float %nv, %san
  store float %nr, ptr addrspace(1) %np
  %njn = add i32 %nj, 1
  %nmore = icmp slt i32 %njn, %cols
  br i1 %nmore, label %normloop, label %done
" + FOOTER
}

/// `out` must be zeroed by the host before the launch.
fn reduce(name: &str, product: bool) -> String {
    let (params, loads, value) = if product {
        (
            "ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %n",
            "  %pb = getelementptr float, ptr addrspace(1) %b, i32 %i\n  %y = load float, ptr addrspace(1) %pb\n",
            "fmul float %x, %y",
        )
    } else {
        (
            "ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n",
            "",
            "fadd float %x, 0.0",
        )
    };
    header(&format!("{name}_f32"), params, "%n")
        + &format!(
            "body:
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %i
  %x = load float, ptr addrspace(1) %pa
{loads}  %v = {value}
  %old = atomicrmw fadd ptr addrspace(1) %out, float %v seq_cst
  br label %done
"
        )
        + FOOTER
}

fn kernel(op: &str) -> Option<String> {
    Some(match op {
        "add" => binary("add", "fadd"),
        "sub" => binary("sub", "fsub"),
        "mul" => binary("mul", "fmul"),
        "div" => binary("div", "fdiv"),
        "scale" => unary("scale", "", ", float %s", "  %r = fmul float %x, %s"),
        "relu" => unary(
            "relu",
            "",
            "",
            "  %r = call float @llvm.maxnum.f32(float %x, float 0.0)",
        ),
        "sigmoid" => unary(
            "sigmoid",
            "",
            "",
            "  %neg = fneg float %x
  %nl = fmul float %neg, 0x3FF7154760000000
  %e = call float @llvm.nvvm.ex2.approx.f(float %nl)
  %den = fadd float %e, 1.0
  %r = fdiv float 1.0, %den",
        ),
        "matmul" => matmul(),
        "softmax" => softmax(),
        "sum" | "mean" => reduce("sum", false),
        "dot" => reduce("dot", true),
        _ => return None,
    })
}

/// Kernel entry points that serve a `std.tensor` operation.
fn kernel_for(op: &str) -> Option<&'static str> {
    Some(match op {
        "add" => "add_f32",
        "sub" => "sub_f32",
        "mul" => "mul_f32",
        "div" => "div_f32",
        "scale" => "scale_f32",
        "relu" => "relu_f32",
        "sigmoid" => "sigmoid_f32",
        "matmul" => "matmul_f32",
        "softmax" => "softmax_f32",
        "sum" | "mean" => "sum_f32",
        "dot" => "dot_f32",
        _ => return None,
    })
}

/// The `std.tensor` operations in `program` that [`ptx`] has a kernel for.
pub fn used_ops(program: &telvax_hir::hir::Program) -> Vec<String> {
    crate::tensor_ops(program)
        .into_iter()
        .filter(|o| kernel_for(o).is_some())
        .collect()
}

/// Compiles the kernels for the operations `program` uses to PTX for compute capability `sm`
/// (for example `"sm_80"`).
pub fn ptx(program: &telvax_hir::hir::Program, sm: &str) -> Result<Ptx, String> {
    let all = crate::tensor_ops(program);
    let unsupported: Vec<String> = all
        .iter()
        .filter(|o| {
            kernel_for(o).is_none()
                && !matches!(
                    o.as_str(),
                    "zeros" | "from_vec" | "to_vec" | "reshape" | "transpose" | "argmax"
                )
        })
        .cloned()
        .collect();
    let mut seen = Vec::new();
    let mut ir = String::from(PRELUDE);
    for op in &all {
        if let (Some(name), Some(text)) = (kernel_for(op), kernel(op))
            && !seen.contains(&name)
        {
            seen.push(name);
            ir.push_str(&text);
        }
    }
    if seen.is_empty() {
        return Err("the program uses no tensor operation that has a PTX kernel".into());
    }
    let text = compile(&ir, sm)?;
    Ok(Ptx {
        text,
        kernels: seen.into_iter().map(String::from).collect(),
        unsupported,
    })
}

fn compile(ir: &str, sm: &str) -> Result<String, String> {
    Target::initialize_nvptx(&InitializationConfig::default());
    let triple = TargetTriple::create("nvptx64-nvidia-cuda");
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let tm = target
        .create_target_machine(
            &triple,
            sm,
            "",
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("cannot create an NVPTX target machine")?;
    let context = Context::create();
    let buffer = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), "telvax_ptx");
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|e| format!("invalid kernel IR: {e}"))?;
    module.verify().map_err(|e| e.to_string())?;
    let out = tm
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(out.as_slice())
        .trim_end_matches('\0')
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_verifies_and_lowers_to_ptx() {
        let program = {
            let src = "
def f(a: Tensor[float32, [4, 4]], b: Tensor[float32, [4, 4]]) -> Tensor[float32, [4, 4]]:
    return std.tensor.softmax(std.tensor.sigmoid(std.tensor.relu(std.tensor.scale(
        std.tensor.matmul(std.tensor.sub(std.tensor.div(std.tensor.mul(std.tensor.add(a, b), b), b), a), b), 0.5))))

def s(a: Tensor[float32, [16]], b: Tensor[float32, [16]]) -> float32:
    return std.tensor.sum(a) + std.tensor.mean(a) + std.tensor.dot(a, b)
            ";
            let (ast, errs) = telvax_parser::parse(src);
            assert!(errs.is_empty(), "{errs:?}");
            let (hir, d) = telvax_hir::lower(&ast.unwrap());
            assert!(
                d.is_empty(),
                "{:?}",
                d.iter().map(|d| &d.message).collect::<Vec<_>>()
            );
            hir
        };
        let out = ptx(&program, "sm_80").unwrap();
        assert!(out.text.contains(".target sm_80"), "{}", out.text);
        assert!(out.text.contains(".address_size 64"));
        for k in &out.kernels {
            assert!(
                out.text.contains(&format!(".visible .entry {k}(")),
                "no entry `{k}` in\n{}",
                out.text
            );
        }
        assert_eq!(out.kernels.len(), 11, "{:?}", out.kernels);
        assert!(out.text.contains("ex2.approx") && out.text.contains("atom.global.add.f32"));
    }
}
