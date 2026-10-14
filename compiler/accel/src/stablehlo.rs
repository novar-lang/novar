use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use telvax_diagnostics::Diagnostic;
use telvax_hir::hir::{self, BinOp, ExprKind, Lit, Stmt};
use telvax_hir::{Dim, Prim, Ty};

/// The StableHLO module for a program.
#[derive(Debug)]
pub struct StableHlo {
    pub text: String,
    /// Functions that were translated.
    pub emitted: Vec<String>,
    /// Functions left out, each with the reason.
    pub skipped: Vec<(String, String)>,
}

/// Translates every function made of tensor and scalar `f32` arithmetic (straight-line `let`s,
/// `std::tensor` calls, `+ - * /` and calls to other translatable functions). Anything else is
/// reported in [`StableHlo::skipped`].
pub fn stablehlo(program: &hir::Program) -> Result<StableHlo, Vec<Diagnostic>> {
    let mut done: HashMap<String, String> = HashMap::new();
    let mut failed: HashMap<String, String> = HashMap::new();
    loop {
        let mut progress = false;
        for f in &program.fns {
            if done.contains_key(&f.name) {
                continue;
            }
            let known: HashSet<&str> = done.keys().map(String::as_str).collect();
            match Emit::function(f, &known) {
                Ok(text) => {
                    done.insert(f.name.clone(), text);
                    failed.remove(&f.name);
                    progress = true;
                }
                Err(why) => {
                    failed.insert(f.name.clone(), why);
                }
            }
        }
        if !progress {
            break;
        }
    }
    let mut text = String::from("module @telvax {\n");
    let mut emitted = Vec::new();
    for f in &program.fns {
        if let Some(t) = done.get(&f.name) {
            text.push_str(t);
            emitted.push(f.name.clone());
        }
    }
    text.push_str("}\n");
    let skipped = program
        .fns
        .iter()
        .filter_map(|f| failed.get(&f.name).map(|w| (f.name.clone(), w.clone())))
        .collect();
    Ok(StableHlo {
        text,
        emitted,
        skipped,
    })
}

fn elem(t: &Ty) -> Option<&'static str> {
    match t {
        Ty::Prim(Prim::F32) | Ty::FloatVar(_) => Some("f32"),
        Ty::Prim(Prim::F64) => Some("f64"),
        _ => None,
    }
}

/// `tensor<2x3xf32>` for tensors, `tensor<f32>` for scalars.
fn mlir_ty(t: &Ty) -> Result<String, String> {
    match t {
        Ty::Tensor { elem: e, shape } => {
            let el = elem(e).ok_or("only f32/f64 tensors are supported")?;
            let mut dims = String::new();
            for d in shape {
                match d {
                    Dim::Lit(n) => write!(dims, "{n}x").unwrap(),
                    Dim::Named(n) => {
                        return Err(format!("symbolic dimension `{n}` has no static shape"));
                    }
                }
            }
            Ok(format!("tensor<{dims}{el}>"))
        }
        other => elem(other)
            .map(|e| format!("tensor<{e}>"))
            .ok_or_else(|| format!("type {other:?} has no StableHLO form")),
    }
}

fn shape(t: &Ty) -> Vec<u64> {
    match t {
        Ty::Tensor { shape, .. } => shape
            .iter()
            .filter_map(|d| if let Dim::Lit(n) = d { Some(*n) } else { None })
            .collect(),
        _ => Vec::new(),
    }
}

struct Emit<'a> {
    known: &'a HashSet<&'a str>,
    body: String,
    next: usize,
    vars: HashMap<u32, (String, Ty)>,
}

type Val = (String, Ty);

impl<'a> Emit<'a> {
    fn function(f: &hir::FnDef, known: &'a HashSet<&'a str>) -> Result<String, String> {
        let mut e = Emit {
            known,
            body: String::new(),
            next: 0,
            vars: HashMap::new(),
        };
        let mut params = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            let name = format!("%arg{i}");
            params.push(format!("{name}: {}", mlir_ty(&p.ty)?));
            e.vars.insert(p.id.0, (name, p.ty.clone()));
        }
        let ret = mlir_ty(&f.ret)?;
        let v = e.block(&f.body)?;
        let mut out = format!(
            "  func.func @{}({}) -> {ret} {{\n",
            f.name,
            params.join(", ")
        );
        out.push_str(&e.body);
        writeln!(out, "    return {} : {ret}", v.0).unwrap();
        out.push_str("  }\n");
        Ok(out)
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    fn line(&mut self, text: String) {
        self.body.push_str("    ");
        self.body.push_str(&text);
        self.body.push('\n');
    }

    fn block(&mut self, b: &hir::Block) -> Result<Val, String> {
        for s in &b.stmts {
            match s {
                Stmt::Let { local, init } => {
                    if local.mutable {
                        return Err("`let mut` has no StableHLO form".into());
                    }
                    let v = self.expr(init)?;
                    self.vars.insert(local.id.0, v);
                }
                Stmt::Expr(_) => return Err("statements with effects are not supported".into()),
            }
        }
        match &b.tail {
            Some(t) => self.expr(t),
            None => Err("a function must end in a value".into()),
        }
    }

    fn constant(&mut self, value: f64, ty: &Ty) -> Result<Val, String> {
        let t = mlir_ty(ty)?;
        let n = self.fresh();
        let lit = if value == f64::NEG_INFINITY {
            "0xFF800000".to_string()
        } else {
            format!("{value:?}")
        };
        self.line(format!("{n} = stablehlo.constant dense<{lit}> : {t}"));
        Ok((n, ty.clone()))
    }

    fn splat(&mut self, scalar: &Val, to: &Ty) -> Result<Val, String> {
        let (s, st) = scalar;
        let n = self.fresh();
        self.line(format!(
            "{n} = stablehlo.broadcast_in_dim {s}, dims = [] : ({}) -> {}",
            mlir_ty(st)?,
            mlir_ty(to)?
        ));
        Ok((n, to.clone()))
    }

    fn expr(&mut self, e: &hir::Expr) -> Result<Val, String> {
        match &e.kind {
            ExprKind::Local(id) => self.vars.get(&id.0).cloned().ok_or("unknown local".into()),
            ExprKind::Lit(Lit::Float(x)) => {
                let ty = if elem(&e.ty).is_some() {
                    e.ty.clone()
                } else {
                    Ty::Prim(Prim::F32)
                };
                self.constant(*x, &ty)
            }
            ExprKind::Lit(Lit::Int(x)) if elem(&e.ty).is_some() => self.constant(*x as f64, &e.ty),
            ExprKind::Block(b) => self.block(b),
            ExprKind::Binary { op, lhs, rhs } => {
                let name = match op {
                    BinOp::Add => "add",
                    BinOp::Sub => "subtract",
                    BinOp::Mul => "multiply",
                    BinOp::Div => "divide",
                    other => return Err(format!("operator {other:?} is not supported")),
                };
                let (a, b) = (self.expr(lhs)?, self.expr(rhs)?);
                self.binary(name, a, b)
            }
            ExprKind::Unary {
                op: hir::UnOp::Neg,
                expr,
            } => {
                let v = self.expr(expr)?;
                let n = self.fresh();
                let t = mlir_ty(&v.1)?;
                self.line(format!("{n} = stablehlo.negate {} : {t}", v.0));
                Ok((n, v.1))
            }
            ExprKind::Call { callee, args } => self.call(callee, args, &e.ty),
            other => Err(format!("{} is not supported", kind_name(other))),
        }
    }

    fn binary(&mut self, name: &str, a: Val, b: Val) -> Result<Val, String> {
        let (a, b) = match (shape(&a.1).is_empty(), shape(&b.1).is_empty()) {
            (false, true) => {
                let b2 = self.splat(&b, &a.1)?;
                (a, b2)
            }
            (true, false) => {
                let a2 = self.splat(&a, &b.1)?;
                (a2, b)
            }
            _ => (a, b),
        };
        let t = mlir_ty(&a.1)?;
        let n = self.fresh();
        self.line(format!("{n} = stablehlo.{name} {}, {} : {t}", a.0, b.0));
        Ok((n, a.1))
    }

    fn reduce(
        &mut self,
        x: &Val,
        dims: &[usize],
        kind: &str,
        init: f64,
        out: &Ty,
    ) -> Result<Val, String> {
        let el = elem(match &x.1 {
            Ty::Tensor { elem, .. } => elem,
            o => o,
        })
        .ok_or("reduce needs a float tensor")?;
        let zero = self.constant(
            init,
            &Ty::Prim(if el == "f32" { Prim::F32 } else { Prim::F64 }),
        )?;
        let n = self.fresh();
        let list = dims
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        self.line(format!(
            "{n} = stablehlo.reduce({} init: {}) applies stablehlo.{kind} across dimensions = [{list}] : ({}, tensor<{el}>) -> {}",
            x.0,
            zero.0,
            mlir_ty(&x.1)?,
            mlir_ty(out)?
        ));
        Ok((n, out.clone()))
    }

    fn call(&mut self, callee: &hir::Expr, args: &[hir::Expr], ret: &Ty) -> Result<Val, String> {
        let mut vals = Vec::new();
        for a in args {
            vals.push(self.expr(a)?);
        }
        match &callee.kind {
            ExprKind::Fn(name) => {
                if !self.known.contains(name.as_str()) {
                    return Err(format!("calls `{name}`, which cannot be translated"));
                }
                let n = self.fresh();
                let ins = vals
                    .iter()
                    .map(|v| mlir_ty(&v.1))
                    .collect::<Result<Vec<_>, _>>()?;
                let names = vals.iter().map(|v| v.0.clone()).collect::<Vec<_>>();
                self.line(format!(
                    "{n} = func.call @{name}({}) : ({}) -> {}",
                    names.join(", "),
                    ins.join(", "),
                    mlir_ty(ret)?
                ));
                Ok((n, ret.clone()))
            }
            ExprKind::Extern(path)
                if path.len() == 3 && path[0] == "std" && path[1] == "tensor" =>
            {
                self.tensor_op(&path[2], vals, ret)
            }
            _ => Err("only direct calls are supported".into()),
        }
    }

    fn tensor_op(&mut self, op: &str, v: Vec<Val>, ret: &Ty) -> Result<Val, String> {
        let first = v.first().cloned().ok_or("missing argument")?;
        match op {
            "add" | "sub" | "mul" | "div" => {
                let name = match op {
                    "add" => "add",
                    "sub" => "subtract",
                    "mul" => "multiply",
                    _ => "divide",
                };
                self.binary(name, first, v[1].clone())
            }
            "scale" => self.binary("multiply", first, v[1].clone()),
            "relu" => {
                let el = Ty::Prim(Prim::F32);
                let z = self.constant(0.0, &el)?;
                let z = self.splat(&z, &first.1)?;
                self.binary("maximum", first, z)
            }
            "sigmoid" => {
                let n = self.fresh();
                let t = mlir_ty(&first.1)?;
                self.line(format!("{n} = stablehlo.logistic {} : {t}", first.0));
                Ok((n, first.1))
            }
            "softmax" => {
                let rank = shape(&first.1).len();
                if rank == 0 {
                    return Err("softmax needs a tensor".into());
                }
                let last = rank - 1;
                let lead: Vec<u64> = shape(&first.1)[..last].to_vec();
                let red_ty = Ty::Tensor {
                    elem: Box::new(Ty::Prim(Prim::F32)),
                    shape: lead.iter().map(|d| Dim::Lit(*d)).collect(),
                };
                let mx = self.reduce(&first, &[last], "maximum", f64::NEG_INFINITY, &red_ty)?;
                let dims = (0..last)
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let t = mlir_ty(&first.1)?;
                let bm = self.fresh();
                self.line(format!(
                    "{bm} = stablehlo.broadcast_in_dim {}, dims = [{dims}] : ({}) -> {t}",
                    mx.0,
                    mlir_ty(&red_ty)?
                ));
                let shifted = self.binary("subtract", first.clone(), (bm, first.1.clone()))?;
                let ex = self.fresh();
                self.line(format!("{ex} = stablehlo.exponential {} : {t}", shifted.0));
                let ex = (ex, first.1.clone());
                let sum = self.reduce(&ex, &[last], "add", 0.0, &red_ty)?;
                let bs = self.fresh();
                self.line(format!(
                    "{bs} = stablehlo.broadcast_in_dim {}, dims = [{dims}] : ({}) -> {t}",
                    sum.0,
                    mlir_ty(&red_ty)?
                ));
                self.binary("divide", ex, (bs, first.1.clone()))
            }
            "sum" | "mean" => {
                let dims: Vec<usize> = (0..shape(&first.1).len()).collect();
                let s = self.reduce(&first, &dims, "add", 0.0, ret)?;
                if op == "sum" {
                    return Ok(s);
                }
                let count: u64 = shape(&first.1).iter().product();
                let c = self.constant(count as f64, ret)?;
                self.binary("divide", s, c)
            }
            "dot" | "matmul" => {
                let b = v.get(1).cloned().ok_or("missing argument")?;
                let (ra, rb) = (shape(&first.1).len(), shape(&b.1).len());
                let n = self.fresh();
                self.line(format!(
                    "{n} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [0] : ({}, {}) -> {}",
                    first.0,
                    b.0,
                    ra - 1,
                    mlir_ty(&first.1)?,
                    mlir_ty(&b.1)?,
                    mlir_ty(ret)?
                ));
                let _ = rb;
                Ok((n, ret.clone()))
            }
            "transpose" => {
                let n = self.fresh();
                self.line(format!(
                    "{n} = stablehlo.transpose {}, dims = [1, 0] : ({}) -> {}",
                    first.0,
                    mlir_ty(&first.1)?,
                    mlir_ty(ret)?
                ));
                Ok((n, ret.clone()))
            }
            "reshape" => {
                let n = self.fresh();
                self.line(format!(
                    "{n} = stablehlo.reshape {} : ({}) -> {}",
                    first.0,
                    mlir_ty(&first.1)?,
                    mlir_ty(ret)?
                ));
                Ok((n, ret.clone()))
            }
            other => Err(format!("`std::tensor::{other}` has no StableHLO form")),
        }
    }
}

fn kind_name(k: &ExprKind) -> &'static str {
    match k {
        ExprKind::If { .. } => "`if`",
        ExprKind::While { .. } | ExprKind::For { .. } => "a loop",
        ExprKind::Assign { .. } => "assignment",
        ExprKind::Index { .. } => "indexing",
        ExprKind::FString(_) => "an f-string",
        ExprKind::Closure { .. } => "a closure",
        _ => "this expression",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(src: &str) -> StableHlo {
        let (ast, errs) = telvax_parser::parse(src);
        assert!(errs.is_empty(), "{errs:?}");
        let (hir, diags) = telvax_hir::lower(&ast.unwrap());
        assert!(
            diags.is_empty(),
            "{:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        stablehlo(&hir).unwrap()
    }

    #[test]
    fn translates_a_small_network() {
        let out = compile(
            "fn layer(x: Tensor<f32, [2, 3]>, w: Tensor<f32, [3, 4]>) -> Tensor<f32, [2, 4]> {
                 std::tensor::softmax(std::tensor::relu(std::tensor::matmul(x, w)))
             }
             fn total(x: Tensor<f32, [2, 4]>) -> f32 { std::tensor::sum(x) }
             fn twice(x: Tensor<f32, [2, 3]>, w: Tensor<f32, [3, 4]>) -> Tensor<f32, [2, 4]> { layer(x, w) }",
        );
        assert_eq!(
            out.emitted,
            ["layer", "total", "twice"],
            "{:?}",
            out.skipped
        );
        for needle in [
            "func.func @layer(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32>",
            "stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]",
            "stablehlo.maximum",
            "dense<0.0> : tensor<f32>",
            "dense<0xFF800000>",
            "stablehlo.exponential",
            "stablehlo.reduce(",
            "func.call @layer(%arg0, %arg1)",
        ] {
            assert!(
                out.text.contains(needle),
                "missing `{needle}` in\n{}",
                out.text
            );
        }
    }

    #[test]
    fn reports_what_it_cannot_translate() {
        let out = compile(
            "fn looped(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } s }
             fn fine(x: Tensor<f32, [4]>) -> f32 { std::tensor::mean(x) }
             fn uses_loop(n: i32) -> i32 { looped(n) }",
        );
        assert_eq!(out.emitted, ["fine"]);
        let names: Vec<_> = out.skipped.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["looped", "uses_loop"]);
    }
}
