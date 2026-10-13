//! Native `Tensor<f32|f64, [static dims]>` values and the `std::tensor` operations.
//!
//! A tensor is a pointer to a row-major buffer; its shape lives in the static type, so every
//! dimension is a compile-time constant and indexing needs no metadata. Tensors are immutable
//! values: each operation allocates its result (never freed, like strings and vectors).
//!
//! Arithmetic is done in `f64` and rounded back to the element type, exactly like the
//! interpreter, so both agree bit for bit. With [`Quantization::Int8`] the `f32` matrix product
//! is lowered to a symmetric per-tensor int8 kernel with `i32` accumulation.

use super::*;
use novar_hir::{Dim, Quantization};

/// Element type and static dimensions of a tensor type, when the native backend supports it.
pub(super) fn tensor_info(ty: &Ty) -> Option<(Prim, Vec<usize>)> {
    let Ty::Tensor { elem, shape } = ty else {
        return None;
    };
    let Ty::Prim(p @ (Prim::F32 | Prim::F64)) = &**elem else {
        return None;
    };
    let dims = shape
        .iter()
        .map(|d| match d {
            Dim::Lit(n) => Some(*n as usize),
            Dim::Named(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((*p, dims))
}

type Vals<'c, 'b> = Vec<Value<'c, 'b>>;
type Step<'c, 'b> = Result<(BlockRef<'c, 'b>, Vals<'c, 'b>), Diagnostic>;

impl<'c, 'b, 'p> Emitter<'c, 'b, 'p> {
    fn et(&self, p: Prim) -> Type<'c> {
        if p == Prim::F32 {
            Type::float32(self.cx.ctx)
        } else {
            Type::float64(self.cx.ctx)
        }
    }

    fn f64t(&self) -> Type<'c> {
        Type::float64(self.cx.ctx)
    }

    fn i64c(&self, blk: BlockRef<'c, 'b>, v: i64) -> Value<'c, 'b> {
        self.int_const(blk, self.cx.i(64), v)
    }

    fn float_const(&self, blk: BlockRef<'c, 'b>, ty: Type<'c>, v: f64) -> Value<'c, 'b> {
        self.push(
            blk,
            arith::constant(
                self.cx.ctx,
                FloatAttribute::new(self.cx.ctx, ty, v).into(),
                self.loc,
            ),
        )
    }

    /// Widens an element to `f64` (a no-op for `f64` tensors).
    fn to_f64(&self, blk: BlockRef<'c, 'b>, x: Value<'c, 'b>, p: Prim) -> Value<'c, 'b> {
        if p == Prim::F32 {
            self.push(blk, arith::extf(x, self.f64t(), self.loc))
        } else {
            x
        }
    }

    /// Rounds an `f64` back to the element type.
    fn from_f64(&self, blk: BlockRef<'c, 'b>, x: Value<'c, 'b>, p: Prim) -> Value<'c, 'b> {
        if p == Prim::F32 {
            self.push(blk, arith::truncf(x, self.et(p), self.loc))
        } else {
            x
        }
    }

    fn math(&self, blk: BlockRef<'c, 'b>, name: &str, x: Value<'c, 'b>) -> Value<'c, 'b> {
        self.push(
            blk,
            OperationBuilder::new(name, self.loc)
                .add_operands(&[x])
                .add_results(&[x.r#type()])
                .build()
                .expect("valid math op"),
        )
    }

    fn load(
        &self,
        blk: BlockRef<'c, 'b>,
        ptr: Value<'c, 'b>,
        idx: Value<'c, 'b>,
        ty: Type<'c>,
    ) -> Value<'c, 'b> {
        let at = self.elem_ptr(blk, ptr, idx, ty);
        self.push(
            blk,
            llvm::load(self.cx.ctx, at, ty, self.loc, LoadStoreOptions::new()),
        )
    }

    fn store(
        &self,
        blk: BlockRef<'c, 'b>,
        ptr: Value<'c, 'b>,
        idx: Value<'c, 'b>,
        v: Value<'c, 'b>,
    ) {
        let at = self.elem_ptr(blk, ptr, idx, v.r#type());
        blk.append_operation(llvm::store(
            self.cx.ctx,
            v,
            at,
            self.loc,
            LoadStoreOptions::new(),
        ));
    }

    fn alloc(&self, blk: BlockRef<'c, 'b>, elems: usize, bytes_each: i64) -> Value<'c, 'b> {
        let bytes = self.i64c(blk, (elems.max(1) as i64) * bytes_each);
        self.call_libc(blk, Libc::Malloc, &[bytes], self.cx.ptr())
    }

    fn width(p: Prim) -> i64 {
        if p == Prim::F32 { 4 } else { 8 }
    }

    /// `for idx in 0..count { body }` with loop-carried values. `body` returns the block it ended
    /// in and the next values of the carried variables. Returns the exit block and their final values.
    fn counted_loop(
        &mut self,
        blk: BlockRef<'c, 'b>,
        count: Value<'c, 'b>,
        init: &[Value<'c, 'b>],
        mut body: impl FnMut(
            &mut Self,
            BlockRef<'c, 'b>,
            Value<'c, 'b>,
            &[Value<'c, 'b>],
        ) -> Step<'c, 'b>,
    ) -> Step<'c, 'b> {
        let ctx = self.cx.ctx;
        let i64_ty = self.cx.i(64);
        let carried_tys: Vec<Type<'c>> = init.iter().map(|v| v.r#type()).collect();
        let mut header_tys = vec![i64_ty];
        header_tys.extend(&carried_tys);
        let header = self.new_block(&header_tys);
        let body_blk = self.new_block(&[]);
        let exit = self.new_block(&carried_tys);

        let zero = self.i64c(blk, 0);
        let mut entry_args = vec![zero];
        entry_args.extend(init);
        blk.append_operation(cf::br(&header, &entry_args, self.loc));

        let idx: Value = header.argument(0).expect("index").into();
        let carried: Vals = (1..header_tys.len())
            .map(|i| header.argument(i).expect("carried").into())
            .collect();
        let go = self.push(
            header,
            arith::cmpi(ctx, CmpiPredicate::Ult, idx, count, self.loc),
        );
        header.append_operation(cf::cond_br(
            ctx,
            go,
            &body_blk,
            &exit,
            &[],
            &carried,
            self.loc,
        ));

        let (end, next) = body(self, body_blk, idx, &carried)?;
        let one = self.i64c(end, 1);
        let next_idx = self.push(end, arith::addi(idx, one, self.loc));
        let mut back = vec![next_idx];
        back.extend(next);
        end.append_operation(cf::br(&header, &back, self.loc));

        let out = (0..carried_tys.len())
            .map(|i| exit.argument(i).expect("exit value").into())
            .collect();
        Ok((exit, out))
    }

    fn tensor_of(&self, e: &hir::Expr) -> Result<(Prim, Vec<usize>), Diagnostic> {
        tensor_info(&e.ty).ok_or_else(|| {
            self.unsupported(
                "tensors with symbolic dimensions or non-float elements",
                &e.span,
            )
        })
    }

    /// `a[idx] op b[idx]` in `f64`, rounded to the element type.
    fn fop(
        &self,
        blk: BlockRef<'c, 'b>,
        op: &str,
        x: Value<'c, 'b>,
        y: Value<'c, 'b>,
        p: Prim,
    ) -> Value<'c, 'b> {
        let (x, y) = (self.to_f64(blk, x, p), self.to_f64(blk, y, p));
        let r = match op {
            "add" => arith::addf(x, y, self.loc),
            "sub" => arith::subf(x, y, self.loc),
            "mul" => arith::mulf(x, y, self.loc),
            _ => arith::divf(x, y, self.loc),
        };
        let r = self.push(blk, r);
        self.from_f64(blk, r, p)
    }

    pub(super) fn tensor_call(
        &mut self,
        name: &str,
        e: &hir::Expr,
        args: &[hir::Expr],
        v: &[Value<'c, 'b>],
        blk: BlockRef<'c, 'b>,
    ) -> Out<'c, 'b> {
        let ok = |blk, vals: Vals<'c, 'b>| Ok((blk, Flow::Value(vals)));
        match name {
            "add" | "sub" | "mul" | "div" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let n: usize = dims.iter().product();
                let out = self.alloc(blk, n, Self::width(p));
                let count = self.i64c(blk, n as i64);
                let (a, b, et) = (v[0], v[1], self.et(p));
                let (exit, _) = self.counted_loop(blk, count, &[], |this, bb, i, _| {
                    let (x, y) = (this.load(bb, a, i, et), this.load(bb, b, i, et));
                    let r = this.fop(bb, name, x, y, p);
                    this.store(bb, out, i, r);
                    Ok((bb, vec![]))
                })?;
                ok(exit, vec![out])
            }
            "scale" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let n: usize = dims.iter().product();
                let et = self.et(p);
                // The factor is rounded to the tensor's precision first, as the interpreter does.
                let s = if v[1].r#type() == et {
                    v[1]
                } else if p == Prim::F32 {
                    self.push(blk, arith::truncf(v[1], et, self.loc))
                } else {
                    self.push(blk, arith::extf(v[1], et, self.loc))
                };
                let out = self.alloc(blk, n, Self::width(p));
                let count = self.i64c(blk, n as i64);
                let a = v[0];
                let (exit, _) = self.counted_loop(blk, count, &[], |this, bb, i, _| {
                    let x = this.load(bb, a, i, et);
                    let r = this.fop(bb, "mul", x, s, p);
                    this.store(bb, out, i, r);
                    Ok((bb, vec![]))
                })?;
                ok(exit, vec![out])
            }
            "relu" | "sigmoid" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let n: usize = dims.iter().product();
                let et = self.et(p);
                let out = self.alloc(blk, n, Self::width(p));
                let count = self.i64c(blk, n as i64);
                let a = v[0];
                let (exit, _) = self.counted_loop(blk, count, &[], |this, bb, i, _| {
                    let x = this.load(bb, a, i, et);
                    let r = if name == "relu" {
                        let zero = this.float_const(bb, et, 0.0);
                        let pos = this.push(
                            bb,
                            arith::cmpf(this.cx.ctx, CmpfPredicate::Ogt, x, zero, this.loc),
                        );
                        this.push(bb, arith::select(pos, x, zero, this.loc))
                    } else {
                        let xf = this.to_f64(bb, x, p);
                        let neg = this.push(bb, arith::negf(xf, this.loc));
                        let ex = this.math(bb, "math.exp", neg);
                        let one = this.float_const(bb, this.f64t(), 1.0);
                        let den = this.push(bb, arith::addf(one, ex, this.loc));
                        let sig = this.push(bb, arith::divf(one, den, this.loc));
                        this.from_f64(bb, sig, p)
                    };
                    this.store(bb, out, i, r);
                    Ok((bb, vec![]))
                })?;
                ok(exit, vec![out])
            }
            "sum" | "mean" | "dot" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let n: usize = dims.iter().product();
                let et = self.et(p);
                let count = self.i64c(blk, n as i64);
                let zero = self.float_const(blk, self.f64t(), 0.0);
                let (a, b) = (v[0], v.get(1).copied());
                let (exit, acc) = self.counted_loop(blk, count, &[zero], |this, bb, i, c| {
                    let x = this.to_f64(bb, this.load(bb, a, i, et), p);
                    let term = match b {
                        Some(b) => {
                            let y = this.to_f64(bb, this.load(bb, b, i, et), p);
                            this.push(bb, arith::mulf(x, y, this.loc))
                        }
                        None => x,
                    };
                    let next = this.push(bb, arith::addf(c[0], term, this.loc));
                    Ok((bb, vec![next]))
                })?;
                let mut total = acc[0];
                if name == "mean" {
                    let denom = self.float_const(exit, self.f64t(), n.max(1) as f64);
                    total = self.push(exit, arith::divf(total, denom, self.loc));
                }
                let r = self.from_f64(exit, total, p);
                ok(exit, vec![r])
            }
            "argmax" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let n: usize = dims.iter().product();
                let et = self.et(p);
                let count = self.i64c(blk, n as i64);
                let a = v[0];
                let zero_idx = self.i64c(blk, 0);
                let first = {
                    let z = self.i64c(blk, 0);
                    let x = self.load(blk, a, z, et);
                    self.to_f64(blk, x, p)
                };
                let (exit, best) =
                    self.counted_loop(blk, count, &[zero_idx, first], |this, bb, i, c| {
                        let x = this.to_f64(bb, this.load(bb, a, i, et), p);
                        let gt = this.push(
                            bb,
                            arith::cmpf(this.cx.ctx, CmpfPredicate::Ogt, x, c[1], this.loc),
                        );
                        let val = this.push(bb, arith::select(gt, x, c[1], this.loc));
                        let idx = this.push(bb, arith::select(gt, i, c[0], this.loc));
                        Ok((bb, vec![idx, val]))
                    })?;
                ok(exit, vec![best[0]])
            }
            "matmul" => self.matmul(args, v, blk),
            "transpose" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let [m, n] = dims[..] else {
                    return Err(self.unsupported("transpose of non-matrices", &e.span));
                };
                let et = self.et(p);
                let out = self.alloc(blk, m * n, Self::width(p));
                let (rows, cols) = (self.i64c(blk, m as i64), self.i64c(blk, n as i64));
                let a = v[0];
                let (exit, _) = self.counted_loop(blk, rows, &[], |this, bb, i, _| {
                    let (exit, _) = this.counted_loop(bb, cols, &[], |this, b2, j, _| {
                        let cols = this.i64c(b2, n as i64);
                        let rows = this.i64c(b2, m as i64);
                        let src_row = this.push(b2, arith::muli(i, cols, this.loc));
                        let src = this.push(b2, arith::addi(src_row, j, this.loc));
                        let dst_row = this.push(b2, arith::muli(j, rows, this.loc));
                        let dst = this.push(b2, arith::addi(dst_row, i, this.loc));
                        let x = this.load(b2, a, src, et);
                        this.store(b2, out, dst, x);
                        Ok((b2, vec![]))
                    })?;
                    Ok((exit, vec![]))
                })?;
                ok(exit, vec![out])
            }
            "zeros" => {
                let (p, dims) = self.tensor_of(e)?;
                let n: usize = dims.iter().product();
                let out = self.alloc(blk, n, Self::width(p));
                let et = self.et(p);
                let count = self.i64c(blk, n as i64);
                let (exit, _) = self.counted_loop(blk, count, &[], |this, bb, i, _| {
                    let z = this.float_const(bb, et, 0.0);
                    this.store(bb, out, i, z);
                    Ok((bb, vec![]))
                })?;
                ok(exit, vec![out])
            }
            "reshape" => ok(blk, vec![v[0]]),
            "from_vec" => {
                let (p, dims) = self.tensor_of(e)?;
                let Ty::Vec(elem) = &args[0].ty else {
                    return Err(self.unsupported("from_vec of non-vectors", &e.span));
                };
                let Ty::Prim(src_p @ (Prim::F32 | Prim::F64)) = **elem else {
                    return Err(
                        self.unsupported("converting vectors of non-float elements", &e.span)
                    );
                };
                let n: usize = dims.iter().product();
                // The vector must have exactly the tensor's number of elements.
                let want = self.i64c(blk, n as i64);
                let same = self.push(
                    blk,
                    arith::cmpi(self.cx.ctx, CmpiPredicate::Eq, v[1], want, self.loc),
                );
                let ok_blk = self.new_block(&[]);
                let bad = self.new_block(&[]);
                blk.append_operation(cf::cond_br(
                    self.cx.ctx,
                    same,
                    &ok_blk,
                    &bad,
                    &[],
                    &[],
                    self.loc,
                ));
                self.trap(bad, "from_vec: wrong number of elements\n");
                let out = self.alloc(ok_blk, n, Self::width(p));
                let count = self.i64c(ok_blk, n as i64);
                let (src_ty, dst_ty) = (self.et(src_p), self.et(p));
                let src = v[0];
                // Copies element by element, converting between f32 and f64 when they differ.
                let (exit, _) = self.counted_loop(ok_blk, count, &[], |this, bb, i, _| {
                    let x = this.load(bb, src, i, src_ty);
                    let x = match (src_p, p) {
                        (Prim::F64, Prim::F32) => this.push(bb, arith::truncf(x, dst_ty, this.loc)),
                        (Prim::F32, Prim::F64) => this.push(bb, arith::extf(x, dst_ty, this.loc)),
                        _ => x,
                    };
                    this.store(bb, out, i, x);
                    Ok((bb, vec![]))
                })?;
                ok(exit, vec![out])
            }
            "to_vec" => {
                let (p, dims) = self.tensor_of(&args[0])?;
                let n: usize = dims.iter().product();
                let out = self.alloc(blk, n, Self::width(p));
                let bytes = self.i64c(blk, n as i64 * Self::width(p));
                self.call_libc(blk, Libc::Memcpy, &[out, v[0], bytes], self.cx.ptr());
                let len = self.i64c(blk, n as i64);
                ok(blk, vec![out, len])
            }
            other => Err(self.unsupported(&format!("`std::tensor::{other}`"), &e.span)),
        }
    }

    fn matmul(
        &mut self,
        args: &[hir::Expr],
        v: &[Value<'c, 'b>],
        blk: BlockRef<'c, 'b>,
    ) -> Out<'c, 'b> {
        let (p, ad) = self.tensor_of(&args[0])?;
        let (_, bd) = self.tensor_of(&args[1])?;
        let (m, k) = match ad[..] {
            [m, k] => (m, k),
            _ => return Err(self.unsupported("matmul of a non-matrix", &args[0].span)),
        };
        let (n, vector) = match bd[..] {
            [_, n] => (n, false),
            [_] => (1, true),
            _ => return Err(self.unsupported("matmul with a rank-3 operand", &args[1].span)),
        };
        let _ = vector;
        let (a, b) = (v[0], v[1]);
        let quantized = p == Prim::F32 && self.cx.quant == Quantization::Int8;
        if quantized {
            return self.qmatmul(a, b, (m, k, n), blk);
        }
        let et = self.et(p);
        let out = self.alloc(blk, m * n, Self::width(p));
        let (mc, nc, kc) = (
            self.i64c(blk, m as i64),
            self.i64c(blk, n as i64),
            self.i64c(blk, k as i64),
        );
        let (exit, _) = self.counted_loop(blk, mc, &[], |this, bi, i, _| {
            let (exit, _) = this.counted_loop(bi, nc, &[], |this, bj, j, _| {
                let zero = this.float_const(bj, this.f64t(), 0.0);
                let kc = this.i64c(bj, k as i64);
                let (e2, acc) = this.counted_loop(bj, kc, &[zero], |this, bp, pp, c| {
                    let kk = this.i64c(bp, k as i64);
                    let nn = this.i64c(bp, n as i64);
                    let arow = this.push(bp, arith::muli(i, kk, this.loc));
                    let aidx = this.push(bp, arith::addi(arow, pp, this.loc));
                    let brow = this.push(bp, arith::muli(pp, nn, this.loc));
                    let bidx = this.push(bp, arith::addi(brow, j, this.loc));
                    let x = this.to_f64(bp, this.load(bp, a, aidx, et), p);
                    let y = this.to_f64(bp, this.load(bp, b, bidx, et), p);
                    let prod = this.push(bp, arith::mulf(x, y, this.loc));
                    let next = this.push(bp, arith::addf(c[0], prod, this.loc));
                    Ok((bp, vec![next]))
                })?;
                let nn = this.i64c(e2, n as i64);
                let orow = this.push(e2, arith::muli(i, nn, this.loc));
                let oidx = this.push(e2, arith::addi(orow, j, this.loc));
                let r = this.from_f64(e2, acc[0], p);
                this.store(e2, out, oidx, r);
                Ok((e2, vec![]))
            })?;
            Ok((exit, vec![]))
        })?;
        let _ = kc;
        Ok((exit, Flow::Value(vec![out])))
    }

    /// `max |x| / 127` of the `count` floats at `ptr`, or 1 for all zeros.
    fn int8_scale(
        &mut self,
        blk: BlockRef<'c, 'b>,
        ptr: Value<'c, 'b>,
        count: usize,
    ) -> Result<(BlockRef<'c, 'b>, Value<'c, 'b>), Diagnostic> {
        let f32t = Type::float32(self.cx.ctx);
        let zero = self.float_const(blk, f32t, 0.0);
        let n = self.i64c(blk, count as i64);
        let (exit, amax) = self.counted_loop(blk, n, &[zero], |this, bb, i, c| {
            let x = this.load(bb, ptr, i, f32t);
            let ax = this.math(bb, "math.absf", x);
            let gt = this.push(
                bb,
                arith::cmpf(this.cx.ctx, CmpfPredicate::Ogt, ax, c[0], this.loc),
            );
            let m = this.push(bb, arith::select(gt, ax, c[0], this.loc));
            Ok((bb, vec![m]))
        })?;
        let zero = self.float_const(exit, f32t, 0.0);
        let one = self.float_const(exit, f32t, 1.0);
        let is_zero = self.push(
            exit,
            arith::cmpf(self.cx.ctx, CmpfPredicate::Oeq, amax[0], zero, self.loc),
        );
        let c127 = self.float_const(exit, f32t, 127.0);
        let scaled = self.push(exit, arith::divf(amax[0], c127, self.loc));
        let scale = self.push(exit, arith::select(is_zero, one, scaled, self.loc));
        Ok((exit, scale))
    }

    /// `round(x / scale)` clamped to ±127, stored as `i8`.
    fn quantize_buffer(
        &mut self,
        blk: BlockRef<'c, 'b>,
        src: Value<'c, 'b>,
        count: usize,
        scale: Value<'c, 'b>,
    ) -> Result<(BlockRef<'c, 'b>, Value<'c, 'b>), Diagnostic> {
        let (f32t, i8t) = (Type::float32(self.cx.ctx), self.cx.i(8));
        let dst = self.alloc(blk, count, 1);
        let n = self.i64c(blk, count as i64);
        let (exit, _) = self.counted_loop(blk, n, &[], |this, bb, i, _| {
            let x = this.load(bb, src, i, f32t);
            let q = this.push(bb, arith::divf(x, scale, this.loc));
            let r = this.math(bb, "math.round", q);
            let lo = this.float_const(bb, f32t, -127.0);
            let hi = this.float_const(bb, f32t, 127.0);
            let below = this.push(
                bb,
                arith::cmpf(this.cx.ctx, CmpfPredicate::Olt, r, lo, this.loc),
            );
            let r = this.push(bb, arith::select(below, lo, r, this.loc));
            let above = this.push(
                bb,
                arith::cmpf(this.cx.ctx, CmpfPredicate::Ogt, r, hi, this.loc),
            );
            let r = this.push(bb, arith::select(above, hi, r, this.loc));
            let qi = this.push(bb, arith::fptosi(r, i8t, this.loc));
            this.store(bb, dst, i, qi);
            Ok((bb, vec![]))
        })?;
        Ok((exit, dst))
    }

    /// The int8 kernel: quantize both operands, multiply with `i32` accumulation, rescale.
    fn qmatmul(
        &mut self,
        a: Value<'c, 'b>,
        b: Value<'c, 'b>,
        (m, k, n): (usize, usize, usize),
        blk: BlockRef<'c, 'b>,
    ) -> Out<'c, 'b> {
        let (f32t, i8t, i32t) = (Type::float32(self.cx.ctx), self.cx.i(8), self.cx.i(32));
        let (blk, sa) = self.int8_scale(blk, a, m * k)?;
        let (blk, sb) = self.int8_scale(blk, b, k * n)?;
        let (blk, qa) = self.quantize_buffer(blk, a, m * k, sa)?;
        let (blk, qb) = self.quantize_buffer(blk, b, k * n, sb)?;
        let scale = self.push(blk, arith::mulf(sa, sb, self.loc));
        let out = self.alloc(blk, m * n, 4);
        let (mc, nc) = (self.i64c(blk, m as i64), self.i64c(blk, n as i64));
        let (exit, _) = self.counted_loop(blk, mc, &[], |this, bi, i, _| {
            let (exit, _) = this.counted_loop(bi, nc, &[], |this, bj, j, _| {
                let zero = this.int_const(bj, i32t, 0);
                let kc = this.i64c(bj, k as i64);
                let (e2, acc) = this.counted_loop(bj, kc, &[zero], |this, bp, pp, c| {
                    let kk = this.i64c(bp, k as i64);
                    let nn = this.i64c(bp, n as i64);
                    let arow = this.push(bp, arith::muli(i, kk, this.loc));
                    let aidx = this.push(bp, arith::addi(arow, pp, this.loc));
                    let brow = this.push(bp, arith::muli(pp, nn, this.loc));
                    let bidx = this.push(bp, arith::addi(brow, j, this.loc));
                    let x = this.load(bp, qa, aidx, i8t);
                    let y = this.load(bp, qb, bidx, i8t);
                    let (x, y) = (
                        this.push(bp, arith::extsi(x, i32t, this.loc)),
                        this.push(bp, arith::extsi(y, i32t, this.loc)),
                    );
                    let prod = this.push(bp, arith::muli(x, y, this.loc));
                    let next = this.push(bp, arith::addi(c[0], prod, this.loc));
                    Ok((bp, vec![next]))
                })?;
                let nn = this.i64c(e2, n as i64);
                let orow = this.push(e2, arith::muli(i, nn, this.loc));
                let oidx = this.push(e2, arith::addi(orow, j, this.loc));
                let f = this.push(e2, arith::sitofp(acc[0], f32t, this.loc));
                let r = this.push(e2, arith::mulf(f, scale, this.loc));
                this.store(e2, out, oidx, r);
                Ok((e2, vec![]))
            })?;
            Ok((exit, vec![]))
        })?;
        Ok((exit, Flow::Value(vec![out])))
    }

    /// `t[i]` on a tensor: an element for vectors, otherwise a view of one row of the buffer.
    pub(super) fn tensor_index(
        &mut self,
        e: &hir::Expr,
        base: &hir::Expr,
        index: &hir::Expr,
        blk: BlockRef<'c, 'b>,
        env: &mut Env<'c, 'b>,
    ) -> Out<'c, 'b> {
        let ctx = self.cx.ctx;
        let (p, dims) = self.tensor_of(base)?;
        let (cur, bf) = self.expr(base, blk, env)?;
        let Flow::Value(t) = bf else {
            return Ok((cur, bf));
        };
        let (cur, xf) = self.expr(index, cur, env)?;
        let Flow::Value(x) = xf else {
            return Ok((cur, xf));
        };
        let idx = self.widen_index(cur, x[0], &index.ty);
        let len = self.i64c(cur, dims[0] as i64);
        let in_bounds = self.push(
            cur,
            arith::cmpi(ctx, CmpiPredicate::Ult, idx, len, self.loc),
        );
        let ok_blk = self.new_block(&[]);
        let fail = self.new_block(&[]);
        cur.append_operation(cf::cond_br(
            ctx,
            in_bounds,
            &ok_blk,
            &fail,
            &[],
            &[],
            self.loc,
        ));
        self.trap(fail, "index out of bounds\n");

        let et = self.et(p);
        let stride: usize = dims[1..].iter().product();
        let result = if dims.len() == 1 {
            self.load(ok_blk, t[0], idx, et)
        } else {
            let row = self.i64c(ok_blk, stride as i64);
            let off = self.push(ok_blk, arith::muli(idx, row, self.loc));
            self.elem_ptr(ok_blk, t[0], off, et)
        };
        let _ = e;
        Ok((ok_blk, Flow::Value(vec![result])))
    }
}
