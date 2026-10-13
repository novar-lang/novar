//! Semantic types.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Prim {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Str,
}

impl Prim {
    pub fn from_name(name: &str) -> Option<Prim> {
        Some(match name {
            "i8" => Prim::I8,
            "i16" => Prim::I16,
            "i32" => Prim::I32,
            "i64" => Prim::I64,
            "u8" => Prim::U8,
            "u16" => Prim::U16,
            "u32" => Prim::U32,
            "u64" => Prim::U64,
            "f32" => Prim::F32,
            "f64" => Prim::F64,
            "bool" => Prim::Bool,
            "String" => Prim::Str,
            _ => return None,
        })
    }

    pub fn is_int(self) -> bool {
        matches!(
            self,
            Prim::I8
                | Prim::I16
                | Prim::I32
                | Prim::I64
                | Prim::U8
                | Prim::U16
                | Prim::U32
                | Prim::U64
        )
    }

    pub fn is_float(self) -> bool {
        matches!(self, Prim::F32 | Prim::F64)
    }

    pub fn is_numeric(self) -> bool {
        self.is_int() || self.is_float()
    }

    pub fn name(self) -> &'static str {
        match self {
            Prim::I8 => "i8",
            Prim::I16 => "i16",
            Prim::I32 => "i32",
            Prim::I64 => "i64",
            Prim::U8 => "u8",
            Prim::U16 => "u16",
            Prim::U32 => "u32",
            Prim::U64 => "u64",
            Prim::F32 => "f32",
            Prim::F64 => "f64",
            Prim::Bool => "bool",
            Prim::Str => "String",
        }
    }
}

/// One dimension of a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Lit(u64),
    /// A symbolic dimension (e.g. `N`); compatible with any size.
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Ty {
    Unit,
    Prim(Prim),
    /// A user-declared struct.
    Struct(String),
    /// An imported type whose definition is not known to the checker.
    Named(String),
    Vec(Box<Ty>),
    Tensor {
        elem: Box<Ty>,
        shape: Vec<Dim>,
    },
    Result(Box<Ty>, Box<Ty>),
    Fn(Vec<Ty>, Box<Ty>),
    /// Type of expressions that never produce a value (`return`).
    Never,
    /// Type of an unsuffixed integer literal that is still waiting to be pinned down by its
    /// uses; defaults to `i32`. Only exists while a function is being checked.
    IntVar(u32),
    /// Like `IntVar`, for float literals; defaults to `f64`.
    FloatVar(u32),
    /// Not known (opaque std items) or already reported as an error; compatible with everything.
    #[default]
    Unknown,
}

impl Ty {
    pub fn str() -> Ty {
        Ty::Prim(Prim::Str)
    }

    pub fn bool() -> Ty {
        Ty::Prim(Prim::Bool)
    }

    /// Whether a value of type `actual` can be used where `expected` is required.
    pub fn accepts(&self, actual: &Ty) -> bool {
        use Ty::*;
        match (self, actual) {
            (Unknown, _) | (_, Unknown) | (_, Never) | (Never, _) => true,
            (IntVar(_), IntVar(_)) | (FloatVar(_), FloatVar(_)) => true,
            (IntVar(_), Prim(p)) | (Prim(p), IntVar(_)) => p.is_int(),
            (FloatVar(_), Prim(p)) | (Prim(p), FloatVar(_)) => p.is_float(),
            (Unit, Unit) => true,
            (Prim(a), Prim(b)) => a == b,
            (Struct(a), Struct(b)) | (Named(a), Named(b)) => a == b,
            (Vec(a), Vec(b)) => a.accepts(b),
            (
                Tensor {
                    elem: e1,
                    shape: s1,
                },
                Tensor {
                    elem: e2,
                    shape: s2,
                },
            ) => {
                e1.accepts(e2)
                    && s1.len() == s2.len()
                    && s1.iter().zip(s2).all(|(a, b)| match (a, b) {
                        (Dim::Lit(x), Dim::Lit(y)) => x == y,
                        _ => true,
                    })
            }
            (Result(a, b), Result(c, d)) => a.accepts(c) && b.accepts(d),
            (Fn(p1, r1), Fn(p2, r2)) => {
                p1.len() == p2.len()
                    && p1.iter().zip(p2).all(|(a, b)| a.accepts(b))
                    && r1.accepts(r2)
            }
            _ => false,
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Lit(n) => write!(f, "{n}"),
            Dim::Named(n) => f.write_str(n),
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Unit => f.write_str("()"),
            Ty::Prim(p) => f.write_str(p.name()),
            Ty::Struct(n) | Ty::Named(n) => f.write_str(n),
            Ty::Vec(t) => write!(f, "Vec<{t}>"),
            Ty::Tensor { elem, shape } => {
                let dims: Vec<_> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "Tensor<{elem}, [{}]>", dims.join(", "))
            }
            Ty::Result(a, b) => write!(f, "Result<{a}, {b}>"),
            Ty::Fn(params, ret) => {
                let ps: Vec<_> = params.iter().map(|p| p.to_string()).collect();
                write!(f, "fn({}) -> {ret}", ps.join(", "))
            }
            Ty::IntVar(_) => f.write_str("{integer}"),
            Ty::FloatVar(_) => f.write_str("{float}"),
            Ty::Never => f.write_str("!"),
            Ty::Unknown => f.write_str("_"),
        }
    }
}
