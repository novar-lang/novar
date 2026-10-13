//! Abstract syntax tree definitions for Novar.

use std::ops::Range;

pub type Span = Range<usize>;

/// A node together with its byte span in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

pub type Ident = Spanned<String>;
pub type Expr = Spanned<ExprKind>;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Import(Import),
    /// Top-level `let` (global resource setup).
    Let(Let),
    Struct(Struct),
    Fn(Fn),
}

/// `import std::http::{Request, Response};` or `import a::b::C;`
#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    pub path: Vec<Ident>,
    /// Empty for a plain import; names inside `{...}` otherwise.
    pub group: Vec<Ident>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Let {
    pub mutable: bool,
    pub name: Ident,
    pub ty: Option<Type>,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Struct {
    pub is_pub: bool,
    pub name: Ident,
    pub fields: Vec<Field>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: Ident,
    pub ty: Type,
}

/// `@endpoint(path = "/x", method = "POST")`
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: Ident,
    pub args: Vec<(Ident, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fn {
    pub attrs: Vec<Attribute>,
    pub is_pub: bool,
    pub name: Ident,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: Type,
}

/// `Name`, `a::B`, `Vec<T>`, `Tensor<f32, [768]>`
#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub path: Vec<Ident>,
    pub args: Vec<TypeArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeArg {
    Type(Type),
    /// Tensor shape such as `[768]` or `[N, 4]`.
    Shape(Vec<Dim>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Lit(u64),
    Named(Ident),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// Trailing expression without `;`, the block's value.
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let(Let),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Lit(Lit),
    /// `x` or `std::text::trim`
    Path(Vec<Ident>),
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Field {
        base: Box<Expr>,
        name: Ident,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `lhs |> rhs`; the lowest-precedence, left-associative operator.
    Pipe {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Closure {
        params: Vec<Ident>,
        body: Box<Expr>,
    },
    StructLit {
        path: Vec<Ident>,
        fields: Vec<(Ident, Expr)>,
    },
    If {
        cond: Box<Expr>,
        then: Block,
        otherwise: Option<Box<Expr>>,
    },
    Block(Block),
    Return(Option<Box<Expr>>),
    FString(Vec<FPart>),
    /// Placeholder left by parser error recovery.
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FPart {
    Text(String),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(u64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Or,
    And,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}
