//! The typed, name-resolved High-Level IR produced by the checker.

use crate::ty::Ty;
pub use novar_ast::{BinOp, Lit, UnOp};

pub type Span = std::ops::Range<usize>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    /// Fully qualified names brought in by `import`.
    pub imports: Vec<Vec<String>>,
    pub structs: Vec<StructDef>,
    pub globals: Vec<Global>,
    pub fns: Vec<FnDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub is_pub: bool,
    pub fields: Vec<(String, Ty)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Global {
    pub name: String,
    pub ty: Ty,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub id: LocalId,
    pub name: String,
    pub ty: Ty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

/// Validated `@endpoint(...)` metadata; input for the server generator.
#[derive(Debug, Clone, PartialEq)]
pub struct Endpoint {
    pub path: String,
    pub method: HttpMethod,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnDef {
    pub name: String,
    pub is_pub: bool,
    pub params: Vec<Local>,
    pub ret: Ty,
    pub body: Block,
    pub endpoint: Option<Endpoint>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub ty: Ty,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let { local: Local, init: Expr },
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Lit(Lit),
    Local(LocalId),
    Global(String),
    Fn(String),
    /// Path into an imported / standard-library item the checker cannot see into.
    Extern(Vec<String>),
    /// Pipelines are desugared: `a |> f(b)` becomes `f(a, b)`.
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Field {
        base: Box<Expr>,
        name: String,
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
    Closure {
        params: Vec<Local>,
        body: Box<Expr>,
    },
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    If {
        cond: Box<Expr>,
        then: Block,
        otherwise: Option<Box<Expr>>,
    },
    Block(Block),
    Return(Option<Box<Expr>>),
    FString(Vec<FPart>),
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FPart {
    Text(String),
    Expr(Expr),
}
