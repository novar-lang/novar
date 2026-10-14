//! Zero-copy tokenizer for Telvax source code.

use std::ops::Range;

use logos::Logos;

pub type Span = Range<usize>;

/// Raised for input that matches no token (stray character, unterminated string, ...).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LexError;

#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(error = LexError)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip(r"//[^\n]*", allow_greedy = true))]
pub enum Token<'src> {
    // Keywords
    #[token("import")]
    Import,
    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("pub")]
    Pub,
    #[token("match")]
    Match,
    #[token("impl")]
    Impl,
    #[token("trait")]
    Trait,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("true")]
    True,
    #[token("false")]
    False,

    // Literals and identifiers
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    Ident(&'src str),
    #[regex(r"[0-9][0-9_]*")]
    Int(&'src str),
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?")]
    Float(&'src str),
    #[regex(r#""([^"\\]|\\.)*""#)]
    Str(&'src str),
    /// `f"..."`; interpolation is split by the parser.
    #[regex(r#"f"([^"\\]|\\.)*""#)]
    FStr(&'src str),

    // Operators
    #[token("|>")]
    Pipe,
    #[token("..=")]
    DotDotEq,
    #[token("..")]
    DotDot,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("::")]
    PathSep,
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("=")]
    Eq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("|")]
    Bar,
    #[token("!")]
    Bang,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(";")]
    Semi,
    #[token(":")]
    Colon,
    #[token("@")]
    At,
    #[token("?")]
    Question,

    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
}

impl std::fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Token::*;
        let s = match self {
            Import => "import",
            Fn => "fn",
            Let => "let",
            Mut => "mut",
            Struct => "struct",
            Enum => "enum",
            Pub => "pub",
            Match => "match",
            Impl => "impl",
            Trait => "trait",
            Return => "return",
            Break => "break",
            Continue => "continue",
            If => "if",
            Else => "else",
            For => "for",
            In => "in",
            While => "while",
            True => "true",
            False => "false",
            Ident(s) | Int(s) | Float(s) | Str(s) | FStr(s) => s,
            Pipe => "|>",
            DotDotEq => "..=",
            DotDot => "..",
            PlusEq => "+=",
            MinusEq => "-=",
            StarEq => "*=",
            SlashEq => "/=",
            PathSep => "::",
            Arrow => "->",
            FatArrow => "=>",
            EqEq => "==",
            NotEq => "!=",
            LtEq => "<=",
            GtEq => ">=",
            AndAnd => "&&",
            OrOr => "||",
            Eq => "=",
            Lt => "<",
            Gt => ">",
            Bar => "|",
            Bang => "!",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Dot => ".",
            Comma => ",",
            Semi => ";",
            Colon => ":",
            At => "@",
            Question => "?",
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
        };
        f.write_str(s)
    }
}

/// Tokenizes `src`, collecting lexical errors instead of stopping at the first one.
pub fn tokenize(src: &str) -> (Vec<(Token<'_>, Span)>, Vec<Span>) {
    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    for (res, span) in Token::lexer(src).spanned() {
        match res {
            Ok(tok) => tokens.push((tok, span)),
            Err(LexError) => errors.push(span),
        }
    }
    (tokens, errors)
}
