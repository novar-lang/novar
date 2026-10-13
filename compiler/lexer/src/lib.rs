//! Zero-copy tokenizer for Novar source code.

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
    #[token("return")]
    Return,
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
            Return => "return",
            If => "if",
            Else => "else",
            For => "for",
            In => "in",
            While => "while",
            True => "true",
            False => "false",
            Ident(s) | Int(s) | Float(s) | Str(s) | FStr(s) => s,
            Pipe => "|>",
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

#[cfg(test)]
mod tests {
    use super::*;
    use Token::*;

    fn kinds(src: &str) -> Vec<Token<'_>> {
        let (toks, errs) = tokenize(src);
        assert!(errs.is_empty(), "unexpected lex errors: {errs:?}");
        toks.into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn pipeline() {
        assert_eq!(
            kinds("text |> std::text::trim()"),
            [
                Ident("text"),
                Pipe,
                Ident("std"),
                PathSep,
                Ident("text"),
                PathSep,
                Ident("trim"),
                LParen,
                RParen
            ]
        );
    }

    #[test]
    fn closure_does_not_collide_with_pipe() {
        assert_eq!(
            kinds("|d| d.text"),
            [Bar, Ident("d"), Bar, Ident("d"), Dot, Ident("text")]
        );
    }

    #[test]
    fn tensor_type() {
        assert_eq!(
            kinds("-> Tensor<f32, [768]>"),
            [
                Arrow,
                Ident("Tensor"),
                Lt,
                Ident("f32"),
                Comma,
                LBracket,
                Int("768"),
                RBracket,
                Gt
            ]
        );
    }

    #[test]
    fn endpoint_annotation() {
        assert_eq!(
            kinds(r#"@endpoint(path = "/v1/rag/query", method = "POST")"#),
            [
                At,
                Ident("endpoint"),
                LParen,
                Ident("path"),
                Eq,
                Str(r#""/v1/rag/query""#),
                Comma,
                Ident("method"),
                Eq,
                Str(r#""POST""#),
                RParen
            ]
        );
    }

    #[test]
    fn fstring_is_not_ident_plus_string() {
        assert_eq!(kinds(r#"f"a {b} \" c""#), [FStr(r#"f"a {b} \" c""#)]);
        assert_eq!(kinds(r#"f "x""#), [Ident("f"), Str(r#""x""#)]);
    }

    #[test]
    fn numbers() {
        assert_eq!(
            kinds("1_000 3.14 2.5e-3"),
            [Int("1_000"), Float("3.14"), Float("2.5e-3")]
        );
    }

    #[test]
    fn keywords_vs_identifiers() {
        assert_eq!(kinds("let letter"), [Let, Ident("letter")]);
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(kinds("a // hi\nb"), [Ident("a"), Ident("b")]);
    }

    #[test]
    fn spans_are_byte_offsets() {
        let (toks, _) = tokenize("let x");
        assert_eq!(toks[1].1, 4..5);
    }

    #[test]
    fn errors_are_collected_and_lexing_continues() {
        let (toks, errs) = tokenize("a # b \"open");
        assert_eq!(toks.len(), 2);
        assert_eq!(errs, [2..3, 6..11]);
    }

    #[test]
    fn rag_example_lexes_cleanly() {
        let src = include_str!("../../../examples/rag-service.nv");
        let (toks, errs) = tokenize(src);
        assert!(errs.is_empty(), "{errs:?}");
        assert!(toks.iter().any(|(t, _)| *t == Pipe));
    }
}
