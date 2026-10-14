//! Zero-copy tokenizer for Telvax source code.
//!
//! Blocks are made by indentation, as in Python. [`tokenize`] turns the indentation into
//! [`Token::Indent`], [`Token::Dedent`] and the ends of lines into [`Token::Newline`], so the parser
//! never looks at whitespace. Line breaks inside `()`, `[]` and `{}` are ignored, and a line that
//! starts with `|>` continues the line before it.

use std::ops::Range;

use logos::Logos;

pub type Span = Range<usize>;

/// Raised for input that matches no token (stray character, unterminated string, ...).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LexError;

#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(error = LexError)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip(r"#[^\n]*", allow_greedy = true))]
pub enum Token<'src> {
    // Keywords
    #[token("import")]
    Import,
    #[token("from")]
    From,
    #[token("def")]
    Def,
    #[token("class")]
    Class,
    #[token("enum")]
    Enum,
    #[token("trait")]
    Trait,
    #[token("extend")]
    Extend,
    #[token("match")]
    Match,
    #[token("case")]
    Case,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("pass")]
    Pass,
    #[token("if")]
    If,
    #[token("elif")]
    Elif,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("is")]
    Is,
    #[token("while")]
    While,
    #[token("lambda")]
    Lambda,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,
    #[token("True")]
    True,
    #[token("False")]
    False,

    // Literals and identifiers
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    Ident(&'src str),
    #[regex(r"[0-9][0-9_]*")]
    Int(&'src str),
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?")]
    Float(&'src str),
    #[regex(r#""([^"\\]|\\.)*"|'([^'\\]|\\.)*'"#)]
    Str(&'src str),
    /// `f"..."`; interpolation is split by the parser.
    #[regex(r#"f"([^"\\]|\\.)*"|f'([^'\\]|\\.)*'"#)]
    FStr(&'src str),

    // Operators
    #[token("|>")]
    Pipe,
    #[token("...")]
    Ellipsis,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("->")]
    Arrow,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("=")]
    Eq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("|")]
    Bar,
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

    // Layout, produced by `tokenize` and never by the lexer itself.
    /// End of a logical line.
    Newline,
    /// The next line is indented one more level.
    Indent,
    /// The block ends.
    Dedent,
}

impl std::fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Token::*;
        let s = match self {
            Import => "import",
            From => "from",
            Def => "def",
            Class => "class",
            Enum => "enum",
            Trait => "trait",
            Extend => "extend",
            Match => "match",
            Case => "case",
            Return => "return",
            Break => "break",
            Continue => "continue",
            Pass => "pass",
            If => "if",
            Elif => "elif",
            Else => "else",
            For => "for",
            In => "in",
            Is => "is",
            While => "while",
            Lambda => "lambda",
            And => "and",
            Or => "or",
            Not => "not",
            True => "True",
            False => "False",
            Ident(s) | Int(s) | Float(s) | Str(s) | FStr(s) => s,
            Pipe => "|>",
            Ellipsis => "...",
            PlusEq => "+=",
            MinusEq => "-=",
            StarEq => "*=",
            SlashEq => "/=",
            Arrow => "->",
            EqEq => "==",
            NotEq => "!=",
            LtEq => "<=",
            GtEq => ">=",
            Eq => "=",
            Lt => "<",
            Gt => ">",
            Bar => "|",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Dot => ".",
            Comma => ",",
            Colon => ":",
            At => "@",
            Question => "?",
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
            Newline => "end of line",
            Indent => "indent",
            Dedent => "dedent",
        };
        f.write_str(s)
    }
}

/// Tokenizes `src` without any layout tokens, collecting lexical errors instead of stopping at the
/// first one. Used for pieces that never span lines, such as the expressions inside an f-string.
pub fn tokenize_flat(src: &str) -> (Vec<(Token<'_>, Span)>, Vec<Span>) {
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

/// Tokenizes a whole file: [`tokenize_flat`] plus `Newline`, `Indent` and `Dedent`.
///
/// A `Newline` ends every logical line; `Indent` and `Dedent` open and close blocks. Tabs in the
/// indentation and a dedent to a level that was never opened are lexical errors.
pub fn tokenize(src: &str) -> (Vec<(Token<'_>, Span)>, Vec<Span>) {
    let (raw, mut errors) = tokenize_flat(src);
    let mut out: Vec<(Token<'_>, Span)> = Vec::with_capacity(raw.len() + raw.len() / 4);
    let mut levels = vec![0usize];
    let mut depth = 0usize;
    let mut prev_end = 0usize;
    for (tok, span) in raw {
        let first = out.is_empty();
        let new_line = first || src[prev_end..span.start].contains('\n');
        if new_line && depth == 0 && (first || tok != Token::Pipe) {
            if !first {
                out.push((Token::Newline, prev_end..prev_end));
            }
            let line_start = src[..span.start].rfind('\n').map_or(0, |i| i + 1);
            let indent = &src[line_start..span.start];
            if indent.contains('\t') {
                errors.push(line_start..span.start);
            }
            let col = indent.chars().count();
            let top = *levels.last().unwrap_or(&0);
            if col > top {
                levels.push(col);
                out.push((Token::Indent, span.start..span.start));
            } else {
                while col < *levels.last().unwrap_or(&0) {
                    levels.pop();
                    out.push((Token::Dedent, span.start..span.start));
                }
                if col != *levels.last().unwrap_or(&0) {
                    errors.push(line_start..span.start);
                }
            }
        }
        match tok {
            Token::LParen | Token::LBracket | Token::LBrace => depth += 1,
            Token::RParen | Token::RBracket | Token::RBrace => depth = depth.saturating_sub(1),
            _ => {}
        }
        prev_end = span.end;
        out.push((tok, span));
    }
    if !out.is_empty() {
        out.push((Token::Newline, prev_end..prev_end));
        while levels.len() > 1 {
            levels.pop();
            out.push((Token::Dedent, src.len()..src.len()));
        }
    }
    (out, errors)
}
