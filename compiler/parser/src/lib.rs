//! Parser for Telvax: token stream -> AST.
//!
//! The surface syntax reads like Python (blocks by indentation, `elif`, `and` / `or` / `not`,
//! `from a.b import C`) with its own words: `fn`, `let` / `var`, `type`, `x => x + 1`, `Int?`. Everything is rewritten into the same tree the compiler
//! always used, so nothing after the parser knows about the surface: see `telvax_ast::tail` for the
//! rewrites that need more than one statement to decide (declarations and `return`).

use std::collections::HashSet;
use std::rc::Rc;

use chumsky::{
    input::{Stream, ValueInput},
    prelude::*,
};
use telvax_ast::*;
use telvax_ast::{Span, Spanned};
use telvax_diagnostics::Diagnostic;
use telvax_lexer::Token;

/// Names that start a dotted path (`util.clamp`) instead of a field access: `std`, the names
/// brought in by `import`, and anything that starts with a capital letter (`Shape.Circle`).
type Heads = Rc<HashSet<String>>;

macro_rules! run {
    ($parser:expr, $tokens:expr, $eoi:expr, $src:expr, $base:expr) => {{
        let iter = $tokens.into_iter().map(|(t, s)| (t, SimpleSpan::from(s)));
        let stream = Stream::from_iter(iter).map(SimpleSpan::from($eoi..$eoi), |(t, s)| (t, s));
        let (out, errs) = $parser.parse(stream).into_output_errors();
        let errs: Vec<Diagnostic> = errs
            .into_iter()
            .map(|e| friendly(&e, $src, $base))
            .collect();
        (out, errs)
    }};
}

/// Turns a parser error into a readable diagnostic: long lists of acceptable tokens collapse to
/// what was found, layout tokens are described in words, and the mistakes people bring from other
/// languages get a hint.
fn friendly(e: &Rich<'_, Token<'_>>, src: &str, base: usize) -> Diagnostic {
    use chumsky::error::RichPattern;
    let span = e.span().into_range();
    let expects = |t: Token<'_>| {
        e.expected()
            .any(|p| matches!(p, RichPattern::Token(x) if **x == t))
    };
    let rest = src.get(span.start.saturating_sub(base)..).unwrap_or("");
    match e.found() {
        Some(Token::Indent) => {
            return Diagnostic::error(span, "unexpected indentation")
                .with_note("the lines of a block start in the same column");
        }
        Some(Token::Dedent) => {
            return Diagnostic::error(span, "the block ends here, but something is missing");
        }
        Some(Token::LBrace) if expects(Token::Colon) => {
            return Diagnostic::error(span, "a block starts with `:` and is indented, not `{ }`")
                .with_note("write `:` at the end of the line and indent the lines below it");
        }
        Some(Token::RParen | Token::Comma) if expects(Token::Colon) => {
            return Diagnostic::error(span, "a parameter or a field needs a type")
                .with_note("write `name: Type`, for example `x: Int`");
        }
        Some(Token::Ident(_))
            if expects(Token::Newline) && after_colon_in_fn(src, span.start, base) =>
        {
            return Diagnostic::error(span, "a return type is written `-> Type`, not `: Type`")
                .with_note("for example `fn area(r: Float) -> Float:`");
        }
        Some(Token::Newline) if expects(Token::Colon) => {
            return Diagnostic::error(span, "expected `:` at the end of this line")
                .with_note("a line that opens a block ends with `:`");
        }
        Some(Token::Newline) if e.expected().count() > 5 => {
            return Diagnostic::error(
                span,
                "the line ends here, but the statement is not finished",
            );
        }
        Some(Token::Dot)
            if rest.starts_with("..") || src[..span.start.saturating_sub(base)].ends_with('.') =>
        {
            return Diagnostic::error(span, "`..` is not used in Telvax")
                .with_note("write a range of numbers as `range(start, end)`");
        }
        _ => {}
    }
    if expects(Token::Indent) {
        return Diagnostic::error(span, "expected an indented block here")
            .with_note("the lines after a line that ends with `:` are indented");
    }
    if e.expected().count() > 5
        && let Some(found) = e.found()
    {
        return Diagnostic::error(span, format!("unexpected `{found}`"));
    }
    Diagnostic::error(span, e.to_string())
}

/// Whether the error at `start` follows the `:` of a `fn` header (`fn f(x: Int): Int:`).
fn after_colon_in_fn(src: &str, start: usize, base: usize) -> bool {
    let before = &src[..start.saturating_sub(base).min(src.len())];
    let line = before.rsplit('\n').next().unwrap_or("");
    line.trim_start().starts_with("fn ") && line.trim_end().ends_with(':')
}

/// What a lexical error is about, from the text it covers.
fn lex_message(text: &str) -> Diagnostic {
    let span = 0..0;
    let (message, note) = if text.contains('\t') {
        (
            "tabs are not allowed in indentation",
            Some("indent with spaces (four per level)"),
        )
    } else if text.trim().is_empty() {
        (
            "this line is indented differently from the lines above it",
            Some("every line of a block starts in the same column"),
        )
    } else if text == ";" {
        ("`;` is not used in Telvax", Some("end the line instead"))
    } else if text == "&" {
        ("`&&` is not used in Telvax", Some("write `and`"))
    } else if text == "!" {
        ("`!` is not used in Telvax", Some("write `not`"))
    } else if text.starts_with(['"', '\'']) {
        (
            "this string is never closed",
            Some("add the closing quote on the same line"),
        )
    } else {
        ("unexpected character", None)
    };
    let d = Diagnostic::error(span, message);
    match note {
        Some(n) => d.with_note(n),
        None => d,
    }
}

/// Words from other languages that start a line, with what to write in Telvax.
fn foreign_word(word: &str) -> Option<&'static str> {
    Some(match word {
        "def" => "`def` is not used in Telvax: write `fn name(...)`",
        "class" => "`class` is not used in Telvax: write `type Name:`",
        "lambda" => "`lambda` is not used in Telvax: write `x => x + 1`",
        "mut" => "`mut` is not used in Telvax: write `var` for a variable that changes",
        "struct" => "`struct` is not used in Telvax: write `type Name:`",
        "impl" => {
            "`impl` is not used in Telvax: put the methods in the `type`, or use `extend Name:`"
        }
        "use" => "`use` is not used in Telvax: write `from a.b import C` or `import a.b.C`",
        "const" => "`const` is not used in Telvax: write `let` for a name that does not change",
        "function" | "func" | "fun" => "`function` is not used in Telvax: write `fn name(...)`",
        "pub" => {
            "`pub` is not used in Telvax: names that start with `_` are private, all others are public"
        }
        _ => return None,
    })
}

/// The names `import` and `from` bring in, which may start a dotted path: the last name of
/// `import a.b.C`, and the names after `import` in `from a.b import C, D`.
fn import_heads(tokens: &[(Token<'_>, std::ops::Range<usize>)]) -> HashSet<String> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Outside,
        /// `import a.b.C`
        Plain,
        /// `from a.b` before `import`
        FromPath,
        /// `from a.b import C, D`
        FromNames,
    }
    let mut heads = HashSet::from(["std".to_string()]);
    let mut mode = Mode::Outside;
    let mut last: Option<&str> = None;
    for (t, _) in tokens {
        match (t, mode) {
            (Token::Import, Mode::Outside) => mode = Mode::Plain,
            (Token::Import, Mode::FromPath) => mode = Mode::FromNames,
            (Token::From, Mode::Outside) => mode = Mode::FromPath,
            (Token::Ident(name), Mode::Plain) => last = Some(name),
            (Token::Ident(name), Mode::FromNames) => {
                heads.insert((*name).to_string());
            }
            (Token::Newline, _) => {
                if let (Mode::Plain, Some(name)) = (mode, last) {
                    heads.insert(name.to_string());
                }
                mode = Mode::Outside;
                last = None;
            }
            _ => {}
        }
    }
    heads
}

/// Lexes and parses `src`. Returns the program (when recovery allowed one) and all errors.
pub fn parse(src: &str) -> (Option<Program>, Vec<Diagnostic>) {
    parse_at(src, 0)
}

/// Like [`parse`], with every span shifted by `base`. Lets several files share one offset space,
/// so a single span identifies a place in a whole multi-file program.
pub fn parse_at(src: &str, base: usize) -> (Option<Program>, Vec<Diagnostic>) {
    let (mut tokens, lex_errors) = telvax_lexer::tokenize(src);
    let heads: Heads = Rc::new(import_heads(&tokens));
    for (_, span) in &mut tokens {
        *span = span.start + base..span.end + base;
    }
    let first_words: Vec<(String, std::ops::Range<usize>)> = {
        let mut out = Vec::new();
        let mut at_start = true;
        for (t, span) in &tokens {
            match t {
                Token::Newline | Token::Indent | Token::Dedent => at_start = true,
                Token::Ident(w) if at_start => {
                    out.push(((*w).to_string(), span.clone()));
                    at_start = false;
                }
                _ => at_start = false,
            }
        }
        out
    };
    let mut errors: Vec<Diagnostic> = lex_errors
        .into_iter()
        .map(|span| {
            let mut d = lex_message(&src[span.clone()]);
            d.span = span.start + base..span.end + base;
            d
        })
        .collect();
    let (program, parse_errors) = run!(program(heads), tokens, src.len() + base, src, base);
    errors.extend(parse_errors);
    if !errors.is_empty() {
        // Someone who writes `fn` or `let` at the start of a line needs to hear what to write.
        for (word, span) in first_words {
            if let Some(hint) = foreign_word(&word) {
                // The word's own message says more than "unexpected `def`".
                errors.retain(|e| e.span != span);
                errors.push(Diagnostic::error(span, hint));
            }
        }
    }
    errors.sort_by_key(|e| e.span.start);
    // After a mistake, the lines that hang from it are not worth reporting.
    let mut kept: Vec<Diagnostic> = Vec::new();
    for e in errors {
        let after_error = kept.last().is_some_and(|prev| {
            src[prev.span.end.saturating_sub(base)
                ..e.span
                    .start
                    .saturating_sub(base)
                    .max(prev.span.end.saturating_sub(base))]
                .matches('\n')
                .count()
                <= 1
        });
        if e.message == "unexpected indentation" && after_error {
            continue;
        }
        kept.push(e);
    }
    (program, kept)
}

/// Parses a single expression (used for f-string interpolations and tests).
pub fn parse_expr(src: &str) -> (Option<Expr>, Vec<Diagnostic>) {
    parse_expr_at(src, 0, &Rc::new(HashSet::from(["std".to_string()])))
}

fn parse_expr_at(src: &str, offset: usize, heads: &Heads) -> (Option<Expr>, Vec<Diagnostic>) {
    let (mut tokens, lex_errors) = telvax_lexer::tokenize_flat(src);
    for (_, span) in &mut tokens {
        *span = span.start + offset..span.end + offset;
    }
    let mut errors: Vec<Diagnostic> = lex_errors
        .into_iter()
        .map(|s| Diagnostic::error(s.start + offset..s.end + offset, "invalid token"))
        .collect();
    let (expr, parse_errors) = run!(
        expr(heads.clone()).then_ignore(end()),
        tokens,
        src.len() + offset,
        src,
        offset
    );
    errors.extend(parse_errors);
    (expr, errors)
}

type Extra<'a, 's> = extra::Err<Rich<'a, Token<'s>, SimpleSpan>>;
fn rng(span: SimpleSpan) -> Span {
    span.into_range()
}

fn spanned<T>(node: T, span: SimpleSpan) -> Spanned<T> {
    Spanned {
        node,
        span: span.into_range(),
    }
}

fn ident<'a, 's: 'a, I>() -> impl Parser<'a, I, Ident, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    select! { Token::Ident(s) => s.to_string() }
        .map_with(|s, e| spanned(s, e.span()))
        .labelled("identifier")
}

/// `a.b.c`: a name with optional dotted parts (types, patterns, imports).
fn path<'a, 's: 'a, I>() -> impl Parser<'a, I, Vec<Ident>, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    ident()
        .separated_by(just(Token::Dot))
        .at_least(1)
        .collect::<Vec<_>>()
}

/// A bare name with this exact spelling (`None`, `self`).
fn word<'a, 's: 'a, I>(w: &'static str) -> impl Parser<'a, I, (), Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    select! { Token::Ident(s) if s == w => () }
}

/// `Tensor[f32, 2, 3]`: after the element type, the arguments are the dimensions.
fn tensor_shape(args: Vec<TypeArg>) -> Vec<TypeArg> {
    let mut out = Vec::new();
    let mut dims = Vec::new();
    for (i, a) in args.into_iter().enumerate() {
        match a {
            TypeArg::Shape(d) if i > 0 => dims.extend(d),
            TypeArg::Type(t) if i > 0 && t.path.len() == 1 && t.args.is_empty() => {
                dims.push(Dim::Named(t.path[0].clone()));
            }
            other => out.push(other),
        }
    }
    if !dims.is_empty() {
        out.push(TypeArg::Shape(dims));
    }
    out
}

fn type_<'a, 's: 'a, I>() -> impl Parser<'a, I, Type, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    recursive(|ty| {
        let dim = select! { Token::Int(n) => n }.try_map(|n, span| {
            n.replace('_', "")
                .parse::<u64>()
                .map(|v| TypeArg::Shape(vec![Dim::Lit(v)]))
                .map_err(|_| Rich::custom(span, "integer literal out of range"))
        });
        let row = ident()
            .then_ignore(just(Token::Colon))
            .then(ty.clone())
            .map(|(name, ty)| Field {
                name,
                ty,
                default: None,
            })
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(TypeArg::Row);
        let arg = ty.clone().map(TypeArg::Type).or(dim).or(row);
        let args = arg
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket));
        // `(Int, Text)`: a tuple type is spelled with a name no program can write.
        let tuple_ty = ty
            .clone()
            .separated_by(just(Token::Comma))
            .at_least(2)
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map_with(|items, e| {
                let span = rng(e.span());
                Type {
                    path: vec![Ident {
                        node: "(tuple)".into(),
                        span: span.clone(),
                    }],
                    args: items.into_iter().map(TypeArg::Type).collect(),
                    span,
                }
            });
        // `(Int, Int) -> Int`, `() -> Text`: the type of a function. The same spelling as a
        // function header's return type, with the parameter types in parentheses. `()` is the
        // type of a function that returns nothing: `(Int) -> ()`.
        let unit_ty = just(Token::LParen)
            .then(just(Token::RParen))
            .map_with(|_, e| {
                let span = rng(e.span());
                Type {
                    path: vec![Ident {
                        node: "(unit)".into(),
                        span: span.clone(),
                    }],
                    args: Vec::new(),
                    span,
                }
            });
        let fn_ty = ty
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .then_ignore(just(Token::Arrow))
            .then(ty.clone())
            .map_with(|(mut params, ret), e| {
                let span = rng(e.span());
                params.push(ret);
                Type {
                    path: vec![Ident {
                        node: "(fn)".into(),
                        span: span.clone(),
                    }],
                    args: params.into_iter().map(TypeArg::Type).collect(),
                    span,
                }
            });
        let named = path().then(args.or_not()).map_with(|(mut path, args), e| {
            if path.len() == 1 {
                let canon = telvax_ast::canonical_type_name(&path[0].node).to_string();
                path[0].node = canon;
            }
            let mut args = args.unwrap_or_default();
            if path.last().is_some_and(|p| p.node == "Tensor") {
                args = tensor_shape(args);
            }
            Type {
                path,
                args,
                span: rng(e.span()),
            }
        });
        // `T?` is `Option<T>`.
        fn_ty
            .or(tuple_ty)
            .or(unit_ty)
            .or(named)
            .then(just(Token::Question).or_not())
            .map_with(|(base, optional), e| match optional {
                None => base,
                Some(_) => {
                    let span = rng(e.span());
                    Type {
                        path: vec![Ident {
                            node: "Option".into(),
                            span: span.clone(),
                        }],
                        args: vec![TypeArg::Type(base)],
                        span,
                    }
                }
            })
    })
}

fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

/// Splits the body of an f-string (without `f"` and `"`) into text and `{expr}` parts.
/// `offset` is the byte offset of `body` in the source file.
fn fstring_parts(body: &str, offset: usize, heads: &Heads) -> Result<Vec<FPart>, Diagnostic> {
    let err = |start: usize, end: usize, message: &str| {
        Diagnostic::error(offset + start..offset + end, message)
    };
    let mut parts = Vec::new();
    let mut text = String::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => {
                let next = body[i + 1..].chars().next().unwrap();
                text.push_str(&unescape(&body[i..i + 1 + next.len_utf8()]));
                i += 1 + next.len_utf8();
            }
            b'{' => {
                let start = i + 1;
                let mut depth = 1;
                let mut j = start;
                while j < bytes.len() && depth > 0 {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    return Err(err(i, bytes.len(), "unclosed `{` in f-string"));
                }
                if !text.is_empty() {
                    parts.push(FPart::Text(std::mem::take(&mut text)));
                }
                let inner = &body[start..j - 1];
                match parse_expr_at(inner, offset + start, heads) {
                    (Some(e), errs) if errs.is_empty() => parts.push(FPart::Expr(e)),
                    (_, mut errs) => {
                        return Err(errs.drain(..).next().unwrap_or_else(|| {
                            err(start, j - 1, "invalid expression in f-string")
                        }));
                    }
                }
                i = j;
            }
            _ => {
                let c = body[i..].chars().next().unwrap();
                text.push(c);
                i += c.len_utf8();
            }
        }
    }
    if !text.is_empty() {
        parts.push(FPart::Text(text));
    }
    Ok(parts)
}

/// What one line of a block stands for.
#[derive(Clone)]
enum BlockItem {
    Let(Let),
    /// `a, b = value` becomes a temporary and one `let` per name.
    Lets(Vec<Let>),
    Expr(Expr),
    /// `pass`
    Nothing,
}

fn make_block(items: Vec<BlockItem>, span: SimpleSpan) -> Block {
    let mut stmts = Vec::new();
    for item in items {
        match item {
            BlockItem::Let(l) => stmts.push(Stmt::Let(l)),
            BlockItem::Lets(ls) => stmts.extend(ls.into_iter().map(Stmt::Let)),
            BlockItem::Expr(e) => stmts.push(Stmt::Expr(e)),
            BlockItem::Nothing => {}
        }
    }
    Block {
        stmts,
        tail: None,
        span: span.into_range(),
    }
}

fn unit_block(span: SimpleSpan) -> Block {
    Block {
        stmts: Vec::new(),
        tail: None,
        span: rng(span),
    }
}

fn value_block(value: Expr, span: SimpleSpan) -> Block {
    Block {
        stmts: Vec::new(),
        tail: Some(Box::new(value)),
        span: rng(span),
    }
}

/// `[T, U: Shape, V: (A, B)]`: the type parameters of a function, type or trait.
fn generics<'a, 's: 'a, I>() -> impl Parser<'a, I, Vec<GenericParam>, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    let bounds = ident()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .or(ident().map(|i| vec![i]));
    ident()
        .then(just(Token::Colon).ignore_then(bounds).or_not())
        .map(|(name, bounds)| GenericParam {
            name,
            bounds: bounds.unwrap_or_default(),
        })
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
        .or_not()
        .map(Option::unwrap_or_default)
}

/// `a, b, _` or `(a, b, _)`: names for the parts of a tuple (`_` skips one).
fn tuple_names<'a, 's: 'a, I>() -> impl Parser<'a, I, Vec<Ident>, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    ident()
        .separated_by(just(Token::Comma))
        .at_least(2)
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
}

/// The `let`s a tuple pattern stands for: a temporary holding the value, then `name = temp.N`.
fn tuple_lets(
    names: Vec<Ident>,
    temp: Ident,
    init: Option<Expr>,
    mutable: bool,
    span: Span,
) -> Vec<Let> {
    let mut out = Vec::new();
    if let Some(init) = init {
        out.push(Let {
            mutable: false,
            name: temp.clone(),
            ty: None,
            init,
            span: span.clone(),
        });
    }
    for (i, name) in names.into_iter().enumerate() {
        if name.node == "_" {
            continue;
        }
        let at = name.span.clone();
        out.push(Let {
            mutable,
            name,
            ty: None,
            init: Spanned {
                node: ExprKind::Field {
                    base: Box::new(Spanned {
                        node: ExprKind::Path(vec![temp.clone()]),
                        span: at.clone(),
                    }),
                    name: Spanned {
                        node: i.to_string(),
                        span: at.clone(),
                    },
                },
                span: at,
            },
            span: span.clone(),
        });
    }
    out
}

fn pattern<'a, 's: 'a, I>() -> impl Parser<'a, I, Pattern, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    recursive(|pat| {
        let int = select! { Token::Int(n) => n }.try_map(|n, span| {
            n.replace('_', "")
                .parse::<u64>()
                .map(|v| PatKind::Lit(Lit::Int(v)))
                .map_err(|_| Rich::custom(span, "integer literal out of range"))
        });
        let float = select! { Token::Float(n) => n }.try_map(|n, span| {
            n.replace('_', "")
                .parse::<f64>()
                .map(|v| PatKind::Lit(Lit::Float(v)))
                .map_err(|_| Rich::custom(span, "invalid float literal"))
        });
        let string = select! { Token::Str(s) => s }
            .map(|s| PatKind::Lit(Lit::Str(unescape(&s[1..s.len() - 1]))));
        let boolean = select! { Token::True => true, Token::False => false }
            .map(|b| PatKind::Lit(Lit::Bool(b)));
        let variant = path()
            .then(
                pat.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LParen), just(Token::RParen))
                    .or_not(),
            )
            .map(|(path, args)| {
                if path.len() == 1 && args.is_none() {
                    let name = &path[0].node;
                    if name == "_" {
                        PatKind::Wild
                    } else {
                        PatKind::Bind(name.clone())
                    }
                } else {
                    PatKind::Variant {
                        path,
                        args: args.unwrap_or_default(),
                    }
                }
            });
        let tuple = pat
            .clone()
            .separated_by(just(Token::Comma))
            .at_least(2)
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(PatKind::Tuple);
        choice((int, float, string, boolean, tuple, variant)).map_with(|k, e| spanned(k, e.span()))
    })
}

fn is_head(name: &str, heads: &HashSet<String>) -> bool {
    heads.contains(name) || name.starts_with(|c: char| c.is_uppercase())
}

fn expr<'a, 's: 'a, I>(heads: Heads) -> impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    recursive(move |expr| {
        let int = select! { Token::Int(n) => n }.try_map(|n, span| {
            n.replace('_', "")
                .parse::<u64>()
                .map(|v| ExprKind::Lit(Lit::Int(v)))
                .map_err(|_| Rich::custom(span, "integer literal out of range"))
        });
        let float = select! { Token::Float(n) => n }.try_map(|n, span| {
            n.replace('_', "")
                .parse::<f64>()
                .map(|v| ExprKind::Lit(Lit::Float(v)))
                .map_err(|_| Rich::custom(span, "invalid float literal"))
        });
        let string = select! { Token::Str(s) => s }
            .map(|s| ExprKind::Lit(Lit::Str(unescape(&s[1..s.len() - 1]))));
        let boolean = select! { Token::True => true, Token::False => false }
            .map(|b| ExprKind::Lit(Lit::Bool(b)));
        let fstring_heads = heads.clone();
        let fstring = select! { Token::FStr(s) => s }
            .map_with(|s, e| (s, rng(e.span()).start))
            .try_map(move |(s, start), _| {
                // Strip `f"` and the closing quote.
                fstring_parts(&s[2..s.len() - 1], start + 2, &fstring_heads)
                    .map(ExprKind::FString)
                    .map_err(|e| Rich::custom(SimpleSpan::from(e.span), e.message))
            });

        // `(a, b)` is a tuple; `(a)` is just `a` in parentheses.
        let tuple = expr
            .clone()
            .then(
                just(Token::Comma)
                    .ignore_then(expr.clone())
                    .repeated()
                    .at_least(1)
                    .collect::<Vec<_>>(),
            )
            .then_ignore(just(Token::Comma).or_not())
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(|(first, rest)| {
                let mut items = vec![first];
                items.extend(rest);
                ExprKind::Tuple(items)
            });
        let paren = tuple
            .or(expr
                .clone()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .map(|e| e.node))
            .recover_with(via_parser(nested_delimiters(
                Token::LParen,
                Token::RParen,
                [
                    (Token::LBrace, Token::RBrace),
                    (Token::LBracket, Token::RBracket),
                ],
                |_| ExprKind::Error,
            )));

        let array = expr
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(ExprKind::Array);

        let primary = choice((
            float,
            int,
            string,
            fstring,
            boolean,
            ident().map(|i| ExprKind::Path(vec![i])),
            paren,
            array,
        ))
        .map_with(|k, e| spanned(k, e.span()));

        #[derive(Clone)]
        enum Postfix {
            Call(Vec<Expr>),
            Field(Ident),
            Index(Expr),
            Try,
            Parts(Vec<Ident>),
        }
        // An argument is `value` or `name: value`.
        let argument = ident()
            .then_ignore(just(Token::Colon))
            .then(expr.clone())
            .map_with(|(name, value), e| {
                spanned(
                    ExprKind::Named {
                        name,
                        value: Box::new(value),
                    },
                    e.span(),
                )
            })
            .or(expr.clone());
        let call = argument
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(Postfix::Call)
            .recover_with(via_parser(nested_delimiters(
                Token::LParen,
                Token::RParen,
                [
                    (Token::LBrace, Token::RBrace),
                    (Token::LBracket, Token::RBracket),
                ],
                |_| Postfix::Call(Vec::new()),
            )));
        // `pair.1` lexes as `.` and an integer, and `pair.0.1` as `.` and the float `0.1`.
        let numeric_field =
            select! { Token::Int(n) => n.to_string(), Token::Float(n) => n.to_string() }.map_with(
                |text, e| {
                    let at = rng(e.span());
                    Postfix::Parts(
                        text.split('.')
                            .map(|part| Ident {
                                node: part.to_string(),
                                span: at.clone(),
                            })
                            .collect(),
                    )
                },
            );
        let field = just(Token::Dot).ignore_then(ident().map(Postfix::Field).or(numeric_field));
        let try_ = just(Token::Question).to(Postfix::Try);
        // `xs[i]`; with commas, `f[Int, Text]` names the types a generic function is called with
        // and the index is the tuple of them.
        let index = expr
            .clone()
            .separated_by(just(Token::Comma))
            .at_least(1)
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with(|mut items, e| {
                Postfix::Index(if items.len() == 1 {
                    items.remove(0)
                } else {
                    spanned(ExprKind::Tuple(items), e.span())
                })
            });
        let path_heads = heads.clone();
        let postfix = primary.foldl_with(
            call.or(field).or(index).or(try_).repeated(),
            move |base, op, e| {
                let kind = match op {
                    Postfix::Index(index) => ExprKind::Index {
                        base: Box::new(base),
                        index: Box::new(index),
                    },
                    Postfix::Call(args) => {
                        // `Point(x: 1, y: 2)`: a capitalized name with every argument named builds
                        // a value of that type.
                        let builds = !args.is_empty()
                            && args
                                .iter()
                                .all(|a| matches!(a.node, ExprKind::Named { .. }))
                            && matches!(&base.node, ExprKind::Path(p)
                                if p.last().is_some_and(|i| i.node.starts_with(|c: char| c.is_uppercase())));
                        if builds {
                            let ExprKind::Path(path) = base.node else {
                                unreachable!()
                            };
                            let fields = args
                                .into_iter()
                                .filter_map(|a| match a.node {
                                    ExprKind::Named { name, value } => Some((name, *value)),
                                    _ => None,
                                })
                                .collect();
                            ExprKind::StructLit { path, fields }
                        } else {
                            ExprKind::Call {
                                callee: Box::new(base),
                                args,
                            }
                        }
                    }
                    Postfix::Try => ExprKind::Try(Box::new(base)),
                    Postfix::Parts(names) => {
                        let mut current = base;
                        for name in names {
                            current = spanned(
                                ExprKind::Field {
                                    base: Box::new(current),
                                    name,
                                },
                                e.span(),
                            );
                        }
                        current.node
                    }
                    Postfix::Field(name) => match base.node {
                        // `std.text.trim`, `Shape.Circle`, `util.clamp`: a path, not a field.
                        ExprKind::Path(mut path) if is_head(&path[0].node, &path_heads) => {
                            path.push(name);
                            ExprKind::Path(path)
                        }
                        node => ExprKind::Field {
                            base: Box::new(spanned(node, SimpleSpan::from(base.span))),
                            name,
                        },
                    },
                };
                spanned(kind, e.span())
            },
        );

        let unary =
            select! { Token::Minus => UnOp::Neg }
                .repeated()
                .foldr_with(postfix, |op, rhs, e| {
                    spanned(
                        ExprKind::Unary {
                            op,
                            expr: Box::new(rhs),
                        },
                        e.span(),
                    )
                });

        let binary = |lower: Boxed<'a, 'a, I, Expr, Extra<'a, 's>>,
                      ops: Boxed<'a, 'a, I, BinOp, Extra<'a, 's>>| {
            lower
                .clone()
                .foldl_with(ops.then(lower).repeated(), |lhs, (op, rhs), e| {
                    spanned(
                        ExprKind::Binary {
                            op,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        e.span(),
                    )
                })
                .boxed()
        };

        let mul = binary(
            unary.boxed(),
            select! { Token::Star => BinOp::Mul, Token::Slash => BinOp::Div, Token::Percent => BinOp::Rem }
                .boxed(),
        );
        let add = binary(
            mul,
            select! { Token::Plus => BinOp::Add, Token::Minus => BinOp::Sub }.boxed(),
        );

        // `<`, `<=`, `>`, `>=`, and the membership tests `in` / `not in`.
        #[derive(Clone, Copy)]
        enum Cmp {
            Op(BinOp),
            In,
            NotIn,
        }
        let cmp_op = select! {
            Token::Lt => Cmp::Op(BinOp::Lt), Token::LtEq => Cmp::Op(BinOp::LtEq),
            Token::Gt => Cmp::Op(BinOp::Gt), Token::GtEq => Cmp::Op(BinOp::GtEq),
            Token::In => Cmp::In,
        }
        .or(just(Token::Not).ignore_then(just(Token::In)).to(Cmp::NotIn));
        let cmp = add
            .clone()
            .foldl_with(cmp_op.then(add).repeated(), |lhs, (op, rhs), e| {
                let kind = match op {
                    Cmp::Op(op) => ExprKind::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    Cmp::In | Cmp::NotIn => {
                        // `a in b` is `b.contains(a)`.
                        let at = rhs.span.clone();
                        let contains = spanned(
                            ExprKind::Call {
                                callee: Box::new(Spanned {
                                    node: ExprKind::Field {
                                        base: Box::new(rhs),
                                        name: Ident {
                                            node: "contains".into(),
                                            span: at.clone(),
                                        },
                                    },
                                    span: at,
                                }),
                                args: vec![lhs],
                            },
                            e.span(),
                        );
                        if matches!(op, Cmp::NotIn) {
                            ExprKind::Unary {
                                op: UnOp::Not,
                                expr: Box::new(contains),
                            }
                        } else {
                            contains.node
                        }
                    }
                };
                spanned(kind, e.span())
            })
            .boxed();
        let eq = binary(
            cmp,
            select! { Token::EqEq => BinOp::Eq, Token::NotEq => BinOp::NotEq }.boxed(),
        );
        let not = just(Token::Not)
            .repeated()
            .foldr_with(eq, |_, rhs, e| {
                spanned(
                    ExprKind::Unary {
                        op: UnOp::Not,
                        expr: Box::new(rhs),
                    },
                    e.span(),
                )
            })
            .boxed();
        let and = binary(not, select! { Token::And => BinOp::And }.boxed());
        let or = binary(and, select! { Token::Or => BinOp::Or }.boxed());

        let pipe = or
            .clone()
            .foldl_with(
                just(Token::Pipe).ignore_then(or).repeated(),
                |lhs, rhs, e| {
                    spanned(
                        ExprKind::Pipe {
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        e.span(),
                    )
                },
            )
            .boxed();

        // `a if cond else b`
        let ternary = pipe
            .clone()
            .then(
                just(Token::If)
                    .ignore_then(pipe)
                    .then_ignore(just(Token::Else))
                    .then(expr.clone())
                    .or_not(),
            )
            .map_with(|(then, rest), e| match rest {
                None => then,
                Some((cond, otherwise)) => {
                    let (a, b) = (then.span.clone(), otherwise.span.clone());
                    spanned(
                        ExprKind::If {
                            cond: Box::new(cond),
                            then: value_block(then, SimpleSpan::from(a)),
                            otherwise: Some(Box::new(spanned(
                                ExprKind::Block(value_block(
                                    otherwise,
                                    SimpleSpan::from(b.clone()),
                                )),
                                SimpleSpan::from(b),
                            ))),
                        },
                        e.span(),
                    )
                }
            });

        // `x => x + 1`, `(a, b) => a + b`, `() => 1`; a parameter may also be a tuple of names:
        // `((i, s)) => ..`.
        // A name may carry its type: `(x: Int) => x + 1`.
        let arrow_param = ident()
            .then(just(Token::Colon).ignore_then(type_()).or_not())
            .map(|(name, ty)| (name, Vec::new(), ty))
            .or(tuple_names().map_with(|names, e| {
                let at = rng(e.span());
                (
                    Ident {
                        node: format!("__arg{}", at.start),
                        span: at,
                    },
                    names,
                    None,
                )
            }));
        let arrow_params = arrow_param
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .or(arrow_param.map(|p| vec![p]));
        let lambda = arrow_params
            .then_ignore(just(Token::FatArrow))
            .then(expr.clone())
            .map_with(|(params, body), e| {
                let mut names = Vec::new();
                let mut types = Vec::new();
                let mut lets = Vec::new();
                for (name, parts, ty) in params {
                    types.push(ty);
                    if !parts.is_empty() {
                        let span = name.span.clone();
                        lets.extend(tuple_lets(parts, name.clone(), None, false, span));
                    }
                    names.push(name);
                }
                // The parts of tuple parameters are bound at the top of the body.
                let body = if lets.is_empty() {
                    body
                } else {
                    spanned(
                        ExprKind::Block(Block {
                            stmts: lets.into_iter().map(Stmt::Let).collect(),
                            tail: Some(Box::new(body)),
                            span: rng(e.span()),
                        }),
                        e.span(),
                    )
                };
                spanned(
                    ExprKind::Closure {
                        params: names,
                        types,
                        body: Box::new(body),
                    },
                    e.span(),
                )
            });

        lambda.or(ternary)
    })
}

/// `if PATTERN` -> a `match` with a catch-all arm, as `if x is Some(v):` needs.
fn if_let(
    pat: Pattern,
    scrutinee: Expr,
    then: Block,
    otherwise: Option<Expr>,
    span: SimpleSpan,
) -> Expr {
    let then_span = then.span.clone();
    spanned(
        ExprKind::Match {
            scrutinee: Box::new(scrutinee),
            arms: vec![
                Arm {
                    pat,
                    body: Spanned {
                        node: ExprKind::Block(then),
                        span: then_span,
                    },
                },
                Arm {
                    pat: spanned(PatKind::Wild, span),
                    body: otherwise
                        .unwrap_or_else(|| spanned(ExprKind::Block(unit_block(span)), span)),
                },
            ],
        },
        span,
    )
}

/// `while x is Some(v):` loops until the pattern stops matching.
fn while_let(pat: Pattern, scrutinee: Expr, body: Block, span: SimpleSpan) -> Expr {
    let leave = Block {
        stmts: vec![Stmt::Expr(spanned(ExprKind::Break, span))],
        tail: None,
        span: rng(span),
    };
    let body_span = body.span.clone();
    let matched = spanned(
        ExprKind::Match {
            scrutinee: Box::new(scrutinee),
            arms: vec![
                Arm {
                    pat,
                    body: Spanned {
                        node: ExprKind::Block(body),
                        span: body_span,
                    },
                },
                Arm {
                    pat: spanned(PatKind::Wild, span),
                    body: spanned(ExprKind::Block(leave), span),
                },
            ],
        },
        span,
    );
    spanned(
        ExprKind::While {
            cond: Box::new(spanned(ExprKind::Lit(Lit::Bool(true)), span)),
            body: Block {
                stmts: Vec::new(),
                tail: Some(Box::new(matched)),
                span: rng(span),
            },
        },
        span,
    )
}

/// `: NEWLINE INDENT statements DEDENT`
fn block_of<'a, 's: 'a, I>(
    stmt: impl Parser<'a, I, BlockItem, Extra<'a, 's>> + Clone,
) -> impl Parser<'a, I, Block, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    just(Token::Colon)
        .ignore_then(just(Token::Newline))
        .ignore_then(just(Token::Indent))
        .ignore_then(stmt.repeated().at_least(1).collect::<Vec<_>>())
        .then_ignore(just(Token::Dedent))
        .map_with(|items, e| make_block(items, e.span()))
}

/// One statement of a block. Simple statements end with a line break; compound ones end with
/// their own block.
fn statement<'a, 's: 'a, I>(
    expr: impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone + 'a,
) -> impl Parser<'a, I, BlockItem, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan> + 'a,
{
    recursive(move |stmt| {
        let block = block_of(stmt.clone());
        let pattern = pattern();

        // The test of an `if`, `elif` or `while`: an expression, or `expr is PATTERN`.
        let test = expr
            .clone()
            .then(just(Token::Is).ignore_then(pattern.clone()).or_not());

        let else_block = just(Token::Else).ignore_then(block.clone());
        // `if` / `elif` chains are built from the last branch backwards.
        let branch = |kw: Token<'s>| {
            just(kw)
                .ignore_then(test.clone())
                .then(block.clone())
                .map_with(|((cond, pat), then), e| (cond, pat, then, e.span()))
        };
        let if_ = branch(Token::If)
            .then(branch(Token::Elif).repeated().collect::<Vec<_>>())
            .then(else_block.or_not())
            .map_with(|((first, rest), last), e| {
                let total = e.span();
                let mut branches = vec![first];
                branches.extend(rest);
                let mut tail: Option<Expr> = last.map(|b| {
                    let span = SimpleSpan::from(b.span.clone());
                    spanned(ExprKind::Block(b), span)
                });
                while let Some((cond, pat, then, _)) = branches.pop() {
                    let span = SimpleSpan::from(
                        total.start..tail.as_ref().map_or(then.span.end, |t| t.span.end),
                    );
                    tail = Some(match pat {
                        Some(pat) => if_let(pat, cond, then, tail.take(), span),
                        None => spanned(
                            ExprKind::If {
                                cond: Box::new(cond),
                                then,
                                otherwise: tail.take().map(Box::new),
                            },
                            span,
                        ),
                    });
                }
                BlockItem::Expr(tail.expect("an `if` has at least one branch"))
            });

        let while_ = just(Token::While)
            .ignore_then(test)
            .then(block.clone())
            .map_with(|((cond, pat), body), e| {
                BlockItem::Expr(match pat {
                    Some(pat) => while_let(pat, cond, body, e.span()),
                    None => spanned(
                        ExprKind::While {
                            cond: Box::new(cond),
                            body,
                        },
                        e.span(),
                    ),
                })
            });

        // `for a, b in items` loops over a temporary and binds the parts at the top of the body.
        let for_var = tuple_names()
            .or(ident()
                .separated_by(just(Token::Comma))
                .at_least(2)
                .collect::<Vec<_>>())
            .map_with(|names, e| {
                let at = rng(e.span());
                (
                    Ident {
                        node: format!("__item{}", at.start),
                        span: at,
                    },
                    names,
                )
            })
            .or(ident().map(|v| (v, Vec::new())));
        let for_ = just(Token::For)
            .ignore_then(for_var)
            .then_ignore(just(Token::In))
            .then(expr.clone())
            .then(block.clone())
            .map_with(|(((var, names), iter), mut body), e| {
                if !names.is_empty() {
                    let lets = tuple_lets(names, var.clone(), None, false, var.span.clone());
                    let mut stmts: Vec<Stmt> = lets.into_iter().map(Stmt::Let).collect();
                    stmts.append(&mut body.stmts);
                    body.stmts = stmts;
                }
                BlockItem::Expr(spanned(
                    ExprKind::For {
                        var,
                        iter: for_iter(iter),
                        body,
                    },
                    e.span(),
                ))
            });

        let case = just(Token::Case)
            .ignore_then(pattern)
            .then(block.clone())
            .map_with(|(pat, body), e| Arm {
                pat,
                body: spanned(ExprKind::Block(body), e.span()),
            });
        let match_ = just(Token::Match)
            .ignore_then(expr.clone())
            .then_ignore(just(Token::Colon))
            .then_ignore(just(Token::Newline))
            .then_ignore(just(Token::Indent))
            .then(case.repeated().at_least(1).collect::<Vec<_>>())
            .then_ignore(just(Token::Dedent))
            .map_with(|(scrutinee, arms), e| {
                BlockItem::Expr(spanned(
                    ExprKind::Match {
                        scrutinee: Box::new(scrutinee),
                        arms,
                    },
                    e.span(),
                ))
            });

        // `def add(a, b: int):` inside a function is a closure bound to a name.
        let local_param = ident().then(just(Token::Colon).ignore_then(type_()).or_not());
        let local_def = just(Token::Fn)
            .ignore_then(ident())
            .then(
                local_param
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .then_ignore(just(Token::Arrow).ignore_then(type_()).or_not())
            .then(block.clone())
            .map_with(|((name, params), mut body), e| {
                telvax_ast::tail::tailify_block(&mut body);
                let (names, types): (Vec<_>, Vec<_>) = params.into_iter().unzip();
                let body_span = body.span.clone();
                BlockItem::Let(Let {
                    mutable: false,
                    name,
                    ty: None,
                    init: spanned(
                        ExprKind::Closure {
                            params: names,
                            types,
                            body: Box::new(Spanned {
                                node: ExprKind::Block(body),
                                span: body_span,
                            }),
                        },
                        e.span(),
                    ),
                    span: rng(e.span()),
                })
            });

        let return_ = just(Token::Return)
            .ignore_then(expr.clone().or_not())
            .map_with(|v, e| BlockItem::Expr(spanned(ExprKind::Return(v.map(Box::new)), e.span())));
        let break_ =
            just(Token::Break).map_with(|_, e| BlockItem::Expr(spanned(ExprKind::Break, e.span())));
        let continue_ = just(Token::Continue)
            .map_with(|_, e| BlockItem::Expr(spanned(ExprKind::Continue, e.span())));
        let pass = just(Token::Pass)
            .or(just(Token::Ellipsis))
            .to(BlockItem::Nothing);

        // `let name: Type = value` and `var name = value`
        let binding = choice((just(Token::Let).to(false), just(Token::Var).to(true)));
        let binding_one = binding
            .then(ident())
            .then(just(Token::Colon).ignore_then(type_()).or_not())
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .map_with(|(((mutable, name), ty), init), e| {
                BlockItem::Let(Let {
                    mutable,
                    name,
                    ty,
                    init,
                    span: rng(e.span()),
                })
            });
        // `let a, b = value`
        let binding_many = binding
            .then(
                ident()
                    .separated_by(just(Token::Comma))
                    .at_least(2)
                    .allow_trailing()
                    .collect::<Vec<_>>(),
            )
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .map_with(|((mutable, names), init), e| {
                let span = rng(e.span());
                let temp = Ident {
                    node: format!("__tuple{}", span.start),
                    span: span.clone(),
                };
                BlockItem::Lets(tuple_lets(names, temp, Some(init), mutable, span))
            });
        // `target = value` and the compound forms, which desugar to `target = target op value`.
        let assign_op = select! {
            Token::Eq => None,
            Token::PlusEq => Some(BinOp::Add),
            Token::MinusEq => Some(BinOp::Sub),
            Token::StarEq => Some(BinOp::Mul),
            Token::SlashEq => Some(BinOp::Div),
        };
        let simple_expr =
            expr.clone()
                .then(assign_op.then(expr).or_not())
                .map_with(|(target, rest), e| {
                    BlockItem::Expr(match rest {
                        None => target,
                        Some((op, value)) => {
                            let value = match op {
                                None => value,
                                Some(op) => spanned(
                                    ExprKind::Binary {
                                        op,
                                        lhs: Box::new(target.clone()),
                                        rhs: Box::new(value.clone()),
                                    },
                                    SimpleSpan::from(target.span.start..value.span.end),
                                ),
                            };
                            spanned(
                                ExprKind::Assign {
                                    target: Box::new(target),
                                    value: Box::new(value),
                                },
                                e.span(),
                            )
                        }
                    })
                });

        let simple = choice((
            return_,
            break_,
            continue_,
            pass,
            binding_many,
            binding_one,
            simple_expr,
        ))
        .then_ignore(just(Token::Newline));

        choice((if_, while_, for_, match_, local_def, simple)).recover_with(skip_until(
            any().and_is(just(Token::Dedent).not()).ignored(),
            just(Token::Newline).ignored(),
            || BlockItem::Nothing,
        ))
    })
}

/// `range(n)` and `range(a, b)` are counting loops; anything else is walked as a list.
fn for_iter(iter: Expr) -> ForIter {
    if let ExprKind::Call { callee, args } = &iter.node
        && matches!(&callee.node, ExprKind::Path(p) if p.len() == 1 && p[0].node == "range")
        && (1..=2).contains(&args.len())
    {
        let mut args = args.clone();
        let end = args.pop().expect("one or two arguments");
        let start = args.pop().unwrap_or(Spanned {
            node: ExprKind::Lit(Lit::Int(0)),
            span: end.span.start..end.span.start,
        });
        return ForIter::Range {
            start: Box::new(start),
            end: Box::new(end),
            inclusive: false,
        };
    }
    ForIter::Expr(Box::new(iter))
}

/// What can sit in the body of a `class`.
#[derive(Clone)]
enum Member {
    Field(Field),
    Method(Fn),
    Nothing,
}

/// `public` unless the name starts with an underscore.
fn is_public(name: &str) -> bool {
    !name.starts_with('_')
}

fn program<'a, 's: 'a, I>(heads: Heads) -> impl Parser<'a, I, Program, Extra<'a, 's>>
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan> + 'a,
{
    let expr = expr(heads);
    let stmt = statement(expr.clone()).boxed();
    let body = block_of(stmt);

    // `import a.b.C` and `from a.b import C, D`
    // `as` is only a word here, so it stays a plain name everywhere else.
    let alias = select! { Token::Ident("as") => () }
        .ignore_then(ident())
        .or_not();
    let import = just(Token::Import)
        .ignore_then(path())
        .then(alias.clone())
        .then_ignore(just(Token::Newline))
        .map_with(|(path, alias), e| {
            Item::Import(Import {
                path,
                group: Vec::new(),
                aliases: vec![alias],
                span: rng(e.span()),
            })
        });
    let names = ident()
        .then(alias)
        .separated_by(just(Token::Comma))
        .at_least(1)
        .allow_trailing()
        .collect::<Vec<_>>();
    let from = just(Token::From)
        .ignore_then(path())
        .then_ignore(just(Token::Import))
        .then(
            names
                .clone()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or(names),
        )
        .then_ignore(just(Token::Newline))
        .map_with(|(path, names), e| {
            let (group, aliases) = names.into_iter().unzip();
            Item::Import(Import {
                path,
                group,
                aliases,
                span: rng(e.span()),
            })
        });

    // `@endpoint(POST, "/x")`, or with names: `@endpoint(method: POST, path: "/x")`. A bare word
    // is the HTTP method and a string is the path.
    let text_value = |name: &'static str| {
        move |value: Expr| -> Expr {
            match value.node {
                ExprKind::Path(ref p) if name == "method" && p.len() == 1 => Expr {
                    span: value.span.clone(),
                    node: ExprKind::Lit(Lit::Str(p[0].node.clone())),
                },
                _ => value,
            }
        }
    };
    let attr_named = ident()
        .then_ignore(just(Token::Colon))
        .then(expr.clone())
        .map(move |(key, value)| {
            let fix = text_value(if key.node == "method" { "method" } else { "" });
            (key, fix(value))
        });
    let attr_method = ident().map_with(|word, e| {
        let span = rng(e.span());
        (
            Ident {
                node: "method".into(),
                span: span.clone(),
            },
            Expr {
                node: ExprKind::Lit(Lit::Str(word.node)),
                span,
            },
        )
    });
    let attr_path = select! { Token::Str(s) => s }.map_with(|text, e| {
        let span = rng(e.span());
        (
            Ident {
                node: "path".into(),
                span: span.clone(),
            },
            Expr {
                node: ExprKind::Lit(Lit::Str(unescape(&text[1..text.len() - 1]))),
                span,
            },
        )
    });
    let attr = just(Token::At)
        .ignore_then(ident())
        .then(
            choice((attr_named, attr_path, attr_method))
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .then_ignore(just(Token::Newline))
        .map_with(|(name, args), e| Attribute {
            name,
            args: args.unwrap_or_default(),
            span: rng(e.span()),
        });

    // `self` takes the type of the `class` it sits in.
    let self_param = word("self").map_with(|(), e| {
        let span = rng(e.span());
        Param {
            mutable: true,
            name: Ident {
                node: "self".into(),
                span: span.clone(),
            },
            ty: Type {
                path: vec![Ident {
                    node: "Self".into(),
                    span: span.clone(),
                }],
                args: Vec::new(),
                span,
            },
            default: None,
        }
    });
    let param = self_param.or(ident()
        .then_ignore(just(Token::Colon))
        .then(type_())
        .then(just(Token::Eq).ignore_then(expr.clone()).or_not())
        .map(|((name, ty), default)| Param {
            mutable: true,
            name,
            ty,
            default,
        }));

    let func = attr
        .repeated()
        .collect::<Vec<_>>()
        .then_ignore(just(Token::Fn))
        .then(ident())
        .then(generics())
        .then(
            param
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(just(Token::Arrow).ignore_then(type_()).or_not())
        // A body, or `: ...` for a function a trait requires.
        .then(
            body.map(Some).or(just(Token::Colon)
                .ignore_then(just(Token::Ellipsis))
                .ignore_then(just(Token::Newline))
                .to(None)),
        )
        .map_with(|(((((attrs, name), generics), params), ret), body), e| {
            let span = rng(e.span());
            Fn {
                attrs,
                is_pub: is_public(&name.node),
                name,
                generics,
                params,
                ret,
                required: body.is_none(),
                body: body.unwrap_or(Block {
                    stmts: Vec::new(),
                    tail: None,
                    span: span.clone(),
                }),
                span,
            }
        });

    // The members of a block that holds only functions (`trait`, `extend`).
    let fns_block = func
        .clone()
        .repeated()
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(
            just(Token::Colon)
                .then(just(Token::Newline))
                .then(just(Token::Indent)),
            just(Token::Dedent),
        );

    // `class Name[T](Trait):` with fields (`x: float`) and methods in the body.
    let field = ident()
        .then_ignore(just(Token::Colon))
        .then(type_())
        .then(just(Token::Eq).ignore_then(expr.clone()).or_not())
        .then_ignore(just(Token::Newline))
        .map(|((name, ty), default)| Member::Field(Field { name, ty, default }));
    let member = choice((
        func.clone().map(Member::Method),
        field,
        just(Token::Pass)
            .then(just(Token::Newline))
            .to(Member::Nothing),
    ));
    let class = word("type")
        .ignore_then(ident())
        .then(generics())
        .then(
            ident()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .then(
            member
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(
                    just(Token::Colon)
                        .then(just(Token::Newline))
                        .then(just(Token::Indent)),
                    just(Token::Dedent),
                ),
        )
        .map_with(|(((name, generics), trait_name), members), e| {
            let span = rng(e.span());
            let mut fields = Vec::new();
            let mut fns = Vec::new();
            for m in members {
                match m {
                    Member::Field(f) => fields.push(f),
                    Member::Method(f) => fns.push(f),
                    Member::Nothing => {}
                }
            }
            let mut items = vec![Item::Struct(Struct {
                is_pub: is_public(&name.node),
                name: name.clone(),
                generics: generics.clone(),
                fields,
                span: span.clone(),
            })];
            if !fns.is_empty() || trait_name.is_some() {
                items.push(Item::Impl(Impl {
                    ty_args: type_params_as_args(&generics),
                    generics,
                    ty: name,
                    trait_name,
                    fns,
                    span,
                }));
            }
            items
        });

    // `extend Name[T](Trait):` adds methods to a type declared elsewhere.
    let extend = just(Token::Extend)
        .ignore_then(ident())
        .then(generics())
        .then(
            ident()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .then(fns_block.clone())
        .map_with(|(((ty, generics), trait_name), fns), e| {
            vec![Item::Impl(Impl {
                ty_args: type_params_as_args(&generics),
                generics,
                ty,
                trait_name,
                fns,
                span: rng(e.span()),
            })]
        });

    let trait_ = just(Token::Trait)
        .ignore_then(ident())
        .then(fns_block)
        .map_with(|(name, fns), e| {
            vec![Item::Trait(Trait {
                is_pub: is_public(&name.node),
                name,
                fns,
                span: rng(e.span()),
            })]
        });

    // `enum Shape:` followed by one variant per line.
    let variant = ident()
        .then(
            type_()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .then_ignore(just(Token::Newline))
        .map(|(name, fields)| Variant {
            name,
            fields: fields.unwrap_or_default(),
        });
    let enum_ = just(Token::Enum)
        .ignore_then(ident())
        .then(generics())
        .then(
            variant
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(
                    just(Token::Colon)
                        .then(just(Token::Newline))
                        .then(just(Token::Indent)),
                    just(Token::Dedent),
                ),
        )
        .map_with(|((name, generics), variants), e| {
            vec![Item::Enum(Enum {
                is_pub: is_public(&name.node),
                name,
                generics,
                variants,
                span: rng(e.span()),
            })]
        });

    // A name set at the top of the file: `embedder = OllamaEmbedder.new("x")`.
    let global = just(Token::Let)
        .ignore_then(ident())
        .then(just(Token::Colon).ignore_then(type_()).or_not())
        .then_ignore(just(Token::Eq))
        .then(expr)
        .then_ignore(just(Token::Newline))
        .map_with(|((name, ty), init), e| {
            vec![Item::Let(Let {
                mutable: false,
                name,
                ty,
                init,
                span: rng(e.span()),
            })]
        });

    let item_start = one_of([
        Token::Import,
        Token::From,
        Token::Enum,
        Token::Extend,
        Token::Trait,
        Token::Fn,
        Token::Let,
        Token::At,
    ])
    .ignored()
    .or(word("type"));

    choice((
        import.map(|i| vec![i]),
        from.map(|i| vec![i]),
        class,
        enum_,
        extend,
        trait_,
        func.map(|f| vec![Item::Fn(f)]),
        global,
    ))
    .recover_with(skip_then_retry_until(any().ignored(), item_start))
    .repeated()
    .collect::<Vec<_>>()
    .then_ignore(end())
    .map(|groups| {
        let mut items: Vec<Item> = groups.into_iter().flatten().collect();
        for item in &mut items {
            match item {
                Item::Fn(f) => telvax_ast::tail::finish_fn(f),
                Item::Impl(i) => i.fns.iter_mut().for_each(telvax_ast::tail::finish_fn),
                Item::Trait(t) => t.fns.iter_mut().for_each(telvax_ast::tail::finish_fn),
                _ => {}
            }
        }
        Program { items }
    })
}

/// `Stack[T]` in `class Stack[T]`: the type arguments the `impl` names are the parameters.
fn type_params_as_args(generics: &[GenericParam]) -> Vec<Type> {
    generics
        .iter()
        .map(|g| Type {
            path: vec![g.name.clone()],
            args: Vec::new(),
            span: g.name.span.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests;
