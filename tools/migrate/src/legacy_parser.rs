//! Parser for Telvax: token stream -> AST.

use crate::legacy_lexer::Token;
use chumsky::{
    input::{Stream, ValueInput},
    prelude::*,
};
use telvax_ast::*;
use telvax_ast::{Span, Spanned};
use telvax_diagnostics::Diagnostic;

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

/// Turns a parser error into a readable diagnostic. A missing `;` is reported where it belongs
/// (after the previous token), and long lists of acceptable tokens collapse to what was found.
fn friendly(e: &Rich<'_, Token<'_>>, src: &str, base: usize) -> Diagnostic {
    use chumsky::error::RichPattern;
    let span = e.span().into_range();
    let expects_semi = e
        .expected()
        .any(|p| matches!(p, RichPattern::Token(t) if **t == Token::Semi));
    if expects_semi && span.start > base {
        // The previous token ends at the last non-blank character before the offending one.
        let before = &src[..(span.start - base).min(src.len())];
        let trimmed = before.trim_end();
        if !trimmed.is_empty() {
            let end = base + trimmed.len();
            let last = trimmed.chars().next_back().map_or(1, char::len_utf8);
            return Diagnostic::error(end - last..end, "expected `;` after this").with_note(
                "statements end with `;`; only the last expression of a block may omit it",
            );
        }
    }
    if e.expected().count() > 5
        && let Some(found) = e.found()
    {
        return Diagnostic::error(span, format!("unexpected `{found}`"));
    }
    Diagnostic::error(span, e.to_string())
}

/// Lexes and parses `src`. Returns the program (when recovery allowed one) and all errors.
pub fn parse(src: &str) -> (Option<Program>, Vec<Diagnostic>) {
    parse_at(src, 0)
}

/// Like [`parse`], with every span shifted by `base`. Lets several files share one offset space,
/// so a single span identifies a place in a whole multi-file program.
pub fn parse_at(src: &str, base: usize) -> (Option<Program>, Vec<Diagnostic>) {
    let (mut tokens, lex_errors) = crate::legacy_lexer::tokenize(src);
    for (_, span) in &mut tokens {
        *span = span.start + base..span.end + base;
    }
    let mut errors: Vec<Diagnostic> = lex_errors
        .into_iter()
        .map(|span| Diagnostic::error(span.start + base..span.end + base, "invalid token"))
        .collect();
    let (program, parse_errors) = run!(program(), tokens, src.len() + base, src, base);
    errors.extend(parse_errors);
    errors.sort_by_key(|e| e.span.start);
    (program, errors)
}

/// Parses a single expression (used for f-string interpolations and tests).
pub fn parse_expr(src: &str) -> (Option<Expr>, Vec<Diagnostic>) {
    parse_expr_at(src, 0)
}

fn parse_expr_at(src: &str, offset: usize) -> (Option<Expr>, Vec<Diagnostic>) {
    let (mut tokens, lex_errors) = crate::legacy_lexer::tokenize(src);
    for (_, span) in &mut tokens {
        *span = span.start + offset..span.end + offset;
    }
    let mut errors: Vec<Diagnostic> = lex_errors
        .into_iter()
        .map(|s| Diagnostic::error(s.start + offset..s.end + offset, "invalid token"))
        .collect();
    let (expr, parse_errors) = run!(
        expr().then_ignore(end()),
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

fn path<'a, 's: 'a, I>() -> impl Parser<'a, I, Vec<Ident>, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    ident()
        .separated_by(just(Token::PathSep))
        .at_least(1)
        .collect::<Vec<_>>()
}

fn type_<'a, 's: 'a, I>() -> impl Parser<'a, I, Type, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    recursive(|ty| {
        let dim = select! { Token::Int(n) => n }
            .try_map(|n, span| {
                n.replace('_', "")
                    .parse::<u64>()
                    .map(Dim::Lit)
                    .map_err(|_| Rich::custom(span, "integer literal out of range"))
            })
            .or(ident().map(Dim::Named));
        let shape = dim
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(TypeArg::Shape);
        let row = ident()
            .then_ignore(just(Token::Colon))
            .then(ty.clone())
            .map(|(name, ty)| Field { name, ty })
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(TypeArg::Row);
        let arg = ty.clone().map(TypeArg::Type).or(shape).or(row);
        let args = arg
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::Lt), just(Token::Gt));
        // `(i64, String)`: a tuple type is spelled with a name no program can write.
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
        tuple_ty.or(path().then(args.or_not()).map_with(|(path, args), e| Type {
            path,
            args: args.unwrap_or_default(),
            span: rng(e.span()),
        }))
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
fn fstring_parts(body: &str, offset: usize) -> Result<Vec<FPart>, Diagnostic> {
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
                match parse_expr_at(inner, offset + start) {
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

enum BlockItem {
    Let(Let),
    /// `let (a, b) = value;` becomes a temporary and one `let` per name.
    Lets(Vec<Let>),
    Expr(Expr, bool),
}

fn is_block_like(e: &Expr) -> bool {
    matches!(
        e.node,
        ExprKind::If { .. }
            | ExprKind::Block(_)
            | ExprKind::While { .. }
            | ExprKind::For { .. }
            | ExprKind::Match { .. }
    )
}

fn let_stmt<'a, 's: 'a, I>(
    expr: impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone,
) -> impl Parser<'a, I, Let, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    just(Token::Let)
        .ignore_then(just(Token::Mut).or_not().map(|m| m.is_some()))
        .then(ident())
        .then(just(Token::Colon).ignore_then(type_()).or_not())
        .then_ignore(just(Token::Eq))
        .then(expr)
        .then_ignore(just(Token::Semi))
        .map_with(|(((mutable, name), ty), init), e| Let {
            mutable,
            name,
            ty,
            init,
            span: rng(e.span()),
        })
}

/// `<T, U: Shape + Other>`: the type parameters of a function, type or `impl`.
fn generics<'a, 's: 'a, I>() -> impl Parser<'a, I, Vec<GenericParam>, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    ident()
        .then(
            just(Token::Colon)
                .ignore_then(
                    ident()
                        .separated_by(just(Token::Plus))
                        .at_least(1)
                        .collect::<Vec<_>>(),
                )
                .or_not(),
        )
        .map(|(name, bounds)| GenericParam {
            name,
            bounds: bounds.unwrap_or_default(),
        })
        .separated_by(just(Token::Comma))
        .allow_trailing()
        .at_least(1)
        .collect::<Vec<_>>()
        .delimited_by(just(Token::Lt), just(Token::Gt))
        .or_not()
        .map(Option::unwrap_or_default)
}

/// `(a, b, _)`: names for the parts of a tuple (`_` skips one).
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

fn let_tuple_stmt<'a, 's: 'a, I>(
    expr: impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone,
) -> impl Parser<'a, I, Vec<Let>, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    just(Token::Let)
        .ignore_then(just(Token::Mut).or_not().map(|m| m.is_some()))
        .then(tuple_names())
        .then_ignore(just(Token::Eq))
        .then(expr)
        .then_ignore(just(Token::Semi))
        .map_with(|((mutable, names), init), e| {
            let span = rng(e.span());
            let temp = Ident {
                node: format!("__tuple{}", span.start),
                span: span.clone(),
            };
            tuple_lets(names, temp, Some(init), mutable, span)
        })
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

fn expr<'a, 's: 'a, I>() -> impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    recursive(|expr| {
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
        let fstring = select! { Token::FStr(s) => s }
            .map_with(|s, e| (s, rng(e.span()).start))
            .try_map(|(s, start), _| {
                // Strip `f"` and the closing quote.
                fstring_parts(&s[2..s.len() - 1], start + 2)
                    .map(ExprKind::FString)
                    .map_err(|e| Rich::custom(SimpleSpan::from(e.span), e.message))
            });

        let block = block(expr.clone());

        let pattern = pattern();

        let if_ = recursive(|if_| {
            let else_branch = just(Token::Else).ignore_then(
                if_.clone().or(block
                    .clone()
                    .map_with(|b, e| spanned(ExprKind::Block(b), e.span()))),
            );
            // `if let PATTERN = value { .. } else { .. }` is a `match` with a catch-all arm.
            let if_let = just(Token::If)
                .ignore_then(just(Token::Let))
                .ignore_then(pattern.clone())
                .then_ignore(just(Token::Eq))
                .then(expr.clone())
                .then(block.clone())
                .then(else_branch.or_not())
                .map_with(|(((pat, scrutinee), then), otherwise), e| {
                    let span = e.span();
                    let unit = || {
                        spanned(
                            ExprKind::Block(Block {
                                stmts: Vec::new(),
                                tail: None,
                                span: rng(span),
                            }),
                            span,
                        )
                    };
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
                                    body: otherwise.unwrap_or_else(unit),
                                },
                            ],
                        },
                        span,
                    )
                });
            let plain_if = just(Token::If)
                .ignore_then(expr.clone())
                .then(block.clone())
                .then(
                    just(Token::Else)
                        .ignore_then(
                            if_.or(block
                                .clone()
                                .map_with(|b, e| spanned(ExprKind::Block(b), e.span()))),
                        )
                        .or_not(),
                )
                .map_with(|((cond, then), otherwise), e| {
                    spanned(
                        ExprKind::If {
                            cond: Box::new(cond),
                            then,
                            otherwise: otherwise.map(Box::new),
                        },
                        e.span(),
                    )
                });
            if_let.or(plain_if)
        });

        // A closure parameter is a name (optionally typed) or a tuple of names: `|(i, s)| ..`.
        let closure_param = ident()
            .then(just(Token::Colon).ignore_then(type_()).or_not())
            .map(|(name, ty)| (name, ty, Vec::new()))
            .or(tuple_names().map_with(|names, e| {
                let at = rng(e.span());
                (
                    Ident {
                        node: format!("__arg{}", at.start),
                        span: at,
                    },
                    None,
                    names,
                )
            }));
        let closure_params = choice((
            just(Token::OrOr).to(Vec::new()),
            closure_param
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::Bar), just(Token::Bar)),
        ));
        let closure = closure_params
            .then(expr.clone())
            .map_with(|(params, body), e| {
                let mut names = Vec::new();
                let mut types = Vec::new();
                let mut lets = Vec::new();
                for (name, ty, parts) in params {
                    if !parts.is_empty() {
                        let span = name.span.clone();
                        lets.extend(tuple_lets(parts, name.clone(), None, false, span));
                    }
                    names.push(name);
                    types.push(ty);
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
                ExprKind::Closure {
                    params: names,
                    types,
                    body: Box::new(body),
                }
            });

        let return_ = just(Token::Return)
            .ignore_then(expr.clone().or_not())
            .map(|v| ExprKind::Return(v.map(Box::new)));

        let break_ = just(Token::Break).to(ExprKind::Break);
        let continue_ = just(Token::Continue).to(ExprKind::Continue);

        let struct_lit = path()
            .filter(|p| {
                p.last()
                    .is_some_and(|i| i.node.starts_with(|c: char| c.is_uppercase()))
            })
            .then(
                ident()
                    .then_ignore(just(Token::Colon))
                    .then(expr.clone())
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(|(path, fields)| ExprKind::StructLit { path, fields });

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

        // `while let PATTERN = value { body }` loops until the pattern stops matching.
        let while_let = just(Token::While)
            .ignore_then(just(Token::Let))
            .ignore_then(pattern.clone())
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then(block.clone())
            .map_with(|((pat, scrutinee), body), e| {
                let span = e.span();
                let body_span = body.span.clone();
                let leave = Block {
                    stmts: vec![Stmt::Expr(spanned(ExprKind::Break, span))],
                    tail: None,
                    span: rng(span),
                };
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
            });
        let while_plain = just(Token::While)
            .ignore_then(expr.clone())
            .then(block.clone())
            .map_with(|(cond, body), e| {
                spanned(
                    ExprKind::While {
                        cond: Box::new(cond),
                        body,
                    },
                    e.span(),
                )
            });
        let while_ = while_let.or(while_plain);

        let for_iter = expr
            .clone()
            .then(
                choice((
                    just(Token::DotDot).to(false),
                    just(Token::DotDotEq).to(true),
                ))
                .then(expr.clone())
                .or_not(),
            )
            .map(|(first, range)| match range {
                Some((inclusive, end)) => ForIter::Range {
                    start: Box::new(first),
                    end: Box::new(end),
                    inclusive,
                },
                None => ForIter::Expr(Box::new(first)),
            });
        // `for (a, b) in items` loops over a temporary and binds the parts at the top of the body.
        let for_var = ident()
            .map(|v| (v, Vec::new()))
            .or(tuple_names().map_with(|names, e| {
                let at = rng(e.span());
                (
                    Ident {
                        node: format!("__item{}", at.start),
                        span: at,
                    },
                    names,
                )
            }));
        let for_ = just(Token::For)
            .ignore_then(for_var)
            .then_ignore(just(Token::In))
            .then(for_iter)
            .then(block.clone())
            .map_with(|(((var, names), iter), mut body), e| {
                if !names.is_empty() {
                    let lets = tuple_lets(names, var.clone(), None, false, var.span.clone());
                    let mut stmts: Vec<Stmt> = lets.into_iter().map(Stmt::Let).collect();
                    stmts.append(&mut body.stmts);
                    body.stmts = stmts;
                }
                spanned(ExprKind::For { var, iter, body }, e.span())
            });

        let match_ = just(Token::Match)
            .ignore_then(expr.clone())
            .then(
                pattern
                    .then_ignore(just(Token::FatArrow))
                    .then(expr.clone())
                    .map(|(pat, body)| Arm { pat, body })
                    .then_ignore(just(Token::Comma).or_not())
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map_with(|(scrutinee, arms), e| {
                spanned(
                    ExprKind::Match {
                        scrutinee: Box::new(scrutinee),
                        arms,
                    },
                    e.span(),
                )
            });

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
            closure,
            return_,
            break_,
            continue_,
            struct_lit,
            path().map(ExprKind::Path),
            paren,
            array,
        ))
        .map_with(|k, e| spanned(k, e.span()))
        .or(if_)
        .or(while_)
        .or(for_)
        .or(match_)
        .or(block.map_with(|b, e| spanned(ExprKind::Block(b), e.span())));

        #[derive(Clone)]
        enum Postfix {
            Call(Vec<Expr>),
            Field(Ident),
            Index(Expr),
            Try,
            Fields(Vec<Ident>),
        }
        let call = expr
            .clone()
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
                    Postfix::Fields(
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
        let index = expr
            .clone()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(Postfix::Index);
        let postfix = primary.foldl_with(
            call.or(field).or(index).or(try_).repeated(),
            |base, op, e| {
                let kind = match op {
                    Postfix::Index(index) => ExprKind::Index {
                        base: Box::new(base),
                        index: Box::new(index),
                    },
                    Postfix::Call(args) => ExprKind::Call {
                        callee: Box::new(base),
                        args,
                    },
                    Postfix::Try => ExprKind::Try(Box::new(base)),
                    Postfix::Fields(names) => {
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
                    Postfix::Field(name) => ExprKind::Field {
                        base: Box::new(base),
                        name,
                    },
                };
                spanned(kind, e.span())
            },
        );

        let unary = select! { Token::Minus => UnOp::Neg, Token::Bang => UnOp::Not }
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
        let cmp = binary(
            add,
            select! {
                Token::Lt => BinOp::Lt, Token::LtEq => BinOp::LtEq,
                Token::Gt => BinOp::Gt, Token::GtEq => BinOp::GtEq,
            }
            .boxed(),
        );
        let eq = binary(
            cmp,
            select! { Token::EqEq => BinOp::Eq, Token::NotEq => BinOp::NotEq }.boxed(),
        );
        let and = binary(eq, select! { Token::AndAnd => BinOp::And }.boxed());
        let or = binary(and, select! { Token::OrOr => BinOp::Or }.boxed());

        let pipe = or.clone().foldl_with(
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
        );

        // `target = value` and the compound forms, which desugar to `target = target op value`.
        let assign_op = select! {
            Token::Eq => None,
            Token::PlusEq => Some(BinOp::Add),
            Token::MinusEq => Some(BinOp::Sub),
            Token::StarEq => Some(BinOp::Mul),
            Token::SlashEq => Some(BinOp::Div),
        };
        pipe.then(assign_op.then(expr.clone()).or_not())
            .map_with(|(target, rest), e| match rest {
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
                            (target.span.start..value.span.end).into(),
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
    })
}

fn block<'a, 's: 'a, I>(
    expr: impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone,
) -> impl Parser<'a, I, Block, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    let item = let_stmt(expr.clone())
        .map(BlockItem::Let)
        .or(let_tuple_stmt(expr.clone()).map(BlockItem::Lets))
        .or(expr
            .then(just(Token::Semi).or_not())
            .map(|(e, semi)| BlockItem::Expr(e, semi.is_some())))
        .recover_with(skip_until(
            any().and_is(just(Token::RBrace).not()).ignored(),
            just(Token::Semi).ignored(),
            || BlockItem::Expr(spanned(ExprKind::Error, (0..0).into()), true),
        ));
    item.repeated()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .try_map(|items, span| {
            let n = items.len();
            let mut stmts = Vec::new();
            let mut tail = None;
            for (i, item) in items.into_iter().enumerate() {
                match item {
                    BlockItem::Let(l) => stmts.push(Stmt::Let(l)),
                    BlockItem::Lets(ls) => stmts.extend(ls.into_iter().map(Stmt::Let)),
                    BlockItem::Expr(e, true) => stmts.push(Stmt::Expr(e)),
                    BlockItem::Expr(e, false) if i + 1 == n => tail = Some(Box::new(e)),
                    BlockItem::Expr(e, false) if is_block_like(&e) => stmts.push(Stmt::Expr(e)),
                    BlockItem::Expr(e, false) => {
                        return Err(Rich::custom(
                            SimpleSpan::from(e.span.end..e.span.end),
                            "expected `;` after expression",
                        ));
                    }
                }
            }
            Ok(Block {
                stmts,
                tail,
                span: span.into_range(),
            })
        })
        .recover_with(via_parser(nested_delimiters(
            Token::LBrace,
            Token::RBrace,
            [
                (Token::LParen, Token::RParen),
                (Token::LBracket, Token::RBracket),
            ],
            |span: SimpleSpan| Block {
                stmts: Vec::new(),
                tail: None,
                span: span.into_range(),
            },
        )))
}

fn program<'a, 's: 'a, I>() -> impl Parser<'a, I, Program, Extra<'a, 's>>
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    let expr = expr();

    let import = just(Token::Import)
        .ignore_then(path())
        .then(
            just(Token::PathSep)
                .ignore_then(
                    ident()
                        .separated_by(just(Token::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace)),
                )
                .or_not(),
        )
        .then_ignore(just(Token::Semi))
        .map_with(|(path, group), e| {
            Item::Import(Import {
                path,
                group: group.unwrap_or_default(),
                span: rng(e.span()),
            })
        });

    let struct_ = just(Token::Pub)
        .or_not()
        .then_ignore(just(Token::Struct))
        .then(ident())
        .then(generics())
        .then(
            ident()
                .then_ignore(just(Token::Colon))
                .then(type_())
                .map(|(name, ty)| Field { name, ty })
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|(((is_pub, name), generics), fields), e| {
            Item::Struct(Struct {
                is_pub: is_pub.is_some(),
                name,
                generics,
                fields,
                span: rng(e.span()),
            })
        });

    let enum_ = just(Token::Pub)
        .or_not()
        .then_ignore(just(Token::Enum))
        .then(ident())
        .then(generics())
        .then(
            ident()
                .then(
                    type_()
                        .separated_by(just(Token::Comma))
                        .allow_trailing()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LParen), just(Token::RParen))
                        .or_not(),
                )
                .map(|(name, fields)| Variant {
                    name,
                    fields: fields.unwrap_or_default(),
                })
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|(((is_pub, name), generics), variants), e| {
            Item::Enum(Enum {
                is_pub: is_pub.is_some(),
                name,
                generics,
                variants,
                span: rng(e.span()),
            })
        });

    let attr = just(Token::At)
        .ignore_then(ident())
        .then(
            ident()
                .then_ignore(just(Token::Eq))
                .then(expr.clone())
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .map_with(|(name, args), e| Attribute {
            name,
            args: args.unwrap_or_default(),
            span: rng(e.span()),
        });

    // `self` and `mut self` take the type of the `impl` they sit in.
    let self_param = just(Token::Mut)
        .or_not()
        .then(select! { Token::Ident(s) if s == "self" => s.to_string() })
        .map_with(|(m, name), e| {
            let span = rng(e.span());
            Param {
                mutable: m.is_some(),
                name: Ident {
                    node: name,
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
            }
        });
    let param = self_param.or(just(Token::Mut)
        .or_not()
        .then(ident())
        .then_ignore(just(Token::Colon))
        .then(type_())
        .map(|((m, name), ty)| Param {
            mutable: m.is_some(),
            name,
            ty,
        }));

    let func = attr
        .repeated()
        .collect::<Vec<_>>()
        .then(just(Token::Pub).or_not())
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
        .then(block(expr.clone()).map(Some).or(just(Token::Semi).to(None)))
        .map_with(
            |((((((attrs, is_pub), name), generics), params), ret), body), e| {
                let span = rng(e.span());
                Item::Fn(Fn {
                    attrs,
                    is_pub: is_pub.is_some(),
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
                })
            },
        );

    let fns_block = func
        .clone()
        .repeated()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LBrace), just(Token::RBrace))
        .map(|items| {
            items
                .into_iter()
                .filter_map(|i| match i {
                    Item::Fn(f) => Some(f),
                    _ => None,
                })
                .collect::<Vec<_>>()
        });
    // `impl Type { .. }` or `impl Trait for Type { .. }`.
    // The type arguments after a name: `Stack<T>`.
    let type_args = type_()
        .separated_by(just(Token::Comma))
        .at_least(1)
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::Lt), just(Token::Gt))
        .or_not()
        .map(Option::unwrap_or_default);
    let impl_ = just(Token::Impl)
        .ignore_then(generics())
        .then(ident())
        .then(type_args.clone())
        .then(
            just(Token::For)
                .ignore_then(ident())
                .then(type_args)
                .or_not(),
        )
        .then(fns_block.clone())
        .map_with(|((((generics, first), first_args), second), fns), e| {
            let (ty, ty_args, trait_name) = match second {
                Some((ty, args)) => (ty, args, Some(first)),
                None => (first, first_args, None),
            };
            Item::Impl(Impl {
                generics,
                ty,
                ty_args,
                trait_name,
                fns,
                span: rng(e.span()),
            })
        });
    let trait_ = just(Token::Pub)
        .or_not()
        .then_ignore(just(Token::Trait))
        .then(ident())
        .then(fns_block)
        .map_with(|((is_pub, name), fns), e| {
            Item::Trait(Trait {
                is_pub: is_pub.is_some(),
                name,
                fns,
                span: rng(e.span()),
            })
        });

    let global_let = let_stmt(expr).map(Item::Let);

    let item_start = one_of([
        Token::Import,
        Token::Struct,
        Token::Enum,
        Token::Impl,
        Token::Trait,
        Token::Fn,
        Token::Pub,
        Token::At,
        Token::Let,
    ])
    .ignored();

    choice((import, struct_, enum_, impl_, trait_, func, global_let))
        .recover_with(skip_then_retry_until(any().ignored(), item_start))
        .repeated()
        .collect::<Vec<_>>()
        .then_ignore(end())
        .map(|items| Program { items })
}
