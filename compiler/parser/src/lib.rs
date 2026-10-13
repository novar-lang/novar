//! Parser for Novar: token stream -> AST.

use chumsky::{
    input::{Stream, ValueInput},
    prelude::*,
};
use novar_ast::*;
use novar_ast::{Span, Spanned};
use novar_lexer::Token;

macro_rules! run {
    ($parser:expr, $tokens:expr, $eoi:expr) => {{
        let iter = $tokens.into_iter().map(|(t, s)| (t, SimpleSpan::from(s)));
        let stream = Stream::from_iter(iter).map(SimpleSpan::from($eoi..$eoi), |(t, s)| (t, s));
        let (out, errs) = $parser.parse(stream).into_output_errors();
        let errs: Vec<ParseError> = errs
            .into_iter()
            .map(|e| ParseError {
                span: e.span().into_range(),
                message: e.to_string(),
            })
            .collect();
        (out, errs)
    }};
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

/// Lexes and parses `src`. Returns the program (when recovery allowed one) and all errors.
pub fn parse(src: &str) -> (Option<Program>, Vec<ParseError>) {
    let (tokens, lex_errors) = novar_lexer::tokenize(src);
    let mut errors: Vec<ParseError> = lex_errors
        .into_iter()
        .map(|span| ParseError {
            span,
            message: "invalid token".into(),
        })
        .collect();
    let (program, parse_errors) = run!(program(), tokens, src.len());
    errors.extend(parse_errors);
    errors.sort_by_key(|e| e.span.start);
    (program, errors)
}

/// Parses a single expression (used for f-string interpolations and tests).
pub fn parse_expr(src: &str) -> (Option<Expr>, Vec<ParseError>) {
    parse_expr_at(src, 0)
}

fn parse_expr_at(src: &str, offset: usize) -> (Option<Expr>, Vec<ParseError>) {
    let (mut tokens, lex_errors) = novar_lexer::tokenize(src);
    for (_, span) in &mut tokens {
        *span = span.start + offset..span.end + offset;
    }
    let mut errors: Vec<ParseError> = lex_errors
        .into_iter()
        .map(|s| ParseError {
            span: s.start + offset..s.end + offset,
            message: "invalid token".into(),
        })
        .collect();
    let (expr, parse_errors) = run!(expr().then_ignore(end()), tokens, src.len() + offset);
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
        let arg = ty.map(TypeArg::Type).or(shape);
        let args = arg
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::Lt), just(Token::Gt));
        path().then(args.or_not()).map_with(|(path, args), e| Type {
            path,
            args: args.unwrap_or_default(),
            span: rng(e.span()),
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
fn fstring_parts(body: &str, offset: usize) -> Result<Vec<FPart>, ParseError> {
    let err = |start: usize, end: usize, message: &str| ParseError {
        span: offset + start..offset + end,
        message: message.into(),
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
    Expr(Expr, bool),
}

fn is_block_like(e: &Expr) -> bool {
    matches!(e.node, ExprKind::If { .. } | ExprKind::Block(_))
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

        let if_ = recursive(|if_| {
            just(Token::If)
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
                })
        });

        let closure_params = choice((
            just(Token::OrOr).to(Vec::new()),
            ident()
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::Bar), just(Token::Bar)),
        ));
        let closure = closure_params
            .then(expr.clone())
            .map(|(params, body)| ExprKind::Closure {
                params,
                body: Box::new(body),
            });

        let return_ = just(Token::Return)
            .ignore_then(expr.clone().or_not())
            .map(|v| ExprKind::Return(v.map(Box::new)));

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

        let paren = expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(|e| e.node);

        let primary = choice((
            float,
            int,
            string,
            fstring,
            boolean,
            closure,
            return_,
            struct_lit,
            path().map(ExprKind::Path),
            paren,
        ))
        .map_with(|k, e| spanned(k, e.span()))
        .or(if_)
        .or(block.map_with(|b, e| spanned(ExprKind::Block(b), e.span())));

        enum Postfix {
            Call(Vec<Expr>),
            Field(Ident),
        }
        let call = expr
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(Postfix::Call);
        let field = just(Token::Dot).ignore_then(ident()).map(Postfix::Field);
        let postfix = primary.foldl_with(call.or(field).repeated(), |base, op, e| {
            let kind = match op {
                Postfix::Call(args) => ExprKind::Call {
                    callee: Box::new(base),
                    args,
                },
                Postfix::Field(name) => ExprKind::Field {
                    base: Box::new(base),
                    name,
                },
            };
            spanned(kind, e.span())
        });

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

        or.clone().foldl_with(
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
    })
}

fn block<'a, 's: 'a, I>(
    expr: impl Parser<'a, I, Expr, Extra<'a, 's>> + Clone,
) -> impl Parser<'a, I, Block, Extra<'a, 's>> + Clone
where
    I: ValueInput<'a, Token = Token<'s>, Span = SimpleSpan>,
{
    let item = let_stmt(expr.clone()).map(BlockItem::Let).or(expr
        .then(just(Token::Semi).or_not())
        .map(|(e, semi)| BlockItem::Expr(e, semi.is_some())));
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
        .map_with(|((is_pub, name), fields), e| {
            Item::Struct(Struct {
                is_pub: is_pub.is_some(),
                name,
                fields,
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

    let param = ident()
        .then_ignore(just(Token::Colon))
        .then(type_())
        .map(|(name, ty)| Param { name, ty });

    let func = attr
        .repeated()
        .collect::<Vec<_>>()
        .then(just(Token::Pub).or_not())
        .then_ignore(just(Token::Fn))
        .then(ident())
        .then(
            param
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(just(Token::Arrow).ignore_then(type_()).or_not())
        .then(block(expr.clone()))
        .map_with(|(((((attrs, is_pub), name), params), ret), body), e| {
            Item::Fn(Fn {
                attrs,
                is_pub: is_pub.is_some(),
                name,
                params,
                ret,
                body,
                span: rng(e.span()),
            })
        });

    let global_let = let_stmt(expr).map(Item::Let);

    choice((import, struct_, func, global_let))
        .repeated()
        .collect::<Vec<_>>()
        .then_ignore(end())
        .map(|items| Program { items })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_expr(src: &str) -> ExprKind {
        let (e, errs) = parse_expr(src);
        assert!(errs.is_empty(), "{src}: {errs:?}");
        e.unwrap().node
    }

    fn show(e: &Expr) -> String {
        match &e.node {
            ExprKind::Lit(Lit::Int(n)) => n.to_string(),
            ExprKind::Path(p) => p
                .iter()
                .map(|i| i.node.as_str())
                .collect::<Vec<_>>()
                .join("::"),
            ExprKind::Binary { op, lhs, rhs } => format!("({:?} {} {})", op, show(lhs), show(rhs)),
            ExprKind::Pipe { lhs, rhs } => format!("(|> {} {})", show(lhs), show(rhs)),
            ExprKind::Call { callee, args } => {
                let a: Vec<_> = args.iter().map(show).collect();
                format!("{}({})", show(callee), a.join(", "))
            }
            ExprKind::Field { base, name } => format!("{}.{}", show(base), name.node),
            ExprKind::Unary { op, expr } => format!("({:?} {})", op, show(expr)),
            other => format!("{other:?}"),
        }
    }

    fn sexp(src: &str) -> String {
        let (e, errs) = parse_expr(src);
        assert!(errs.is_empty(), "{src}: {errs:?}");
        show(&e.unwrap())
    }

    #[test]
    fn precedence() {
        assert_eq!(sexp("1 + 2 * 3"), "(Add 1 (Mul 2 3))");
        assert_eq!(sexp("a || b && c == d"), "(Or a (And b (Eq c d)))");
        assert_eq!(sexp("-a.b"), "(Neg a.b)");
        assert_eq!(sexp("(1 + 2) * 3"), "(Mul (Add 1 2) 3)");
    }

    #[test]
    fn pipe_is_lowest_and_left_associative() {
        assert_eq!(sexp("a + 1 |> f() |> g(2)"), "(|> (|> (Add a 1) f()) g(2))");
    }

    #[test]
    fn calls_and_fields() {
        assert_eq!(sexp("a.b.c(1, 2).d"), "a.b.c(1, 2).d");
        assert_eq!(sexp("std::text::trim(x,)"), "std::text::trim(x)");
    }

    #[test]
    fn closure_in_call() {
        assert!(matches!(ok_expr("map(|d| d.text)"), ExprKind::Call { .. }));
        let ExprKind::Call { args, .. } = ok_expr("f(|a, b| a + b, || 1)") else {
            panic!()
        };
        assert!(matches!(&args[0].node, ExprKind::Closure { params, .. } if params.len() == 2));
        assert!(matches!(&args[1].node, ExprKind::Closure { params, .. } if params.is_empty()));
    }

    #[test]
    fn struct_literal_only_for_capitalized_paths() {
        assert!(matches!(
            ok_expr("Foo { a: 1, b: 2, }"),
            ExprKind::StructLit { .. }
        ));
        let (p, errs) = parse("fn f() { if x { 1 } else { 2 } }");
        assert!(errs.is_empty(), "{errs:?}");
        assert!(p.is_some());
    }

    #[test]
    fn fstring_interpolation() {
        let ExprKind::FString(parts) = ok_expr(r#"f"hi {name}!\n{a + 1}""#) else {
            panic!()
        };
        assert_eq!(parts.len(), 4);
        assert!(matches!(&parts[0], FPart::Text(t) if t == "hi "));
        assert!(matches!(&parts[1], FPart::Expr(_)));
        assert!(matches!(&parts[2], FPart::Text(t) if t == "!\n"));
        let FPart::Expr(e) = &parts[3] else { panic!() };
        assert_eq!(show(e), "(Add a 1)");
        // Spans of interpolated expressions point into the original source.
        assert_eq!(e.span, 15..20);
    }

    #[test]
    fn imports() {
        let (p, errs) = parse("import std::http::{Request, Response};\nimport a::B;");
        assert!(errs.is_empty(), "{errs:?}");
        let p = p.unwrap();
        let Item::Import(i) = &p.items[0] else {
            panic!()
        };
        assert_eq!(i.path.len(), 2);
        assert_eq!(i.group.len(), 2);
        let Item::Import(i) = &p.items[1] else {
            panic!()
        };
        assert_eq!((i.path.len(), i.group.len()), (2, 0));
    }

    #[test]
    fn tensor_types() {
        let (p, errs) = parse("fn f(x: Tensor<f32, [768]>, y: Vec<Tensor<i8, [N, 4]>>) -> Vec {}");
        assert!(errs.is_empty(), "{errs:?}");
        let Item::Fn(f) = &p.unwrap().items[0] else {
            panic!()
        };
        assert_eq!(f.params[0].ty.args.len(), 2);
        assert!(matches!(&f.params[0].ty.args[1], TypeArg::Shape(d) if d == &[Dim::Lit(768)]));
    }

    #[test]
    fn missing_semicolon_is_reported() {
        let (_, errs) = parse("fn f() { a b }");
        assert!(!errs.is_empty());
    }

    #[test]
    fn rag_example_parses() {
        let src = include_str!("../../../examples/rag-service.nv");
        let (p, errs) = parse(src);
        assert!(errs.is_empty(), "{errs:#?}");
        let p = p.unwrap();
        let fns: Vec<_> = p
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) => Some(f),
                _ => None,
            })
            .collect();
        assert_eq!(fns.len(), 4);
        let handler = fns.iter().find(|f| f.name.node == "handle_rag").unwrap();
        assert!(handler.is_pub);
        assert_eq!(handler.attrs[0].name.node, "endpoint");
        assert_eq!(handler.attrs[0].args.len(), 2);
        let embed = fns
            .iter()
            .find(|f| f.name.node == "generate_embedding")
            .unwrap();
        assert!(matches!(
            embed.body.tail.as_deref(),
            Some(Spanned {
                node: ExprKind::Pipe { .. },
                ..
            })
        ));
    }
}
