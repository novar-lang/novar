//! A conservative formatter for Novar source.
//!
//! It works on the token stream, so it never changes what a program means: line breaks you chose
//! stay where they are, comments are kept, and only indentation, spacing inside a line and runs of
//! blank lines are normalised. After formatting, the result is tokenised again and compared with
//! the input; if the tokens differ, formatting fails instead of returning something wrong.

use novar_lexer::{Token, tokenize};

const INDENT: &str = "    ";

/// Formats `src`. Fails on lexical errors or if the result would not tokenise to the same tokens.
pub fn format(src: &str) -> Result<String, String> {
    let (tokens, errors) = tokenize(src);
    if let Some(span) = errors.first() {
        return Err(format!(
            "cannot format: unexpected input at byte {}",
            span.start
        ));
    }
    let out = layout(src, &tokens);
    let (again, errs) = tokenize(&out);
    let same = errs.is_empty()
        && again.len() == tokens.len()
        && again
            .iter()
            .zip(&tokens)
            .all(|((a, sa), (b, sb))| a == b && out[sa.clone()] == src[sb.clone()]);
    if !same {
        return Err("internal error: formatting would change the program".into());
    }
    Ok(out)
}

/// Whether `src` is already formatted.
pub fn is_formatted(src: &str) -> Result<bool, String> {
    format(src).map(|f| f == src)
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Word,
    Literal,
    Open(char),
    Close(char),
    Comma,
    Semi,
    Colon,
    Dot,
    Path,
    At,
    Question,
    Range,
    Arrow,
    Pipe,
    Op,
    Bar,
    Lt,
    Gt,
    Keyword,
}

fn kind(t: &Token) -> Kind {
    use Token::*;
    match t {
        Import | Fn | Let | Mut | Struct | Enum | Pub | Return | Break | Continue | Match | If
        | Else | For | In | While => Kind::Keyword,
        True | False | Int(_) | Float(_) | Str(_) | FStr(_) => Kind::Literal,
        Ident(_) => Kind::Word,
        LParen => Kind::Open('('),
        LBracket => Kind::Open('['),
        LBrace => Kind::Open('{'),
        RParen => Kind::Close('('),
        RBracket => Kind::Close('['),
        RBrace => Kind::Close('{'),
        Comma => Kind::Comma,
        Semi => Kind::Semi,
        Colon => Kind::Colon,
        Dot => Kind::Dot,
        PathSep => Kind::Path,
        At => Kind::At,
        DotDot | DotDotEq => Kind::Range,
        Arrow | FatArrow => Kind::Arrow,
        Pipe => Kind::Pipe,
        Bar | OrOr => Kind::Bar,
        Lt => Kind::Lt,
        Gt => Kind::Gt,
        _ => Kind::Op,
    }
}

fn is_question(src: &str) -> bool {
    src == "?"
}

struct Piece<'a> {
    text: &'a str,
    kind: Kind,
    /// Newlines in the gap before this token (comments excluded).
    comments: Vec<(&'a str, bool, usize)>,
    newlines_before: usize,
}

fn layout(src: &str, tokens: &[(Token<'_>, std::ops::Range<usize>)]) -> String {
    // Split every gap between tokens into comments and line breaks.
    let mut pieces: Vec<Piece> = Vec::new();
    let mut prev_end = 0;
    let mut in_import = false;
    for (tok, span) in tokens {
        let gap = &src[prev_end..span.start];
        let (comments, newlines_before) = scan_gap(gap, prev_end == 0);
        let mut k = kind(tok);
        let text = &src[span.clone()];
        if k == Kind::Op && is_question(text) {
            k = Kind::Question;
        }
        // `import a::{b, c};` keeps its braces tight, like parentheses.
        match k {
            Kind::Keyword if text == "import" => in_import = true,
            Kind::Semi => in_import = false,
            Kind::Open('{') if in_import => k = Kind::Open('('),
            Kind::Close('{') if in_import => k = Kind::Close('('),
            _ => {}
        }
        pieces.push(Piece {
            text,
            kind: k,
            comments,
            newlines_before,
        });
        prev_end = span.end;
    }
    let tail = &src[prev_end..];
    let (tail_comments, _) = scan_gap(tail, false);

    let mut out = String::new();
    // For every open delimiter, the output line it was opened on. Delimiters opened on the same
    // line add only one level of indentation (`Ok(Response {` ... `});`).
    let mut open_lines: Vec<usize> = Vec::new();
    let mut line_no = 0usize;
    let mut generic = 0usize; // open generic <
    let mut closure_open = false;
    let mut at_line_start = true;
    let mut prev: Option<&Piece> = None;
    let mut prev_prev_kind: Option<Kind> = None;

    for (i, p) in pieces.iter().enumerate() {
        // Comments that precede this token.
        for (text, trailing, blank_before) in &p.comments {
            if *trailing && !at_line_start {
                out.push(' ');
                out.push_str(text);
                {
                    out.push('\n');
                    line_no += 1;
                }
                at_line_start = true;
            } else {
                if !at_line_start {
                    {
                        out.push('\n');
                        line_no += 1;
                    }
                    at_line_start = true;
                }
                if *blank_before > 0 && !out.is_empty() && !out.ends_with("\n\n") {
                    {
                        out.push('\n');
                        line_no += 1;
                    }
                }
                out.push_str(&INDENT.repeat(levels(&open_lines)));
                out.push_str(text);
                {
                    out.push('\n');
                    line_no += 1;
                }
            }
        }
        let comment_ended_line = !p.comments.is_empty() && at_line_start;
        let breaks = if comment_ended_line {
            // After a comment line the token starts a new line; keep a blank line if one was there.
            usize::from(p.newlines_before > 1) * 2
        } else {
            p.newlines_before
        };
        if i > 0 && breaks > 0 && !at_line_start {
            {
                out.push('\n');
                line_no += 1;
            }
            at_line_start = true;
        }
        if breaks > 1 && !out.is_empty() && !out.ends_with("\n\n") && at_line_start {
            {
                out.push('\n');
                line_no += 1;
            }
        }

        // Closers at the start of a line are accounted for before indenting that line.
        let closers_here = if at_line_start || (i > 0 && breaks > 0) {
            pieces[i..]
                .iter()
                .enumerate()
                .take_while(|(n, q)| {
                    matches!(q.kind, Kind::Close(_)) && (*n == 0 || q.newlines_before == 0)
                })
                .count()
        } else {
            0
        };
        let indent_now = if at_line_start {
            levels(&open_lines[..open_lines.len().saturating_sub(closers_here)])
        } else {
            0
        };
        if let Kind::Close(_) = p.kind {
            open_lines.pop();
        }
        let generic_for_space = generic;
        if p.kind == Kind::Gt && generic > 0 {
            generic -= 1;
        }

        if at_line_start {
            // Continuation lines are indented one more level: a line that starts with `|>` or
            // `.`, and the line after one that ends with `=`.
            let after_assign = prev.is_some_and(|q| q.kind == Kind::Op && q.text == "=");
            let extra = usize::from(matches!(p.kind, Kind::Pipe | Kind::Dot) || after_assign);
            out.push_str(&INDENT.repeat(indent_now + extra));
            at_line_start = false;
        } else if let Some(prev) = prev
            && space_between(prev, p, prev_prev_kind, generic_for_space, closure_open)
        {
            out.push(' ');
        }
        out.push_str(p.text);

        match p.kind {
            Kind::Open(_) => open_lines.push(line_no),
            Kind::Lt if is_generic_open(prev, p) => generic += 1,
            Kind::Bar if p.text == "|" => {
                closure_open = !closure_open;
            }
            _ => {}
        }
        prev_prev_kind = prev.map(|x| x.kind);
        prev = Some(p);
    }
    for (text, _, blank_before) in &tail_comments {
        if !at_line_start {
            out.push('\n');
            at_line_start = true;
        }
        if *blank_before > 0 && !out.ends_with("\n\n") && !out.is_empty() {
            out.push('\n');
        }
        out.push_str(text);
        out.push('\n');
    }
    if !at_line_start {
        out.push('\n');
    }
    out
}

/// Comments in a gap as `(text, on the same line as the previous token, blank lines before)`, and
/// the number of line breaks that separate the previous token from the next one.
fn scan_gap(gap: &str, first: bool) -> (Vec<(&str, bool, usize)>, usize) {
    let mut comments = Vec::new();
    let mut newlines = 0usize;
    let mut blank = 0usize;
    let mut total = 0usize;
    let mut rest = gap;
    while !rest.is_empty() {
        if let Some(pos) = rest.find("//") {
            let (before, after) = rest.split_at(pos);
            let nl = before.matches('\n').count();
            total += nl;
            newlines += nl;
            let end = after.find('\n').unwrap_or(after.len());
            let text = after[..end].trim_end();
            let trailing = newlines == 0 && !first;
            let blanks = if newlines > 1 { 1 } else { blank };
            comments.push((text, trailing, blanks));
            rest = &after[end..];
            newlines = 0;
            blank = 0;
        } else {
            let nl = rest.matches('\n').count();
            total += nl;
            newlines += nl;
            rest = "";
        }
    }
    // Trailing newlines after the last comment decide the spacing before the token.
    (comments, if total == 0 { 0 } else { newlines.max(1) })
}

/// Indentation levels for delimiters opened on the given lines (one per distinct line).
fn levels(open_lines: &[usize]) -> usize {
    let mut n = 0;
    let mut last = None;
    for l in open_lines {
        if last != Some(*l) {
            n += 1;
            last = Some(*l);
        }
    }
    n
}

fn is_generic_open(prev: Option<&Piece>, p: &Piece) -> bool {
    p.kind == Kind::Lt
        && prev
            .is_some_and(|q| q.kind == Kind::Word && q.text.starts_with(|c: char| c.is_uppercase()))
}

fn is_value_end(k: Kind) -> bool {
    matches!(
        k,
        Kind::Word | Kind::Literal | Kind::Close(_) | Kind::Question | Kind::Gt
    )
}

fn space_between(
    prev: &Piece,
    next: &Piece,
    before_prev: Option<Kind>,
    generic: usize,
    closure_open: bool,
) -> bool {
    use Kind::*;
    let unary_prev = |p: &Piece, before: Option<Kind>| {
        // Is `p` (an `-` or `!`) a prefix operator?
        matches!(p.kind, Op)
            && matches!(p.text, "-" | "!" | "&" | "*")
            && !before.is_some_and(is_value_end)
    };
    match (prev.kind, next.kind) {
        // Nothing before these.
        (_, Comma | Semi | Dot | Path | Question) => false,
        (_, Close('{')) => !matches!(prev.kind, Open('{')),
        (_, Close(_)) => false,
        (Open('{'), _) => true,
        (Open(_), _) => false,
        (Dot | Path | At, _) => false,
        (Range, _) | (_, Range) => false,
        // Colons hug the name and take a space after.
        (_, Colon) => false,
        (Colon, _) => true,
        (Comma | Semi, _) => true,
        // Calls and indexing.
        (Word | Close(_) | Question, Open('(' | '[')) => false,
        (Gt, Open('(')) if generic == 0 => false,
        (Keyword, Open('(' | '[')) => true,
        // Generic arguments hug their name.
        (Word, Lt) if is_generic_open(Some(prev), next) => false,
        (Lt, _) if generic > 0 => false,
        (_, Gt) if generic > 0 => false,
        // Closure bars: `|x, y| body`.
        (Bar, _) if prev.text == "|" => !closure_open,
        (_, Bar) if next.text == "|" => !closure_open && !matches!(prev.kind, Open(_) | Comma),
        // Prefix operators stick to their operand.
        (Op, _) if unary_prev(prev, before_prev) => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(src: &str) -> String {
        format(src).unwrap()
    }

    #[test]
    fn normalises_spacing_and_indentation() {
        let out = fmt("fn add(a:i32,b:i32)->i32{\nlet x=a+b*2;\n  if x>3{return x;}\nx - 1\n}\n");
        assert_eq!(
            out,
            "fn add(a: i32, b: i32) -> i32 {\n    let x = a + b * 2;\n    if x > 3 { return x; }\n    x - 1\n}\n"
        );
    }

    #[test]
    fn keeps_comments_and_limits_blank_lines() {
        let src = "// header\n\n\n\nfn f() -> i32 {   // trailing\n  // inside\n  1 // one\n}\n";
        assert_eq!(
            fmt(src),
            "// header\n\nfn f() -> i32 { // trailing\n    // inside\n    1 // one\n}\n"
        );
    }

    #[test]
    fn handles_generics_unary_calls_ranges_and_pipes() {
        let src = "fn f(t:Tensor<f32,[2,3]>,v:Vec<Vec<i32>>)->Result<i32,String>{\nlet a= -1;\nlet b=!true;\nfor i in 0..n{g(i,-a);}\nlet c=x\n|>f()\n|>g();\nlet d=|x,y|x+y;\nOk(t[0]  [1])\n}";
        let out = fmt(src);
        assert_eq!(
            out,
            "fn f(t: Tensor<f32, [2, 3]>, v: Vec<Vec<i32>>) -> Result<i32, String> {\n    let a = -1;\n    let b = !true;\n    for i in 0..n { g(i, -a); }\n    let c = x\n        |> f()\n        |> g();\n    let d = |x, y| x + y;\n    Ok(t[0][1])\n}\n"
        );
    }

    #[test]
    fn is_idempotent_on_the_bundled_examples() {
        for src in [
            include_str!("../../../examples/rag-service.nv"),
            include_str!("../../../examples/basics/primes.nv"),
            include_str!("../../../examples/basics/sieve.nv"),
            include_str!("../../../examples/hello-world/hello.nv"),
        ] {
            let once = fmt(src);
            assert_eq!(fmt(&once), once, "not idempotent");
        }
    }

    #[test]
    fn enums_match_and_break() {
        let src = "enum S{A(i32),B}\nfn f(s:S)->i32{match s{S::A(n)=>n,S::B=>0,}}\nfn g(){while true{break;}}";
        let out = fmt(src);
        assert_eq!(fmt(&out), out);
        assert!(out.contains("enum S { A(i32), B }"), "{out}");
        assert!(out.contains("S::A(n) => n,"), "{out}");
    }

    #[test]
    fn rejects_input_it_cannot_lex() {
        assert!(format("fn f() { let s = \"unterminated; }").is_err());
    }

    #[test]
    fn nested_blocks_import_braces_and_multiline_literals() {
        let src = "import std::http::{Request,Status};\nfn f() -> R {\n    while a {\n        if b {\n            c();\n        }\n    }\n    Ok(Response {\n        x: 1,\n        y: 2\n    })\n}\n";
        let out = fmt(src);
        assert_eq!(
            out,
            "import std::http::{Request, Status};\nfn f() -> R {\n    while a {\n        if b {\n            c();\n        }\n    }\n    Ok(Response {\n        x: 1,\n        y: 2\n    })\n}\n"
        );
    }

    #[test]
    fn a_value_on_the_line_after_an_equals_sign_is_indented() {
        let src = "fn f() {\n    let long_name: Type =\n    call(1);\n}\n";
        assert_eq!(
            fmt(src),
            "fn f() {\n    let long_name: Type =\n        call(1);\n}\n"
        );
        assert_eq!(fmt(&fmt(src)), fmt(src));
    }
}
