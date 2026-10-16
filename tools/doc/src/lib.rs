//! `telvax doc`: Markdown reference built from the types and signatures of a source file, plus the
//! `#` comment lines written directly above each item.

use telvax_ast::{Alias, Enum, Fn, GenericParam, Item, Param, Program, Span, Struct, Trait, Type};

/// What to include.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Skip names that start with an underscore.
    pub public_only: bool,
}

/// The Markdown for one file. `title` is the heading (usually the path).
pub fn generate(src: &str, title: &str, opts: Options) -> Result<String, String> {
    let (program, diags) = telvax_parser::parse(src);
    let Some(program) = program else {
        let msg = diags
            .first()
            .map_or("syntax error".to_string(), |d| d.message.clone());
        return Err(msg);
    };
    Ok(render(src, title, &program, opts))
}

fn render(src: &str, title: &str, program: &Program, opts: Options) -> String {
    let mut out = format!("## `{title}`\n");
    let mut body = String::new();
    for item in &program.items {
        match item {
            Item::Fn(f) if keep(f.is_pub, opts) => body.push_str(&function(src, f, "###")),
            Item::Struct(s) if keep(s.is_pub, opts) => body.push_str(&structure(src, s)),
            Item::Enum(e) if keep(e.is_pub, opts) => body.push_str(&enumeration(src, e)),
            Item::Alias(a) if keep(a.is_pub, opts) => body.push_str(&alias(src, a)),
            Item::Trait(t) if keep(t.is_pub, opts) => body.push_str(&interface(src, t)),
            Item::Impl(i) => {
                let fns: String = i
                    .fns
                    .iter()
                    .filter(|f| keep(f.is_pub, opts))
                    .map(|f| function(src, f, "####"))
                    .collect();
                if !fns.is_empty() {
                    let ty = text(src, &i.ty.span);
                    let head = match &i.trait_name {
                        Some(t) => format!("### `extend {ty}` as `{}`\n\n", t.node),
                        None => format!("### `extend {ty}`\n\n"),
                    };
                    body.push_str(&head);
                    body.push_str(&fns);
                }
            }
            _ => {}
        }
    }
    if body.is_empty() {
        out.push_str("\nNothing to document.\n");
    } else {
        out.push('\n');
        out.push_str(&body);
    }
    out
}

fn keep(is_pub: bool, opts: Options) -> bool {
    is_pub || !opts.public_only
}

fn text<'a>(src: &'a str, span: &Span) -> &'a str {
    src.get(span.clone()).unwrap_or("").trim()
}

fn ty(src: &str, t: &Type) -> String {
    text(src, &t.span).to_string()
}

fn generics(params: &[GenericParam]) -> String {
    if params.is_empty() {
        return String::new();
    }
    let list: Vec<String> = params
        .iter()
        .map(|p| {
            if p.bounds.is_empty() {
                p.name.node.clone()
            } else {
                let bounds: Vec<&str> = p.bounds.iter().map(|b| b.node.as_str()).collect();
                format!("{}: {}", p.name.node, bounds.join(" + "))
            }
        })
        .collect();
    format!("[{}]", list.join(", "))
}

fn param(src: &str, p: &Param) -> String {
    let mut s = String::new();
    s.push_str(&p.name.node);
    // `self` has no written type; its type span is empty.
    let t = ty(src, &p.ty);
    if !t.is_empty() && t != p.name.node {
        s.push_str(": ");
        s.push_str(&t);
    }
    if let Some(d) = &p.default {
        s.push_str(" = ");
        s.push_str(text(src, &d.span));
    }
    s
}

fn signature(src: &str, f: &Fn) -> String {
    let mut s = String::new();
    if f.is_async {
        s.push_str("async ");
    }
    s.push_str("fn ");
    s.push_str(&f.name.node);
    s.push_str(&generics(&f.generics));
    let params: Vec<String> = f.params.iter().map(|p| param(src, p)).collect();
    s.push_str(&format!("({})", params.join(", ")));
    if let Some(r) = &f.ret {
        s.push_str(" -> ");
        s.push_str(&ty(src, r));
    }
    s
}

fn function(src: &str, f: &Fn, level: &str) -> String {
    let start = f
        .attrs
        .first()
        .map_or(f.span.start, |a| a.span.start.min(f.span.start));
    let mut s = format!("{level} `{}`\n\n```tvx\n", f.name.node);
    for a in &f.attrs {
        s.push_str(text(src, &a.span));
        s.push('\n');
    }
    s.push_str(&signature(src, f));
    s.push_str("\n```\n\n");
    s.push_str(&comment_before(src, start));
    s
}

fn structure(src: &str, st: &Struct) -> String {
    let mut s = format!("### `type {}{}`\n\n", st.name.node, generics(&st.generics));
    s.push_str(&comment_before(src, st.span.start));
    if !st.fields.is_empty() {
        s.push_str("| Field | Type | Default |\n|---|---|---|\n");
        for f in &st.fields {
            let d = f
                .default
                .as_ref()
                .map_or(String::new(), |d| format!("`{}`", text(src, &d.span)));
            s.push_str(&format!(
                "| `{}` | `{}` | {d} |\n",
                f.name.node,
                ty(src, &f.ty)
            ));
        }
        s.push('\n');
    }
    s
}

fn enumeration(src: &str, e: &Enum) -> String {
    let mut s = format!("### `enum {}{}`\n\n", e.name.node, generics(&e.generics));
    s.push_str(&comment_before(src, e.span.start));
    if !e.variants.is_empty() {
        s.push_str("| Variant | Values |\n|---|---|\n");
        for v in &e.variants {
            let values: Vec<String> = v
                .fields
                .iter()
                .enumerate()
                .map(|(i, t)| match v.names.get(i) {
                    Some(n) => format!("{}: {}", n.node, ty(src, t)),
                    None => ty(src, t),
                })
                .collect();
            let shown = if values.is_empty() {
                String::new()
            } else {
                format!("`({})`", values.join(", "))
            };
            let from = if v.from { " (`from`)" } else { "" };
            s.push_str(&format!("| `{}`{from} | {shown} |\n", v.name.node));
        }
        s.push('\n');
    }
    s
}

fn alias(src: &str, a: &Alias) -> String {
    let mut s = format!(
        "### `type {}{}`\n\nAnother name for `{}`.\n\n",
        a.name.node,
        generics(&a.generics),
        ty(src, &a.ty)
    );
    s.push_str(&comment_before(src, a.span.start));
    s
}

fn interface(src: &str, t: &Trait) -> String {
    let mut s = format!("### `trait {}`\n\n", t.name.node);
    s.push_str(&comment_before(src, t.span.start));
    for f in &t.fns {
        s.push_str(&function(src, f, "####"));
    }
    s
}

/// The `#` lines directly above byte `start` (no blank line between), as paragraphs ending in a
/// blank line; empty when there are none.
fn comment_before(src: &str, start: usize) -> String {
    let line_start = src[..start.min(src.len())].rfind('\n').map_or(0, |i| i + 1);
    let mut lines: Vec<&str> = Vec::new();
    for line in src[..line_start].lines().rev() {
        match line.trim_start().strip_prefix('#') {
            Some(rest) => lines.push(rest.strip_prefix(' ').unwrap_or(rest)),
            None => break,
        }
    }
    if lines.is_empty() {
        return String::new();
    }
    lines.reverse();
    format!("{}\n\n", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(src: &str) -> String {
        generate(src, "m.tvx", Options::default()).unwrap()
    }

    #[test]
    fn function_signature_and_comment() {
        let md =
            doc("# Adds two numbers.\n# Slowly.\nfn add(a: Int, b: Int = 2) -> Int:\n    a + b\n");
        assert!(md.contains("fn add(a: Int, b: Int = 2) -> Int"), "{md}");
        assert!(md.contains("Adds two numbers.\nSlowly."), "{md}");
    }

    #[test]
    fn blank_line_detaches_comment() {
        let md = doc("# File header.\n\nfn f() -> Int:\n    1\n");
        assert!(!md.contains("File header"), "{md}");
    }

    #[test]
    fn types_enums_and_methods() {
        let md = doc(
            "# A point.\ntype Point:\n    x: Float\n    y: Float = 0.0\n\nenum Shape:\n    Circle(Float)\n    Empty\n\nextend Point:\n    fn norm(self) -> Float:\n        self.x\n",
        );
        assert!(md.contains("### `type Point`"), "{md}");
        assert!(md.contains("A point."), "{md}");
        assert!(md.contains("| `y` | `Float` | `0.0` |"), "{md}");
        assert!(md.contains("| `Circle` | `(Float)` |"), "{md}");
        assert!(md.contains("### `extend Point`"), "{md}");
        assert!(md.contains("fn norm(self) -> Float"), "{md}");
    }

    #[test]
    fn public_only_skips_private() {
        let src = "fn _hidden() -> Int:\n    1\n\nfn shown() -> Int:\n    2\n";
        let md = generate(src, "m.tvx", Options { public_only: true }).unwrap();
        assert!(md.contains("shown") && !md.contains("_hidden"), "{md}");
    }
}
