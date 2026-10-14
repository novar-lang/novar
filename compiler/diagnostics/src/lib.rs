//! Compiler diagnostics shared by every phase, rendered with `ariadne`.

use std::ops::Range;

pub type Span = Range<usize>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Span,
    /// Extra annotated spans (e.g. "expected because of this").
    pub labels: Vec<(Span, String)>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, span, message)
    }

    pub fn warning(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, span, message)
    }

    fn new(severity: Severity, span: Span, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            span,
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn with_label(mut self, span: Span, text: impl Into<String>) -> Self {
        self.labels.push((span, text.into()));
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// Renders `diagnostics` against `src`. Spans are byte offsets.
pub fn render(file_name: &str, src: &str, diagnostics: &[Diagnostic], color: bool) -> String {
    use ariadne::{Color, Config, IndexType, Label, Report, ReportKind, Source};

    let mut out = Vec::new();
    for d in diagnostics {
        let (kind, tint) = match d.severity {
            Severity::Error => (ReportKind::Error, Color::Red),
            Severity::Warning => (ReportKind::Warning, Color::Yellow),
        };
        let at = |s: &Span| (file_name, clamp(s, src.len()));
        let mut report = Report::build(kind, at(&d.span))
            .with_config(
                Config::default()
                    .with_color(color)
                    .with_index_type(IndexType::Byte),
            )
            .with_message(&d.message)
            .with_label(
                Label::new(at(&d.span))
                    .with_message(&d.message)
                    .with_color(tint),
            );
        for (span, text) in &d.labels {
            report = report.with_label(Label::new(at(span)).with_message(text));
        }
        for note in &d.notes {
            report = report.with_note(note);
        }
        report
            .finish()
            .write((file_name, Source::from(src)), &mut out)
            .expect("writing to a Vec cannot fail");
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn clamp(span: &Span, len: usize) -> Span {
    span.start.min(len)..span.end.min(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_message_label_and_note() {
        let src = "let x = ;\n";
        let d = Diagnostic::error(8..9, "expected expression")
            .with_label(0..3, "statement starts here")
            .with_note("try adding a value");
        let out = render("a.tvx", src, &[d], false);
        assert!(out.contains("Error: expected expression"), "{out}");
        assert!(out.contains("a.tvx:1:9"), "{out}");
        assert!(out.contains("statement starts here"), "{out}");
        assert!(out.contains("try adding a value"), "{out}");
    }

    #[test]
    fn out_of_range_span_does_not_panic() {
        let out = render("a.tvx", "x", &[Diagnostic::warning(5..9, "eof")], false);
        assert!(out.contains("Warning"));
    }

    #[test]
    fn multibyte_source_uses_byte_offsets() {
        let src = "é = ;";
        let out = render("a.tvx", src, &[Diagnostic::error(5..6, "bad")], false);
        assert!(out.contains("a.tvx:1:5"), "{out}");
    }
}
