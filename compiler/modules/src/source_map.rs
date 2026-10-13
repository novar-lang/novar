//! Several source files in one offset space, so a single span names a place in a whole program.

use std::collections::HashMap;

use novar_diagnostics::Diagnostic;

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub name: String,
    pub text: String,
    /// Offset of the file's first byte in the shared space.
    pub base: usize,
}

#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    pub files: Vec<SourceFile>,
    /// Mangled item name -> the name the user wrote (`pkg__mod__f` -> `pkg::mod::f`).
    pub display_names: HashMap<String, String>,
}

impl SourceMap {
    /// Registers a file and returns its base offset. Files are separated by one unused byte so no
    /// span ever touches two of them.
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> usize {
        let base = self.files.last().map_or(0, |f| f.base + f.text.len() + 1);
        self.files.push(SourceFile {
            name: name.into(),
            text: text.into(),
            base,
        });
        base
    }

    /// The file containing global `offset`, and the offset inside it.
    pub fn locate(&self, offset: usize) -> Option<(&SourceFile, usize)> {
        let idx = self
            .files
            .partition_point(|f| f.base <= offset)
            .checked_sub(1)?;
        let file = &self.files[idx];
        (offset <= file.base + file.text.len()).then(|| (file, offset - file.base))
    }

    /// Replaces mangled item names in `text` with the names the user wrote.
    pub fn demangle(&self, text: &str) -> String {
        let mut out = text.to_string();
        // Longest first so `a__b__c` is replaced before `a__b`.
        let mut names: Vec<_> = self.display_names.iter().collect();
        names.sort_by_key(|(m, _)| std::cmp::Reverse(m.len()));
        for (mangled, shown) in names {
            out = out.replace(mangled.as_str(), shown);
        }
        out
    }

    /// Renders diagnostics, each against the file it points into.
    pub fn render(&self, diagnostics: &[Diagnostic], color: bool) -> String {
        let mut out = String::new();
        for d in diagnostics {
            let Some((file, local)) = self.locate(d.span.start) else {
                out.push_str(&format!("error: {}\n", self.demangle(&d.message)));
                continue;
            };
            let shift = |s: &std::ops::Range<usize>| {
                s.start.saturating_sub(file.base)
                    ..s.end.saturating_sub(file.base).min(file.text.len())
            };
            let mut local_diag = d.clone();
            local_diag.message = self.demangle(&d.message);
            local_diag.span = local
                ..d.span
                    .end
                    .saturating_sub(file.base)
                    .min(file.text.len())
                    .max(local);
            // Labels that point into other files cannot be drawn on this one.
            local_diag.labels = d
                .labels
                .iter()
                .filter(|(s, _)| {
                    self.locate(s.start)
                        .is_some_and(|(f, _)| f.base == file.base)
                })
                .map(|(s, t)| (shift(s), self.demangle(t)))
                .collect();
            local_diag.notes = d.notes.iter().map(|n| self.demangle(n)).collect();
            out.push_str(&novar_diagnostics::render(
                &file.name,
                &file.text,
                &[local_diag],
                color,
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_get_disjoint_ranges_and_can_be_located() {
        let mut m = SourceMap::default();
        let a = m.add("a.nv", "fn a() {}");
        let b = m.add("b.nv", "fn b() {}");
        assert_eq!((a, b), (0, 10));
        assert_eq!(
            m.locate(3).map(|(f, o)| (f.name.as_str(), o)),
            Some(("a.nv", 3))
        );
        assert_eq!(
            m.locate(12).map(|(f, o)| (f.name.as_str(), o)),
            Some(("b.nv", 2))
        );
        assert!(m.locate(500).is_none());
    }

    #[test]
    fn diagnostics_render_against_their_own_file_with_demangled_names() {
        let mut m = SourceMap::default();
        m.add("a.nv", "fn a() {}");
        let base = m.add("b.nv", "let x = 1;\nlet y = pkg__f;");
        m.display_names.insert("pkg__f".into(), "pkg::f".into());
        let at = base + "let x = 1;\nlet y = ".len();
        let d = Diagnostic::error(at..at + 6, "cannot find `pkg__f` in this scope");
        let text = m.render(&[d], false);
        assert!(text.contains("b.nv:2:9"), "{text}");
        assert!(
            text.contains("cannot find `pkg::f` in this scope"),
            "{text}"
        );
    }
}
