//! Byte offsets <-> LSP positions (line, UTF-16 column).

use std::ops::Range;

pub use lsp_types::Position;
use lsp_types::Range as LspRange;

#[derive(Debug)]
pub struct LineIndex {
    /// Byte offset where each line starts.
    starts: Vec<usize>,
    text: String,
}

impl LineIndex {
    pub fn new(text: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
        Self {
            starts,
            text: text.to_string(),
        }
    }

    pub fn position(&self, offset: usize) -> Position {
        let offset = offset.min(self.text.len());
        let line = self.starts.partition_point(|&s| s <= offset) - 1;
        let start = self.starts[line];
        // Back up to a char boundary so slicing cannot panic on bad offsets.
        let mut end = offset;
        while !self.text.is_char_boundary(end) {
            end -= 1;
        }
        let character = self.text[start.min(end)..end].encode_utf16().count();
        Position {
            line: line as u32,
            character: character as u32,
        }
    }

    pub fn range(&self, span: &Range<usize>) -> LspRange {
        LspRange {
            start: self.position(span.start),
            end: self.position(span.end),
        }
    }

    /// The byte offset of `pos`; positions past the end of a line clamp to its end.
    pub fn offset(&self, pos: Position) -> usize {
        let Some(&start) = self.starts.get(pos.line as usize) else {
            return self.text.len();
        };
        let line_end = self
            .starts
            .get(pos.line as usize + 1)
            .copied()
            .unwrap_or(self.text.len());
        let line = &self.text[start..line_end];
        let mut units = 0u32;
        for (i, c) in line.char_indices() {
            if units >= pos.character || c == '\n' {
                return start + i;
            }
            units += c.len_utf16() as u32;
        }
        line_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ascii_and_crlf() {
        let text = "fn a() {}\r\nlet x = 1;\n";
        let idx = LineIndex::new(text);
        assert_eq!(
            idx.position(0),
            Position {
                line: 0,
                character: 0
            }
        );
        let x = text.find("x =").unwrap();
        let p = idx.position(x);
        assert_eq!(
            p,
            Position {
                line: 1,
                character: 4
            }
        );
        assert_eq!(idx.offset(p), x);
    }

    #[test]
    fn columns_count_utf16_units() {
        // 'é' is 2 bytes / 1 unit; the emoji is 4 bytes / 2 units.
        let text = "let s = \"é😀\"; x";
        let idx = LineIndex::new(text);
        let x = text.rfind('x').unwrap();
        assert_eq!(idx.position(x).character, 8 + 1 + 1 + 2 + 1 + 1 + 1);
        assert_eq!(idx.offset(idx.position(x)), x);
    }

    #[test]
    fn out_of_range_positions_clamp() {
        let idx = LineIndex::new("abc\ndef");
        assert_eq!(
            idx.offset(Position {
                line: 0,
                character: 99
            }),
            3
        );
        assert_eq!(
            idx.offset(Position {
                line: 9,
                character: 0
            }),
            7
        );
        assert_eq!(
            idx.position(1000),
            Position {
                line: 1,
                character: 3
            }
        );
    }
}
