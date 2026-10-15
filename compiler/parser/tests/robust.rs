//! The parser is fed programs that are cut, damaged and made up. Whatever it is given it must
//! finish, never panic, and report errors whose positions are inside the text.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use telvax_parser::parse;

/// A program that uses most of the syntax once.
const SAMPLE: &str = r#"# a sample
import std.http.Status
from std.text import trim as strip
from util import (clamp, Pt as Point)

trait Shape:
    fn area(self) -> Float: ...

    fn describe(self) -> Text:
        f"area {self.area()}"

type Circle(Shape):
    r: Float = 1.0

    fn area(self) -> Float:
        self.r * self.r * 3.14

type Stack[T]:
    items: List[T]

    fn map[U](self, f: (T) -> U) -> Stack[U]:
        let out: List[U] = []
        for x in self.items:
            out.push(f(x))
        Stack(items: out)

enum Maybe[T]:
    Just(T)
    Nothing

extend Maybe[T]:
    fn or_default(self, d: T) -> T:
        match self:
            case Maybe.Just(v):
                v
            case Maybe.Nothing:
                d

fn pick[T: (Num, Ord)](a: T, b: T) -> T:
    if a > b:
        a
    elif a == b:
        b
    else:
        a

fn compose(f: (Int) -> Int, g: (Int) -> Int) -> (Int) -> Int:
    x => g(f(x))

@endpoint(POST, "/x")
fn handle(i: Circle) -> Result[Circle, Status]:
    let xs = [1, 2, 3]
    let t = (1, "a")
    let a, b = t
    var n = 0
    while n < 3:
        n += 1
        if n == 2:
            continue
    let ys = xs.map(x => x * 2) |> std.iter.sum
    let m: Int? = Some(1)
    if m is Some(v):
        print(f"{v} {ys}")
    let z = pick[Int](1, 2)
    Ok(i)
"#;

fn examples() -> Vec<(String, String)> {
    let mut out = vec![("sample".to_string(), SAMPLE.to_string())];
    let mut stack = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path: PathBuf = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "tvx") {
                out.push((
                    path.display().to_string(),
                    std::fs::read_to_string(&path).unwrap(),
                ));
            }
        }
    }
    out.sort();
    out
}

/// Parses `src` and checks everything that must hold for any input.
fn check(name: &str, src: &str) {
    let (program, diags) = parse(src);
    for d in &diags {
        assert!(!d.message.is_empty(), "{name}: empty message");
        assert!(
            d.span.start <= d.span.end && d.span.end <= src.len(),
            "{name}: span {:?} outside the {} bytes of {src:?}",
            d.span,
            src.len()
        );
        for (span, _) in &d.labels {
            assert!(
                span.start <= span.end && span.end <= src.len(),
                "{name}: label {span:?}"
            );
        }
    }
    if diags.is_empty() {
        assert!(program.is_some(), "{name}: no errors but no program");
    }
}

/// Runs `body` on another thread so a parser that never returns fails the test.
fn within_a_minute(body: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        body();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("the parser did not finish in a minute"),
        Err(e) => panic!("the parser panicked ({e})"),
    }
}

/// A small deterministic generator, so a failure can be reproduced.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The largest character boundary of `s` that is at most `at`.
fn floor(s: &str, mut at: usize) -> usize {
    at = at.min(s.len());
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn boundaries(s: &str) -> Vec<usize> {
    (0..=s.len()).filter(|&i| s.is_char_boundary(i)).collect()
}

#[test]
fn the_sample_and_the_examples_parse_without_errors() {
    for (name, src) in examples() {
        let (program, diags) = parse(&src);
        assert!(diags.is_empty(), "{name}: {diags:?}");
        assert!(program.is_some(), "{name}");
    }
}

#[test]
fn parsing_is_deterministic() {
    for (name, src) in examples() {
        let a = parse(&src);
        let b = parse(&src);
        assert_eq!(a.0, b.0, "{name}");
        assert_eq!(
            a.1.iter()
                .map(|d| (&d.message, &d.span))
                .collect::<Vec<_>>(),
            b.1.iter()
                .map(|d| (&d.message, &d.span))
                .collect::<Vec<_>>(),
            "{name}"
        );
    }
}

#[test]
fn a_program_cut_anywhere_is_handled() {
    within_a_minute(|| {
        for (name, src) in examples() {
            for end in boundaries(&src) {
                check(&format!("{name} cut at {end}"), &src[..end]);
            }
        }
    });
}

#[test]
fn a_program_with_its_beginning_cut_off_is_handled() {
    within_a_minute(|| {
        for (name, src) in examples() {
            for start in boundaries(&src) {
                check(&format!("{name} from {start}"), &src[start..]);
            }
        }
    });
}

#[test]
fn a_program_missing_one_line_is_handled() {
    within_a_minute(|| {
        for (name, src) in examples() {
            let lines: Vec<&str> = src.split_inclusive('\n').collect();
            for skip in 0..lines.len() {
                let text: String = lines
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != skip)
                    .map(|(_, l)| *l)
                    .collect();
                check(&format!("{name} without line {skip}"), &text);
            }
        }
    });
}

#[test]
fn a_program_missing_one_character_is_handled() {
    within_a_minute(|| {
        for (name, src) in examples() {
            let at = boundaries(&src);
            for pair in at.windows(2) {
                let mut text = src.clone();
                text.replace_range(pair[0]..pair[1], "");
                check(&format!("{name} without byte {}", pair[0]), &text);
            }
        }
    });
}

#[test]
fn randomly_damaged_programs_are_handled() {
    within_a_minute(|| {
        const BITS: &[&str] = &[
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ":",
            ",",
            ".",
            "=>",
            "->",
            "|>",
            "?",
            "@",
            "#",
            "\"",
            "'",
            "\n",
            "\n    ",
            "\n        ",
            "\t",
            "fn ",
            "type ",
            "let ",
            "match ",
            "case ",
            "import ",
            "from ",
            "as ",
            "if ",
            "else:",
            "0",
            "1.5",
            "_",
            "é",
            "日本",
            ";",
            "$",
        ];
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for (name, src) in examples() {
            for round in 0..400 {
                let mut text = src.clone();
                for _ in 0..=rng.below(4) {
                    let at = boundaries(&text);
                    let i = at[rng.below(at.len())];
                    match rng.below(3) {
                        0 => text.insert_str(i, BITS[rng.below(BITS.len())]),
                        1 => {
                            let j = at[rng.below(at.len())];
                            let (a, b) = (i.min(j), i.max(j));
                            let end = floor(&text, b.min(a + 12));
                            text.replace_range(a..end, "");
                        }
                        _ => {
                            let j = at[rng.below(at.len())];
                            let (a, b) = (i.min(j), i.max(j));
                            let chunk = text[a..floor(&text, b.min(a + 20))].to_string();
                            text.insert_str(i, &chunk);
                        }
                    }
                }
                check(&format!("{name} round {round}"), &text);
            }
        }
    });
}

#[test]
fn made_up_text_is_handled() {
    within_a_minute(|| {
        const WORDS: &[&str] = &[
            "fn",
            "let",
            "var",
            "type",
            "enum",
            "trait",
            "extend",
            "import",
            "from",
            "match",
            "case",
            "if",
            "elif",
            "else",
            "while",
            "for",
            "in",
            "is",
            "return",
            "and",
            "or",
            "not",
            "x",
            "Int",
            "T",
            "main",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ":",
            ",",
            ".",
            "=",
            "==",
            "=>",
            "->",
            "|>",
            "+",
            "-",
            "*",
            "/",
            "?",
            "@",
            "\"s\"",
            "'c'",
            "f\"{x}\"",
            "1",
            "2.5",
            "\n",
            "\n    ",
            "\n        ",
            "#c",
        ];
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for round in 0..3000 {
            let n = 1 + rng.below(40);
            let mut text = String::new();
            for _ in 0..n {
                text.push_str(WORDS[rng.below(WORDS.len())]);
                text.push(' ');
            }
            check(&format!("made up {round}"), &text);
        }
    });
}

#[test]
fn deep_nesting_does_not_overflow_the_stack() {
    within_a_minute(|| {
        for depth in [10usize, 100, 400] {
            let parens = format!("let x = {}1{}\n", "(".repeat(depth), ")".repeat(depth));
            check(&format!("parens {depth}"), &parens);
            let lists = format!("let x = {}1{}\n", "[".repeat(depth), "]".repeat(depth));
            check(&format!("lists {depth}"), &lists);
            let mut blocks = String::from("fn f():\n");
            for level in 0..depth.min(60) {
                blocks.push_str(&format!("{}if True:\n", "    ".repeat(level + 1)));
            }
            blocks.push_str(&format!("{}pass\n", "    ".repeat(depth.min(60) + 1)));
            check(&format!("blocks {depth}"), &blocks);
        }
    });
}
