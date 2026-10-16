use super::*;

const PROGRAM: &str = "\
trait Shape:
    fn area(self) -> Float: ...

type Circle(Shape):
    r: Float

    fn area(self) -> Float:
        self.r * self.r * 3.14

enum Kind:
    Round(Float)
    Flat

type Config:
    top: Float = 5.0

fn describe(s: Shape) -> Float:
    s.area()

fn build() -> Float:
    let c = Circle(r: 2.0)
    let k = Kind.Round(1.0)
    let cfg = Config(top: 1.0)
    match k:
        case Kind.Round(x):
            c.area() + c.r + x + cfg.top
        case Kind.Flat:
            describe(c)
";

fn snapshot() -> Snapshot {
    let snap = Snapshot::analyze(PROGRAM.to_string());
    assert!(snap.diagnostics.is_empty(), "{:?}", snap.diagnostics);
    snap
}

/// The offset of the `nth` (from 0) occurrence of `needle`, plus `skip` bytes.
fn find(text: &str, needle: &str, nth: usize, skip: usize) -> usize {
    text.match_indices(needle).nth(nth).unwrap().0 + skip
}

/// The lines (from 0) where the symbol at `offset` appears, one entry per occurrence.
fn lines(snap: &Snapshot, offset: usize) -> Vec<usize> {
    let found = snap.occurrences(offset).unwrap();
    found
        .spans
        .iter()
        .map(|s| snap.text[..s.start].matches('\n').count())
        .collect()
}

#[test]
fn types_enums_and_traits_are_found_wherever_they_are_written() {
    let snap = snapshot();
    // `Circle`: its declaration and the construction.
    assert_eq!(lines(&snap, find(PROGRAM, "Circle", 0, 1)), [3, 20]);
    // `Shape`: the trait, `type Circle(Shape)` and the parameter type.
    assert_eq!(lines(&snap, find(PROGRAM, "Shape", 0, 1)), [0, 3, 16]);
    // `Kind`: the enum and its three uses.
    assert_eq!(lines(&snap, find(PROGRAM, "Kind", 0, 1)), [9, 21, 24, 26]);
    // `Config`.
    assert_eq!(lines(&snap, find(PROGRAM, "Config", 0, 1)), [13, 22]);
}

#[test]
fn variants_fields_and_methods_are_found_through_their_types() {
    let snap = snapshot();
    assert_eq!(lines(&snap, find(PROGRAM, "Round", 0, 1)), [10, 21, 24]);
    assert_eq!(lines(&snap, find(PROGRAM, "Flat", 0, 1)), [11, 26]);
    // Field `r`: declared, in the construction, twice through `self`, once through `c`.
    assert_eq!(
        lines(&snap, find(PROGRAM, "r: Float", 0, 0)),
        [4, 7, 7, 20, 25]
    );
    assert_eq!(lines(&snap, find(PROGRAM, "top", 0, 1)), [14, 22, 25]);
    // `area`: the trait, the type that implements it, and both calls.
    assert_eq!(lines(&snap, find(PROGRAM, "area", 0, 1)), [1, 6, 17, 25]);
}

#[test]
fn definition_jumps_to_types_variants_fields_and_methods() {
    let snap = snapshot();
    let at = |needle: &str, nth: usize, skip: usize| {
        let span = snap
            .definition(find(PROGRAM, needle, nth, skip))
            .unwrap_or_else(|| panic!("no definition for {needle}"));
        PROGRAM[..span.start].matches('\n').count()
    };
    assert_eq!(at("Shape", 2, 1), 0, "type annotation");
    assert_eq!(at("Kind", 3, 1), 9, "enum in a pattern");
    assert_eq!(at("Round", 2, 1), 10, "variant in a pattern");
    assert_eq!(at("c.r", 0, 2), 4, "field");
    assert_eq!(at("c.area", 0, 2), 6, "method");
    assert_eq!(at("s.area", 0, 2), 1, "method of a trait");
}

#[test]
fn rename_changes_every_use_of_a_type_a_field_and_a_method() {
    let snap = snapshot();
    let apply = |offset: usize, name: &str| {
        let mut text = PROGRAM.to_string();
        let mut edits = snap.rename(offset, name).unwrap();
        edits.sort_by_key(|(s, _)| std::cmp::Reverse(s.start));
        for (span, new) in edits {
            text.replace_range(span, &new);
        }
        let again = Snapshot::analyze(text.clone());
        assert!(
            again.diagnostics.is_empty(),
            "{:?}\n{text}",
            again.diagnostics
        );
        text
    };
    let text = apply(find(PROGRAM, "Circle", 0, 1), "Disc");
    assert!(text.contains("type Disc(Shape)") && text.contains("Disc(r: 2.0)"));
    let text = apply(find(PROGRAM, "Round", 0, 1), "Wide");
    assert!(text.contains("Kind.Wide(1.0)") && text.contains("case Kind.Wide(x)"));
    let text = apply(find(PROGRAM, "c.r", 0, 2), "radius");
    assert!(
        text.contains("radius: Float") && text.contains("self.radius") && text.contains("c.radius")
    );
    assert!(text.contains("Circle(radius: 2.0)"));
    // Renaming a method of the type renames it in the trait and at the call through the trait.
    let text = apply(find(PROGRAM, "c.area", 0, 2), "surface");
    assert!(text.contains("fn surface(self) -> Float: ..."));
    assert!(text.contains("s.surface()") && text.contains("c.surface()"));
}

#[test]
fn globals_are_found_and_renamed() {
    let src = "let limit = 3\n\nfn f() -> Int:\n    limit + limit\n";
    let snap = Snapshot::analyze(src.to_string());
    let offset = src.find("limit").unwrap() + 1;
    assert_eq!(lines(&snap, offset), [0, 3, 3]);
    assert_eq!(snap.rename(offset, "cap").unwrap().len(), 3);
    assert_eq!(snap.definition(src.rfind("limit").unwrap() + 1), Some(4..9));
}

#[test]
fn navigation_still_works_while_the_text_does_not_parse() {
    let first = snapshot();
    // A half-typed line after the code: everything before it keeps its place.
    let broken = format!("{PROGRAM}\nfn broken(:\n");
    let snap = Snapshot::analyze_after(broken.clone(), None, Some(&first));
    assert!(snap.stale);
    assert_eq!(lines(&snap, find(PROGRAM, "Kind", 0, 1)), [9, 21, 24, 26]);
    assert!(snap.definition(find(PROGRAM, "Round", 2, 1)).is_some());
    // An edit before the references moves them.
    let shifted = format!("# note\n{PROGRAM}");
    let good = Snapshot::analyze_after(shifted.clone(), None, Some(&first));
    let snap = Snapshot::analyze_after(format!("{shifted}\nfn broken(:\n"), None, Some(&good));
    assert!(snap.stale);
    assert_eq!(lines(&snap, find(&shifted, "Kind", 0, 1)), [10, 22, 25, 27]);
    // But a rename is refused until the syntax is right.
    let err = snap.rename(find(&shifted, "Kind", 0, 1), "Sort");
    assert!(err.unwrap_err().contains("syntax errors"));
}
