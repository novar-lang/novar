//! Completion in context: types, members of every kind of value, table columns, closure
//! parameters and ranking.

use telvax_lsp::analysis::{Completion, CompletionKind, Snapshot};

fn complete_after_edit(before: &str, typed: &str) -> Vec<Completion> {
    let good = Snapshot::analyze(before.replacen("/*here*/", "", 1));
    assert!(good.hir.is_some(), "{:?}", good.diagnostics);
    let offset = before.find("/*here*/").unwrap();
    let broken = before.replacen("/*here*/", typed, 1);
    Snapshot::analyze_after(broken, None, Some(&good)).completions(offset + typed.len())
}

fn labels(items: &[Completion]) -> Vec<&str> {
    items.iter().map(|c| c.label.as_str()).collect()
}

fn has(items: &[Completion], wanted: &[&str]) {
    let l = labels(items);
    for w in wanted {
        assert!(l.contains(w), "missing {w}: {l:?}");
    }
}

fn lacks(items: &[Completion], unwanted: &[&str]) {
    let l = labels(items);
    for w in unwanted {
        assert!(!l.contains(w), "unexpected {w}: {l:?}");
    }
}

const DECLS: &str = "\
type Point:
    x: Float
    y: Float

enum Shape:
    Circle(Float)
    Rect(Float, Float)
    Empty

trait Named:
    fn name(self) -> Text: ...

    fn greet(self) -> Text:
        f\"hi {self.name()}\"

type Id = Int

type Row:
    id: Int
    score: f32
    label: Text

";

fn with(body: &str) -> String {
    format!("{DECLS}{body}")
}

#[test]
fn offers_types_where_a_type_is_written() {
    let src = with("fn f(a: Int) -> Int:\n    /*here*/\n    0\n");
    for typed in ["let v: ", "var v: "] {
        let items = complete_after_edit(&src, typed);
        has(
            &items,
            &[
                "Int", "Text", "List", "Map", "Table", "Point", "Shape", "Named", "Id",
            ],
        );
        lacks(&items, &["fn", "let", "a", "f"]);
    }
    let src = with("fn f(a: Int):\n    pass\n\n/*here*/\n");
    let items = complete_after_edit(&src, "fn g(p: ");
    has(&items, &["Point", "Shape", "Float"]);
    let items = complete_after_edit(&src, "fn g() -> ");
    has(&items, &["Point", "Shape", "Result"]);
    lacks(&items, &["return", "f"]);
    let items = complete_after_edit(&src, "fn g(xs: List[");
    has(&items, &["Point", "Int"]);
    lacks(&items, &["return"]);
    let items = complete_after_edit(&src, "type Alias = ");
    has(&items, &["Point", "Int"]);
    lacks(&items, &["return"]);
}

#[test]
fn offers_types_for_fields_and_type_parameters() {
    let src = with(
        "type Wrapper:\n    a: Int\n    /*here*/\n\nfn first[T](xs: List[T]) -> T:\n    xs[0]\n",
    );
    let items = complete_after_edit(&src, "inner: ");
    has(&items, &["Point", "Shape", "Id", "Text"]);
    lacks(&items, &["return", "fn"]);
    let src = with("fn first[T](xs: List[T]) -> T:\n    /*here*/\n    xs[0]\n");
    let items = complete_after_edit(&src, "let y: ");
    has(&items, &["T", "Point"]);
}

#[test]
fn offers_enums_traits_and_aliases_by_kind() {
    let src = with("fn f():\n    /*here*/\n    pass\n");
    let items = complete_after_edit(&src, "");
    let kind = |name: &str| items.iter().find(|c| c.label == name).unwrap().kind;
    assert_eq!(kind("Shape"), CompletionKind::Enum);
    assert_eq!(kind("Point"), CompletionKind::Struct);
    lacks(&items, &["Named", "Id"]);
    let items = complete_after_edit(&src, "let t: ");
    assert_eq!(
        items.iter().find(|c| c.label == "Named").unwrap().kind,
        CompletionKind::Trait
    );
}

#[test]
fn offers_enum_variants_and_associated_functions() {
    let src = with(
        "extend Point:\n    fn origin() -> Point:\n        Point(x=0.0, y=0.0)\n\n    fn norm(self) -> Float:\n        self.x\n\nfn f():\n    /*here*/\n    pass\n",
    );
    let items = complete_after_edit(&src, "Shape.");
    assert_eq!(labels(&items), ["Circle", "Empty", "Rect"]);
    assert!(items.iter().all(|c| c.kind == CompletionKind::EnumMember));
    assert_eq!(
        items
            .iter()
            .find(|c| c.label == "Rect")
            .unwrap()
            .detail
            .as_deref(),
        Some("Shape.Rect(Float, Float)")
    );
    let items = complete_after_edit(&src, "Point.");
    assert_eq!(labels(&items), ["origin"]);
    let items = complete_after_edit(&src, "Point.origin().");
    has(&items, &["x", "y", "norm"]);
}

#[test]
fn offers_methods_of_user_types_and_traits() {
    let src = with(
        "type Dog(Named):\n    name_text: Text\n\n    fn name(self) -> Text:\n        self.name_text\n\nfn f(d: Dog, n: Named):\n    /*here*/\n    pass\n",
    );
    let items = complete_after_edit(&src, "d.");
    has(&items, &["name_text", "name", "greet"]);
    let items = complete_after_edit(&src, "n.");
    assert_eq!(labels(&items), ["greet", "name"]);
}

#[test]
fn offers_the_methods_of_every_builtin_type() {
    let src = with(
        "fn f(s: Text, xs: List[Point], m: Map[Text, Int], o: Int?, r: Result[Int, Text], t: Table[Row]):\n    /*here*/\n    pass\n",
    );
    has(
        &complete_after_edit(&src, "s."),
        &["split", "starts_with", "to_int", "lines"],
    );
    has(
        &complete_after_edit(&src, "xs."),
        &["push", "map", "filter", "fold", "join", "sort"],
    );
    has(
        &complete_after_edit(&src, "m."),
        &["insert", "keys", "entries", "contains_key"],
    );
    has(
        &complete_after_edit(&src, "o."),
        &["unwrap_or", "is_some", "ok_or"],
    );
    has(
        &complete_after_edit(&src, "r."),
        &["map_err", "is_ok", "err"],
    );
    has(
        &complete_after_edit(&src, "t."),
        &["sql", "len", "id", "score", "label"],
    );
    let items = complete_after_edit(&src, "s.sp");
    assert_eq!(labels(&items), ["split"]);
    assert!(items[0].detail.as_deref().unwrap().contains("List[Text]"));
}

#[test]
fn follows_method_calls_in_a_chain() {
    let src =
        with("fn f(s: Text, xs: List[Point], m: Map[Text, Point]):\n    /*here*/\n    pass\n");
    has(&complete_after_edit(&src, "s.trim()."), &["split", "trim"]);
    has(
        &complete_after_edit(&src, "xs.first().unwrap()."),
        &["x", "y"],
    );
    has(
        &complete_after_edit(&src, "m.get(\"a\").unwrap()."),
        &["x", "y"],
    );
    has(
        &complete_after_edit(&src, "s.split(\",\")."),
        &["push", "map"],
    );
}

#[test]
fn types_the_parameter_of_a_closure_being_written() {
    let src = with("fn f(xs: List[Point], t: Table[Row], o: Point?):\n    /*here*/\n    pass\n");
    assert_eq!(
        labels(&complete_after_edit(&src, "xs.map(p => p.")),
        ["x", "y"]
    );
    assert_eq!(
        labels(&complete_after_edit(
            &src,
            "xs.filter(p => p.x > 1.0 and p."
        )),
        ["x", "y"]
    );
    has(
        &complete_after_edit(&src, "std.table.filter(t, r => r."),
        &["id", "score", "label"],
    );
    assert_eq!(
        labels(&complete_after_edit(&src, "o.map(p => p.")),
        ["x", "y"]
    );
    assert_eq!(
        labels(&complete_after_edit(&src, "xs.fold(0.0, (acc, p) => p.")),
        ["x", "y"]
    );
    // The closure is over: `p` is not in scope after it.
    let after = complete_after_edit(&src, "xs.map(p => p.x).first().");
    lacks(&after, &["y"]);
}

#[test]
fn offers_table_columns_inside_queries() {
    let src = with("fn f(t: Table[Row], n: Int):\n    /*here*/\n    pass\n");
    let items = complete_after_edit(&src, "t.sql(\"SELECT ");
    assert_eq!(labels(&items), ["id", "label", "score"]);
    assert_eq!(items[0].detail.as_deref(), Some("column: Int"));
    let items = complete_after_edit(&src, "t.sql(\"SELECT id FROM t WHERE sc");
    assert_eq!(labels(&items), ["score"]);
    let items = complete_after_edit(&src, "t[\"");
    assert_eq!(labels(&items), ["id", "label", "score"]);
    // Other strings and comments are not code.
    assert!(complete_after_edit(&src, "print(\"hel").is_empty());
    assert!(complete_after_edit(&src, "# let").is_empty());
    // The braces of an f-string are code again.
    has(&complete_after_edit(&src, "let s = f\"{"), &["n", "t"]);
}

#[test]
fn ranks_what_fits_the_expected_type_first() {
    let src = with(
        "fn repeat(count: Int, word: Text) -> Text:\n    word\n\nfn f(a: Int, b: Text) -> Text:\n    /*here*/\n    b\n",
    );
    let rank =
        |items: &[Completion], name: &str| items.iter().find(|c| c.label == name).unwrap().rank;
    let items = complete_after_edit(&src, "repeat(a, ");
    assert_eq!(rank(&items, "b"), 0);
    assert!(rank(&items, "a") > 0, "{items:?}");
    let items = complete_after_edit(&src, "return ");
    assert_eq!(rank(&items, "b"), 0);
    assert!(rank(&items, "a") > 0);
    // Declarations come after names in scope, keywords last.
    let items = complete_after_edit(&src, "");
    assert!(rank(&items, "a") < rank(&items, "repeat"));
    assert!(rank(&items, "repeat") < rank(&items, "while"));
}

#[test]
fn describes_what_it_offers() {
    let src = with("fn f():\n    /*here*/\n    pass\n");
    let items = complete_after_edit(&src, "");
    for name in ["while", "async", "await", "Int", "List", "std"] {
        let c = items.iter().find(|c| c.label == name).unwrap();
        assert!(c.detail.is_some() && c.doc.is_some(), "{c:?}");
    }
    let items = complete_after_edit(&src, "std.text.tr");
    assert!(
        items[0]
            .detail
            .as_deref()
            .unwrap()
            .starts_with("std function")
    );
}

#[test]
fn offers_some_and_ok_in_a_match_arm() {
    let src = with("fn f(o: Int?):\n    match o:\n        /*here*/\n        pass\n");
    let _ = src;
    let src = with("fn f(o: Int?):\n    /*here*/\n    pass\n");
    let items = complete_after_edit(&src, "match o:\n        case ");
    has(&items, &["Some", "None", "Shape"]);
}

#[test]
fn every_builtin_method_offered_is_accepted_by_the_checker() {
    use telvax_hir::{Prim, Ty};
    let i32_ = || Box::new(Ty::Prim(Prim::I32));
    let src = with(
        "fn f(s: Text, xs: List[Int], m: Map[Text, Int], o: Int?, r: Result[Int, Text], t: Table[Row]):\n    pass\n",
    );
    for (var, ty) in [
        ("s", Ty::str()),
        ("xs", Ty::Vec(i32_())),
        ("m", Ty::Map(Box::new(Ty::str()), i32_())),
        ("o", Ty::Option(i32_())),
        ("r", Ty::Result(i32_(), Box::new(Ty::str()))),
        ("t", Ty::Table("Row".into())),
    ] {
        for name in telvax_hir::stdlib::builtin_method_names(&ty) {
            let program = src.replace("    pass\n", &format!("    {var}.{name}\n"));
            let snap = Snapshot::analyze(program);
            assert!(
                snap.diagnostics
                    .iter()
                    .all(|d| !d.message.contains("no method")),
                "{var}.{name}: {:?}",
                snap.diagnostics
            );
        }
    }
}
