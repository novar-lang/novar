//! The language server on multi-file projects.

use std::path::PathBuf;

use telvax_lsp::analysis::Snapshot;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("telvax-lsp-mod-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    fn write(&self, rel: &str, text: &str) -> PathBuf {
        let p = self.0.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, text).unwrap();
        p
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const MAIN: &str = r#"
from util import double, Pair

fn main() -> i32:
    let p = Pair(a: 1, b: 2)
    double(p.a) + double(p.b)
"#;
const UTIL: &str = r#"
type Pair:
    a: i32
    b: i32

fn double(x: i32) -> i32:
    x * 2
"#;

fn project(tag: &str) -> (Dir, PathBuf) {
    let d = Dir::new(tag);
    d.write("util.tvx", UTIL);
    let main = d.write("main.tvx", MAIN);
    (d, main)
}

#[test]
fn imports_resolve_and_the_file_is_clean() {
    let (_d, main) = project("clean");
    let s = Snapshot::analyze_in(MAIN.to_string(), Some(&main));
    assert!(s.diagnostics.is_empty(), "{:?}", s.diagnostics);
    assert!(s.hir.is_some());
}

#[test]
fn without_a_path_the_import_is_reported() {
    let s = Snapshot::analyze(MAIN.to_string());
    assert!(
        s.diagnostics
            .iter()
            .any(|d| d.message.contains("cannot find the module")),
        "{:?}",
        s.diagnostics
    );
}

#[test]
fn hover_shows_the_names_the_user_wrote() {
    let (_d, main) = project("hover");
    let s = Snapshot::analyze_in(MAIN.to_string(), Some(&main));
    let at = MAIN.find("double(p.a)").unwrap() + 2;
    let (text, _) = s.hover(at).unwrap();
    assert!(text.contains("fn util.double(x: i32) -> i32"), "{text}");
    assert!(!text.contains("__"), "{text}");
}

#[test]
fn go_to_definition_crosses_files() {
    let (_d, main) = project("definition");
    let s = Snapshot::analyze_in(MAIN.to_string(), Some(&main));
    let at = MAIN.find("double(p.a)").unwrap() + 2;
    let (file, span) = s.definition_location(at).unwrap();
    assert!(file.ends_with("util.tvx"), "{file}");
    assert_eq!(&UTIL[span], "double");
    let at = MAIN.find("Pair(a: 1").unwrap() + 1;
    let (file, span) = s.definition_location(at).unwrap();
    assert!(
        file.ends_with("util.tvx") && UTIL[span].starts_with("Pair"),
        "{file}"
    );
}

#[test]
fn only_this_documents_problems_and_symbols_are_listed() {
    let d = Dir::new("scope");
    d.write(
        "util.tvx",
        r#"
fn broken() -> i32:
    "nope"
"#,
    );
    let main_text = r#"
import util.broken

fn main() -> i32:
    broken()

fn other():
    pass
"#;
    let main = d.write("main.tvx", main_text);
    let s = Snapshot::analyze_in(main_text.to_string(), Some(&main));
    assert!(
        s.diagnostics.is_empty(),
        "util.tvx's error belongs to util.tvx: {:?}",
        s.diagnostics
    );
    let names: Vec<_> = s.symbols().into_iter().map(|x| x.name).collect();
    assert_eq!(
        names,
        ["main", "other"],
        "imported items are not this document's symbols"
    );

    // Opening the broken file itself reports the error there.
    let util_text = std::fs::read_to_string(d.0.join("util.tvx")).unwrap();
    let u = Snapshot::analyze_in(util_text, Some(&d.0.join("util.tvx")));
    assert!(
        u.diagnostics
            .iter()
            .any(|x| x.message.contains("expected `i32`, found `Text`"))
    );
}

#[test]
fn completion_offers_imported_items_without_mangled_names() {
    let (_d, main) = project("complete");
    let s = Snapshot::analyze_in(MAIN.to_string(), Some(&main));
    let at = MAIN.find("    double(p.a)").unwrap() + 4;
    let items = s.completions(at);
    let labels: Vec<_> = items.iter().map(|c| c.label.as_str()).collect();
    assert!(
        labels.contains(&"double") && labels.contains(&"Pair") && labels.contains(&"main"),
        "{labels:?}"
    );
    assert!(labels.iter().all(|l| !l.contains("__")), "{labels:?}");
    let double = items.iter().find(|c| c.label == "double").unwrap();
    assert!(
        double.detail.as_deref().unwrap().contains("util.double"),
        "{double:?}"
    );
}

#[test]
fn project_dependencies_are_found_through_telvax_toml() {
    let d = Dir::new("pkg");
    d.write("telvax.toml", "[package]\nname = \"app\"\nversion = \"0.1.0\"\n[dependencies]\nshared = { path = \"shared\" }\n");
    d.write(
        "shared/telvax.toml",
        "[package]\nname = \"shared\"\nversion = \"0.1.0\"\n",
    );
    d.write(
        "shared/src/lib.tvx",
        r#"
fn answer() -> i32:
    42
"#,
    );
    // The locked path dependency is what the compiler consults.
    let project = telvax_pkg::Project::open(&d.0).unwrap();
    project.lock(None).unwrap();
    let text = r#"
import shared.answer

fn main() -> i32:
    answer()
"#;
    let main = d.write("src/main.tvx", text);
    let s = Snapshot::analyze_in(text.to_string(), Some(&main));
    assert!(s.diagnostics.is_empty(), "{:?}", s.diagnostics);
    let (hover, _) = s.hover(text.find("answer()").unwrap() + 1).unwrap();
    assert!(hover.contains("fn shared.answer() -> i32"), "{hover}");
}
