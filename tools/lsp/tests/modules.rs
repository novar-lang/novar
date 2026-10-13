//! The language server on multi-file projects.

use std::path::PathBuf;

use novar_lsp::analysis::Snapshot;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("novar-lsp-mod-{}-{tag}", std::process::id()));
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

const MAIN: &str = "import util::{double, Pair};\n\nfn main() -> i32 {\n    let p = Pair { a: 1, b: 2 };\n    double(p.a) + double(p.b)\n}\n";
const UTIL: &str =
    "pub struct Pair { a: i32, b: i32 }\n\npub fn double(x: i32) -> i32 {\n    x * 2\n}\n";

fn project(tag: &str) -> (Dir, PathBuf) {
    let d = Dir::new(tag);
    d.write("util.nv", UTIL);
    let main = d.write("main.nv", MAIN);
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
    assert!(text.contains("fn util::double(x: i32) -> i32"), "{text}");
    assert!(!text.contains("__"), "{text}");
}

#[test]
fn go_to_definition_crosses_files() {
    let (_d, main) = project("definition");
    let s = Snapshot::analyze_in(MAIN.to_string(), Some(&main));
    let at = MAIN.find("double(p.a)").unwrap() + 2;
    let (file, span) = s.definition_location(at).unwrap();
    assert!(file.ends_with("util.nv"), "{file}");
    assert_eq!(&UTIL[span], "double");
    let at = MAIN.find("Pair {").unwrap() + 1;
    let (file, span) = s.definition_location(at).unwrap();
    assert!(
        file.ends_with("util.nv") && UTIL[span].starts_with("Pair"),
        "{file}"
    );
}

#[test]
fn only_this_documents_problems_and_symbols_are_listed() {
    let d = Dir::new("scope");
    d.write("util.nv", "pub fn broken() -> i32 {\n    \"nope\"\n}\n");
    let main_text = "import util::broken;\nfn main() -> i32 { broken() }\nfn other() {}\n";
    let main = d.write("main.nv", main_text);
    let s = Snapshot::analyze_in(main_text.to_string(), Some(&main));
    assert!(
        s.diagnostics.is_empty(),
        "util.nv's error belongs to util.nv: {:?}",
        s.diagnostics
    );
    let names: Vec<_> = s.symbols().into_iter().map(|x| x.name).collect();
    assert_eq!(
        names,
        ["main", "other"],
        "imported items are not this document's symbols"
    );

    // Opening the broken file itself reports the error there.
    let util_text = std::fs::read_to_string(d.0.join("util.nv")).unwrap();
    let u = Snapshot::analyze_in(util_text, Some(&d.0.join("util.nv")));
    assert!(
        u.diagnostics
            .iter()
            .any(|x| x.message.contains("expected `i32`, found `String`"))
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
        double.detail.as_deref().unwrap().contains("util::double"),
        "{double:?}"
    );
}

#[test]
fn project_dependencies_are_found_through_novar_toml() {
    let d = Dir::new("pkg");
    d.write("novar.toml", "[package]\nname = \"app\"\nversion = \"0.1.0\"\n[dependencies]\nshared = { path = \"shared\" }\n");
    d.write(
        "shared/novar.toml",
        "[package]\nname = \"shared\"\nversion = \"0.1.0\"\n",
    );
    d.write("shared/src/lib.nv", "pub fn answer() -> i32 { 42 }\n");
    // The locked path dependency is what the compiler consults.
    let project = novar_pkg::Project::open(&d.0).unwrap();
    project.lock(None).unwrap();
    let text = "import shared::answer;\nfn main() -> i32 { answer() }\n";
    let main = d.write("src/main.nv", text);
    let s = Snapshot::analyze_in(text.to_string(), Some(&main));
    assert!(s.diagnostics.is_empty(), "{:?}", s.diagnostics);
    let (hover, _) = s.hover(text.find("answer()").unwrap() + 1).unwrap();
    assert!(hover.contains("fn shared::answer() -> i32"), "{hover}");
}
