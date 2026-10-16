//! References and rename across the files of a project.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use telvax_lsp::analysis::Snapshot;
use telvax_lsp::workspace;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("telvax-lsp-ws-{}-{tag}", std::process::id()));
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

const UTIL: &str = "\
type Pair:
    a: i32
    b: i32

enum Mode:
    Fast
    Slow

fn double(x: i32) -> i32:
    x * 2
";
const MAIN: &str = "\
from util import double, Pair, Mode

fn pick(m: Mode) -> i32:
    match m:
        case Mode.Fast:
            1
        case Mode.Slow:
            2

fn main() -> i32:
    let p = Pair(a: 1, b: 2)
    double(p.a) + double(p.b) + pick(Mode.Fast)
";
const OTHER: &str = "\
from util import double as twice

fn run() -> i32:
    twice(4)
";

struct Project {
    dir: Dir,
    util: PathBuf,
    main: PathBuf,
    other: PathBuf,
}

fn project(tag: &str) -> Project {
    let dir = Dir::new(tag);
    let util = dir.write("util.tvx", UTIL);
    let main = dir.write("main.tvx", MAIN);
    let other = dir.write("other.tvx", OTHER);
    Project {
        dir,
        util,
        main,
        other,
    }
}

fn at(text: &str, needle: &str, skip: usize) -> usize {
    text.find(needle).unwrap() + skip
}

/// How many places each file has, by file name.
fn places(doc: &Path, text: &str, offset: usize) -> Vec<(String, usize)> {
    let snap = Snapshot::analyze_in(text.to_string(), Some(doc));
    let (item, _) = snap.item_at(offset).expect("an item here");
    let mut out: Vec<(String, usize)> = workspace::find(&item, &snap, doc, &HashMap::new())
        .into_iter()
        .map(|h| {
            let name = h.path.file_name().unwrap().to_string_lossy().to_string();
            (name, h.spans.len())
        })
        .collect();
    out.sort();
    out
}

/// Renames the item at `offset` of `doc`, writes the result and checks every file is clean.
fn rename(p: &Project, doc: &Path, text: &str, offset: usize, name: &str) -> Vec<(String, String)> {
    let snap = Snapshot::analyze_in(text.to_string(), Some(doc));
    let (item, _) = snap.item_at(offset).expect("an item here");
    let files = workspace::rename(&item, name, &snap, doc, &HashMap::new()).unwrap();
    let mut out = Vec::new();
    for file in files {
        let (path, mut new, mut edits) = (file.path, file.text, file.edits);
        edits.sort_by_key(|(s, _)| std::cmp::Reverse(s.start));
        for (span, t) in edits {
            new.replace_range(span, &t);
        }
        std::fs::write(&path, &new).unwrap();
        out.push((path.file_name().unwrap().to_string_lossy().to_string(), new));
    }
    for f in [&p.util, &p.main, &p.other] {
        let text = std::fs::read_to_string(f).unwrap();
        let s = Snapshot::analyze_in(text.clone(), Some(f));
        assert!(
            s.diagnostics.is_empty(),
            "{f:?}: {:?}\n{text}",
            s.diagnostics
        );
    }
    out.sort();
    out
}

#[test]
fn a_function_is_found_in_every_file_that_imports_it() {
    let p = project("fn-refs");
    let from_main = places(&p.main, MAIN, at(MAIN, "double(p.a)", 1));
    // util: the declaration; main: the import and two calls; other: the import and the call.
    assert_eq!(
        from_main,
        [
            ("main.tvx".into(), 3),
            ("other.tvx".into(), 2),
            ("util.tvx".into(), 1)
        ]
    );
    let from_util = places(&p.util, UTIL, at(UTIL, "double", 1));
    assert_eq!(from_util, from_main);
}

#[test]
fn renaming_a_function_changes_the_import_and_keeps_the_alias() {
    let p = project("fn-rename");
    let files = rename(&p, &p.main, MAIN, at(MAIN, "double(p.a)", 1), "triple");
    let get = |n: &str| files.iter().find(|(f, _)| f == n).unwrap().1.clone();
    assert!(get("util.tvx").contains("fn triple(x: i32)"));
    assert!(get("main.tvx").contains("from util import triple, Pair, Mode"));
    assert!(get("main.tvx").contains("triple(p.a) + triple(p.b)"));
    // `as twice` stays: only the name before it is changed.
    assert!(get("other.tvx").contains("from util import triple as twice"));
    assert!(get("other.tvx").contains("twice(4)"));
}

#[test]
fn types_variants_and_fields_are_renamed_in_all_files() {
    let p = project("members");
    // The enum: its declaration, the import, the annotation and the three paths.
    let files = rename(&p, &p.main, MAIN, at(MAIN, "m: Mode", 3), "Kind");
    let get = |n: &str| files.iter().find(|(f, _)| f == n).unwrap().1.clone();
    assert!(get("util.tvx").contains("enum Kind:"));
    assert!(get("main.tvx").contains("pick(m: Kind)") && get("main.tvx").contains("Kind.Fast"));

    // A variant, asked for from the file that declares it.
    let util = std::fs::read_to_string(&p.util).unwrap();
    let files = rename(&p, &p.util, &util, at(&util, "Fast", 1), "Quick");
    let main = &files.iter().find(|(f, _)| f == "main.tvx").unwrap().1;
    assert_eq!(main.matches("Kind.Quick").count(), 2, "{main}");

    // A field, asked for from a file that only builds the type and reads the field.
    let main = std::fs::read_to_string(&p.main).unwrap();
    let files = rename(&p, &p.main, &main, at(&main, "Pair(a:", 5), "first");
    let main = &files.iter().find(|(f, _)| f == "main.tvx").unwrap().1;
    assert!(
        main.contains("Pair(first: 1") && main.contains("p.first"),
        "{main}"
    );
    assert!(
        files
            .iter()
            .find(|(f, _)| f == "util.tvx")
            .unwrap()
            .1
            .contains("first: i32")
    );
}

#[test]
fn text_open_in_the_editor_wins_over_the_file_on_disk() {
    let p = project("open");
    let snap = Snapshot::analyze_in(UTIL.to_string(), Some(&p.util));
    let (item, _) = snap.item_at(at(UTIL, "double", 1)).unwrap();
    let unsaved = format!("{OTHER}\nfn again() -> i32:\n    twice(5)\n");
    let open = HashMap::from([(p.other.clone(), unsaved)]);
    let hits = workspace::find(&item, &snap, &p.util, &open);
    let other = hits.iter().find(|h| h.path == p.other).unwrap();
    assert_eq!(other.spans.len(), 3, "{other:?}");
}

#[test]
fn go_to_definition_of_an_imported_type_and_a_dependency_is_not_renamed() {
    let p = project("def");
    let snap = Snapshot::analyze_in(MAIN.to_string(), Some(&p.main));
    let (file, span) = snap
        .definition_location(at(MAIN, "m: Mode", 3))
        .expect("definition");
    assert!(file.ends_with("util.tvx"), "{file}");
    assert_eq!(&UTIL[span], "Mode");

    // A file outside the project (here, outside the directory with `telvax.toml`) is refused.
    p.dir.write(
        "app/telvax.toml",
        "[package]\nname = \"app\"\nversion = \"0.1.0\"\n",
    );
    let app_main = p.dir.write(
        "app/main.tvx",
        "from util import double\n\nfn main() -> i32:\n    double(1)\n",
    );
    let text = std::fs::read_to_string(&app_main).unwrap();
    let snap = Snapshot::analyze_in(text.clone(), Some(&app_main));
    // The compiler does not find the module from `app/`, but the import line still names it.
    let (item, _) = snap.item_at(at(&text, "double", 1)).expect("an item here");
    let message = workspace::rename(&item, "triple", &snap, &app_main, &HashMap::new())
        .expect_err("util.tvx is outside the project");
    assert!(message.contains("outside this project"), "{message}");
}
