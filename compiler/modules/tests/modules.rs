use std::path::{Path, PathBuf};

use novar_modules::{Loaded, Package, load};

struct Project(PathBuf);

impl Project {
    fn new(name: &str) -> Project {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "novar-mod-{}-{}-{name}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Project(dir)
    }

    fn file(&self, rel: &str, text: &str) -> &Project {
        let path = self.0.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
        self
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn dep(name: &str, project: &Project, src: &str) -> Package {
    Package {
        name: name.into(),
        src_dir: project.path(src),
    }
}

/// Loads, type-checks and returns (value of `main` as text, all messages).
fn run(loaded: &Loaded) -> (String, Vec<String>) {
    let mut messages: Vec<String> = loaded
        .diagnostics
        .iter()
        .map(|d| d.message.clone())
        .collect();
    let (hir, diags) = novar_hir::lower(&loaded.program);
    messages.extend(diags.into_iter().map(|d| d.message));
    if !messages.is_empty() {
        return (String::new(), messages);
    }
    let interp = novar_runtime::Interp::new(hir).unwrap();
    (interp.call("main", vec![]).unwrap().to_string(), messages)
}

fn ok(loaded: &Loaded) -> String {
    let (value, messages) = run(loaded);
    assert!(messages.is_empty(), "{messages:#?}");
    value
}

#[test]
fn sibling_modules_are_imported_by_path() {
    let p = Project::new("sibling");
    p.file(
        "main.nv",
        "import util::double;\nfn main() -> i32 { double(21) }\n",
    )
    .file("util.nv", "pub fn double(x: i32) -> i32 { x * 2 }\n");
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "42");
}

#[test]
fn structs_group_imports_and_nested_modules() {
    let p = Project::new("nested");
    p.file(
        "main.nv",
        "import crate::geometry::{Point, norm2};\nimport crate::geometry::shapes::area;\n\
         fn main() -> i32 { let p = Point { x: 3, y: 4 }; norm2(p) + area(2, 5) }\n",
    )
    .file("geometry.nv", "pub struct Point { x: i32, y: i32 }\npub fn norm2(p: Point) -> i32 { p.x * p.x + p.y * p.y }\n")
    .file("geometry/shapes.nv", "pub fn area(w: i32, h: i32) -> i32 { w * h }\n");
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "35");
}

#[test]
fn mod_rs_style_directories_work() {
    let p = Project::new("modfile");
    p.file(
        "main.nv",
        "import tools::answer;\nfn main() -> i32 { answer() }\n",
    )
    .file("tools/mod.nv", "pub fn answer() -> i32 { 42 }\n");
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "42");
}

#[test]
fn dependencies_are_imported_by_package_name() {
    let p = Project::new("deps");
    p.file("app/main.nv", "import mathx::dot;\nimport mathx::vec::scale;\nfn main() -> i32 { dot(2, 3) + scale(4) }\n")
        .file("mathx/src/lib.nv", "pub fn dot(a: i32, b: i32) -> i32 { a * b }\n")
        .file("mathx/src/vec.nv", "import crate::dot;\npub fn scale(x: i32) -> i32 { dot(x, 10) }\n");
    let deps = [dep("mathx", &p, "mathx/src")];
    assert_eq!(ok(&load(&p.path("app/main.nv"), &deps).unwrap()), "46");
}

#[test]
fn same_private_names_in_different_modules_do_not_collide() {
    let p = Project::new("collide");
    p.file("main.nv", "import a::value;\nimport b::other;\nfn helper() -> i32 { 1 }\nfn main() -> i32 { helper() + value() + other() }\n")
        .file("a.nv", "fn helper() -> i32 { 100 }\npub fn value() -> i32 { helper() }\n")
        .file("b.nv", "fn helper() -> i32 { 10000 }\npub fn other() -> i32 { helper() }\n");
    assert_eq!(
        ok(&load(&p.path("main.nv"), &[]).unwrap()),
        "10101",
        "each module keeps its own `helper`"
    );
}

#[test]
fn locals_shadow_imports() {
    let p = Project::new("shadow");
    p.file(
        "main.nv",
        "import util::double;\nfn main() -> i32 { let double = 5; double + 1 }\n",
    )
    .file("util.nv", "pub fn double(x: i32) -> i32 { x * 2 }\n");
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "6");
}

#[test]
fn std_imports_survive_merging() {
    let p = Project::new("std");
    p.file(
        "main.nv",
        "import util::shout;\nfn main() -> String { shout(\"hi\") }\n",
    )
    .file(
        "util.nv",
        "pub fn shout(s: String) -> String { std::text::uppercase(s) }\n",
    );
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "HI");
}

#[test]
fn import_cycles_between_modules_are_allowed() {
    let p = Project::new("cycle");
    p.file("main.nv", "import even::is_even;\nfn main() -> bool { is_even(10) }\n")
        .file("even.nv", "import odd::is_odd;\npub fn is_even(n: i32) -> bool { if n == 0 { true } else { is_odd(n - 1) } }\n")
        .file("odd.nv", "import even::is_even;\npub fn is_odd(n: i32) -> bool { if n == 0 { false } else { is_even(n - 1) } }\n");
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "true");
}

fn messages(p: &Project, deps: &[Package]) -> Vec<String> {
    run(&load(&p.path("main.nv"), deps).unwrap()).1
}

#[test]
fn import_problems_are_diagnosed() {
    let p = Project::new("errors");
    p.file("main.nv", "import nope::f;\nfn main() {}\n");
    let m = messages(&p, &[]);
    assert!(
        m.iter()
            .any(|m| m.contains("cannot find the module `crate::nope`")),
        "{m:?}"
    );

    p.file("main.nv", "import util::hidden;\nfn main() {}\n")
        .file("util.nv", "fn hidden() {}\npub fn shown() {}\n");
    assert!(
        messages(&p, &[])
            .iter()
            .any(|m| m.contains("`hidden` is private"))
    );

    p.file("main.nv", "import util::missing;\nfn main() {}\n");
    assert!(
        messages(&p, &[])
            .iter()
            .any(|m| m.contains("has no function or struct `missing`"))
    );

    p.file("main.nv", "import crate;\nfn main() {}\n");
    assert!(
        messages(&p, &[])
            .iter()
            .any(|m| m.contains("must name an item"))
    );
}

#[test]
fn diagnostics_point_into_the_right_file_with_original_names() {
    let p = Project::new("render");
    p.file(
        "main.nv",
        "import util::broken;\nfn main() -> i32 { broken() }\n",
    )
    .file(
        "util.nv",
        "pub fn broken() -> i32 {\n    \"not a number\"\n}\n",
    );
    let loaded = load(&p.path("main.nv"), &[]).unwrap();
    assert!(loaded.diagnostics.is_empty());
    let (_, diags) = novar_hir::lower(&loaded.program);
    let text = loaded.map.render(&diags, false);
    assert!(text.contains("util.nv:2:5"), "{text}");
    assert!(text.contains("expected `i32`, found `String`"), "{text}");

    // Errors that mention a renamed item show the name the user wrote.
    p.file(
        "main.nv",
        "import util::broken;\nfn main() -> String { broken() }\n",
    )
    .file("util.nv", "pub fn broken() -> i32 { 1 }\n");
    let loaded = load(&p.path("main.nv"), &[]).unwrap();
    let (_, diags) = novar_hir::lower(&loaded.program);
    assert!(!diags.is_empty());
    let shown = loaded.map.render(&diags, false);
    assert!(shown.contains("main.nv:2:"), "{shown}");
}

#[test]
fn syntax_errors_in_a_module_are_reported_with_that_file() {
    let p = Project::new("syntax");
    p.file("main.nv", "import util::f;\nfn main() -> i32 { f() }\n")
        .file("util.nv", "pub fn f() -> i32 { let x = ; 1 }\n");
    let loaded = load(&p.path("main.nv"), &[]).unwrap();
    assert!(!loaded.diagnostics.is_empty());
    assert!(
        loaded
            .map
            .render(&loaded.diagnostics, false)
            .contains("util.nv:1")
    );
}

#[test]
fn load_text_uses_the_unsaved_buffer() {
    let p = Project::new("buffer");
    p.file("util.nv", "pub fn one() -> i32 { 1 }\n");
    let loaded = novar_modules::load_text(
        "<buffer>",
        "import util::one;\nfn main() -> i32 { one() + 1 }\n",
        Some(Path::new(&p.0)),
        &[],
    );
    assert_eq!(ok(&loaded), "2");
}

#[test]
fn bundled_sources_rebuild_the_same_program_without_the_file_system() {
    use novar_modules::{MemFiles, load_in};
    let p = Project::new("bundle");
    p.file(
        "app/main.nv",
        "import mathx::dot;\nimport helper::twice;\nfn main() -> i32 { twice(dot(2, 3)) }\n",
    )
    .file("app/helper.nv", "pub fn twice(x: i32) -> i32 { x * 2 }\n")
    .file(
        "mathx/src/lib.nv",
        "pub fn dot(a: i32, b: i32) -> i32 { a * b }\n",
    );
    let deps = [dep("mathx", &p, "mathx/src")];
    let entry = p.path("app/main.nv");
    let first = load(&entry, &deps).unwrap();
    assert_eq!(ok(&first), "12");

    let mem = MemFiles(
        first
            .sources()
            .into_iter()
            .map(|(n, t)| (PathBuf::from(n), t))
            .collect(),
    );
    let text = std::fs::read_to_string(&entry).unwrap();
    drop(p); // the files are gone from disk
    let again = load_in(
        &mem,
        &entry.display().to_string(),
        &text,
        entry.parent(),
        &deps,
    );
    assert!(again.diagnostics.is_empty(), "{:?}", again.diagnostics);
    assert_eq!(ok(&again), "12");
}

#[test]
fn enums_cross_modules_and_do_not_collide() {
    let p = Project::new("enums");
    p.file(
        "main.nv",
        "import crate::shapes::{Shape, area};\n\
         enum Shape2 { Dot }\n\
         fn main() -> i32 {\n\
             let s = Shape::Square(3);\n\
             let n = match s { Shape::Square(k) => k, Shape::Dot => 0 };\n\
             area(Shape::Square(4)) + n\n\
         }\n",
    )
    .file(
        "shapes.nv",
        "pub enum Shape { Square(i32), Dot }\n\
         pub fn area(s: Shape) -> i32 { match s { Shape::Square(k) => k * k, Shape::Dot => 0 } }\n",
    );
    assert_eq!(ok(&load(&p.path("main.nv"), &[]).unwrap()), "19");
}
