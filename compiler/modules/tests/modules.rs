use std::path::{Path, PathBuf};

use telvax_modules::{Loaded, Package, load};

struct Project(PathBuf);

impl Project {
    fn new(name: &str) -> Project {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "telvax-mod-{}-{}-{name}",
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
    let (hir, diags) = telvax_hir::lower(&loaded.program);
    messages.extend(diags.into_iter().map(|d| d.message));
    if !messages.is_empty() {
        return (String::new(), messages);
    }
    let interp = telvax_runtime::Interp::new(hir).unwrap();
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
        "main.tvx",
        r#"
import util.double

def main() -> int32:
    return double(21)
"#,
    )
    .file(
        "util.tvx",
        r#"
def double(x: int32) -> int32:
    return x * 2
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "42");
}

#[test]
fn structs_group_imports_and_nested_modules() {
    let p = Project::new("nested");
    p.file(
        "main.tvx",
        r#"
from crate.geometry import Point, norm2
import crate.geometry.shapes.area

def main() -> int32:
    p = Point(x=3, y=4)
    return norm2(p) + area(2, 5)
"#,
    )
    .file(
        "geometry.tvx",
        r#"
class Point:
    x: int32
    y: int32

def norm2(p: Point) -> int32:
    return p.x * p.x + p.y * p.y
"#,
    )
    .file(
        "geometry/shapes.tvx",
        r#"
def area(w: int32, h: int32) -> int32:
    return w * h
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "35");
}

#[test]
fn mod_rs_style_directories_work() {
    let p = Project::new("modfile");
    p.file(
        "main.tvx",
        r#"
import tools.answer

def main() -> int32:
    return answer()
"#,
    )
    .file(
        "tools/mod.tvx",
        r#"
def answer() -> int32:
    return 42
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "42");
}

#[test]
fn dependencies_are_imported_by_package_name() {
    let p = Project::new("deps");
    p.file(
        "app/main.tvx",
        r#"
import mathx.dot
import mathx.vec.scale

def main() -> int32:
    return dot(2, 3) + scale(4)
"#,
    )
    .file(
        "mathx/src/lib.tvx",
        r#"
def dot(a: int32, b: int32) -> int32:
    return a * b
"#,
    )
    .file(
        "mathx/src/vec.tvx",
        r#"
import crate.dot

def scale(x: int32) -> int32:
    return dot(x, 10)
"#,
    );
    let deps = [dep("mathx", &p, "mathx/src")];
    assert_eq!(ok(&load(&p.path("app/main.tvx"), &deps).unwrap()), "46");
}

#[test]
fn same_private_names_in_different_modules_do_not_collide() {
    let p = Project::new("collide");
    p.file(
        "main.tvx",
        r#"
import a.value
import b.other

def helper() -> int32:
    return 1

def main() -> int32:
    return helper() + value() + other()
"#,
    )
    .file(
        "a.tvx",
        r#"
def helper() -> int32:
    return 100

def value() -> int32:
    return helper()
"#,
    )
    .file(
        "b.tvx",
        r#"
def helper() -> int32:
    return 10000

def other() -> int32:
    return helper()
"#,
    );
    assert_eq!(
        ok(&load(&p.path("main.tvx"), &[]).unwrap()),
        "10101",
        "each module keeps its own `helper`"
    );
}

#[test]
fn locals_shadow_imports() {
    let p = Project::new("shadow");
    p.file(
        "main.tvx",
        r#"
import util.double

def main() -> int32:
    double = 5
    return double + 1
"#,
    )
    .file(
        "util.tvx",
        r#"
def double(x: int32) -> int32:
    return x * 2
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "6");
}

#[test]
fn std_imports_survive_merging() {
    let p = Project::new("std");
    p.file(
        "main.tvx",
        r#"
import util.shout

def main() -> str:
    return shout("hi")
"#,
    )
    .file(
        "util.tvx",
        r#"
def shout(s: str) -> str:
    return std.text.uppercase(s)
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "HI");
}

#[test]
fn import_cycles_between_modules_are_allowed() {
    let p = Project::new("cycle");
    p.file(
        "main.tvx",
        r#"
import even.is_even

def main() -> bool:
    return is_even(10)
"#,
    )
    .file(
        "even.tvx",
        r#"
import odd.is_odd

def is_even(n: int32) -> bool:
    if n == 0:
        return True
    else:
        return is_odd(n - 1)
"#,
    )
    .file(
        "odd.tvx",
        r#"
import even.is_even

def is_odd(n: int32) -> bool:
    if n == 0:
        return False
    else:
        return is_even(n - 1)
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "True");
}

fn messages(p: &Project, deps: &[Package]) -> Vec<String> {
    run(&load(&p.path("main.tvx"), deps).unwrap()).1
}

#[test]
fn import_problems_are_diagnosed() {
    let p = Project::new("errors");
    p.file(
        "main.tvx",
        r#"
import nope.f

def main():
    pass
"#,
    );
    let m = messages(&p, &[]);
    assert!(
        m.iter()
            .any(|m| m.contains("cannot find the module `crate.nope`")),
        "{m:?}"
    );

    p.file(
        "main.tvx",
        r#"
import util._hidden

def main():
    pass
"#,
    )
    .file(
        "util.tvx",
        r#"
def _hidden():
    pass

def shown():
    pass
"#,
    );
    assert!(
        messages(&p, &[])
            .iter()
            .any(|m| m.contains("`_hidden` is private"))
    );

    p.file(
        "main.tvx",
        r#"
import util.missing

def main():
    pass
"#,
    );
    assert!(
        messages(&p, &[])
            .iter()
            .any(|m| m.contains("has no function or class `missing`"))
    );

    p.file(
        "main.tvx",
        r#"
import crate

def main():
    pass
"#,
    );
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
        "main.tvx",
        r#"
import util.broken

def main() -> int32:
    return broken()
"#,
    )
    .file(
        "util.tvx",
        r#"
def broken() -> int32:
    return "not a number"
"#,
    );
    let loaded = load(&p.path("main.tvx"), &[]).unwrap();
    assert!(loaded.diagnostics.is_empty());
    let (_, diags) = telvax_hir::lower(&loaded.program);
    let text = loaded.map.render(&diags, false);
    assert!(text.contains("util.tvx:3:12"), "{text}");
    assert!(text.contains("expected `int32`, found `str`"), "{text}");

    // Errors that mention a renamed item show the name the user wrote.
    p.file(
        "main.tvx",
        r#"
import util.broken

def main() -> str:
    return broken()
"#,
    )
    .file(
        "util.tvx",
        r#"
def broken() -> int32:
    return 1
"#,
    );
    let loaded = load(&p.path("main.tvx"), &[]).unwrap();
    let (_, diags) = telvax_hir::lower(&loaded.program);
    assert!(!diags.is_empty());
    let shown = loaded.map.render(&diags, false);
    assert!(shown.contains("main.tvx:5:"), "{shown}");
}

#[test]
fn syntax_errors_in_a_module_are_reported_with_that_file() {
    let p = Project::new("syntax");
    p.file(
        "main.tvx",
        r#"
import util.f

def main() -> int32:
    return f()
"#,
    )
    .file("util.tvx", "pub fn f() -> i32 { let x = ; 1 }\n");
    let loaded = load(&p.path("main.tvx"), &[]).unwrap();
    assert!(!loaded.diagnostics.is_empty());
    assert!(
        loaded
            .map
            .render(&loaded.diagnostics, false)
            .contains("util.tvx:1")
    );
}

#[test]
fn load_text_uses_the_unsaved_buffer() {
    let p = Project::new("buffer");
    p.file(
        "util.tvx",
        r#"
def one() -> int32:
    return 1
"#,
    );
    let loaded = telvax_modules::load_text(
        "<buffer>",
        r#"
import util.one

def main() -> int32:
    return one() + 1
"#,
        Some(Path::new(&p.0)),
        &[],
    );
    assert_eq!(ok(&loaded), "2");
}

#[test]
fn bundled_sources_rebuild_the_same_program_without_the_file_system() {
    use telvax_modules::{MemFiles, load_in};
    let p = Project::new("bundle");
    p.file(
        "app/main.tvx",
        r#"
import mathx.dot
import helper.twice

def main() -> int32:
    return twice(dot(2, 3))
"#,
    )
    .file(
        "app/helper.tvx",
        r#"
def twice(x: int32) -> int32:
    return x * 2
"#,
    )
    .file(
        "mathx/src/lib.tvx",
        r#"
def dot(a: int32, b: int32) -> int32:
    return a * b
"#,
    );
    let deps = [dep("mathx", &p, "mathx/src")];
    let entry = p.path("app/main.tvx");
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
        "main.tvx",
        r#"
from crate.shapes import Shape, area

enum Shape2:
    Dot

def main() -> int32:
    s = Shape.Square(3)
    n = 0
    match s:
        case Shape.Square(k):
            n = k
        case Shape.Dot:
            n = 0
    return area(Shape.Square(4)) + n
"#,
    )
    .file(
        "shapes.tvx",
        r#"
enum Shape:
    Square(int32)
    Dot

def area(s: Shape) -> int32:
    match s:
        case Shape.Square(k):
            return k * k
        case Shape.Dot:
            return 0
"#,
    );
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "19");
}

#[test]
fn impl_blocks_travel_with_their_type() {
    let p = Project::new("impls");
    p.file(
        "main.tvx",
        r#"
import crate.shapes.Rect

class Rect2:
    w: int32

    def area(self) -> int32:
        return self.w * self.w

def area() -> int32:
    return 1

def main() -> int32:
    return Rect.new(3, 4).area() + Rect2(w=2).area() + area()
"#,
    )
    .file(
        "shapes.tvx",
        r#"
class Rect:
    w: int32
    h: int32

def area() -> int32:
    return 100

extend Rect:
    def new(w: int32, h: int32) -> Rect:
        return Rect(w=w, h=h)

    def area(self) -> int32:
        return self.w * self.h
"#,
    );
    // `area` the method, `area` the function of each module, and the one in main do not clash.
    assert_eq!(ok(&load(&p.path("main.tvx"), &[]).unwrap()), "17");
}
