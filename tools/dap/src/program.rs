//! The program being debugged: loaded, checked, and mapped between source lines and the byte
//! offsets the interpreter reports.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use telvax_diagnostics::Diagnostic;
use telvax_hir::hir;
use telvax_modules::SourceMap;

/// Where the statements of the program are, by file and line.
#[derive(Debug)]
pub struct Program {
    pub map: SourceMap,
    /// For each file, the lines that start a statement, with the offsets of those statements.
    lines: BTreeMap<String, BTreeMap<usize, Vec<usize>>>,
    /// A file as the editor writes its path -> the name the program knows it by.
    names: HashMap<String, String>,
}

impl Program {
    /// Loads `file` with its modules and the dependencies of its project, and checks it.
    /// `Err` is the diagnostics, rendered, or what went wrong.
    pub fn load(file: &str) -> Result<(Program, hir::Program), String> {
        let path = Path::new(file);
        let deps = match telvax_pkg::Project::discover(path) {
            Ok(Some(project)) => project.dependency_packages()?,
            Ok(None) => Vec::new(),
            Err(e) => return Err(e),
        };
        let loaded = telvax_modules::load(path, &deps)?;
        let (hir, diags) = telvax_hir::lower(&loaded.program);
        let all: Vec<Diagnostic> = loaded.diagnostics.iter().cloned().chain(diags).collect();
        if all.iter().any(Diagnostic::is_error) {
            return Err(loaded.map.render(&all, false));
        }
        let mut program = Program {
            map: loaded.map,
            lines: BTreeMap::new(),
            names: HashMap::new(),
        };
        for f in &program.map.files {
            program.names.insert(canonical(&f.name), f.name.clone());
        }
        for offset in telvax_hir::coverage::points(&hir, |_| false) {
            if let Some((file, line, _)) = program.position(offset) {
                program
                    .lines
                    .entry(file)
                    .or_default()
                    .entry(line)
                    .or_default()
                    .push(offset);
            }
        }
        Ok((program, hir))
    }

    /// The file, line and column (all from 1) of `offset`.
    pub fn position(&self, offset: usize) -> Option<(String, usize, usize)> {
        let (file, local) = self.map.locate(offset)?;
        let before = file.text.get(..local)?;
        let line = before.matches('\n').count() + 1;
        let column = local - before.rfind('\n').map_or(0, |i| i + 1) + 1;
        Some((file.name.clone(), line, column))
    }

    /// The statements a breakpoint on `line` of `path` stops at, and the line they are on: the
    /// line itself, or the next one that has a statement. `None` when the file has no statement
    /// there or after.
    pub fn breakpoint(&self, path: &str, line: usize) -> Option<(usize, &[usize])> {
        let name = self.names.get(&canonical(path))?;
        let (line, offsets) = self.lines.get(name)?.range(line..).next()?;
        Some((*line, offsets))
    }
}

/// `path` in a form that two spellings of the same file share.
fn canonical(path: &str) -> String {
    std::fs::canonicalize(path)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.to_string())
}
