//! A project on disk: its manifest, lockfile and installed dependencies.

use std::path::{Path, PathBuf};

use novar_modules::Package;

use crate::lock::{LOCKFILE, Lock};
use crate::manifest::{MANIFEST, Manifest};
use crate::registry::{self, Registry};
use crate::resolver::{self, Resolution};

#[derive(Debug, Clone)]
pub struct Project {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

/// Where downloaded registry packages are unpacked: `$NVPM_HOME` or `~/.nvpm`.
pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("NVPM_HOME") {
        return PathBuf::from(h);
    }
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".nvpm"))
        .unwrap_or_else(|| PathBuf::from(".nvpm"))
}

impl Project {
    /// The project containing `start` (a file or directory): the nearest ancestor with a `novar.toml`.
    pub fn discover(start: &Path) -> Result<Option<Project>, String> {
        let start = std::fs::canonicalize(start).unwrap_or_else(|_| start.to_path_buf());
        let mut dir = if start.is_dir() {
            Some(start.as_path())
        } else {
            start.parent()
        };
        while let Some(d) = dir {
            if d.join(MANIFEST).is_file() {
                let manifest = Manifest::read(d).map_err(|e| e.0)?;
                return Ok(Some(Project {
                    dir: d.to_path_buf(),
                    manifest,
                }));
            }
            dir = d.parent();
        }
        Ok(None)
    }

    pub fn open(dir: &Path) -> Result<Project, String> {
        let dir = std::fs::canonicalize(dir)
            .map_err(|e| format!("cannot open {}: {e}", dir.display()))?;
        let manifest = Manifest::read(&dir).map_err(|e| e.0)?;
        Ok(Project { dir, manifest })
    }

    /// The file to run: `package.entry`, else `src/main.nv`, else `src/lib.nv`.
    pub fn entry(&self) -> Result<PathBuf, String> {
        let candidates: Vec<PathBuf> = match &self.manifest.package.entry {
            Some(e) => vec![self.dir.join(e)],
            None => vec![self.dir.join("src/main.nv"), self.dir.join("src/lib.nv")],
        };
        candidates
            .iter()
            .find(|p| p.is_file())
            .cloned()
            .ok_or_else(|| {
                format!(
                    "no entry file found (looked for {})",
                    candidates
                        .iter()
                        .map(|c| c.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }

    /// Resolves the dependency graph and writes `novar.lock`.
    pub fn lock(&self, registry: Option<&dyn Registry>) -> Result<(Resolution, Lock), String> {
        let resolution = resolver::resolve(&self.dir, &self.manifest, registry)?;
        let lock = resolution.to_lock(&self.manifest);
        lock.write(&self.dir)?;
        Ok((resolution, lock))
    }

    /// Where an installed package's modules live.
    fn package_src(&self, name: &str, source: &str, version: &semver::Version) -> PathBuf {
        match source.strip_prefix("path+") {
            Some(dir) => PathBuf::from(dir).join("src"),
            None => home()
                .join("pkgs")
                .join(format!("{name}-{version}"))
                .join("src"),
        }
    }

    /// Downloads and unpacks every registry package of the lockfile that is not installed yet.
    pub fn install(&self, registry: Option<&dyn Registry>) -> Result<Vec<String>, String> {
        let lock = Lock::read(&self.dir)?
            .ok_or_else(|| format!("no {LOCKFILE}; run `nvpm lock` first"))?;
        let mut fetched = Vec::new();
        for p in &lock.packages {
            if p.source.as_deref() != Some("registry") {
                continue;
            }
            let dest = home()
                .join("pkgs")
                .join(format!("{}-{}", p.name, p.version));
            if dest.join(MANIFEST).is_file() {
                continue;
            }
            let reg = registry.ok_or_else(|| {
                format!(
                    "`{}` is not installed and no registry is configured",
                    p.name
                )
            })?;
            let archive = reg.download(&p.name, &p.version)?;
            let cksum = p
                .checksum
                .as_deref()
                .ok_or_else(|| format!("{LOCKFILE} has no checksum for `{}`", p.name))?;
            registry::unpack(&archive, cksum, &dest)
                .map_err(|e| format!("`{}` {}: {e}", p.name, p.version))?;
            fetched.push(format!("{} {}", p.name, p.version));
        }
        Ok(fetched)
    }

    /// The dependencies for the compiler, from `novar.lock`. Fails when the lockfile is missing or
    /// does not cover the manifest, or a package is not installed.
    pub fn dependency_packages(&self) -> Result<Vec<Package>, String> {
        if self.manifest.dependencies.is_empty() {
            return Ok(Vec::new());
        }
        let lock = Lock::read(&self.dir)?.ok_or_else(|| {
            format!(
                "{} has dependencies but no {LOCKFILE}; run `nvpm install`",
                MANIFEST
            )
        })?;
        let mut out = Vec::new();
        for p in &lock.packages {
            let Some(source) = &p.source else { continue };
            let src_dir = self.package_src(&p.name, source, &p.version);
            if !src_dir.is_dir() {
                return Err(format!(
                    "package `{}` {} is not installed; run `nvpm install`",
                    p.name, p.version
                ));
            }
            out.push(Package {
                name: p.name.clone(),
                src_dir,
            });
        }
        for name in self.manifest.dependencies.keys() {
            if lock.get(name).is_none() {
                return Err(format!(
                    "dependency `{name}` is not in {LOCKFILE}; run `nvpm lock`"
                ));
            }
        }
        Ok(out)
    }

    /// A printable dependency tree.
    pub fn tree(&self) -> Result<String, String> {
        let lock = Lock::read(&self.dir)?
            .ok_or_else(|| format!("no {LOCKFILE}; run `nvpm lock` first"))?;
        let mut out = format!(
            "{} {}\n",
            self.manifest.package.name, self.manifest.package.version
        );
        fn walk(
            lock: &Lock,
            names: &[String],
            prefix: &str,
            seen: &mut Vec<String>,
            out: &mut String,
        ) {
            for (i, name) in names.iter().enumerate() {
                let last = i + 1 == names.len();
                let Some(p) = lock.get(name) else { continue };
                let src = match p.source.as_deref() {
                    Some(s) if s.starts_with("path+") => format!(" ({})", &s[5..]),
                    _ => String::new(),
                };
                let again = seen.contains(name);
                out.push_str(&format!(
                    "{prefix}{} {} {}{src}{}\n",
                    if last { "└──" } else { "├──" },
                    p.name,
                    p.version,
                    if again { " (*)" } else { "" }
                ));
                if !again {
                    seen.push(name.clone());
                    let child = format!("{prefix}{}", if last { "    " } else { "│   " });
                    walk(lock, &p.dependencies, &child, seen, out);
                }
            }
        }
        let roots: Vec<String> = self.manifest.dependencies.keys().cloned().collect();
        walk(&lock, &roots, "", &mut Vec::new(), &mut out);
        Ok(out)
    }

    /// Whether the lockfile covers the manifest with versions that satisfy it.
    pub fn lock_is_fresh(&self) -> bool {
        let Ok(Some(lock)) = Lock::read(&self.dir) else {
            return false;
        };
        self.manifest
            .dependencies
            .iter()
            .all(|(name, dep)| match (lock.get(name), dep.path()) {
                (Some(p), None) => {
                    dep.requirement().matches(&p.version) && p.source.as_deref() == Some("registry")
                }
                (Some(p), Some(_)) => p.source.as_deref().is_some_and(|s| s.starts_with("path+")),
                (None, _) => false,
            })
    }

    pub fn has_registry_deps(&self) -> bool {
        self.manifest
            .dependencies
            .values()
            .any(|d| d.path().is_none())
    }
}
