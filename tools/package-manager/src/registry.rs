//! Registries: where published packages live.
//!
//! The layout works as a plain directory and as a static website (no server logic needed):
//!
//! ```text
//! index/<name>.json              one JSON line per published version
//! packages/<name>-<version>.nvpkg   gzip-compressed tar: telvax.toml + src/**
//! ```

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::manifest::{Dependency, MANIFEST, Manifest};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexDep {
    pub name: String,
    pub req: VersionReq,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexEntry {
    pub name: String,
    pub vers: Version,
    pub deps: Vec<IndexDep>,
    /// Hex SHA-256 of the package archive.
    pub cksum: String,
    #[serde(default)]
    pub yanked: bool,
}

pub trait Registry {
    /// Every published version of `name` (empty when the package is unknown).
    fn versions(&self, name: &str) -> Result<Vec<IndexEntry>, String>;
    /// The package archive of `name` at `version`.
    fn download(&self, name: &str, version: &Version) -> Result<Vec<u8>, String>;
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn parse_index(name: &str, text: &str) -> Result<Vec<IndexEntry>, String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).map_err(|e| format!("corrupt registry index for `{name}`: {e}"))
        })
        .collect()
}

/// A registry in a local directory.
#[derive(Debug, Clone)]
pub struct DirRegistry {
    pub root: PathBuf,
}

impl DirRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn index_file(&self, name: &str) -> PathBuf {
        self.root.join("index").join(format!("{name}.json"))
    }

    fn package_file(&self, name: &str, version: &Version) -> PathBuf {
        self.root
            .join("packages")
            .join(format!("{name}-{version}.nvpkg"))
    }

    /// Names of the published packages containing `query`.
    pub fn search(&self, query: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.root.join("index"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                e.path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
            })
            .filter(|n| n.contains(query))
            .collect();
        names.sort();
        names
    }
}

impl Registry for DirRegistry {
    fn versions(&self, name: &str) -> Result<Vec<IndexEntry>, String> {
        match std::fs::read_to_string(self.index_file(name)) {
            Ok(text) => parse_index(name, &text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(format!("cannot read the index for `{name}`: {e}")),
        }
    }

    fn download(&self, name: &str, version: &Version) -> Result<Vec<u8>, String> {
        let file = self.package_file(name, version);
        std::fs::read(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))
    }
}

/// A registry served over HTTP(S) with the same layout (any static file host works).
#[derive(Debug, Clone)]
pub struct HttpRegistry {
    pub base: String,
}

impl HttpRegistry {
    fn get(&self, path: &str) -> Result<Option<Vec<u8>>, String> {
        let url = format!("{}/{path}", self.base.trim_end_matches('/'));
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into();
        let mut resp = agent
            .get(&url)
            .call()
            .map_err(|e| format!("request to {url} failed: {e}"))?;
        match resp.status().as_u16() {
            200 => resp
                .body_mut()
                .read_to_vec()
                .map(Some)
                .map_err(|e| format!("cannot read {url}: {e}")),
            404 => Ok(None),
            code => Err(format!("{url} answered HTTP {code}")),
        }
    }
}

impl Registry for HttpRegistry {
    fn versions(&self, name: &str) -> Result<Vec<IndexEntry>, String> {
        match self.get(&format!("index/{name}.json"))? {
            Some(bytes) => parse_index(name, &String::from_utf8_lossy(&bytes)),
            None => Ok(Vec::new()),
        }
    }

    fn download(&self, name: &str, version: &Version) -> Result<Vec<u8>, String> {
        self.get(&format!("packages/{name}-{version}.nvpkg"))?
            .ok_or_else(|| format!("package `{name}` {version} is not in the registry"))
    }
}

/// Opens a registry from a directory path or an `http(s)://` URL.
pub fn open(spec: &str) -> Box<dyn Registry> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        Box::new(HttpRegistry {
            base: spec.to_string(),
        })
    } else {
        Box::new(DirRegistry::new(spec))
    }
}

fn collect(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect(&path, root, out)?;
        } else if path.extension().is_some_and(|x| x == "tvx") {
            out.push(path.strip_prefix(root).expect("inside root").to_path_buf());
        }
    }
    Ok(())
}

/// Archives the project at `dir`: its manifest, `src/**/*.tvx` and README / LICENSE files.
pub fn pack(dir: &Path) -> Result<(Manifest, Vec<u8>), String> {
    let manifest = Manifest::read(dir).map_err(|e| e.0)?;
    if let Some((name, _)) = manifest
        .dependencies
        .iter()
        .find(|(_, d)| d.path().is_some())
    {
        return Err(format!(
            "`{name}` is a path dependency; published packages may only depend on registry packages"
        ));
    }
    let src = dir.join("src");
    if !src.is_dir() {
        return Err("the package has no `src/` directory".into());
    }
    let mut files = vec![PathBuf::from(MANIFEST)];
    collect(&src, dir, &mut files).map_err(|e| e.to_string())?;
    if files.len() == 1 {
        return Err("`src/` contains no .tvx files".into());
    }
    for extra in ["README.md", "LICENSE", "LICENSE-MIT", "LICENSE-APACHE"] {
        if dir.join(extra).is_file() {
            files.push(PathBuf::from(extra));
        }
    }
    let mut tar = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::default()));
    for f in &files {
        tar.append_path_with_name(dir.join(f), f)
            .map_err(|e| format!("cannot archive {}: {e}", f.display()))?;
    }
    let bytes = tar
        .into_inner()
        .and_then(|gz| gz.finish())
        .map_err(|e| e.to_string())?;
    Ok((manifest, bytes))
}

/// Publishes the project at `project` to the directory registry at `registry`.
pub fn publish(registry: &Path, project: &Path) -> Result<IndexEntry, String> {
    let (manifest, archive) = pack(project)?;
    let reg = DirRegistry::new(registry);
    let name = manifest.package.name.clone();
    let version = manifest.package.version.clone();
    if reg.versions(&name)?.iter().any(|e| e.vers == version) {
        return Err(format!(
            "`{name}` {version} is already published; bump the version"
        ));
    }
    let entry = IndexEntry {
        name: name.clone(),
        vers: version.clone(),
        deps: manifest
            .dependencies
            .iter()
            .map(|(n, d): (&String, &Dependency)| IndexDep {
                name: n.clone(),
                req: d.requirement(),
            })
            .collect(),
        cksum: sha256_hex(&archive),
        yanked: false,
    };
    std::fs::create_dir_all(registry.join("index"))
        .and_then(|()| std::fs::create_dir_all(registry.join("packages")))
        .map_err(|e| e.to_string())?;
    std::fs::write(reg.package_file(&name, &version), &archive).map_err(|e| e.to_string())?;
    let line = serde_json::to_string(&entry).map_err(|e| e.to_string())?;
    let index = reg.index_file(&name);
    let mut text = std::fs::read_to_string(&index).unwrap_or_default();
    text.push_str(&line);
    text.push('\n');
    std::fs::write(&index, text).map_err(|e| e.to_string())?;
    Ok(entry)
}

/// Marks a published version as yanked: it stays installable from lockfiles but new resolutions
/// skip it.
pub fn yank(registry: &Path, name: &str, version: &Version, yanked: bool) -> Result<(), String> {
    let reg = DirRegistry::new(registry);
    let mut entries = reg.versions(name)?;
    let entry = entries
        .iter_mut()
        .find(|e| &e.vers == version)
        .ok_or_else(|| format!("`{name}` {version} is not published"))?;
    entry.yanked = yanked;
    let text: String = entries
        .iter()
        .map(|e| serde_json::to_string(e).map(|l| l + "\n"))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    std::fs::write(reg.index_file(name), text).map_err(|e| e.to_string())
}

/// Extracts `archive` into `dest` after checking it against `cksum`.
pub fn unpack(archive: &[u8], cksum: &str, dest: &Path) -> Result<(), String> {
    let actual = sha256_hex(archive);
    if actual != cksum {
        return Err(format!(
            "checksum mismatch: the index says {cksum}, the download is {actual}"
        ));
    }
    let _ = std::fs::remove_dir_all(dest);
    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    let mut tar = tar::Archive::new(GzDecoder::new(archive));
    for entry in tar.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path().map_err(|e| e.to_string())?.into_owned();
        if path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(format!("unsafe path in the archive: {}", path.display()));
        }
        entry.unpack_in(dest).map_err(|e| e.to_string())?;
    }
    let mut buf = String::new();
    let _ = std::fs::File::open(dest.join(MANIFEST))
        .and_then(|mut f| f.read_to_string(&mut buf))
        .map_err(|_| "the package has no telvax.toml".to_string())?;
    Ok(())
}
