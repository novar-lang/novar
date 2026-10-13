use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use novar_pkg::manifest::{Dependency, MANIFEST, validate_name};
use novar_pkg::registry::{self, DirRegistry, Registry};
use novar_pkg::{Manifest, Project};
use semver::{Version, VersionReq};

const USAGE: &str = "\
nvpm: the Novar package manager

usage:
  nvpm new <name> [--lib]            create a project in ./<name>
  nvpm init                          create novar.toml in the current directory
  nvpm add <name>[@req] [--path DIR] add a dependency (registry, or a local directory)
  nvpm remove <name>                 remove a dependency
  nvpm lock                          resolve versions and write novar.lock
  nvpm install                       lock (if needed) and download what is missing
  nvpm tree                          print the dependency tree
  nvpm search <text>                 find packages in a directory registry
  nvpm publish                       publish this package to the registry
  nvpm yank <name> <version> [--undo]
  nvpm check | run [fn] | serve [flags] | build [flags]
                                     install, then run the matching `novar` command

The registry is a directory or an http(s) URL: --registry <where> or NVPM_REGISTRY.";

type Res<T> = Result<T, String>;

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let registry =
        take_flag(&mut args, "--registry").or_else(|| std::env::var("NVPM_REGISTRY").ok());
    match run(args, registry) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) if msg.is_empty() => ExitCode::from(2),
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Removes `--flag value` from `args` and returns the value.
fn take_flag(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.remove(i);
    (i < args.len()).then(|| args.remove(i))
}

fn take_switch(args: &mut Vec<String>, flag: &str) -> bool {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

fn run(mut args: Vec<String>, registry_spec: Option<String>) -> Res<()> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{USAGE}");
        return Ok(());
    }
    let cmd = args.remove(0);
    let reg = registry_spec.as_deref().map(registry::open);
    let reg_ref = reg.as_deref();
    match cmd.as_str() {
        "new" => {
            let lib = take_switch(&mut args, "--lib");
            let name = args
                .first()
                .ok_or_else(|| String::from("usage: nvpm new <name>"))?;
            new_project(Path::new(name), name, lib)
        }
        "init" => {
            let dir = std::env::current_dir().map_err(|e| e.to_string())?;
            let name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("app")
                .replace('-', "_")
                .to_lowercase();
            new_project(&dir, &name, false)
        }
        "add" => add(&mut args, reg_ref),
        "remove" => {
            let name = args
                .first()
                .ok_or_else(|| String::from("usage: nvpm remove <name>"))?;
            let project = current_project()?;
            let mut m = project.manifest.clone();
            if m.dependencies.remove(name).is_none() {
                return Err(format!("`{name}` is not a dependency"));
            }
            m.write(&project.dir).map_err(|e| e.0)?;
            println!("removed {name}");
            Ok(())
        }
        "lock" => {
            let project = current_project()?;
            lock(&project, reg_ref)
        }
        "install" => install(&current_project()?, reg_ref),
        "tree" => {
            print!("{}", current_project()?.tree()?);
            Ok(())
        }
        "search" => {
            let q = args.first().map_or("", String::as_str);
            let spec = registry_spec.ok_or("search needs --registry (a directory)")?;
            let dir = DirRegistry::new(spec);
            let found = dir.search(q);
            if found.is_empty() {
                println!("no packages match `{q}`");
            }
            for name in found {
                let newest = dir
                    .versions(&name)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|e| !e.yanked)
                    .map(|e| e.vers)
                    .max();
                println!("{name} {}", newest.map_or(String::new(), |v| v.to_string()));
            }
            Ok(())
        }
        "publish" => {
            let spec = registry_spec.ok_or("publish needs --registry <directory>")?;
            if spec.starts_with("http") {
                return Err("publishing to an HTTP registry is not supported; publish to a directory and serve it".into());
            }
            let project = current_project()?;
            let entry = registry::publish(Path::new(&spec), &project.dir)?;
            println!(
                "published {} {} (sha256 {})",
                entry.name,
                entry.vers,
                &entry.cksum[..16]
            );
            Ok(())
        }
        "yank" => {
            let undo = take_switch(&mut args, "--undo");
            let spec = registry_spec.ok_or("yank needs --registry <directory>")?;
            let [name, version] = args.as_slice() else {
                return Err("usage: nvpm yank <name> <version> [--undo]".into());
            };
            let version: Version = version
                .parse()
                .map_err(|e| format!("bad version `{version}`: {e}"))?;
            registry::yank(Path::new(&spec), name, &version, !undo)?;
            println!(
                "{} {name} {version}",
                if undo { "unyanked" } else { "yanked" }
            );
            Ok(())
        }
        "check" | "run" | "serve" | "build" => {
            let project = current_project()?;
            if !project.lock_is_fresh() && !project.manifest.dependencies.is_empty() {
                lock(&project, reg_ref)?;
            }
            for fetched in project.install(reg_ref)? {
                eprintln!("downloaded {fetched}");
            }
            let entry = project.entry()?;
            let mut command = Command::new(novar_binary());
            command.arg(&cmd);
            if cmd == "build" && args.first().is_some_and(|a| a == "--serve") {
                args.remove(0);
                command.arg("--serve");
            }
            command.arg(&entry);
            if cmd == "run" && args.is_empty() {
                args.push("main".into());
            }
            let status = command
                .args(&args)
                .status()
                .map_err(|e| format!("cannot run `novar` (set NOVAR_BIN): {e}"))?;
            if status.success() {
                Ok(())
            } else {
                Err(String::new())
            }
        }
        other => {
            eprintln!("unknown command `{other}`\n\n{USAGE}");
            Err(String::new())
        }
    }
}

fn novar_binary() -> PathBuf {
    if let Some(p) = std::env::var_os("NOVAR_BIN") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.with_file_name("novar");
        if sibling.is_file() {
            return sibling;
        }
    }
    PathBuf::from("novar")
}

fn current_project() -> Res<Project> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    Project::discover(&cwd)?
        .ok_or_else(|| format!("no {MANIFEST} here or in a parent directory; run `nvpm init`"))
}

fn new_project(dir: &Path, name: &str, lib: bool) -> Res<()> {
    validate_name(name).map_err(|e| e.0)?;
    if dir.join(MANIFEST).exists() {
        return Err(format!("{} already exists", dir.join(MANIFEST).display()));
    }
    std::fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    Manifest {
        package: novar_pkg::manifest::PackageInfo {
            name: name.to_string(),
            version: Version::new(0, 1, 0),
            description: None,
            entry: None,
        },
        dependencies: Default::default(),
    }
    .write(dir)
    .map_err(|e| e.0)?;
    let (file, body) = if lib {
        (
            "lib.nv",
            "pub fn hello() -> String {\n    \"hello from a library\"\n}\n",
        )
    } else {
        (
            "main.nv",
            "fn main() -> i32 {\n    std::io::println(\"hello, world\");\n    0\n}\n",
        )
    };
    let path = dir.join("src").join(file);
    if !path.exists() {
        std::fs::write(&path, body).map_err(|e| e.to_string())?;
    }
    let ignore = dir.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(ignore, "target/\n.nvpm/\n").map_err(|e| e.to_string())?;
    }
    println!(
        "created {} `{name}`",
        if lib { "library" } else { "binary package" }
    );
    Ok(())
}

fn add(args: &mut Vec<String>, reg: Option<&dyn Registry>) -> Res<()> {
    let path = take_flag(args, "--path");
    let spec = args
        .first()
        .ok_or_else(|| String::from("usage: nvpm add <name>[@req] [--path DIR]"))?
        .clone();
    let (name, req) = match spec.split_once('@') {
        Some((n, r)) => (n.to_string(), Some(r.to_string())),
        None => (spec, None),
    };
    validate_name(&name).map_err(|e| e.0)?;
    let project = current_project()?;
    let mut manifest = project.manifest.clone();
    let dep = match (path, req) {
        (Some(p), _) => {
            let dep_dir = project.dir.join(&p);
            let m =
                Manifest::read(&dep_dir).map_err(|e| format!("{}: {}", dep_dir.display(), e.0))?;
            if m.package.name != name {
                return Err(format!(
                    "{} is the package `{}`, not `{name}`",
                    dep_dir.display(),
                    m.package.name
                ));
            }
            Dependency::Detailed {
                path: Some(p),
                version: None,
            }
        }
        (None, Some(r)) => Dependency::Version(
            r.parse::<VersionReq>()
                .map_err(|e| format!("bad version requirement `{r}`: {e}"))?,
        ),
        (None, None) => {
            // No requirement given: depend on the newest published version, compatibly.
            let reg =
                reg.ok_or("give a version (`nvpm add name@1.2`) or a --registry to look it up")?;
            let newest = reg
                .versions(&name)?
                .into_iter()
                .filter(|e| !e.yanked)
                .map(|e| e.vers)
                .max()
                .ok_or_else(|| format!("`{name}` is not in the registry"))?;
            Dependency::Version(format!("^{newest}").parse().map_err(|e| format!("{e}"))?)
        }
    };
    manifest.dependencies.insert(name.clone(), dep);
    manifest.write(&project.dir).map_err(|e| e.0)?;
    println!("added {name}");
    Ok(())
}

fn lock(project: &Project, reg: Option<&dyn Registry>) -> Res<()> {
    let (resolution, _) = project.lock(reg)?;
    println!("locked {} package(s)", resolution.packages.len());
    for p in &resolution.packages {
        println!("  {} {}", p.name, p.version);
    }
    Ok(())
}

fn install(project: &Project, reg: Option<&dyn Registry>) -> Res<()> {
    if !project.lock_is_fresh() {
        lock(project, reg)?;
    }
    let fetched = project.install(reg)?;
    if fetched.is_empty() {
        println!("everything is installed");
    }
    for f in fetched {
        println!("downloaded {f}");
    }
    Ok(())
}
