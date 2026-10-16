//! `telvax doctor`: what this binary can do and how it is set up, for bug reports and for
//! working out why `build` or `--live` does not do what you expect.

/// The commands `telvax` knows, for the "did you mean" of an unknown one.
pub const COMMANDS: &[&str] = &[
    "check", "lint", "fmt", "test", "run", "profile", "dap", "lsp", "worker", "serve", "build",
    "jit", "doctor", "version", "help",
];

/// The closest command to `word`, if one is near enough to be a typing slip.
pub fn closest_command(word: &str) -> Option<&'static str> {
    COMMANDS
        .iter()
        .map(|c| (distance(word, c), *c))
        .filter(|(d, _)| *d <= if word.chars().count() >= 4 { 2 } else { 1 })
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut diagonal = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let above = row[j];
            row[j] = (row[j] + 1)
                .min(row[j - 1] + 1)
                .min(diagonal + usize::from(a[i - 1] != b[j - 1]));
            diagonal = above;
        }
    }
    row[b.len()]
}

fn yes_no(on: bool) -> &'static str {
    if on { "yes" } else { "no" }
}

/// Whether an environment variable is set to something, without showing it (it may be a key).
fn is_set(name: &str) -> &'static str {
    if std::env::var(name).is_ok_and(|v| !v.is_empty()) {
        "set"
    } else {
        "not set"
    }
}

pub fn report() -> String {
    let editions = telvax_pkg::manifest::EDITIONS.join(", ");
    let value = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let mut out = String::new();
    let mut line = |text: String| {
        out.push_str(&text);
        out.push('\n');
    };
    line(format!("telvax {}", env!("CARGO_PKG_VERSION")));
    line(format!(
        "syntax editions read: {editions} (new packages use {})",
        telvax_pkg::manifest::CURRENT_EDITION
    ));
    line(String::new());
    line("built with:".into());
    line(format!(
        "  tables (CSV, Parquet, SQL)  {}",
        yes_no(cfg!(feature = "tables"))
    ));
    line(format!(
        "  live services (--live)      {}",
        yes_no(cfg!(feature = "live"))
    ));
    line(format!(
        "  GPU (--device gpu)          {}",
        yes_no(cfg!(feature = "gpu"))
    ));
    line(format!(
        "  native backend (build, jit) {}",
        yes_no(cfg!(feature = "llvm"))
    ));
    if !cfg!(feature = "llvm") {
        line("    rebuild with `--features llvm` to use it (see scripts/llvm-env.sh)".into());
    }
    line(String::new());
    line("environment:".into());
    line(format!(
        "  TELVAX_LIVE           {}",
        value("TELVAX_LIVE", "not set")
    ));
    line(format!(
        "  OLLAMA_HOST           {}",
        value("OLLAMA_HOST", "http://localhost:11434 (default)")
    ));
    line(format!(
        "  OPENAI_API_KEY        {}",
        is_set("OPENAI_API_KEY")
    ));
    line(format!(
        "  TELVAX_WORKERS        {}",
        value("TELVAX_WORKERS", "not set")
    ));
    line(format!(
        "  TELVAX_CLUSTER_TOKEN  {}",
        is_set("TELVAX_CLUSTER_TOKEN")
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_typo_finds_the_command() {
        assert_eq!(closest_command("chek"), Some("check"));
        assert_eq!(closest_command("biuld"), Some("build"));
        assert_eq!(closest_command("docter"), Some("doctor"));
        assert_eq!(closest_command("frobnicate"), None);
        assert_eq!(closest_command("x"), None);
    }
}
