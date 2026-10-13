//! `novar serve` and the server half of bundled binaries.

use std::collections::BTreeMap;
use std::sync::Arc;

use novar_hir::hir;
use novar_runtime::Interp;
use novar_server::{HandlerError, handler};

use crate::{CliError, failed};

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub host: String,
    pub port: u16,
    pub seed: Option<String>,
    pub live: bool,
}

impl Options {
    /// Parses `--port N`, `--host H`, `--seed FILE` and `--live`.
    pub fn parse(default_host: &str, flags: &[&str]) -> Result<Options, CliError> {
        let mut opts = Options {
            host: default_host.to_string(),
            port: 8080,
            seed: None,
            live: false,
        };
        let mut it = flags.iter();
        while let Some(flag) = it.next() {
            if *flag == "--live" {
                opts.live = true;
                continue;
            }
            match (*flag, it.next()) {
                ("--port", Some(p)) => {
                    opts.port = p
                        .parse()
                        .or_else(|_| failed(format!("invalid port `{p}`")))?
                }
                ("--host", Some(h)) => opts.host = h.to_string(),
                ("--seed", Some(f)) => opts.seed = Some(f.to_string()),
                _ => return Err(CliError::Usage),
            }
        }
        Ok(opts)
    }
}

/// Loads `{"collection": .., "id": .., "text": ..}` lines into every database the program opened.
fn seed(interp: &Interp, path: &str) -> Result<usize, CliError> {
    let data =
        std::fs::read_to_string(path).or_else(|e| failed(format!("cannot read `{path}`: {e}")))?;
    let dbs = interp.dbs();
    let mut n = 0;
    for (i, line) in data
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let doc: serde_json::Value =
            serde_json::from_str(line).or_else(|e| failed(format!("{path}:{}: {e}", i + 1)))?;
        let field = |k: &str| doc[k].as_str().map(str::to_string);
        let (Some(col), Some(id), Some(text)) = (field("collection"), field("id"), field("text"))
        else {
            return failed(format!(
                "{path}:{}: expected string fields `collection`, `id`, `text`",
                i + 1
            ));
        };
        let vector = interp.embed(&text).or_else(|e| failed(e.to_string()))?;
        for db in &dbs {
            db.insert(&col, &id, &text, vector.clone())
                .or_else(|e| failed(e.to_string()))?;
        }
        n += 1;
    }
    Ok(n)
}

/// Serves the `@endpoint` functions of `program` until Ctrl-C / SIGTERM.
pub fn run(program: hir::Program, opts: &Options) -> Result<(), CliError> {
    let routes = novar_server::routes(&program);
    if routes.is_empty() {
        return failed("no `@endpoint` functions found, nothing to serve");
    }
    let interp =
        Arc::new(Interp::with_live(program, opts.live).or_else(|e| failed(e.to_string()))?);
    if opts.live {
        eprintln!("live mode: using the real Ollama / OpenAI / PostgreSQL services");
    }
    if let Some(path) = &opts.seed {
        let n = seed(&interp, path)?;
        eprintln!("seeded {n} document(s) from {path}");
    }

    let mut handlers = BTreeMap::new();
    for route in &routes {
        let (interp, name) = (interp.clone(), route.name.clone());
        handlers.insert(
            route.name.clone(),
            handler(move |payload| {
                let (interp, name) = (interp.clone(), name.clone());
                async move {
                    // The evaluator is synchronous and may be CPU-heavy; keep it off the async workers.
                    tokio::task::spawn_blocking(move || interp.call_endpoint(&name, &payload))
                        .await
                        .map_err(|e| HandlerError::new(500, format!("handler panicked: {e}")))?
                        .map_err(|e| HandlerError::new(e.status, e.message))
                }
            }),
        );
    }
    let app = novar_server::router(&routes, handlers).or_else(|e| failed(e.to_string()))?;

    let rt = tokio::runtime::Runtime::new().or_else(|e| failed(e.to_string()))?;
    rt.block_on(async {
        let addr = (opts.host.as_str(), opts.port);
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .or_else(|e| failed(format!("cannot bind {}:{}: {e}", opts.host, opts.port)))?;
        for r in &routes {
            eprintln!(
                "  {:?} http://{}:{}{}",
                r.method, opts.host, opts.port, r.path
            );
        }
        eprintln!(
            "listening on http://{}:{} (Ctrl-C to stop)",
            opts.host, opts.port
        );
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .or_else(|e| failed(e.to_string()))
    })
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}
