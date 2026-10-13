//! The language server loop (JSON-RPC over stdio).

use std::collections::HashMap;
use std::error::Error;

use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification as _,
    PublishDiagnostics,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, InlayHintRequest, Request as _,
};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    DiagnosticRelatedInformation, DiagnosticSeverity, DocumentSymbol, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents,
    HoverParams, HoverProviderCapability, InitializeParams, InlayHint, InlayHintKind,
    InlayHintLabel, InlayHintParams, Location, MarkupContent, MarkupKind, OneOf,
    PublishDiagnosticsParams, ServerCapabilities, SymbolKind, TextDocumentSyncCapability,
    TextDocumentSyncKind, Uri,
};

use crate::analysis::{CompletionKind, Snapshot, SymbolKind as Sym};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub fn capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".into(), ":".into(), "@".into()]),
            ..CompletionOptions::default()
        }),
        ..ServerCapabilities::default()
    }
}

/// Runs the server on stdin / stdout until the client shuts it down.
pub fn run_stdio() -> Result<()> {
    let (connection, io_threads) = Connection::stdio();
    let caps = serde_json::to_value(capabilities())?;
    let init = connection.initialize(caps)?;
    let _params: InitializeParams = serde_json::from_value(init)?;
    serve(&connection)?;
    // `handle_shutdown` returned only after the client's `exit`, so the shutdown reply has long
    // been written. The reader thread may sit in a blocking read until the editor closes stdin,
    // so do not wait for it: returning ends the process.
    drop(connection);
    drop(io_threads);
    Ok(())
}

struct Documents {
    docs: HashMap<Uri, Snapshot>,
}

fn serve(conn: &Connection) -> Result<()> {
    let mut state = Documents {
        docs: HashMap::new(),
    };
    for msg in &conn.receiver {
        match msg {
            Message::Request(req) => {
                if conn.handle_shutdown(&req)? {
                    return Ok(());
                }
                let resp = handle_request(&state, req);
                conn.sender.send(Message::Response(resp))?;
            }
            Message::Notification(note) => handle_notification(conn, &mut state, note)?,
            Message::Response(_) => {}
        }
    }
    Ok(())
}

fn handle_notification(conn: &Connection, state: &mut Documents, note: Notification) -> Result<()> {
    let (uri, text) = match note.method.as_str() {
        DidOpenTextDocument::METHOD => {
            let p: lsp_types::DidOpenTextDocumentParams = serde_json::from_value(note.params)?;
            (p.text_document.uri, p.text_document.text)
        }
        DidChangeTextDocument::METHOD => {
            let p: lsp_types::DidChangeTextDocumentParams = serde_json::from_value(note.params)?;
            // Full synchronisation: the last change event carries the whole document.
            let Some(change) = p.content_changes.into_iter().last() else {
                return Ok(());
            };
            (p.text_document.uri, change.text)
        }
        DidCloseTextDocument::METHOD => {
            let p: lsp_types::DidCloseTextDocumentParams = serde_json::from_value(note.params)?;
            state.docs.remove(&p.text_document.uri);
            publish(conn, p.text_document.uri, Vec::new())?;
            return Ok(());
        }
        _ => return Ok(()),
    };
    let path = file_path(&uri);
    let snap = Snapshot::analyze_after(text, path.as_deref(), state.docs.get(&uri));
    let diags = diagnostics(&snap, &uri);
    state.docs.insert(uri.clone(), snap);
    publish(conn, uri, diags)
}

fn publish(conn: &Connection, uri: Uri, diagnostics: Vec<lsp_types::Diagnostic>) -> Result<()> {
    let params = PublishDiagnosticsParams {
        uri,
        diagnostics,
        version: None,
    };
    conn.sender.send(Message::Notification(Notification::new(
        PublishDiagnostics::METHOD.to_string(),
        params,
    )))?;
    Ok(())
}

/// The local path of a `file://` URI.
fn file_path(uri: &Uri) -> Option<std::path::PathBuf> {
    let rest = uri.as_str().strip_prefix("file://")?;
    let bytes = rest.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&rest[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    Some(std::path::PathBuf::from(
        String::from_utf8_lossy(&out).into_owned(),
    ))
}

pub fn diagnostics(snap: &Snapshot, uri: &Uri) -> Vec<lsp_types::Diagnostic> {
    snap.diagnostics
        .iter()
        .map(|d| {
            let mut message = d.message.clone();
            for note in &d.notes {
                message.push('\n');
                message.push_str(note);
            }
            lsp_types::Diagnostic {
                range: snap.index.range(&d.span),
                severity: Some(if d.is_error() {
                    DiagnosticSeverity::ERROR
                } else {
                    DiagnosticSeverity::WARNING
                }),
                source: Some("novar".into()),
                message,
                related_information: Some(
                    d.labels
                        .iter()
                        .map(|(span, text)| DiagnosticRelatedInformation {
                            location: Location {
                                uri: uri.clone(),
                                range: snap.index.range(span),
                            },
                            message: text.clone(),
                        })
                        .collect(),
                )
                .filter(|v: &Vec<_>| !v.is_empty()),
                ..lsp_types::Diagnostic::default()
            }
        })
        .collect()
}

fn reply<T: serde::Serialize>(id: RequestId, result: T) -> Response {
    Response {
        id,
        result: Some(serde_json::to_value(result).unwrap_or(serde_json::Value::Null)),
        error: None,
    }
}

fn handle_request(state: &Documents, req: Request) -> Response {
    let id = req.id.clone();
    let unknown = |id: RequestId| Response {
        id,
        result: Some(serde_json::Value::Null),
        error: None,
    };
    macro_rules! params {
        ($ty:ty) => {
            match serde_json::from_value::<$ty>(req.params.clone()) {
                Ok(p) => p,
                Err(_) => return unknown(id),
            }
        };
    }
    match req.method.as_str() {
        HoverRequest::METHOD => {
            let p = params!(HoverParams);
            let pos = p.text_document_position_params;
            let Some(snap) = state.docs.get(&pos.text_document.uri) else {
                return unknown(id);
            };
            let hover = snap
                .hover(snap.index.offset(pos.position))
                .map(|(text, span)| Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: text,
                    }),
                    range: Some(snap.index.range(&span)),
                });
            reply(id, hover)
        }
        GotoDefinition::METHOD => {
            let p = params!(GotoDefinitionParams);
            let pos = p.text_document_position_params;
            let Some(snap) = state.docs.get(&pos.text_document.uri) else {
                return unknown(id);
            };
            let target = snap
                .definition_location(snap.index.offset(pos.position))
                .and_then(|(file, span)| {
                    let own = file_path(&pos.text_document.uri).map(|p| p.display().to_string());
                    if own.as_deref() == Some(file.as_str()) || file == "<buffer>" {
                        let uri = pos.text_document.uri.clone();
                        return Some(GotoDefinitionResponse::Scalar(Location {
                            uri,
                            range: snap.index.range(&span),
                        }));
                    }
                    let text = std::fs::read_to_string(&file).ok()?;
                    let range = crate::position::LineIndex::new(&text).range(&span);
                    let uri = format!("file://{file}").parse::<Uri>().ok()?;
                    Some(GotoDefinitionResponse::Scalar(Location { uri, range }))
                });
            reply(id, target)
        }
        Completion::METHOD => {
            let p = params!(CompletionParams);
            let pos = p.text_document_position;
            let Some(snap) = state.docs.get(&pos.text_document.uri) else {
                return unknown(id);
            };
            let items: Vec<CompletionItem> = snap
                .completions(snap.index.offset(pos.position))
                .into_iter()
                .map(|c| CompletionItem {
                    label: c.label,
                    kind: Some(match c.kind {
                        CompletionKind::Keyword => CompletionItemKind::KEYWORD,
                        CompletionKind::Function => CompletionItemKind::FUNCTION,
                        CompletionKind::Struct => CompletionItemKind::STRUCT,
                        CompletionKind::Variable => CompletionItemKind::VARIABLE,
                        CompletionKind::Field => CompletionItemKind::FIELD,
                        CompletionKind::Method => CompletionItemKind::METHOD,
                        CompletionKind::Module => CompletionItemKind::MODULE,
                        CompletionKind::Type => CompletionItemKind::CLASS,
                    }),
                    detail: c.detail,
                    ..CompletionItem::default()
                })
                .collect();
            reply(id, CompletionResponse::Array(items))
        }
        DocumentSymbolRequest::METHOD => {
            let p = params!(DocumentSymbolParams);
            let Some(snap) = state.docs.get(&p.text_document.uri) else {
                return unknown(id);
            };
            #[allow(deprecated)]
            let symbols: Vec<DocumentSymbol> = snap
                .symbols()
                .into_iter()
                .map(|s| DocumentSymbol {
                    name: s.name,
                    detail: Some(s.detail),
                    kind: match s.kind {
                        Sym::Function => SymbolKind::FUNCTION,
                        Sym::Struct => SymbolKind::STRUCT,
                        Sym::Global => SymbolKind::VARIABLE,
                    },
                    tags: None,
                    deprecated: None,
                    range: snap.index.range(&s.span),
                    selection_range: snap.index.range(&s.name_span),
                    children: None,
                })
                .collect();
            reply(id, DocumentSymbolResponse::Nested(symbols))
        }
        InlayHintRequest::METHOD => {
            let p = params!(InlayHintParams);
            let Some(snap) = state.docs.get(&p.text_document.uri) else {
                return unknown(id);
            };
            let hints: Vec<InlayHint> = snap
                .inlay_hints()
                .into_iter()
                .map(|(offset, label)| InlayHint {
                    position: snap.index.position(offset),
                    label: InlayHintLabel::String(label),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: None,
                    padding_left: Some(false),
                    padding_right: Some(false),
                    data: None,
                })
                .collect();
            reply(id, hints)
        }
        _ => Response {
            id,
            result: None,
            error: Some(lsp_server::ResponseError {
                code: lsp_server::ErrorCode::MethodNotFound as i32,
                message: format!("unsupported request `{}`", req.method),
                data: None,
            }),
        },
    }
}
