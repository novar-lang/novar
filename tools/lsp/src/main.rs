fn main() {
    if let Err(e) = novar_lsp::run_stdio() {
        eprintln!("novar-lsp: {e}");
        std::process::exit(1);
    }
}
