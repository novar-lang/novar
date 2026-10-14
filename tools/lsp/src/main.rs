fn main() {
    if let Err(e) = telvax_lsp::run_stdio() {
        eprintln!("telvax-lsp: {e}");
        std::process::exit(1);
    }
}
