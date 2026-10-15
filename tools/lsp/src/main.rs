/// See `STACK_BYTES` in the command line tool: deep programs need more than a main thread has.
const STACK_BYTES: usize = 512 * 1024 * 1024;

fn serve() {
    if let Err(e) = telvax_lsp::run_stdio() {
        eprintln!("telvax-lsp: {e}");
        std::process::exit(1);
    }
}

fn main() {
    match std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(serve)
    {
        Ok(worker) => {
            if worker.join().is_err() {
                std::process::exit(1);
            }
        }
        Err(_) => serve(),
    }
}
