# Telvax playground

The type checker and interpreter, compiled to WebAssembly, behind a small web page: edit, see
diagnostics as you type, hover for inferred types, and run functions with JSON arguments. Nothing
leaves the browser. The model, vector database and LLM are the offline stand-ins, and runs stop
after a fixed number of steps.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
scripts/build-playground.sh                 # writes playground/pkg
python3 -m http.server -d playground 8000   # open http://localhost:8000

scripts/build-playground.sh --node && node --test playground/test    # engine tests on the real .wasm
```

`src/lib.rs` exports `check`, `functions`, `run`, `hover` and `complete` (JSON in, JSON out), and for the
console `repl_eval`, `repl_load`, `repl_type`, `repl_vars`, `repl_defs`, `repl_reset`, `repl_is_complete` and
`repl_complete`.

The console under the editor is the same session as `telvax repl`: definitions and variables last from one
input to the next, a block ends with an empty line, Tab completes, the arrow keys go through earlier lines, and
`:type`, `:vars`, `:defs`, `:load`, `:reset`, `:clear` and `:help` are commands. **Load the program** enters the
functions, types and variables of the editor into the console. It has no files, environment or network, and an
input stops after the same number of steps as a run.
`index.html`, `app.js` and `examples.js` are the page. The engine tests run the compiled module
under Node; the page is exercised in Chromium by `playground/e2e`.
