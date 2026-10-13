# Novar playground

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

`src/lib.rs` exports `check`, `functions`, `run`, `hover` and `complete` (JSON in, JSON out).
`index.html`, `app.js` and `examples.js` are the page. The engine tests run the compiled module
under Node; the page itself has not been exercised in an automated browser.
