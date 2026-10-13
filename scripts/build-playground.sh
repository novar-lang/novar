#!/usr/bin/env sh
# Builds the WebAssembly engine and its JS glue.
#   scripts/build-playground.sh            -> playground/pkg       (ES module for the web page)
#   scripts/build-playground.sh --node     -> playground/pkg-node  (CommonJS, used by the tests)
# Needs: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.129
set -eu
cd "$(dirname "$0")/.."
cargo build -p novar_wasm --target wasm32-unknown-unknown --release
if [ "${1:-}" = "--node" ]; then
  wasm-bindgen --target nodejs --out-dir playground/pkg-node --out-name novar_wasm \
    target/wasm32-unknown-unknown/release/novar_wasm.wasm
else
  wasm-bindgen --target web --out-dir playground/pkg --out-name novar_wasm \
    target/wasm32-unknown-unknown/release/novar_wasm.wasm
fi
# wasm-opt (binaryen) shrinks the module further when installed.
if command -v wasm-opt >/dev/null 2>&1; then
  for f in playground/pkg/novar_wasm_bg.wasm playground/pkg-node/novar_wasm_bg.wasm; do
    [ -f "$f" ] && wasm-opt -Oz "$f" -o "$f"
  done
fi
