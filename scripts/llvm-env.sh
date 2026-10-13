#!/usr/bin/env sh
# Source this file to build the native backend:
#   . scripts/llvm-env.sh && cargo test --workspace --features llvm
#
# Needs LLVM 22 with MLIR. On macOS: `brew install llvm@22`.
# On Debian/Ubuntu (apt.llvm.org): llvm-22-dev libmlir-22-dev mlir-22-tools libzstd-dev.
if [ -z "${LLVM_PREFIX:-}" ]; then
  if command -v brew >/dev/null 2>&1 && [ -d "$(brew --prefix llvm@22 2>/dev/null)" ]; then
    LLVM_PREFIX="$(brew --prefix llvm@22)"
    export LIBRARY_PATH="$(brew --prefix)/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  elif [ -d /usr/lib/llvm-22 ]; then
    LLVM_PREFIX=/usr/lib/llvm-22
  else
    echo "llvm-env.sh: LLVM 22 not found; set LLVM_PREFIX to its install prefix" >&2
    return 1 2>/dev/null || exit 1
  fi
fi
export LLVM_SYS_221_PREFIX="$LLVM_PREFIX"
export MLIR_SYS_220_PREFIX="$LLVM_PREFIX"
export TABLEGEN_220_PREFIX="$LLVM_PREFIX"
export PATH="$LLVM_PREFIX/bin:$PATH"
