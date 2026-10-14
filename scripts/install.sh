#!/bin/sh
# Installs `novar` and `nvpm` from a release archive or, failing that, builds them from source.
# Usage: scripts/install.sh [version]     (default: the latest release)
set -eu

repo="ntsation/novar"
version="${1:-latest}"
prefix="${NOVAR_HOME:-$HOME/.novar}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target="aarch64-apple-darwin" ;;
  Darwin-x86_64) target="x86_64-apple-darwin" ;;
  Linux-x86_64) target="x86_64-unknown-linux-gnu" ;;
  *) target="" ;;
esac

if [ -n "$target" ] && command -v curl >/dev/null 2>&1; then
  if [ "$version" = latest ]; then
    url="https://github.com/$repo/releases/latest/download/novar-$target.tar.gz"
  else
    url="https://github.com/$repo/releases/download/v$version/novar-$target.tar.gz"
  fi
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  if curl -fsSL "$url" -o "$tmp/novar.tar.gz"; then
    mkdir -p "$prefix/bin"
    tar -xzf "$tmp/novar.tar.gz" -C "$prefix/bin"
    echo "installed to $prefix/bin; add it to PATH:  export PATH=\"$prefix/bin:\$PATH\""
    exit 0
  fi
  echo "no prebuilt binary for $target, building from source" >&2
fi

command -v cargo >/dev/null 2>&1 || { echo "install Rust first: https://rustup.rs" >&2; exit 1; }
ref=""
[ "$version" = latest ] || ref="--tag v$version"
cargo install --locked --git "https://github.com/$repo" $ref novar_cli novar_pkg
