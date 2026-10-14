#!/bin/sh
# Installs `telvax` and `tvpm` from a release archive or, failing that, builds them from source.
# Usage: scripts/install.sh [version]     (default: the latest release)
#        TELVAX_LITE=1 scripts/install.sh   a ~10 MB build without SQL and Parquet
set -eu

repo="telvax/telvax"
version="${1:-latest}"
prefix="${TELVAX_HOME:-$HOME/.telvax}"
asset=telvax
[ -n "${TELVAX_LITE:-}" ] && asset=telvax-lite

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target="aarch64-apple-darwin" ;;
  Darwin-x86_64) target="x86_64-apple-darwin" ;;
  Linux-aarch64 | Linux-arm64) target="aarch64-unknown-linux-gnu" ;;
  Linux-x86_64) target="x86_64-unknown-linux-gnu" ;;
  *) target="" ;;
esac

if [ -n "$target" ] && command -v curl >/dev/null 2>&1; then
  if [ "$version" = latest ]; then
    # `releases/latest` skips pre-releases, which is all there is during the alpha.
    tag="$(curl -fsSL "https://api.github.com/repos/$repo/releases?per_page=1" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)"
  else
    tag="v$version"
  fi
  url="https://github.com/$repo/releases/download/$tag/$asset-$target.tar.gz"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  if curl -fsSL "$url" -o "$tmp/telvax.tar.gz"; then
    mkdir -p "$prefix/bin"
    tar -xzf "$tmp/telvax.tar.gz" -C "$prefix/bin"
    echo "installed to $prefix/bin; add it to PATH:  export PATH=\"$prefix/bin:\$PATH\""
    exit 0
  fi
  echo "no prebuilt binary for $target, building from source" >&2
fi

command -v cargo >/dev/null 2>&1 || { echo "install Rust first: https://rustup.rs" >&2; exit 1; }
ref=""
[ "$version" = latest ] || ref="--tag v$version"
cargo install --locked --git "https://github.com/$repo" $ref telvax_cli telvax_pkg
