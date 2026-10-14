#!/bin/sh
# Installs `novar` and `nvpm` from a release archive or, failing that, builds them from source.
# Usage: scripts/install.sh [version]     (default: the latest release)
#        NOVAR_LITE=1 scripts/install.sh   a ~10 MB build without tables / SQL (CSV, Parquet)
set -eu

repo="ntsation/novar"
version="${1:-latest}"
prefix="${NOVAR_HOME:-$HOME/.novar}"
asset=novar
[ -n "${NOVAR_LITE:-}" ] && asset=novar-lite

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
