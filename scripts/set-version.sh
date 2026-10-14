#!/bin/sh
# Sets the workspace version for a build: scripts/set-version.sh 0.0.5
# Used by the release workflow so a release does not need a commit that edits Cargo.toml.
set -eu
version="$1"
perl -0pi -e 's/(\[workspace\.package\][^\[]*?\nversion = ")[^"]*(")/${1}'"$version"'${2}/' Cargo.toml
grep -A3 '^\[workspace.package\]' Cargo.toml | grep -q "^version = \"$version\"" || { echo "could not set the version" >&2; exit 1; }
cargo update --workspace --quiet
