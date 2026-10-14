#!/bin/sh
# Works out the version a merge to main releases, from the pull request title (the first line of the
# squash commit) and the latest tag. Prints the version, or "none" when the change is not released.
#   scripts/next-version.sh <latest tag version or ""> <version declared in Cargo.toml> <commit message>
#
#   feat!: / BREAKING CHANGE:   major  (1.2.3 -> 2.0.0; while the major is 0, minor: 0.1.2 -> 0.2.0)
#   feat:                       minor  (0.1.2 -> 0.2.0)
#   fix, perf, refactor, build,
#   revert, or no prefix        patch  (0.1.2 -> 0.1.3)
#   docs, ci, test, style,
#   chore                       no release
#   [skip release]              no release
# A Cargo.toml version above the latest tag wins, so a version can still be chosen by hand.
set -eu
latest="$1" declared="$2" message="$3"

case "$message" in *"[skip release]"*) echo none; exit 0 ;; esac

if [ -z "$latest" ]; then echo "$declared"; exit 0; fi
if [ "$declared" != "$latest" ] && [ "$(printf '%s\n%s\n' "$latest" "$declared" | sort -V | tail -n 1)" = "$declared" ]; then
  echo "$declared"; exit 0
fi

subject="$(printf '%s\n' "$message" | head -n 1)"
type="$(printf '%s\n' "$subject" | sed -n 's/^\([A-Za-z]*\)\(([^)]*)\)\{0,1\}\(!\{0,1\}\):.*/\1/p' | tr 'A-Z' 'a-z')"
bang="$(printf '%s\n' "$subject" | sed -n 's/^[A-Za-z]*\(([^)]*)\)\{0,1\}\(!\)\{1\}:.*/!/p')"
breaking=false
[ -n "$bang" ] && breaking=true
printf '%s\n' "$message" | grep -q '^BREAKING[ -]CHANGE:' && breaking=true

case "$type" in
  docs | ci | test | style | chore) [ "$breaking" = true ] || { echo none; exit 0; } ;;
esac

major="${latest%%.*}"; rest="${latest#*.}"; minor="${rest%%.*}"; patch="${rest#*.}"
if [ "$breaking" = true ]; then
  if [ "$major" = 0 ]; then echo "0.$((minor + 1)).0"; else echo "$((major + 1)).0.0"; fi
elif [ "$type" = feat ]; then
  echo "$major.$((minor + 1)).0"
else
  echo "$major.$minor.$((patch + 1))"
fi
