#!/bin/sh
# Prints which parts of the repository a push or pull request touches, as GITHUB_OUTPUT lines:
#   core=true|false     anything that is not documentation, the extension or the Python package
#   python=true|false   core, or the Python package
#   vscode=true|false   core, or the VS Code extension
# CI uses it to skip pipelines a change cannot affect: a documentation-only change runs none of them.
# When the change set cannot be worked out, everything runs.
# Usage (in CI): scripts/changed-areas.sh >> "$GITHUB_OUTPUT"
set -eu

if [ "${GITHUB_EVENT_NAME:-}" = pull_request ]; then
  files="$(git diff --name-only "origin/${GITHUB_BASE_REF}...HEAD" 2>/dev/null || echo __unknown__)"
else
  before="${BEFORE_SHA:-}"
  case "$before" in
    "" | 0000000000000000000000000000000000000000) files=__unknown__ ;;
    *) files="$(git diff --name-only "$before" HEAD 2>/dev/null || echo __unknown__)" ;;
  esac
fi

core=false python=false vscode=false
if [ "$files" = __unknown__ ] || [ -z "$files" ]; then
  core=true
else
  for f in $files; do
    case "$f" in
      docs/* | rfcs/* | spec/* | *.md | LICENSE* | .github/FUNDING.yml | .github/CODEOWNERS | .gitattributes) ;;
      python/*) python=true ;;
      editors/vscode/*) vscode=true ;;
      *) core=true ;;
    esac
  done
fi
[ "$core" = true ] && python=true && vscode=true
echo "core=$core"
echo "python=$python"
echo "vscode=$vscode"
