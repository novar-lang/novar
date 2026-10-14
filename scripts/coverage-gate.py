#!/usr/bin/env python3
"""Reads `cargo llvm-cov --json --summary-only` output, reports it and enforces a floor.

    python3 scripts/coverage-gate.py coverage.json

The floor is the number in scripts/coverage-floor.txt (line coverage, percent). Raise it when
coverage goes up; it never needs to go down.
"""
import json
import os
import pathlib
import sys

report = json.load(open(sys.argv[1]))
totals = report["data"][0]["totals"]
lines = totals["lines"]["percent"]
functions = totals["functions"]["percent"]
regions = totals["regions"]["percent"]
floor = float(pathlib.Path(__file__).with_name("coverage-floor.txt").read_text().strip() or 0)

summary = (
    "### Rust coverage\n\n"
    "| | covered | total | percent |\n|---|---|---|---|\n"
    f"| lines | {totals['lines']['covered']} | {totals['lines']['count']} | {lines:.2f}% |\n"
    f"| functions | {totals['functions']['covered']} | {totals['functions']['count']} | {functions:.2f}% |\n"
    f"| regions | {totals['regions']['covered']} | {totals['regions']['count']} | {regions:.2f}% |\n\n"
    f"Floor: {floor:.2f}% of lines.\n"
)
print(summary)
if "GITHUB_STEP_SUMMARY" in os.environ:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(summary)

if lines + 1e-9 < floor:
    sys.exit(f"line coverage {lines:.2f}% is below the floor of {floor:.2f}%")
