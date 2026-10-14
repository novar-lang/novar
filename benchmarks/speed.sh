#!/usr/bin/env bash
# Fails when the interpreter gets much slower: runs two small programs and compares the wall
# time with generous limits (about ten times what they take on a laptop).
#   benchmarks/speed.sh target/release/telvax
set -euo pipefail
telvax="${1:?usage: speed.sh path/to/telvax}"
here="$(cd "$(dirname "$0")" && pwd)"
status=0

# name function argument expected-result limit-in-milliseconds
check() {
  local name="$1" fn="$2" arg="$3" want="$4" limit="$5"
  local start end ms out
  start=$(date +%s%N)
  out="$("$telvax" run "$here/speed.tvx" "$fn" "$arg" 2>/dev/null)"
  end=$(date +%s%N)
  ms=$(( (end - start) / 1000000 ))
  if [ "$out" != "$want" ]; then
    echo "FAIL $name: expected $want, got '$out'"; status=1; return
  fi
  if [ "$ms" -gt "$limit" ]; then
    echo "FAIL $name: ${ms} ms is over the ${limit} ms limit"; status=1
  else
    echo "ok   $name: ${ms} ms (limit ${limit} ms)"
  fi
}

check "fib(27)" fib 27 196418 2000
check "loop of 3,000,000 iterations" loop_sum 3000000 8999997 6000
exit $status
