#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CIRCOM_BIN="${1:-$ROOT/../circom/target/release/circom}"
WORKDIR="$ROOT/target/sha256-optimized-cache/sha256_2048b"
CPP_DIR="$WORKDIR/sha256_2048b_cpp"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

"$CIRCOM_BIN" \
  "$ROOT/tests/circuits/optimized/sha256_2048b.circom" \
  --prime koalabear --r1cs --c --O2 -o "$WORKDIR"

includes=(-I "$CPP_DIR")
gmp_linker=(-lgmp)
for dir in /opt/homebrew/include /usr/local/include; do
  if [ -f "$dir/gmp.h" ]; then
    includes+=(-I "$dir")
  fi
done
for dir in /opt/homebrew/lib /usr/local/lib; do
  if [ -f "$dir/libgmp.dylib" ] || [ -f "$dir/libgmp.so" ] || [ -f "$dir/libgmp.a" ]; then
    gmp_linker=(-L "$dir" -lgmp)
    break
  fi
done

case "$(uname -s)" in
  Darwin)
    library="$WORKDIR/libsha256_2048b_witness.dylib"
    linker=(-dynamiclib -Wl,-install_name,@rpath/libsha256_2048b_witness.dylib)
    ;;
  *)
    library="$WORKDIR/libsha256_2048b_witness.so"
    linker=(-shared)
    ;;
esac

"${CXX:-c++}" \
  -std=c++11 -O3 -fPIC -fvisibility=hidden -UNDEBUG \
  -DCIRCOM_LINKED_WITNESS_ONLY \
  "${includes[@]}" \
  "$CPP_DIR/calcwit.cpp" \
  "$CPP_DIR/fr.cpp" \
  "$CPP_DIR/main.cpp" \
  "$CPP_DIR/sha256_2048b.cpp" \
  "${linker[@]}" \
  "${gmp_linker[@]}" \
  -o "$library"

printf 'wrote %s\n' "$library"
