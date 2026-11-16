#!/usr/bin/env bash
# Coût du semis d'historique MTP dans le prefill DFlash, mesuré dans le moteur
# (ligne dflash_prefill ms=) : semis forcé par la bascule armée mais jamais
# déclenchée (SWITCH_MIN=1000, WARMUP énorme) contre sans semis. ABBA.

B=${BIN:-target/release/saragossa}
M=${MODEL:?MODEL requis}
D=${DRAFTER:?DRAFTER requis}
one() { # $1=label $2=seed(0/1) puis args prompt
  local label=$1 seed=$2; shift 2
  local min=0; [[ $seed == 1 ]] && min=1000
  python3 -I -c "import os,sys; sys.exit(0 if os.getloadavg()[0] < 2 else 1)" || { while ! python3 -I -c "import os,sys; sys.exit(0 if os.getloadavg()[0] < 2 else 1)"; do sleep 10; done; }
  local l; l=$(python3 -I -c "import os; print(round(os.getloadavg()[0],2))")
  local ms; ms=$(env -i HOME=$HOME PATH=$PATH SARAGOSSA_RUST_DFLASH_DRAFT_DIR=$D SARAGOSSA_RUST_DFLASH_QUANT_BITS=4 SARAGOSSA_RUST_PREFIX_CACHE=0 SARAGOSSA_RUST_MTP_LOOKUP_DRAFT=0 SARAGOSSA_RUST_DFLASH_PROFILE=1 SARAGOSSA_RUST_SPEC_ROUTE=0 SARAGOSSA_RUST_DFLASH_SWITCH_MIN=$min SARAGOSSA_RUST_DFLASH_SWITCH_WARMUP=100000000 taskpolicy -b $B --model-dir $M --max-tokens 8 "$@" 2>&1 | grep "dflash_prefill source=full" | tail -1 | sed 's/.*ms=//')
  echo "$label seed=$seed ms=$ms load=$l"
}
for _ in 1 2 3; do
  for s in 1 0 0 1; do one C1 $s --prompt "Write a complete Python implementation of Dijkstra's shortest path algorithm, then explain the algorithm step by step."; done
done
for _ in 1 2 3; do
  for s in 1 0 0 1; do one C3 $s --prompt "Implement a TypeScript async retry function with exponential backoff, jitter, a maximum attempt count and AbortSignal support. Include usage examples."; done
done
for _ in 1 2; do
  for s in 1 0 0 1; do one T32k $s --prompt-tokens 32768; done
done
