#!/usr/bin/env bash
# Qualité des GGUF (llama-perplexity, KL) et ligne llama-bench, sous le verrou GPU.
#
# Usage : quality_llama.sh CORPUS OUT_DIR
#   CORPUS : wikitext-2-raw/wiki.test.raw (sha256 publié dans le protocole)
# Le BF16 sert de référence : ses logits sont sauvés une fois, puis Q4_0 pur
# et Q4_K_M sont comparés à lui (PPL, KL, top-1 identique).
set -euo pipefail
source "$(dirname "$0")/paths.sh"

corpus="$1"
out="$2"
mkdir -p "${out}"
common=(-f "${corpus}" -c 512 --chunks 32 -ngl all -fa on -b 512 -ub 512)

llama-perplexity -m "${AI_MODEL}/Qwen3.8-27B-GGUF-ggml/Qwen3.8-27B-BF16.gguf" \
  "${common[@]}" --kl-divergence-base "${out}/bf16.kld" 2>&1 | tee "${out}/ppl-bf16.log"
for name in Q4_0-pure Q4_K_M; do
  llama-perplexity -m "${GGUF_DIR}/Qwen3.8-27B-${name}.gguf" "${common[@]}" \
    --kl-divergence-base "${out}/bf16.kld" --kl-divergence 2>&1 | tee "${out}/ppl-${name}.log"
done

# Référence habituelle de la communauté, à côté du tableau client (hors T1).
for name in Q4_0-pure Q4_K_M; do
  llama-bench -m "${GGUF_DIR}/Qwen3.8-27B-${name}.gguf" -p 4096 -n 400 -fa 1 -ngl 99 -r 3 \
    -o json 2>&1 | tee "${out}/llama-bench-${name}.json"
done
