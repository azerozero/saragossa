#!/usr/bin/env bash
# Qualité des GGUF (llama-perplexity, KL) et ligne llama-bench, sous le verrou GPU.
#
# Usage : quality_llama.sh OUT_DIR [PORT]
#   Corpus : $WIKITEXT_TEST (paths.sh), sha256 vérifié par quality.py.
# 1. Contrôle bloquant : ids du corpus selon llama-server (/tokenize) égaux à
#    ceux du tokenizer HF, sinon arrêt avant toute PPL.
# 2. Le BF16 sert de référence : ses logits sont sauvés une fois, puis Q4_0
#    pur et Q4_K_M sont comparés à lui (PPL, KL, top-1 identique).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "${here}/paths.sh"

out="$1"
port="${2:-18081}"
mkdir -p "${out}"
venv=("${MLX_VENV}/bin/python" -I -B)

llama-server -m "${GGUF_Q4_0}" --host 127.0.0.1 --port "${port}" -np 1 -c 4096 \
  > "${out}/llama-ids-server.log" 2>&1 &
server=$!
trap 'kill "${server}" 2>/dev/null || true; wait "${server}" 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:${port}/health" > /dev/null; do
  kill -0 "${server}" 2>/dev/null || { echo "llama-server arrêté" >&2; exit 1; }
  sleep 1
done
"${venv[@]}" "${here}/quality.py" llama-ids "${WIKITEXT_TEST}" "${port}" | tee "${out}/llama-ids.json"
kill "${server}"
wait "${server}" 2>/dev/null || true
trap - EXIT

common=(-f "${WIKITEXT_TEST}" -c 512 --chunks 32 -ngl all -fa on -b 512 -ub 512)
llama-perplexity -m "${GGUF_BF16}" \
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
