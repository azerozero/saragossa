#!/usr/bin/env bash
# Prépare les GGUF llama.cpp du bench concurrents, sans GPU (CPU et disque).
#
# Entrées : BF16 officiel ggml-org (tronc et tête MTP) et drafter DFlash2 BF16
# z-lab. Sorties dans $OUT : Q4_0 pur et Q4_K_M (recette standard, sans
# imatrix) du tronc, Q4_0 pur de la tête MTP, journaux et sha256.
#
# Usage : prepare_weights.sh [--check-only]
#   --check-only : sha256 des entrées seulement, sans quantifier.
set -euo pipefail
source "$(dirname "$0")/paths.sh"

DFLASH_GGUF="${GGUF_DFLASH}"
OUT="${GGUF_DIR}"
QUANTIZE="${QUANTIZE:-llama-quantize}"

trunk_bf16="${GGUF_BF16}"
mtp_bf16="${GGUF_MTP_BF16}"

mkdir -p "${OUT}"
cd "${OUT}"

# Les sha256 des entrées sont calculés en parallèle : 60 Go de lecture.
sha_inputs() {
  local file
  for file in "${trunk_bf16}" "${mtp_bf16}" "${DFLASH_GGUF}"; do
    shasum -a 256 "${file}" &
  done
  wait
}

sha_inputs | tee inputs.sha256
if [[ "${1:-}" == "--check-only" ]]; then
  exit 0
fi

"${QUANTIZE}" --version 2>&1 | tee quantize-version.txt || true

# Tronc, budget nominal proche du MLX u4 gs64 : tous les tenseurs quantifiables
# en Q4_0 (bloc 32, échelle f16), embeddings et tête de sortie compris.
"${QUANTIZE}" --pure "${trunk_bf16}" Qwen3.8-27B-Q4_0-pure.gguf Q4_0 \
  2>&1 | tee quantize-Q4_0-pure.log

# Tronc, recette communautaire standard (mélange k-quants), sans imatrix.
"${QUANTIZE}" "${trunk_bf16}" Qwen3.8-27B-Q4_K_M.gguf Q4_K_M \
  2>&1 | tee quantize-Q4_K_M.log

# Tête MTP : même précision nominale que mtp.safetensors (u4 affine gs64).
"${QUANTIZE}" --pure "${mtp_bf16}" mtp-Qwen3.8-27B-Q4_0-pure.gguf Q4_0 \
  2>&1 | tee quantize-mtp-Q4_0-pure.log

shasum -a 256 Qwen3.8-27B-Q4_0-pure.gguf Qwen3.8-27B-Q4_K_M.gguf \
  mtp-Qwen3.8-27B-Q4_0-pure.gguf | tee outputs.sha256
