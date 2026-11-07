#!/usr/bin/env bash
# Lance llmd (ZML, brew 20261029) pour le bench concurrents.
#
# Usage : launch_llmd.sh MODE PORT CTX [ARGS...]
#   MODE : ar | native (tête MTP bf16 du checkpoint NVFP4)
#   CTX  : --max-context-len (128000 par défaut chez llmd : toujours explicite)
#   ARGS : réglages de la grille (--token-batch-size, --prefill-chunk-size,
#          --gpu-memory-fraction, --cache-memory-fraction, --topk,
#          --speculative-draft-count), ajoutés après la base.
#
# llmd ne lit pas le MLX : checkpoint NVFP4 nvidia (MLP NVFP4 g16, attention
# FP8, embeddings bf16). Son chat_template.jinja local force
# enable_thinking=false en tête (copie d'origine : chat_template.jinja.orig).
set -euo pipefail
source "$(dirname "$0")/paths.sh"

mode="$1"
port="$2"
ctx="$3"
shift 3

args=(
  --model="${NVFP4_CKPT}" --listen="127.0.0.1:${port}"
  --batch-size=1 --max-context-len="${ctx}"
)

case "${mode}" in
  ar) ;;
  native) args+=(--speculative-drafter-model=native) ;;
  *) echo "mode inconnu: ${mode}" >&2; exit 2 ;;
esac

run_engine llmd "${args[@]}" "$@"
