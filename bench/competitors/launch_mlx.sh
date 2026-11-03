#!/usr/bin/env bash
# Lance mlx_lm.server (mlx-lm 0.32.2) pour le bench concurrents.
#
# Usage : launch_mlx.sh MODE PORT CTX [ARGS...]
#   MODE : ar (seul mode : mlx-lm 0.32.2 jette les poids mtp.* du checkpoint
#          et n'a pas de drafter DFlash2)
#   CTX  : ignoré (le serveur n'a pas de --max-kv-size : KV non borné, sans
#          rotation).
#   ARGS : réglages de la grille (--prefill-step-size…).
set -euo pipefail
source "$(dirname "$0")/paths.sh"

mode="$1"
port="$2"
shift 3

if [[ "${mode}" != "ar" ]]; then
  echo "mlx-lm: mode non supporté: ${mode}" >&2
  exit 3
fi

run_engine "${MLX_VENV}/bin/python" -I -m mlx_lm.server \
  --model "${MLX_CKPT}" --host 127.0.0.1 --port "${port}" \
  --decode-concurrency 1 --prompt-concurrency 1 --prompt-cache-size 0 \
  --temp 0 --top-p 1 --top-k 0 --min-p 0 \
  --chat-template-args '{"enable_thinking":false}' \
  "$@"
