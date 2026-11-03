#!/usr/bin/env bash
# Lance llama-server (llama.cpp) pour le bench concurrents.
#
# Usage : launch_llama.sh FORMAT MODE PORT CTX [ARGS...]
#   FORMAT : q4_0 (Q4_0 pur, budget nominal proche) | q4_k_m (recette standard)
#   MODE   : ar | mtp | dflash
#   ARGS   : réglages de la grille (-ub, -b, -bs, --spec-draft-n-max, -ctxcp…),
#            ajoutés après la base et donc prioritaires.
set -euo pipefail
source "$(dirname "$0")/paths.sh"

format="$1"
mode="$2"
port="$3"
ctx="$4"
shift 4

case "${format}" in
  q4_0) model="${GGUF_Q4_0}" ;;
  q4_k_m) model="${GGUF_Q4_K_M}" ;;
  *) echo "format inconnu: ${format}" >&2; exit 2 ;;
esac

# Base commune : un slot, tout sur le GPU, pas de cache de prompt (ni slot ni
# RAM), pas de rotation du contexte, template du checkpoint MLX sans thinking,
# sampling neutre par défaut (chaque requête fixe aussi ses paramètres).
args=(
  -m "${model}" --host 127.0.0.1 --port "${port}"
  -np 1 -ngl all -fa on -c "${ctx}"
  -ctk f16 -ctv f16
  --no-cache-prompt -cram 0 --no-context-shift
  --jinja --chat-template-file "${MLX_CKPT}/chat_template.jinja"
  --reasoning off --reasoning-format none
  --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1 -s 0
)

case "${mode}" in
  ar) ;;
  mtp)
    args+=(--spec-type draft-mtp -md "${GGUF_MTP}" -ngld all -ctkd f16 -ctvd f16)
    ;;
  dflash)
    args+=(--spec-type draft-dflash -md "${GGUF_DFLASH}" -ngld all -ctkd f16 -ctvd f16)
    ;;
  *) echo "mode inconnu: ${mode}" >&2; exit 2 ;;
esac

run_engine llama-server "${args[@]}" "$@"
