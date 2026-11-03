#!/usr/bin/env bash
# Lance saragossa serve pour le bench concurrents.
#
# Usage : launch_saragossa.sh MODE PORT CTX [ARGS...]
#   MODE : ar | mtp | dflash
#   CTX  : ignoré (saragossa dimensionne le KV par requête), gardé pour
#          l'interface commune des lanceurs.
# Réglages de la grille passés par l'environnement, s'ils sont définis :
#   SARAGOSSA_RUST_MTP_MAX_DRAFT (MTP), SARAGOSSA_RUST_DFLASH_TREE (DFlash).
set -euo pipefail
source "$(dirname "$0")/paths.sh"

mode="$1"
port="$2"
shift 3

# Les deux caches de préfixe sont coupés : celui de serve (AR) et celui du
# décodeur (MTP, DFlash).
export SARAGOSSA_SERVE_PREFIX_CACHE=0
export SARAGOSSA_RUST_PREFIX_CACHE=0
export SARAGOSSA_API_KEY="${SARAGOSSA_API_KEY:-bench}"

case "${mode}" in
  ar)
    export SARAGOSSA_RUST_MTP_DECODE=0
    unset SARAGOSSA_RUST_DFLASH_DRAFT_DIR
    ;;
  mtp)
    export SARAGOSSA_RUST_MTP_DECODE=1
    unset SARAGOSSA_RUST_DFLASH_DRAFT_DIR
    ;;
  dflash)
    # Le drafter passe par le chemin MTP : il exige la tête chargée.
    export SARAGOSSA_RUST_MTP_DECODE=1
    export SARAGOSSA_RUST_DFLASH_DRAFT_DIR="${DFLASH_DIR}"
    ;;
  *)
    echo "mode inconnu: ${mode}" >&2
    exit 2
    ;;
esac

env | grep '^SARAGOSSA_' | grep -v '^SARAGOSSA_API_KEY=' | sort | sed 's/^/bench-env: /' >&2
run_engine "${SARAGOSSA_BIN}" serve --port "${port}" \
  --model "qwen38=${MLX_CKPT}" --preload "$@"
