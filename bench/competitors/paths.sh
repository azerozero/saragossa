# shellcheck shell=bash
# Chemins du bench concurrents, seule source pour les scripts et Python
# (`config.py` lit ce fichier). Les valeurs locales viennent de `local.env`
# (non versionné, voir `local.env.example`) ou de l'environnement.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${BENCH_ENV:-${_here}/local.env}" ]]; then
  # shellcheck source=/dev/null
  source "${BENCH_ENV:-${_here}/local.env}"
fi

REPO="${REPO:-$(cd "${_here}/../.." && pwd)}"
: "${MODELS:?MODELS non défini (local.env)}"
: "${AI_MODEL:?AI_MODEL non défini (local.env)}"
: "${RUST_SRC:?RUST_SRC non défini (local.env)}"
: "${CORPUS_DIR:?CORPUS_DIR non défini (local.env)}"

MLX_CKPT="${MLX_CKPT:-${MODELS}/Qwen3.8-27B-4bit}"
MLX_VENV="${MLX_VENV:-${MODELS}/mlxenv}"
NVFP4_CKPT="${NVFP4_CKPT:-${MODELS}/nvfp4/Qwen3.8-27B-NVFP4}"
DFLASH_DIR="${DFLASH_DIR:-${MODELS}/Qwen3.8-27B-DFlash2}"

GGUF_SRC_DIR="${GGUF_SRC_DIR:-${AI_MODEL}/Qwen3.8-27B-GGUF-ggml}"
GGUF_DIR="${GGUF_DIR:-${AI_MODEL}/Qwen3.8-27B-GGUF-bench}"
GGUF_BF16="${GGUF_BF16:-${GGUF_SRC_DIR}/Qwen3.8-27B-BF16.gguf}"
GGUF_MTP_BF16="${GGUF_MTP_BF16:-${GGUF_SRC_DIR}/mtp-Qwen3.8-27B-BF16.gguf}"
GGUF_Q4_0="${GGUF_Q4_0:-${GGUF_DIR}/Qwen3.8-27B-Q4_0-pure.gguf}"
GGUF_Q4_K_M="${GGUF_Q4_K_M:-${GGUF_DIR}/Qwen3.8-27B-Q4_K_M.gguf}"
GGUF_MTP="${GGUF_MTP:-${GGUF_DIR}/mtp-Qwen3.8-27B-Q4_0-pure.gguf}"
GGUF_DFLASH="${GGUF_DFLASH:-${AI_MODEL}/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf}"

WIKITEXT_TRAIN="${WIKITEXT_TRAIN:-${CORPUS_DIR}/wikitext-2-raw/wiki.train.raw}"
WIKITEXT_TEST="${WIKITEXT_TEST:-${CORPUS_DIR}/wikitext-2-raw/wiki.test.raw}"

SARAGOSSA_BIN="${SARAGOSSA_BIN:-${REPO}/target/release/saragossa}"

# Clés exportées vers Python (config.py), lues par le script qui source ce fichier.
# shellcheck disable=SC2034
BENCH_PATH_KEYS=(REPO MODELS AI_MODEL RUST_SRC CORPUS_DIR MLX_CKPT MLX_VENV NVFP4_CKPT
  DFLASH_DIR GGUF_SRC_DIR GGUF_DIR GGUF_BF16 GGUF_MTP_BF16 GGUF_Q4_0 GGUF_Q4_K_M GGUF_MTP
  GGUF_DFLASH WIKITEXT_TRAIN WIKITEXT_TEST SARAGOSSA_BIN)

# Affiche la commande exacte sur stderr (journal du moteur), puis l'exécute :
# le PID suivi par l'ordonnanceur est celui du moteur.
run_engine() {
  printf 'bench-launch:' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  exec "$@"
}
