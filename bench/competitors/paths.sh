# Chemins du bench concurrents, sourcés par les scripts de lancement.
# Chaque valeur peut être surchargée par l'environnement.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODELS="${MODELS:-/Users/ludwig/workspace/saragossa-oss/models}"
AI_MODEL="${AI_MODEL:-/Users/ludwig/workspace/ai_model}"

MLX_CKPT="${MLX_CKPT:-${MODELS}/Qwen3.8-27B-4bit}"
MLX_VENV="${MLX_VENV:-${MODELS}/mlxenv}"
NVFP4_CKPT="${NVFP4_CKPT:-${MODELS}/nvfp4/Qwen3.8-27B-NVFP4}"
DFLASH_DIR="${DFLASH_DIR:-${MODELS}/Qwen3.8-27B-DFlash2}"

GGUF_DIR="${GGUF_DIR:-${AI_MODEL}/Qwen3.8-27B-GGUF-bench}"
GGUF_Q4_0="${GGUF_Q4_0:-${GGUF_DIR}/Qwen3.8-27B-Q4_0-pure.gguf}"
GGUF_Q4_K_M="${GGUF_Q4_K_M:-${GGUF_DIR}/Qwen3.8-27B-Q4_K_M.gguf}"
GGUF_MTP="${GGUF_MTP:-${GGUF_DIR}/mtp-Qwen3.8-27B-Q4_0-pure.gguf}"
GGUF_DFLASH="${GGUF_DFLASH:-${AI_MODEL}/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf}"

SARAGOSSA_BIN="${SARAGOSSA_BIN:-${REPO}/target/release/saragossa}"

# Affiche la commande exacte sur stderr (journal du moteur), puis l'exécute :
# le PID suivi par l'ordonnanceur est celui du moteur.
run_engine() {
  printf 'bench-launch:' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  exec "$@"
}
