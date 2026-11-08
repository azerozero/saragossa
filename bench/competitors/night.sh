#!/usr/bin/env bash
# Enchaîne une nuit de campagne (protocole v5, §12). Machine libre, secteur.
#
# Usage : night.sh validate|1|2|partial RUN_DIR
#   validate : essai de validation (prompts de réglage 4k et sondes EOS)
#   1        : réglage, gel des réglages (freeze), puis eval-S et eval-M
#   2        : qualité (ids du corpus, PPL, KL, llama-bench), échantillonné
#              exploratoire, 256k
#   partial  : qualité, eval-S, sampled-S, eval-L-mlx, à partir d’un gel existant
# RUN_DIR doit contenir prompts/prompts.json (prompts.py build).
#
# Reprise : relancer la même nuit reprend l'étape interrompue. Un plan déjà tiré
# (plan-<phases>.json) n'est jamais retiré : il est réutilisé tel quel, et
# `run` saute les entrées terminées. Un gel déjà écrit n'est pas refait.
# Toute étape en échec (empreintes en désaccord, gel impossible, moteur
# survivant) arrête la nuit : rien n'est mesuré sur une base non validée.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "${here}/paths.sh"

night="$1"
run="$2"

# Environnement maîtrisé pour tout ce que la nuit lance (Python, moteurs de la
# phase qualité) : rien n'est hérité de l'environnement ambiant hors de cette liste.
clean=(env -i PATH="${PATH}" HOME="${HOME}" TMPDIR="${TMPDIR:-/tmp}" USER="${USER:-}"
  LANG=en_US.UTF-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  MODELS="${MODELS}" AI_MODEL="${AI_MODEL}" RUST_SRC="${RUST_SRC}" CORPUS_DIR="${CORPUS_DIR}"
  MLX_CKPT="${MLX_CKPT}" BENCH_MLX_CKPT="${BENCH_MLX_CKPT:-}"
  BENCH_ENV="${BENCH_ENV:-${here}/local.env}")
sys_py=("${clean[@]}" python3 -I -B)
venv=("${clean[@]}" "${MLX_VENV}/bin/python" -I -B)

"${sys_py[@]}" "${here}/run_matrix.py"
"${sys_py[@]}" "${here}/analyze.py"
if [[ "${night}" == partial ]]; then
  "${sys_py[@]}" "${here}/night_checks.py" partial-freeze "${run}"
fi

GPU_LOCK="${GPU_LOCK:-${HOME}/.cache/saragossa/gpu.lock}"
mkdir -p "$(dirname "${GPU_LOCK}")"
if [[ -d "${GPU_LOCK}" ]]; then
  GPU_LOCK="$(cd -P "${GPU_LOCK}" && pwd -P)"
else
  GPU_LOCK="$(cd -P "$(dirname "${GPU_LOCK}")" && pwd -P)/$(basename "${GPU_LOCK}")"
fi
if [[ "${GPU_LOCK_HELD:-}" != "${GPU_LOCK}" ]]; then
  until mkdir "${GPU_LOCK}" 2>/dev/null; do sleep 60; done
  echo "bench-competitors" > "${GPU_LOCK}/owner"
  trap 'rm -rf "${GPU_LOCK}"' EXIT
fi

phase_run() {
  local name
  name="$(IFS=+; echo "$*")"
  if [[ -f "${run}/plan-${name}.json" ]]; then
    cp "${run}/plan-${name}.json" "${run}/plan.json"
  else
    "${sys_py[@]}" "${here}/run_matrix.py" plan "${run}" "$@"
  fi
  "${venv[@]}" "${here}/prompts.py" expect "${run}/prompts" "${run}/plan.json"
  "${sys_py[@]}" "${here}/run_matrix.py" run "${run}"
}

stamp="$(date +%Y%m%dT%H%M%S)"
"${sys_py[@]}" "${here}/run_matrix.py" env "${run}" "${night}-${stamp}" > /dev/null
case "${night}" in
  validate)
    phase_run validate
    ;;
  1)
    phase_run tune
    if [[ ! -f "${run}/matrix.frozen.json" ]]; then
      "${venv[@]}" "${here}/run_matrix.py" freeze "${run}"
    fi
    phase_run eval-S eval-M
    ;;
  partial)
    "${sys_py[@]}" "${here}/night_checks.py" power
    "${clean[@]}" "${here}/quality_llama.sh" "${run}/quality"
    "${sys_py[@]}" "${here}/night_checks.py" power
    "${venv[@]}" "${here}/quality.py" mlx-ppl "${WIKITEXT_TEST}" > "${run}/quality/ppl-mlx.json"
    phase_run eval-S
    phase_run sampled-S
    phase_run eval-L-mlx
    ;;
  2)
    "${sys_py[@]}" "${here}/night_checks.py" power
    "${clean[@]}" "${here}/quality_llama.sh" "${run}/quality"
    "${sys_py[@]}" "${here}/night_checks.py" power
    "${venv[@]}" "${here}/quality.py" mlx-ppl "${WIKITEXT_TEST}" > "${run}/quality/ppl-mlx.json"
    phase_run sampled-S eval-L
    ;;
  *)
    echo "nuit inconnue: ${night}" >&2
    exit 2
    ;;
esac
"${venv[@]}" "${here}/analyze.py" "${run}" > "${run}/analyze-${night}-${stamp}.log"
