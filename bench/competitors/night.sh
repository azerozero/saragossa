#!/usr/bin/env bash
# Enchaîne une nuit de campagne (protocole v3, §11). Machine libre, secteur.
#
# Usage : night.sh validate|1|2 RUN_DIR
#   validate : essai de validation (1 prompt de réglage à 4k, chaque config)
#   1        : réglage, gel des réglages (freeze), puis eval-S et eval-M
#   2        : qualité (PPL, KL, llama-bench), échantillonné exploratoire, 256k
# RUN_DIR doit contenir prompts/prompts.json (prompts.py build).
# Le verrou GPU des agents est pris s'il est défini (GPU_LOCK) et rendu à la fin.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "${here}/paths.sh"

night="$1"
run="$2"
py=(python3 -I -B "${here}/run_matrix.py")
venv=("${MLX_VENV}/bin/python" -I -B)
corpus="${CORPUS:-${run}/corpus/wikitext-2-raw/wiki.test.raw}"

if [[ -n "${GPU_LOCK:-}" ]]; then
  until mkdir "${GPU_LOCK}" 2>/dev/null; do sleep 60; done
  echo "bench-competitors" > "${GPU_LOCK}/owner"
  trap 'rm -rf "${GPU_LOCK}"' EXIT
fi

phase_run() {
  "${py[@]}" plan "${run}" "$@"
  "${venv[@]}" "${here}/prompts.py" expect "${run}/prompts" "${run}/plan.json"
  "${py[@]}" run "${run}"
}

"${py[@]}" env "${run}" > /dev/null
case "${night}" in
  validate)
    phase_run validate
    ;;
  1)
    phase_run tune
    "${py[@]}" freeze "${run}"
    phase_run eval-S eval-M
    ;;
  2)
    "${here}/quality_llama.sh" "${corpus}" "${run}/quality"
    "${venv[@]}" "${here}/quality.py" mlx-ppl "${corpus}" > "${run}/quality/ppl-mlx.json"
    phase_run sampled-S eval-L
    ;;
  *)
    echo "nuit inconnue: ${night}" >&2
    exit 2
    ;;
esac
"${venv[@]}" "${here}/analyze.py" "${run}" > "${run}/analyze-${night}.log"
