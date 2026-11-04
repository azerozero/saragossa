#!/usr/bin/env bash
# Enchaîne une nuit de campagne (protocole v4, §12). Machine libre, secteur.
#
# Usage : night.sh validate|1|2 RUN_DIR
#   validate : essai de validation (1 prompt de réglage à 4k, chaque config)
#   1        : réglage, gel des réglages (freeze), puis eval-S et eval-M
#   2        : qualité (ids du corpus, PPL, KL, llama-bench), échantillonné
#              exploratoire, 256k
# RUN_DIR doit contenir prompts/prompts.json (prompts.py build).
# Le verrou GPU des agents est pris s'il est défini (GPU_LOCK) et rendu à la fin.
# Toute étape en échec (empreintes en désaccord, gel impossible, moteur
# survivant) arrête la nuit : rien n'est mesuré sur une base non validée.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "${here}/paths.sh"

night="$1"
run="$2"
sys_py=(python3 -I -B)
venv=("${MLX_VENV}/bin/python" -I -B)

if [[ -n "${GPU_LOCK:-}" ]]; then
  until mkdir "${GPU_LOCK}" 2>/dev/null; do sleep 60; done
  echo "bench-competitors" > "${GPU_LOCK}/owner"
  trap 'rm -rf "${GPU_LOCK}"' EXIT
fi

phase_run() {
  "${sys_py[@]}" "${here}/run_matrix.py" plan "${run}" "$@"
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
    "${venv[@]}" "${here}/run_matrix.py" freeze "${run}"
    phase_run eval-S eval-M
    ;;
  2)
    "${here}/quality_llama.sh" "${run}/quality"
    "${venv[@]}" "${here}/quality.py" mlx-ppl "${WIKITEXT_TEST}" > "${run}/quality/ppl-mlx.json"
    phase_run sampled-S eval-L
    ;;
  *)
    echo "nuit inconnue: ${night}" >&2
    exit 2
    ;;
esac
"${venv[@]}" "${here}/analyze.py" "${run}" > "${run}/analyze-${night}-${stamp}.log"
