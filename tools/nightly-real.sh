#!/usr/bin/env bash
# Tests ignorés de modèles réels : verrou coopératif obligatoire.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export GPU_LOCK="${GPU_LOCK:-${TMPDIR:-/tmp}/saragossa-gpu.lock}"
export SARAGOSSA_REQUIRE_REAL_MODELS=1
export SARAGOSSA_REQUIRE_METAL=1
export HF_HUB_OFFLINE=1
export CARGO_NET_OFFLINE=true
log_dir=${NIGHTLY_LOG_DIR:-target/nightly-real}
mkdir -p "$log_dir"
log_file="$log_dir/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"

# Les filtres libtest sont réunis en une seule invocation (union sans doublon).
# gpulock possède le groupe de processus et propage les signaux à ses enfants.
set +e
python3 -I -B "${GPU_LOCK_TOOL:-$root/tools/gpulock.py}" run \
    --owner "${GPU_LOCK_OWNER:-nightly-real}" --prio "${GPU_LOCK_PRIO:-5}" -- \
    taskpolicy -b cargo test --locked --release --all-features -- \
    --ignored live_ golden_ quality_ external_ --test-threads=1 --nocapture \
    2>&1 | tee "$log_file"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" == 0 ]] && grep -Eiq "(^|[[:space:]])skip:" "$log_file"; then
    printf "nightly-real: précondition sautée malgré le mode strict\n" | tee -a "$log_file"
    rc=1
fi
printf "nightly-real rc=%s\n" "$rc" | tee -a "$log_file"
exit "$rc"
