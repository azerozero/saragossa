"""Compare trois processus indépendants à 32k/128k, sous tools/gpulock.py.

Ne télécharge aucun poids et ne prend pas le verrou lui-même. Compilez d'abord
l'exemple prefix_ssd ; les JSON/logs restent dans un nouveau répertoire privé.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/prefix_ssd"))
    parser.add_argument("--mode", choices=("ar", "mtp"), default="ar")
    parser.add_argument("--tokens", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Les mêmes réglages dans les trois processus ; aucune interaction avec le
    # répertoire diagnostic et aucun drafter implicite dans l'oracle MTP.
    environment = os.environ.copy()
    for name in ("SARAGOSSA_RUST_STATE_SNAPSHOT", "RETI_RUST_STATE_SNAPSHOT",
                 "SARAGOSSA_RUST_DFLASH_DRAFT_DIR", "RETI_RUST_DFLASH_DRAFT_DIR"):
        environment.pop(name, None)
    for length in args.tokens:
        reports = []
        with tempfile.TemporaryDirectory(prefix=f"ssd-{length}-", dir=args.output) as cache:
            for phase in ("baseline", "cold", "warm"):
                output = args.output / f"{length}-{phase}.json"
                log_path = args.output / f"{length}-{phase}.log"
                try:
                    with log_path.open("wb") as log:
                        subprocess.run([str(args.binary.resolve()), str(args.model.resolve()), cache,
                                        str(length), phase, args.mode, str(output.resolve())],
                                       stdout=log, stderr=log, env=environment, check=True)
                except subprocess.CalledProcessError:
                    print(f"Échec à {length} tokens ({phase}) ; journal : {log_path}",
                          file=sys.stderr, flush=True)
                    # Relire après fermeture pour inclure tout le stderr, sans
                    # garder un journal potentiellement volumineux en mémoire.
                    with log_path.open(encoding="utf-8", errors="replace") as log:
                        for line in log:
                            print(line, end="", file=sys.stderr)
                    raise
                reports.append(json.loads(output.read_text()))
            if not reports[0]["generated_tokens"] == reports[1]["generated_tokens"] == reports[2]["generated_tokens"]:
                raise RuntimeError(f"divergence greedy à {length} tokens ; voir les JSON")
            baseline, cold, warm = (report["prefill_or_reload_ms"] for report in reports)
            print(f"tokens={length} prefill_ms={baseline:.3f} staging_ms={cold:.3f} "
                  f"reload_ms={warm:.3f} ratio={baseline / warm:.3f} ids_identiques=true", flush=True)


if __name__ == "__main__":
    main()
