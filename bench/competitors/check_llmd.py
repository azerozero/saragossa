"""Contrôle court de llmd, sous verrou GPU : mémoire selon les fractions et
`prompt_tokens` de chaque prompt d'évaluation (décision post-validation).

  python3 -I check_llmd.py RUN_DIR CTX GPU_FRAC CACHE_FRAC [PROMPT...] > sortie.json

Lance llmd (lanceur du bench, environnement maîtrisé) avec ces fractions,
relève l'empreinte après chargement (`proc_pid_rusage`, `vmmap`), puis envoie
chaque prompt avec `max_tokens=1` et compare le `prompt_tokens` rapporté à la
longueur attendue de `prompts.json` (le nonce y compte toujours 16 tokens).
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import client  # noqa: E402
import memwatch  # noqa: E402
import run_matrix  # noqa: E402


def main(run_dir, ctx, gpu_frac, cache_frac, prompt_ids):
    run_dir = pathlib.Path(run_dir)
    matrix, _ = run_matrix.load_matrix()
    manifest = json.loads((run_dir / "prompts" / "prompts.json").read_text())
    manifest["by_id"] = {p["id"]: p for p in manifest["prompts"]}
    spec = run_matrix.resolve(matrix, "llmd-ar")
    spec["args"] = [a for a in spec["args"] if "memory-fraction" not in a] + [
        f"--gpu-memory-fraction={gpu_frac}", f"--cache-memory-fraction={cache_frac}"]
    entry = {"config_spec": spec, "class": "check", "ctx": ctx}
    (run_dir / "logs").mkdir(exist_ok=True)
    log = run_dir / "logs" / f"check-llmd-{ctx}-{gpu_frac}-{cache_frac}-{int(time.time())}.log"
    if not run_matrix.wait_port_free(matrix["port"]):
        raise SystemExit("port occupé")
    eng = run_matrix.Engine(matrix, entry, log)
    report = {"ctx": ctx, "gpu_memory_fraction": gpu_frac, "cache_memory_fraction": cache_frac,
              "argv": eng.argv, "log": str(log), "prompts": []}
    try:
        report["ready"] = eng.wait_ready(matrix["load_timeout_s"])
        if report["ready"]:
            report["rusage_after_load"] = memwatch.rusage(eng.proc.pid)
            report["vmmap_after_load"] = memwatch.file_backed(eng.proc.pid)
            for prompt_id in prompt_ids:
                content = run_matrix.content_for(manifest, prompt_id, "0" * 16)
                body = run_matrix.request_body(matrix, "llmd", "greedy", content, None, 1)
                record = client.chat_stream("127.0.0.1", eng.port, body, deadline_s=3600)
                reported = (record.get("usage") or {}).get("prompt_tokens")
                expected = manifest["by_id"][prompt_id]["prompt_tokens"]
                report["prompts"].append({
                    "prompt": prompt_id, "status": record["status"], "reported": reported,
                    "expected": expected, "equal": reported == expected,
                    "seconds": round(record["t_end_ns"] / 1e9, 1),
                    "error": record.get("error_body") or record.get("error")})
                print(json.dumps(report["prompts"][-1]), file=sys.stderr, flush=True)
            report["rusage_after_prompts"] = memwatch.rusage(eng.proc.pid)
    finally:
        report["group_gone"] = eng.stop()
    print(json.dumps(report, indent=1))
    if not report["group_gone"]:
        raise SystemExit("llmd survit à l'arrêt")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:])
