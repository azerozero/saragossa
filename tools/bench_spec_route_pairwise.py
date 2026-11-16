#!/usr/bin/env python3
"""ABBA du routage spéculatif : AR, MTP, DFlash et routage, un worker JSONL.

Même protocole que la campagne du 15 novembre (tools/bench_mtp_dflash_pairwise.py
de perf/mtp-dflash-fusion) : 256 tokens greedy sans EOS, une chauffe par mode
exclue, ordre direct puis inverse, préfixe froid, MTP sans lookup, drafter u4.
Construire examples/spec_route_measure.rs en release, lancer sous gpulock avec
Python -I ; aucun téléchargement ni écriture de poids.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import select
import statistics
import subprocess
import time

MODES = ["ar", "mtp", "dflash", "route"]


def environment(drafter):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SARAGOSSA_", "RETI_"))}
    env.update(SARAGOSSA_RUST_PREFIX_CACHE="0", SARAGOSSA_RUST_DFLASH_SWITCH_MIN="0",
               SARAGOSSA_RUST_MTP_LOOKUP_DRAFT="0", SARAGOSSA_RUST_DFLASH_QUANT_BITS="4",
               SARAGOSSA_RUST_DFLASH_DRAFT_DIR=str(drafter))
    return env


class Worker:
    def __init__(self, binary, args, log):
        self.process = subprocess.Popen(
            [str(binary.resolve()), str(args.model.resolve())],
            env=environment(args.drafter.resolve()), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        try:
            if self.read() != {"ready": True}:
                raise RuntimeError("worker non prêt")
        except BaseException:
            self.close()
            raise

    def read(self):
        if not select.select([self.process.stdout], [], [], 600)[0]:
            raise TimeoutError("worker sans réponse depuis 600 secondes")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"worker interrompu : code {self.process.poll()}")
        return json.loads(line)

    def run(self, case, mode, tokens):
        self.process.stdin.write(json.dumps(dict(prompt=case["prompt"], mode=mode,
            context_tokens=case.get("context_tokens", 0), tokens=tokens)) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=30)


def mode_order(repeat):
    return list(reversed(MODES)) if repeat >= 0 and repeat % 2 else list(MODES)


def bootstrap(values):
    rng = random.Random(0)
    samples = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(10000))
    return [samples[249], samples[9749]]


def host_load():
    listing = subprocess.check_output(["ps", "-axo", "pcpu,comm"], text=True)
    compilers = []
    for line in listing.splitlines()[1:]:
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and Path(fields[1]).name in ("rustc", "clippy-driver", "cc1"):
            compilers.append(float(fields[0].replace(",", ".")))
    return dict(load_average=os.getloadavg(), compiler_count=len(compilers),
                compiler_cpu=sum(compilers))


def wait_calm(seconds):
    deadline = time.monotonic() + seconds
    load = host_load()
    while (load["load_average"][0] >= 2 or load["compiler_count"]) and time.monotonic() < deadline:
        time.sleep(10)
        load = host_load()
    return load


def model_manifest(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors"):
            stat = path.stat()
            entry = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if path.suffix == ".json":
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result[path.name] = entry
    return result


def paired_gain(subset, mode, base_mode):
    base = {r["repeat"]: r for r in subset if r["mode"] == base_mode}
    return [100 * (r["decode_tok_s"] / base[r["repeat"]]["decode_tok_s"] - 1)
            for r in subset if r["mode"] == mode and r["repeat"] in base]


def summarize(rows):
    result = []
    for case in dict.fromkeys(r["case"] for r in rows):
        subset = [r for r in rows if r["case"] == case and not r["warmup"]]
        modes = {}
        for mode in MODES:
            runs = [r for r in subset if r["mode"] == mode]
            if not runs:
                continue
            rates = [r["decode_tok_s"] for r in runs]
            modes[mode] = dict(n=len(runs), tok_s=statistics.median(rates),
                tok_s_ci95=bootstrap(rates),
                ttft_ms=statistics.median(r["ttft_ms"] for r in runs),
                total_ms=statistics.median(r["total_ms"] for r in runs),
                acceptance=(statistics.median(r["accepted"] / r["proposed"]
                    for r in runs) if all(r["proposed"] for r in runs) else None),
                routed=sorted({r.get("routed") or "ar" for r in runs}),
                ids_equal=all(r["ids_equal"] for r in runs))
        entry = dict(case=case, modes=modes)
        best = max(("mtp", "dflash"), key=lambda m: modes.get(m, {}).get("tok_s", 0))
        gains = paired_gain(subset, "route", best)
        if gains:
            entry.update(best=best, route_vs_best_pct=statistics.median(gains),
                         route_vs_best_ci95=bootstrap(gains))
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--prompts", type=Path,
                        default=Path(__file__).parent / "fixtures/spec_route_prompts.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--calm-wait", type=float, default=0,
                        help="avant chaque passage, attendre au plus N s une charge 1 min < 2 sans compilateur")
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats % 2:
        parser.error("repeats doit être pair et >= 2")
    if not os.environ.get("GPU_LOCK_HELD"):
        parser.error("lancer sous gpulock run")
    cases = json.loads(args.prompts.read_text())
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
        if {c["id"] for c in cases} != set(args.ids):
            parser.error("ID inconnu")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(created=time.time(), repeats=args.repeats, cases=cases,
                    binary=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    model=model_manifest(args.model), drafter=model_manifest(args.drafter),
                    flags={k: v for k, v in environment(args.drafter).items()
                           if k.startswith("SARAGOSSA_") and not k.endswith("_DIR")})
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    rows = []
    worker = None
    try:
        with (args.output / "worker.stderr").open("x") as log, \
             (args.output / "results.jsonl").open("x") as report:
            worker = Worker(args.binary, args, log)
            for case in cases:
                reference = None
                for repeat in range(-1, args.repeats):
                    load = wait_calm(args.calm_wait)
                    for mode in mode_order(repeat):
                        row = worker.run(case, mode, 256)
                        if reference is None:
                            reference = row["ids"]
                        row.update(case=case["id"], prompt_class=case["class"],
                                   repeat=repeat, warmup=repeat < 0,
                                   ids_equal=row["ids"] == reference, host_load=load)
                        row["first_divergence"] = next((i for i, (a, b) in
                            enumerate(zip(reference, row["ids"])) if a != b), None)
                        report.write(json.dumps(row) + "\n")
                        report.flush()
                        rows.append(row)
                        print(f'{case["id"]} r={repeat} {mode}: {row["decode_tok_s"]:.2f} tok/s '
                              f'route={row["routed"]} IDs={row["ids_equal"]} '
                              f'diff={row["first_divergence"]}', flush=True)
                (args.output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    finally:
        if worker:
            worker.close()


if __name__ == "__main__":
    main()
