"""Comparaison A/B CLI, stdlib ; sans argument : auto-tests factices.

Lancer sous le verrou externe, sans l'imbriquer ici :
  python3 -I -B tools/gpulock.py run --owner ab --prio 10 -- \
    python3 -I -B bench/ab.py run OUT --a BIN --b BIN --model-dir DIR \
      --prompt-tokens 4096 --max-tokens 256
  python3 -I -B bench/ab.py report OUT [--json]

Un bloc contient ABBA (ou BAAB) pour chaque prompt. Reprise à configuration
identique : les blocs interrompus sont rejoués entièrement ; leurs anciennes
tentatives restent au journal, exclues explicitement du rapport. Les échecs
d'un bloc terminé restent invalidants. Ne pas modifier les entrées pendant
la campagne. Le modèle est référencé par son chemin, sans lire ses poids.

IC de Student à 95 % sur les blocs : chaque bloc fournit la moyenne des
log-ratios de ses prompts, de poids égaux. Au moins quatre blocs complets.
La conclusion porte sur ces prompts uniquement. Indistinct ne prouve pas
l'équivalence. La latence de prefill est prompt_tokens / prefill_tok_s,
jamais un temps au premier token ; sous une seconde, elle est imprécise.
AR compte n−1 decode_tokens, MTP en compte n : si ces comptes diffèrent,
decode_tok_s et le prefill (estimé par soustraction en MTP) sont non comparables.
Utiliser alors --primary tok_s ; l'identité des ids reste exigée.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "competitors"))
import sensors  # noqa: E402

METRICS = ("decode_tok_s", "prefill_ms", "tok_s")
CODES = {"gain": 0, "perte": 1, "indistinct": 2, "invalide": 3}
# Quantiles t(0,975 ; ddl), indexés à partir d'un degré de liberté.
STUDENT_975 = (
    12.7062047364, 4.3026527297, 3.1824463053, 2.7764451052, 2.5705818356,
    2.4469118511, 2.3646242516, 2.3060041352, 2.2621571629, 2.2281388520,
    2.2009851601, 2.1788128297, 2.1603686565, 2.1447866879, 2.1314495456,
    2.1199052992, 2.1098155778, 2.1009220402, 2.0930240544, 2.0859634473,
    2.0796138447, 2.0738730679, 2.0686576104, 2.0638985616, 2.0595385528,
    2.0555294386, 2.0518305165, 2.0484071418, 2.0452296421, 2.0422724563,
)


def student_critical(df):
    if df <= len(STUDENT_975):
        return STUDENT_975[df - 1]
    # Hors table, inversion numérique de la CDF : l'intégrale de cos^(ddl−1)
    # se calcule par récurrence, sans approximation normale ni dépendance.
    factor = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi)
    low, high = 0.0, STUDENT_975[-1]
    for _ in range(50):
        middle = (low + high) / 2
        angle = math.atan(middle / math.sqrt(df))
        sine, cosine = math.sin(angle), math.cos(angle)
        integral = angle if df % 2 else sine
        for exponent in range(2 if df % 2 else 3, df, 2):
            integral = (sine * cosine ** (exponent - 1) + (exponent - 1) * integral) / exponent
        if 0.5 + factor * integral < 0.975:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def validate_metrics(values, max_tokens=None):
    for key in ("decode_tok_s", "tok_s", "prefill_tok_s", "prompt_tokens", "decode_tokens", "tokens"):
        zero_decode = key in ("decode_tokens", "decode_tok_s") and values.get("decode_tokens") == 0
        if key not in values or not math.isfinite(values[key]) or values[key] < 0 or (values[key] == 0 and not zero_decode):
            raise ValueError(f"métrique absente ou non positive : {key}")
    if "prefill_ms" not in values or not math.isfinite(values["prefill_ms"]) or values["prefill_ms"] < 0:
        raise ValueError("prefill_ms absent ou négatif")
    for key in ("prompt_tokens", "decode_tokens", "tokens"):
        if not float(values[key]).is_integer():
            raise ValueError(f"compte non entier : {key}")
    if values["decode_tokens"] not in (values["tokens"] - 1, values["tokens"]):
        raise ValueError("decode_tokens doit valoir tokens - 1 (AR) ou tokens (MTP)")
    if max_tokens is not None and values["tokens"] != max_tokens:
        raise ValueError("tokens différent de --max-tokens avec --ignore-stop-tokens")
    latency = values["prompt_tokens"] / values["prefill_tok_s"]
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("latence de prefill non finie ou non positive")
    return latency


def parse(raw, identity):
    """Seulement des lignes stderr complètes, ancrées ; aucun texte stdout."""
    lines = [line[:-1].rstrip("\r") for line in raw.decode("utf-8").splitlines(True)
             if line.endswith("\n")]
    metrics = [line[8:] for line in lines if line.startswith("metrics ")]
    if len(metrics) != 1:
        raise ValueError("ligne metrics absente ou multiple")
    pairs = [item.split("=", 1) for item in metrics[0].split()]
    values = {key: float(value) for key, value in pairs}
    if len(values) != len(pairs) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("métrique dupliquée ou non finie")
    validate_metrics(values)
    ids = [line[11:] for line in lines if line.startswith("oracle_ids ")]
    if identity == "ids":
        if len(ids) != 1 or any(not item.isascii() or not item.isdecimal()
                                for item in ids[0].split(",")):
            raise ValueError("oracle_ids absent ou mal formé")
        ids = [int(item) for item in ids[0].split(",")]
        if len(ids) != values["tokens"]:
            raise ValueError("oracle_ids incomplet")
        return values, ids
    return values, None


def digest(path):
    with open(path, "rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def plan(args):
    rng = random.Random(args.random_seed)
    prompts = ([["--prompt-file", str(pathlib.Path(p).resolve())] for p in args.prompt_file]
               if args.prompt_file else [["--prompt-tokens", str(args.prompt_tokens)]])
    common = ["--model-dir", str(pathlib.Path(args.model_dir).resolve()), "--max-tokens",
              str(args.max_tokens), "--temperature", str(args.temperature), "--seed",
              str(args.seed), "--ignore-stop-tokens", "--metrics"]
    sides = {}
    for side in "ab":
        env = {}
        for item in getattr(args, "env_" + side):
            key, sep, value = item.partition("=")
            if not sep or not key or "\0" in item:
                raise ValueError("environnement attendu : K=V")
            env[key] = value
        if args.identity == "ids":
            env["SARAGOSSA_RUST_ORACLE_DUMP_IDS"] = "1"
        binary = str(pathlib.Path(getattr(args, side)).resolve())
        sides[side] = {"binary": binary, "sha256": digest(binary), "env": env}
    return {"version": 1, "sides": sides, "args": common, "prompts": prompts,
            "prompt_sha256": [digest(p[1]) if p[0] == "--prompt-file" else None for p in prompts],
            "orders": [rng.choice(("abba", "baab")) if args.order == "random" else "abba"
                       for _ in range(args.blocks)], "order": args.order,
            "random_seed": args.random_seed, "identity": args.identity,
            "primary": args.primary, "min_effect": args.min_effect,
            "allow_battery": args.allow_battery}


def journal(root, repair=False):
    path = root / "results.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    tail = raw[raw.rfind(b"\n") + 1:]
    if tail and repair:
        (root / "raw" / f"journal-interrompu-{uuid.uuid4().hex}").write_bytes(tail)
        path.write_bytes(raw[:-len(tail)])
    rows = [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b"\n")]
    return rows, bool(tail)


def selected(config, rows):
    active, excluded = {}, []
    for block, order in enumerate(config["orders"]):
        group = [r for r in rows if r["block"] == block]
        attempt = group[-1]["attempt"] if group else None
        current = [r for r in group if r["attempt"] == attempt]
        excluded.extend(r for r in group if r["attempt"] != attempt)
        expected = [(p, i, side) for p in range(len(config["prompts"])) for i, side in enumerate(order)]
        if ([(r["prompt"], r["slot"], r["side"]) for r in current] == expected
                and current[-1].get("block_done")):
            active[block] = current
        else:
            excluded.extend(current)
    return active, excluded


def passage(root, config, sampler, block, attempt, prompt, slot, side):
    spec = config["sides"][side]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SARAGOSSA_")}
    env.update(spec["env"])
    argv = [spec["binary"], *config["args"], *config["prompts"][prompt]]
    stem = f"raw/{block}-{attempt}-{prompt}-{slot}-{side}"
    row = dict(block=block, attempt=attempt, prompt=prompt, slot=slot, side=side,
               argv=argv, env=spec["env"], stdout=stem + ".stdout", stderr=stem + ".stderr")
    row["idle"] = sensors.wait_idle({}, sampler, require_sensor=True)
    row["power_before"] = sensors.power_state()
    row["thermal_before"] = sensors.thermal(sampler)
    errors = []
    start = time.perf_counter_ns()
    with (root / row["stdout"]).open("wb") as out, (root / row["stderr"]).open("wb") as err:
        try:
            row["returncode"] = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                                timeout=3600).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            row["returncode"] = None
            errors.append(str(error))
    end = time.perf_counter_ns()
    row["power_after"] = sensors.power_state()
    row["thermal_after"] = sensors.thermal(sampler)
    row["sensors"] = sampler.snapshot(start, end)
    row["start_ns"], row["end_ns"] = start, end
    if row["returncode"] != 0:
        errors.append("passage en échec")
    try:
        row["metrics"], row["ids"] = parse((root / row["stderr"]).read_bytes(), config["identity"])
    except (ValueError, UnicodeError) as error:
        errors.append(str(error))
    row["errors"] = errors
    return row


def run(root, config):
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    manifest = root / "plan.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("reprise refusée : configuration, binaire ou prompt modifié")
    else:
        manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    rows, _ = journal(root, repair=True)
    active, _ = selected(config, rows)
    with sensors.Sampler(root / "sensors.jsonl") as sampler:
        for block, order in enumerate(config["orders"]):
            if block in active:
                continue
            attempt = uuid.uuid4().hex
            for prompt in range(len(config["prompts"])):
                for slot, side in enumerate(order):
                    row = passage(root, config, sampler, block, attempt, prompt, slot, side)
                    row["block_done"] = prompt == len(config["prompts"]) - 1 and slot == 3
                    with (root / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())


def estimate(strata, minimum):
    if len({len(values) for values in strata}) > 1:
        raise ValueError("blocs non alignés entre prompts")
    logs = [statistics.fmean(block) for block in zip(*strata)]
    mean = statistics.fmean(logs) if logs else None
    ratio = math.exp(mean) if mean is not None else None
    if len(logs) < 4:
        return dict(ratio=ratio, interval=None, verdict="indistinct", note="insuffisant : moins de 4 blocs")
    margin = student_critical(len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs))
    interval = [math.exp(mean - margin), math.exp(mean + margin)]
    verdict = "gain" if interval[0] > 1 + minimum else "perte" if interval[1] < 1 - minimum else "indistinct"
    return dict(ratio=ratio, interval=interval, verdict=verdict, note="indistinct ne prouve pas l'équivalence")


def report(root):
    config = json.loads((root / "plan.json").read_text())
    rows, tail = journal(root)
    active, excluded = selected(config, rows)
    errors = ["blocs incomplets ou absents"] if len(active) != len(config["orders"]) else []
    if tail:
        errors.append("journal interrompu")
    indicative_reasons = ["alimentation sur batterie autorisée"] if config["allow_battery"] else []
    non_comparable, imprecise_prefill, decode_unavailable = False, False, False
    max_tokens = (int(config["args"][config["args"].index("--max-tokens") + 1])
                  if "--ignore-stop-tokens" in config["args"] else None)
    ids_by_prompt = {}
    for group in active.values():
        for row in group:
            label = f"bloc {row['block']} prompt {row['prompt']} {row['side']}{row['slot']}"
            reasons = list(row["errors"])
            unknown = []
            if "metrics" in row:
                try:
                    latency = validate_metrics(row["metrics"], max_tokens)
                    imprecise_prefill |= latency < 1
                    decode_unavailable |= row["metrics"]["decode_tokens"] == 0
                except ValueError as error:
                    reasons.append(str(error))
            for phase in ("before", "after"):
                state = row["power_" + phase]
                if not config["allow_battery"] and (state["on_ac"] is not True or state["low_power"] is not False):
                    reasons.append("alimentation hors secteur, mode éco ou inconnue")
                state = row["thermal_" + phase]
                if state.get("pressure") is None:
                    unknown.append(f"pression thermique inconnue ({phase})")
                if state.get("error"):
                    unknown.append(f"lecture thermique indisponible ({phase})")
                if state["throttled"] or (state.get("pressure") is not None and state["pressure"] >= 2):
                    reasons.append("bridage thermique")
            if row["idle"]["machine_busy"]:
                reasons.append("machine occupée à expiration de l'attente")
            if row["idle"].get("idle_established") is not True and not row["idle"]["machine_busy"]:
                unknown.append("repos GPU non établi : " + row["idle"].get("reason", "preuve de repos absente"))
            sample = row.get("sensors") or {}
            if not sample.get("n_samples"):
                unknown.append("capteur absent ou aucun échantillon pendant le passage")
            elif any(sample.get(key) is None for key in
                     ("gpu_temp_c_start", "gpu_temp_c_end", "gpu_freq_mhz_median")):
                unknown.append("capteur GPU indisponible (température ou fréquence)")
            if config["identity"] == "ids" and "ids" in row:
                reference = ids_by_prompt.setdefault(row["prompt"], row["ids"])
                if reference != row["ids"]:
                    reasons.append("identité d'ids violée")
            errors.extend(f"{label} : {reason}" for reason in reasons)
            indicative_reasons.extend(f"{label} : {reason}" for reason in unknown)
        for prompt in range(len(config["prompts"])):
            samples = [r.get("metrics", {}) for r in group if r["prompt"] == prompt]
            for key in ("prompt_tokens", "tokens"):
                if len({m.get(key) for m in samples}) != 1:
                    errors.append(f"bloc {group[0]['block']} prompt {prompt} : {key} différents")
            counts = [{r.get("metrics", {}).get("decode_tokens") for r in group
                       if r["prompt"] == prompt and r["side"] == side} for side in "ab"]
            if any(len(count) != 1 for count in counts):
                errors.append(f"bloc {group[0]['block']} prompt {prompt} : decode_tokens instables par côté")
            elif counts[0] != counts[1]:
                non_comparable = True
    if non_comparable and config["primary"] != "tok_s":
        errors.append("decode_tok_s et prefill non comparables (AR/MTP) : utiliser --primary tok_s")
    if decode_unavailable and config["primary"] == "decode_tok_s":
        errors.append("decode_tok_s indisponible : aucun token décodé ; utiliser --primary tok_s")
    metrics = {}
    for metric in METRICS:
        unavailable = decode_unavailable and metric == "decode_tok_s"
        comparable = (not non_comparable or metric == "tok_s") and not unavailable
        strata = []
        if not errors and comparable:
            for prompt in range(len(config["prompts"])):
                logs = []
                for group in active.values():
                    means = [statistics.fmean(
                        r["metrics"]["prompt_tokens"] / r["metrics"]["prefill_tok_s"]
                        if metric == "prefill_ms" else r["metrics"][metric] for r in group
                                             if r["prompt"] == prompt and r["side"] == side) for side in "ab"]
                    logs.append((math.log(means[1]) - math.log(means[0])) * (-1 if metric == "prefill_ms" else 1))
                strata.append(logs)
        metrics[metric] = (dict(ratio=None, interval=None, verdict="invalide") if errors
                           else estimate(strata, config["min_effect"]) if comparable
                           else dict(ratio=None, interval=None, verdict="indistinct"))
        metrics[metric]["comparable"] = comparable
        if unavailable:
            metrics[metric]["note"] = "indisponible : aucun token décodé"
        elif not comparable:
            metrics[metric]["note"] = "non comparables : comptes decode AR/MTP ; prefill MTP estimé par soustraction"
        elif metric == "prefill_ms" and imprecise_prefill:
            metrics[metric]["note"] = (metrics[metric].get("note", "") +
                                       " ; imprécise : latence de prefill sous 1 s").lstrip(" ;")
            if not errors:
                metrics[metric]["verdict"] = "indistinct"
    return dict(metrics=metrics, primary=config["primary"], blocks=len(active), errors=errors,
                identity=config["identity"], indicative=bool(indicative_reasons),
                indicative_reasons=indicative_reasons,
                identity_status=("désactivée" if config["identity"] == "none" else
                                 "non établie" if errors else "vérifiée"),
                excluded=[{k: r[k] for k in ("block", "attempt", "prompt", "slot")} for r in excluded],
                scope="La conclusion porte sur ces prompts uniquement ; indistinct ne prouve pas l'équivalence.",
                sensors=[dict(block=r["block"], prompt=r["prompt"], side=r["side"], **(r.get("sensors") or {}))
                         for group in active.values() for r in group])


def display(result, as_json):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return
    print(result["scope"])
    print(f"Blocs : {result['blocks']} ; identité : {result['identity']} ({result['identity_status']}) ; "
          f"tentatives exclues (passages) : {len(result['excluded'])}" + (" ; INDICATIF" if result["indicative"] else ""))
    print("métrique          ratio favorable B    IC 95 %                 verdict")
    for metric, value in result["metrics"].items():
        ratio = f"{value['ratio']:.4f}" if value["ratio"] is not None else "—"
        interval = " / ".join(f"{v:.4f}" for v in value["interval"]) if value["interval"] else "—"
        print(f"{metric:17} {ratio:20} {interval:23} {value['verdict']} {value.get('note', '')}")
    print("bloc prompt côté   GPU °C début / fin   fréquence GPU MHz (médiane par passage)")
    for sample in result["sensors"]:
        print(f"{sample['block']:4} {sample['prompt']:6} {sample['side']:4}   "
              f"{sample.get('gpu_temp_c_start')} / {sample.get('gpu_temp_c_end')}   {sample.get('gpu_freq_mhz_median')}")
    for error in result["errors"]:
        print("invalide : " + error)
    for reason in result["indicative_reasons"]:
        print("indicatif : " + reason)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("out_dir", type=pathlib.Path)
    for side in "ab":
        launch.add_argument("--" + side, required=True)
        launch.add_argument("--env-" + side, action="append", default=[])
    launch.add_argument("--model-dir", required=True)
    prompt = launch.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-file", nargs="+", action="extend")
    prompt.add_argument("--prompt-tokens", type=int)
    for name, default in (("max-tokens", 256), ("blocks", 6), ("seed", 1), ("random-seed", 42)):
        launch.add_argument("--" + name, type=int, default=default)
    launch.add_argument("--temperature", type=float, default=0)
    launch.add_argument("--min-effect", type=float, default=0.01)
    launch.add_argument("--order", choices=("abba", "random"), default="abba")
    launch.add_argument("--identity", choices=("ids", "none"), default="ids")
    launch.add_argument("--primary", choices=METRICS, default="decode_tok_s")
    launch.add_argument("--allow-battery", action="store_true")
    read = sub.add_parser("report")
    read.add_argument("out_dir", type=pathlib.Path)
    read.add_argument("--json", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if error.code == 0:  # --help est la seule sortie normale du parseur.
            return 0
        raise ValueError("arguments invalides") from error
    if args.command == "run":
        if (min(args.blocks, args.max_tokens, (args.prompt_tokens if args.prompt_tokens is not None else 1)) <= 0 or args.seed < 0
                or not math.isfinite(args.temperature) or args.temperature < 0
                or not 0 <= args.min_effect < 1):
            raise ValueError("paramètre numérique hors limites")
        run(args.out_dir, plan(args))
    result = report(args.out_dir)
    display(result, getattr(args, "json", False))
    return 3 if result["errors"] else 2 if result["indicative"] else CODES[result["metrics"][result["primary"]]["verdict"]]


def main(argv=None):
    try:
        return cli(argv)
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        print(f"invalide : {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            import runpy
            tests = runpy.run_path(str(pathlib.Path(__file__).with_name("ab_test.py")), init_globals={"ab": sys.modules[__name__]})
            if tests.get("TESTS_COMPLETED") is not True:
                raise RuntimeError("auto-tests non exécutés jusqu'au bout")
            code = 0
        else:
            code = main()
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        print(f"invalide : {type(error).__name__} : {error}", file=sys.stderr)
        code = 3
    sys.exit(code)
