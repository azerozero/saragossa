"""Analyse du bench concurrents : contrôles, métriques par requête, tableaux T1/T2/T3.

  models/mlxenv/bin/python -I bench/competitors/analyze.py RUN_DIR [REFERENCE]

Le tokenizer HF du checkpoint (paquet `tokenizers` du venv MLX) sert au
recoupement du compte de tokens ; sans lui, ce recoupement est omis.

Règles (protocole v3, §8 et §9) :
- k, tokens du premier événement : exact (= 1) quand les événements de sortie
  et les tokens sont en bijection (événements = N ou N − 1 si l'EOS compte
  sans texte) ; sinon retokenisation du premier fragment, marquée « ≈ ».
- débit de livraison D = (N_texte − k) / (t_dernier − t_premier) ; indéfini
  avec un seul événement de sortie.
- unité statistique : le prompt. Médiane par (config, prompt) sur blocs et
  graines, rapport apparié par prompt, moyenne géométrique sur les prompts,
  IC 95 % par bootstrap hiérarchique (prompts, puis répétitions).
"""

import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prompts import CHECKPOINT  # noqa: E402

BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20261104
PRACTICAL = 0.05
SPEC_PAIRS = {
    "sar-mtp": "sar-ar", "sar-dflash": "sar-ar",
    "llq40-mtp": "llq40-ar", "llq40-dflash": "llq40-ar",
    "llkm-mtp": "llkm-ar", "llkm-dflash": "llkm-ar",
    "llmd-native": "llmd-ar", "llmd-native-s": "llmd-ar-s",
}


def tokenizer():
    try:
        import tokenizers
    except ImportError:
        return None
    return tokenizers.Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def derive(line, expected, tok):
    """Métriques et contrôles d'une requête mesurée."""
    out_events = [e for e in line["events"]
                  if e.get("content_chars", 0) + e.get("reasoning_chars", 0) > 0]
    usage = line.get("usage") or {}
    n_usage = usage.get("completion_tokens")
    text = line.get("reasoning", "") + line.get("content", "")
    n_retok = len(tok.encode(text, add_special_tokens=False).ids) if tok else None
    m = {"n_usage": n_usage, "n_retok": n_retok, "events_out": len(out_events),
         "finish_reason": line.get("finish_reason"), "checks": {}}
    if not out_events:
        return m
    t_first, t_last = out_events[0]["t_ns"], out_events[-1]["t_ns"]
    m["ttft_s"] = t_first / 1e9
    m["total_s"] = t_last / 1e9
    gaps = [(b["t_ns"] - a["t_ns"]) / 1e6 for a, b in zip(out_events, out_events[1:])]
    m["gap_p50_ms"] = percentile(gaps, 0.5)
    m["gap_p95_ms"] = percentile(gaps, 0.95)
    bijective = n_usage is not None and len(out_events) in (n_usage, n_usage - 1)
    m["exact_k"] = bijective
    if bijective:
        n_text, k = len(out_events), 1
        m["cum_tokens"] = list(range(1, len(out_events) + 1))
    else:
        first_chars = out_events[0]["content_chars"] + out_events[0]["reasoning_chars"]
        k = len(tok.encode(text[:first_chars], add_special_tokens=False).ids) if tok else 1
        n_text = n_retok if n_retok is not None else (n_usage or 0)
    m["k"] = k
    m["n_text"] = n_text
    if len(out_events) > 1 and t_last > t_first:
        m["delivery_tok_s"] = (n_text - k) / ((t_last - t_first) / 1e9)
    if m.get("cum_tokens"):
        m["t_at_token_s"] = [e["t_ns"] / 1e9 for e in out_events]
    # Contrôles d'entrée : ids, longueur et absence de réutilisation.
    exp = expected.get(f"{line['prompt']}|{line['nonce']}")
    checks = m["checks"]
    if exp:
        checks["prompt_tokens"] = usage.get("prompt_tokens") == exp["prompt_tokens"]
        ingest = line.get("ingest") or {}
        if "prompt_fnv1a64" in ingest:
            checks["ids_fingerprint"] = ingest["prompt_fnv1a64"] == exp["fnv1a64"]
        elif ingest.get("kind") == "offline":
            key = "llmd_offline_fnv1a64" if line["engine"] == "llmd" else "transformers_fnv1a64"
            checks["ids_fingerprint_offline"] = exp[key] == exp["fnv1a64"]
    ingest = line.get("ingest") or {}
    if "reused_prefix_tokens" in ingest:
        checks["no_reuse"] = ingest["reused_prefix_tokens"] == "0"
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        checks["no_reuse"] = cached == 0
    timings = line.get("timings") or {}
    if "cache_n" in timings:
        checks["no_reuse"] = timings["cache_n"] == 0
    if n_usage is not None and n_retok is not None:
        checks["usage_vs_retok_delta"] = n_usage - n_retok
    return m


def load(run_dir):
    run_dir = pathlib.Path(run_dir)
    expected_path = run_dir / "prompts" / "expected.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    tok = tokenizer()
    rows, others = [], []
    for raw in (run_dir / "results.jsonl").read_text().splitlines():
        line = json.loads(raw)
        if line.get("kind") == "measure":
            if line.get("status") == "ok":
                line["m"] = derive(line, expected, tok)
            rows.append(line)
        else:
            others.append(line)
    return rows, others


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def integrity(rows):
    """Sorties greedy du spéculatif comparées à l'AR du même moteur, même requête."""
    by_key = {(r["config"], r["prompt"], r["nonce"], r.get("seed"), r.get("repeat")): r
              for r in rows if r.get("status") == "ok" and r["sampling"] == "greedy"}
    result = {}
    for (config, prompt, nonce, seed, repeat), row in by_key.items():
        base = SPEC_PAIRS.get(config)
        ref = by_key.get((base, prompt, nonce, seed, repeat)) if base else None
        if ref is None:
            continue
        div = first_divergence(row["content"], ref["content"])
        stats = result.setdefault(config, {"pairs": 0, "identical": 0, "first_divergences": []})
        stats["pairs"] += 1
        if div is None:
            stats["identical"] += 1
        else:
            stats["first_divergences"].append({"prompt": prompt, "char": div})
    return result


def repeatability(rows):
    """Greedy : deux requêtes identiques (graines différentes quand le moteur les lit)
    doivent rendre le même texte. Preuve de répétabilité, pas d'argmax."""
    groups = {}
    for row in rows:
        if row.get("status") == "ok" and row["sampling"] == "greedy":
            groups.setdefault((row["config"], row["prompt"], row["nonce"]), []).append(row)
    result = {}
    for (config, prompt, _), group in groups.items():
        if len(group) < 2:
            continue
        stats = result.setdefault(config, {"groups": 0, "identical": 0, "divergent": []})
        stats["groups"] += 1
        if len({row["content"] for row in group}) == 1:
            stats["identical"] += 1
        else:
            stats["divergent"].append(
                {"prompt": prompt, "char": first_divergence(group[0]["content"],
                                                            group[1]["content"])})
    return result


LOG_PATTERNS = ("bench-launch", "bench-env", "MTP actif", "drafter DFlash", "n_ctx", "n_batch",
                "n_ubatch", "flash_attn", "type_k", "type_v", "spec", "draft", "n_max",
                "backend_sampling", "checkpoint", "warn", "error", "fail")


def log_flags(loads):
    """Lignes utiles des journaux de chargement : réglages effectifs lus, pas demandés."""
    flags = {}
    for load in loads:
        path = pathlib.Path(load.get("log", ""))
        if not path.is_file():
            continue
        lines = []
        for line in path.read_text(errors="replace").splitlines():
            lowered = line.lower()
            if any(pattern.lower() in lowered for pattern in LOG_PATTERNS) and line not in lines:
                lines.append(line.strip()[:300])
        flags[f"{load['phase']}|{load['config']}|{load.get('attempt', 0)}"] = lines[:60]
    return flags


def per_prompt(rows, metric):
    """{config: {prompt: [valeurs sur blocs et graines]}}."""
    table = {}
    for row in rows:
        value = (row.get("m") or {}).get(metric)
        if value is None:
            continue
        table.setdefault(row["config"], {}).setdefault(row["prompt"], []).append(value)
    return table


def geo_ratio(a, b, prompts):
    logs = [math.log(statistics.median(a[p]) / statistics.median(b[p])) for p in prompts]
    return math.exp(sum(logs) / len(logs))


def bootstrap_ratio(a, b, rng):
    """Moyenne géométrique des rapports appariés par prompt et IC 95 % hiérarchique."""
    prompts = sorted(set(a) & set(b))
    if not prompts:
        return None
    point = geo_ratio(a, b, prompts)
    if len(prompts) < 3:
        return {"ratio": point, "ci": None, "n_prompts": len(prompts)}
    draws = []
    for _ in range(BOOTSTRAP):
        picked = [rng.choice(prompts) for _ in prompts]
        logs = []
        for p in picked:
            ra = [rng.choice(a[p]) for _ in a[p]]
            rb = [rng.choice(b[p]) for _ in b[p]]
            logs.append(math.log(statistics.median(ra) / statistics.median(rb)))
        draws.append(math.exp(sum(logs) / len(logs)))
    return {"ratio": point, "ci": (percentile(draws, 0.025), percentile(draws, 0.975)),
            "n_prompts": len(prompts)}


def verdict(stat, higher_is_better=True):
    if stat is None:
        return "n/d"
    if stat["ci"] is None:
        return "indicatif (moins de 3 prompts)"
    low, high = stat["ci"]
    if low <= 1 <= high:
        return "indistinct"
    better = (low > 1) == higher_is_better
    notable = abs(stat["ratio"] - 1) >= PRACTICAL
    return ("meilleur" if better else "moins bon") + (" (notable)" if notable else " (< 5 %)")


def fmt(stat):
    if stat is None:
        return "n/d"
    ci = stat["ci"]
    ci_text = f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""
    return f"{stat['ratio']:.2f}×{ci_text}"


def medians(table, config):
    values = [statistics.median(v) for v in table.get(config, {}).values()]
    return statistics.median(values) if values else None


METRICS = [("delivery_tok_s", True, "débit de livraison (tok/s)"),
           ("ttft_s", False, "TTFT (s)"),
           ("latency_L_s", False, "latence jusqu'à L tokens communs (s)"),
           ("total_s", False, "latence totale, longueur naturelle (s)")]


def add_common_length(rows):
    """L = plus courte sortie exacte du prompt dans la phase ; temps d'arrivée du L-ième token.

    Une sortie sans bijection événements/tokens n'a pas d'horodatage par token :
    elle n'entre ni dans L ni dans cette métrique.
    """
    lengths = {}
    for row in rows:
        m = row.get("m") or {}
        if m.get("t_at_token_s"):
            key = (row["phase"], row["prompt"])
            lengths[key] = min(lengths.get(key, 10**9), len(m["t_at_token_s"]))
    for row in rows:
        m = row.get("m") or {}
        common = lengths.get((row["phase"], row["prompt"]))
        if m.get("t_at_token_s") and common:
            m["latency_L_s"] = m["t_at_token_s"][common - 1]
            m["common_length"] = common


def tables(rows, reference):
    rng = random.Random(BOOTSTRAP_SEED)
    add_common_length(rows)
    lines = []
    for phase in sorted({r["phase"] for r in rows}):
        prows = [r for r in rows if r["phase"] == phase]
        configs = sorted({r["config"] for r in prows})
        for metric, higher, label in METRICS:
            table = per_prompt(prows, metric)
            lines.append(f"\n### {phase} — {label}, rapport à {reference}\n")
            lines.append("| config | médiane | rapport [IC 95 %] | verdict |")
            lines.append("|---|---|---|---|")
            for config in configs:
                med = medians(table, config)
                stat = None
                if config != reference and reference in table and config in table:
                    stat = bootstrap_ratio(table[config], table[reference], rng)
                lines.append(f"| {config} | {med if med is None else round(med, 2)} | "
                             f"{fmt(stat)} | {verdict(stat, higher) if stat else '—'} |")
        spec = [c for c in configs if SPEC_PAIRS.get(c) in configs]
        if spec:
            lines.append(f"\n### {phase} — T2, gain spéculatif contre l'AR du même moteur\n")
            lines.append("| config | contre | débit | TTFT | latence |")
            lines.append("|---|---|---|---|---|")
            for config in spec:
                cells = []
                for metric, _, _ in METRICS:
                    table = per_prompt(prows, metric)
                    cells.append(fmt(bootstrap_ratio(table.get(config, {}),
                                                     table.get(SPEC_PAIRS[config], {}), rng)))
                lines.append(f"| {config} | {SPEC_PAIRS[config]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def checks_summary(rows):
    summary = {}
    for row in rows:
        entry = summary.setdefault(row["config"], {"ok": 0, "statuses": {}, "failed_checks": {}})
        status = row.get("status")
        entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
        if status != "ok":
            continue
        entry["ok"] += 1
        for name, value in row["m"]["checks"].items():
            if value is False or (name == "usage_vs_retok_delta" and abs(value) > 1):
                entry["failed_checks"].setdefault(name, []).append(
                    {"prompt": row["prompt"], "value": value})
        if not row["m"].get("exact_k"):
            entry["failed_checks"].setdefault("approx_k", []).append(row["prompt"])
        if (row["m"].get("n_usage") or 0) < 300:
            entry["failed_checks"].setdefault("short_output", []).append(
                {"prompt": row["prompt"], "n": row["m"].get("n_usage")})
    return summary


def _self_check():
    a = {"p1": [2.0, 2.0], "p2": [4.0], "p3": [1.0]}
    b = {"p1": [1.0], "p2": [2.0], "p3": [0.5]}
    stat = bootstrap_ratio(a, b, random.Random(1))
    assert abs(stat["ratio"] - 2.0) < 1e-9 and stat["ci"] == (2.0, 2.0)
    assert verdict(stat) == "meilleur (notable)"
    assert first_divergence("abc", "abd") == 2 and first_divergence("ab", "ab") is None
    line = {"events": [{"t_ns": 1_000_000_000, "content_chars": 1, "reasoning_chars": 0},
                       {"t_ns": 2_000_000_000, "content_chars": 1, "reasoning_chars": 0},
                       {"t_ns": 3_000_000_000, "content_chars": 1, "reasoning_chars": 0}],
            "usage": {"completion_tokens": 4}, "content": "abc", "prompt": "p",
            "nonce": "0", "engine": "llama"}
    m = derive(line, {}, None)
    assert m["exact_k"] and m["k"] == 1 and m["delivery_tok_s"] == 1.0 and m["ttft_s"] == 1.0


if __name__ == "__main__":
    _self_check()
    if len(sys.argv) < 2:
        print("analyze: auto-test ok")
        raise SystemExit(0)
    run_dir = pathlib.Path(sys.argv[1])
    reference = sys.argv[2] if len(sys.argv) > 2 else "llkm-ar"
    rows, others = load(run_dir)
    report = {
        "checks": checks_summary(rows),
        "integrity": integrity(rows),
        "repeatability": repeatability(rows),
        "loads": [o for o in others if o.get("kind") == "load"],
        "log_flags": log_flags([o for o in others if o.get("kind") == "load"]),
    }
    (run_dir / "analysis.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    per_request = [{k: r.get(k) for k in ("phase", "block", "config", "prompt", "seed", "status")}
                   | {k: v for k, v in (r.get("m") or {}).items()
                      if k not in ("cum_tokens", "t_at_token_s")}
                   for r in rows]
    (run_dir / "per_request.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in per_request) + "\n")
    (run_dir / "tables.md").write_text(tables([r for r in rows if r.get("status") == "ok"],
                                              reference))
    print(json.dumps(report["checks"], indent=1, ensure_ascii=False))
    print(json.dumps(report["integrity"], indent=1, ensure_ascii=False))
    print(json.dumps(report["repeatability"], indent=1, ensure_ascii=False))
