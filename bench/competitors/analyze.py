"""Analyse du bench concurrents : admissibilité, métriques par requête, T1/T2/T3.

  VENV_PY -I bench/competitors/analyze.py RUN_DIR [REFERENCE]

Le tokenizer HF du checkpoint (`tokenizers`, venv MLX) compte les tokens de
chaque événement ; sans lui, aucun débit n'est exact.

Règles (protocole v4) :
- Comptage par événement : retokenisation des préfixes cumulés des fragments
  bruts ; nᵢ = |tok(préfixe i)| − |tok(préfixe i−1)|. Le débit est **exact**
  si et seulement si la convention EOS du moteur est fixée
  (`usage_counts_eos` de matrix.json), que chaque nᵢ vaut 1, que le nombre
  d'événements égale N_texte = N_usage − [EOS compté et arrêt « stop »] et
  que la retokenisation du texte entier l'égale aussi. Sinon il est marqué
  « ≈ » et la requête est exclue de la latence à L tokens.
- Admissibilité : statut `ok` et toutes les preuves requises vraies ; une
  preuve fausse donne `validation_failed`, une preuve absente `unverified`.
  Seules les lignes `admissible` entrent dans les tableaux et le gel.
- Unité statistique : le prompt. Les observations sont appariées par clé
  (bloc, graine, répétition), donc par nonce ; IC 95 % par bootstrap
  hiérarchique : blocs (chargements), puis prompts, puis clés du bloc.
"""

import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prompts import CHECKPOINT  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20261104
PRACTICAL = 0.05
# Un second TTFT d'une requête identique sous 80 % du premier signale un cache.
CACHE_PROBE_RATIO = 0.8
SPEC_PAIRS = {
    "sar-mtp": "sar-ar", "sar-dflash": "sar-ar",
    "llq40-mtp": "llq40-ar", "llq40-dflash": "llq40-ar",
    "llkm-mtp": "llkm-ar", "llkm-dflash": "llkm-ar",
    "llmd-native": "llmd-ar", "llmd-native-s": "llmd-ar-s",
}
# Comparaisons de textes entre moteurs, à titre informatif (formats différents).
CROSS_PAIRS = {"mlx-ar": "sar-ar", "llq40-ar": "sar-ar", "llkm-ar": "sar-ar", "llmd-ar": "sar-ar"}
REQUIRED = ("usage", "prompt_tokens", "ids", "no_reuse")


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


def count_tokens(tok, text):
    return len(tok.encode(text, add_special_tokens=False).ids)


def derive(line, tok, eos_counted):
    """Métriques d'une requête `ok` ; `exact` dit si le comptage par événement est prouvé."""
    out_events = [e for e in line["events"]
                  if e.get("content_chars", 0) + e.get("reasoning_chars", 0) > 0]
    usage = line.get("usage") or {}
    n_usage = usage.get("completion_tokens")
    m = {"n_usage": n_usage, "events_out": len(out_events), "exact": False,
         "finish_reason": line.get("finish_reason")}
    if not out_events:
        return m
    t_first, t_last = out_events[0]["t_ns"], out_events[-1]["t_ns"]
    m["ttft_s"] = t_first / 1e9
    m["total_s"] = t_last / 1e9
    gaps = [(b["t_ns"] - a["t_ns"]) / 1e6 for a, b in zip(out_events, out_events[1:])]
    m["gap_p50_ms"] = percentile(gaps, 0.5)
    m["gap_p95_ms"] = percentile(gaps, 0.95)
    counts, n_retok = None, None
    if tok is not None:
        prefix, previous, counts = "", 0, []
        for event in out_events:
            prefix += event.get("text", "")
            current = count_tokens(tok, prefix)
            counts.append(current - previous)
            previous = current
        n_retok = previous
    m["n_retok"] = n_retok
    m["event_tokens_max"] = max(counts) if counts else None
    expected = None
    if eos_counted is not None and n_usage is not None:
        expected = n_usage - (1 if eos_counted and m["finish_reason"] == "stop" else 0)
    m["n_text_expected"] = expected
    exact = (counts is not None and expected is not None and all(c == 1 for c in counts)
             and len(out_events) == expected == n_retok)
    m["exact"] = exact
    if exact:
        k, n_text = 1, len(out_events)
        m["t_at_token_s"] = [e["t_ns"] / 1e9 for e in out_events]
    else:
        k = max(counts[0], 1) if counts else 1
        n_text = n_retok if n_retok is not None else (n_usage or 0)
    m["k"], m["n_text"] = k, n_text
    if len(out_events) > 1 and t_last > t_first and n_text > k:
        m["delivery_tok_s"] = (n_text - k) / ((t_last - t_first) / 1e9)
    # Mémoire : maxima échantillonnés pendant le prefill puis le decode.
    samples = (line.get("memory") or {}).get("samples") or []
    prefill = [v for t, v in samples if t <= t_first]
    decode = [v for t, v in samples if t_first < t <= t_last]
    m["mem_prefill_max"] = max(prefill) if prefill else None
    m["mem_decode_max"] = max(decode) if decode else None
    m["mem_lifetime_max"] = ((line.get("memory") or {}).get("after") or {}).get(
        "lifetime_max_phys_footprint")
    return m


def proofs(line, expected, cache_probe):
    """Preuves d'entrée : True, False, ou None si absente (« non vérifiée »)."""
    usage = line.get("usage") or {}
    ingest = line.get("ingest") or {}
    exp = expected.get(f"{line['prompt']}|{line['nonce']}")
    checks = {"usage": bool(usage.get("completion_tokens") is not None
                            and usage.get("prompt_tokens") is not None) or None}
    checks["prompt_tokens"] = (None if not exp or usage.get("prompt_tokens") is None
                               else usage["prompt_tokens"] == exp["prompt_tokens"])
    if not exp:
        checks["ids"] = None
    elif "prompt_fnv1a64" in ingest:
        checks["ids"] = ingest["prompt_fnv1a64"] == exp["fnv1a64"]
    elif ingest.get("kind") == "offline":
        key = "llmd_offline_fnv1a64" if line["engine"] == "llmd" else "transformers_fnv1a64"
        checks["ids"] = exp[key] == exp["fnv1a64"]
    else:
        checks["ids"] = None
    no_reuse = None
    if "reused_prefix_tokens" in ingest:
        no_reuse = ingest["reused_prefix_tokens"] == "0"
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        no_reuse = cached == 0
    timings = line.get("timings") or {}
    if "cache_n" in timings:
        no_reuse = timings["cache_n"] == 0
    if no_reuse is None:
        # Moteur sans compteur (llmd) : seul le diagnostic de validation prouve.
        no_reuse = cache_probe.get(line["engine"])
    checks["no_reuse"] = no_reuse
    return checks


def cache_probes(rows):
    """Diagnostic de validation : requêtes identiques consécutives d'un même chargement.

    Vrai si chaque second TTFT reste ≥ 80 % du premier, faux sinon ; absent
    si le moteur n'a pas de paire mesurée.
    """
    groups = {}
    for row in rows:
        if row["phase"] == "validate" and row.get("status") == "ok" and row.get("m", {}).get(
                "ttft_s"):
            key = (row["config"], row.get("attempt"), row["prompt"], row["nonce"])
            groups.setdefault(key, []).append(row)
    verdicts, ratios = {}, {}
    for (config, *_), group in groups.items():
        if len(group) < 2:
            continue
        engine = group[0]["engine"]
        ordered = sorted(group, key=lambda r: r["t0_wall"])
        ratio = ordered[1]["m"]["ttft_s"] / ordered[0]["m"]["ttft_s"]
        ratios.setdefault(engine, []).append(round(ratio, 3))
        verdicts[engine] = verdicts.get(engine, True) and ratio >= CACHE_PROBE_RATIO
    return verdicts, ratios


def load_matrix(run_dir):
    frozen = pathlib.Path(run_dir) / "matrix.frozen.json"
    return json.loads((frozen if frozen.exists() else HERE / "matrix.json").read_text())


def load(run_dir):
    """Lignes de mesure non remplacées, avec métriques, preuves et admissibilité."""
    run_dir = pathlib.Path(run_dir)
    matrix = load_matrix(run_dir)
    expected_path = run_dir / "prompts" / "expected.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    manifest = json.loads((run_dir / "prompts" / "prompts.json").read_text())
    kinds = {p["id"]: p["kind"] for p in manifest["prompts"]}
    tok = tokenizer()
    lines = [json.loads(raw) for raw in (run_dir / "results.jsonl").read_text().splitlines()]
    superseded = {(l["entry_key"], l["superseded_run_seq"]) for l in lines
                  if l.get("kind") == "superseded"}
    throttled = {(l["entry_key"], l["run_seq"], l.get("attempt")) for l in lines
                 if l.get("kind") == "unload" and l.get("thermal_after", {}).get("throttled")}
    rows, others = [], []
    for line in lines:
        if (line.get("entry_key"), line.get("run_seq")) in superseded:
            continue
        if line.get("kind") != "measure":
            others.append(line)
            continue
        line["prompt_kind"] = kinds.get(line["prompt"], "?")
        if line.get("status") == "ok":
            eos = matrix["engines"][line["engine"]].get("usage_counts_eos")
            line["m"] = derive(line, tok, eos)
        line["throttled"] = bool(line.get("thermal_before_load")) or (
            line.get("entry_key"), line.get("run_seq"), line.get("attempt")) in throttled
        rows.append(line)
    probes, probe_ratios = cache_probes(rows)
    for row in rows:
        if row.get("status") != "ok":
            row["admissibility"] = f"failed:{row.get('status')}"
            continue
        row["proofs"] = proofs(row, expected, probes)
        values = [row["proofs"][name] for name in REQUIRED]
        row["admissibility"] = ("validation_failed" if False in values
                                else "unverified" if None in values else "admissible")
    divergences = integrity(rows)
    for row in rows:
        if row.get("request_key") in divergences and row["admissibility"] == "admissible":
            row["admissibility"] = "validation_failed"
            row.setdefault("proofs", {})["greedy_identity"] = False
    return rows, others, {"cache_probe": probes, "cache_probe_ratios": probe_ratios}


def obs_key(row):
    return (row["block"], row.get("seed"), row.get("repeat"))


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def integrity(rows, pairs=None, mark=True):
    """Greedy : texte du spéculatif contre l'AR du même moteur, même requête (même nonce).

    Rend les clés des requêtes divergentes (`mark`) ou le rapport de comparaison.
    """
    pairs = pairs or SPEC_PAIRS
    by_key = {}
    for row in rows:
        if row.get("status") == "ok" and row["sampling"] == "greedy":
            key = (row["phase"], row["prompt"], row["nonce"]) + obs_key(row)
            row["request_key"] = (row["config"],) + key
            by_key[row["request_key"]] = row
    divergent, report = set(), {}
    for (config, *key), row in by_key.items():
        base = pairs.get(config)
        ref = by_key.get((base, *key)) if base else None
        stats = report.setdefault(config, {"pairs": 0, "identical": 0, "missing_pair": 0,
                                           "first_divergences": []}) if base else None
        if base is None:
            continue
        if ref is None:
            stats["missing_pair"] += 1
            continue
        stats["pairs"] += 1
        div = first_divergence(row["content"], ref["content"])
        if div is None:
            stats["identical"] += 1
        else:
            stats["first_divergences"].append({"prompt": row["prompt"], "char": div})
            divergent.add(row["request_key"])
    return divergent if mark else report


def repeatability(rows):
    """Greedy, validation : requêtes identiques (graines 1 et 2) → même texte."""
    groups = {}
    for row in rows:
        if row.get("status") == "ok" and row["sampling"] == "greedy" and row["phase"] == "validate":
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
            stats["divergent"].append({"prompt": prompt, "char": first_divergence(
                group[0]["content"], group[1]["content"])})
    return result


def eos_table(rows):
    """Par moteur, N_usage − |retok(texte)| des requêtes arrêtées et tronquées."""
    table = {}
    for row in rows:
        m = row.get("m") or {}
        if m.get("n_usage") is None or m.get("n_retok") is None:
            continue
        entry = table.setdefault(row["engine"], {}).setdefault(m["finish_reason"], {})
        delta = str(m["n_usage"] - m["n_retok"])
        entry[delta] = entry.get(delta, 0) + 1
    return table


def geomean_of_medians(per_prompt):
    logs = [math.log(statistics.median(v)) for v in per_prompt.values()]
    return math.exp(sum(logs) / len(logs))


def observations(rows, metric):
    """{config: {prompt: {clé d'observation: valeur}}} des lignes admissibles."""
    table = {}
    for row in rows:
        value = (row.get("m") or {}).get(metric)
        if row.get("admissibility") != "admissible" or value is None:
            continue
        table.setdefault(row["config"], {}).setdefault(row["prompt"], {})[obs_key(row)] = value
    return table


def paired_bootstrap(a, b, rng):
    """Rapport apparié par prompt (médianes sur les clés communes), moyenne géométrique.

    IC 95 % : tirage des blocs (chargements), puis des prompts, puis des clés de
    chaque bloc tiré. Publie les paires disponibles et manquantes.
    """
    pairs = {p: {k: (a[p][k], b[p][k]) for k in set(a[p]) & set(b[p])}
             for p in set(a) & set(b)}
    pairs = {p: v for p, v in pairs.items() if v}
    missing = sum(len(set(a.get(p, {})) ^ set(b.get(p, {}))) for p in set(a) | set(b))
    if not pairs:
        return None

    def ratio(selection):
        return statistics.median(x for x, _ in selection) / statistics.median(
            y for _, y in selection)

    prompts = sorted(pairs)
    point = math.exp(statistics.fmean(math.log(ratio(list(pairs[p].values())))
                                      for p in prompts))
    result = {"ratio": point, "ci": None, "n_prompts": len(prompts),
              "n_pairs": sum(len(v) for v in pairs.values()), "n_missing": missing}
    if len(prompts) < 3:
        return result
    blocks = sorted({k[0] for v in pairs.values() for k in v})
    draws = []
    for _ in range(BOOTSTRAP):
        drawn_blocks = [rng.choice(blocks) for _ in blocks]
        logs = []
        for p in (rng.choice(prompts) for _ in prompts):
            selection = []
            for block in drawn_blocks:
                keys = [k for k in pairs[p] if k[0] == block]
                selection += [pairs[p][rng.choice(keys)] for _ in keys]
            if selection:
                logs.append(math.log(ratio(selection)))
        if logs:
            draws.append(math.exp(statistics.fmean(logs)))
    result["ci"] = (percentile(draws, 0.025), percentile(draws, 0.975))
    return result


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


def fmt(stat, approx=False):
    if stat is None:
        return "n/d"
    ci = stat["ci"]
    ci_text = f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""
    mark = "≈" if approx else ""
    return f"{mark}{stat['ratio']:.2f}×{ci_text} ({stat['n_pairs']} paires, {stat['n_missing']} manquantes)"


# (métrique, plus grand = meilleur, libellé, comparée par rapport)
METRICS = [("delivery_tok_s", True, "débit de livraison (tok/s)", True),
           ("ttft_s", False, "TTFT (s)", True),
           ("latency_L_s", False, "latence jusqu'à L tokens communs (s)", True),
           ("total_s", False, "latence totale, longueur naturelle (s) — descriptive", False)]


def add_common_length(rows):
    """L = plus courte sortie exacte du prompt dans le groupe ; heure du L-ième token."""
    lengths = {}
    for row in rows:
        m = row.get("m") or {}
        if row.get("admissibility") == "admissible" and m.get("t_at_token_s"):
            key = (row["phase"], row["prompt"])
            lengths[key] = min(lengths.get(key, 10**9), len(m["t_at_token_s"]))
    for row in rows:
        m = row.get("m") or {}
        common = lengths.get((row["phase"], row["prompt"]))
        if m.get("t_at_token_s") and common:
            m["latency_L_s"] = m["t_at_token_s"][common - 1]
            m["common_length"] = common


def approx_configs(rows):
    return {row["config"] for row in rows if row.get("admissibility") == "admissible"
            and not (row.get("m") or {}).get("exact")}


def tables(rows, reference):
    rng = random.Random(BOOTSTRAP_SEED)
    add_common_length(rows)
    approx = approx_configs(rows)
    lines = []
    for group in sorted({(r["phase"], r["prompt_kind"]) for r in rows}):
        grows = [r for r in rows if (r["phase"], r["prompt_kind"]) == group]
        configs = sorted({r["config"] for r in grows})
        label = f"{group[0]} / {group[1]}"
        for metric, higher, title, compared in METRICS:
            table = observations(grows, metric)
            lines.append(f"\n### {label} — {title}\n")
            if not compared:
                lines.append("| config | médiane | N médian | requêtes |")
                lines.append("|---|---|---|---|")
                n_table = observations(grows, "n_text")
                for config in configs:
                    values = [v for p in table.get(config, {}).values() for v in p.values()]
                    ns = [v for p in n_table.get(config, {}).values() for v in p.values()]
                    if values:
                        lines.append(f"| {config} | {statistics.median(values):.2f} | "
                                     f"{statistics.median(ns):.0f} | {len(values)} |")
                continue
            lines.append(f"| config | médiane | rapport à {reference} [IC 95 %] | verdict |")
            lines.append("|---|---|---|---|")
            for config in configs:
                values = [v for p in table.get(config, {}).values() for v in p.values()]
                if not values:
                    lines.append(f"| {config} | n/d | n/d | n/d |")
                    continue
                is_approx = metric == "delivery_tok_s" and (config in approx or reference in approx)
                stat = None
                if config != reference and reference in table:
                    stat = paired_bootstrap(table[config], table[reference], rng)
                mark = "≈" if is_approx else ""
                lines.append(f"| {config} | {mark}{statistics.median(values):.2f} | "
                             f"{fmt(stat, is_approx) if stat else '—'} | "
                             f"{verdict(stat, higher) if stat else '—'} |")
        spec = [c for c in configs if SPEC_PAIRS.get(c) in configs]
        if spec:
            compared = [m for m in METRICS if m[3]]
            lines.append(f"\n### {label} — T2, gain spéculatif contre l'AR du même moteur\n")
            lines.append("| config | contre | " + " | ".join(m[2] for m in compared) + " |")
            lines.append("|---|---|" + "---|" * len(compared))
            for config in spec:
                cells = []
                for metric, _, _, _ in compared:
                    table = observations(grows, metric)
                    is_approx = metric == "delivery_tok_s" and (
                        config in approx or SPEC_PAIRS[config] in approx)
                    cells.append(fmt(paired_bootstrap(table.get(config, {}),
                                                      table.get(SPEC_PAIRS[config], {}), rng),
                                     is_approx))
                lines.append(f"| {config} | {SPEC_PAIRS[config]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def drift(rows):
    """Pente de log(débit) en fonction du rang dans le bloc et de l'heure, par config."""
    result = {}
    for config in sorted({r["config"] for r in rows}):
        points = [(r["position"], r["t0_wall"], math.log(r["m"]["delivery_tok_s"]))
                  for r in rows if r["config"] == config and r.get("admissibility") == "admissible"
                  and r["m"].get("delivery_tok_s")]
        if len(points) < 3:
            continue
        entry = {}
        for name, index in (("rank", 0), ("hour", 1)):
            xs = [p[index] / (3600 if name == "hour" else 1) for p in points]
            ys = [p[2] for p in points]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            var = sum((x - mx) ** 2 for x in xs)
            if var == 0:
                continue
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
            span = max(xs) - min(xs)
            change = math.exp(slope * span) - 1
            entry[name] = {"slope_log_per_unit": slope, "change_over_span": change,
                           "flag": abs(change) > PRACTICAL}
        result[config] = entry
    return result


def memory_summary(rows):
    summary = {}
    for row in rows:
        m = row.get("m") or {}
        if row.get("admissibility") != "admissible":
            continue
        entry = summary.setdefault(row["config"], {"prefill": [], "decode": [], "lifetime": []})
        for key, name in (("mem_prefill_max", "prefill"), ("mem_decode_max", "decode"),
                          ("mem_lifetime_max", "lifetime")):
            if m.get(key):
                entry[name].append(m[key])
    return {config: {name: (max(values) / 1e9 if values else None) for name, values in e.items()}
            for config, e in summary.items()}


def admissibility_summary(rows):
    summary = {}
    for row in rows:
        entry = summary.setdefault(row["config"], {"total": 0, "by_status": {}, "failed_proofs": {},
                                                   "approx": 0, "throttled": 0, "short": 0})
        entry["total"] += 1
        entry["by_status"][row["admissibility"]] = entry["by_status"].get(row["admissibility"], 0) + 1
        for name, value in (row.get("proofs") or {}).items():
            if value is not True:
                entry["failed_proofs"].setdefault(f"{name}={value}", []).append(row["prompt"])
        m = row.get("m") or {}
        if row.get("status") == "ok":
            entry["approx"] += not m.get("exact")
            entry["short"] += (m.get("n_usage") or 0) < 300
        entry["throttled"] += row.get("throttled", False)
    return summary


LOG_PATTERNS = ("bench-launch", "bench-env", "MTP actif", "drafter DFlash", "n_ctx", "n_batch",
                "n_ubatch", "flash_attn", "type_k", "type_v", "spec", "draft", "n_max",
                "backend_sampling", "checkpoint", "warn", "error", "fail")


def log_flags(loads):
    """Lignes utiles des journaux de chargement : réglages effectifs lus, pas demandés."""
    flags = {}
    for load_line in loads:
        path = pathlib.Path(load_line.get("log", "")).expanduser()
        if not path.is_file():
            continue
        lines = []
        for line in path.read_text(errors="replace").splitlines():
            lowered = line.lower()
            if any(pattern.lower() in lowered for pattern in LOG_PATTERNS) and line not in lines:
                lines.append(line.strip()[:300])
        flags[f"{load_line['entry_key']}|r{load_line['run_seq']}|a{load_line['attempt']}"] = lines[:60]
    return flags


def _self_check():
    class Tok:
        """Un caractère = un token, sauf « ab » qui fusionne."""

        def encode(self, text, add_special_tokens=False):
            class Out:
                ids = list(text.replace("ab", "\0"))
            return Out()

    def line(texts, n_usage, finish="stop"):
        events = [{"t_ns": (i + 1) * 10**9, "content_chars": len(t), "reasoning_chars": 0,
                   "text": t} for i, t in enumerate(texts)]
        return {"events": events, "usage": {"completion_tokens": n_usage},
                "finish_reason": finish}

    tok = Tok()
    m = derive(line(["x", "y", "z"], 4), tok, True)
    assert m["exact"] and m["k"] == 1 and m["delivery_tok_s"] == 1.0
    # Deux événements pour trois tokens (« y », puis « zw ») : jamais exact.
    m = derive(line(["y", "zw"], 3), tok, False)
    assert not m["exact"] and m["event_tokens_max"] == 2 and "t_at_token_s" not in m
    # Convention EOS inconnue : jamais exact.
    assert not derive(line(["x", "y"], 2), tok, None)["exact"]
    # Fusion « a » + « b » en un token : le comptage cumulé le voit (0 au 2e).
    assert not derive(line(["a", "b", "c"], 2), tok, False)["exact"]
    a = {"p1": {(0, None, 0): 2.0, (1, None, 0): 2.0}, "p2": {(0, None, 0): 4.0},
         "p3": {(0, None, 0): 1.0}}
    b = {"p1": {(0, None, 0): 1.0, (1, None, 0): 1.0}, "p2": {(0, None, 0): 2.0},
         "p3": {(0, None, 0): 0.5, (1, None, 0): 0.5}}
    stat = paired_bootstrap(a, b, random.Random(1))
    assert abs(stat["ratio"] - 2.0) < 1e-9 and stat["ci"] == (2.0, 2.0)
    assert stat["n_pairs"] == 4 and stat["n_missing"] == 1
    assert verdict(stat) == "meilleur (notable)"
    assert first_divergence("abc", "abd") == 2 and first_divergence("ab", "ab") is None
    row = {"prompt": "p", "nonce": "n", "engine": "llama", "usage": {
        "completion_tokens": 3, "prompt_tokens": 10}, "timings": {"cache_n": 4},
        "ingest": {"prompt_fnv1a64": "aa"}}
    checks = proofs(row, {"p|n": {"prompt_tokens": 10, "fnv1a64": "aa"}}, {})
    assert checks == {"usage": True, "prompt_tokens": True, "ids": True, "no_reuse": False}
    row = {"prompt": "p", "nonce": "n", "engine": "llmd", "usage": {
        "completion_tokens": 3, "prompt_tokens": 10}, "ingest": {"kind": "offline"}}
    exp = {"p|n": {"prompt_tokens": 10, "fnv1a64": "aa", "llmd_offline_fnv1a64": "aa"}}
    assert proofs(row, exp, {})["no_reuse"] is None
    assert proofs(row, exp, {"llmd": True})["no_reuse"] is True


if __name__ == "__main__":
    _self_check()
    if len(sys.argv) < 2:
        print("analyze: auto-test ok")
        raise SystemExit(0)
    run_dir = pathlib.Path(sys.argv[1])
    reference = sys.argv[2] if len(sys.argv) > 2 else "llkm-ar"
    rows, others, probes = load(run_dir)
    loads = [o for o in others if o.get("kind") == "load"]
    report = {
        "admissibility": admissibility_summary(rows),
        **probes,
        "eos": eos_table(rows),
        "integrity": integrity(rows, mark=False),
        "cross_engine_text_identity_informative": integrity(rows, CROSS_PAIRS, mark=False),
        "repeatability": repeatability(rows),
        "drift": drift(rows),
        "memory_gb": memory_summary(rows),
        "argmax_checks": [o for o in others if o.get("kind") == "argmax_check"],
        "loads": loads,
        "log_flags": log_flags(loads),
    }
    (run_dir / "analysis.json").write_text(json.dumps(report, indent=1, ensure_ascii=False,
                                                      default=str))
    keep = ("phase", "block", "config", "prompt", "prompt_kind", "seed", "repeat", "status",
            "admissibility", "throttled", "proofs", "request_id")
    per_request = [{k: r.get(k) for k in keep}
                   | {k: v for k, v in (r.get("m") or {}).items() if k != "t_at_token_s"}
                   for r in rows]
    (run_dir / "per_request.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in per_request) + "\n")
    (run_dir / "tables.md").write_text(tables(rows, reference))
    print(json.dumps({k: report[k] for k in ("admissibility", "cache_probe", "eos",
                                             "integrity", "repeatability")},
                     indent=1, ensure_ascii=False))
