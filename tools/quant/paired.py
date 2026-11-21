"""Comparaisons appariées par prompt des sorties `chat_kl.py eval --json`.

  python3 -I paired.py diff CANDIDAT.json REFERENCE.json [--metric kl_mean]
  python3 -I paired.py factorial A0F0.json A0F1.json A1F0.json A1F1.json
  python3 -I paired.py                                   # auto-tests stdlib

Unité statistique : le prompt (positions successives non indépendantes).
IC 95 % par bootstrap percentile sur les prompts, 10 000 tirages, graine 0.
`diff` retient un gain si l'IC de (candidat − référence) est entièrement sous 0
et si aucune catégorie ne régresse de plus de 10 % en moyenne relative.
`factorial` (plan 2×2, facteurs A et F, niveaux 0/1) donne les effets
principaux et l'interaction, contrastes par prompt, avec le même bootstrap.
"""
import argparse
import json
from pathlib import Path
import random
import sys

DRAWS, SEED, MAX_REGRESSION = 10_000, 0, 0.10


def load(path, metric):
    data = json.loads(Path(path).read_text())
    return {k: v[metric] for k, v in data["prompts"].items()}, {k: v["category"] for k, v in data["prompts"].items()}, \
        data.get("ref_sha256")


def bootstrap_ci(values, draws=DRAWS, seed=SEED):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def aligned(paths, metric):
    loaded = [load(p, metric) for p in paths]
    keys = set(loaded[0][0])
    if any(set(v) != keys for v, _, _ in loaded) or len({sha for _, _, sha in loaded}) != 1:
        raise ValueError("comparaison refusée : prompts ou référence différents")
    order = sorted(keys)
    return [[v[k] for k in order] for v, _, _ in loaded], [loaded[0][1][k] for k in order], order


def diff(candidate, reference, metric="kl_mean"):
    (c, r), categories, order = aligned([candidate, reference], metric)
    d = [a - b for a, b in zip(c, r)]
    low, high = bootstrap_ci(d)
    by_category = {}
    for cat in sorted(set(categories)):
        idx = [i for i, x in enumerate(categories) if x == cat]
        mc, mr = sum(c[i] for i in idx) / len(idx), sum(r[i] for i in idx) / len(idx)
        by_category[cat] = dict(candidate=mc, reference=mr, relative=(mc - mr) / mr if mr else 0.0)
    regress = [cat for cat, v in by_category.items() if v["relative"] > MAX_REGRESSION]
    return dict(metric=metric, prompts=len(d), candidate=sum(c) / len(c), reference=sum(r) / len(r),
                mean_diff=sum(d) / len(d), ci95=[low, high], categories=by_category,
                regressed=regress, gain=high < 0 and not regress,
                worst_prompt=max(zip(d, order))[1])


def factorial(a0f0, a0f1, a1f0, a1f1, metric="kl_mean"):
    (y00, y01, y10, y11), _, _ = aligned([a0f0, a0f1, a1f0, a1f1], metric)
    contrasts = {
        "A": [(d + c - b - a) / 2 for a, b, c, d in zip(y00, y01, y10, y11)],
        "F": [(d + b - c - a) / 2 for a, b, c, d in zip(y00, y01, y10, y11)],
        "AxF": [(d - c - b + a) / 2 for a, b, c, d in zip(y00, y01, y10, y11)],
    }
    result = {}
    for name, values in contrasts.items():
        low, high = bootstrap_ci(values)
        result[name] = dict(effect=sum(values) / len(values), ci95=[low, high], excludes_zero=low > 0 or high < 0)
    result["cells"] = {name: sum(y) / len(y) for name, y in zip(("A0F0", "A0F1", "A1F0", "A1F1"),
                                                                 (y00, y01, y10, y11))}
    return result


def _self_check():
    import tempfile
    assert bootstrap_ci([1.0] * 5) == (1.0, 1.0)
    low, high = bootstrap_ci([-1.0, -2.0, -1.5, -0.5])
    assert -2.0 <= low <= high < 0
    with tempfile.TemporaryDirectory() as tmp:
        def write(name, values, sha="x"):
            path = Path(tmp) / name
            path.write_text(json.dumps(dict(ref_sha256=sha, prompts={
                f"p{i}": dict(kl_mean=v, category="code" if i % 2 else "prose") for i, v in enumerate(values)})))
            return path
        ref = write("r.json", [0.02, 0.03, 0.02, 0.04])
        better = write("b.json", [0.01, 0.02, 0.015, 0.03])
        out = diff(better, ref)
        assert out["gain"] and out["ci95"][1] < 0 and out["regressed"] == []
        mixed = write("m.json", [0.01, 0.05, 0.01, 0.06])  # code +60 % : refus
        out = diff(mixed, ref)
        assert not out["gain"] and out["regressed"] == ["code"]
        try:
            diff(better, write("o.json", [0.02, 0.03, 0.02, 0.04], sha="y"))
            raise AssertionError("références différentes acceptées")
        except ValueError:
            pass
        # Effet A pur de −0,01, aucun effet F ni interaction.
        y = [0.02, 0.03, 0.025, 0.04]
        cells = [write(n, [v + da for v in y]) for n, da in (("00", 0), ("01", 0), ("10", -0.01), ("11", -0.01))]
        out = factorial(*cells)
        assert abs(out["A"]["effect"] + 0.01) < 1e-12 and out["A"]["excludes_zero"]
        assert abs(out["F"]["effect"]) < 1e-12 and abs(out["AxF"]["effect"]) < 1e-12


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("paired: auto-tests stdlib ok")
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    d = sub.add_parser("diff")
    d.add_argument("candidate")
    d.add_argument("reference")
    f = sub.add_parser("factorial")
    f.add_argument("cells", nargs=4, metavar="A?F?.json")
    for p in (d, f):
        p.add_argument("--metric", default="kl_mean")
    args = parser.parse_args()
    out = (diff(args.candidate, args.reference, args.metric) if args.action == "diff"
           else factorial(*args.cells, metric=args.metric))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
