#!/usr/bin/env python3
"""Compare two complete layer dumps on CPU; first divergence is an observation, not a root-cause proof."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_dump import FORMAT, expected_points, key, validate_ids


def load_index(directory):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if index["format"] != FORMAT or index["logits"] != "last-token-after-softcap":
        raise ValueError("unsupported dump format or logits semantics")
    validate_ids(index["prompt_ids"])
    if type(index["num_layers"]) is not int or index["num_layers"] <= 0:
        raise ValueError("invalid layer count")
    entries = {}
    for entry in index["entries"]:
        name = key(entry["layer"], entry["point"])
        shape = entry["shape"]
        if name in entries or entry["dtype"] != "<f4":
            raise ValueError(f"duplicate point or unsupported dtype: {name}")
        if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError(f"invalid shape: {name}")
        rows = 1 if entry["point"] in ("final_norm", "logits") else len(index["prompt_ids"])
        if shape[0] != rows or Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"invalid rows or filename: {name}")
        entries[name] = entry
    if set(entries) != {key(*p) for p in expected_points(index["num_layers"])}:
        raise ValueError("missing or unexpected capture points")
    return index, entries


def read_tensor(directory, entry):
    import numpy as np
    path = Path(directory) / entry["file"]
    rows, cols = entry["shape"]
    if path.stat().st_size != rows * cols * 4:
        raise ValueError(f"truncated or oversized tensor: {path}")
    values = np.fromfile(path, dtype="<f4").reshape(rows, cols).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite tensor: {path}")
    return values


def cosine(a, b):
    import numpy as np
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.clip(np.sum(a * b) / (na * nb), -1, 1))


def metrics(ref, actual, atol, rtol, min_cosine):
    import numpy as np
    if ref.shape != actual.shape:
        raise ValueError(f"shape mismatch: {ref.shape} / {actual.shape}")
    diff = np.abs(actual - ref)
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    row_cosines = [cosine(a, b) for a, b in zip(ref, actual)]
    bad = diff > (atol + rtol * np.abs(ref))
    bad_rows = np.flatnonzero(np.any(bad, axis=1) | (np.asarray(row_cosines) < min_cosine))
    location = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return dict(max_abs=float(diff.max()), relative_linf=float(diff.max()) / scale,
                max_relative=float(np.max(diff / np.maximum(np.abs(ref), max(atol, 1e-30)))),
                cosine=cosine(ref, actual), min_row_cosine=min(row_cosines),
                last_row_max_abs=float(diff[-1].max()),
                outside_tolerance=int(bad.sum()), elements=int(diff.size),
                max_location=[int(i) for i in location],
                first_bad_row=int(bad_rows[0]) if len(bad_rows) else None,
                significant=bool(len(bad_rows)))


def compare(reference, candidate, atol=1e-5, rtol=1e-3, min_cosine=0.99999):
    import numpy as np
    if not np.isfinite([atol, rtol, min_cosine]).all() or atol < 0 or rtol < 0 or not -1 <= min_cosine <= 1:
        raise ValueError("invalid thresholds")
    ref_index, ref_entries = load_index(reference)
    got_index, got_entries = load_index(candidate)
    if ref_index["prompt_ids"] != got_index["prompt_ids"] or ref_index["num_layers"] != got_index["num_layers"]:
        raise ValueError("different prompts or layer counts")
    if ref_index.get("config_sha256") and got_index.get("config_sha256") and ref_index["config_sha256"] != got_index["config_sha256"]:
        raise ValueError("different checkpoint configurations")
    report = dict(reference_backend=ref_index["backend"], candidate_backend=got_index["backend"],
                  thresholds=dict(atol=atol, rtol=rtol, min_cosine=min_cosine), points=[], first_divergence=None)
    router_scores = {}
    for layer, point in expected_points(ref_index["num_layers"]):
        name = key(layer, point)
        ref = read_tensor(reference, ref_entries[name])
        got = read_tensor(candidate, got_entries[name])
        result = dict(name=name, layer=layer, point=point, **metrics(ref, got, atol, rtol, min_cosine))
        if point == "router_logits":
            router_scores[layer] = (ref, got)
        if point == "router_indices":
            if any(not np.equal(x, np.floor(x)).all() or (np.diff(x, axis=1) <= 0).any() for x in (ref, got)):
                raise ValueError(f"expert IDs must be sorted, distinct integers: {name}")
            changed = np.flatnonzero(np.any(ref != got, axis=1))
            result["changed_expert_rows"] = [int(i) for i in changed]
            if len(changed):
                result.update(significant=True, first_bad_row=int(changed[0]))
            scores = router_scores.pop(layer)
            k = ref.shape[1]
            if k > scores[0].shape[1] or any((ids < 0).any() or (ids >= s.shape[1]).any() for ids, s in zip((ref, got), scores)):
                raise ValueError(f"expert IDs outside router range: {name}")
            result["routing"] = []
            for row in changed:
                margins = []
                for s in scores:
                    sorted_scores = np.sort(s[row])
                    margins.append(float(sorted_scores[-k] - sorted_scores[-k - 1]) if k < len(sorted_scores) else None)
                result["routing"].append(dict(row=int(row), reference=ref[row].astype(int).tolist(),
                                              candidate=got[row].astype(int).tolist(), boundary_margins=margins))
        if point == "logits":
            result["greedy"] = [int(ref.argmax()), int(got.argmax())]
        report["points"].append(result)
        if result["significant"] and report["first_divergence"] is None:
            report["first_divergence"] = name
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--min-cosine", type=float, default=0.99999)
    parser.add_argument("--json", type=Path, help="new JSON report file")
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.candidate, args.atol, args.rtol, args.min_cosine)
        print("point                                  max_abs   rel_Linf  max_rel   cosine    min_cos_row last_abs row flags")
        for p in report["points"]:
            print(f"{p['name']:38} {p['max_abs']:9.3g} {p['relative_linf']:9.3g} {p['max_relative']:8.3g} "
                  f"{p['cosine']:.7f} {p['min_row_cosine']:.7f} {p['last_row_max_abs']:8.3g} "
                  f"{str(p['first_bad_row']):>3} {'DIFF' if p['significant'] else 'ok'}")
            for routing in p.get("routing", []):
                print(f"  routing: {routing}")
            if "greedy" in p:
                print(f"  greedy reference/candidate: {p['greedy']}")
        print(f"Thresholds: |diff| > {args.atol} + {args.rtol}*|reference| OR row cosine < {args.min_cosine}; expert IDs exact.")
        print(f"First significant divergence: {report['first_divergence'] or 'none'}")
        layer_first = next((p["name"] for p in report["points"] if p["layer"] is not None and p["significant"]), None)
        print(f"First layer/substep: {layer_first or 'none'} (in canonical dependency order; branches are parallel)")
        print("A propagated input difference is not proof of a fault in the next operation.")
        if args.json:
            with args.json.open("x") as f:
                json.dump(report, f, indent=2, allow_nan=False)
                f.write("\n")
        return 1 if report["first_divergence"] else 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"invalid capture: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
