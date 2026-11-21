"""Audit des octets sérialisés d'un checkpoint MLX : bpw langage et total.

  python3 -I audit_bytes.py MODELE [--limit 4.2] [--params 26895998464]
  python3 -I audit_bytes.py                      # auto-tests stdlib

Lit seulement les en-têtes safetensors. Langage = tenseurs `language_model.*`
(codes, échelles, biais et tenseurs fixes) ; le reste (vision, MTP) entre dans
le total publié à côté. Le dénominateur est le nombre de paramètres langage du
BF16 (Qwen3.8-27B par défaut). Refus (code 1) si le bpw langage dépasse la
limite ou s'écarte de `allocation.json.actual_bpw` quand il existe.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

QWEN38_27B_LANGUAGE_PARAMS = 26_895_998_464


def header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))


def audit(model, params=QWEN38_27B_LANGUAGE_PARAMS, limit=None):
    language = other = 0
    files = sorted(Path(model).glob("*.safetensors"))
    if not files:
        raise ValueError(f"aucun safetensors dans {model}")
    for path in files:
        for key, entry in header(path).items():
            if key == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            if key.startswith("language_model."):
                language += end - start
            else:
                other += end - start
    report = dict(model=str(model), language_bytes=language, other_bytes=other,
                  language_bpw=language * 8 / params, total_bpw=(language + other) * 8 / params,
                  params=params, files=[p.name for p in files])
    allocation = Path(model) / "allocation.json"
    if allocation.is_file():
        planned = json.loads(allocation.read_text())["actual_bpw"]
        report["allocation_bpw"] = planned
        if abs(planned - report["language_bpw"]) > 1e-6:
            report["refused"] = "écart avec allocation.json"
    if limit is not None and report["language_bpw"] > limit:
        report["refused"] = f"au-dessus de {limit} bpw"
    return report


def _self_check():
    import tempfile

    def write(path, tensors):
        head = {k: dict(dtype="U8", shape=[n], data_offsets=[0, n]) for k, n in tensors.items()}
        raw = json.dumps(head).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write(root / "model-1.safetensors", {"language_model.a": 50, "vision_tower.b": 10, "__metadata__": 0})
        write(root / "mtp.safetensors", {"fc.weight": 5})
        out = audit(root, params=100, limit=4.2)
        assert out["language_bpw"] == 4.0 and out["total_bpw"] == 5.2 and "refused" not in out
        assert "refused" in audit(root, params=100, limit=3.9)
        (root / "allocation.json").write_text(json.dumps(dict(actual_bpw=4.1)))
        assert audit(root, params=100)["refused"] == "écart avec allocation.json"


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("audit_bytes: auto-tests stdlib ok")
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--params", type=int, default=QWEN38_27B_LANGUAGE_PARAMS)
    args = parser.parse_args()
    report = audit(args.model, args.params, args.limit)
    print(json.dumps(report, indent=1))
    return 1 if "refused" in report else 0


if __name__ == "__main__":
    sys.exit(main())
