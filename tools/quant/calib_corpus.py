"""Corpus de calibration imatrix publiable, règle déterministe.

  python3 -I -B calib_corpus.py --no-robots DIR --aya DIR --out corpus.txt

Entrées : pages JSON de datasets-server (`/rows`, 100 lignes par page) des
splits **train** de HuggingFaceH4/no_robots (pages aux offsets 0, 100, …) et
CohereLabs/aya_dataset (offsets 0, 10 000, …, 190 000). Les ensembles de
sélection et de validation viennent des splits test : `imatrix.py` refuse
de toute façon un corpus qui en contient un prompt.

Règle : lignes dans l'ordre des offsets ; une ligne no_robots = ses messages
séparés par une ligne vide ; une ligne aya = question puis réponse. Les deux
flux sont entrelacés à raison de NR_PER_AYA lignes no_robots par ligne aya
jusqu'à épuisement d'aya, puis no_robots est tronqué à NR_ROWS lignes. Sans
argument : auto-test stdlib.
"""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_sets import _rows  # noqa: E402

NR_ROWS = 1200
NR_PER_AYA = 1


def no_robots_text(row):
    return "\n\n".join(m["content"] for m in row["messages"])


def aya_text(row):
    return row["inputs"] + "\n\n" + row["targets"]


def build(nr_rows, aya_rows):
    nr = [no_robots_text(r) for _, r in nr_rows][:NR_ROWS]
    aya = [aya_text(r) for _, r in aya_rows]
    parts = []
    while nr or aya:
        parts += nr[:NR_PER_AYA]
        del nr[:NR_PER_AYA]
        if aya:
            parts.append(aya.pop(0))
    return "\n\n\n".join(parts) + "\n"


def _self_check():
    nr = [(i, dict(messages=[dict(content=f"q{i}"), dict(content=f"r{i}")])) for i in range(3)]
    aya = [(0, dict(inputs="a", targets="b"))]
    assert build(nr, aya) == "q0\n\nr0\n\n\na\n\nb\n\n\nq1\n\nr1\n\n\nq2\n\nr2\n"


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("calib_corpus: auto-test stdlib ok")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-robots", type=Path, required=True)
    parser.add_argument("--aya", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"destination existante refusée : {args.out}")
    text = build(list(_rows(args.no_robots)), list(_rows(args.aya)))
    from imatrix import reject_eval_text
    reject_eval_text(text)
    args.out.write_text(text, encoding="utf-8")
    print(args.out, len(text), "caractères, sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
