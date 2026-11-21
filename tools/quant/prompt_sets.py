"""Ensembles de prompts figés de la campagne quant : sélection, validation aveugle.

  python3 -I prompt_sets.py                          # auto-tests stdlib
  python3 -I prompt_sets.py check sets/*.json        # disjonction + SHA-256
  python3 -I prompt_sets.py build-validation --no-robots DIR --aya DIR --out sets/validation.json

La validation vient d'autres provenances que la sélection (no_robots test,
aya_dataset test) ; elle n'est ouverte que sur les finalistes. Deux ensembles
sont disjoints s'ils ne partagent aucune fenêtre de 32 caractères alphanumériques
après normalisation NFKC et casse : cela couvre la copie, pas la paraphrase.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

WINDOW = 32
CATEGORIES = ("code", "prose", "multilingue")


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def windows(text):
    s = normalize(text)
    return {s} if len(s) <= WINDOW else {s[i:i + WINDOW] for i in range(len(s) - WINDOW + 1)}


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = data["prompts"]
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"identifiants dupliqués dans {path}")
    for p in prompts:
        if p["category"] not in CATEGORIES or not p["prompt"].strip():
            raise ValueError(f"prompt invalide : {p['id']}")
    return prompts


def overlaps(a, b):
    """Paires (id_a, id_b) qui partagent une fenêtre normalisée."""
    index = {}
    for p in a:
        for w in windows(p["prompt"]):
            index.setdefault(w, p["id"])
    found = set()
    for p in b:
        for w in windows(p["prompt"]):
            if w in index and index[w] != p["id"]:
                found.add((index[w], p["id"]))
    return sorted(found)


def check(paths):
    sets = {path: load(path) for path in paths}
    names = list(sets)
    problems = []
    for name in names:
        # Doublons internes : même test, en excluant la paire (p, p).
        problems += [(name, x, y) for x, y in overlaps(sets[name], sets[name])]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            problems += [(f"{a}|{b}", x, y) for x, y in overlaps(sets[a], sets[b])]
    if problems:
        raise ValueError(f"ensembles non disjoints : {problems}")
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


def _rows(directory):
    for path in sorted(glob.glob(str(Path(directory) / "*.json")),
                       key=lambda p: int(Path(p).stem.rsplit("_", 1)[1])):
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]:
            yield row["row_idx"], row["row"]


def build_validation(no_robots, aya):
    """Règle déterministe, fixée avant toute lecture des réponses du modèle.

    no_robots : premier tour utilisateur seul (pas de message système), 40-600
    caractères, réponse de référence > 600 caractères ; 5 Coding, puis 2 de
    chacune des catégories Generation, Open QA, Brainstorm, dans l'ordre des lignes.
    aya : 1 prompt par langue (turc, arabe, portugais, chinois, telugu), 30-400
    caractères, réponse de référence > 400 caractères.
    """
    quota = {"Coding": 5, "Generation": 2, "Open QA": 2, "Brainstorm": 2}
    prompts = []
    for idx, row in _rows(no_robots):
        messages = row["messages"]
        if (quota.get(row["category"], 0) == 0 or messages[0]["role"] != "user"
                or len(messages) != 2 or not 40 <= len(messages[0]["content"]) <= 600
                or len(messages[1]["content"]) <= 600):
            continue
        quota[row["category"]] -= 1
        prompts.append(dict(id=f"val-nr-{idx}", category="code" if row["category"] == "Coding" else "prose",
                            source=f"HuggingFaceH4/no_robots test row {idx} ({row['category']})",
                            prompt=messages[0]["content"]))
    languages = ["Turkish", "Standard Arabic", "Portuguese", "Simplified Chinese", "Telugu"]
    for idx, row in _rows(aya):
        if (row["language"] in languages and 30 <= len(row["inputs"]) <= 400
                and len(row["targets"]) > 400):
            languages.remove(row["language"])
            prompts.append(dict(id=f"val-aya-{idx}", category="multilingue",
                                source=f"CohereLabs/aya_dataset test row {idx} ({row['language']})",
                                prompt=row["inputs"]))
    if any(quota.values()) or languages:
        raise ValueError(f"sources insuffisantes : {quota}, {languages}")
    return dict(name="validation", provenance="HuggingFaceH4/no_robots (test) + CohereLabs/aya_dataset (test), "
                "datasets-server, 21 nov. 2026 ; ouverte une seule fois sur les finalistes (E4)",
                prompts=prompts)


def _self_check():
    a = [dict(id="a", category="prose", prompt="Explain how a refrigerator works, in two paragraphs.")]
    b = [dict(id="b", category="prose", prompt="  EXPLAIN how a refrigerator works,\nin two paragraphs!! ")]
    c = [dict(id="c", category="code", prompt="Write a Rust parser for key=value lines.")]
    assert overlaps(a, b) == [("a", "b")] and overlaps(a, c) == []
    assert overlaps(a + c, a + c) == []
    assert windows("ab") == {"ab"} and len(windows("x" * 40)) == 1
    here = Path(__file__).resolve().parent / "sets"
    assert len(load(here / "selection.json")) == 24
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from chat_kl import PROMPTS
    texts = [p["prompt"] for p in load(here / "selection.json")]
    assert all(p in texts for p in PROMPTS)
    if (here / "validation.json").is_file():
        check([here / "selection.json", here / "validation.json"])


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("prompt_sets: auto-tests stdlib ok")
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    c = sub.add_parser("check")
    c.add_argument("paths", nargs="+", type=Path)
    v = sub.add_parser("build-validation")
    v.add_argument("--no-robots", type=Path, required=True)
    v.add_argument("--aya", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps({str(k): v for k, v in check(args.paths).items()}, indent=2))
    else:
        if args.out.exists():
            raise SystemExit(f"destination existante refusée : {args.out}")
        args.out.write_text(json.dumps(build_validation(args.no_robots, args.aya), ensure_ascii=False, indent=1)
                            + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
