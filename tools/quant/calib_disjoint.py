"""Disjonction du corpus de calibration et des prompts réservés : SHA exact, préfixe et n-grammes de 64 tokens.

  python3 -I -B calib_disjoint.py tools/quant DL_CALIB TOKENIZER_JSON > disjoint.json

DL_CALIB contient les pages no_robots/ et aya/ lues par calib_corpus.py. Exige
le paquet tokenizers. Chaque compteur doit valoir 0 ; un prompt touché est
cité par son SHA-256, jamais par son texte.
"""
import hashlib, json, sys, unicodedata
from pathlib import Path
from tokenizers import Tokenizer
sys.path.insert(0, sys.argv[1])
from calib_corpus import no_robots_text, aya_text, NR_ROWS
from prompt_sets import _rows
from imatrix import reserved_prompts
dl, tok = Path(sys.argv[2]), Tokenizer.from_file(sys.argv[3])
docs = [no_robots_text(r) for _, r in _rows(dl / "no_robots")][:NR_ROWS] + [aya_text(r) for _, r in _rows(dl / "aya")]
# Unités comparées : chaque document et chacun de ses messages (question, réponse).
units = set()
for _, r in _rows(dl / "no_robots"):
    units.update(m["content"] for m in r["messages"])
for _, r in _rows(dl / "aya"):
    units.update((r["inputs"], r["targets"]))
units.update(docs)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
unit_sha = {sha(u) for u in units} | {sha(u.strip()) for u in units}
N = 64
grams, prefixes, full = set(), set(), []
for d in docs:
    ids = tok.encode(d, add_special_tokens=False).ids
    full.append(ids)
    prefixes.add(tuple(ids[:N]))
    for i in range(len(ids) - N + 1):
        grams.add(hash(tuple(ids[i:i + N])))
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKC", s).casefold() if c.isalnum())
corpus_norm = norm("\n".join(docs))
prompts = reserved_prompts()
res = dict(prompts=len(prompts), documents=len(docs), units=len(units), ngram=N, grams=len(grams),
           exact_sha=0, prefix=0, ngram_hits=0, short_contained=0, normalized_contained=0, details=[])
for p in prompts:
    ids = tok.encode(p, add_special_tokens=False).ids
    hit = []
    if sha(p) in unit_sha or sha(p.strip()) in unit_sha: hit.append("exact_sha")
    if len(ids) >= N and tuple(ids[:N]) in prefixes: hit.append("prefix")
    if len(ids) >= N and any(hash(tuple(ids[i:i + N])) in grams for i in range(len(ids) - N + 1)): hit.append("ngram_hits")
    if len(ids) < N:  # prompt plus court que 64 tokens : recherche de la séquence entière
        t = tuple(ids)
        if any(any(tuple(d[i:i + len(t)]) == t for i in range(len(d) - len(t) + 1)) for d in full): hit.append("short_contained")
    if norm(p) in corpus_norm: hit.append("normalized_contained")
    for h in hit: res[h] += 1
    if hit: res["details"].append(dict(prompt_sha256=sha(p), hits=hit))
print(json.dumps(res, indent=1))
