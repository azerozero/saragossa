"""Transcrit des WAV avec mlx_whisper (large-v3-turbo local, hors ligne) et calcule le WER.
Usage : wer_eval.py <jobs.json> <out.json> ; jobs = [{"id","wav","lang","text"}]"""
import json, os, re, sys, unicodedata
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
import mlx_whisper
REPO = "mlx-community/whisper-large-v3-turbo"
def norm(t):
    t = unicodedata.normalize("NFKC", t.lower())
    t = re.sub(r"[^\w\s']", " ", t)
    return t.split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)
jobs = json.load(open(sys.argv[1])); res = []
for j in jobs:
    x, sr = sf.read(j["wav"], dtype="float32")
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    hyp = mlx_whisper.transcribe(x, path_or_hf_repo=REPO, language=j["lang"], temperature=0.0,
                                 condition_on_previous_text=False)["text"]
    e, n = wer(j["text"], hyp)
    res.append({**j, "hyp": hyp, "err": e, "n": n})
json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
