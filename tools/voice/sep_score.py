"""SI-SDR de séparation avec meilleure permutation, et SI-SDRi contre le mélange.
Usage : sep_score.py a.wav b.wav mix.wav est0.wav est1.wav"""
import sys, numpy as np, soundfile as sf
def sisdr(s, e):
    n = min(len(s), len(e)); s, e = s[:n] - s[:n].mean(), e[:n] - e[:n].mean()
    a = np.sum(e * s) / (np.sum(s * s) + 1e-12) * s
    return 10 * np.log10(np.sum(a * a) / (np.sum((e - a) ** 2) + 1e-12))
r = lambda f: sf.read(f, dtype="float64")[0]
a, b, mix, e0, e1 = map(r, sys.argv[1:6])
p0 = (sisdr(a, e0) + sisdr(b, e1)) / 2; p1 = (sisdr(a, e1) + sisdr(b, e0)) / 2
perm = 0 if p0 >= p1 else 1
sa, sb = (sisdr(a, e0), sisdr(b, e1)) if perm == 0 else (sisdr(a, e1), sisdr(b, e0))
base = (sisdr(a, mix) + sisdr(b, mix)) / 2
print(f"perm {perm} SI-SDR a {sa:.2f} b {sb:.2f} moyenne {(sa+sb)/2:.2f} mélange {base:.2f} SI-SDRi {(sa+sb)/2-base:.2f}")
