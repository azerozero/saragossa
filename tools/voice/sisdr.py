"""SI-SDR (dB) de chaque estimation contre la référence. Usage : sisdr.py ref.wav est1.wav [...]"""
import sys, numpy as np, soundfile as sf
def sisdr(s, e):
    n = min(len(s), len(e)); s, e = s[:n] - s[:n].mean(), e[:n] - e[:n].mean()
    a = np.sum(e * s) / (np.sum(s * s) + 1e-12) * s
    return 10 * np.log10(np.sum(a * a) / (np.sum((e - a) ** 2) + 1e-12))
ref = sf.read(sys.argv[1], dtype="float64")[0]
for f in sys.argv[2:]:
    print(f"{f}: {sisdr(ref, sf.read(f, dtype='float64')[0]):.2f} dB")
