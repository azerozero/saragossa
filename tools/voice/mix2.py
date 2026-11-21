"""Superpose deux clips FLEURS 16 kHz à 0 dB (même RMS), tronqués au plus court.
Usage : mix2.py a.wav b.wav out_prefix"""
import sys, numpy as np, soundfile as sf
a, _ = sf.read(sys.argv[1], dtype="float64"); b, _ = sf.read(sys.argv[2], dtype="float64")
n = min(len(a), len(b)); a, b = a[:n], b[:n]
a /= np.sqrt((a**2).mean()); b /= np.sqrt((b**2).mean())
a *= 0.05; b *= 0.05
p = sys.argv[3]
sf.write(p + "_mix.wav", (a + b).astype(np.float32), 16000, subtype="FLOAT")
sf.write(p + "_a.wav", a.astype(np.float32), 16000, subtype="FLOAT")
sf.write(p + "_b.wav", b.astype(np.float32), 16000, subtype="FLOAT")
