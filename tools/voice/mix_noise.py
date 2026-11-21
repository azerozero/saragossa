"""Mélange un clip FLEURS (16 kHz) avec du bruit synthétique déterministe, sortie 48 kHz.
Usage : mix_noise.py <clean.wav> <snr_db> <seed> <out_noisy.wav> <out_clean48.wav> [max_s]"""
import sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, lfilter
src, snr, seed, out_n, out_c = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
max_s = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
x, sr = sf.read(src, dtype="float64"); assert sr == 16000
x = resample_poly(x, 3, 1)[: int(max_s * 48000)]
x = 0.25 * x / (np.abs(x).max() + 1e-9)
rng = np.random.default_rng(seed)
w = rng.standard_normal(len(x))
# bruit rose (filtre de Kellet) + ronflement 100 Hz modulé + babillage bande 300-3k simulé
pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
t = np.arange(len(x)) / 48000
hum = np.sin(2*np.pi*100*t) * (1 + 0.5*np.sin(2*np.pi*0.7*t))
n = pink / pink.std() + 0.3 * hum / hum.std()
n *= np.sqrt((x**2).mean() / (n**2).mean() / 10**(snr/10))
sf.write(out_n, (x + n).astype(np.float32), 48000, subtype="FLOAT")
sf.write(out_c, x.astype(np.float32), 48000, subtype="FLOAT")
