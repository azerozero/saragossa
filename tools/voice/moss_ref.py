"""Référence MossFormer2_SS_16K : code officiel ClearerVoice (copié dans shim_moss), poids
chargés en weights_only, décodage en une passe + renormalisation RMS (decode.py).
Usage : moss_ref.py <shim_dir> <ckpt> <device cpu|mps> <mix.wav> <out_prefix> [repeats]"""
import sys, time, types
import numpy as np, soundfile as sf, torch
shim, ckpt, dev, wav, prefix = sys.argv[1:6]
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
sys.path.insert(0, shim)
from mossformer2_ss.mossformer2 import MossFormer2_SS_16K
args = types.SimpleNamespace(encoder_embedding_dim=512, mossformer_sequence_dim=512,
                             num_mossformer_layer=24, encoder_kernel_size=16, num_spks=2)
model = MossFormer2_SS_16K(args).model
sd = torch.load(ckpt, weights_only=True, map_location="cpu")["model"]
print(model.load_state_dict(sd, strict=True))
model.eval().to(dev)
x, sr = sf.read(wav, dtype="float32"); assert sr == 16000 and x.ndim == 1
inp = torch.from_numpy(x)[None].to(dev)
times = []
with torch.no_grad():
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = model(inp)
        outs = [o[0].float().cpu().numpy() for o in outs]
        times.append(time.perf_counter() - t0)
rms_in = np.sqrt((x ** 2).mean())
for i, o in enumerate(outs):
    o = o / np.sqrt((o ** 2).mean()) * rms_in
    sf.write(f"{prefix}_s{i}.wav", o, 16000, subtype="FLOAT")
dur = len(x) / 16000
print(f"device {dev} audio {dur:.2f}s temps {min(times):.3f}s (médiane {np.median(times):.3f}) RTF {min(times)/dur:.4f}")
