"""Oracle onnxruntime du réseau DFN3 exporté (enc → erb_dec, df_dec), sans DSP.

Usage : dfn3_onnx_ref.py <export_dir> <golden_dfn3.safetensors> <out.safetensors>

Rejoue les features `erb_feat` [T, 32] et `spec_feat` [T, 96, 2] d'un golden
existant (`dfn3_ref.py` : elles ne dépendent pas du modèle) dans les trois
graphes ONNX, sur toute la séquence d'un coup, et écrit les entrées, le masque
[T, 32], les coefficients DF [T, 96, 10] et le LSNR [T]. Sans anticipation
(`conv_lookahead = 0`), la trame τ du réseau lit la feature τ.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file, save_file

src, golden, dst = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
g = load_file(golden)
erb = g["erb_feat"][None, None]  # [1, 1, T, 32]
spec = np.ascontiguousarray(g["spec_feat"].transpose(2, 0, 1)[None])  # [1, 2, T, 96]


def run(name, feeds):
    sess = ort.InferenceSession(str(src / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    wanted = {i.name for i in sess.get_inputs()}
    return dict(zip(names, sess.run(None, {k: v for k, v in feeds.items() if k in wanted})))


enc = run("enc", {"feat_erb": erb, "feat_spec": spec})
mask = run("erb_dec", enc)["m"][0, 0]
coefs = run("df_dec", enc)["coefs"][0]
save_file(
    {
        "erb_feat": g["erb_feat"],
        "spec_feat": g["spec_feat"],
        "mask": np.ascontiguousarray(mask, dtype=np.float32),
        "coefs": np.ascontiguousarray(coefs, dtype=np.float32),
        "lsnr": np.ascontiguousarray(enc["lsnr"][0, :, 0], dtype=np.float32),
    },
    dst,
)
print(f"{mask.shape[0]} trames -> {dst}")
