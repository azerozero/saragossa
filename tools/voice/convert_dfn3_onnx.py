"""Convertit un export ONNX DeepFilterNet3 (enc, erb_dec, df_dec) en safetensors.

Usage : python3 -I -B tools/voice/convert_dfn3_onnx.py <export_dir> dfn3_ll.safetensors

`export_dir` contient `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` et `config.ini`
(archive officielle `DeepFilterNet3_ll_onnx.tar.gz`). Seuls les poids sont lus,
le graphe n'est jamais exécuté. Les noms rendus suivent le state dict PyTorch
attendu par `src/denoise/network.rs` :
- BatchNorm déjà repliée par l'export : la convolution porte `{bloc}.{i}.bias`
  sans `running_mean`, ce que `Conv::load` traite comme une échelle unité ;
- GRU ONNX (portes z, r, h ; biais concaténés) remis dans l'ordre PyTorch r, z, n ;
- les hyperparamètres de `config.ini` partent en métadonnées (`Config` côté Rust).
Requiert `onnx`, `numpy` et `safetensors` (déjà présents dans un venv local).
"""
import configparser
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

META_KEYS = {
    "df": ["sr", "fft_size", "hop_size", "nb_erb", "nb_df", "df_order", "df_lookahead"],
    "deepfilternet": [
        "conv_lookahead", "conv_ch", "conv_kernel", "conv_kernel_inp", "emb_hidden_dim",
        "emb_num_layers", "enc_linear_groups", "linear_groups", "df_hidden_dim",
        "df_num_layers", "df_pathway_kernel_size_t",
    ],
}


def tensors(graph):
    """Initialiseurs et sorties de nœuds `Constant` : l'export range les GRU dans les deux."""
    out = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    out[node.output[0]] = numpy_helper.to_array(a.t)
    return out


def gru_torch(w, r, b):
    """ONNX `[1, 3H, ·]` en (z, r, h) vers PyTorch `[3H, ·]` en (r, z, n)."""
    h = w.shape[1] // 3

    def reorder(x):
        return np.concatenate([x[h:2 * h], x[:h], x[2 * h:]])

    return reorder(w[0]), reorder(r[0]), reorder(b[0, :3 * h]), reorder(b[0, 3 * h:])


def convert(path, sub, out):
    graph = onnx.load(str(path)).graph
    vals = tensors(graph)
    convs, grus = {}, {}
    for node in graph.node:
        seg = node.name.strip("/").split("/")
        if node.op_type in ("Conv", "ConvTranspose"):
            convs.setdefault(seg[0], []).append(node)
        elif node.op_type == "GRU":
            grus.setdefault(seg[0], []).append(node)
        elif node.op_type == "Einsum":
            name = node.input[1]
            out[f"{sub}.{name}"] = vals[name]
        elif node.op_type == "MatMul":
            # `nn.Linear(n, 1)` exportée en MatMul `[n, 1]` + Add du biais.
            mod = seg[1] if "." in seg[1] else ".".join(seg[:2])
            out[f"{sub}.{mod}.weight"] = vals[node.input[1]].T
            out[f"{sub}.{mod}.bias"] = vals[f"{mod}.bias"]
    for mod, nodes in convs.items():
        main = nodes[0]
        kt = vals[main.input[1]].shape[2]
        idx = 1 if kt > 1 else 0  # un ConstantPad2d causal précède la conv si kt > 1
        out[f"{sub}.{mod}.{idx}.weight"] = vals[main.input[1]]
        last = nodes[-1]
        if len(nodes) == 2:
            idx += 1
            out[f"{sub}.{mod}.{idx}.weight"] = vals[last.input[1]]
        if len(last.input) < 3:
            raise SystemExit(f"{sub}.{mod}: biais replié attendu")
        out[f"{sub}.{mod}.{idx + 1}.bias"] = vals[last.input[2]]
    for mod, nodes in grus.items():
        name = f"{sub}.{mod}.gru"
        for layer, node in enumerate(nodes):
            w_ih, w_hh, b_ih, b_hh = gru_torch(*(vals[i] for i in node.input[1:4]))
            out[f"{name}.weight_ih_l{layer}"] = w_ih
            out[f"{name}.weight_hh_l{layer}"] = w_hh
            out[f"{name}.bias_ih_l{layer}"] = b_ih
            out[f"{name}.bias_hh_l{layer}"] = b_hh


def main():
    src, dst = Path(sys.argv[1]), sys.argv[2]
    out = {}
    for sub in ("enc", "erb_dec", "df_dec"):
        convert(src / f"{sub}.onnx", sub, out)
    out = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}
    cfg = configparser.ConfigParser()
    cfg.read(src / "config.ini")
    meta = {k: cfg[s][k].replace(" ", "") for s, keys in META_KEYS.items() for k in keys}
    meta["source"] = f"Rikorose/DeepFilterNet ONNX {src.name}"
    meta["license"] = "MIT OR Apache-2.0"
    save_file(out, dst, metadata=meta)
    print(f"{len(out)} tenseurs, {sum(v.size for v in out.values())} paramètres -> {dst}")


if __name__ == "__main__":
    main()
