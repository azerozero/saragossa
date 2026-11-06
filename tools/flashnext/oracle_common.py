"""Outils communs aux oracles qwen4_exp : lecture des poids q4e8, construction
des couches officielles transformers et utilitaires de position.

Dépendances de développement hors crate (voir README) : torch, transformers
(qwen4_exp), numpy, tokenizers. Toujours lancer Python avec `-I`.
"""

import json
import os
import resource
import struct
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig  # noqa: E402

torch.set_grad_enabled(False)

E4M3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
E4M3[[0x7F, 0xFF]] = np.nan


def ngram_table(meta):
    """Chemin de la table n-gram relatif à `ngram/`, même règle que le loader
    Rust : ni absolu ni `..`, sans quoi la provenance hacherait un autre fichier."""
    name = meta["file"]
    if not isinstance(name, str) or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
        raise SystemExit(f"ngram/meta.json: file {name!r} hors de ngram/")
    return f"ngram/{name}"


def footprint_gb():
    """Pic RSS du processus (octets sous macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def write_safetensors(path, tensors, metadata=None):
    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for blob in blobs:
            fh.write(blob)


class Shards:
    """Lecteur safetensors minimal par memmap."""

    def __init__(self, directory, extra=()):
        self.dir = Path(directory)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        files = sorted(set(index["weight_map"].values())) + list(extra)
        self.entries, self.maps = {}, {}
        for name in files:
            path = self.dir / name
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            header.pop("__metadata__", None)
            self.maps[name] = np.memmap(path, dtype=np.uint8, mode="r")
            for key, entry in header.items():
                self.entries[key] = (name, 8 + n, entry)

    def has(self, key):
        return key in self.entries

    def array(self, key):
        name, start, entry = self.entries[key]
        a, b = entry["data_offsets"]
        data, dtype, shape = self.maps[name][start + a : start + b], entry["dtype"], entry["shape"]
        if dtype == "BF16":
            return (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dtype == "F32":
            return data.view(np.float32).reshape(shape)
        if dtype == "F8_E4M3":
            return E4M3[data].reshape(shape)
        if dtype == "U32":
            return data.view(np.uint32).reshape(shape)
        if dtype == "I64":
            return data.view(np.int64).reshape(shape)
        raise ValueError(f"{key}: dtype {dtype}")


def unpack_codes(words, bits, cols):
    """Codes MLX LSB-first en flux de bits par ligne : [N, P] uint32 → [N, cols]."""
    w = words.astype(np.uint64)
    pos = np.arange(cols, dtype=np.int64) * bits
    word, shift = pos >> 5, (pos & 31).astype(np.uint64)
    nxt = np.minimum(word + 1, words.shape[1] - 1)
    lo = w[:, word] >> shift
    spill = (shift + np.uint64(bits)) > 32
    hi = np.where(spill, (w[:, nxt] << (np.uint64(32) - shift)) & np.uint64(0xFFFFFFFF), 0)
    return ((lo | hi) & np.uint64((1 << bits) - 1)).astype(np.int64)


def mlx_dequant(words, scales, biases, bits, gs=64, chunk=4096):
    """`w = q·scale + bias` (u4/u8 en lanes, u6 chevauchant deux u32)."""
    rows, packed = words.shape
    cols = packed * 32 // bits
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(0, rows, chunk):
        codes = unpack_codes(np.asarray(words[r : r + chunk]), bits, cols).astype(np.float32)
        out[r : r + chunk] = (codes * np.repeat(scales[r : r + chunk], gs, axis=1)
                              + np.repeat(biases[r : r + chunk], gs, axis=1))
    return out


def q_name(name):
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    return name


class QWeights:
    """Sortie q4e8 de convert-qwen4exp : affines déquantifiés, copies telles quelles."""

    def __init__(self, qdir):
        self.dir = Path(qdir)
        extra = ("mtp.safetensors",) if (self.dir / "mtp.safetensors").exists() else ()
        self.sh = Shards(self.dir, extra=extra)
        quant = json.loads((self.dir / "config.json").read_text())["quantization"]
        self.default_bits = quant["bits"]
        self.group_size = quant["group_size"]
        self.overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}

    def _affine(self, base, index=None):
        words, scales, biases = (self.sh.array(base + s) for s in (".weight", ".scales", ".biases"))
        if index is not None:
            words, scales, biases = words[index], scales[index], biases[index]
        bits = self.overrides.get(base, {}).get("bits", self.default_bits)
        return mlx_dequant(words, scales, biases, bits, self.group_size)

    def dense(self, name):
        out = q_name(name)
        base = out[: -len(".weight")] if out.endswith(".weight") else None
        if base is not None and self.sh.has(base + ".scales"):
            return torch.from_numpy(self._affine(base))
        return torch.from_numpy(np.array(self.sh.array(out)))

    def rows(self, name, ids):
        base = q_name(name)[: -len(".weight")]
        bits = self.overrides.get(base, {}).get("bits", self.default_bits)
        return torch.from_numpy(mlx_dequant(self.sh.array(base + ".weight")[ids],
                                            self.sh.array(base + ".scales")[ids],
                                            self.sh.array(base + ".biases")[ids], bits, self.group_size))

    def expert(self, prefix, e):
        qprefix = q_name(prefix + "x")[:-1]
        return [torch.from_numpy(self._affine(f"{qprefix}mlp.switch_mlp.{p}", e))
                for p in ("gate_proj", "up_proj", "down_proj")]

    def ngram_scale(self, ple_layer):
        return float(self.sh.array(
            f"language_model.model.layers.{ple_layer}.ple.ple_embedding.ngram_embedding.weight_scale")[0])


class MmapNgram(torch.nn.Module):
    """Remplace `nn.Embedding` n-gram : gather E4M3 × échelle per-tensor dans ngram/."""

    def __init__(self, qdir, scale):
        super().__init__()
        meta = json.loads((Path(qdir) / "ngram" / "meta.json").read_text())
        rows, width = meta["shape"]
        self.table = np.memmap(Path(qdir) / ngram_table(meta), dtype=np.uint8, mode="r", shape=(rows, width))
        self.scale = scale
        self.register_buffer("weight", torch.empty(0), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1).numpy()
        uniq, inverse = np.unique(flat, return_inverse=True)
        values = E4M3[np.asarray(self.table[uniq])] * self.scale
        return torch.from_numpy(values[inverse].reshape(*ids.shape, -1))


def text_config(qdir):
    config = Qwen4ExpConfig.from_dict(json.loads((Path(qdir) / "config.json").read_text()))
    tc = config.text_config
    tc._attn_implementation = "sdpa"
    return tc


def build_layer(tc, index, weights, with_experts=True):
    """Couche officielle `Qwen4ExpTextDecoderLayer` sur les poids q4e8, experts f32
    dans le module officiel `Qwen4ExpTextExperts` (MoE non réimplémenté)."""
    with torch.device("meta"):
        layer = M.Qwen4ExpTextDecoderLayer(tc, index)
    layer.mlp.experts = torch.nn.Identity()
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = torch.nn.Identity()
    layer = layer.to_empty(device="cpu")
    prefix = f"model.language_model.layers.{index}."
    state = {}
    for key, param in layer.state_dict().items():
        if key.split(".")[-1] in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
            state[key] = torch.from_numpy(np.array(weights.sh.array(q_name(prefix + key))))
            continue
        tensor = weights.dense(prefix + key)
        state[key] = tensor.reshape(param.shape).to(param.dtype)
    missing, unexpected = layer.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = MmapNgram(weights.dir, weights.ngram_scale(index))
    if with_experts:
        with torch.device("meta"):
            experts = M.Qwen4ExpTextExperts(tc)
        gate_up = torch.empty(tc.num_experts, 2 * tc.moe_intermediate_size, tc.hidden_size)
        down = torch.empty(tc.num_experts, tc.hidden_size, tc.moe_intermediate_size)
        for e in range(tc.num_experts):
            g, u, d = weights.expert(prefix, e)
            gate_up[e] = torch.cat([g, u], dim=0)
            down[e] = d
        experts.gate_up_proj = torch.nn.Parameter(gate_up, requires_grad=False)
        experts.down_proj = torch.nn.Parameter(down, requires_grad=False)
        layer.mlp.experts = experts
    return layer.eval()


def rope(tc, length):
    rotary = M.Qwen4ExpTextRotaryEmbedding(tc)
    pos = torch.arange(length)[None, None].expand(3, 1, length)
    return rotary(torch.zeros(1, length, tc.hidden_size), pos)


def causal_mask(length):
    return torch.tril(torch.ones(length, length, dtype=torch.bool))[None, None]


def embed(tc, weights, ids):
    rows = weights.rows("model.language_model.embed_tokens.weight", ids.reshape(-1).numpy())
    return rows.reshape(1, ids.shape[1], -1).repeat(1, 1, tc.hc_count)


class TopkCapture:
    """Capture les top-k des routeurs (`torch.topk`) et de l'indexeur QSA
    (`Tensor.topk` sur des scores 1-D), pour figer les scores de l'oracle."""

    def __init__(self):
        self.indexer = []
        self._topk, self._method = torch.topk, torch.Tensor.topk

    def __enter__(self):
        capture = self

        def method(values, k, *args, **kwargs):
            if values.dim() == 1:
                capture.indexer.append(values.detach().float().clone())
            return capture._method(values, k, *args, **kwargs)

        torch.Tensor.topk = method
        return self

    def __exit__(self, *exc):
        torch.Tensor.topk = self._method
        return False


def flatten_scores(scores):
    """Scores par requête → (valeurs aplaties F32, offsets I64 de longueur n+1)."""
    offsets = [0]
    for row in scores:
        offsets.append(offsets[-1] + len(row))
    flat = torch.cat(scores) if scores else torch.zeros(0)
    return flat.numpy().astype(np.float32), np.array(offsets, dtype=np.int64)
