"""Ventilation des octets de poids par catégorie (GGUF et safetensors).

Stdlib seulement. Les tailles viennent des offsets du fichier (padding
d'alignement compris), pas d'une estimation par type.

Usage : python3 -I weights_report.py FICHIER_OU_DOSSIER... > weights.json
"""

import hashlib
import json
import pathlib
import struct
import sys

GGUF_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16",
}

# Taille fixe des scalaires GGUF par identifiant de type de valeur.
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
            10: "Q", 11: "q", 12: "d"}


def category(name):
    """Catégorie d'un tenseur d'après son nom (conventions HF, MLX et GGUF)."""
    lowered = name.lower()
    if "mtp" in lowered or "nextn" in lowered:
        return "mtp"
    if lowered.startswith(("v.", "mm.", "visual", "model.visual", "vision")) or ".visual." in lowered:
        return "vision"
    if "embed_tokens" in lowered or lowered.startswith("token_embd"):
        return "embeddings"
    if "lm_head" in lowered or lowered.startswith("output."):
        return "lm_head"
    return "trunk"


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.handle.read(size))

    def string(self):
        (length,) = self.unpack("Q")
        return self.handle.read(length).decode("utf-8", "replace")

    def value(self, kind):
        if kind in _SCALARS:
            return self.unpack(_SCALARS[kind])[0]
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind, count = self.unpack("IQ")
            return [self.value(item_kind) for _ in range(count)]
        raise ValueError(f"type de valeur GGUF inconnu: {kind}")


def gguf_tensors(path):
    """(nom, type, octets) de chaque tenseur d'un GGUF, et ses métadonnées utiles."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        reader = _Reader(handle)
        if handle.read(4) != b"GGUF":
            raise ValueError(f"{path}: magie GGUF absente")
        (version,) = reader.unpack("I")
        n_tensors, n_kv = reader.unpack("QQ")
        meta = {}
        for _ in range(n_kv):
            key = reader.string()
            (kind,) = reader.unpack("I")
            value = reader.value(kind)
            # Les tableaux (vocabulaire, merges) ne servent pas au rapport.
            if not isinstance(value, list):
                meta[key] = value
        infos = []
        for _ in range(n_tensors):
            name = reader.string()
            (n_dims,) = reader.unpack("I")
            dims = reader.unpack("Q" * n_dims)
            kind, offset = reader.unpack("IQ")
            infos.append((name, kind, offset, dims))
        alignment = int(meta.get("general.alignment", 32))
        data_start = -(-handle.tell() // alignment) * alignment
    infos.sort(key=lambda info: info[2])
    ends = [info[2] for info in infos[1:]] + [size - data_start]
    tensors = [
        (name, GGUF_TYPES.get(kind, str(kind)), end - offset, dims)
        for (name, kind, offset, dims), end in zip(infos, ends)
    ]
    template = meta.pop("tokenizer.chat_template", None)
    keep = {key: value for key, value in meta.items()
            if key.startswith("general.") or key.endswith(("block_count", "file_type"))
            or key.startswith("tokenizer.ggml.") and not key.endswith("chat_template")}
    if template is not None:
        keep["chat_template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return {"format": "gguf", "version": version, "metadata": keep}, tensors


def safetensors_tensors(path):
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors.append((name, info["dtype"], end - start, tuple(info["shape"])))
    return {"format": "safetensors", "metadata": header.get("__metadata__", {})}, tensors


def report(path):
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path}: aucun fichier de poids")
    per_file = []
    categories = {}
    for file in files:
        reader = gguf_tensors if file.suffix == ".gguf" else safetensors_tensors
        header, tensors = reader(file)
        per_file.append({"file": file.name, "bytes_on_disk": file.stat().st_size, **header})
        # Un fichier de tête MTP (sidecar MLX, GGUF mtp-*) est MTP en entier :
        # ses tenseurs n'en portent pas toujours le nom. Sa propre copie des
        # embeddings ou de la tête de sortie reste distinguée (mtp/embeddings).
        whole_file_mtp = "mtp" in file.name.lower()
        for name, dtype, nbytes, _ in tensors:
            kind = category(name)
            if whole_file_mtp and kind != "mtp":
                kind = f"mtp/{kind}"
            entry = categories.setdefault(kind, {"bytes": 0, "tensors": 0, "dtypes": {}})
            entry["bytes"] += nbytes
            entry["tensors"] += 1
            entry["dtypes"][dtype] = entry["dtypes"].get(dtype, 0) + nbytes
    for entry in categories.values():
        entry["GB"] = round(entry["bytes"] / 1e9, 3)
        entry["GiB"] = round(entry["bytes"] / 2**30, 3)
    total = sum(entry["bytes"] for entry in categories.values())
    disk = sum(item["bytes_on_disk"] for item in per_file)
    return {
        "path": str(path),
        "files": per_file,
        "categories": categories,
        "tensor_bytes": total,
        "tensor_GB": round(total / 1e9, 3),
        "tensor_GiB": round(total / 2**30, 3),
        "disk_GB": round(disk / 1e9, 3),
        "disk_GiB": round(disk / 2**30, 3),
    }


def _self_check():
    assert category("token_embd.weight") == "embeddings"
    assert category("language_model.model.embed_tokens.scales") == "embeddings"
    assert category("output.weight") == "lm_head"
    assert category("blk.3.attn_q.weight") == "trunk"
    assert category("mtp.layers.0.mlp.gate_proj.weight") == "mtp"
    assert category("v.blk.0.attn_q.weight") == "vision"
    assert category("model.visual.blocks.0.attn.qkv.weight") == "vision"


if __name__ == "__main__":
    _self_check()
    json.dump([report(pathlib.Path(arg)) for arg in sys.argv[1:]], sys.stdout, indent=1)
    print()
