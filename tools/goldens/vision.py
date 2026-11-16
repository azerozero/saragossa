"""Oracle vision local : inventaire strict et embeddings eager CPU f32.

Exécuter avec l'interpréteur de référence en -I -B. Aucun AutoModel,
trust_remote_code, téléchargement, import ou écriture depuis les checkpoints.
Les poids BF16 sont lus un tenseur à la fois, seul le ViT devient résident.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import MethodType, SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

MODELS = ("Qwen3.8-27B-BF16", "Qwen3.6-35B-A3B-4bit", "THX-02", "d3-flash")
PREFIXES = ("model.visual.", "vision_tower.", "visual.")
SOURCES = (
    "models/qwen3_5/modeling_qwen3_5.py",
    "models/qwen3_5/configuration_qwen3_5.py",
    "models/qwen2_vl/image_processing_qwen2_vl.py",
    "models/qwen3_vl/processing_qwen3_vl.py",
    "models/qwen3_vl/video_processing_qwen3_vl.py",
    "image_processing_backends.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"clé JSON dupliquée : {key}")
        out[key] = value
    return out


def read_json(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"JSON trop grand : {path.name}")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object)


def environment():
    import transformers

    versions = {}
    for package in ("torch", "transformers", "numpy", "safetensors", "torchvision", "tokenizers", "pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if versions["transformers"] != "5.8.1" or versions["torch"] != "2.12.0":
        raise RuntimeError("oracle figé pour transformers 5.8.1 / torch 2.12.0")
    root = Path(transformers.__file__).parent
    sources = {name: digest((root / name).read_bytes()) for name in SOURCES}
    return {
        "versions": versions,
        "transformers_source_sha256": sources,
        "transformers_sha256": digest(json.dumps(sources, sort_keys=True).encode()),
        "transformers_git_sha": None,
        "git_sha_note": "wheel installé sans métadonnée Git ; SHA256 des sources exécutées",
        "python": sys.version.split()[0],
        "reference_script_sha256": digest(Path(__file__).read_bytes()),
    }


def canonical(name):
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return prefix, name[len(prefix):]
    return None


def expected_shapes(config, mlx=False):
    h, mid, out = (config[k] for k in ("hidden_size", "intermediate_size", "out_hidden_size"))
    profile = (config["depth"], h, mid, out)
    if profile not in ((27, 1152, 4304, 5120), (27, 1152, 4304, 2048),
                       (24, 1024, 4096, 2560), (27, 1152, 4304, 4096)):
        raise ValueError(f"profil vision inconnu : {profile}")
    for key, value in {"patch_size": 16, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "in_channels": 3, "num_heads": 16,
                       "num_position_embeddings": 2304, "deepstack_visual_indexes": [],
                       "hidden_act": "gelu_pytorch_tanh"}.items():
        if config.get(key) != value:
            raise ValueError(f"configuration vision incompatible : {key}")
    shapes = {"patch_embed.proj.weight": [h, 2, 16, 16, 3] if mlx else [h, 3, 2, 16, 16],
              "patch_embed.proj.bias": [h], "pos_embed.weight": [2304, h],
              "merger.norm.weight": [h], "merger.norm.bias": [h],
              "merger.linear_fc1.weight": [h * 4, h * 4], "merger.linear_fc1.bias": [h * 4],
              "merger.linear_fc2.weight": [out, h * 4], "merger.linear_fc2.bias": [out]}
    for layer in range(config["depth"]):
        for stem, shape in {"norm1": [h], "norm2": [h], "attn.qkv": [h * 3, h],
                            "attn.proj": [h, h], "mlp.linear_fc1": [mid, h],
                            "mlp.linear_fc2": [h, mid]}.items():
            shapes[f"blocks.{layer}.{stem}.weight"] = shape
            shapes[f"blocks.{layer}.{stem}.bias"] = shape[:1]
    return shapes


def inspect_model(path):
    raw = read_json(path / "config.json")
    config = raw.get("backbone_config", raw)
    vision = config["vision_config"]
    expected = expected_shapes(vision, path.name == MODELS[1])
    tensors, headers, all_keys, seen = [], {}, set(), set()
    prefixes = set()
    for shard in sorted(path.glob("*.safetensors")):
        with shard.open("rb") as stream:
            prefix_bytes = stream.read(8)
            if len(prefix_bytes) != 8:
                raise ValueError("header tronqué")
            length = struct.unpack("<Q", prefix_bytes)[0]
            if not 0 < length <= 128 * 1024 * 1024:
                raise ValueError("taille header invalide")
            encoded = stream.read(length)
            if len(encoded) != length:
                raise ValueError("header tronqué")
            header = json.loads(encoded, object_pairs_hook=unique_object)
            headers[shard.name] = {"header_sha256": digest(encoded), "size_bytes": shard.stat().st_size}
            for name, entry in sorted(header.items()):
                if name == "__metadata__":
                    continue
                # Tous les noms sont inspectés, y compris ceux hors index HF.
                if name in all_keys:
                    raise ValueError(f"tenseur dupliqué : {name}")
                all_keys.add(name)
                match = canonical(name)
                if match is None:
                    if any(part in name.lower() for part in ("vision", "visual", "image", "deepstack")):
                        raise ValueError(f"tenseur multimodal non classé : {name}")
                    continue
                prefix, key = match
                prefixes.add(prefix)
                if key in seen or key not in expected or entry["shape"] != expected[key] or entry["dtype"] != "BF16":
                    raise ValueError(f"tenseur vision incompatible : {name}")
                seen.add(key)
                start, end = entry["data_offsets"]
                count = math.prod(entry["shape"]) * 2
                if start < 0 or end - start != count or 8 + length + end > shard.stat().st_size:
                    raise ValueError(f"offsets invalides : {name}")
                stream.seek(8 + length + start)
                payload_hash = hashlib.sha256()
                remaining = count
                while remaining:
                    chunk = stream.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("payload tronqué")
                    payload_hash.update(chunk)
                    remaining -= len(chunk)
                tensors.append({"name": name, "canonical_name": key, "shard": shard.name,
                                "shape": entry["shape"], "dtype": entry["dtype"],
                                "data_offsets": [start, end], "sha256": payload_hash.hexdigest()})
    if seen != expected.keys() or len(prefixes) != 1:
        raise ValueError(f"inventaire incomplet : {sorted(expected.keys() - seen)}")
    processors = {p.name: {"sha256": digest(p.read_bytes()), "config": read_json(p)}
                  for p in sorted(path.glob("*processor*.json"))}
    if not processors:
        raise ValueError("processeur absent")
    processor_file = processors.get("processor_config.json")
    image_processor = (processor_file["config"]["image_processor"] if processor_file
                       else processors["preprocessor_config.json"]["config"])
    for key, value in {"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                       "image_mean": [0.5] * 3, "image_std": [0.5] * 3}.items():
        if image_processor.get(key) != value:
            raise ValueError(f"processeur image incompatible : {key}")
    if image_processor["size"] != {"shortest_edge": 65536,
                                    "longest_edge": 1638400 if path.name == "THX-02" else 16777216}:
        raise ValueError("bornes du processeur image incompatibles")
    return {
        "checkpoint": path.name, "model_type": config["model_type"], "vision_config": vision,
        "text_rope": config["text_config"]["rope_parameters"],
        "full_attention_interval": config["text_config"]["full_attention_interval"],
        "declared_transformers_version": config.get("transformers_version"),
        "config_sha256": digest((path / "config.json").read_bytes()),
        "tokenizer_sha256": digest((path / "tokenizer.json").read_bytes()),
        "runtime_source_sha256": {p.name: digest(p.read_bytes()) for p in
                                  [path / "d3_runtime.py"] if p.is_file()},
        "prefix": next(iter(prefixes)), "conv_layout": "OTHWI" if path.name == MODELS[1] else "OITHW",
        "linear_layout": "OI", "processors": processors,
        "effective_image_pixels": [65536, 1638400 if path.name in MODELS[2:] else 16777216],
        "effective_pixels_source": "THX processor; d3 runtime contract; Qwen processor",
        "shards": headers, "total_header_tensors": len(all_keys),
        "expected_vision_tensors": len(expected), "identified_vision_tensors": len(seen),
        "unknown_vision_tensors": [], "tensors": sorted(tensors, key=lambda item: item["name"]),
        "vision_payload_sha256": digest("".join(t["sha256"] for t in sorted(tensors, key=lambda t: t["name"])).encode()),
    }


def synthetic_images():
    import numpy as np

    for label, height, width in (("square", 256, 256), ("landscape", 224, 320), ("portrait", 320, 224)):
        y, x = np.indices((height, width), dtype=np.uint32)
        rgb = np.stack(((x * 3 + y * 5) % 256, ((x // 8 + y // 8) % 2) * 255,
                        (x * 7 + y * 11 + 19) % 256), axis=-1).astype(np.uint8)
        yield label, rgb


def patches_without_resize(rgb):
    """Chemin sans resize du processeur 5.8.1, normalisation fusionnée.

    Pas de substitution PIL au backend torchvision absent. Le layout est
    [bloc_h,bloc_w,intra_h,intra_w,C,T,patch_h,patch_w].
    """
    import torch

    h, w, _ = rgb.shape
    if h % 32 or w % 32 or not 65536 <= h * w <= 1638400:
        raise ValueError("fixture hors porte sans redimensionnement")
    pixels = (torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() - 127.5) / 127.5
    patches = pixels.reshape(3, h // 32, 2, 16, w // 32, 2, 16)
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1)
    return patches.reshape(h * w // 256, 1536), torch.tensor([[1, h // 16, w // 16]])


def encode_tensor(value):
    import torch

    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    array = value.numpy()
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    payload = array.tobytes()
    return payload, {"dtype": str(array.dtype), "shape": list(array.shape),
                     "sha256": digest(payload), "size_bytes": len(payload)}


def save_tensor(directory, name, value, shared=False):
    payload, metadata = encode_tensor(value)
    path = directory / f"{name}.bin"
    if shared and path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"table partagée différente : {path.name}")
    else:
        path.write_bytes(payload)
    return {"file": path.name, **metadata}


def save_table(output, name, value):
    directory = output / "tables"
    directory.mkdir(exist_ok=True)
    metadata = save_tensor(directory, name, value, shared=True)
    metadata["file"] = "../tables/" + metadata["file"]
    return metadata


def interpolation_tables(model, grid):
    """Capture les variables exactes de la fonction installée, sans linspace recodé."""
    captured = {}
    target = model.fast_pos_embed_interpolate.__func__.__code__

    def profiler(frame, event, arg):
        if frame.f_code is target and event == "return":
            for name in ("idx_tensor", "weight_tensor"):
                captured[name] = frame.f_locals[name].detach().clone()

    previous = sys.getprofile()
    try:
        sys.setprofile(profiler)
        interpolated = model.fast_pos_embed_interpolate(grid)
    finally:
        sys.setprofile(previous)
    if set(captured) != {"idx_tensor", "weight_tensor"}:
        raise RuntimeError("fonction d'interpolation modifiée : capture refusée")
    # Rejouer les tables exportées et vérifier l'ordre 2x2, bit pour bit.
    lookup = model.pos_embed(captured["idx_tensor"]) * captured["weight_tensor"][:, :, None]
    summed = lookup[0] + lookup[1] + lookup[2] + lookup[3]
    t, h, w = grid[0].tolist()
    replay = summed.repeat(t, 1).view(t, h // 2, 2, w // 2, 2, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
    if not replay.equal(interpolated):
        raise RuntimeError("rejeu d'interpolation différent de Transformers")
    return captured


def capture(path, manifest, output):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    config = Qwen3_5VisionConfig(**manifest["vision_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5VisionModel(config)
    # Charger seulement le ViT, sans allouer un décodeur ni du code de checkpoint.
    state = {}
    for shard_name in sorted({t["shard"] for t in manifest["tensors"]}):
        with safe_open(path / shard_name, framework="pt", device="cpu") as source:
            for entry in manifest["tensors"]:
                if entry["shard"] != shard_name:
                    continue
                weight = source.get_tensor(entry["name"])
                # Vérifie les octets réellement consommés, pas seulement l'ancien manifeste.
                if digest(weight.view(torch.uint8).numpy().tobytes()) != entry["sha256"]:
                    raise ValueError("poids modifié depuis l'inventaire")
                if entry["canonical_name"] == "patch_embed.proj.weight" and manifest["conv_layout"] == "OTHWI":
                    weight = weight.permute(0, 4, 1, 2, 3).contiguous()
                state[entry["canonical_name"]] = weight.float()
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Buffer non persistant créé sur meta : reconstruire via la classe installée.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
    model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
    model.eval()
    directory = output / path.name
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for label, rgb in synthetic_images():
            patches, grid = patches_without_resize(rgb)
            tables = interpolation_tables(model, grid)
            result = model(patches, grid_thw=grid).pooler_output
            if not torch.isfinite(result).all():
                raise ValueError("embeddings non finis")
            payloads = {"embeddings": save_tensor(directory, label + "-embeddings", result),
                        "interpolation_indices": save_table(output, label + "-indices", tables["idx_tensor"]),
                        "interpolation_weights": save_table(output, label + "-weights", tables["weight_tensor"])}
            records.append({"image": label, "rgb_sha256": digest(rgb.tobytes()), "grid_thw": grid.tolist(),
                            "placeholder_count": result.shape[0], "payloads": payloads})
            print(f"{path.name}/{label}: {list(result.shape)}", flush=True)
    write_json(directory / "fixtures.json", {"schema_version": 1, "checkpoint": path.name,
               "environment": environment(), "vision_payload_sha256": manifest["vision_payload_sha256"],
               "dtype": "f32", "weights_source_dtype": "BF16", "device": "cpu", "attention": "eager",
               "resize": False, "preprocessing": "no-resize port of transformers 5.8.1 fused normalize",
               "bf16_logit_calibration": "pending stage 3; never inferred from embedding noise",
               "cases": records})
    capture_positions(path, directory, records)


def capture_positions(path, directory, records):
    """Six préfixes et 16 positions de decode, sans instancier le LLM."""
    import torch
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    oracle = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    oracle.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, oracle)
    images = {r["image"]: r for r in records}
    cases = []
    for labels in (("square",), ("landscape",), ("portrait",),
                   ("square", "landscape"), ("landscape", "portrait"), ("portrait", "square")):
        ids = tokenizer.encode("Décris les couleurs et les formes. ", add_special_tokens=False).ids
        types = [0] * len(ids)
        grids, spans = [], []
        for label in labels:
            start, pad, end = [tokenizer.token_to_id(t) for t in
                               ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
            if any(t is None for t in (start, pad, end)):
                raise ValueError("marqueurs vision absents")
            ids.append(start)
            types.append(0)
            count = images[label]["placeholder_count"]
            spans.append({"image": label, "start": len(ids), "length": count,
                          "embedding_file": images[label]["payloads"]["embeddings"]["file"]})
            ids.extend([pad] * count + [end])
            types.extend([1] * count + [0])
            grids.extend(images[label]["grid_thw"])
        suffix = tokenizer.encode(" Réponds brièvement.", add_special_tokens=False).ids
        ids.extend(suffix)
        types.extend([0] * len(suffix))
        positions, delta = Qwen3_5Model.get_rope_index(
            oracle, torch.tensor([ids]), torch.tensor([types]), image_grid_thw=torch.tensor(grids))
        d = delta.item()
        forced = tokenizer.encode(" Rouge bleu vert noir blanc. Les formes sont régulières.",
                                  add_special_tokens=False).ids
        forced = (forced * 16)[:16]
        cases.append({"images": list(labels), "input_ids": ids, "mm_token_type_ids": types,
                      "image_grid_thw": grids, "image_spans": spans,
                      "position_ids": positions.tolist(), "rope_delta": d,
                      "forced_decode_ids": forced, "decode_rope_positions": [len(ids) + i + d for i in range(16)],
                      "decode_kv_positions": [len(ids) + i for i in range(16)]})
    write_json(directory / "mrope-inputs.json", {"schema_version": 1, "cases": cases,
               "oracle": "Qwen3_5Model.get_rope_index; no LLM weights loaded",
               "logits": "not captured in stage 0; calibration and comparison belong to stage 3"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "capture"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("exécuter en -I -B")
    root, output = args.model_root.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(root) or "ai_model" in output.parts or "models" in output.parts:
        raise ValueError("sortie interdite dans un répertoire de poids")
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    if args.action == "inventory":
        manifests = [inspect_model(root / name) for name in MODELS]
        write_json(output / "manifest.json", {"schema_version": 1, "environment": env, "checkpoints": manifests})
        print([(m["checkpoint"], m["identified_vision_tensors"]) for m in manifests])
    else:
        manifest = read_json(output / "manifest.json")
        if manifest["environment"] != env:
            raise ValueError("environnement modifié depuis l'inventaire")
        shared = output / "images"
        shared.mkdir(exist_ok=True)
        images = []
        import torch
        for label, rgb in synthetic_images():
            patches, grid = patches_without_resize(rgb)
            images.append({"image": label, "grid_thw": grid.tolist(),
                           "rgb": save_tensor(shared, label + "-rgb", torch.from_numpy(rgb)),
                           "patches": encode_tensor(patches)[1]})
        write_json(shared / "images.json", images)
        for item in manifest["checkpoints"]:
            if args.model is None or item["checkpoint"] == args.model:
                capture(root / item["checkpoint"], item, output)


if __name__ == "__main__":
    main()
