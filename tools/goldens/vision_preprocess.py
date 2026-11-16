#!/usr/bin/env python3
"""Capture CPU sans poids : smart_resize installé, resize et normalisation.

Exécuter avec l'interpréteur existant, -I -B ; aucune installation implicite.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_smart_resize():
    # Exécuter seulement cette fonction pure : importer le module entier exige
    # torchvision même quand la référence de repli PIL est choisie.
    package = importlib.util.find_spec("transformers")
    source = Path(package.origin).parent / "models/qwen2_vl/image_processing_qwen2_vl.py"
    data = source.read_bytes()
    tree = ast.parse(data)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "smart_resize")
    namespace = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), source.name, "exec"), namespace)
    return namespace["smart_resize"], digest(data)


def synthetic(height, width, pattern):
    import numpy as np

    y, x, c = np.indices((height, width, 3), dtype=np.uint32)
    if pattern == "ramp":
        values = x * 13 + y * 7 + c * 73
    elif pattern == "checker":
        values = ((x + y + c) % 2) * 255
    elif pattern == "colors":
        values = c * 127 + (x // 16 + y // 16) % 2
    elif pattern == "impulse":
        values = ((x == width // 2) & (y == height // 2)) * 255 + c * 0
    else:
        raise ValueError(pattern)
    return (values % 256).astype(np.uint8)


def capture(output):
    import numpy as np
    import PIL
    from PIL import Image
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    smart_resize, source_sha = load_smart_resize()
    has_torchvision = importlib.util.find_spec("torchvision") is not None
    if has_torchvision:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2.functional import resize
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("square", 256, 256, "ramp", "Qwen38"),
        ("portrait", 320, 224, "ramp", "Qwen36"),
        ("landscape", 224, 320, "ramp", "Qwen38"),
        ("colors", 256, 256, "colors", "Thx02"),
        ("checker", 256, 256, "checker", "D3Flash"),
        ("ties", 272, 304, "ramp", "Qwen38"),
        ("tiny", 17, 31, "ramp", "Qwen36"),
        ("up-portrait", 129, 63, "checker", "Thx02"),
        ("up-landscape", 63, 129, "impulse", "D3Flash"),
        ("down-budget", 1313, 1297, "ramp", "Thx02"),
        ("down-checker", 513, 777, "checker", "Qwen38"),
        ("one-axis", 256, 511, "ramp", "Qwen36"),
    ]
    records = []
    for name, h, w, pattern, profile in cases:
        maximum = 16777216 if profile in ("Qwen38", "Qwen36") else 1638400
        rh, rw = smart_resize(h, w, 32, 65536, maximum)
        rgb = synthetic(h, w, pattern)
        if has_torchvision:
            resized = resize(torch.from_numpy(rgb).permute(2, 0, 1), [rh, rw],
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
            resized = resized.permute(1, 2, 0).numpy()
        else:
            resized = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BICUBIC))
        data = resized.tobytes()
        filename = name + "-rgb.bin"
        (output / filename).write_bytes(data)
        # Oracle indépendant du parcours Rust, identique à la permutation HF.
        pixels = torch.from_numpy(resized.copy()).permute(2, 0, 1).float()
        pixels = (pixels - 127.5) / 127.5
        frames = pixels.unsqueeze(0).repeat(2, 1, 1, 1)
        patches = frames.reshape(1, 2, 3, rh // 32, 2, 16, rw // 32, 2, 16)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 1536)
        records.append({"name": name, "height": h, "width": w, "pattern": pattern,
                        "profile": profile, "resized": [rh, rw], "file": filename,
                        "sha256": digest(data), "grid_thw": [1, rh // 16, rw // 16],
                        "patches_sha256": digest(patches.numpy().astype("<f4").tobytes())})
    # 256 niveaux f32 suffisent à contrôler chaque valeur des fixtures sans
    # stocker six copies flottantes de chaque image.
    levels = ((torch.arange(256, dtype=torch.float32) - 127.5) / 127.5).numpy()
    (output / "normalized-f32.bin").write_bytes(levels.astype("<f4").tobytes())
    boundary_cases = []
    for profile, maximum in [("Qwen38", 16777216), ("Qwen36", 16777216),
                             ("Thx02", 1638400), ("D3Flash", 1638400)]:
        for h, w in [(1, 1), (1, 200), (200, 1), (240, 272), (272, 304),
                     (255, 256), (257, 257), (4096, 4096), (4112, 4112),
                     (4113, 4113), (1280, 1280), (1296, 1296), (1297, 1297),
                     (6000, 3000), (2**32 - 1, 2**32 - 1)]:
            boundary_cases.append({"profile": profile, "input": [h, w],
                                   "output": smart_resize(h, w, 32, 65536, maximum)})
    manifest = {"backend": "torchvision" if has_torchvision else "PIL",
                "pillow": PIL.__version__, "torch": torch.__version__,
                "torchvision": importlib.metadata.version("torchvision") if has_torchvision else None,
                "transformers": importlib.metadata.version("transformers"),
                "smart_resize_source_sha256": source_sha,
                "generator_sha256": digest(Path(__file__).read_bytes()),
                "normalized_sha256": digest((output / "normalized-f32.bin").read_bytes()),
                "fixtures": records, "smart_resize": boundary_cases}
    probes = []
    for h, w, rh, rw, pattern in [(257, 259, 32, 32, "checker"),
                                 (33, 49, 7, 11, "ramp"),
                                 (1, 73, 32, 32, "ramp"),
                                 (73, 1, 32, 32, "ramp"),
                                 (64, 64, 17, 64, "impulse"),
                                 (64, 64, 64, 17, "colors")]:
        rgb = synthetic(h, w, pattern)
        if has_torchvision:
            resized = resize(torch.from_numpy(rgb).permute(2, 0, 1), [rh, rw],
                             interpolation=InterpolationMode.BICUBIC, antialias=True)
            resized = resized.permute(1, 2, 0).numpy()
        else:
            resized = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BICUBIC))
        name = f"probe-{len(probes)}"
        data = resized.tobytes()
        filename = name + "-rgb.bin"
        (output / filename).write_bytes(data)
        probes.append({"name": name, "height": h, "width": w, "pattern": pattern,
                       "profile": "Qwen38", "resized": [rh, rw], "file": filename,
                       "sha256": digest(data)})
    manifest["filter_probes"] = probes
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(records)} fixtures, référence {manifest['backend']}, sortie {output}")


if __name__ == "__main__":
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise SystemExit("Exécuter avec -I -B")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args().output)
