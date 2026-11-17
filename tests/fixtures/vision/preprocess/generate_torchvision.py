#!/usr/bin/env python3
"""Ajoute un oracle torchvision CPU aux captures PIL, sans poids ni réseau.

Exécuter avec python -I -B. Les sorties RGB restent sous 1 Mio au total.
Le pipeline est celui du backend HF 5.8.1 : uint8 NCHW contigu,
tvF.resize BICUBIC antialias=True, puis normalisation f32 et patchs.
Les tailles smart_resize sont celles du manifeste PIL déjà capturé.
"""

import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
from torchvision.transforms.v2 import functional as tvF


def digest(data):
    return hashlib.sha256(data).hexdigest()


def synthetic(record):
    height, width = record["height"], record["width"]
    y, x, c = np.indices((height, width, 3), dtype=np.uint32)
    pattern = record["pattern"]
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


def capture(directory):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reference = {
        "backend": "torchvision",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "hf_pipeline": "https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/image_processing_backends.py",
        "hf_processor": "https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py",
        "input": "CPU uint8 contiguous NCHW, batch=1",
        "interpolation": "BICUBIC",
        "antialias": True,
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "resize_source_sha256": digest(Path(inspect.getfile(tvF.resize)).read_bytes()),
    }
    # Un cas inchangé, un resize sur deux axes et deux agrandissements ;
    # les petites sondes ajoutent réduction forte, impulsion et axes unitaires.
    selected = {"square", "ties", "tiny", "up-portrait"}
    for group in ("fixtures", "filter_probes"):
        records = []
        for original in manifest[group]:
            if group == "fixtures" and original["name"] not in selected:
                continue
            rgb = synthetic(original)
            rh, rw = original["resized"]
            pil = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BICUBIC))
            # Ne jamais remplacer silencieusement l'oracle PIL existant.
            assert digest(pil.tobytes()) == original["sha256"], original["name"]
            assert (directory / original["file"]).read_bytes() == pil.tobytes()
            batch = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().unsqueeze(0)
            resized = tvF.resize(batch, [rh, rw], interpolation=tvF.InterpolationMode.BICUBIC,
                                 antialias=True)
            assert resized.dtype == torch.uint8
            data = resized[0].permute(1, 2, 0).contiguous().numpy().tobytes()
            record = dict(original)
            record["file"] = original["name"] + "-torchvision-rgb.bin"
            record["sha256"] = digest(data)
            record["pil_max_abs_uint8"] = int(np.abs(
                np.frombuffer(data, dtype=np.uint8).astype(np.int16)
                - pil.reshape(-1).astype(np.int16)).max())
            if group == "fixtures":
                # Normalisation fusionnée et permutation du processeur HF.
                pixels = tvF.normalize(resized.float(), [127.5] * 3, [127.5] * 3)
                patches = pixels.reshape(1, 3, rh // 32, 2, 16, rw // 32, 2, 16)
                patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
                patches = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1)
                record["patches_sha256"] = digest(patches.reshape(-1, 1536).numpy().astype("<f4").tobytes())
            (directory / record["file"]).write_bytes(data)
            records.append(record)
            print(f'{record["name"]}: PIL/torchvision max_abs_uint8={record["pil_max_abs_uint8"]}')
        reference[group] = records
    manifest["torchvision_reference"] = reference
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise SystemExit("Exécuter avec -I -B")
    capture(Path(__file__).resolve().parent)
