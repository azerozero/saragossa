"""Convertit le checkpoint officiel DeepFilterNet3 (`model_120.ckpt.best`) en safetensors.

Usage : python3 -I -B tools/voice/convert_dfn3.py <DeepFilterNet3>/checkpoints/model_120.ckpt.best dfn3.safetensors

Le checkpoint est un pickle PyTorch : il est lu en `weights_only=True`, sans
exécuter de code arbitraire. Seuls les tenseurs flottants sont gardés
(`num_batches_tracked` est entier et inutile en inférence).
"""
import sys

import torch
from safetensors.torch import save_file

src, dst = sys.argv[1:3]
state = torch.load(src, weights_only=True, map_location="cpu")
tensors = {k: v.float().contiguous() for k, v in state.items() if v.dtype.is_floating_point}
save_file(tensors, dst, metadata={"source": "Rikorose/DeepFilterNet DeepFilterNet3", "license": "MIT OR Apache-2.0"})
print(f"{len(tensors)} tenseurs, {sum(v.numel() for v in tensors.values())} paramètres -> {dst}")
