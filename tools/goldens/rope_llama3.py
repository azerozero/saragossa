"""Bits f32 de la table RoPE llama3 de transformers, figés dans src/rope.rs.

Référence CPU : transformers et torch aux PINS de common.py, sans poids ni
réseau. Lancer avec python -I.
"""
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import require_versions


def main():
    require_versions(("transformers", "torch"))
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    # Llama 3.2 1B (d64, ×32) puis Llama 3.2 3B / 3.1 8B (d128, ×8).
    for head_dim, factor in [(64, 32.0), (128, 8.0)]:
        config = LlamaConfig(hidden_size=head_dim * 32, num_attention_heads=32, head_dim=head_dim,
                             max_position_embeddings=131072,
                             rope_parameters=dict(rope_type="llama3", rope_theta=500000.0,
                                                  factor=factor, low_freq_factor=1.0,
                                                  high_freq_factor=4.0,
                                                  original_max_position_embeddings=8192))
        inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
        bits = [struct.unpack("<I", struct.pack("<f", value))[0] for value in inv_freq.tolist()]
        print(f"d{head_dim} x{factor}:", ", ".join(f"0x{b:08x}" for b in bits))


if __name__ == "__main__":
    main()
