#!/usr/bin/env python3
"""Gemma 4 MoE local capture. Run inference explicitly under the operator's GPU lock.

--golden META.json --write-ids IDS.json only extracts IDs; it never imports MLX.
The inference mode wraps mlx-lm 0.32.2 methods in memory without replacing math.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer_dump import CallHooks, Writer, golden_ids, validate_ids


def instrument(text, writer, hooks, mx):
    import numpy as np

    def save(layer, point, value):
        dtype = str(value.dtype)
        value = value.astype(mx.float32)
        mx.eval(value)
        writer.save(layer, point, np.asarray(value), dtype)

    for i, layer in enumerate(text.model.layers):
        def setup(i, layer):
            residual = []

            def layer_before(x, *args, **kwargs):
                if i == 0:
                    save(None, "embeddings", x)  # The actual value after embed_scale.
                save(i, "input", x)

            hooks.add(layer, before=layer_before, after=lambda out: save(i, "output", out[0]))
            hooks.add(layer.input_layernorm, after=lambda out: save(i, "attention_input", out))
            hooks.add(layer.self_attn, after=lambda out: save(i, "attention_raw", out[0]))
            hooks.add(layer.post_attention_layernorm, after=lambda out: save(i, "attention_out", out))

            def dense_before(x):
                residual[:] = [x]
                save(i, "post_attention_residual", x)

            hooks.add(layer.pre_feedforward_layernorm, before=dense_before,
                      after=lambda out: save(i, "dense_input", out))
            hooks.add(layer.mlp, after=lambda out: save(i, "dense_raw", out))
            hooks.add(layer.post_feedforward_layernorm_1, after=lambda out: save(i, "dense_out", out))
            hooks.add(layer.router.proj, before=lambda x: save(i, "router_input", x),
                      after=lambda out: save(i, "router_logits", out))

            def routing(out):
                indices, weights = out
                order = mx.argsort(indices, axis=-1)
                save(i, "router_indices", mx.take_along_axis(indices, order, axis=-1))
                save(i, "router_weights", mx.take_along_axis(weights, order, axis=-1))

            hooks.add(layer.router, after=routing)
            hooks.add(layer.pre_feedforward_layernorm_2, after=lambda out: save(i, "moe_input", out))
            hooks.add(layer.experts, after=lambda out: save(i, "moe_raw", out))
            hooks.add(layer.post_feedforward_layernorm_2, after=lambda out: save(i, "moe_out", out))

            def ffn_after(out):
                save(i, "ffn_out", out)
                # No module exists at the residual add. Reproduce this single
                # expression for observation only, with the original dtype.
                save(i, "pre_scalar", residual[0] + out)

            hooks.add(layer.post_feedforward_layernorm, before=lambda x: save(i, "ffn_sum", x), after=ffn_after)

        setup(i, layer)
    hooks.add(text.model.norm, after=lambda out: save(None, "final_norm", out[:, -1:, :]))
    return save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ids-file", type=Path, help="JSON array of token IDs; no tokenization")
    source.add_argument("--ids", help="JSON array of token IDs")
    source.add_argument("--golden", type=Path, help="golden metadata JSON with adjacent .bin payload")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--write-ids", type=Path, help="extract IDs only, without model or GPU")
    parser.add_argument("--model", type=Path, help="existing local checkpoint directory")
    parser.add_argument("--out", type=Path, help="new output directory")
    parser.add_argument("--float32", action="store_true", help="control run: cast floating weights/scales to f32, keep packed quantized weights")
    args = parser.parse_args()
    ids = (golden_ids(args.golden, args.prompt_index) if args.golden else
           validate_ids(json.loads(args.ids_file.read_text() if args.ids_file else args.ids)))
    if args.write_ids:
        with args.write_ids.open("x") as f:
            json.dump(ids, f)
            f.write("\n")
        print(f"{len(ids)} IDs: {ids}")
        return
    if args.model is None or args.out is None:
        parser.error("inference requires --model and --out")
    path = args.model.resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        parser.error("expected an existing local checkpoint with config.json and safetensors")
    if args.out.exists():
        parser.error("output already exists")
    versions = {p: version(p) for p in ("mlx-lm", "mlx")}
    if versions != {"mlx-lm": "0.32.2", "mlx": "0.32.4"}:
        raise RuntimeError(f"expected golden versions mlx-lm=0.32.2, mlx=0.32.4; got {versions}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.models import gemma4_text
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    # load_model(Path) reads local shards directly and never calls _download.
    model, _ = load_model(path)
    text = getattr(model, "language_model", model)
    if not isinstance(text, gemma4_text.Model) or not text.args.enable_moe_block or text.args.hidden_size_per_layer_input:
        raise ValueError("capture supports Gemma 4 MoE without per-layer input gating")
    if args.float32:
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    writer = Writer(args.out, ids, len(text.model.layers), "mlx-prefill",
                    versions=versions, float32_control=args.float32,
                    config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest())
    with CallHooks() as hooks:
        save = instrument(text, writer, hooks, mx)
        logits = model(mx.array([ids]), cache=make_prompt_cache(model))[:, -1:, :]
        save(None, "logits", logits)
    writer.finish()
    print(f"capture complete; greedy={int(mx.argmax(logits).item())}")


if __name__ == "__main__":
    main()
