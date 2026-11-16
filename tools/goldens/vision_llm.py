"""Calibration irréversible des logits image Qwen dense, sans encodeur ni réseau.

Charge une couche à la fois, exécute le prefill puis 16 pas imposés avec le
cache Transformers. La calibration est publiée avant toute comparaison Rust.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    output = args.fixtures / "llm-calibration.json"
    if output.exists():
        raise RuntimeError("calibration déjà figée ; recalibrage interdit")
    # Le timeout du verrou borne seulement la file : borner aussi le ticket actif.
    if args.device == "mps":
        signal.alarm(1740)
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
    )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    helpers = runpy.run_path(str(Path(__file__).with_name("vision.py")))
    environment = helpers["environment"]()
    environment["llm_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw = json.loads((args.model / "config.json").read_bytes())
    if raw["model_type"] != "qwen3_5":
        raise ValueError("calibration dense BF16 seulement ; quantifié exige une référence déquantifiée distincte")
    config = Qwen3_5TextConfig(**raw["text_config"])
    config._attn_implementation = "eager"
    mapping = json.loads((args.model / "model.safetensors.index.json").read_bytes())["weight_map"]
    cases = json.loads((args.fixtures / "mrope-inputs.json").read_bytes())["cases"]
    assert len(cases) == 6 and all(len(c["forced_decode_ids"]) == 16 for c in cases)
    embedding_hashes = {
        name: hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest()
        for name in sorted({span["embedding_file"] for case in cases for span in case["image_spans"]})
    }
    prefix = "model.language_model."
    weight_hashes = {}

    def tensor(name, dtype, device):
        with safe_open(args.model / mapping[name], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(name)
        if value.dtype != torch.bfloat16:
            raise ValueError(f"poids non BF16 : {name}")
        if name not in weight_hashes:
            weight_hashes[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        return value.to(device=device, dtype=dtype)

    def load(module, stem, dtype):
        state = {name: tensor(stem + name, dtype, args.device) for name in module.state_dict()}
        module.load_state_dict(state, strict=True, assign=True)
        return module.eval()

    @torch.inference_mode()
    def capture(dtype):
        started = time.monotonic()
        # Embedding CPU : seuls les rangs utiles sont envoyés sur le GPU.
        embedding = tensor(prefix + "embed_tokens.weight", dtype, "cpu")
        hidden, positions = [], []
        for case in cases:
            n = len(case["input_ids"])
            ids = case["input_ids"] + case["forced_decode_ids"]
            h = embedding[torch.tensor(ids)].unsqueeze(0).clone()
            for span in case["image_spans"]:
                data = np.fromfile(args.fixtures / span["embedding_file"], dtype="<f4").copy()
                value = torch.from_numpy(data.reshape(span["length"], config.hidden_size)).to(dtype)
                h[0, span["start"]:span["start"] + span["length"]] = value
            p = torch.tensor(case["position_ids"], dtype=torch.long)
            decode = torch.arange(n, n + 16) + case["rope_delta"]
            p = torch.cat((p, decode.view(1, 1, -1).expand(3, 1, -1)), dim=2)
            hidden.append(h.to(args.device))
            positions.append(p.to(args.device))
        del embedding
        rotary = Qwen3_5TextRotaryEmbedding(config).to(args.device)
        rotations = [rotary(h, p) for h, p in zip(hidden, positions)]
        for index in range(config.num_hidden_layers):
            with torch.device("meta"):
                layer = Qwen3_5DecoderLayer(config, index)
            layer = load(layer, prefix + f"layers.{index}.", dtype)
            for case_id, case in enumerate(cases):
                n = len(case["input_ids"])
                cache = DynamicCache(config=config)
                outputs = []
                for start, end in [(0, n)] + [(n + i, n + i + 1) for i in range(16)]:
                    mask = None
                    if config.layer_types[index] == "full_attention":
                        mask = torch.zeros((1, 1, end - start, end), dtype=dtype, device=args.device)
                        if start == 0:
                            mask.masked_fill_(torch.ones((end, end), dtype=torch.bool, device=args.device).triu(1), torch.finfo(dtype).min)
                    outputs.append(layer(hidden[case_id][:, start:end],
                        position_embeddings=tuple(r[:, start:end] for r in rotations[case_id]),
                        attention_mask=mask, past_key_values=cache, use_cache=True))
                hidden[case_id] = torch.cat(outputs, dim=1)
                del cache, outputs
            del layer
            gc.collect()
            if args.device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            print(f"{dtype} layer={index+1}/{config.num_hidden_layers} seconds={time.monotonic()-started:.1f}", flush=True)
        with torch.device("meta"):
            norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm = load(norm, prefix + "norm.", dtype)
        final = [norm(h[:, len(c["input_ids"]) - 1:]) for h, c in zip(hidden, cases)]
        del hidden, norm
        # La tête f32 dépasse 4 Gio : borner chaque transfert MPS, sans
        # tronquer le vocabulaire ni changer la somme sur hidden_size.
        head = tensor("lm_head.weight", dtype, "cpu")
        pieces = [[] for _ in final]
        for start in range(0, config.vocab_size, 32768):
            chunk = head[start:start + 32768].to(args.device)
            for h, output in zip(final, pieces):
                output.append((h @ chunk.T).float().cpu().numpy()[0])
            del chunk
        result = [np.concatenate(output, axis=1) for output in pieces]
        del head, final
        if args.device == "mps":
            torch.mps.empty_cache()
        return result

    f32 = capture(torch.float32)
    bf16 = capture(torch.bfloat16)
    records = []
    for index, (ref, noisy) in enumerate(zip(f32, bf16)):
        assert ref.shape == noisy.shape == (17, config.vocab_size)
        assert np.isfinite(ref).all() and np.isfinite(noisy).all()
        error = noisy.astype(np.float64) - ref.astype(np.float64)
        def softmax(x):
            x = x.astype(np.float64)
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        noise = {"rms": np.sqrt((error * error).mean(axis=1)),
                 "max_abs": np.abs(error).max(axis=1),
                 "max_probability": np.abs(softmax(ref) - softmax(noisy)).max(axis=1)}
        payloads = {}
        for label, value in [("f32", ref), ("bf16", noisy)]:
            path = args.fixtures / f"llm-{index}-{label}.bin"
            data = value.astype("<f4").tobytes()
            with path.open("xb") as file:
                file.write(data)
            payloads[label] = {"file": path.name, "sha256": hashlib.sha256(data).hexdigest()}
        records.append({"shape": list(ref.shape), "payloads": payloads,
                        "noise": {k: v.tolist() for k, v in noise.items()},
                        "threshold": {k: (2 * v).tolist() for k, v in noise.items()}})
    import transformers
    source_root = Path(transformers.__file__).parent
    environment["loaded_transformers_sources_sha256"] = helpers["hash_records"]({
        str(Path(module.__file__).relative_to(source_root)): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in sorted(sys.modules.items())
        if name.startswith("transformers.") and getattr(module, "__file__", None)
        and Path(module.__file__).suffix == ".py"
        and Path(module.__file__).is_relative_to(source_root)
    })
    result = {"schema_version": 1, "checkpoint": args.model.name, "environment": environment,
              "device": args.device, "attention": "eager", "weights_source_dtype": "BF16", "head_chunk_rows": 32768,
              "weight_sha256": helpers["hash_records"](weight_hashes),
              "embedding_sha256": helpers["hash_records"](embedding_hashes),
              "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "mrope_inputs_sha256": hashlib.sha256((args.fixtures / "mrope-inputs.json").read_bytes()).hexdigest(),
              "rule": "exactly twice BF16 versus f32; frozen before Rust comparison", "cases": records}
    with output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
