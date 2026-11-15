"""Référence chat longue Gemma 4 : poids packés, activations f32, bruit bf16.

Sous gpulock, avec SARAGOSSA_GOLDEN_CAPTURE=1 et SARAGOSSA_GEMMA4_DIR.
--verify recalcule les captures et exige des payloads identiques, sans écrire.
Le corpus public et les versions sont ceux de kld.py. Aucun téléchargement.
"""
import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LOGITS_PACKAGES, OUT, local_model, provenance, require_versions, write, write_json
from kld import canonical_embed_scale, stats

LENGTHS, SCORED, K = (1024, 2048, 4100), 128, 64
PACKAGES = (*LOGITS_PACKAGES, "numpy")
INSTRUCTION = (
    "Read the following excerpt from A Scandal in Bohemia. Write a detailed "
    "literary analysis of at least 600 words, discussing the narrator's point "
    "of view, characterization, evidence and inference. Support each argument "
    "with specific details from the excerpt.\n\nExcerpt:\n"
)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prompts(tokenizer):
    text = (OUT / "kld/corpus.txt").read_text()
    result = []
    # Longest character prefix fitting the target: no padding or repetition.
    for target in LENGTHS:
        lo, hi = 0, len(text)
        while lo < hi:
            middle = (lo + hi + 1) // 2
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION + text[:middle]}],
                add_generation_prompt=True, return_dict=False,
            )
            if len(ids) <= target:
                lo = middle
            else:
                hi = middle - 1
        content = INSTRUCTION + text[:lo]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, return_dict=False)
        if not target - 4 <= len(ids) <= target:
            raise RuntimeError(f"longueur inattendue : {len(ids)} / {target}")
        result.append((target, content, ids))
    return result


def audit_memory(mx, model, dtype):
    from mlx.utils import tree_flatten
    arrays = tree_flatten(model.parameters())
    sizes = collections.Counter()
    for _, value in arrays:
        sizes[str(value.dtype)] += value.nbytes
        if mx.issubdtype(value.dtype, mx.floating) and value.dtype != dtype:
            raise RuntimeError(f"paramètre flottant inattendu : {value.dtype}")
    quantized = [m for m in model.modules() if hasattr(m, "bits")]
    bits = collections.Counter(m.bits for m in quantized)
    if set(bits) != {4, 8} or any(m.weight.dtype != mx.uint32 for m in quantized):
        raise RuntimeError("poids mixtes packés 4/8 bits requis")
    info = dict(parameter_bytes=dict(sizes), quantized_modules={str(k): v for k, v in bits.items()})
    print("memory", json.dumps(info), "active", mx.get_active_memory(), flush=True)
    return info


def distributions(mx, model, prompt, forced=None):
    """Un prefill entier, puis 127 appels decode; pas de chunking implicite.

    Seuls les derniers logits sont retenus. La normalisation f64 CPU reproduit
    metrics() en Rust, y compris pour les logits bf16 convertis en f32.
    """
    import numpy as np
    from mlx.utils import tree_flatten
    cache = model.make_cache()
    inputs = prompt
    for position in range(SCORED):
        logits = model(mx.array([inputs]), cache=cache)[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
        if position == 0:
            cache_dtypes = sorted({str(v.dtype) for _, v in tree_flatten([c.state for c in cache])
                                   if isinstance(v, mx.array)})
            expected = model.language_model.model.embed_tokens.scales.dtype
            if logits.dtype != expected:
                raise RuntimeError(f"activations inattendues : {logits.dtype} / {expected}")
            if expected == mx.float32 and cache_dtypes != [str(mx.float32)]:
                raise RuntimeError(f"cache f32 attendu : {cache_dtypes}")
            print("prefill", len(prompt), "logits", logits.dtype,
                  "cache_dtypes", cache_dtypes, "peak_bytes", mx.get_peak_memory(), flush=True)
        values = np.asarray(logits.astype(mx.float32))[0].astype(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("logits non finis")
        top = int(values.argmax())
        log_z = values.max() + np.log(np.exp(values - values.max()).sum())
        yield values - log_z, top
        inputs = [top if forced is None else forced[position]]


def capture(mx, model, prompt, tokenizer):
    import numpy as np
    rows, answer = [], []
    for lp, token in distributions(mx, model, prompt):
        if token in tokenizer.eos_token_ids:
            raise RuntimeError(f"EOS avant les {SCORED} positions : {len(answer)}")
        ids = np.argsort(-lp, kind="stable")[:K].tolist()
        rows.append((ids, lp[ids].astype(np.float32).tolist()))
        answer.append(token)
    return rows, answer


def measure_noise(mx, model, prompt, answer, reference):
    import numpy as np
    kls, deltas, same = [], [], []
    for (lp, token), (ids, ref) in zip(distributions(mx, model, prompt, answer), reference):
        p, q = np.exp(np.array(ref, dtype=np.float64)), np.exp(lp[ids])
        p = np.append(p, max(1.0 - p.sum(), 0.0))
        q = np.append(q, max(1.0 - q.sum(), 0.0))
        valid = p > 0
        kls.append(float((p[valid] * np.log(p[valid] / np.maximum(q[valid], np.finfo(np.float64).tiny))).sum()))
        deltas.append(float(np.abs(p - q).max()))
        same.append(token == ids[0])
    return stats(kls, deltas, same)


def main():
    require_versions(PACKAGES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("GPU_LOCK_HELD"):
        raise RuntimeError("lancer sous tools/gpulock.py")
    path = local_model("gemma4")
    import mlx.core as mx
    from mlx_lm import load
    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    print("device", device, flush=True)
    # Bound allocator/cache without dequantizing the checkpoint.
    budget = min(64 * 2**30, device["max_recommended_working_set_size"])
    mx.set_memory_limit(budget)
    mx.set_cache_limit(2 * 2**30)
    source = provenance("gemma4", path, PACKAGES)
    source.update(float32_control=True, model=path.name,
                  weight_sha256={p.name: digest(p) for p in sorted(path.glob("*.safetensors"))},
                  corpus_sha256=digest(OUT / "kld/corpus.txt"),
                  protocol="full-prefill, cached greedy f32, 128 positions, top64+rest, f64 softmax")
    if not source["weight_sha256"]:
        raise RuntimeError("poids safetensors absents")
    model, tokenizer = load(path)
    model.set_dtype(mx.float32)
    canonical_embed_scale(mx, model)
    mx.eval(model.parameters())
    source["f32_memory"] = audit_memory(mx, model, mx.float32)
    cases = prompts(tokenizer)
    references = []
    for target, content, prompt in cases:
        print("capture f32", target, "prompt_tokens", len(prompt), flush=True)
        rows, answer = capture(mx, model, prompt, tokenizer)
        references.append((rows, answer))
        mx.clear_cache()
    del model
    gc.collect()
    mx.clear_cache()
    model, _ = load(path)
    mx.eval(model.parameters())
    source["bf16_memory"] = audit_memory(mx, model, mx.bfloat16)
    for (target, content, prompt), (rows, answer) in zip(cases, references):
        noise = measure_noise(mx, model, prompt, answer, rows)
        print("noise", target, json.dumps(noise), flush=True)
        name = f"gemma4_long_{target}"
        flat = [x for ids, values in rows for x in (*map(float, ids), *values)]
        payload = struct.pack(f"<{len(flat)}f", *flat)
        meta = dict(model=name, chunks=[prompt + answer], scored=SCORED, top_k=K,
                    mlx_bf16_noise=noise, prompt_tokens=len(prompt), chat_content=content)
        if args.verify:
            if payload != (OUT / f"kld/{name}.bin").read_bytes():
                raise RuntimeError(f"{name}: référence f32 non reproductible")
            if meta != json.loads((OUT / f"kld/{name}_meta.bin").read_bytes()):
                raise RuntimeError(f"{name}: tokens/bruit bf16 non reproductibles")
            header = json.loads((OUT / f"kld/{name}.json").read_text())
            if header["source"] != source:
                raise RuntimeError(f"{name}: provenance différente")
            print(name, "reproduction exacte", flush=True)
        else:
            tolerance = "KL moyen et p99 delta_p <= 1.5 x bruit bf16 local; same_top >= bf16"
            write("kld", name, payload, "f32", [SCORED, 2, K], source, tolerance)
            write_json("kld", name + "_meta", meta, source, tolerance)
        mx.clear_cache()


if __name__ == "__main__":
    main()
