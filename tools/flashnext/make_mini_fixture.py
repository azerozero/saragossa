"""Fixture N1 : petit modèle qwen4_exp aléatoire, exactement représentable en q4e8.

Écrit `<sortie>/source/` (checkpoint au format FP8 officiel attendu par
`convert-qwen4exp`) et `<sortie>/golden.safetensors` (sorties officielles
transformers 5.20.2, f32, sdpa, eval). Les poids sont choisis pour que la
conversion soit sans perte :
  - experts : e4m3 entiers dans [-8, 7] × échelle 2^-6, chaque groupe de 64
    contient -8 et 7 → u4 exact ;
  - denses u8 : entiers [-128, 127] × 2^-10, chaque groupe contient les deux
    extrêmes → u8 exact ;
  - lm_head u6 : entiers [-32, 31] × 2^-8 → u6 exact ;
  - copies (normes, conv, routeur, gates…) : bf16 quelconques.
Usage : python -I tools/flashnext/make_mini_fixture.py tests/fixtures/qwen4exp_mini
"""

import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_common import TopkCapture, flatten_scores, write_safetensors  # noqa: E402

torch.set_grad_enabled(False)
SEED = int(os.environ.get("FIXTURE_SEED", "20261110"))
EOS = 5
TOKENS = 40

TEXT = dict(
    hidden_size=64,
    num_hidden_layers=3,
    layer_types=["linear_attention", "linear_attention", "full_attention"],
    full_attention_interval=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    partial_rotary_factor=0.25,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                     "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    hc_count=2,
    hc_lowrank=64,
    indexer_budget=16,
    indexer_compress_ratio=4,
    indexer_head_dim=16,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    ple_layer_ids=[2],
    ple_embed_dim=64,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=37,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=2,
    vocab_size=128,
    output_gate_type="sigmoid",
    rms_norm_eps=1e-6,
    hidden_act="silu",
    max_position_embeddings=4096,
    mtp_num_hidden_layers=0,
    eos_token_id=EOS,
    bos_token_id=EOS,
    tie_word_embeddings=False,
)


def bf16_round(t):
    return t.to(torch.bfloat16).to(torch.float32)


def grid(shape, lo, hi, step, gen, group=64):
    """Entiers dans [lo, hi] × step, chaque groupe de `group` colonnes contenant lo et hi."""
    ints = torch.randint(lo, hi + 1, shape, generator=gen)
    flat = ints.reshape(-1, shape[-1])
    for start in range(0, shape[-1], group):
        flat[:, start] = lo
        flat[:, start + 1] = hi
    return (flat.reshape(shape).to(torch.float32)) * step


def bf16_bytes(t):
    return t.contiguous().to(torch.bfloat16).view(torch.int16).numpy().tobytes()


def main(out):
    out = Path(out)
    (out / "source").mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)
    config = Qwen4ExpTextConfig(**TEXT)
    config._attn_implementation = "sdpa"
    model = M.Qwen4ExpForCausalLM(config).eval()

    exact_u8 = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "input_mix_weight_down", "input_mix_weight_up", "index_qk_proj", "key_proj", "value_proj",
                "embed_tokens"}
    sd = {}
    for name, param in model.state_dict().items():
        shape = tuple(param.shape)
        leaf = name.split(".")[-2] if name.endswith(".weight") else name.split(".")[-1]
        if "ngram_embedding" in name:
            continue  # table n-gram : remplie plus bas en e4m3 × échelle
        if name.endswith("layer_multipliers") or name.endswith("vocab_sizes") or name.endswith("heads_offsets"):
            sd[name] = param.clone()
            continue
        if ".mlp.experts." in name:
            continue  # experts : plus bas
        if name == "lm_head.weight":
            sd[name] = grid(shape, -32, 31, 2.0 ** -8, gen)
        elif leaf in exact_u8 or (".shared_expert." in name):
            sd[name] = grid(shape, -128, 127, 2.0 ** -10, gen)
        elif name.endswith("A_log"):
            sd[name] = bf16_round(torch.empty(shape).uniform_(-1.0, 1.0, generator=gen))
        elif name.endswith("dt_bias"):
            sd[name] = bf16_round(torch.empty(shape).uniform_(-0.5, 0.5, generator=gen))
        elif name.endswith("linear_attn.norm.weight"):
            sd[name] = bf16_round(1.0 + torch.empty(shape).uniform_(-0.2, 0.2, generator=gen))
        elif "norm" in leaf or "layernorm" in leaf:
            sd[name] = bf16_round(torch.empty(shape).uniform_(-0.2, 0.2, generator=gen))
        else:  # conv, routeur, gates, block_inject, in_proj_a/b
            sd[name] = bf16_round(torch.empty(shape).uniform_(-0.3, 0.3, generator=gen))

    # Experts : e4m3 entiers [-8, 7] × 2^-6 par expert ; échelle par bloc 2^-6.
    expert_src = {}
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        e, i, h = config.num_experts, config.moe_intermediate_size, config.hidden_size
        gate_up = torch.zeros(e, 2 * i, h)
        down = torch.zeros(e, h, i)
        for expert in range(e):
            for proj, shape in (("gate_proj", (i, h)), ("up_proj", (i, h)), ("down_proj", (h, i))):
                ints = grid(shape, -8, 7, 1.0, gen)
                w = ints * 2.0 ** -6
                if proj == "gate_proj":
                    gate_up[expert, :i] = w
                elif proj == "up_proj":
                    gate_up[expert, i:] = w
                else:
                    down[expert] = w
                expert_src[(layer, expert, proj)] = ints
        sd[prefix + "gate_up_proj"] = gate_up
        sd[prefix + "down_proj"] = down

    # Table n-gram : e4m3 aléatoires (sans NaN) × échelle per-tensor bf16.
    ple = model.model.layers[1].ple.ple_embedding
    rows, width = ple.ngram_embedding.weight.shape
    codes = torch.randint(0, 254, (rows, width), generator=gen).to(torch.uint8)
    codes = torch.where(codes >= 0x7F, codes + 1, codes)
    scale = bf16_round(torch.tensor([0.0625]))
    table = codes.view(torch.float8_e4m3fn).to(torch.float32) * scale
    sd["model.layers.1.ple.ple_embedding.ngram_embedding.weight"] = table

    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing, unexpected)

    # Entrées : un EOS au milieu (redémarrage du contexte n-gram).
    ids = torch.randint(6, config.vocab_size, (1, TOKENS), generator=gen)
    ids[0, 17] = EOS
    last = config.num_hidden_layers - 1
    captures = {"router": {}, "routing": {}, "qsa": None, "last": None}

    def router_hook(index):
        def hook(_m, _inp, out):
            captures["router"][index] = out[0].clone()
            captures["routing"][index] = out[2].clone()
        return hook

    for index, layer in enumerate(model.model.layers):
        layer.mlp.gate.register_forward_hook(router_hook(index))
        if layer.layer_type != "linear_attention":
            layer.self_attn.indexer.register_forward_hook(
                lambda _m, _i, out: captures.__setitem__("qsa", out.clone()))
    # Sortie de la dernière couche (visible sinon seulement à travers le mixer).
    model.model.layers[last].register_forward_hook(
        lambda _m, _i, out: captures.__setitem__("last", (out[0] if isinstance(out, tuple) else out).clone()))
    with TopkCapture() as topk:
        result = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden = result.hidden_states  # (entrée, couches 0..L-2, sortie du mixer final)
    assert len(hidden) == config.num_hidden_layers + 1

    # Marge des routeurs au seuil : le golden doit être non ambigu sur le routage.
    worst = float("inf")
    for logits in captures["router"].values():
        probs = torch.softmax(logits.float(), dim=-1).sort(dim=-1, descending=True).values
        k = config.num_experts_per_tok
        worst = min(worst, float((probs[:, k - 1] - probs[:, k]).min()))
    assert worst > 1e-4, f"marge des routeurs {worst:.2e} : changer de graine"
    print("marge minimale des routeurs", worst)

    golden = {"input_ids": ("I64", ids.shape[1:], ids[0].numpy().astype(np.int64).tobytes()),
              "logits": ("F32", result.logits.shape[1:], result.logits[0].float().numpy().tobytes())}
    # L'embedding (u8 exact) n'est pas figé : le test Rust le recalcule à l'identique.
    for index in range(config.num_hidden_layers - 1):
        h = hidden[index + 1][0].float()
        golden[f"layer.{index}"] = ("F32", h.shape, h.numpy().tobytes())
    h = captures["last"][0].float()
    golden[f"layer.{last}"] = ("F32", h.shape, h.numpy().tobytes())
    golden["final_mixer"] = ("F32", hidden[-1][0].shape, hidden[-1][0].float().numpy().tobytes())
    for index in range(config.num_hidden_layers):
        r = captures["routing"][index].to(torch.int64)
        golden[f"routing.{index}"] = ("I64", r.shape, r.numpy().tobytes())
        logits = captures["router"][index].float()
        golden[f"router_logits.{index}"] = ("F32", logits.shape, logits.numpy().tobytes())
    q = captures["qsa"][0, 0].to(torch.uint8)
    golden[f"qsa_mask.{last}"] = ("U8", q.shape, q.numpy().tobytes())
    # Scores torch de l'indexeur par requête (vides sans bloc complet).
    compress = config.indexer_compress_ratio
    rows_scores, it = [], iter(topk.indexer)
    for i in range(TOKENS):
        rows_scores.append(next(it) if (i + 1) // compress > 0 else torch.zeros(0))
    assert next(it, None) is None, "scores d'indexeur en surnombre"
    flat, offsets = flatten_scores(rows_scores)
    golden[f"qsa_scores.{last}"] = ("F32", flat.shape, flat.tobytes())
    golden[f"qsa_offsets.{last}"] = ("I64", offsets.shape, offsets.tobytes())

    # Golden QSA à marge nette : module d'attention officiel de la dernière couche
    # seul, entrée aléatoire, graine cherchée jusqu'à une marge > 1e-3 au seuil.
    attn = model.model.layers[last].self_attn
    keep = config.indexer_budget // compress
    pos = M.Qwen4ExpTextRotaryEmbedding(config)(
        torch.zeros(1, TOKENS, config.hidden_size), torch.arange(TOKENS)[None, None].expand(3, 1, TOKENS))
    mask = torch.tril(torch.ones(TOKENS, TOKENS, dtype=torch.bool))[None, None]
    for attempt in range(1000):
        xg = torch.Generator().manual_seed(SEED + 1 + attempt)
        x = bf16_round(torch.randn(1, TOKENS, config.hidden_size, generator=xg))
        sel = {}
        hook = attn.indexer.register_forward_hook(lambda _m, _i, out: sel.__setitem__("m", out.clone()))
        with TopkCapture() as clear:
            out_x, _ = attn(x, pos, attention_mask=mask, past_key_values=None)
        hook.remove()
        margins = [float(v.sort(descending=True).values[keep - 1] - v.sort(descending=True).values[keep])
                   for v in clear.indexer if v.numel() > keep]
        if margins and min(margins) > 1e-3:
            break
    else:
        raise SystemExit("aucune entrée QSA à marge nette")
    print("golden QSA net : tentative", attempt, "marge", min(margins))
    golden["qsa_clear.x"] = ("F32", x.shape[1:], x[0].numpy().tobytes())
    golden["qsa_clear.out"] = ("F32", out_x.shape[1:], out_x[0].float().numpy().tobytes())
    m = sel["m"][0, 0].to(torch.uint8)
    golden["qsa_clear.mask"] = ("U8", m.shape, m.numpy().tobytes())
    write_safetensors(out / "golden.safetensors", golden)

    # Source au format FP8 officiel (noms model.language_model.*, experts découpés).
    src = {}
    for name, value in sd.items():
        if ".mlp.experts." in name or "ngram_embedding" in name:
            continue
        target = "lm_head.weight" if name == "lm_head.weight" else "model.language_model." + name[len("model."):]
        if value.dtype == torch.int64:
            src[target] = ("I64", value.shape, value.numpy().astype(np.int64).tobytes())
        else:
            assert torch.equal(bf16_round(value), value), name
            src[target] = ("BF16", value.shape, bf16_bytes(value))
    for (layer, expert, proj), ints in expert_src.items():
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
        fp8 = ints.to(torch.float8_e4m3fn)
        assert torch.equal(fp8.to(torch.float32), ints)
        src[base + ".weight"] = ("F8_E4M3", ints.shape, fp8.view(torch.uint8).numpy().tobytes())
        blocks = (-(-ints.shape[0] // 128), -(-ints.shape[1] // 128))
        src[base + ".weight_scale_inv"] = ("BF16", blocks, bf16_bytes(torch.full(blocks, 2.0 ** -6)))
    half = rows // config.split_ngram_parts
    for part in range(config.split_ngram_parts):
        chunk = codes[part * half : (part + 1) * half]
        src[f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{part}.weight"] = (
            "F8_E4M3", chunk.shape, chunk.numpy().tobytes())
    src["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"] = (
        "BF16", (1,), bf16_bytes(scale))
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(out / "source" / shard, src)
    (out / "source" / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: shard for k in sorted(src)}}, separators=(",", ":")))
    text = dict(TEXT)
    text["model_type"] = "qwen4_exp_text"
    (out / "source" / "config.json").write_text(json.dumps(
        {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": text},
        indent=1))
    for side in ("tokenizer.json", "tokenizer_config.json"):
        (out / "source" / side).write_text(f"fixture synthétique ({side}), graine {SEED}\n")
    (out / "source" / "LICENSE").write_text(
        "Fixture synthétique générée par tools/flashnext/make_mini_fixture.py (poids aléatoires,\n"
        "aucune donnée tierce). Distribuée sous la licence du dépôt saragossa : Apache-2.0\n"
        "(voir LICENSE-APACHE à la racine).\n")
    (out / "RECIPE.json").write_text(json.dumps({
        "script": "tools/flashnext/make_mini_fixture.py",
        "command": "python -I tools/flashnext/make_mini_fixture.py tests/fixtures/qwen4exp_mini",
        "seed": SEED, "torch": torch.__version__,
        "transformers": transformers.__version__, "attn_implementation": "sdpa", "dtype": "float32",
        "tokens": TOKENS, "eos_position": 17}, indent=1))
    print("ok", sum(f.stat().st_size for f in out.rglob("*") if f.is_file()), "octets")


if __name__ == "__main__":
    main(sys.argv[1])
