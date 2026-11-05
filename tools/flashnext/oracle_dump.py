"""Goldens N2 : couches officielles transformers sur les poids q4e8 déquantifiés.

Chaque couche est le module officiel `Qwen4ExpTextDecoderLayer` (sdpa, f32,
eval, sans autocast), experts déquantifiés en f32 dans `Qwen4ExpTextExperts` :
le MoE n'est pas réimplémenté. Sont figés : sorties de couche, logits et
indices du routeur, masque et scores de l'indexeur QSA. `provenance.json`
contient une entrée par golden : son empreinte, celles des fichiers de poids
recalculées sur disque, la config, les versions et le backend. Un même dossier
refuse des goldens d'origines différentes. Les tests Rust recalculent les
empreintes des fichiers ouverts avant l'inférence.

  --layers : une séquence de 8000 tokens (couches 0-3) ; ses préfixes couvrent
             les longueurs 64, 2047-2052 et 3000 (tout est causal).
  --full   : logits du modèle complet sur quatre prompts courts.

Usage :
  python -I tools/flashnext/oracle_dump.py --selftest
  python -I tools/flashnext/oracle_dump.py --layers <q4e8> <tokenizer.json> <texte> <sortie>
  python -I tools/flashnext/oracle_dump.py --full   <q4e8> <tokenizer.json> <sortie>
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import oracle_common as C  # noqa: E402

PROMPTS = [
    "Le moteur saragossa exécute les modèles Qwen sur Apple Silicon.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    '{"name": "get_weather", "arguments": {"city": "Lyon", "date": "2026-11-06"}}',
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_files(qdir):
    """Fichiers réellement lus par les oracles : shards indexés, MTP, `ngram/meta.json`
    et la table qu'il désigne (pas un nom supposé)."""
    qdir = Path(qdir)
    index = json.loads((qdir / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if (qdir / "mtp.safetensors").exists():
        names.append("mtp.safetensors")
    meta = json.loads((qdir / "ngram" / "meta.json").read_text())
    names += ["ngram/meta.json", f"ngram/{meta['file']}"]
    return names


def environment():
    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "attn_implementation": "sdpa",
        "dtype": "float32",
        "mode": "eval, no_grad, sans autocast",
    }


COMPAT_KEYS = ("weights", "config_sha256", "transformers", "torch", "attn_implementation", "dtype")


def source_entry(qdir):
    """Origine d'un golden : empreintes des fichiers de poids **recalculées sur
    disque** (confrontées au manifeste), config et environnement."""
    qdir = Path(qdir)
    manifest = {f["name"]: f["sha256"] for f in json.loads((qdir / "convert_manifest.json").read_text())["files"]}
    weights = {}
    for name in weight_files(qdir):
        weights[name] = sha256(qdir / name)
        if name in manifest and manifest[name] != weights[name]:
            raise SystemExit(f"{name}: sha disque {weights[name]} ≠ manifeste {manifest[name]}")
    return {"weights_dir": str(qdir), "weights": weights, "config_sha256": sha256(qdir / "config.json"),
            **environment()}


def load_provenance(path):
    prov = json.loads(path.read_text()) if path.exists() else {"goldens": {}}
    if "goldens" not in prov:
        raise SystemExit(f"{path}: format de provenance inconnu, dossier à recréer")
    return prov


def check_compatible(out, golden_file, source):
    """Refuse, **avant toute écriture**, un golden d'une autre origine que ceux du dossier."""
    for other, existing in load_provenance(out / "provenance.json")["goldens"].items():
        if other == golden_file:
            continue
        for key in COMPAT_KEYS:
            if existing.get(key) != source.get(key):
                raise SystemExit(f"{out}: {golden_file} et {other} diffèrent sur {key} ; dossier séparé requis")


def write_atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def publish(out, golden_file, tmp, source, run):
    """Publie un golden écrit dans `tmp` : compatibilité revérifiée, empreinte,
    provenance puis golden renommés atomiquement."""
    check_compatible(out, golden_file, source)
    prov = load_provenance(out / "provenance.json")
    prov["goldens"][golden_file] = {**source, "sha256": sha256(tmp), "run": run}
    os.replace(tmp, out / golden_file)
    write_atomic(out / "provenance.json", json.dumps(prov, indent=1, ensure_ascii=False).encode())


def selftest():
    """Refus de mélange avant écriture : les fichiers existants restent intacts."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        a = {"weights": {"x": "1"}, "config_sha256": "c", **environment()}
        for name, data in (("layers.safetensors", b"A-layers"), ("full.safetensors", b"A-full")):
            staged = out / (name + ".new")
            staged.write_bytes(data)
            publish(out, name, staged, a, {})
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        for key, value in (("weights", {"x": "2"}), ("config_sha256", "d"), ("torch", "0")):
            b = {**a, key: value}
            staged = out / "full.safetensors.new"
            staged.write_bytes(b"B-full")
            try:
                publish(out, "full.safetensors", staged, b, {})
            except SystemExit:
                staged.unlink()
            else:
                raise AssertionError(f"mélange accepté sur {key}")
            after = {p.name: p.read_bytes() for p in out.iterdir()}
            assert after == before, f"fichiers modifiés malgré le refus ({key})"
        staged = out / "full.safetensors.new"
        staged.write_bytes(b"A-full-2")
        publish(out, "full.safetensors", staged, a, {})  # même origine : remplacement admis
        assert (out / "full.safetensors").read_bytes() == b"A-full-2"
    print("selftest provenance : ok")


def run_layers(qdir, tokenizer, text, out):
    source = source_entry(qdir)
    check_compatible(out, "layers.safetensors", source)
    tc = C.text_config(qdir)
    weights = C.QWeights(qdir)
    ids_list = Tokenizer.from_file(tokenizer).encode(Path(text).read_text()).ids[:8000]
    assert len(ids_list) == 8000, len(ids_list)
    ids = torch.tensor([ids_list])
    n = ids.shape[1]
    pos, mask = C.rope(tc, n), C.causal_mask(n)
    h = C.embed(tc, weights, ids)
    tensors = {"input_ids": ("I64", (n,), np.array(ids_list, dtype=np.int64).tobytes())}
    for index in range(4):
        started = time.time()
        layer = C.build_layer(tc, index, weights)
        cap = {}
        layer.mlp.gate.register_forward_hook(
            lambda _m, _i, o: cap.update(logits=o[0].clone(), ids=o[2].clone()))
        if layer.layer_type != "linear_attention":
            layer.self_attn.indexer.register_forward_hook(lambda _m, _i, o: cap.update(mask=o.clone()))
        with C.TopkCapture() as topk:
            h = layer(h, position_embeddings=pos, attention_mask=mask, conv_mask=None,
                      past_key_values=None, ple_input_ids=ids)
        tensors[f"layer.{index}"] = ("F32", tuple(h.shape[1:]), h[0].float().numpy().tobytes())
        r = cap["ids"].to(torch.int64)
        tensors[f"routing.{index}"] = ("I64", tuple(r.shape), r.numpy().tobytes())
        lg = cap["logits"].float()
        tensors[f"router_logits.{index}"] = ("F32", tuple(lg.shape), lg.numpy().tobytes())
        if "mask" in cap:
            m = cap["mask"][0, 0].to(torch.uint8)
            tensors[f"qsa_mask.{index}"] = ("U8", tuple(m.shape), m.numpy().tobytes())
            compress = tc.indexer_compress_ratio
            it = iter(topk.indexer)
            rows = [next(it) if (i + 1) // compress > 0 else torch.zeros(0) for i in range(n)]
            assert next(it, None) is None, "scores d'indexeur en surnombre"
            flat, offsets = C.flatten_scores(rows)
            tensors[f"qsa_scores.{index}"] = ("F32", flat.shape, flat.tobytes())
            tensors[f"qsa_offsets.{index}"] = ("I64", offsets.shape, offsets.tobytes())
        del layer
        print(f"couche {index}: {time.time() - started:.0f} s, pic RSS {C.footprint_gb():.1f} Go", flush=True)
    tmp = out / "layers.safetensors.tmp"
    C.write_safetensors(tmp, tensors, {"tokens": n})
    publish(out, "layers.safetensors", tmp, source, {"tokens": n, "text": str(text), "text_sha256": sha256(text)})


def run_full(qdir, tokenizer, out):
    source = source_entry(qdir)
    check_compatible(out, "full.safetensors", source)
    tc = C.text_config(qdir)
    weights = C.QWeights(qdir)
    tok = Tokenizer.from_file(tokenizer)
    ids = [torch.tensor([tok.encode(p).ids]) for p in PROMPTS]
    pos = [C.rope(tc, t.shape[1]) for t in ids]
    masks = [C.causal_mask(t.shape[1]) for t in ids]
    hs = [C.embed(tc, weights, t) for t in ids]
    started = time.time()
    for index in range(tc.num_hidden_layers):
        layer = C.build_layer(tc, index, weights)
        hs = [layer(h, position_embeddings=p, attention_mask=m, conv_mask=None, past_key_values=None,
                    ple_input_ids=t) for h, p, m, t in zip(hs, pos, masks, ids)]
        del layer
        print(f"couche {index}: {time.time() - started:.0f} s, pic RSS {C.footprint_gb():.1f} Go", flush=True)
    mixer = C.M.Qwen4ExpTextGatedResidual(tc, use_combine=False)
    mixer.load_state_dict({k: weights.dense(f"model.language_model.hyper_connection_mixer.{k}")
                           for k in mixer.state_dict()})
    head = weights.dense("lm_head.weight")
    tensors = {}
    for i, (h, t) in enumerate(zip(hs, ids)):
        logits = mixer(h)[0] @ head.T
        tensors[f"prompt.{i}.ids"] = ("I64", (t.shape[1],), t[0].numpy().astype(np.int64).tobytes())
        tensors[f"prompt.{i}.logits"] = ("F32", tuple(logits.shape), logits.float().numpy().tobytes())
    tmp = out / "full.safetensors.tmp"
    C.write_safetensors(tmp, tensors, {"prompts": json.dumps(PROMPTS)})
    publish(out, "full.safetensors", tmp, source, {"prompts": PROMPTS})


def main():
    torch.set_num_threads(os.cpu_count() or 8)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--selftest":
        selftest()
        return
    if mode == "--layers":
        _, _, qdir, tokenizer, text, out = sys.argv
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        run_layers(qdir, tokenizer, text, out)
    elif mode == "--full":
        _, _, qdir, tokenizer, out = sys.argv
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        run_full(qdir, tokenizer, out)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
