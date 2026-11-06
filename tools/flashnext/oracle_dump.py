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
  --long   : modèle complet sur 4096 tokens (QSA creuse au-delà de 2051) :
             routages, scores et masques de l'indexeur, logits à une position sur 16
             et aux 64 dernières.
  --mtp    : couche MTP (absente de transformers) sur les paires (h_t, tok_{t+1})
             du golden long : h_t recalculé par la couche 47 officielle depuis
             `qsa_input.47`, entrée fc transcrite de `graph_mtp` (llama.cpp),
             normes et couche QSA + MoE officielles (poids `mtp.*`), mixer de tête
             officiel, logits aux positions de `long_positions`.

Usage :
  python -I tools/flashnext/oracle_dump.py --selftest
  python -I tools/flashnext/oracle_dump.py --layers <q4e8> <tokenizer.json> <texte> <sortie>
  python -I tools/flashnext/oracle_dump.py --full   <q4e8> <tokenizer.json> <sortie>
  python -I tools/flashnext/oracle_dump.py --long   <q4e8> <tokenizer.json> <texte> <sortie>
  python -I tools/flashnext/oracle_dump.py --mtp    <q4e8> <sortie contenant long.safetensors>
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
    names += ["ngram/meta.json", C.ngram_table(meta)]
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
        # `ngram/meta.json` est absent du manifeste du convertisseur ; tout autre poids y figure.
        if name == "ngram/meta.json":
            continue
        if manifest.get(name) != weights[name]:
            raise SystemExit(f"{name}: sha disque {weights[name]} ≠ manifeste {manifest.get(name)}")
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


def publish(out, golden_file, tmp, source, run):
    """Publie un golden écrit dans `tmp`. Golden et provenance sont préparés en
    `.tmp` et vérifiés ; l'ancien golden est mis de côté puis le nouveau et la
    provenance sont renommés, la provenance en dernier. Sur échec, l'ancien état
    est restauré : un golden publié a toujours une provenance cohérente."""
    prov_path, golden = out / "provenance.json", out / golden_file
    prov_tmp, old = out / "provenance.json.tmp", out / (golden_file + ".old")
    try:
        check_compatible(out, golden_file, source)
        prov = load_provenance(prov_path)
        prov["goldens"][golden_file] = {**source, "sha256": sha256(tmp), "run": run}
        prov_tmp.write_bytes(json.dumps(prov, indent=1, ensure_ascii=False).encode())
        if json.loads(prov_tmp.read_text())["goldens"][golden_file]["sha256"] != sha256(tmp):
            raise SystemExit(f"{prov_tmp}: provenance relue incohérente")
        if golden.exists():
            os.replace(golden, old)
        try:
            os.replace(tmp, golden)
            os.replace(prov_tmp, prov_path)
        except BaseException:
            if old.exists():
                os.replace(old, golden)
            else:
                golden.unlink(missing_ok=True)
            raise
        old.unlink(missing_ok=True)
    finally:
        prov_tmp.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def selftest():
    """Refus de mélange avant écriture, échec de publication sans golden orphelin,
    chemins n-gram hors de `ngram/` refusés."""
    import tempfile
    from unittest import mock

    for bad in ("/external/table.bin", "../table.bin", "a/../../t.bin", ""):
        try:
            C.ngram_table({"file": bad})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"chemin n-gram {bad!r} accepté")
    assert C.ngram_table({"file": "ngram_e4m3.bin"}) == "ngram/ngram_e4m3.bin"

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        a = {"weights": {"x": "1"}, "config_sha256": "c", **environment()}

        def stage(name, data):
            staged = out / (name + ".new")
            staged.write_bytes(data)
            return staged

        for name, data in (("layers.safetensors", b"A-layers"), ("full.safetensors", b"A-full")):
            publish(out, name, stage(name, data), a, {})
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        for key, value in (("weights", {"x": "2"}), ("config_sha256", "d"), ("torch", "0")):
            b = {**a, key: value}
            try:
                publish(out, "full.safetensors", stage("full.safetensors", b"B-full"), b, {})
            except SystemExit:
                pass
            else:
                raise AssertionError(f"mélange accepté sur {key}")
            after = {p.name: p.read_bytes() for p in out.iterdir()}
            assert after == before, f"fichiers modifiés malgré le refus ({key})"

        # Échec à chaque étape d'écriture : l'état publié reste l'ancien, sans résidu.
        real_replace = os.replace
        for fail_at in range(3):
            calls = []

            def flaky(src, dst, fail_at=fail_at, calls=calls):
                calls.append(dst)
                if len(calls) == fail_at + 1:
                    raise OSError("échec simulé")
                real_replace(src, dst)

            with mock.patch.object(os, "replace", flaky):
                try:
                    publish(out, "full.safetensors", stage("full.safetensors", b"A-full-x"), a, {})
                except OSError:
                    pass
                else:
                    raise AssertionError(f"échec simulé {fail_at} ignoré")
            after = {p.name: p.read_bytes() for p in out.iterdir()}
            assert after == before, f"état modifié après l'échec {fail_at}"
        staged = stage("full.safetensors", b"A-full-y")
        with mock.patch.object(Path, "write_bytes", side_effect=OSError("disque plein")):
            try:
                publish(out, "full.safetensors", staged, a, {})
            except OSError:
                pass
            else:
                raise AssertionError("échec d'écriture de la provenance ignoré")
        assert {p.name: p.read_bytes() for p in out.iterdir()} == before

        publish(out, "full.safetensors", stage("full.safetensors", b"A-full-2"), a, {})  # même origine : admis
        assert (out / "full.safetensors").read_bytes() == b"A-full-2"
        prov = json.loads((out / "provenance.json").read_text())
        assert prov["goldens"]["full.safetensors"]["sha256"] == sha256(out / "full.safetensors")
        assert sorted(p.name for p in out.iterdir()) == ["full.safetensors", "layers.safetensors", "provenance.json"]
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


# Positions dont `--long` fige les logits : une sur 16, plus les 64 dernières (les
# logits complets pèseraient 4 Go à 4096 tokens).
LONG_TOKENS = 4096


def long_positions(n):
    return sorted(set(range(0, n, 16)) | set(range(max(0, n - 64), n)))


def run_long(qdir, tokenizer, text, out):
    """Modèle complet sur les 4096 premiers tokens du texte : routages et logits de
    routeur de chaque couche, masque et scores de l'indexeur des couches QSA
    (sélection creuse au-delà de 2051 tokens), logits finaux aux positions de
    `long_positions`. Aux couches QSA, résiduel d'entrée complet et sortie aux
    positions de `long_positions` : l'oracle Rust ré-ancre chaque couche QSA sur
    l'entrée torch, sans dérive accumulée."""
    source = source_entry(qdir)
    check_compatible(out, "long.safetensors", source)
    tc = C.text_config(qdir)
    weights = C.QWeights(qdir)
    ids_list = Tokenizer.from_file(tokenizer).encode(Path(text).read_text()).ids[:LONG_TOKENS]
    assert len(ids_list) == LONG_TOKENS, len(ids_list)
    ids = torch.tensor([ids_list])
    n = ids.shape[1]
    pos, mask = C.rope(tc, n), C.causal_mask(n)
    h = C.embed(tc, weights, ids)
    tensors = {"input_ids": ("I64", (n,), np.array(ids_list, dtype=np.int64).tobytes())}
    keep = torch.tensor(long_positions(n))
    started = time.time()
    for index in range(tc.num_hidden_layers):
        layer = C.build_layer(tc, index, weights)
        cap = {}
        layer.mlp.gate.register_forward_hook(
            lambda _m, _i, o: cap.update(logits=o[0].clone(), ids=o[2].clone()))
        qsa = layer.layer_type != "linear_attention"
        if qsa:
            layer.self_attn.indexer.register_forward_hook(lambda _m, _i, o: cap.update(mask=o.clone()))
            x = h[0].float()
            tensors[f"qsa_input.{index}"] = ("F32", tuple(x.shape), x.numpy().tobytes())
        with C.TopkCapture() as topk:
            h = layer(h, position_embeddings=pos, attention_mask=mask, conv_mask=None,
                      past_key_values=None, ple_input_ids=ids)
        if qsa:
            y = h[0][keep].float()
            tensors[f"qsa_output.{index}"] = ("F32", tuple(y.shape), y.numpy().tobytes())
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
    mixer = C.M.Qwen4ExpTextGatedResidual(tc, use_combine=False)
    mixer.load_state_dict({k: weights.dense(f"model.language_model.hyper_connection_mixer.{k}")
                           for k in mixer.state_dict()})
    head = weights.dense("lm_head.weight")
    logits = mixer(h)[0][keep] @ head.T
    tensors["logit_positions"] = ("I64", tuple(keep.shape), keep.numpy().astype(np.int64).tobytes())
    tensors["logits"] = ("F32", tuple(logits.shape), logits.float().numpy().tobytes())
    tmp = out / "long.safetensors.tmp"
    C.write_safetensors(tmp, tensors, {"tokens": n})
    publish(out, "long.safetensors", tmp, source,
            {"tokens": n, "text": str(text), "text_sha256": sha256(text), "logit_positions": "16k + 64 dernières",
             "qsa_anchors": "qsa_input complet, qsa_output aux logit_positions"})


def run_mtp(qdir, out):
    """Couche MTP sur les 4095 paires `(h_t, tok_{t+1})` du golden long. Ligne `t`
    de l'historique MTP à la position RoPE `t` (RoPE relative : un décalage
    constant ne change l'attention qu'à l'arrondi près). Sont figés : `h` et les
    tokens d'entrée, l'entrée fc et la sortie hc aux positions, routages, masque
    et scores de l'indexeur, logits de tête aux positions."""
    from safetensors import safe_open

    source = source_entry(qdir)
    check_compatible(out, "mtp_golden.safetensors", source)
    tc = C.text_config(qdir)
    weights = C.QWeights(qdir)
    last = tc.num_hidden_layers - 1
    # La couche MTP est construite comme la couche `last` : QSA, sans PLE.
    assert tc.layer_types[last] == "full_attention" and last + 1 not in tc.ple_layer_ids, \
        "la couche de référence de la MTP doit être QSA et sans PLE"
    recorded = load_provenance(out / "provenance.json")["goldens"].get("long.safetensors", {})
    long_sha = sha256(out / "long.safetensors")
    if recorded.get("sha256") != long_sha:
        raise SystemExit(f"{out}/long.safetensors : empreinte {long_sha} ≠ provenance")
    with safe_open(out / "long.safetensors", framework="np") as golden:
        ids_list = golden.get_tensor("input_ids").tolist()
        x = torch.from_numpy(golden.get_tensor(f"qsa_input.{last}")).float()[None]
    n = len(ids_list)
    ids = torch.tensor([ids_list])
    started = time.time()
    layer = C.build_layer(tc, last, weights)
    with C.TopkCapture():
        h = layer(x, position_embeddings=C.rope(tc, n), attention_mask=C.causal_mask(n),
                  conv_mask=None, past_key_values=None, ple_input_ids=ids)[0]
    del layer
    print(f"couche {last} recalculée : {time.time() - started:.0f} s", flush=True)
    m = n - 1
    h_in, next_ids = h[:m], ids[0, 1:]
    hidden, hc = tc.hidden_size, tc.hc_count
    e_norm = C.M.Qwen4ExpTextRMSNorm(hidden, eps=tc.rms_norm_eps)
    e_norm.weight.data = weights.dense("mtp.pre_fc_norm_embedding.weight").float()
    h_norm = C.M.Qwen4ExpTextRMSNorm(hc * hidden, group_size=hidden, eps=tc.rms_norm_eps)
    h_norm.weight.data = weights.dense("mtp.pre_fc_norm_hidden.weight").float()
    fc_e, fc_h = weights.dense("mtp.fc_embedding.weight"), weights.dense("mtp.fc_hidden.weight")
    e = e_norm(weights.rows("model.language_model.embed_tokens.weight", next_ids.numpy())) @ fc_e.T
    hs = h_norm(h_in).reshape(m, hc, hidden) @ fc_h.T
    res = (hs + e[:, None, :]).reshape(1, m, hc * hidden)
    # Couche de type QSA sans PLE : l'indice `last` fixe le type, `prefix` les poids.
    layer = C.build_layer(tc, last, weights, prefix="mtp.layers.0.")
    cap = {}
    layer.mlp.gate.register_forward_hook(
        lambda _m, _i, o: cap.update(logits=o[0].clone(), ids=o[2].clone()))
    layer.self_attn.indexer.register_forward_hook(lambda _m, _i, o: cap.update(mask=o.clone()))
    with C.TopkCapture() as topk:
        y = layer(res, position_embeddings=C.rope(tc, m), attention_mask=C.causal_mask(m),
                  conv_mask=None, past_key_values=None, ple_input_ids=None)
    del layer
    print(f"couche MTP : {time.time() - started:.0f} s, pic RSS {C.footprint_gb():.1f} Go", flush=True)
    keep = torch.tensor(long_positions(m))
    tensors = {
        "mtp_h": ("F32", tuple(h_in.shape), h_in.contiguous().numpy().tobytes()),
        "mtp_next_ids": ("I64", (m,), next_ids.numpy().astype(np.int64).tobytes()),
        "mtp_positions": ("I64", tuple(keep.shape), keep.numpy().astype(np.int64).tobytes()),
    }
    xin, yout = res[0][keep], y[0][keep]
    tensors["mtp_input"] = ("F32", tuple(xin.shape), xin.contiguous().numpy().tobytes())
    tensors["mtp_output"] = ("F32", tuple(yout.shape), yout.contiguous().numpy().tobytes())
    r = cap["ids"].to(torch.int64)
    tensors["routing.0"] = ("I64", tuple(r.shape), r.numpy().tobytes())
    lg = cap["logits"].float()
    tensors["router_logits.0"] = ("F32", tuple(lg.shape), lg.numpy().tobytes())
    mask = cap["mask"][0, 0].to(torch.uint8)
    tensors["qsa_mask.0"] = ("U8", tuple(mask.shape), mask.numpy().tobytes())
    it = iter(topk.indexer)
    rows = [next(it) if (i + 1) // tc.indexer_compress_ratio > 0 else torch.zeros(0) for i in range(m)]
    assert next(it, None) is None, "scores d'indexeur en surnombre"
    flat, offsets = C.flatten_scores(rows)
    tensors["qsa_scores.0"] = ("F32", flat.shape, flat.tobytes())
    tensors["qsa_offsets.0"] = ("I64", offsets.shape, offsets.tobytes())
    mixer = C.M.Qwen4ExpTextGatedResidual(tc, use_combine=False)
    mixer.load_state_dict({k: weights.dense(f"mtp.hyper_connection_mixer.{k}") for k in mixer.state_dict()})
    logits = mixer(y)[0][keep] @ weights.dense("lm_head.weight").T
    tensors["logits"] = ("F32", tuple(logits.shape), logits.float().numpy().tobytes())
    tmp = out / "mtp_golden.safetensors.tmp"
    C.write_safetensors(tmp, tensors, {"pairs": m})
    publish(out, "mtp_golden.safetensors", tmp, source,
            {"pairs": m, "long_sha256": long_sha,
             "positions": "ligne t à la position RoPE t ; logits 16k + 64 dernières"})


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
    elif mode == "--long":
        _, _, qdir, tokenizer, text, out = sys.argv
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        run_long(qdir, tokenizer, text, out)
    elif mode == "--mtp":
        _, _, qdir, out = sys.argv
        run_mtp(qdir, Path(out))
    elif mode == "--full":
        _, _, qdir, tokenizer, out = sys.argv
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        run_full(qdir, tokenizer, out)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
