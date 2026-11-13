"""KL sur corpus, à la llama-perplexity --kl-divergence.

Référence mlx-lm float32 sur kld/corpus.txt (domaine public : Project
Gutenberg n°1661, extrait de « A Scandal in Bohemia »). 8 morceaux de 512
tokens ; les 256 dernières positions de chaque morceau sont notées. Chaque
position garde le top-64 float32 et la masse restante. Le même script mesure
ensuite mlx-lm bf16 contre cette référence : c'est le bruit propre de la
référence, dénominateur du ratio vérifié en Rust.

Lancer sous le verrou GPU, SARAGOSSA_GOLDEN_CAPTURE=1, sans téléchargement.
"""
import argparse
import gc
import math
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LOGITS_PACKAGES, OUT, local_model, provenance, require_versions, write, write_json

CHUNKS, CTX, SCORED, K = 8, 512, 256, 64


def chunk_ids(tokenizer):
    ids = tokenizer.encode((OUT / "kld/corpus.txt").read_text(), add_special_tokens=False)
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    width = CTX - len(bos)
    if len(ids) < CHUNKS * width:
        raise RuntimeError(f"corpus trop court : {len(ids)} tokens")
    return [bos + ids[i * width:(i + 1) * width] for i in range(CHUNKS)]


def logprobs(mx, model, chunk):
    # La position t prédit le token t + 1 : on note les SCORED derniers tokens.
    logits = model(mx.array([chunk]))[0, CTX - SCORED - 1:CTX - 1].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def stats(kls, deltas, same):
    deltas = sorted(deltas)
    return dict(positions=len(kls), mean_kl=sum(kls) / len(kls),
                p99_delta_p=deltas[int(0.99 * (len(deltas) - 1))], max_delta_p=deltas[-1],
                same_top=sum(same) / len(same))


def main():
    require_versions(LOGITS_PACKAGES)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen38", "qwen36_moe", "gemma3", "gemma4"], required=True)
    args = parser.parse_args()
    path = local_model(args.model)
    import mlx.core as mx
    from mlx_lm import load
    mx.set_default_device(mx.gpu)

    model, tokenizer = load(path)
    chunks = chunk_ids(tokenizer)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    reference = []  # par position : (ids top-K, logprobs top-K)
    for chunk in chunks:
        lp = logprobs(mx, model, chunk)
        top = mx.argsort(-lp, axis=-1)[:, :K]
        values = mx.take_along_axis(lp, top, axis=-1)
        mx.eval(top, values)
        reference += zip(top.tolist(), values.tolist())
    del model
    gc.collect()
    mx.clear_cache()

    model, _ = load(path)  # dtype natif du checkpoint, bf16 pour ces modèles
    kls, deltas, same = [], [], []
    for index, chunk in enumerate(chunks):
        lp = logprobs(mx, model, chunk)
        argmax = mx.argmax(lp, axis=-1)
        mx.eval(lp, argmax)
        for row in range(SCORED):
            ids, ref = reference[index * SCORED + row]
            p = [math.exp(v) for v in ref]
            q = [math.exp(v) for v in lp[row][mx.array(ids)].tolist()]
            pairs = list(zip(p, q)) + [(max(1 - sum(p), 0.0), max(1 - sum(q), 0.0))]
            kls.append(sum(a * math.log(a / max(b, 1e-300)) for a, b in pairs if a > 0))
            deltas.append(max(abs(a - b) for a, b in pairs))
            same.append(int(argmax[row].item()) == ids[0])
    noise = stats(kls, deltas, same)
    print(args.model, "mlx bf16 / f32 :", noise)

    source = provenance(args.model, path, LOGITS_PACKAGES)
    source["float32_control"] = True
    flat = [x for ids, values in reference for x in (*map(float, ids), *values)]
    write("kld", args.model, struct.pack(f"<{len(flat)}f", *flat), "f32",
          [CHUNKS * SCORED, 2, K], source, "ratio au bruit mlx bf16 / f32, voir README",
          layout="par position : K ids (entiers exacts en f32) puis K logprobs float32")
    write_json("kld", f"{args.model}_meta", dict(model=args.model, chunks=chunks, scored=SCORED,
               top_k=K, mlx_bf16_noise=noise), source, "ratio au bruit mlx bf16 / f32, voir README")


if __name__ == "__main__":
    main()
