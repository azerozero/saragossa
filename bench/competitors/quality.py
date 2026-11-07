"""Contrôle de qualité des formats : perplexité MLX au protocole de llama-perplexity.

  models/mlxenv/bin/python -I bench/competitors/quality.py mlx-ppl CORPUS [CHECKPOINT]
  models/mlxenv/bin/python -I bench/competitors/quality.py llama-ids CORPUS PORT

Protocole commun (identique à `llama-perplexity -c 512 --chunks 32`) :
corpus wikitext-2-raw `wiki.test.raw` tokenisé d'un bloc, sans BOS (Qwen
n'en ajoute pas : `tokenizer.ggml.add_bos_token` à relever dans le GGUF) ;
fenêtres contiguës non chevauchantes de 512 tokens, les 32 premières ; dans
chaque fenêtre, seuls les tokens 257 à 511 sont notés (la position j prédit
le token j + 1 pour j de 256 à 510, soit 255 tokens), avec le contexte de la
fenêtre seul. PPL = exp(somme des log-vraisemblances négatives / 8160).

`llama-ids` compare les ids du corpus selon llama-server (`/tokenize`, GGUF
chargé sur PORT) à ceux du tokenizer HF : la PPL n'est comparée entre outils
que si les ids sont identiques.
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prompts import CHECKPOINT, fnv1a64, sha256_bytes  # noqa: E402

# sha256 publié de wiki.test.raw (wikitext-2-raw-v1.zip, sha256 ef7edb56…5a11).
WIKITEXT_TEST_SHA256 = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"

N_CTX = 512
N_CHUNKS = 32
FIRST = N_CTX // 2


def corpus_text(corpus):
    data = pathlib.Path(corpus).read_bytes()
    if sha256_bytes(data) != WIKITEXT_TEST_SHA256:
        raise SystemExit(f"{corpus}: sha256 inattendu, corpus de perplexité refusé")
    return data.decode("utf-8")


def corpus_ids(corpus):
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    return tok.encode(corpus_text(corpus), add_special_tokens=False).ids


def mlx_ppl(corpus, checkpoint=CHECKPOINT):
    import mlx.core as mx
    from mlx_lm import load

    ids = corpus_ids(corpus)
    model, _ = load(str(checkpoint))
    nll, count, per_chunk = 0.0, 0, []
    for chunk in range(N_CHUNKS):
        window = ids[chunk * N_CTX:(chunk + 1) * N_CTX]
        logits = model(mx.array([window]))[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(window[FIRST + 1:])
        picked = mx.take_along_axis(logprobs[FIRST:N_CTX - 1], targets[:, None], axis=-1)
        chunk_nll = -float(picked.sum().item())
        nll += chunk_nll
        count += N_CTX - FIRST - 1
        per_chunk.append(math.exp(nll / count))
    print(json.dumps({"tool": "mlx-lm forward", "checkpoint": pathlib.Path(checkpoint).name,
                      "ppl": math.exp(nll / count),
                      "scored_tokens": count, "running_ppl": per_chunk,
                      "corpus_ids_fnv1a64": fnv1a64(ids[:N_CTX * N_CHUNKS])}, indent=1))


def llama_ids(corpus, port):
    from client import http_json

    text = corpus_text(corpus)
    status, body = http_json("127.0.0.1", int(port), "POST", "/tokenize",
                             {"content": text, "add_special": True, "parse_special": False})
    if status != 200:
        raise SystemExit(f"/tokenize: HTTP {status}")
    llama, hf = body["tokens"], corpus_ids(corpus)
    span = N_CTX * N_CHUNKS
    first_diff = next((i for i, (a, b) in enumerate(zip(llama, hf)) if a != b), None)
    identical = llama[:span] == hf[:span]
    print(json.dumps({"llama_tokens": len(llama), "hf_tokens": len(hf),
                      "scored_span_identical": identical,
                      "first_difference": first_diff,
                      "llama_fnv1a64": fnv1a64(llama[:span]),
                      "hf_fnv1a64": fnv1a64(hf[:span])}, indent=1))
    # Bloquant : sans ids identiques, les PPL ne sont pas comparées comme telles.
    if not identical:
        raise SystemExit("ids du corpus différents entre llama.cpp et le tokenizer HF")


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "mlx-ppl":
        mlx_ppl(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "llama-ids":
        llama_ids(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
