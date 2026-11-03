"""Contrôle de qualité des formats : perplexité MLX au protocole de llama-perplexity.

  models/mlxenv/bin/python -I bench/competitors/quality.py mlx-ppl CORPUS
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
from prompts import CHECKPOINT, fnv1a64  # noqa: E402

N_CTX = 512
N_CHUNKS = 32
FIRST = N_CTX // 2


def corpus_ids(corpus):
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    return tok.encode(pathlib.Path(corpus).read_text(), add_special_tokens=False).ids


def mlx_ppl(corpus):
    import mlx.core as mx
    from mlx_lm import load

    ids = corpus_ids(corpus)
    model, _ = load(str(CHECKPOINT))
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
    print(json.dumps({"tool": "mlx-lm forward", "ppl": math.exp(nll / count),
                      "scored_tokens": count, "running_ppl": per_chunk,
                      "corpus_ids_fnv1a64": fnv1a64(ids[:N_CTX * N_CHUNKS])}, indent=1))


def llama_ids(corpus, port):
    from client import http_json

    text = pathlib.Path(corpus).read_text()
    status, body = http_json("127.0.0.1", int(port), "POST", "/tokenize",
                             {"content": text, "add_special": True, "parse_special": False})
    if status != 200:
        raise SystemExit(f"/tokenize: HTTP {status}")
    llama, hf = body["tokens"], corpus_ids(corpus)
    span = N_CTX * N_CHUNKS
    first_diff = next((i for i, (a, b) in enumerate(zip(llama, hf)) if a != b), None)
    print(json.dumps({"llama_tokens": len(llama), "hf_tokens": len(hf),
                      "scored_span_identical": llama[:span] == hf[:span],
                      "first_difference": first_diff,
                      "llama_fnv1a64": fnv1a64(llama[:span]),
                      "hf_fnv1a64": fnv1a64(hf[:span])}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "mlx-ppl":
        mlx_ppl(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "llama-ids":
        llama_ids(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
