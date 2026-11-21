"""KL en régime chat contre les poids officiels bf16.

  python3 -I chat_kl.py ref  --model BF16_OFFICIEL --out ref.json [--prompts sets/selection.json]
  python3 -I chat_kl.py eval --model CANDIDAT_MLX  --ref ref.json [--json par_prompt.json] [--reverse]
  python3 -I chat_kl.py                              # auto-tests stdlib

`ref` : le modèle officiel répond en greedy (192 tokens au plus) à chaque prompt
(PROMPTS par défaut, ou un ensemble figé de prompt_sets.py) ; on garde, par
position de réponse, le top-64 de ses logprobs. Avec --full-vocab N, les N
premiers prompts gardent aussi la distribution complète (`.npy` f32 à côté).
`eval` : KL sur la partition top-64 + reste, accord top-1 et NLL du texte
forcé ; --json écrit, par prompt, KL moyenne, p99, max, top-1, NLL et, si
disponible, la KL exacte sur le vocabulaire complet. --reverse évalue les
prompts en ordre inverse (contrôle de répétabilité). Pour un modèle instruct,
ce régime remplace le texte brut, que Gemma 4 ne sait pas prédire.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PROMPTS = [
    "Explain how a refrigerator works, in two paragraphs.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main causes of the French Revolution?",
    "Write a Python function that checks whether a string is a palindrome, with comments.",
    "Summarize the plot of Romeo and Juliet.",
    "Give me five tips to improve my sleep, with a sentence of explanation each.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Describe the water cycle step by step.",
]
TOP_K, MAX_TOKENS = 64, 192


def partition_kl(p_logprobs, q_logprobs):
    """KL(p ‖ q) sur les ids de référence + masse restante."""
    p = [math.exp(v) for v in p_logprobs]
    q = [math.exp(v) for v in q_logprobs]
    pairs = list(zip(p, q)) + [(max(1 - sum(p), 0.0), max(1 - sum(q), 0.0))]
    return sum(a * math.log(a / max(b, 1e-300)) for a, b in pairs if a > 0)


def quantile(values, q):
    """Rang le plus proche, sans interpolation : reproductible et défini dès 1 valeur."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def prompt_stats(kls, same, nll):
    n = len(kls)
    return dict(positions=n, kl_mean=sum(kls) / n, kl_p99=quantile(kls, 0.99), kl_max=max(kls),
                top1=sum(same) / n, nll=sum(nll) / n)


def load_cases(prompts):
    if prompts is None:
        return [dict(id=f"hist-{i + 1}", category="code" if "Python" in p else "prose", prompt=p)
                for i, p in enumerate(PROMPTS)], None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompt_sets import load
    return load(prompts), hashlib.sha256(Path(prompts).read_bytes()).hexdigest()


def reference(model_dir, out, prompts=None, full_vocab=0, min_positions=0):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"destination existante refusée : {out}")
    entries, prompts_sha = load_cases(prompts)
    tok = AutoTokenizer.from_pretrained(model_dir)
    # Sans accelerate : chargement CPU puis copie sur le GPU.
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("mps").eval()
    cases, short = [], []
    with torch.no_grad():
        for i, entry in enumerate(entries):
            x = tok.apply_chat_template([{"role": "user", "content": entry["prompt"]}], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)["input_ids"].to("mps")
            seq = model.generate(x, max_new_tokens=MAX_TOKENS, do_sample=False)[0]
            lp = torch.log_softmax(model(seq[None]).logits[0].float(), -1)[x.shape[1] - 1:len(seq) - 1]
            values, ids = lp.topk(TOP_K, -1)
            case = dict(id=entry["id"], category=entry["category"], prompt_len=x.shape[1], ids=seq.tolist(),
                        top_ids=ids.cpu().tolist(), top_lp=values.cpu().tolist())
            if i < full_vocab:
                path = out.with_name(f"{out.stem}.{entry['id']}.npy")
                np.save(path, lp.cpu().numpy().astype(np.float32))
                case["full_vocab"] = path.name
                case["full_vocab_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            cases.append(case)
            if len(case["top_ids"]) < min_positions:
                short.append((entry["id"], len(case["top_ids"])))
            print(entry["id"], len(case["top_ids"]), "positions", flush=True)
    out.write_text(json.dumps(dict(prompts_sha256=prompts_sha, model=str(model_dir), cases=cases)))
    print("écrit", out, hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)
    if short:
        raise SystemExit(f"prompts sous {min_positions} positions, à remplacer avant toute mesure : {short}")


def evaluate(model_dir, ref, json_out=None, reverse=False):
    import mlx.core as mx
    from mlx_lm import load

    if json_out is not None and Path(json_out).exists():
        raise FileExistsError(f"destination existante refusée : {json_out}")
    ref = Path(ref)
    data = json.loads(ref.read_text())
    # Ancien format : une liste de cas sans identifiants.
    cases = data["cases"] if isinstance(data, dict) else [dict(c, id=f"hist-{i + 1}", category="prose")
                                                          for i, c in enumerate(data)]
    model, _ = load(model_dir)
    per_prompt = {}
    for case in (reversed(cases) if reverse else cases):
        ids, start = case["ids"], case["prompt_len"]
        logits = model(mx.array([ids]))[0, start - 1:len(ids) - 1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, -1, keepdims=True)
        mx.eval(lp)
        kls, same, nll = [], [], []
        for row, (top, values) in enumerate(zip(case["top_ids"], case["top_lp"])):
            kls.append(partition_kl(values, lp[row][mx.array(top)].tolist()))
            same.append(int(mx.argmax(lp[row]).item()) == top[0])
            nll.append(-lp[row][ids[start + row]].item())
        stats = dict(prompt_stats(kls, same, nll), category=case["category"])
        if "full_vocab" in case:
            import numpy as np
            full = mx.array(np.load(ref.with_name(case["full_vocab"])))
            exact = (mx.exp(full) * (full - lp)).sum(-1)
            stats["kl_full_vocab_mean"] = exact.mean().item()
        per_prompt[case["id"]] = stats
        print(case["id"], {k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()}, flush=True)
        del logits, lp
    total = sum(s["positions"] for s in per_prompt.values())
    summary = dict(positions=total,
                   kl_positions=sum(s["kl_mean"] * s["positions"] for s in per_prompt.values()) / total,
                   kl_prompts=sum(s["kl_mean"] for s in per_prompt.values()) / len(per_prompt),
                   top1=sum(s["top1"] * s["positions"] for s in per_prompt.values()) / total,
                   ppl=math.exp(sum(s["nll"] * s["positions"] for s in per_prompt.values()) / total))
    print(f"positions {total}  KL {summary['kl_positions']:.4f}  même top-1 {summary['top1']:.2%}"
          f"  PPL {summary['ppl']:.3f}")
    if json_out is not None:
        Path(json_out).write_text(json.dumps(dict(
            model=str(model_dir), ref=str(ref), ref_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
            reverse=reverse, summary=summary, prompts=per_prompt), indent=1) + "\n")


def _self_check():
    assert abs(partition_kl([math.log(0.5), math.log(0.5)], [math.log(0.5), math.log(0.5)])) < 1e-12
    assert partition_kl([math.log(0.9)], [math.log(0.5)]) > 0
    assert quantile([3, 1, 2], 0.99) == 3 and quantile([5], 0.99) == 5
    assert quantile(list(range(1, 201)), 0.99) == 198
    s = prompt_stats([0.1, 0.3], [True, False], [1.0, 2.0])
    assert s == dict(positions=2, kl_mean=0.2, kl_p99=0.3, kl_max=0.3, top1=0.5, nll=1.5)
    entries, sha = load_cases(None)
    assert len(entries) == 8 and sha is None and entries[3]["category"] == "code"
    assert "mlx.core" not in sys.modules


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("chat_kl: auto-tests stdlib ok")
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("ref", "eval"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out")
    parser.add_argument("--ref")
    parser.add_argument("--prompts", type=Path, help="ensemble figé (prompt_sets.py) ; défaut : PROMPTS")
    parser.add_argument("--full-vocab", type=int, default=0, help="ref : distribution complète des N premiers prompts")
    parser.add_argument("--min-positions", type=int, default=0, help="ref : échec si un prompt en a moins")
    parser.add_argument("--json", help="eval : résultats par prompt")
    parser.add_argument("--reverse", action="store_true", help="eval : prompts en ordre inverse")
    args = parser.parse_args()
    if args.action == "ref":
        reference(args.model, args.out or parser.error("--out requis"), args.prompts, args.full_vocab,
                  args.min_positions)
    else:
        evaluate(args.model, args.ref or parser.error("--ref requis"), args.json, args.reverse)


if __name__ == "__main__":
    main()
