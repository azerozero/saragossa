"""KL en régime chat contre les poids officiels bf16.

  python3 -I chat_kl.py ref  --model BF16_OFFICIEL --out ref.json   # transformers + torch
  python3 -I chat_kl.py eval --model CANDIDAT_MLX  --ref ref.json   # mlx-lm

`ref` : le modèle officiel répond en greedy (192 tokens au plus) à PROMPTS ; on
garde, par position de réponse, le top-64 de ses logprobs. `eval` : KL sur la
partition top-64 + reste, accord top-1 et perplexité des réponses, en teacher
forcing. Pour un modèle instruct, ce régime remplace le texte brut, que Gemma 4
ne sait pas prédire.
"""
import argparse
import json
import math

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


def reference(model_dir, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    # Sans accelerate : chargement CPU puis copie sur le GPU.
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("mps").eval()
    cases = []
    with torch.no_grad():
        for prompt in PROMPTS:
            x = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)["input_ids"].to("mps")
            seq = model.generate(x, max_new_tokens=MAX_TOKENS, do_sample=False)[0]
            lp = torch.log_softmax(model(seq[None]).logits[0].float(), -1)[x.shape[1] - 1:len(seq) - 1]
            values, ids = lp.topk(TOP_K, -1)
            cases.append(dict(prompt_len=x.shape[1], ids=seq.tolist(), top_ids=ids.cpu().tolist(),
                              top_lp=values.cpu().tolist()))
    with open(out, "w") as f:
        json.dump(cases, f)


def evaluate(model_dir, ref):
    import mlx.core as mx
    from mlx_lm import load

    model, _ = load(model_dir)
    kls, same, nll = [], [], []
    with open(ref) as f:
        cases = json.load(f)
    for case in cases:
        ids, start = case["ids"], case["prompt_len"]
        logits = model(mx.array([ids]))[0, start - 1:len(ids) - 1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, -1, keepdims=True)
        mx.eval(lp)
        for row, (top, values) in enumerate(zip(case["top_ids"], case["top_lp"])):
            kls.append(partition_kl(values, lp[row][mx.array(top)].tolist()))
            same.append(int(mx.argmax(lp[row]).item()) == top[0])
            nll.append(-lp[row][ids[start + row]].item())
    print(f"positions {len(kls)}  KL {sum(kls) / len(kls):.4f}  même top-1 {sum(same) / len(same):.2%}"
          f"  PPL {math.exp(sum(nll) / len(nll)):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("ref", "eval"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out")
    parser.add_argument("--ref")
    args = parser.parse_args()
    if args.action == "ref":
        reference(args.model, args.out or parser.error("--out requis"))
    else:
        evaluate(args.model, args.ref or parser.error("--ref requis"))


if __name__ == "__main__":
    main()
