"""GPU reference: mlx-lm 0.32.2, transformers 5.22.2.
pip install mlx-lm==0.32.2 transformers==5.22.2

Run explicitly under the operator's GPU lock; never downloads models.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import local_model, provenance, write_json

PROMPTS = ["Bonjour, je m'appelle", "The capital of France is", "2 + 2 =",
           "def square(x):\n    return", "Translate to English: le chat dort.",
           "Explique la gravité en une phrase."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen38", "qwen36_moe", "gemma3", "gemma4"], required=True)
    args = parser.parse_args()
    path = local_model(args.model)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_default_device(mx.gpu)
    model, tokenizer = load(path)
    cases = []
    for index, prompt in enumerate(PROMPTS):
        chat = index == len(PROMPTS) - 1
        rendered = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) if chat else prompt
        ids = tokenizer.encode(rendered, add_special_tokens=not chat)
        cache = make_prompt_cache(model)
        inputs = mx.array([ids])
        steps = []
        for _ in range(16):
            logits = model(inputs, cache=cache)[:, -1, :].astype(mx.float32)[0]
            logprobs = logits - mx.logsumexp(logits)
            top = mx.argsort(logprobs)[-20:][::-1]
            mx.eval(top, logprobs)
            top_ids = top.tolist()
            generated = int(mx.argmax(logits).item())
            pairs = list(zip(top_ids, logprobs[top].tolist()))
            # Preserve argmax's tie decision even if argsort picks another tied ID.
            if generated not in top_ids:
                pairs[-1] = (generated, float(logprobs[generated].item()))
            pairs.sort(key=lambda pair: (-pair[1], pair[0] != generated, pair[0]))
            steps.append(dict(token=generated, top_ids=[p[0] for p in pairs],
                              logprobs=[p[1] for p in pairs]))
            inputs = mx.array([[generated]])
        cases.append(dict(prompt=prompt, rendered=rendered, chat=chat, prompt_ids=ids, steps=steps))
    write_json("logits", args.model, dict(model=args.model, cases=cases),
               provenance(args.model, path, ["mlx-lm", "mlx", "transformers"]),
               "greedy exact; near-tie logprob margin <= 0.01; top20+other KL <= 0.02; max delta-p <= 0.02",
               decoding="16 steps including EOS; no sampling or logits processors; reference-prefix teacher forcing")


if __name__ == "__main__":
    main()
