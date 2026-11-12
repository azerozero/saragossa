"""Bruit propre de la référence : capture mlx bf16 contre capture float32.

Usage : python3 -I tools/goldens/reference_noise.py qwen36_moe gemma4
(depuis tests/fixtures/goldens/logits). Sert à calibrer MAX_DELTA_P.
"""
import json, math, sys
def load(p): return json.load(open(p))
for name in sys.argv[1:]:
    a, b = load(f"{name}.bin"), load(f"{name}_f32.bin")
    dps, kls, n, missing = [], [], 0, 0
    for ca, cb in zip(a["cases"], b["cases"]):
        assert ca["prompt_ids"] == cb["prompt_ids"]
        for sa, sb in zip(ca["steps"], cb["steps"]):
            qmap = {i: math.exp(l) for i, l in zip(sb["top_ids"], sb["logprobs"])}
            floor = math.exp(sb["logprobs"][-1])
            p = [math.exp(l) for l in sa["logprobs"]]
            q = [qmap.get(i) for i in sa["top_ids"]]
            missing += sum(x is None for x in q)
            # id absent du top-20 f32 : q borné par la 20e probabilité f32, on prend la borne.
            q = [floor if x is None else x for x in q]
            pairs = list(zip(p, q)) + [(max(1 - sum(p), 0), max(1 - sum(q), 0))]
            dps.append(max(abs(x - y) for x, y in pairs))
            kls.append(sum(x * math.log(x / max(y, 1e-300)) for x, y in pairs if x > 0))
            n += 1
            if sa["token"] != sb["token"]:
                break  # préfixes différents ensuite
    dps.sort(); kls.sort()
    pct = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
    print(f"{name}: steps={n} missing_ids={missing} dp p50={pct(dps,.5):.4f} p95={pct(dps,.95):.4f} max={dps[-1]:.4f} | KL p50={pct(kls,.5):.5f} p95={pct(kls,.95):.5f} max={kls[-1]:.5f}")
