"""Chevauchement des experts routés entre jetons consécutifs (étape 6 D).

Lit les logits du routeur du golden long (`router_logits.<couche>`, `[T, 512]`)
et publie, pour R = 2, 3 et 4, le nombre moyen d'experts distincts parmi les
top-10 de R jetons consécutifs (fenêtres disjointes), moyenne puis min et max
sur les couches. C'est le facteur des experts d'une vérification en lignes qui
lirait chaque expert distinct une fois.

Usage : python -I tools/flashnext/expert_overlap.py <n2_golden_v3>/long.safetensors
"""

import sys

from safetensors import safe_open


def main(path: str, top_k: int = 10) -> None:
    with safe_open(path, "pt") as golden:
        keys = sorted(
            (k for k in golden.keys() if k.startswith("router_logits.")),
            key=lambda k: int(k.split(".")[1]),
        )
        per_rows = {2: [], 3: [], 4: []}
        for key in keys:
            top = golden.get_tensor(key).float().topk(top_k, dim=-1).indices
            tokens = top.shape[0]
            for rows, means in per_rows.items():
                distinct = [
                    len(set(top[t : t + rows].flatten().tolist()))
                    for t in range(0, tokens - rows, rows)
                ]
                means.append(sum(distinct) / len(distinct))
    print(f"{len(keys)} couches, {tokens} positions")
    for rows, means in per_rows.items():
        print(
            f"R = {rows} : {sum(means) / len(means):.2f} experts distincts sur {top_k * rows}"
            f" (couches : min {min(means):.1f}, max {max(means):.1f})"
        )


if __name__ == "__main__":
    main(sys.argv[1])
