"""Chevauchement des experts routés entre jetons consécutifs (étape 6 D).

Lit les logits du routeur du golden long (`router_logits.<couche>`, `[T, 512]`)
et publie, pour R = 2, 3 et 4, le nombre moyen d'experts distincts parmi les
top-10 de R jetons consécutifs (fenêtres disjointes), moyenne puis min et max
sur les couches, et la multiplicité : nombre moyen d'experts distincts servant
exactement m des R lignes (m = 1..R). Une vérification en lignes qui lit chaque
expert distinct une fois coûte `Σ_m N_m · ρ_expert(m)` fois le coût d'un
expert mono, contre 10 en pas mono.

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
        multiplicity = {rows: [0.0] * (rows + 1) for rows in per_rows}
        windows = {rows: 0 for rows in per_rows}
        for key in keys:
            top = golden.get_tensor(key).float().topk(top_k, dim=-1).indices
            tokens = top.shape[0]
            for rows, means in per_rows.items():
                distinct = []
                for t in range(0, tokens - rows + 1, rows):
                    counts = {}
                    for row in top[t : t + rows].tolist():
                        for expert in row:
                            counts[expert] = counts.get(expert, 0) + 1
                    distinct.append(len(counts))
                    for m in counts.values():
                        multiplicity[rows][m] += 1
                    windows[rows] += 1
                means.append(sum(distinct) / len(distinct))
    print(f"{len(keys)} couches, {tokens} positions")
    for rows, means in per_rows.items():
        print(
            f"R = {rows} : {sum(means) / len(means):.2f} experts distincts sur {top_k * rows}"
            f" (couches : min {min(means):.1f}, max {max(means):.1f})"
        )
        shares = ", ".join(
            f"N_{m} = {multiplicity[rows][m] / windows[rows]:.2f}" for m in range(1, rows + 1)
        )
        print(f"  multiplicité : {shares}")


if __name__ == "__main__":
    main(sys.argv[1])
