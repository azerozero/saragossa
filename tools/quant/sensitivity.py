"""Allocation 4/6/8 bits au gain d'erreur pondérée par bit stocké, budget langage.

Sans argument : auto-tests stdlib. Utilisé par mixed_quant.py --target-bpw.
Le score somme_j E[x_j²] (W_quant - W)² approxime l'erreur de sortie avec une
covariance diagonale ; il ne modélise ni les corrélations ni la récurrence SSM.
"""
import math
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_quant
from imatrix import importance_for, is_embedding, language_modules

WIDTHS = (4, 6, 8)


def storage_bits(size, bits):
    # Deux paramètres BF16 par groupe de 64, soit 0,5 bpw en plus des codes.
    return size * bits + (size // 64) * 32


def decision_units(names):
    """Les projections des experts sont indissociables, même si une seule est épinglée."""
    groups = {}
    for name in sorted(names):
        stem, _, proj = name.rpartition(".")
        expert = any(part in stem.split(".") for part in ("switch_mlp", "switch_glu", "experts"))
        key = stem if expert and proj in ("gate_proj", "up_proj", "down_proj") else name
        groups.setdefault(key, []).append(name)
    for key, members in groups.items():
        if key not in names and len(members) != 3:
            raise ValueError(f"trio MoE incomplet : {members}")
    return list(groups.values())


def allocate(scores, sizes, total_params, fixed_bits, target_bpw, forced=(), high_bits=6):
    """Glouton déterministe, promotions 4→6/8 ou 6→8 ; jamais de dépassement."""
    if (not math.isfinite(target_bpw) or target_bpw <= 0 or high_bits not in (6, 8)
            or total_params <= 0 or fixed_bits < 0 or not sizes
            or set(scores) != set(sizes) or not set(forced) <= set(sizes)):
        raise ValueError("budget, modules ou largeur d'épinglage invalides")
    for name, size in sizes.items():
        if size <= 0 or size % 64 or set(scores[name]) != set(WIDTHS):
            raise ValueError(f"taille ou largeurs invalides : {name}")
        if any(not math.isfinite(e) or e < 0 for e in scores[name].values()):
            raise ValueError(f"score invalide : {name}")
    units = decision_units(sizes)
    assignment = {}
    for members in units:
        width = high_bits if any(name in forced for name in members) else 4
        assignment.update((name, width) for name in members)
    used = fixed_bits + sum(storage_bits(sizes[n], b) for n, b in assignment.items())
    budget = math.floor(target_bpw * total_params)
    if used > budget:
        raise ValueError(f"budget impossible : minimum {used / total_params:.6f} bpw "
                         f"avec les modules épinglés, cible {target_bpw}")
    while True:
        best = None
        for members in units:
            old = assignment[members[0]]
            # Les épinglés sont un minimum ; ils peuvent encore passer de 6 à 8.
            for new in WIDTHS:
                if new <= old:
                    continue
                cost = sum(sizes[n] * (new - old) for n in members)
                gain = sum(scores[n][old] - scores[n][new] for n in members)
                if gain <= 0 or used + cost > budget:
                    continue
                candidate = (gain / cost, gain, -cost, tuple(members), new, cost)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _, _, _, members, new, cost = best
        assignment.update((name, new) for name in members)
        used += cost
    return assignment, used / total_params


def scan(model, matrix, force_high, target_bpw, high_bits):
    """Score le fit réellement utilisé par convert_ls, par blocs pour borner la mémoire."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    modules = language_modules(model)
    sizes = {name: int(m.weight.size) for name, m in modules.items()}
    params = {k: v for k, v in tree_flatten(model.parameters()) if k.startswith("language_model.")}
    total = sum(v.size for v in params.values())
    fixed = sum(v.nbytes * 8 for k, v in params.items() if k.removesuffix(".weight") not in modules)
    forced = {name for name in modules if force_high and re.search(force_high, name)}
    # Vérifier la faisabilité et toute l'imatrix avant les trois fits par module.
    zeros = {name: dict.fromkeys(WIDTHS, 0.0) for name in modules}
    allocate(zeros, sizes, total, fixed, target_bpw, forced, high_bits)
    for name, module in modules.items():
        importance_for(matrix, name, module.weight.shape[-1], is_embedding(module))
    scores = {}
    for name, module in modules.items():
        columns = module.weight.shape[-1]
        importance = importance_for(matrix, name, columns, is_embedding(module))
        mass = sum(importance.tolist()) if importance is not None else columns
        rows = module.weight.reshape(-1, columns)
        errors = dict.fromkeys(WIDTHS, 0.0)
        for start in range(0, rows.shape[0], 256):
            block = rows[start:start + 256]
            for bits in WIDTHS:
                packed, scales, biases, _, rmse = opt_quant.fit(block, bits=bits, weight=importance)
                errors[bits] += rmse.item() ** 2 * mass * block.shape[0]
                del packed, scales, biases, rmse
        scores[name] = errors
        mx.clear_cache()
        print(f"score {name} : {errors}", flush=True)
    assignment, bpw = allocate(scores, sizes, total, fixed, target_bpw, forced, high_bits)
    report = dict(target_bpw=target_bpw, actual_bpw=bpw, total_parameters=total,
                  fixed_bits=fixed, scores=scores, sizes=sizes, assignment=assignment,
                  forced=sorted(forced), pinned_bits=high_bits,
                  metric="sum E[x²] * (W_quant-W)²; embedding uniforme")
    return assignment, report


def _self_check():
    scores = {"a": {4: 10, 6: 2, 8: 0}, "b": {4: 2, 6: 1, 8: 0}}
    sizes = {"a": 64, "b": 64}
    result, bpw = allocate(scores, sizes, 128, 0, 5.5)
    assert result == {"a": 6, "b": 4} and bpw == 5.5
    result, bpw = allocate(scores, sizes, 128, 0, 5.5, {"b"})
    assert result == {"a": 4, "b": 6} and bpw == 5.5
    # Les tenseurs non quantifiés et les échelles/biais comptent réellement.
    result, bpw = allocate(scores, sizes, 192, 64 * 16, 9)
    assert bpw <= 9 and result == {"a": 6, "b": 4}
    names = ["layer.switch_mlp." + p + "_proj" for p in ("gate", "up", "down")]
    trio_scores = {n: {4: 10, 6: 2, 8: 0} for n in names}
    result, bpw = allocate(trio_scores, dict.fromkeys(names, 64), 192, 0, 6.5, {names[0]})
    assert set(result.values()) == {6} and bpw == 6.5
    # Un saut direct vers 8 reste possible si 6 n'améliore pas le score.
    result, bpw = allocate({"a": {4: 10, 6: 12, 8: 0}}, {"a": 64}, 64, 0, 8.5)
    assert result == {"a": 8} and bpw == 8.5
    for target in (4.49, -1, math.nan, math.inf):
        try:
            allocate(scores, sizes, 128, 0, target)
            raise AssertionError("budget invalide accepté")
        except ValueError:
            pass
    try:
        allocate(scores, sizes, 128, 0, 4.8, {"b"})
        raise AssertionError("épinglage hors budget accepté")
    except ValueError:
        pass
    try:
        decision_units(names[:2])
        raise AssertionError("trio incomplet accepté")
    except ValueError:
        pass
    assert "mlx.core" not in sys.modules


if __name__ == "__main__":
    _self_check()
    print("sensitivity: auto-tests stdlib ok")
