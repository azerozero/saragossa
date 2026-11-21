"""Allocation 3/4/5/6/8 bits au gain d'erreur pondérée par bit stocké, budget langage.

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

WIDTHS = (3, 4, 5, 6, 8)


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


def allocate(scores, sizes, total_params, fixed_bits, target_bpw, forced=(), high_bits=6,
             base_bits=4):
    """Glouton déterministe, promotions 3→4→5→6→8 ; base 4 par défaut."""
    if (not math.isfinite(target_bpw) or target_bpw <= 0 or high_bits not in (5, 6, 8)
            or base_bits not in WIDTHS or base_bits > high_bits
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
        width = high_bits if any(name in forced for name in members) else base_bits
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
            # Les épinglés fixent un minimum ; chaque promotion suit WIDTHS.
            next_index = WIDTHS.index(old) + 1
            if next_index == len(WIDTHS):
                continue
            new = WIDTHS[next_index]
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


def weight_space_score(errors, energy, size):
    """Erreurs absolues ‖ΔW‖² d'un module → size · ‖ΔW‖² / ‖W‖²."""
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("énergie de poids invalide")
    if energy == 0:
        return dict.fromkeys(errors, 0.0)
    return {bits: size * error / energy for bits, error in errors.items()}


def scan(model, matrix, force_high, target_bpw, high_bits, imatrix_blend=False, base_bits=4):
    """Score le fit réellement utilisé par convert_ls, par blocs pour borner la mémoire."""
    if imatrix_blend and matrix is None:
        raise ValueError("imatrix_blend exige une imatrix")
    import mlx.core as mx
    from mlx.utils import tree_flatten

    modules = language_modules(model)
    sizes = {name: int(m.weight.size) for name, m in modules.items()}
    params = {k: v for k, v in tree_flatten(model.parameters()) if k.startswith("language_model.")}
    total = sum(v.size for v in params.values())
    fixed = sum(v.nbytes * 8 for k, v in params.items() if k.removesuffix(".weight") not in modules)
    forced = {name for name in modules if force_high and re.search(force_high, name)}
    # Vérifier la faisabilité et toute l'imatrix avant les cinq fits par module.
    zeros = {name: dict.fromkeys(WIDTHS, 0.0) for name in modules}
    allocate(zeros, sizes, total, fixed, target_bpw, forced, high_bits, base_bits)
    for name, module in modules.items():
        importance_for(matrix, name, module.weight.shape[-1], is_embedding(module))
    scores = {}
    for name, module in modules.items():
        columns = module.weight.shape[-1]
        embedding = is_embedding(module)
        importance = importance_for(matrix, name, columns, embedding)
        rows = module.weight.reshape(-1, columns)
        blend = imatrix_blend and importance is not None
        errors = dict.fromkeys(WIDTHS, 0.0)
        energy = 0.0
        for start in range(0, rows.shape[0], 256):
            block = rows[start:start + 256]
            if matrix is None:
                energy += (block.astype(mx.float32) ** 2).sum().item()
            mass = opt_quant.error_mass(block, importance, blend)
            # L'entrée one-hot choisit une ligne avec probabilité 1/vocab.
            # Le dénominateur est celui de la table entière, jamais du bloc.
            if embedding:
                mass /= rows.shape[0]
            for bits in WIDTHS:
                packed, scales, biases, _, rmse = opt_quant.fit(
                    block, bits=bits, weight=importance, imatrix_blend=blend)
                errors[bits] += rmse.item() ** 2 * mass
                del packed, scales, biases, rmse
        if matrix is None:
            # Sans corpus : erreur relative ‖ΔW‖²/‖W‖² pondérée par la taille,
            # soit l'objectif additif du classement rel_err historique.
            errors = weight_space_score(errors, energy, sizes[name])
        scores[name] = errors
        mx.clear_cache()
        print(f"score {name} : {errors}", flush=True)
    assignment, bpw = allocate(scores, sizes, total, fixed, target_bpw, forced, high_bits, base_bits)
    report = dict(allocation="imatrix" if matrix is not None else "weights", target_bpw=target_bpw, actual_bpw=bpw, total_parameters=total,
                  fixed_bits=fixed, scores=scores, sizes=sizes, assignment=assignment,
                  forced=sorted(forced), pinned_bits=high_bits, base_bits=base_bits,
                  widths=list(WIDTHS), imatrix_blend=imatrix_blend,
                  metric=("sum E[x²] * sqrt(mean_row(W²) + W_ij²) * (W_quant-W)²" if imatrix_blend
                          else "sum E[x²] * (W_quant-W)²" if matrix is not None
                          else "size * sum (W_quant-W)² / sum W², fit LS uniforme"),
                  embedding_metric="sum (W_quant-W)² / vocab; fréquence uniforme")
    return assignment, report


def _self_check():
    scores = {"a": {3: 30, 4: 10, 5: 5, 6: 2, 8: 0}, "b": {3: 3, 4: 2, 5: 1.5, 6: 1, 8: 0}}
    sizes = {"a": 64, "b": 64}
    result, bpw = allocate(scores, sizes, 128, 0, 5.5)
    assert result == {"a": 6, "b": 4} and bpw == 5.5
    result, bpw = allocate(scores, sizes, 128, 0, 5.5, {"b"})
    assert result == {"a": 4, "b": 6} and bpw == 5.5
    result, bpw = allocate(scores, sizes, 128, 0, 5.0, {"b"}, high_bits=5)
    assert result == {"a": 4, "b": 5} and bpw == 5.0
    # Les tenseurs non quantifiés et les échelles/biais comptent réellement.
    result, bpw = allocate(scores, sizes, 192, 64 * 16, 9)
    assert bpw <= 9 and result == {"a": 6, "b": 4}
    names = ["layer.switch_mlp." + p + "_proj" for p in ("gate", "up", "down")]
    trio_scores = {n: scores["a"] for n in names}
    result, bpw = allocate(trio_scores, dict.fromkeys(names, 64), 192, 0, 6.5, {names[0]})
    assert set(result.values()) == {6} and bpw == 6.5
    # Chaque pas doit apporter un gain ; pas de saut au-dessus d'un pas nuisible.
    result, bpw = allocate({"a": {3: 30, 4: 10, 5: 12, 6: 2, 8: 0}}, {"a": 64}, 64, 0, 8.5)
    assert result == {"a": 4} and bpw == 4.5
    for width in WIDTHS:
        result, bpw = allocate({"a": scores["a"]}, {"a": 64}, 64, 0, width + 0.5, base_bits=3)
        assert result == {"a": width} and bpw == width + 0.5
    result, bpw = allocate(scores, sizes, 128, 0, 4.2, base_bits=3)
    assert result == {"a": 4, "b": 3} and bpw == 4.0
    result, bpw = allocate(scores, sizes, 128, 0, 5.0, {"b"}, base_bits=3)
    assert result == {"a": 3, "b": 6} and bpw == 5.0
    # Le coût fixe peut rendre impossible même une base 3 bits.
    for target, fixed, forced, base in ((3.49, 0, (), 3), (4.2, 128, (), 3),
                                        (4.2, 0, {"b"}, 3), (9, 0, (), 2),
                                        (9, 0, (), 8)):
        try:
            allocate(scores, sizes, 128, fixed, target, forced, base_bits=base)
            raise AssertionError("base ou budget invalide accepté")
        except ValueError:
            pass
    plateau = {"a": {3: 10, 4: 12, 5: 5, 6: 2, 8: 0}}
    assert allocate(plateau, {"a": 64}, 64, 0, 8.5, base_bits=3) == ({"a": 3}, 3.5)
    result, bpw = allocate(trio_scores, dict.fromkeys(names, 64), 192, 0, 4.2, base_bits=3)
    assert set(result.values()) == {3} and bpw == 3.5
    result, bpw = allocate(trio_scores, dict.fromkeys(names, 64), 192, 0, 4.5, base_bits=3)
    assert set(result.values()) == {4} and bpw == 4.5
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
    # Score poids : invariance d'échelle d'un module, et module nul sans gain.
    a = weight_space_score({3: 4.0, 4: 1.0}, 8.0, 64)
    assert a == {3: 32.0, 4: 8.0} and a == weight_space_score({3: 400.0, 4: 100.0}, 800.0, 64)
    assert weight_space_score({3: 0.0}, 0.0, 64) == {3: 0.0}
    assert "mlx.core" not in sys.modules


if __name__ == "__main__":
    _self_check()
    print("sensitivity: auto-tests stdlib ok")
