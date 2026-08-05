#!/usr/bin/env python3
"""Tests de la logique d'allocation de bits, sans dépendance à MLX.

Le classement et l'allocation portent toute la méthode : une erreur ici produit
un modèle silencieusement dégradé ou privé de ses kernels rapides. Ces tests
n'importent pas `mlx`, ils extraient les deux fonctions pures du module afin de
tourner sur une machine sans Apple Silicon (et donc en CI).

    python3 test_allocation.py
"""

from __future__ import annotations

import ast
import collections
import sys
from pathlib import Path

MODULE = Path(__file__).with_name("mixed_quant.py")
PURE_FUNCTIONS = ("rank_candidates", "allocate_bits")


def load_pure_functions():
    """Extrait les fonctions pures de `mixed_quant.py` sans exécuter ses imports.

    Importer le module entraînerait `import mlx.core`, indisponible hors Apple
    Silicon ; on compile donc uniquement les définitions qui n'en dépendent pas.
    """
    tree = ast.parse(MODULE.read_text())
    kept = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in PURE_FUNCTIONS
    ]
    missing = set(PURE_FUNCTIONS) - {node.name for node in kept}
    if missing:
        raise AssertionError(f"fonctions absentes de {MODULE.name}: {sorted(missing)}")

    namespace = {"collections": collections}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(MODULE), "exec"), namespace)
    return namespace["rank_candidates"], namespace["allocate_bits"]


rank_candidates, allocate_bits = load_pure_functions()


def test_ranks_by_descending_sensitivity() -> None:
    candidates = {"a": (0.1, 100), "b": (0.4, 100), "c": (0.3, 100), "d": (0.2, 100)}
    ranked = rank_candidates(candidates, group_trios=False)
    assert [names[0] for _, _, names in ranked] == ["b", "c", "d", "a"]


def test_budget_promotes_only_most_sensitive() -> None:
    candidates = {"a": (0.1, 100), "b": (0.4, 100), "c": (0.3, 100), "d": (0.2, 100)}
    ranked = rank_candidates(candidates, group_trios=False)
    assignment, used = allocate_bits({}, ranked, 400, 0.5, 3, 4)

    assert assignment == {"b": 4, "c": 4, "d": 3, "a": 3}
    assert used == 200, "un budget de 50 % sur 400 paramètres promeut 200"


def test_forced_layers_bypass_budget() -> None:
    # Les couches épinglées sortent en largeur haute même si le budget est nul.
    candidates = {"a": (0.9, 100)}
    ranked = rank_candidates(candidates, group_trios=False)
    assignment, used = allocate_bits({"embed": 50}, ranked, 150, 0.0, 3, 4)

    assert assignment["embed"] == 4, "un module épinglé ignore le budget"
    assert assignment["a"] == 3
    assert used == 0, "le budget ne compte que les promotions par sensibilité"


def test_per_layer_mode_can_split_a_trio() -> None:
    # Vérifie que le mode non groupé produit bien le dépareillage que `--moe`
    # existe pour empêcher : sans ce constat, le test suivant ne prouverait rien.
    candidates = {
        "l0.switch_mlp.gate_proj": (0.9, 100),
        "l0.switch_mlp.up_proj": (0.1, 100),
        "l0.switch_mlp.down_proj": (0.1, 100),
    }
    ranked = rank_candidates(candidates, group_trios=False)
    assignment, _ = allocate_bits({}, ranked, 300, 1 / 3, 3, 4)

    widths = {bits for name, bits in assignment.items() if name.startswith("l0")}
    assert len(widths) > 1, "le mode per-couche doit pouvoir dépareiller un trio"


def test_moe_mode_keeps_every_trio_uniform() -> None:
    # Invariant critique : les kernels experts fusés exigent gate/up/down de
    # même largeur. Un trio dépareillé fait tomber le décodage mesuré de ≥140
    # à 95 tok/s, sans aucune erreur visible.
    candidates = {
        "l0.switch_mlp.gate_proj": (0.9, 100),
        "l0.switch_mlp.up_proj": (0.1, 100),
        "l0.switch_mlp.down_proj": (0.1, 100),
        "l1.switch_mlp.gate_proj": (0.2, 100),
        "l1.switch_mlp.up_proj": (0.2, 100),
        "l1.switch_mlp.down_proj": (0.2, 100),
    }
    ranked = rank_candidates(candidates, group_trios=True)
    assignment, _ = allocate_bits({}, ranked, 600, 0.5, 3, 4)

    for layer in ("l0", "l1"):
        widths = {bits for name, bits in assignment.items() if name.startswith(layer)}
        assert len(widths) == 1, f"trio {layer} dépareillé: {widths}"

    assert assignment["l0.switch_mlp.gate_proj"] == 4, "trio le plus sensible promu"
    assert assignment["l1.switch_mlp.gate_proj"] == 3


def test_moe_ranks_trios_by_weighted_error() -> None:
    # L'erreur d'un trio est la moyenne pondérée par la taille : une projection
    # très sensible mais minuscule ne doit pas promouvoir tout le trio.
    candidates = {
        "big.switch_mlp.gate_proj": (0.5, 1000),
        "big.switch_mlp.up_proj": (0.5, 1000),
        "small.switch_mlp.gate_proj": (0.9, 1),
        "small.switch_mlp.up_proj": (0.1, 1000),
    }
    ranked = rank_candidates(candidates, group_trios=True)
    assert ranked[0][2][0].startswith("big"), "le trio globalement le plus sensible d'abord"


def test_default_force_high_follows_architecture() -> None:
    # Le défaut d'épinglage se lit dans le source sans importer `mlx`. La
    # recette MoE appliquée à un dense épinglerait toute l'attention : mesuré
    # 55,7 % des paramètres et 4,35 bpw au lieu de 4,06 sur un 0,6B.
    source = MODULE.read_text()
    assert 'DENSE_FORCE_HIGH = r"embed|lm_head"' in source, "défaut dense minimal attendu"
    assert "MOE_FORCE_HIGH" in source, "recette MoE distincte attendue"
    assert (
        "MOE_FORCE_HIGH if args.moe else DENSE_FORCE_HIGH" in source
    ), "le défaut doit être dérivé du mode --moe, jamais fixe"

    dense_start = source.index("DENSE_FORCE_HIGH = ")
    dense_line = source[dense_start : source.index("\n", dense_start)]
    for structural in ("linear_attn", "self_attn", "shared_expert"):
        assert structural not in dense_line, (
            f"{structural} n'a pas à être épinglé sur un modèle dense"
        )


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passés")
    return 0


if __name__ == "__main__":
    sys.exit(main())
