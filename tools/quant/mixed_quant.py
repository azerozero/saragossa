#!/usr/bin/env python3
r"""Quantification mixte par sensibilité, sans calibration, au format MLX affine.

Alloue deux largeurs de bits par couche selon l'erreur de quantification
mesurée en espace des poids, puis écrit un modèle MLX chargeable par `mlx-lm`
comme par `saragossa`.

Méthode
-------
Pour chaque couche linéaire, on mesure l'erreur relative de la largeur basse :

    rel_err(W) = ‖W − dequant(quant(W))‖_F / ‖W‖_F

Les couches sont classées par erreur décroissante et reçoivent la largeur haute
tant qu'un budget en paramètres n'est pas épuisé ; les autres gardent la
largeur basse. C'est calibration-free : aucun jeu de données n'intervient, ce
qui rend le résultat déterministe et reproductible.

Sur les architectures hybrides GatedDeltaNet/SSM, ce classement en espace des
poids bat le classement pondéré par activations (façon imatrix) : les
projections de porte récurrentes sont systématiquement sous-protégées par les
proxys d'activation d'entrée. Mesuré : KL 0,065 contre 0,18 à 0,20 pour trois
variantes imatrix à bpw égal.

Deux garde-fous structurels
---------------------------
`--force-high` épingle en largeur haute, hors budget, ce dont l'erreur se
propage. Le défaut dépend de l'architecture : en dense seuls `embed` et
`lm_head` sont épinglés, tandis que `--moe` y ajoute la récurrence SSM, le
routeur, les experts partagés et l'attention pleine. Appliquer la recette MoE
à un dense épinglerait plus de la moitié des paramètres et ferait grimper le
bpw en silence.

`--moe` décide **par trio de couche** (gate/up/down ensemble) : les kernels
experts fusionnés exigent des largeurs identiques, faute de quoi une couche
dépareillée perd tout chemin rapide. Mesuré : 10 couches sur 40 dépareillées
faisaient tomber le décodage de ≥140 à 95 tok/s.

Exemples
--------
    # Dense 27B, mixte 3/4 bits, 30 % des paramètres en 4 bits
    python mixed_quant.py --input MODELE-bf16 --output SORTIE \
        --low-bits 3 --high-bits 4 --high-fraction 0.30

    # MoE 35B-A3B : décision par trio + garde-fous structurels
    python mixed_quant.py --input MODELE-bf16 --output SORTIE \
        --low-bits 3 --high-bits 4 --high-fraction 0.22 --moe

    # Explorer un mixte 2/3 bits plus agressif
    python mixed_quant.py --input MODELE-bf16 --output SORTIE \
        --low-bits 2 --high-bits 3 --high-fraction 0.35
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import quantize_model, save_config, save_model

# Fichiers annexes recopiés tels quels depuis le modèle source (tokenizer,
# gabarit de chat, etc.). `config.json` est réécrit par la quantification et
# les index de shards ne valent plus rien une fois les poids réécrits.
SIDECAR_SUFFIXES = (".json", ".jinja", ".txt", ".model")

# Épinglage minimal, valable pour un modèle dense : seules les tables
# d'embedding et la tête de sortie sortent du budget. Les épingler davantage
# gonflerait le bpw sans que rien ne le signale.
DENSE_FORCE_HIGH = r"embed|lm_head"

# Épinglage des architectures hybrides SSM/MoE : tout ce dont l'erreur se
# propage à travers la récurrence ou le routage reste en largeur haute.
# Appliquer ceci à un modèle dense épinglerait toute l'attention, soit plus de
# la moitié des paramètres (mesuré 55,7 % sur un 0,6B, 4,35 bpw au lieu de 3,85).
MOE_FORCE_HIGH = (
    r"embed|lm_head|linear_attn|shared_expert|self_attn|\.mlp\.gate$"
)


def relative_quant_error(weight, bits: int, group_size: int) -> float:
    """Erreur de quantification relative en norme de Frobenius.

    Le epsilon au dénominateur évite la division par zéro sur une couche de
    poids nuls, cas dégénéré mais possible sur un modèle élagué.
    """
    quantized, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    restored = mx.dequantize(
        quantized, scales, biases, group_size=group_size, bits=bits
    )
    error = mx.linalg.norm((weight - restored).astype(mx.float32))
    magnitude = mx.linalg.norm(weight.astype(mx.float32))
    return float(error / (magnitude + 1e-9))


def scan_layers(model, force_high: re.Pattern | None, bits: int, group_size: int):
    """Sépare les couches épinglées en largeur haute des couches arbitrables.

    Les poids 3D sont les stacks d'experts MoE (`SwitchLinear`) ; ils comptent
    comme candidats au même titre que les 2D.
    """
    forced: dict[str, int] = {}
    candidates: dict[str, tuple[float, int]] = {}

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "ndim") or weight.ndim not in (2, 3):
            continue
        if force_high is not None and force_high.search(name):
            forced[name] = int(weight.size)
            continue
        candidates[name] = (
            relative_quant_error(weight, bits, group_size),
            int(weight.size),
        )
        # Sans ce vidage, le scan d'un 35B accumule les tampons de dequantize
        # et sature la mémoire unifiée avant la fin.
        mx.clear_cache()

    return forced, candidates


def rank_candidates(candidates: dict[str, tuple[float, int]], group_trios: bool):
    """Classe les unités de décision par erreur décroissante.

    En mode `group_trios`, l'unité est la couche entière (gate/up/down), dont
    l'erreur est la moyenne pondérée par le nombre de paramètres.
    """
    if not group_trios:
        return [
            (error, size, [name])
            for name, (error, size) in sorted(
                candidates.items(), key=lambda kv: kv[1][0], reverse=True
            )
        ]

    trios: dict[str, list[tuple[str, float, int]]] = collections.defaultdict(list)
    for name, (error, size) in candidates.items():
        trios[name.rsplit(".", 1)[0]].append((name, error, size))

    ranked = []
    for members in trios.values():
        size = sum(n for _, _, n in members)
        weighted = sum(error * n for _, error, n in members) / max(size, 1)
        ranked.append((weighted, size, [name for name, _, _ in members]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def allocate_bits(forced, ranked, total_params, high_fraction, low_bits, high_bits):
    """Alloue la largeur haute par ordre de sensibilité jusqu'à épuiser le budget."""
    assignment = {name: high_bits for name in forced}
    budget = high_fraction * total_params
    used = 0

    for _, size, names in ranked:
        bits = high_bits if used + size <= budget else low_bits
        for name in names:
            assignment[name] = bits
        if bits == high_bits:
            used += size

    return assignment, used


def copy_sidecars(source: Path, destination: Path) -> None:
    for path in glob.glob(str(source / "*")):
        name = Path(path).name
        if not name.endswith(SIDECAR_SUFFIXES):
            continue
        if name == "config.json" or "index" in name:
            continue
        shutil.copy(path, destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantification mixte par sensibilité (MLX affine).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="modèle source bf16 (répertoire)"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="répertoire du modèle quantifié"
    )
    parser.add_argument("--low-bits", type=int, default=3, help="largeur basse")
    parser.add_argument("--high-bits", type=int, default=4, help="largeur haute")
    parser.add_argument(
        "--high-fraction",
        type=float,
        default=0.30,
        help="part des paramètres promue en largeur haute (0.0 à 1.0)",
    )
    parser.add_argument(
        "--group-size", type=int, default=64, help="taille de groupe MLX affine"
    )
    parser.add_argument(
        "--moe",
        action="store_true",
        help="décider par trio gate/up/down (obligatoire pour les kernels experts fusés)",
    )
    parser.add_argument(
        "--force-high",
        default=None,
        help="regex des modules épinglés en largeur haute, hors budget. "
        "Par défaut: minimal en dense (embed|lm_head), structurel avec --moe "
        "(récurrence, routeur, experts partagés, attention). "
        "Chaîne vide pour désactiver.",
    )
    args = parser.parse_args(argv)
    if args.force_high is None:
        # Le bon épinglage dépend de l'architecture : appliquer la recette MoE
        # à un dense épinglerait toute l'attention et ferait grimper le bpw en
        # silence. On la dérive donc du mode plutôt que de la laisser au hasard.
        args.force_high = MOE_FORCE_HIGH if args.moe else DENSE_FORCE_HIGH
    return args


def validate(args: argparse.Namespace) -> None:
    """Rejette les entrées invalides avant de charger des dizaines de Go."""
    if not args.input.is_dir():
        raise SystemExit(f"modèle source introuvable: {args.input}")
    if not (args.input / "config.json").is_file():
        raise SystemExit(f"config.json absent de {args.input}")
    if not 0.0 <= args.high_fraction <= 1.0:
        raise SystemExit(f"--high-fraction hors de [0, 1]: {args.high_fraction}")
    if args.low_bits >= args.high_bits:
        raise SystemExit(
            f"--low-bits ({args.low_bits}) doit être inférieur à "
            f"--high-bits ({args.high_bits})"
        )
    if args.group_size <= 0:
        raise SystemExit(f"--group-size doit être positif: {args.group_size}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate(args)

    force_high = re.compile(args.force_high) if args.force_high else None

    print(f"chargement de {args.input}...", flush=True)
    model, _ = load(str(args.input))
    config = json.loads((args.input / "config.json").read_text())

    print(f"scan des erreurs {args.low_bits} bits par couche...", flush=True)
    forced, candidates = scan_layers(model, force_high, args.low_bits, args.group_size)
    if not candidates:
        raise SystemExit(
            "aucune couche arbitrable: --force-high est trop large, "
            "tout le modèle serait en largeur haute"
        )

    total = sum(forced.values()) + sum(size for _, size in candidates.values())
    ranked = rank_candidates(candidates, args.moe)
    assignment, used = allocate_bits(
        forced, ranked, total, args.high_fraction, args.low_bits, args.high_bits
    )

    high_count = sum(1 for bits in assignment.values() if bits == args.high_bits)
    pinned_share = sum(forced.values()) / total
    promoted_share = used / total
    print(
        f"épinglé {args.high_bits} bits: {len(forced)} modules "
        f"({pinned_share:.1%} des paramètres) | "
        f"promu par sensibilité: {promoted_share:.1%} | "
        f"total {args.high_bits} bits: {pinned_share + promoted_share:.1%} "
        f"des paramètres, {high_count}/{len(assignment)} modules",
        flush=True,
    )

    def predicate(path: str, module):
        bits = assignment.get(path)
        return {"group_size": args.group_size, "bits": bits} if bits else False

    model, config = quantize_model(
        model, config, args.group_size, args.low_bits, quant_predicate=predicate
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_model(str(args.output), model)
    save_config(config, str(args.output / "config.json"))
    copy_sidecars(args.input, args.output)

    print(f"modèle écrit dans {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
