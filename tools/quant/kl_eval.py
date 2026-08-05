#!/usr/bin/env python3
"""Divergence KL et accord top-1 d'un modèle quantifié contre sa référence bf16.

Mesure, en teacher forcing, à quel point la distribution de sortie du modèle
quantifié s'écarte de celle du modèle bf16 :

    KL(bf16 ‖ quant) moyennée sur les positions de chaque prompt
    accord top-1 : fraction des positions où l'argmax coïncide

Le teacher forcing (un seul forward sur le prompt, pas de génération) rend la
mesure déterministe et indépendante de la stratégie d'échantillonnage.

Les deux modèles sont chargés l'un après l'autre, jamais simultanément : un
bf16 de 27B pèse déjà ~52 Go, les tenir ensemble saturerait la mémoire unifiée.

Exemple
-------
    python kl_eval.py --reference MODELE-bf16 --candidate MODELE-quantifie

Repères d'interprétation : KL < 0,05 excellent, < 0,15 bon, > 0,3 dégradation
notable ; un accord top-1 > 0,95 vaut quasi-sans-perte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

# Prompts courts couvrant prose, code et raisonnement numérique. Le jeu est figé
# dans le fichier pour que deux exécutions restent comparables : changer les
# prompts change les valeurs de KL, pas seulement leur bruit.
DEFAULT_PROMPTS = [
    "Explique en détail pourquoi le ciel est bleu et comment fonctionne la diffusion de Rayleigh.",
    "Résume les causes principales de la Révolution française de 1789.",
    "def fibonacci(n):\n    # renvoie le n-ieme terme\n    if n < 2:\n        return n\n    a, b = 0, 1\n",
    "Un train part de Paris à 14h00 à 120 km/h vers Lyon (450 km). À quelle heure arrive-t-il ?",
    "Écris une fonction Python qui vérifie si une chaîne est un palindrome, en ignorant la casse.",
]


def logits_for(model_path: Path, prompts: list[str]):
    """Forward teacher-forcé sur chaque prompt, puis libère le modèle.

    Le `del` explicite est nécessaire : sans lui, le second chargement
    s'ajoute au premier en mémoire unifiée au lieu de le remplacer.
    """
    model, tokenizer = load(str(model_path))
    outputs = []
    for prompt in prompts:
        ids = mx.array(tokenizer.encode(prompt))[None]
        logits = model(ids)[0]  # [T, V]
        mx.eval(logits)
        outputs.append(logits)
    del model
    mx.clear_cache()
    return outputs


def kl_and_agreement(reference_logits, candidate_logits) -> tuple[float, float]:
    """KL(référence ‖ candidat) moyenne et accord top-1 sur les positions."""
    reference = mx.softmax(reference_logits.astype(mx.float32), axis=-1)
    log_reference = mx.log(reference + 1e-9)
    log_candidate = mx.log(
        mx.softmax(candidate_logits.astype(mx.float32), axis=-1) + 1e-9
    )
    divergence = (reference * (log_reference - log_candidate)).sum(axis=-1)
    agreement = (
        reference_logits.argmax(-1) == candidate_logits.argmax(-1)
    ).astype(mx.float32)
    return divergence.mean().item(), agreement.mean().item()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KL et accord top-1 d'un modèle quantifié contre sa référence bf16.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference", required=True, type=Path, help="modèle de référence (bf16)"
    )
    parser.add_argument(
        "--candidate", required=True, type=Path, help="modèle quantifié à évaluer"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="fichier texte optionnel, un prompt par ligne, en remplacement du jeu par défaut",
    )
    return parser.parse_args(argv)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    if not path.is_file():
        raise SystemExit(f"fichier de prompts introuvable: {path}")
    prompts = [line for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"aucun prompt exploitable dans {path}")
    return prompts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for label, path in (("référence", args.reference), ("candidat", args.candidate)):
        if not path.is_dir():
            raise SystemExit(f"modèle {label} introuvable: {path}")

    prompts = load_prompts(args.prompts)

    print(f"[1/2] forward référence {args.reference}...", flush=True)
    reference_logits = logits_for(args.reference, prompts)
    print(f"[2/2] forward candidat {args.candidate}...", flush=True)
    candidate_logits = logits_for(args.candidate, prompts)

    divergences, agreements = [], []
    for index in range(len(prompts)):
        divergence, agreement = kl_and_agreement(
            reference_logits[index], candidate_logits[index]
        )
        divergences.append(divergence)
        agreements.append(agreement)
        print(
            f"prompt {index}: KL={divergence:.4f}  accord_top1={agreement:.3f}",
            flush=True,
        )

    mean_divergence = sum(divergences) / len(divergences)
    mean_agreement = sum(agreements) / len(agreements)
    print(f"\nmoyenne: KL={mean_divergence:.4f}  accord_top1={mean_agreement:.3f}")
    print(
        "repères: KL < 0,05 excellent, < 0,15 bon, > 0,3 dégradé ; "
        "accord_top1 > 0,95 quasi-sans-perte"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
