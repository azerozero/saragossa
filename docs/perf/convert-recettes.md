# Recettes `saragossa convert` : imatrix, GPTQ, mixte — Qwen3.8-27B (22-26 nov. 2026)

Verdict : **le défaut reste (a), `convert` u4 par moindres carrés.** `--imatrix`
(b) donne la meilleure KL de sélection à budget égal, mais perd 3 à 6 % de débit
decode en MTP et en DFlash. Il reste donc une option.

## Critère pré-enregistré

Le critère a été écrit dans le journal de campagne le 22 nov., avant toute mesure.
Il est cité tel quel :

> **Critère du défaut** : parmi les variantes à bpw langage ≤ 4,6, la meilleure KL moyenne de
> sélection, à condition d'un débit decode ≥ 0,97 × (a) dans chacun des modes AR/MTP/DFlash à 1 k et
> 16 k. Si son écart apparié à (a) a un IC 95 % qui inclut 0, (a) reste le défaut (pas de
> corpus requis). (d) et (e) sont hors budget : repères seulement.

## Variantes et protocole

| Variante | Recette |
|---|---|
| (a) | `saragossa convert` u4 gs64, moindres carrés (défaut actuel) |
| (b) | (a) + `--imatrix` (moindres carrés pondérés par E[x²]) |
| (c) | GPTQ u4 gs64 (`tools/quant/gptq.py`, template u4 uniforme) |
| (d) | mixte ≈ 4,82 bpw, imatrix : u5 `linear_attn` sauf `in_proj_qkv`, u5 `self_attn`, embedding u5, `lm_head` u6 (`--bits-for`) |
| (e) | u8 |

- Imatrix : `tools/quant/imatrix.py --samples 128 --length 512 --chat` sur un corpus
  publiable (`tools/quant/calib_corpus.py` : no_robots train + aya_dataset train,
  937 375 tokens, SHA-256 `c5f353a5…`). Imatrix produite : 497 vecteurs, SHA-256 `94b402ad…`.
- Disjonction (`tools/quant/calib_disjoint.py`) : contrôle fait avant toute KL contre les 48 prompts réservés.
  Résultat : 0 SHA exact, 0 préfixe ni n-gramme commun de 64 tokens, 0 contenu après normalisation.
- Qualité : `chat_kl.py eval --json` sur les 24 prompts de sélection
  (`tools/quant/sets/selection.json`), 4 608 positions. `paired.py` compare chaque variante à (a) :
  IC 95 % par bootstrap, 10 000 tirages, graine 0.
- Budget : `audit_bytes.py`, octets `language_model.*` / 26 895 998 464 paramètres.
- Débit : `bench/ab.py`, ordre ABBA, 4 blocs, 8 passages par côté. 256 tokens greedy, prompts de
  1 024 et 16 384 tokens, caches de préfixe coupés. Les passes rejetées par le harnais
  (bridage thermique, échec) ont été relancées. Versions retenues : `ab-d-ar-16384-r2`,
  `ab-d-mtp-16384-r3`, `ab-d-dflash-16384-r5`, les autres en première passe.

## Qualité

| Variante | bpw langage | KL moy. | top-1 | écart vs (a) | IC 95 % |
|---|---|---|---|---|---|
| (a) convert u4 | 4,501 | 0,0323 | 0,937 | — | — |
| (b) + imatrix | 4,501 | 0,0220 | 0,947 | −0,0104 | [−0,0125 ; −0,0084] |
| (c) GPTQ | 4,501 | 0,0248 | 0,943 | −0,0075 | [−0,0097 ; −0,0053] |
| (d) mixte (repère) | 4,818 | 0,0165 | 0,955 | −0,0158 | [−0,0183 ; −0,0135] |
| (e) u8 (repère) | 8,501 | 0,0007 | 0,991 | −0,0317 | [−0,0356 ; −0,0280] |

(a), (b) et (c) tiennent dans le budget de 4,6 bpw. (b) a la meilleure KL et son IC exclut 0.
(a) ne reste donc pas défaut d'office : la condition de débit décide.

## Débit decode (ratio variante / (a), IC 95 % du harnais)

| Mode | Contexte | (a) tok/s | (b) tok/s | ratio (b) | IC (b) | ≥ 0,97 | (d) tok/s | ratio (d) |
|---|---|---|---|---|---|---|---|---|
| AR | 1 k | 32,85 | 32,89 | 1,0006 | [0,997 ; 1,004] | oui | 30,84 | 0,937 |
| AR | 16 k | 30,41 | 30,41 | 0,9961 | [0,982 ; 1,010] | oui | 28,65 | 0,939 |
| MTP | 1 k | 57,01 | 53,33 | 0,9433 | [0,931 ; 0,956] | **non** | 47,53 | 0,865 |
| MTP | 16 k | 50,70 | 48,58 | 0,9627 | [0,938 ; 0,989] | **non** | 39,98 | 0,840 |
| DFlash | 1 k | 56,64 | 53,22 | 0,9376 | [0,934 ; 0,941] | **non** | 48,22 | 0,861 |
| DFlash | 16 k | 50,73 | 48,64 | 0,9664 | [0,927 ; 1,008] | **non** | 39,82 | 0,835 |

Les débits sont des médianes par côté. La colonne (a) vient des passes (b) contre (a). Les passes
(d) contre (a) mesurent (a) à 6 % près de ces valeurs, et le ratio (d) vient du rapport
de sa propre passe. Le critère porte sur le ratio mesuré. En MTP 1 k et DFlash 1 k, l'IC entier
est sous 0,97 : le verdict ne dépend pas de cette lecture. Le prefill de (b) est indistinct de (a)
(0,97 à 1,01). Celui de (d) perd 10 à 12 %.

## Verdict

- (b) passe en AR, mais échoue en MTP (0,943 / 0,963) et en DFlash (0,938 / 0,966). Le critère
  n'est pas rempli : **(a) reste le défaut**. `--imatrix` reste une option.
- (c) n'a pas de mesure de débit. Sa KL est moins bonne que celle de (b), donc il ne serait pas
  retenu même en passant le débit.
- (d), qui n'est qu'un repère, perd 0,84 à 0,94 du débit decode selon le mode.
- La validation aveugle (`tools/quant/sets/validation.json`) n'a pas été ouverte, puisqu'aucune
  variante ne remplace (a).

## Recette recommandée

- Par défaut, quand le débit spéculatif (MTP, DFlash) compte : `saragossa convert` sans option.
- Quand la qualité prime sur la vitesse spéculative : `saragossa convert --imatrix FICHIER`, avec
  une imatrix de `tools/quant/imatrix.py` calculée sur un corpus disjoint des prompts
  d'évaluation. Gain mesuré à budget et débit AR égaux : Qwen3.8-27B 0,0323 → 0,0220 en KL.
  Mistral-7B v0.3 u4 : 0,188 → 0,126 (`kl_mlx.py`, ablation dans
  [`tools/quant/README.md`](../../tools/quant/README.md)).
- `--bits-for motif=N,…` sert aux recettes mixtes comme (d). Il sort du budget de 4,6 bpw et coûte
  du débit.

## Limites

- La perte MTP/DFlash de (b) vient probablement de l'acceptance des drafts, plus basse avec des
  poids cibles différents, puisque le débit AR est identique. **Non vérifié** : l'acceptance n'a
  pas été extraite des passes.
- (c) GPTQ n'a pas été mesuré en débit.
- Un seul modèle cible pour le critère (Qwen3.8-27B), un prompt par contexte dans le harnais de
  débit, 256 tokens greedy.
- Les données brutes (rapports `ab-*.report.json`, KL par prompt) sont restées hors dépôt, sous
  `~/workspace/ai_model-quant/convert-recettes/`.
