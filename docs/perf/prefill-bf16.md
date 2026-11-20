# Prefill BF16 dense : GEMM à un produit (19 nov. 2026)

Modèle : Qwen3.8-27B-BF16, sur M5 Max. Les noyaux sont comparés à Herbert
(`556fcae`, MIT), qui faisait le même prefill 2,3 fois plus vite.

## Cause

Toutes les projections denses du prefill passaient par
`gemm_nax_bf16_nt_tiled_x2`. Ce noyau scinde l'activation f32 en hi + lo
bf16 et fait deux MMA par pas de K. Herbert n'en fait qu'un, sur une
activation bf16. Le prefill de 4096 tokens représente environ 199 TFLOP
de GEMM :

| Projection | TFLOP |
|---|---:|
| gate/up | 93,5 |
| down | 46,7 |
| qkv linéaire | 20,6 |
| o/out | 16,5 |
| z | 12,4 |
| q | 8,3 |
| k/v | 1,4 |

Herbert tenait 50 TFLOP/s effectifs, saragossa 21.

## Changement

Le GEMM à un produit arrondit l'activation une fois en bf16
(`f32_to_bf16`), puis la multiplie par `gemm_nax_bf16_nt_bn128` (matmul2d,
tuiles 64×128, grille N d'abord). L'accumulation reste en f32. Sur le 27B,
le prefill est 36 à 44 % plus court.

Il s'active par défaut quand toutes ces conditions sont réunies :

- c'est une projection d'une couche du prefill résident d'un LLM Qwen dont
  toutes les queues MLP sont denses en bf16 (portée `DenseBf16X1Scope`) ;
- M ≥ 64 lignes ;
- K et N sont multiples de 64, et N l'est aussi de 128.

Partout ailleurs, le GEMM hi + lo reste en place :

- la tête MTP ;
- le drafter DFlash ;
- Gemma ;
- les tenseurs restés en bf16 dans les modèles quantifiés ;
- le decode et la vérification, même avec un plancher
  `SARAGOSSA_RUST_DENSE_BF16_GEMM_MIN_BATCH` sous 64.

`SARAGOSSA_RUST_DENSE_BF16_GEMM_X1=1` force le GEMM à un produit partout, ces
chemins compris. `=0` le coupe partout.

Le tampon bf16 de A (M×K) est gardé en cache, un par K distinct (5120, 6144
et 17408 sur le 27B). Il occupe 2·M·K octets : 142 Mo pour le down_proj à
4096 lignes, 285 Mo à 8192.

Écartés :

- le double tampon threadgroup du x2 (PR #261) : 6 à 12 % plus lent ;
- la grille N d'abord pour le x2, pourtant égale au bit : 26 % plus lente
  (11,8 s contre 9,4 s à 4096 tokens).

## Vitesse

Prefill en temps mur, tête comprise (`--prefill-only --prefill-repeat 1`).
ABBA ×3 avec un processus neuf par mesure et charge inférieure à 2 au
lancement. Chaque case donne la médiane sur 6 mesures.

| Tokens | hi + lo | X1 | Gain | Herbert |
|---:|---:|---:|---:|---:|
| 512 | 1,117 s | 0,620 s | −44 % | 0,467 s |
| 4096 | 9,357 s | 5,509 s | −41 % | 3,951 s |
| 16384 | 37,89 s | 24,12 s | −36 % | 17,27 s |

Les timestamps GPU par section (`SARAGOSSA_RUST_GPU_TIMESTAMPS=1`) donnent,
sur 4096 tokens, la répartition suivante. La méthode « stage boundary »
attribue mal le temps entre noyaux, ces chiffres sont des ordres de
grandeur.

| Section | hi + lo | X1 |
|---|---:|---:|
| MLP (`tail_dense`) | 6,08 s | 3,49 s |
| linear attention (projections + delta) | 2,31 s | 1,44 s |
| qkv | 0,43 s | 0,24 s |
| o_proj | 0,17 s | 0,10 s |
| attention causale | 0,13 s | 0,13 s |
| normes et porte | 0,08 s | 0,08 s |

Il reste environ 1,6 s d'écart avec Herbert à 4096 tokens. La moitié se
trouve dans le MLP (gate et up séparés, SwiGLU f32), le reste dans la linear
attention, où les projections b/a (N = 48) passent par le qmv scalaire.

## Exactitude

On compare le KL du GEMM X1 au GEMM hi + lo, pris comme référence. Chaque
position garde le top-64 et la masse restante, avec une normalisation en
f64. Le test est `golden::kld` en auto-comparaison.

| Corpus | Positions | KL moyen | KL max | Top-1 | Greedy 64 tokens |
|---|---:|---:|---:|---:|---|
| 8 morceaux de 512 tokens, 256 notées (corpus kld) | 2048 | 7,0e-6 | 5,8e-4 | 100 % | 8/8 identiques |
| 8 prompts du corpus, 64 tokens générés (`--kl-compare`) | 512 | ≤ 3e-5 | — | 99,8 % (511/512) | — |
| 4 fenêtres de 2048 tokens + 64 notées (pg1661) | 256 | 1,8e-5 | 4,5e-4 | 100 % | 3/4 identiques, 1 diverge au 10e token |
| 4 fenêtres de 8192 tokens + 64 notées (pg1661) | 256 | 2,1e-5 | 4,0e-4 | 98,8 % (253/256) | 4/4 identiques |

Les fenêtres longues viennent de « The Adventures of Sherlock Holmes »
(Project Gutenberg n° 1661, domaine public), le même livre que le corpus kld.
Il s'agit de 4 fenêtres réparties sur le livre. Avec
`KLD_SELF_TEXT`/`KLD_SELF_PREFIX`, le test auto-compare a été lancé en local,
puis retiré de la PR. Δp max : 0,012 à 2048 et à 8192.

L'arrondi bf16 de l'activation touche le prefill seulement. Le decode M = 1
reste sur le qmv f32. Voir [greedy-identity.md](greedy-identity.md).

## Fusions opt-in (MLP, b/a)

`SARAGOSSA_RUST_DENSE_BF16_FUSED_MLP=1` calcule gate et up dans un seul
dispatch matmul2d, avec le SwiGLU en épilogue et H écrit en bf16. Le résultat
est égal au bit au chemin X1 (`dense_bf16_fused_mlp_matches_x1`).

`SARAGOSSA_RUST_DENSE_BF16_LA_BA_GEMM=1` remplace les deux qmv des
projections b/a par un GEMM hi + lo sur `[b; a]`. Les deux flags actifs sont
comparés au défaut, dans le même protocole qu'au-dessus :

| Corpus | Positions | KL moyen | KL max | Top-1 | Greedy 64 tokens |
|---|---:|---:|---:|---:|---|
| corpus kld, 8 × 512 | 2048 | 1,1e-5 | 1,7e-3 | 100 % | 7/8 identiques, 1 diverge au 64e token |
| pg1661, 4 fenêtres de 8192 + 64 | 256 | 2,3e-5 | 3,8e-4 | 98,4 % (252/256) | 3/4 identiques, 1 diverge au 43e token |

Le gain en temps n'est pas encore mesuré.
