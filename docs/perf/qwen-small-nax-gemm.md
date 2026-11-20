# Petites projections u4 du 35B-A3B au GEMM NA (bf16×4)

Mesures du 20 novembre 2026 sur M5 Max, modèle Qwen3.6-35B-A3B-4bit.

## Changement

Au prefill, les projections u4 dont une dimension est sous 2048 passaient par le
qmv f32 ligne à ligne : expert partagé (2048→512→2048) et k/v (2048→512). Le
plancher descend à 512 et ces formes passent au GEMM NA tuilé à lhs f32 scindé
en hi + lo bf16, poids déquantifiés scindés de même.

- **bf16×3** (`gemm_nax_coop_qb_tiled_u4_x3`, celui de flint) : hi·hi + hi·lo +
  lo·hi. Le produit lo·lo est abandonné.
- **bf16×4** (`gemm_nax_coop_qb_tiled_u4_x4`) : les quatre produits. L'écart
  relatif max à la référence f64 passe de 3,1-5,1e-7 à 1,0-1,7e-7
  (`qmm_na_fused_tiled_u4_x4_matches_f32_reference`).

Défaut : bf16×4 sur les Qwen MoE seuls. Les Qwen denses, Gemma et flint ne
changent pas. Variable `SARAGOSSA_RUST_QWEN_SMALL_NAX_GEMM` :

| Valeur | Effet |
|---|---|
| absente | bf16×4 si Qwen MoE, sinon rien |
| `0` | kill-switch, retour au qmv f32 |
| `1` ou `x4` | bf16×4 forcé, Gemma et Qwen denses compris |
| `x3` | bf16×3 |

## KL corpus

Protocole `golden::kld::external_kld_qwen36_moe`, 2048 positions. Le KL par
position est exporté par `SARAGOSSA_TEST_KLD_DUMP=<fichier>`. On calcule la
différence appariée avec le défaut (qmv f32) et son IC95 par bootstrap sur
les positions (10 000 tirages).

| Bras | mean_kl | same_top | Δ apparié | IC95 |
|---|---:|---:|---:|---|
| qmv f32 (avant) | 0,010374 | 98,44 % | | |
| bf16×3 | 0,011934 | 98,05 % | +15,0 % | [−0,00025 ; +0,00351] |
| bf16×4 | 0,010030 | 98,58 % | −3,3 % | [−0,00198 ; +0,00129] |
| plancher : `PREFILL_LINEAR_BATCH=0` | 0,009755 | 98,58 % | −6,0 % | [−0,00251 ; +0,00124] |
| plancher : `PREFILL_ATTN_NAX=0` | 0,011768 | 98,05 % | +13,4 % | [−0,00013 ; +0,00339] |

Deux exécutions du défaut sont identiques au bit près.
`PREFILL_LINEAR_NAX=0` et `PREFILL_KV_BF16_STORE=0` ne touchent pas ce
chemin : leur KL est identique au défaut.

Pour estimer le plancher, on a pris deux perturbations sans perte de
précision :
- la récurrence linéaire jeton par jeton, de même précision que le défaut ;
- l'attention Steel f32, plus précise que le défaut.

Elles déplacent le KL de −6 % et de +13 %. Les deux IC95 contiennent 0.

Avec le même IC95, bf16×3 et bf16×4 restent donc dans le bruit. Le bf16×3 se
place au bord haut du plancher. Le bf16×4 reste le plus proche du défaut, et
son écart synthétique est trois fois plus faible : c'est lui qui passe par
défaut.

## Prefill (ABBA ×3, loadavg < 2, médianes)

| Tokens | qmv f32 | bf16×3 | bf16×4 |
|---:|---:|---:|---:|
| 4096 | 1079-1082 ms | 887 ms (−17,8 %) | 907 ms (−16,2 %) |
| 16384 | 5020-5198 ms | 4058 ms (−21,9 %) | 4223 ms (−15,9 %) |

Chaque variante a son propre bras A, d'où les deux valeurs du qmv f32.
