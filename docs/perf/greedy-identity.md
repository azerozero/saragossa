# Greedy spéculatif et AR : état de l'identité (4 nov. 2026)

En greedy, MTP, DFlash et lookup doivent produire le texte de l'AR. Pour cela,
une ligne vérifiée parmi R lignes doit donner exactement les bits du pas
mono-ligne. Le texte reste identique tant qu'une dérive ne fait pas basculer
une quasi-égalité top-1/top-2. Avant ce chantier, cela arrivait pour 1 requête
sur 66 sur le 27B (bench concurrents), et pour 2 prompts sur 5 avec DFlash sur
le 35B, chemin non exposé (voir [mtp-pairwise.md](mtp-pairwise.md),
« DFlash 2 sur le 35B-A3B »).

## Oracles

Trois oracles couvrent l'identité, à différents niveaux :

- **O1** (`src/metal_backend/tests/rows_exact.rs`, plus l'attention dans
  `decode_resident/tests.rs`) : la ligne r d'un lot de R lignes est comparée
  en bits au même calcul mono-ligne, pour R = 1..9, kernel par kernel, sur des
  poids synthétiques aux formes du 27B et du 35B. Les écarts connus sont
  listés, et une exception redevenue exacte fait échouer le test.
- **O2** (`greedy_identity_states_real_model`) : les états finaux des lignes
  vérifiées, relevés depuis les entrées de production (MTP fusionné, lookup,
  DFlash chaîne et arbre), sont comparés en bits à l'AR.
- **O3** : le même test compare les tokens de chaque mode spéculatif à ceux
  de l'AR, et compte les quasi-égalités. Il tourne sur le corpus figé
  `tests/fixtures/rowsexact`, qui contient le cas fixé prose32k-1.

Lancer avec `--ignored` : ces oracles ne valent que pour la configuration du
lancement (GPU, compilateur Metal, flags).

## Corrigé

| Source | Cause démontrée | Correction |
|---|---|---|
| scan gated-delta (y) et rms-gate du bloc linear-attn batché | même source que le mono, contraction FMA différente en fast-math (kernels réels compilés seuls : 470/6144 sorties ≠, 0 avec `contract(off)`) | `#pragma METAL fp contract(off)` dans les kernels mono, en lignes, en arbre et de rejeu |
| norme d'entrée en lignes (linear-attn à 5120, QKV full-attn MoE) | le mono la fusionne au prologue du matmul (réduction de `rms_norm_simd`) | norme en miroir (`encode_rms_norm_rows_matching_solo`) |

Ces kernels mono changent au dernier bit : les sorties AR des couches
linear-attn ne sont plus exactement celles d'avant. Sur le cas prose32k-1, le
texte AR diffère après 834 caractères.

## Mesuré après correction (O2, 96 à 128 tokens, corpus court)

| Modèle et mode | États égaux à l'AR en bits |
|---|---|
| 27B MTP (3 lignes) | 9/9 prompts |
| 27B DFlash chaîne, skinny coupé (`QMM_SKINNY_MIN_BATCH=32`) | 3/3 et 9/9 |
| 35B MTP (sidecar officiel) | 8/9 (code4k-3 : écart à partir de la position 4 105, non isolé) |

Les tokens sont identiques à l'AR dans tous ces cas.

## Restant

- **Skinny u4 dès 5 lignes** (DFlash, lookup) : MMA bf16 hi+lo contre FMA
  scalaire. Le qmv répliqué est exact mais coûte plus cher (mesure à faire).
- **Préfixe NAX** de l'attention en lignes (≥ 16k, KV bf16) : il ne peut pas
  être exact. `SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY=0` le réserve à
  l'échantillonné ; le défaut est inchangé. Avec ce flag, le cas prose32k-1
  garde le texte de l'AR, mais les états divergent encore à partir du
  51e token généré (non isolé).
- **Arbre DFlash** : les états diffèrent de l'AR (attention ou conv en arbre,
  non isolé).

La promesse greedy reste donc « identique à l'AR sauf sur une quasi-égalité ».
Elle n'est démontrée au bit près que pour le 27B en MTP sous 16k, et pour le
27B en DFlash chaîne sans skinny.
