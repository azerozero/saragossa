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
| scan gated-delta (y) et rms-gate du bloc linear-attn batché | même source que le mono, mais contraction FMA différente en fast-math. Kernels réels compilés seuls : 470/6144 sorties ≠, 0 avec `contract(off)` | `#pragma METAL fp contract(off)` dans les kernels mono, en lignes, en arbre et de rejeu |
| norme d'entrée en lignes (linear-attn à 5120, QKV full-attn MoE) | le mono la fusionne au prologue du matmul (réduction de `rms_norm_simd`) | norme en miroir (`encode_rms_norm_rows_matching_solo`) |
| attention en arbre DFlash (flash et 2-passes) | la ligne parcourait les positions stockées en sautant les non-ancêtres. Les clés visibles tombaient donc dans d'autres simdgroups et blocs que dans le pas AR du chemin (jusqu'à 342/512 sorties ≠, nœud par nœud) | parcours en positions virtuelles (préfixe, puis k-ième ancêtre) ; kernel et blocs choisis sur la longueur du chemin |

Les kernels linear-attn mono changent au dernier bit : les sorties AR de ces
couches ne sont plus exactement celles d'avant. Sur le cas prose32k-1, le
texte AR diffère de l'ancien après 834 caractères.

## Mesuré après correction (O2, 96 à 200 tokens)

Coupures appliquées : skinny (`SARAGOSSA_RUST_QMM_SKINNY_MIN_BATCH=32`) et
préfixe NAX en greedy (`SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY=0`).

| Modèle et mode | États égaux à l'AR en bits |
|---|---|
| 27B MTP, corpus court | 9/9 prompts (aussi sans les coupures) |
| 27B DFlash chaîne et arbre B = 8 | chaîne 9/9, arbre 3/3 |
| 27B prose32k-1 (32 768 tokens), MTP | 199/199 lignes |
| 35B MTP, corpus court + code4k-3 | 9/9 prompts |

Les tokens sont identiques à l'AR dans tous ces cas.

## Restant

- **Skinny u4 dès 5 lignes**, utilisé par DFlash, par le lookup, et par le
  MTP avec lookup sur prompt long (8 lignes par défaut). Cause : MMA bf16
  hi+lo contre FMA scalaire.
  - Avec les réglages par défaut, c'est la seule source restante des écarts
    MTP sur prose32k-1 (27B) et code4k-3 (35B) : skinny coupé, ces cas
    deviennent exacts.
  - Pistes de remplacement exactes :
    - qmv répliqué ;
    - qmm2 par paires, déjà exact ;
    - qmv-R à 8 lignes (qmm8). Il est exact à condition que le qmv mono et
      lui portent tous deux `contract(off)` (prototype : 0/5120 sorties ≠ sur
      9 lignes). Mais il change alors l'AR de toutes les couches.
  - Coûts à mesurer.
- **Préfixe NAX** de l'attention en lignes (≥ 16k, KV bf16) : il ne peut pas
  être exact. `SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY=0` le réserve à
  l'échantillonné ; le défaut est inchangé en attendant la mesure.

La promesse greedy reste donc « identique à l'AR sauf sur une quasi-égalité »
avec les réglages par défaut. Elle est démontrée au bit près sur le corpus
avec les deux coupures ci-dessus.
