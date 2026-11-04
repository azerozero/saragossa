# Greedy spéculatif et AR : état de l'identité (4 nov. 2026)

En greedy, MTP, DFlash et lookup doivent produire le texte de l'AR. Pour cela,
une ligne vérifiée parmi R lignes doit donner exactement les bits du pas
mono-ligne.

Avant les corrections ci-dessous, une dérive faisait parfois basculer une
quasi-égalité top-1/top-2 :
- 1 requête sur 66 sur le 27B (bench concurrents, prompt prose de 32k) ;
- 2 prompts sur 5 avec DFlash sur le 35B, chemin non exposé (voir
  [mtp-pairwise.md](mtp-pairwise.md), « DFlash 2 sur le 35B-A3B »).

## Limites actuelles

Deux kernels actifs par défaut ne sont pas bit-identiques au pas mono-ligne.

| Kernel | Quand | Pourquoi |
|---|---|---|
| matmul u4 skinny | lots de 5 lignes et plus : DFlash, lookup, MTP avec drafts lookup sur prompt long (8 lignes) | MMA bf16 hi+lo contre FMA scalaire du qmv |
| préfixe NAX de l'attention en lignes | contexte ≥ 16k, KV bf16, lots de 2 lignes et plus | Q en bf16 et MMA |

Avec ces réglages par défaut, le texte reste celui de l'AR, sauf sur une
quasi-égalité. Deux réglages coupent ces kernels :
- `SARAGOSSA_RUST_QMM_SKINNY_MIN_BATCH=32` remplace le skinny par le qmv
  répliqué ;
- `SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY=0` garde le NAX pour l'échantillonné
  seulement.

Le coût de ces coupures est à mesurer. Le scan gated-delta inverse
(`SARAGOSSA_RUST_LINEAR_INV_DELTA`, opt-in) reste hors périmètre : la
vérification en lignes n'a pas de variante inverse.

## Corrigé

| Source | Cause démontrée | Correction |
|---|---|---|
| scan gated-delta (y) et rms-gate du bloc linear-attn batché | même source que le mono, mais contraction FMA différente en fast-math. Kernels réels compilés seuls : 470/6144 sorties ≠, 0 avec `contract(off)` | `#pragma METAL fp contract(off)` dans les kernels mono, en lignes, en arbre et de rejeu |
| norme d'entrée en lignes : linear-attn concaténé, QKV full-attn (MoE, et dense avec gate de sortie) | le mono la fusionne au prologue du matmul (réduction de `rms_norm_simd`) | norme en miroir (`encode_rms_norm_rows_matching_solo`), sous un prédicat unique partagé avec l'encodeur mono |
| attention en arbre DFlash (flash et 2-passes) | la ligne parcourait les positions stockées en sautant les non-ancêtres. Les clés visibles tombaient donc dans d'autres simdgroups et blocs que dans le pas AR du chemin (jusqu'à 342/512 sorties ≠) | parcours en positions virtuelles (préfixe, puis k-ième ancêtre) ; kernel et blocs choisis sur la longueur du chemin |

Les kernels linear-attn mono changent au dernier bit : les sorties AR de ces
couches ne sont plus exactement celles d'avant. Sur le cas prose32k-1, le
texte AR diffère de l'ancien après 834 caractères. Le coût AR de
`contract(off)` est à mesurer.

## Oracles et commandes

- **O1**, synthétique, kernel par kernel, R = 1..9 :
  `src/metal_backend/tests/rows_exact.rs`, plus l'attention dans
  `decode_resident/tests.rs` et `rows_tree_tests.rs`. Chaque oracle sensible
  au skinny passe deux fois :
  - réglages par défaut : exceptions listées, finies et bornées ;
  - skinny coupé : égalité stricte.

  ```sh
  cargo test --release --all-features --lib -- --ignored \
    metal_backend::tests::rows_exact_ decode_resident::tests::full_attn_rows \
    decode_resident::rows_tree_tests::
  ```

- **O2/O3**, modèle réel (`greedy_identity_states_real_model`). Le test
  vérifie :
  - les tokens de chaque mode contre l'AR ;
  - les états des lignes vérifiées en bits, relevés depuis les entrées de
    production ;
  - le compteur de quasi-égalités ;
  - une couverture obligatoire : des lots de plusieurs lignes comparés pour
    chaque mode, et un arbre réellement branché.

  Le corpus court est versionné. `prose32k-1` est externe : la variable
  `SARAGOSSA_RUST_IDENTITY_EXTERNAL_DIR` désigne son dossier, et le test
  contrôle son empreinte. Son budget est d'au moins 200 tokens.

  Commande stricte :

  ```sh
  SARAGOSSA_REQUIRE_REAL_MODELS=1 \
  SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL=<Qwen3.8-27B-4bit> \
  SARAGOSSA_RUST_DFLASH_DRAFT_DIR=<Qwen3.8-27B-DFlash2> \
  SARAGOSSA_RUST_IDENTITY_EXTERNAL_DIR=<dossier de prose32k-1.txt> \
  SARAGOSSA_RUST_IDENTITY_PROMPTS=chat-1,chat-2,chat-3,chat-4,chat-5,chat-6,code4k-1,code4k-2,code4k-3,prose32k-1 \
  SARAGOSSA_RUST_IDENTITY_MODES=mtp,dflash-chaîne,dflash-arbre \
  SARAGOSSA_RUST_QMM_SKINNY_MIN_BATCH=32 SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY=0 \
  cargo test --release --all-features --lib -- --ignored --exact \
    decoder::tests::greedy_identity_states_real_model
  ```

  - **Avec les deux coupures, le test doit passer.**
  - **Avec les réglages par défaut, il doit échouer** : les états en lignes
    du skinny, et du NAX au-delà de 16k, diffèrent de l'AR, même quand les
    tokens restent identiques.
  - Sur le 35B, la même commande s'utilise avec
    `SARAGOSSA_RUST_MTP_SIDECAR=<modèle>/mtp-official/mtp.safetensors`, sans
    drafter, et `SARAGOSSA_RUST_IDENTITY_MODES=mtp`.
  - Le mode lookup se demande à part : il ne vérifie que si une proposition
    existe.

## Mesuré (O2, coupures appliquées)

| Modèle et mode | États égaux à l'AR en bits |
|---|---|
| 27B MTP, corpus court | 9/9 prompts, aussi sans les coupures |
| 27B DFlash chaîne et arbre B = 8 | chaîne 9/9, arbre 3/3 |
| 27B prose32k-1 (32 768 tokens), MTP, 200 tokens | 199/199 lignes |
| 35B MTP, corpus court | 9/9 prompts |

Les tokens sont identiques à l'AR dans tous ces cas.

Pistes pour un remplacement exact du skinny :
- qmv répliqué ;
- qmm2 par paires, déjà exact ;
- qmv-R à 8 lignes. Il est exact si le qmv mono et lui portent tous deux
  `contract(off)` : prototype à 0/5120 sorties ≠ sur 9 lignes. Mais il
  change alors l'AR de toutes les couches.

Les coûts restent à mesurer.
