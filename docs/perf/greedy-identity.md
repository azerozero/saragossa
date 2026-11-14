# Greedy spéculatif et AR : état de l'identité (5 nov. 2026)

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

Coût mesuré le 3 nov. 2026 (27B, ABBA, M5 Max) :

- skinny coupé : DFlash −49 à −52 % ;
- NAX greedy coupé : DFlash −25 % à 32k et −36 % à 64k, MTP −6 à −9 %.

### Mode strict (`SARAGOSSA_RUST_VERIFY_STRICT=1`, opt-in)

Le flag rend la vérification greedy exacte sans réglage manuel. Sous la portée
greedy, il fait trois choses :

- il coupe le préfixe NAX en lignes ;
- les matmuls u4 gs64 de 3 à 16 lignes passent par le qmv-T au lieu du skinny ;
- l'arbre DFlash greedy passe à B = 3 nœuds quand `SARAGOSSA_RUST_DFLASH_TREE`
  est absent ; une valeur explicite garde la priorité.

La politique greedy vient de la requête (température nulle), quel que soit le
type de tête de vérification, y compris sous `SARAGOSSA_RUST_GPU_ARGMAX=0`.

L'échantillonné et le décodage AR ne changent pas.

Le qmv-T (`affine_qmvt_*`) calcule `xt` et `sum` une fois par ligne et par
bloc, puis convertit les poids une fois pour les R lignes. Chaque ligne garde
le contrat canonique du qmv mono. Il existe en deux variantes :

- device, pour `in_dim` ≤ 8192 (gate, up, qkv) et pour 16 lignes ;
- threadgroup, pour down (S = 8) et pour les sorties d'au moins 65536, dont le
  lm_head (S = 16).

Au-delà de 16 lignes, le mode strict garde le qmv répliqué : le skinny et les
GEMM NA sont exclus à toute largeur. Le qmv-T ne s'engage que là où le pas
mono-ligne prend le qmv rapide (`SARAGOSSA_RUST_QMV_FAST`).

Coût du matmul à 9 lignes (ms) :

| Matmul | skinny | qmv-T | qmv répliqué |
|---|---|---|---|
| gate 17408×5120 | 0,19 | 0,25 | 0,36 |
| down 5120×17408 | 0,21 | 0,28 | 0,38 |
| lm_head 248320×5120 | 2,1 | 3,9 | 5,7 |

Choix du budget d'arbre (27B, DFlash, ABBA, 200 tokens, tok/s) :

| Contexte | défaut rapide (B = 8) | strict B = 8 (qmv-T) | strict B = 3 |
|---|---|---|---|
| 4k | 55,3 / 55,3 | 39,5 / 39,1 | 53,8 / 53,1 |
| 32k | 50,8 / 50,3 | 29,6 / 29,8 | 44,9 / 44,9 |

Le strict B = 3 coûte −4 % à 4k et −12 % à 32k face au défaut rapide. Le
strict B = 8 coûte −29 % et −41 %. Mesures croisées strict B = 8 / B = 3 :
39,5 / 52,4 à 4k et 30,1 / 45,1 à 32k.

Le scan gated-delta inverse
(`SARAGOSSA_RUST_LINEAR_INV_DELTA`, opt-in) reste hors périmètre : la
vérification en lignes n'a pas de variante inverse.

## Corrigé

| Source | Cause démontrée | Correction |
|---|---|---|
| scan gated-delta (y) et rms-gate du bloc linear-attn batché | même source que le mono, mais contraction FMA différente en fast-math. Kernels réels compilés seuls : 470/6144 sorties ≠, 0 avec `contract(off)` | `#pragma METAL fp contract(off)` dans les kernels mono, en lignes, en arbre et de rejeu |
| norme d'entrée en lignes : linear-attn concaténé, QKV full-attn (MoE, et dense avec gate de sortie) | le mono la fusionne au prologue du matmul (réduction de `rms_norm_simd`) | norme en miroir (`encode_rms_norm_rows_matching_solo`), sous un prédicat unique partagé avec l'encodeur mono |
| matmul u4 gs64 en lignes (qmv répliqué, qmm2) contre le qmv mono et ses variantes fusionnées | même source, mais contraction FMA laissée au compilateur, différente d'un kernel à l'autre ; miroirs de norme sans la même contraction que les prologues fusionnés | contrat canonique : produits en `fma()` explicites, `r = fma(scale, accum, fma(sum, bias, r))`, contraction coupée dans le corps (voir « Contrat canonique du qmv u4 gs64 ») |
| attention en arbre DFlash (flash et 2-passes) | la ligne parcourait les positions stockées en sautant les non-ancêtres. Les clés visibles tombaient donc dans d'autres simdgroups et blocs que dans le pas AR du chemin (jusqu'à 342/512 sorties ≠) | parcours en positions virtuelles (préfixe, puis k-ième ancêtre) ; kernel et blocs choisis sur la longueur du chemin |

Les kernels linear-attn mono changent au dernier bit : les sorties AR de ces
couches ne sont plus exactement celles d'avant. Sur le cas prose32k-1, le
texte AR diffère de l'ancien après 834 caractères.

### Contrat canonique du qmv u4 gs64

Périmètre : les 15 kernels u4 gs64 de `kernels.metal` :

- qmv non aligné, aligné, résiduel et `align64` ;
- `qmm2` et `qmm8` ;
- gather qmv et sa queue ;
- gate/up+SiLU, simple et gather ;
- QKV split ;
- prologues rms fusionnés au qmv et au QKV split ;
- o_proj à porte fusionnée ;
- argmax fusionné.

Chacun calcule ses produits en `fma()` explicites et
`r = fma(scale, accum, fma(sum, bias, r))`, sous `#pragma METAL fp
contract(off)` en tête de corps. Les miroirs en lignes `rms_norm_simd_rows_f32`
et `rms_norm_simd_u3_rows_f32` portent la même contraction coupée, comme le
bloc de réduction des prologues rms u3 et u8 (`sumsq` et `sumsq / dim + eps`).

L'oracle `rows_exact_qmv_contract_cpu_oracle` ancre ce contrat sur les
compilations exercées (fast-math, ce compilateur, ce GPU), pas au-delà :

- partiels par voie égaux à une réplique CPU f32 ;
- leur `simd_sum` égal au kernel intact ;
- kernel intact égal au pipeline de production, en aligné et non aligné.

`contract(off)` ne coupe pas la réassociation fast-math : c'est ce test qui
garde l'ordre de la somme.

Le contrat change l'arrondi de l'AR. Bilan O3 (5 nov. 2026, greedy AR,
200 tokens, `main` contre le contrat) : 9 séquences sur 20 diffèrent.

| Modèle | Identiques | Divergence (premier token différent) |
|---|---|---|
| 27B | 7/10 | chat-4 (4), code4k-3 (126), prose32k-1 (182) |
| 35B | 4/10 | chat-2 (22), chat-4 (114), chat-6 (163), code4k-2 (6), code4k-3 (80), prose32k-1 (12) |

Les bascules observées sont des quasi-égalités entre candidats. Exemples :

- « on the coast of Blackwood Point » contre « on the Isle of Skye » ;
- « Rust's String Pattern API » contre « Rust's `Pattern` API ».

Aucune régression de qualité n'est démontrée. Les goldens Whisper, TTS, Mimi
et speaker sont identiques au bit près avant et après.

Gemma (modèles 4 bits du cache HF local, 200 tokens greedy, prompts chat-1,
chat-4 et code4k-1) : Gemma 3 1B et 4B sont identiques sur les trois prompts.
Gemma 4 26B-A4B est identique sur deux, et code4k-1 bascule au token 63.

Débit (ABBA, 200 tokens, tok/s, deux mesures par côté) :

| Cas | `main` | contrat |
|---|---|---|
| AR 27B 4k | 31,74 / 31,43 | 31,75 / 31,43 |
| AR 27B 32k | 28,47 / 28,52 | 28,42 / 28,56 |
| AR 35B 4k | 138,3 / 139,0 | 139,5 / 139,8 |
| AR 35B 32k | 116,1 / 116,4 | 116,8 / 116,4 |
| MTP D2 27B 4k | 47,18 / 47,20 | 49,23 / 48,65 |
| MTP D2 27B 32k | 41,03 / 41,26 | 41,49 / 41,54 |
| MTP D2 35B 4k | 166,98 / 166,91 | 170,29 / 169,90 |
| MTP D2 35B 32k | 136,2 / 136,5 | 142,6 / 142,7 |

L'AR reste dans ±1 %. Les écarts MTP suivent les tokens acceptés, qui
changent avec l'arrondi ; ce n'est pas un gain des kernels.

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
  SARAGOSSA_RUST_IDENTITY_MODES=mtp,mtp-skip,dflash-chaîne,dflash-arbre,dflash-défaut,dflash-cpu \
  SARAGOSSA_RUST_VERIFY_STRICT=1 \
  cargo test --release --all-features --lib -- --ignored --exact \
    decoder::tests::greedy_identity_states_real_model
  ```

  - **En mode strict, le test doit passer.** Les deux coupures manuelles
    (`SARAGOSSA_RUST_QMM_SKINNY_MIN_BATCH=32` et
    `SARAGOSSA_RUST_VERIFY_ROWS_NAX_GREEDY=0`) donnent le même résultat, plus
    lentement.
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

Sous le contrat canonique, le qmv répliqué et le qmm2 par paires sont exacts.
À 9 lignes, le qmv répliqué coûte environ le double du skinny : gate
17408×5120 en 0,36 ms contre 0,18 ms, down en 0,38 ms contre 0,21 ms.
