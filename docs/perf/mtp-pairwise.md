# Criblage pairwise des optimisations MTP

Modèle Qwen3.8-27B 4 bits, Apple M5 Max, décodage greedy de 64 tokens.
Référence de production au moment du criblage : MTP D1 committed,
`SARAGOSSA_RUST_MTP_FRESH_CACHE=0`. Ce document publie le criblage.

Depuis le 2026-10-15, le défaut est PW04 (`adaptive_fresh`, profondeur 2,
fusions p1 déjà actives) : confirmation n24 Holm15 +5,8 % sur T0 et +17,0 % sur T1,
non-infériorité à -2 % sur T2 (n96, médiane +0,2 %). L'isolation factorielle
(n48, harnais V5) n'a trouvé aucun gain ≥2 % des états cachés GPU, de l'embedding
GPU, du recouvrement ni de la fusion QMV-u4+résiduel au-dessus du mode seul ; ces
chemins restent dans le code, désactivés hors harnais. Retour au D1 :
`SARAGOSSA_RUST_MTP_MAX_DRAFT=1 SARAGOSSA_RUST_MTP_FRESH_CACHE=0
SARAGOSSA_RUST_MTP_ADAPTIVE_DEPTH=0`.

Depuis le 2026-10-16, l'historique MTP suit la longueur du prompt : cache fresh
jusqu'à `SARAGOSSA_RUST_MTP_FRESH_MAX_PROMPT` tokens (défaut 128), historique
committed au-delà, à profondeur adaptative égale. Mesure appariée PW03 contre
PW04 (n48 par longueur, Holm6 à +2 %, A/A +0,3 %) : le committed perd -5,5 % à
26 tokens, est neutre à 33 et 556 tokens, et gagne +3,3 % à 204, +6,7 % à 380 et
+6,8 % à 732 tokens. Mécanisme : en fresh la tête MTP n'attend que sur
elle-même ; sur un long prompt l'historique lui rend du contexte et relève
l'acceptance en position 2. `SARAGOSSA_RUST_MTP_FRESH_MAX_PROMPT=0` force le
committed partout ; `SARAGOSSA_RUST_MTP_FRESH_CACHE=0` reste prioritaire.

## Protocole

- 19 configurations effectives, chacune sur trois entrées : T0 (26 tokens),
  T1 (33 tokens) et T2 (732 tokens).
- Chaque cellule compare le contrôle D1 et le candidat sur 6 paires, avec
  préchauffage et ordre alterné (3 contrôle d'abord, 3 candidat d'abord).
- Débit : tokens générés divisés par la durée de la boucle de décodage,
  prefill exclu. Aucune instrumentation pendant ces mesures.
- Chaque génération est comparée token par token à un oracle AR du même
  processus. Les 756 générations du criblage (342 paires D1 et 36 paires
  AR) sont exactes.
- Les activations réelles de chaque chemin ont été vérifiées séparément :
  129 paires de facteurs actives par entrée. Neuf paires du plan initial
  restent inexerçables, car le p3 scalaire n'est jamais atteint en MTP : la
  fusion batch y est déjà imposée. Elles ne sont pas comptées comme réussies.
- Harnais : T0 en v3 ; T1 en v3 pour PW02/PW04 et v4 ailleurs ; T2 en v4.
  La v4 ajoute seulement l'inversion d'ordre des lots, sans changer le calcul.
  Les versions ne sont jamais fusionnées.

## Facteurs

| Code | Signification |
|---|---|
| d1, d2 | profondeur de brouillon fixe 1 ou 2 |
| adaptive | profondeur adaptative |
| fresh | cache MTP recalculé (`FRESH_CACHE=1`) |
| block2 | bloc de deux cycles décidé sur GPU |
| ar | décodage autorégressif, sans MTP |
| H/E/O | états cachés résidents / embedding GPU / recouvrement streaming |
| p1/p2/p3 | fusions de noyaux 1, 2 et 3 |

## Résultats contre D1

Médianes contrôle → candidat en tokens/s, écart entre médianes.
† : contrôle anormalement lent dans ce groupe ; cause non mesurée, lot conservé.

| ID | Mode | H/E/O | p1/p2/p3 | T0 | T1 | T2 |
|---|---|---|---|---|---|---|
| PW00 | d1 (A/A) | 0/0/0 | 1/0/0 | 47,26 → 47,53 (+0,6 %) | 47,29 → 47,25 (−0,1 %) | 37,46 → 37,69 (+0,6 %) |
| PW01 | fresh | 0/0/0 | 1/0/0 | 47,18 → 43,57 (−7,7 %) | 47,10 → 44,79 (−4,9 %) | 37,71 → 37,22 (−1,3 %) |
| PW02 | d2 | 0/0/0 | 1/0/0 | 47,29 → 50,89 (+7,6 %) | 47,19 → 55,97 (+18,6 %) | 37,40 → 37,00 (−1,1 %) |
| PW03 | adaptive | 0/0/0 | 1/0/0 | 47,13 → 46,89 (−0,5 %) | 32,03 → 33,75 (+5,4 %) † | 35,97 → 37,70 (+4,8 %) † |
| PW04 | adaptive_fresh | 0/0/0 | 1/0/0 | 47,12 → 50,01 (+6,1 %) | 47,23 → 55,78 (+18,1 %) | 37,69 → 37,31 (−1,0 %) |
| PW05 | block2 | 0/1/0 | 1/0/0 | 47,02 → 39,52 (−16,0 %) | 47,31 → 41,97 (−11,3 %) | 36,87 → 34,61 (−6,1 %) |
| PW06 | ar | 0/0/0 | 1/0/0 | 35,21 → 27,19 (−22,8 %) † | 47,07 → 32,86 (−30,2 %) | 37,83 → 32,51 (−14,1 %) |
| PW07 | d1 | 1/0/0 | 1/0/0 | 47,15 → 47,11 (−0,1 %) | 41,32 → 41,43 (+0,3 %) † | 37,63 → 37,63 (±0,0 %) |
| PW08 | d1 | 0/1/0 | 1/0/0 | 46,49 → 46,25 (−0,5 %) | 42,00 → 42,14 (+0,3 %) † | 37,70 → 37,72 (±0,0 %) |
| PW09 | d1 | 0/0/1 | 1/0/0 | 47,06 → 47,00 (−0,1 %) | 47,33 → 47,31 (±0,0 %) | 37,71 → 37,85 (+0,4 %) |
| PW10 | d1 | 0/0/0 | 0/0/0 | 46,47 → 46,21 (−0,6 %) | 47,28 → 47,36 (+0,2 %) | 37,66 → 37,77 (+0,3 %) |
| PW11 | d1 | 0/0/0 | 1/1/0 | 47,33 → 47,34 (±0,0 %) | 47,29 → 47,31 (±0,0 %) | 37,73 → 37,54 (−0,5 %) |
| PW18 | ar | 0/0/0 | 0/1/1 | 47,30 → 32,93 (−30,4 %) | 47,23 → 32,80 (−30,6 %) | 37,65 → 32,24 (−14,4 %) |
| EX13 | fresh | 1/1/1 | 0/1/0 | 47,49 → 43,87 (−7,6 %) | 47,21 → 45,07 (−4,5 %) | 36,86 → 36,92 (+0,2 %) |
| EX14 | d2 | 1/1/1 | 0/1/0 | 47,37 → 51,51 (+8,7 %) | 47,27 → 55,95 (+18,4 %) | 37,44 → 37,54 (+0,3 %) |
| EX15 | adaptive | 1/1/1 | 0/1/0 | 47,42 → 47,69 (+0,6 %) | 47,26 → 56,06 (+18,6 %) | 37,62 → 39,99 (+6,3 %) |
| EX16 | adaptive_fresh | 1/1/1 | 0/1/0 | 47,48 → 50,46 (+6,3 %) | 47,26 → 55,95 (+18,4 %) | 37,72 → 37,79 (+0,2 %) |
| EX17 | block2 | 0/1/0 | 0/1/0 | 47,38 → 39,73 (−16,2 %) | 47,04 → 41,88 (−11,0 %) | 37,66 → 35,09 (−6,8 %) |
| EX18 | ar | 0/0/0 | 1/0/1 | 47,21 → 32,95 (−30,2 %) | 47,28 → 32,77 (−30,7 %) | 37,71 → 32,30 (−14,3 %) |

## Fusions AR contre un contrôle AR

Contrôle secondaire, réservé aux lignes AR, harnais v4.

| ID | T0 | T1 | T2 |
|---|---|---|---|
| PW18 | 33,26 → 32,94 (−0,9 %) | 33,23 → 32,94 (−0,9 %) | 32,23 → 31,80 (−1,3 %) |
| EX18 | 33,25 → 33,17 (−0,2 %) | 33,12 → 33,19 (+0,2 %) | 32,54 → 32,25 (−0,9 %) |

## Lecture

- **Bruit de mesure.** L'A/A PW00 varie de +0,6 % sans aucun changement de
  calcul. Les écarts de cet ordre ne prouvent rien.
- **Facteurs isolés.** H, E, O, p1 désactivé et p2 restent dans le bruit,
  à ±0,6 % près. Aucun gain démontré.
- **Régressions nettes.** `fresh` coûte 4,5 à 8 % sur T0/T1. `block2` coûte
  6 à 16 % sur les trois entrées. Les fusions AR n'apportent rien face à AR.
- **Signaux à confirmer.** D2 et les modes adaptatifs (PW02, PW04, EX14,
  EX15, EX16) gagnent environ 18 % sur T1. Sur T0, PW02, PW04, EX14 et EX16
  gagnent 6 à 9 %, EX15 seulement +0,6 %. Sur T2, l'écart disparaît, sauf
  EX15 (+6,3 %). Ces modes existaient déjà. Les lignes EX
  combinent plusieurs facteurs : elles n'attribuent aucun gain à H, E, O ou p2.
- **Entrée longue.** Le contrôle tombe à environ 37,5 tokens/s sur T2, contre
  47,3 sur T0. C'est l'effet du contexte de 732 tokens, pas une régression.

Six paires forment un criblage, pas une preuve. Les p exploratoires ne
descendent pas sous 1/32 à cet effectif et ne sont pas corrigées.

## Confirmation en cours

Plan validé avant mesure : PW02, PW04, EX14, EX15 et EX16 sur T0, T1 et T2,
24 nouvelles paires par cellule (360 au total), un processus neuf par paire.

- Hypothèse par cellule : gain apparié médian supérieur à 2 %, test exact des
  signes, correction de Holm sur les 15 comparaisons.
- Aucune régression : borne basse de l'intervalle de confiance de la médiane
  au moins nulle sur les trois entrées.
- Aucun verdict avant les 360 paires. Les paires du criblage n'y entrent pas.

## Vérification large

Mesures CLI du 16 octobre (M5 Max, Qwen3.8-27B-4bit, sorties identiques) hors
protocole pairwise : elles ne touchent pas le défaut D2 fusionné (3 lignes).

- Kernel « skinny » u4 sur unités matricielles (`gemm_nax_skinny_u4_x3`) pour
  les vérifications de 5 à 31 lignes : T0 à draft fixe 8 passe de 28,5 à
  37,8 tok/s, un appel de vérification à 9 lignes de 140 à 75 ms. Variante
  `SARAGOSSA_RUST_QMM_SKINNY_KERNEL=bf16` (42,6 tok/s, écart 5e-3) et retrait
  `SARAGOSSA_RUST_QMM_SKINNY=0`.
- Couches linear-attention batchées sur les lignes du draft
  (`SARAGOSSA_RUST_LINEAR_ROWS_BATCHED=0` pour la boucle par ligne) : environ
  5 ms de moins par vérification à 9 lignes.
- Lookup n-grammes (`SARAGOSSA_RUST_MTP_LOOKUP_DRAFT=n`) et champ `lookup` des
  cas pairwise : mesuré en phase 6, ci-dessous.

## Phase 6 : lookup n-grammes et plafond adaptatif 4

Protocole V5 (contrôle = ligne de matrice, harnais dans l'arbre), 16 octobre,
`models/pairwise/lookup-plan.md`. Deux facteurs isolés contre le défaut par
longueur (PW04 fresh sur T0 et T1, PW03 committed sur T6 et T2), 48 paires par
cellule, 12 A/A, test des signes contre +2 %, Holm8.

| Facteur | T0 (26) | T1 (33) | T6 (424, code) | T2 (732, prose) |
|---|---:|---:|---:|---:|
| lookup 8 | −0,1 % | −4,2 % | **+21,2 %** [19,9 ; 23,0] | −0,2 % |
| plafond 4 | −9,9 % | −23,0 % | −16,2 % | −16,4 % |

- Lookup 8 : sur T6 la sortie recopie le prompt, 9 pas au lieu de 21, tout
  accepté. Sur T2 et T0 il ne propose jamais, coût nul. Sur T1 une proposition
  de deux tokens au dernier pas vaut moins que la tête MTP et coûte une
  vérification. Décision : actif par défaut quand l'historique MTP est committed
  (prompt au-delà de `SARAGOSSA_RUST_MTP_FRESH_MAX_PROMPT`, 128 tokens), inactif
  en fresh ; la variable force la valeur dans les deux cas.
- Plafond 4 : perte partout, même adaptatif. Reste à 2.
- Kernel skinny bf16 contre x3 (huit prompts, 200 tokens, draft fixe 8, hors
  protocole) : trois sorties sur huit divergent (T0, T1, T5). bf16 reste opt-in.
- Phase 6b (17 octobre, après le correctif de la ligne KV du primaire, HEAD
  `133e47d`, `models/pairwise/lookup-b/`) : lookup 8 contre PW03, 48 paires par
  cellule, 6 A/A : T6 **+21,4 %** [20,6 ; 22,6], 48/48 victoires ; T2 −0,1 %
  [−0,5 ; 0,4] ; A/A +0,2 %. Le défaut sur prompt long est confirmé.

## Cycle D2 fusionné segmenté et contexte long

`SARAGOSSA_RUST_GPU_COUNTERS=1` coupe désormais le cycle D2 fusionné en command
buffers chronométrés CPU (append différé, tête MTP, chaque couche de
vérification, queue) ; mêmes kernels, même état GPU, sortie identique. La coupe
ajoute ~0,2 ms de soumission par segment, soit ~13 ms par cycle : les totaux
segmentés dépassent le cycle non segmenté (63 contre 49 ms sur T0), les parts
restent lisibles. Profils du 17 octobre, 64 tokens, draft fixe 2 (25 cycles) :

| Segment | T0 (26 tokens) | T7 (8297 tokens) |
|---|---:|---:|
| append différé | 1,0 ms | 1,0 ms |
| tête MTP (2 drafts) | 3,9 ms | 8,5 ms |
| vérification, 16 couches full-attn | 13,3 ms (0,83 ms/couche) | 22,6 ms (1,41 ms/couche) |
| vérification, 48 couches linear-attn | 43,1 ms (0,90 ms/couche) | 47,6 ms (0,99 ms/couche) |
| queue (norme, lm_head, copies) | 2,1 ms | 2,1 ms |
| cycle | 63,4 ms | 81,8 ms |

- Surcoût à 8k : +9 ms d'attention du tronc sur 3 lignes. Chaque ligne relit K et
  V f32 en entier (8297 × 1024 × 4 o × 2 = 68 Mo par ligne, 204 Mo par couche) en
  0,58 ms, soit ~350 Go/s : le kernel 2-passes est proche de la bande passante, le
  levier est la relecture par ligne (cf. « Chemin full-attn en lignes fusionné »
  ci-dessous). +4,6 ms de tête MTP (attention sur son historique committed de 8k),
  +4,5 ms sur les couches linear-attn (à confirmer, indépendant du contexte en
  théorie).
- Indépendamment du contexte, les 48 couches linear-attn font 43 des 63 ms du
  cycle pour 3 lignes seulement. Compteurs par site (`SARAGOSSA_RUST_DECODE_PROFILE_SITES`) :
  par couche linéaire, 3 dispatches delta par ligne et 3 glissements d'état conv ;
  par couche full, 6 copies KV et 6 norm/rope par ligne ; ~1000 dispatches par cycle.
- Matmuls à M=3 (microbench `small_m_qmm_microbench`, poids chauds) : le qmv
  répliqué coûte 1,6 à 1,9 fois le M=1 (gate/up 0,175 contre 0,091 ms, down 0,139
  contre 0,078), soit ~18 ms par cycle D2 sur 64 couches. Trois kernels « R lignes
  par thread » (une lecture des poids pour R lignes, ordre d'addition conservé) ont
  été mesurés plus lents en decode réel (0,96 à 1,05 ms par couche linéaire contre
  0,85) et supprimés : à petit M le qmv est borné par la latence par thread, et les
  3× threadgroups du qmv répliqué la masquent mieux.
- Delta linear-attn par séquence avec captures (`linear_attn_gated_delta_seq_capture_dk128_tg4_f32`,
  `SARAGOSSA_RUST_LINEAR_ROWS_SEQ_DELTA=0` pour la boucle) : un dispatch au lieu de
  3 delta + 2 copies d'état par couche à M=3 ; couche linéaire 0,806 contre 0,850 ms
  (−2 ms par cycle D2), sorties T0 et T6 byte-identiques.

### Chemin full-attn en lignes fusionné (`src/decode_resident/rows.rs`)

Pour les `rows` lignes d'une vérification (position hôte, fenêtre pleine, sans
norme de V, KV f32 ou bf16), la boucle par ligne (RoPE q, RoPE k, copie K, copie V,
attention, copie du contexte : 6 dispatches par ligne, KV relu par ligne) est
remplacée par cinq dispatches pour toutes les lignes : `rms_norm_rope_heads_decode_rows_f32`
(grille têtes × lignes, position `base + ligne`) pour Q puis K, `copy_rows_at_{f32,bf16}`
pour K puis V (lignes KV contiguës, stride source de la projection qkv concaténée),
puis `attention_decode_flash_rows_*` (grille q_heads × lignes, longueur `base + ligne + 1`)
ou, dès le seuil 2-passes, `attention_decode_2pass_1_rows_d128_*` : threadgroup
(32, gqa, lignes) par (kv_head, bloc), chaque bloc KV lu une fois pour `gqa × lignes`
simdgroups, passe 2 inchangée sur `q_heads × lignes`, écriture directe dans `ctx[ligne]`.
Les lignes sont découpées en chunks bornés par le threadgroup (5 lignes à gqa 6 en
d256). Les kernels existent en d128 et d256 (le modèle a 24 têtes q, 4 têtes kv,
`head_dim` 256) : un premier essai limité à d128 retombait sur le flash générique en
lignes à 8k, 2,4 ms par couche contre 1,4 pour la boucle, d'où les variantes d256.
Kill-switch `SARAGOSSA_RUST_FULL_ROWS_FUSED=0`, diagnostic `SARAGOSSA_RUST_FULL_ROWS_GROUP=n`
(lignes par threadgroup du 2-passes). Les kernels reprennent l'arithmétique des kernels
mono-ligne dans le même ordre ; quand les lignes chevauchent une frontière de partition
(seuil flash/2-passes à 2048, nombre de blocs split-K tous les 4096 tokens), le chemin
passe en chunks d'une ligne avec le choix et les blocs de chaque ligne.
`full_attn_rows_fused_matches_row_loop` vérifie l'égalité bit à bit (contextes et KV) en
flash et en 2-passes, d128 et d256, f32 et bf16, chunks d'une ligne et frontières compris.

Mesures du 17 octobre (64 tokens, draft fixe 2, profil D2 segmenté, runs alternés OFF/ON,
sorties byte-identiques par hachage sur T0, T6, T7 et T6 avec lookup) :

| | boucle par ligne | en lignes fusionné |
|---|---:|---:|
| T0 (26 tokens), full-attn par couche | 0,77-0,79 ms | 0,74 ms |
| T0, cycle D2 | 57,4-59,2 ms | 56,4-56,7 ms |
| T0, dispatches par run | 36 311 | 31 207 |
| T7 (8297 tokens), full-attn par couche | 1,56 ms | 1,34-1,36 ms |
| T7, cycle D2 | 87,7 ms | 84,1-85,4 ms |
| T6 avec lookup, 200 tokens | 72,2 tok/s | 72,9-73,1 tok/s |

À 8k le gain par couche (−0,2 ms) reste sous l'estimation (−0,4 ms) : le KV n'est plus
relu par ligne, mais les 18 simdgroups par bloc (gqa 6 × 3 lignes) progressent moins
bien que 3 dispatches de 6. Sur T6 avec lookup, la vérification à 9 lignes (chemin
générique) coûte 68 ms par appel contre 58-60 ms pour le cycle D2 à 3 lignes : six
lignes de plus ne coûtent que 10 ms, ce qui fixe la condition du drafter externe.
Les runs où tous les segments ralentissent ensemble (couches linéaires comprises) sont
des contentions GPU externes, pas un effet du chemin : ils ont été rejoués.

## Drafter DFlash 2 (prototype CPU, devtools)

Drafter publié `z-lab/Qwen3.8-27B-DFlash2` (bf16, 3,8 Go, 5 couches, bloc 8,
cibles 5/19/33/47/61, convolution causale dynamique 2 taps par groupes de 16
autour de chaque sous-couche, sélecteur de candidats rang 256 sur le top 16 des
logits). Chargé par `SARAGOSSA_RUST_DFLASH_DRAFT_DIR`, harnais
`SARAGOSSA_RUST_DFLASH_ACCEPTANCE=1`, sémantique d'après `z-lab/dflash`
(`model.py`, MIT). Greedy sans perte : sorties identiques à l'AR sur les quatre
entrées, 64 tokens, 8 drafts par tour.

| Entrée | tours | tokens acceptés | tokens par tour (bonus inclus) | MTP D2 aujourd'hui |
|---|---:|---:|---:|---:|
| T0 (prose courte) | 17 | 46 | 3,8 | ~2,8 |
| T1 (code court) | 11 | 53 | 5,8 | ~2,8 |
| T2 (prose longue) | 20 | 44 | 3,2 | ~2,2 |
| T6 (code qui recopie) | 7 | 56 | 9,1 | 9 (lookup) |

Décision : un drafter GPU ne paie qu'avec une vérification à 9 lignes bien
moins chère que les 75 ms actuels (un pas AR coûte ~35 ms) : à 95 ms par tour,
T1 et T6 gagnent, T0 et T2 perdent. Prérequis : fusion du chemin en lignes des
couches linear-attn, puis mesure. Le prototype CPU (~11 tok/s) sert d'oracle.

### Drafter résident (18-19 octobre, mesuré)

Chantier mesuré machine libre (`dflash_resident.sh` dans le scratchpad : `oracle`,
`quant`, `bench`). Ce qui existe :

- `AffineQuantizedTensor::quantize` : quantification affine Rust (layout MLX, u4/u8
  gs64). `SARAGOSSA_RUST_DFLASH_QUANT_BITS=4|8` quantifie les projections du draft
  au chargement (fc, q/k/v/o, gate/up/down, noyaux dynamiques) : 3,8 Go bf16 →
  ~0,8 Go lus par tour. Normes, sélecteur et codebooks restent en f32.
- `decode_resident/dflash.rs` : mini-décodeur sur `block` lignes dans l'encoder
  partagé (matmuls du tronc, convolution dynamique en un kernel, RoPE/append/
  attention en lignes sur un KV de contexte par couche, repli ligne par ligne
  fenêtré au-delà de la fenêtre glissante de 2048, top-k des logits sur GPU,
  projection du sélecteur). Parcours des codebooks sur CPU. Oracle synthétique
  `dflash_resident_block_matches_cpu` (états, top-k et parcours identiques).
- `decoder/resident/dflash.rs` : boucle `generate_greedy_dflash_resident_with_options`
  (`SARAGOSSA_RUST_DFLASH_RESIDENT=1` dans le harnais d'acceptance) : proposition
  GPU, vérification résidente en lignes avec captures (rollback par troncature,
  sans rejeu, contrairement à la boucle CPU qui rejoue les tokens commités), contexte
  du drafter nourri des hidden states commités. Prefill du contexte limité aux 2048
  dernières positions du prompt (toutes les couches sont fenêtrées). Oracle modèle
  réel ignoré `dflash_resident_matches_cpu_loop_real_model` (résident == CPU == AR).
  `SARAGOSSA_RUST_DFLASH_PROFILE=1` imprime le coût moyen par tour par segment.

Oracle modèle réel (32 tokens) : résident == CPU == AR en dense et en u4, mêmes
acceptations (26 / 30). Acceptance CPU sur 64 tokens, acceptés / proposés :

| Quantification | T0 | T1 | T2 | T6 |
|---|---|---|---|---|
| dense bf16 | 46 / 62 | 53 / 60 | 44 / 63 | 56 / 56 |
| u4 gs64 | 44 / 62 | 53 / 60 | 44 / 63 | 56 / 56 |
| u8 gs64 | 45 / 62 | 53 / 60 | 44 / 63 | 56 / 56 |

u4 coûte deux acceptations sur T0 et rien ailleurs ; c'est la configuration retenue.

Première mesure : le pas bonus séquentiel après un bloc entièrement accepté (un
forward complet à 1 ligne, 33 ms) pesait 33 ms par tour sur T6. Corrigé en
vérifiant l'ancre et les 8 drafts dans un bloc de 9 lignes : la dernière ligne
donne le token suivant. 7 drafts (8 lignes) ne gagnent rien sur T0 et perdent
9 % sur T6, 8 drafts restent le défaut.

Bench apparié, résident u4 contre MTP par défaut, 200 tokens, alterné, deux
répétitions (tok/s de la boucle de decode) :

| Entrée | résident | MTP | écart | ms par tour (draft / vérif / rollback / append) |
|---|---|---|---|---|
| T0 | 42,8 / 43,0 | 45,1 / 45,0 | -5 % | 84,9 (9,6 / 73,7 / 0,9 / 0,8) |
| T1 | 64,8 / 62,6 | 47,1 / 47,2 | +35 % | 83,4 (9,5 / 72,3 / 0,7 / 0,9) |
| T2 | 39,6 / 40,1 | 42,2 / 43,8 | -7 % | 88,7 (10,4 / 76,6 / 0,9 / 0,8) |
| T6 | 103,3 / 104,4 | 65,4 / 75,0 | +45 % | 84,2 (9,1 / 74,0 / 0,0 / 1,0) |

Lecture : le tour coûte 84 à 89 ms contre 77 ms attendus, la vérification à 9
lignes 73-77 ms (68 mesurés isolément) et le drafter 9,5 ms (5 estimés, trois
synchronisations et le sélecteur CPU compris). Gain net sur le code (T1, T6),
perte de 5 à 7 % sur la prose courte et le contexte long. Un premier bench, lancé
juste après la quantification, donnait 25 à 32 tok/s sur T0 des deux côtés :
contention externe, seule la mesure appariée répétée fait foi.

Sous-profil du drafter sur T0 : 0,6 ms d'embeddings et d'encodage, 8,1 ms
d'attente GPU, 0,1 ms de readback et de sélecteur. Le coût est le travail GPU du
bloc, pas les synchronisations ni le CPU. Microbench du command buffer du
drafter (médiane de 20 runs, machine calme) :

| Segment | skinny x3 (défaut tronc) | skinny bf16 | sans skinny (NA tiled) | qmm8 |
|---|---|---|---|---|
| bloc 5 couches, 9 lignes | 4,57 | 3,74 | 7,58 | 13,2 |
| bloc à 2 lignes (qmm2) | 2,80 | | | |
| lm_head 9 lignes | 2,59 | 2,14 | 5,86 | 9,36 |
| lm_head 1 ligne (qmv) | 1,50 | 1,50 | 1,51 | 1,53 |
| candidats (lm_head + top-k + projection) | 3,12 | 2,63 | 6,46 | 9,62 |
| bloc + candidats | 7,37 | 6,10 | 13,6 | 23,2 |

Le kernel skinny lit les 800 Mo du bloc et les 636 Mo du lm_head à ~2,4× le
plancher de bande passante (le qmv à 1 ligne est au plancher). Le drafter ne
produit que des propositions, la vérification garantit la parité : il prend le
kernel skinny bf16 (portée thread-local `install_qmm_skinny_bf16_scope`, kill-switch
`SARAGOSSA_RUST_DFLASH_SKINNY_BF16=0`) alors que le tronc garde le x3, qui seul
préserve le greedy. Drafter 9,6 → 7,9 ms par tour sur T0, 9,1 → 8,2 sur T6,
acceptations identiques (145 et 177 sur 200), oracle modèle réel vert. Le top-k
restant vaut ~0,45 ms par tour (0,5 %), laissé tel quel. Le reste du drafter est
la lecture des poids par le kernel skinny, même levier que la vérification du
tronc à 9 lignes (73-77 ms, 85 % du tour).

Le bench apparié relancé après ce changement a tourné machine chargée (VM,
navigateur) : tous les segments ralentissent ensemble, MTP compris (T0 41-47,
T6 62-65 tok/s), ratios conservés (T1 +40 %, T6 +40 %, T0 -15 %, T2 -10 %),
absolus sans valeur. À rejouer machine calme. Le seuil de rentabilité sur T0 et
T2 reste 77 ms par tour, et la vérification à 9 lignes varie de 66 à 77 ms d'un
run à l'autre : T0 est dans le bruit.

### Bascule par acceptance observée (19 octobre, mesurée, écartée)

Deux politiques comparées sur 200 tokens (×3) et 500 tokens (×2), alternées
A/B/MTP par entrée, via `dflash-acceptance` :

- A : décision par requête, drafter résident seul (`SWITCH_MIN=0`).
- B : résident, puis bascule vers la boucle MTP si l'acceptance observée tombe
  sous 4 tokens par tour après 8 tours (`SARAGOSSA_RUST_DFLASH_SWITCH_MIN=4.0`,
  `SARAGOSSA_RUST_DFLASH_SWITCH_WARMUP=8`). Faute de reprise du cache DFlash
  par la boucle MTP (arène et historique MTP dimensionnés au setup), la bascule
  du harnais re-prefill prompt + généré, re-prefill compris dans le temps.

Machine chargée pendant tout le run (VM 107-267 % CPU, attente de 90 min sans
calme) : absolus bruités (vérification 87 ms par tour contre 73-77 calme),
seules les comparaisons par ligne comptent. tok/s de la boucle, 200 tokens :

| Entrée | A (résident) | B (bascule) | MTP | bascule à | re-prefill + reste |
|---|---|---|---|---|---|
| T0 | 48,1 / 34,7 / 36,5 | 42,4 / 33,9 / 35,7 | 47,1 / 46,0 / 46,7 | 35 | 4,0-4,9 s |
| T1 | 68,1 / 52,7 / 55,8 | 49,7 / 55,2 / 59,4 | 46,4 / 49,7 / 44,3 | jamais | — |
| T2 | 35,1 / 42,2 / 36,2 | 22,5 / 23,6 / 23,0 | 40,6 / 43,0 / 40,7 | 21 | 7,7-8,1 s |
| T6 | 71,9 / 86,3 / 101,2 | 77,6 / 88,9 / 101,3 | 79,3 / 79,7 / 70,9 | jamais | — |

500 tokens : T0 A 52,7 / 52,9, B 39,1 / 42,1, MTP 47,9 / 44,4 (bascule à 35,
re-prefill + reste 11-12 s) ; T1 A 64,8 / 67,3, B 66,1 / 68,1, MTP 45,8 / 47,5.

Lecture :

- Le seuil ne se déclenche jamais sur le code (T1, T6, 10 runs sur 10) et se
  déclenche au premier tour éligible sur la prose (T0 à 35 tokens, T2 à 21).
- B perd partout : son plafond est le débit MTP moins le re-prefill, et il ne
  bascule que sur les entrées où A est déjà à ±7 % de MTP. Sur T2 (prompt 732
  tokens, re-prefill ~3 s) B tombe à 23 tok/s, moitié de A.
- Une bascule intégrée (reprise du cache sans re-prefill) rapporterait au mieux
  l'écart A/MTP sur la prose, +5-7 % à 200 tokens, et rien à 500 tokens où A
  rejoint ou dépasse MTP sur T0 (52,8 contre 46,2, machine chargée). Le coût
  (refonte de `generate_greedy_mtp_inner`, arène à redimensionner) n'est pas
  justifié. Verdict : A, décision par requête ; la bascule reste un outil de
  mesure opt-in du harnais, désactivée par défaut.

Divergence T2 examinée (19 octobre) : après bascule sur T2 le harnais rapporte
`tokens_equal=false` et sort en erreur d'oracle. Ce n'est ni la jonction ni la
boucle MTP : sur T2, toutes les variantes MTP du CLI (défaut adaptatif, lookup
0, profondeur 1, historique fresh) et l'AR du CLI (`SARAGOSSA_RUST_MTP_DECODE=0`)
donnent le même texte, hachage identique sur 64 tokens, et le devtool
`SARAGOSSA_RUST_MTP_ACCEPTANCE=1` confirme MTP == AR. Ce qui diffère, c'est le
prefill : la référence AR du harnais DFlash et le drafter résident passent par
le prefill per-op avec capture de couches (`prefill_cache_state_with_layer_capture`),
le CLI et MTP par le prefill résident fusionné. `SARAGOSSA_RUST_ORACLE_DUMP_LOGITS=3`
sur l'AR du CLI avec chaque prefill : logits décalés de 0,002 à 0,08 à toutes
les positions (déjà 0,002 sur le prompt de warmup de 32 tokens), et au 17e
token généré un near-tie, 9462 contre 218318 : marge 0,016 avec le prefill
résident, 0,006 dans l'autre sens avec le per-op. Classe near-tie de la
politique du 4 juillet, pas une régression. Les kill-switches
`PREFILL_ATTN_BATCH_MID_30B=0` et `PREFILL_ATTN_STEEL_D256=0` ne changent rien
(le 27B, 24/4/256 hybride, n'est éligible à aucun des deux à 732 tokens).

Conséquence pratique pour la solution A : le prefill per-op avec capture coûtait
~21 s sur T2 contre 3 s en résident. Corrigé le 19 octobre : le prefill résident
fusionné copie sur GPU la sortie des couches cibles pendant l'encodage
(`qwen_moe_prefill_resident_with_captures`, une copie par couche capturée vers
un buffer `[n, seq, hidden]`, relu une fois et transposé en `[seq, n * hidden]`)
et `prefill_cache_state_with_layer_capture` l'emprunte avant le repli per-op.
Kill-switch inchangé : `SARAGOSSA_RUST_PREFILL_RESIDENT=0` ramène au per-op.
Mesure T2, 200 tokens, `dflash-acceptance` résident u4 (machine chargée) :

| | per-op avec capture | résident avec captures |
|---|---|---|
| `spec_ms` (prefill + boucle) | 27 474 | 8 318 |
| `ref_ms` (prefill + AR) | 30 186 | 11 194 |
| texte | ≠ AR du CLI (near-tie) | = AR du CLI |

Le drafter résident, sa référence AR et sa boucle CPU partagent désormais le
prefill de l'AR et du MTP du CLI : la divergence T2 disparaît par construction.
Oracle modèle réel toujours vert (résident == CPU == AR, 26/30, dense et u4) ;
test synthétique `prefill_resident_uses_layer_specific_attention_dimensions` :
captures byte-identiques à la sortie de chaque couche.

### Phase 7 : drafter résident contre le défaut MTP, pairwise (nuit du 19 au 20 octobre)

Question : à mode égal (greedy, committed, résident), le drafter DFlash 2 u4
(`mode: dflash`, PW23, 8 drafts par tour, skinny bf16) paie-t-il contre le défaut
de production par entrée, à 64 tokens comme les phases 5-6 et en génération
longue (500 tokens ; 350 sur T6 dont l'AR s'arrête sur EOS avant 400) ?
Harnais V5 (`Mode::Dflash`, commit `70cc013`, action bench seulement), contrôle
`row:ID` : PW04 (`adaptive_fresh`) sur T0/T1, PW03 (`adaptive`) sur T6/T2.
Huit cellules, 48 paires d'un lot alterné chacune, A/A PW04 contre PW04 sur T0
(12 paires). H0 médiane du gain ≤ +2 %, test des signes exact, Holm8 à 0,025.
Byte-identité avec l'oracle AR exigée et obtenue sur les 396 paires. Gel sur
HEAD `70cc013`, binaire archivé, poids du modèle et du drafter épinglés.
Machine sous contention toute la nuit (VM à 600 % CPU sur les 396 attentes de
machine calme) : mesure appariée et alternée, valide pour le signe, bruitée
pour l'amplitude. A/A : −0,17 %, IC [−2,1, +1,8].

| Entrée | Budget | Contrôle | Médianes contrôle / PW23 tok/s | Gain médian | IC simultané | p Holm | Acceptés/proposés | Tokens par tour DFlash / MTP |
|---|---:|---|---|---:|---|---|---|---|
| T0 (26, prose) | 64 | PW04 | 51,96 / 45,38 | −12,8 % | [−13,1, −12,4] | 1 | 44/62 (71 %) | 3,4 / 2,3 |
| T1 (33, code) | 64 | PW04 | 58,80 / 76,85 | +30,7 % | [+29,9, +32,0] | 3e-14 | 53/60 (88 %) | 5,8 / 2,8 |
| T6 (424, code) | 64 | PW03 | 63,14 / 113,60 | +81,3 % | [+78,3, +82,7] | 3e-14 | 56/56 (100 %) | 9,1 / 3,0 |
| T2 (732, prose) | 64 | PW03 | 48,75 / 35,58 | −27,0 % | [−28,8, −24,8] | 1 | 42/63 (67 %) | 2,9 / 2,2 |
| T0 (26, prose) | 500 | PW04 | 48,39 / 59,08 | +22,3 % | [+21,6, +22,8] | 3e-14 | 394/476 (83 %) | 4,7 / 2,2 |
| T1 (33, code) | 500 | PW04 | 51,24 / 74,77 | +46,6 % | [+45,1, +47,9] | 3e-14 | 416/469 (89 %) | 6,0 / 2,4 |
| T6 (424, code) | 350 | PW03 | 59,00 / 110,82 | +88,1 % | [+84,8, +90,3] | 3e-14 | 311/311 (100 %) | 9,0 / 3,0 |
| T2 (732, prose) | 500 | PW03 | 47,78 / 50,33 | +7,0 % | [+3,3, +9,0] | 2e-5 | 380/485 (78 %) | 4,2 / 2,3 |

Tokens par tour = budget divisé par le nombre de forwards du tronc
(`by_position[0]` des stats ; `verifications` compte une position de draft
examinée, pas un forward). Un tour DFlash coûte ~75-80 ms (vérification à
9 lignes et drafter), un tour MTP ~45 ms : le drafter gagne dès qu'il commet
plus de ~1,7 fois plus de tokens par tour que la tête MTP.

Lecture :

- Sur le code, gain massif et stable à toute longueur (+31 à +88 %) : 100 %
  d'acceptation sur l'édition de fichier T6, 88 % sur T1.
- Sur la prose, perte aux 64 premiers tokens (−13 % T0, −27 % T2) puis gain en
  génération longue (+22 % T0, +7 % T2 à 500). Ce n'est pas un coût fixe :
  aucun surcoût sur T1/T6 entre 64 et 500. C'est l'acceptation du début de
  réponse (phrases d'introduction : 2,9-3,4 tokens par tour) qui ne couvre pas
  le tour plus cher, alors que le corps de la réponse monte à 4,2-4,7.
- Le signe dépend donc de la longueur générée, inconnue a priori (EOS), pas
  seulement de l'entrée. Décision automatique du statut : gain sur le code et
  perte sur la prose courte, le drafter reste opt-in, défaut greedy inchangé.
  La bascule intégrée mesurée le 19 octobre (bornée à +5-7 %) reste la seule
  issue pour couvrir le début de réponse en prose.

Bruts : `models/pairwise/dflash/` (hors dépôt) : `results-live.json`,
`results.json`, `status.md`, manifeste et journal.

Mise en service (20 octobre, suite de la phase 7) : le drafter est opt-in dans
le chemin greedy de production. `SARAGOSSA_RUST_DFLASH_DRAFT_DIR=<dossier>`
sur `serve`, `run` ou la CLI le charge à côté du sidecar MTP (u4 gs64 par
défaut, `SARAGOSSA_RUST_DFLASH_QUANT_BITS=0|8` pour dense ou u8) et
`generate_greedy_mtp_inner` aiguille le greedy non contraint vers la boucle
résidente, désormais streamée et annulable ; échantillonnage, contraintes et
stop sequences gardent la tête MTP, et les harnais d'acceptance et pairwise
ne sont pas détournés. Sans la variable, rien ne change. Le prefill DFlash
passe par le cache de préfixe comme MTP : l'entrée porte en plus les captures
des couches cibles sur les 2048 dernières positions (fenêtre du drafter), un
hit est prolongé per-op sur le suffixe avec ses captures. Le seuil partagé
avec MTP au-delà duquel le prefill résident complet repart de zéro passe d'un
tiers à un huitième du prompt (mesure du 20 octobre : 31 ms par token en
extension per-op contre 3,6 ms en prefill résident). Serve multi-tour, tour 2 :
T2 (732 + 86 tokens) 2,7 s contre 3,1 s sans cache ; T7 (8297 + 54) 6,9 s
contre 49 s. Réserve : sur T7 le decode DFlash tombe à 13 tok/s contre 27 en
MTP sur les 32 premiers tokens (vérification à 9 lignes sur 8k de contexte,
début de réponse structuré), cas hors phase 7 qui confirme l'opt-in.

Extension résidente du cache de préfixe (20 octobre, mesurée la nuit du 20 au
21) : le suffixe d'un hit ne passe plus par le forward per-op mais par la
vérification en lignes du decode résident. Une arène temporaire est seedée
depuis le KV CPU de l'entrée, le suffixe est traité par blocs de 256 lignes
(`SARAGOSSA_RUST_PREFIX_EXTEND_ROWS`), les lignes K/V ajoutées sont relues
vers le cache CPU et l'arène rendue. Kill-switch
`SARAGOSSA_RUST_PREFIX_EXTEND_RESIDENT=0` (retour au per-op et au seuil d'un
huitième). Chemin partagé par l'AR, l'extension MTP committed, DFlash
(captures des couches cibles rendues par le même passage) et
`extend_prompt_state` de serve. Serve multi-tour, VM en fond (110-630 % CPU),
prefill du tour 2 (`prefill_ms`, extension en lignes contre per-op) :

| Prompt | Suffixe | MTP résident | MTP per-op | DFlash résident | DFlash per-op |
|---|---|---|---|---|---|
| T2 (732) | 86 | 664-708 ms | 2489 ms | 638-787 ms | 2270 ms |
| T7 (8297) | 54 | 1244-1303 ms | 5865 ms | 1209-1386 ms | 7222 ms |

Trace `prefix_extend` : T2 86 lignes en 467-667 ms (5,4-7,8 ms par token,
seed de l'arène 9-16 ms, relecture 2-3 ms) ; T7 54 lignes en 445-531 ms
(9 ms par token sur 8k de contexte, seed 68-121 ms, relecture 5-8 ms). Le
seed de l'arène pèse moins de 10 % du tour même sur 8k : la réutilisation de
l'arène pour le decode (v2) n'est pas justifiée. En revanche 550-670 ms du tour
T7 sont hors extension (clone de l'entrée du cache de préfixe de 8k à la
lecture puis à l'écriture) : levier suivant sur contexte long. Par token,
l'extension en lignes reste 1,5 à 2 fois plus chère que le prefill résident
complet (4,1 ms par token sur T2, 4,9 sur T7 sans semis MTP) : au-delà de la
moitié du prompt à recalculer, le prefill complet reprend la main (seuil dans
`prefix_cache_hit`, un huitième sans extension résidente).

Bascule intégrée (A DFlash, B DFlash puis repli MTP par acceptance 4,0 après 8
tours, M MTP, harnais d'acceptance, médianes, VM en fond) :

| Prompt | A 64 | B 64 | M 64 | A 500 | B 500 | M 500 |
|---|---|---|---|---|---|---|
| T0 prose | 32,7 | 29,3 | 51,4 | 56,3 | 43,5 | 46,5 |
| T2 prose | 32,2 | 32,2 | 46,7 | 50,1 | 45,2 | 46,9 |
| T1 code | 60,2 | 56,3 | 54,1 | 70,5 | 71,0 | 49,1 |
| T6 code | 110,8 | 111,9 | 75,7 | 110,4 | 111,4 | 74,5 |

Le repli retrouve bien l'entrée du cache (bascule au tour 19-35, suite
identique à l'AR partout) mais B ne rejoint M nulle part en prose : il paie
le départ DFlash lent, et à 500 tokens le DFlash seul dépasse déjà M sur T0 et
T2 (phase 7 confirmée). La bascule reste opt-in. Les cellules M à 64 tokens
(51,4 / 54,1 / 46,7) confirment que les 47-51 tok/s du contrôle d1 tiennent
toujours : les 41-46 relevés le 20 étaient des runs à 200 tokens sous charge.

Spéculatif échantillonné (20 octobre, mesuré la nuit du 20 au 21) : le cœur
MTP vérifie déjà les drafts avec température par accept/reject lossless
(`lossless_speculative_sample`, distribution du draft fournie par la tête MTP),
mais seules les requêtes greedy y étaient routées. `SARAGOSSA_RUST_MTP_SAMPLED=1`
(opt-in) étend le routage de `serve`, `run` et la CLI aux requêtes à
température > 0 sans contrainte ni stop sequence. Oracle réel : à T = 0,05 la
sortie est celle de l'AR greedy sur 32 tokens, une même graine rejoue la même
sortie, 18 drafts acceptés sur 21 à T = 0,7. Mesure CLI, 200 tokens, T = 0,7,
graine 1 : AR échantillonné 32,6 / 32,0 / 32,4 tok/s (T0 / T2 / T6) contre MTP
échantillonné 6,9 / 2,0 / 10,4. Avec température, drafts de la tête MTP,
lm_head et distributions cibles sont per-op sur CPU (493 ms par cycle sur T2,
profondeur moyenne 1,18) : le flag reste à 0 et le chantier suivant est le
draft résident avec distribution (lm_head GPU, relecture des logits, sampling
CPU).

Suite (21 octobre) : la distribution résiduelle du rejet passe en O(V) (scan
du draft par token du target = 248k² par rejet) : MTP échantillonné T2 2,0 →
8,2 tok/s. Puis chemin résident à T>0 sous le même flag : arène résidente
construite comme en greedy, draft de la tête MTP résident avec relecture de la
ligne de logits (`next_mtp_draft_resident_logits`, softmax et tirage CPU),
vérification en lignes avec rollback par captures résidentes (plus de snapshot
CPU ni de rejeu), et distribution complète sans tri du vocabulaire quand
top_p = 1. Mesure CLI, 200 tokens, T = 0,7, graine 1 :

| Prompt | AR T0,7 | MTP échantillonné nuit | MTP échantillonné résident |
|---|---|---|---|
| T0 | 32,2 | 6,9 | 31,2 |
| T2 | 32,5 | 2,0 | 29,1 |
| T6 | 31,5 | 10,4 | 33,7 |

Parité avec l'AR, pas encore de gain : le flag reste opt-in. Reste par cycle
le lm_head des lignes cibles via l'API Tensor (envoi, matmul, relecture par
ligne) et deux softmax de 248k par draft ; levier suivant = logits des lignes
cibles calculés dans le command buffer de vérification et relus en bloc.

Profil du cycle (`SARAGOSSA_RUST_MTP_STEP_PROFILE=1`, T2, 103 cycles pour
200 tokens) : verify 38 ms par cycle, et `bonus_next` 32 ms × 76 cycles : le
chemin général paie un pas de decode complet pour le token bonus après un cycle
entièrement accepté, parce que la ligne du dernier draft n'est pas vérifiée.
Deux leviers pris : les logits des lignes cibles sortent du command buffer de
vérification (`ResidentVerifyHead::Logits`, relecture en bloc), et à T>0 avec
rollback résident la ligne du dernier draft est vérifiée aussi, le bonus se tire
de sa distribution. Même protocole :

| Prompt | AR T0,7 | MTP échantillonné résident | + logits en lignes et bonus |
|---|---|---|---|
| T0 | 32,2 | 31,2 | 43,8 |
| T2 | 32,5 | 29,1 | 39,4 |
| T6 | 31,5 | 33,7 | 52,4 |

Reste par cycle ~44 ms de verify sur T2 dont les softmax CPU de 248k par
ligne. Pris ensuite : quand la distribution est complète (top_k = 0,
top_p = 1), drafts et cibles restent des softmax denses (`Vec<f32>`) et
l'accept/reject travaille dessus (`lossless_speculative_sample_dense`), sans
matérialiser 248k paires (token, probabilité) par ligne : 47,3 / 43,3 / 55,1
tok/s (T0 / T2 / T6), verify 41 ms par cycle sur T2, oracle T = 0,05
bit-identique au chemin par paires.

Oracle réel (T2, T = 0,05, graine 1, 32 tokens) : AR échantillonné et MTP
échantillonné (nuit, #32, #34) donnent le même texte, rejeu de graine
identique ; le greedy diffère d'une bascule near-tie propre au chemin T>0 (KV
bf16). `SARAGOSSA_RUST_MTP_SAMPLED` passe à 1 par défaut le 21 octobre sur
`serve`, `run` et la CLI (kill-switch `=0`).

Comparatif mlx_lm 0.32.2 (même checkpoint MLX u4 gs64, 200 tokens, 2 reps,
VM en fond) :

| Prompt | mlx_lm greedy | MTP défaut | DFlash | mlx_lm T0,7 | AR T0,7 |
|---|---|---|---|---|---|
| T0 | 32,4 | 47,6 | 49,7 | 32,3 | 32,7 |
| T1 | 32,4 | 49,6 | 73,9 | | |
| T2 | 32,2 | 45,5 | 45,8 | 32,0 | 31,8 |
| T6 | 32,3 | 75,0 | 115,1 | | |

Decode : l'AR saragossa est à parité avec mlx_lm (32 tok/s, limite mémoire du
27B 4 bits), le MTP ajoute +41 à +132 % et DFlash +42 à +256 %. Prefill : mlx
annonce 825 tok/s sur T2 (772 tokens avec template) et 727 sur T6 contre
251-304 et 262-282 pour saragossa, soit 2,6 à 3,1 fois plus vite ; les cellules
T0/T1 de mlx (131-257 tok/s) incluent sa compilation sur un prompt court et
ne comptent pas. Le prefill est le prochain levier pour le contexte long
(T7 : 40 s DFlash, 62 s MTP dont 22 s de semis CPU de l'historique MTP).

Prefill (21 octobre, profil par sections `SARAGOSSA_RUST_PREFILL_PROFILE_SECTIONS=1`,
AR greedy, VM en fond) : T2 (732 tokens, 3,44 s) MLP dense 54 %, attention
linéaire 20 %, attention causale 15 % (32 ms par couche, kernel `mid`
per-query), projections q/k/v et o 7 % ; T7 (8297 tokens, 39,2 s) MLP dense
47 %, attention causale 24 % (595 ms par couche, `batch_long_d256`), attention
linéaire 20 %, q/k/v 5 %. Le GEMM u4 tourne à 13-15 TFLOPS effectifs sur les
deux tailles, loin des ~44 que mlx_lm tire des accélérateurs neuronaux du M5
(825 tok/s sur T2, 679 sur T7) : c'est l'écart principal. Premier levier pris :
le kernel Steel causal d256 (déjà qualifié sur le 35B, facteur GQA en
paramètre) étendu au 27B (24/4 têtes) :

| Prompt | Attention causale avant | après | Prefill avant | après |
|---|---|---|---|---|
| T2 (732) | 491 ms | 20 ms | 3481 ms (210 tok/s) | 2935 ms (249 tok/s) |
| T7 (8297) | 9,18 s | 1,56 s | 43,3 s (192 tok/s) | 32,5 s (256 tok/s) |

Suites greedy identiques au binaire précédent (64 tokens sur T2, 8 sur T7),
oracle CPU à 2e-3 sur (24/4) à 257, 1025 et 2049 tokens. Kill-switch
`SARAGOSSA_RUST_PREFILL_ATTN_STEEL_D256=0`.

Deuxième levier pris : le chargeur de tuile B du GEMM u4 gs64 sur accélérateur
neuronal (`gemm_nax_coop_qb_tiled_u4` et sa variante align64, mêmes tuiles
64×64×64 et 2×2 simdgroups que le `qmm_t_nax` de MLX) dé-quantifiait poids par
poids en relisant le mot u32, le scale et le bias à chaque fois (96 lectures
device par thread et par pas de K). Il dé-quantifie maintenant une demi-ligne
par thread (4 mots, 32 poids, un scale et un bias), et la tuile threadgroup est
paddée à 72 colonnes pour étaler les fragments sur les banques. Bit-identique
au chemin u8 et au GEMM générique (oracles existants). Cumul des deux leviers,
même protocole :

| Prompt | MLP dense | Attention linéaire | Prefill avant la nuit | après |
|---|---|---|---|---|
| T2 (732) | 1761 → 820 ms | 702 → 363 ms | 3481 ms (210 tok/s) | 1754 ms (417 tok/s) |
| T7 (8297) | 19,3 → 12,8 s | 7,3 → 5,3 s | 43,3 s (192 tok/s) | 23,2 s (358 tok/s) |

Troisième levier (PR #28) : stores vectorisés de la tuile u4 (`vec<bfloat,8>`
au lieu de huit écritures scalaires) et fragments A chargés sans masque quand
M est aligné sur 32. Bit-identique aux mêmes oracles. Cumul des trois leviers :

| Prompt | MLP dense | Attention linéaire | Prefill avant la nuit | après |
|---|---|---|---|---|
| T2 (732) | 1761 → 663 ms | 702 → 309 ms | 3481 ms (210 tok/s) | 1324 ms (552 tok/s) |
| T7 (8297) | 19,3 → 6,8 s | 7,3 → 3,7 s | 43,3 s (192 tok/s) | 15,3 s (544 tok/s) |

Suites greedy identiques. Reste un facteur 1,5 contre mlx_lm (825 / 679
tok/s) : le GEMM tourne à 38-41 TFLOPS effectifs (BK=32 comme MLX essayé, sans
gain) ; puis attention linéaire (23 %, dont ~100 ms de kernels delta sur T2),
conversions f32→bf16 avant chaque GEMM (3 %), et semis CPU de l'historique MTP
sur contexte long (22 s sur T7 en MTP committed).

Plafond mesuré sur MLX (`mx.quantized_matmul`, mêmes formes, `mx.eval` par
itération) : 48 TFLOPS à M=732, 57 à M=8297, contre 38 / 41 ici : le GEMM
n'explique plus que ~130 des ~440 ms d'écart sur T2. Écartés après mesure :
delta chunké (`SARAGOSSA_RUST_LINEAR_CHUNKED=1`, section linéaire 385-417 ms
contre 319-333, texte différent) et plancher de sortie 1024 du GEMM u4 pour
les k/v_proj (qkv T7 1367 → 691 ms, T2 −20 ms, mais texte greedy T2 différent).
Pris (21 octobre) : MLP dense fusionné gate+up+SwiGLU
(`gemm_nax_coop_qb_tiled_u4_gate_up_swiglu`, deux tuiles B par pas de K,
SwiGLU f32 sur les accumulateurs, H bf16 directement en entrée du down_proj ;
kill-switch `SARAGOSSA_RUST_PREFILL_FUSED_MLP=0`) : bit-identique aux trois
passes séparées (oracle sur M=77, fragments masqués), textes identiques sur T2
et T7. Section `tail_dense`, meilleure de trois (VM en fond) : T2 697 → 582 ms,
T7 (deux runs) 9,4 → 6,5 s ; conversions f32→bf16 agrégées 48 → 21 ms sur T2.

Nettoyage du 21 octobre au soir : le scratch résident dense n'alloue plus ses
trois buffers f32 `[seq, inter]` (gate, up, hidden) quand le MLP fusionné prend
la couche, soit 3 × 0,58 Go de mémoire privée évités à 8k tokens. La décision
est prise une seule fois, à l'allocation, et l'encodage suit le scratch. Textes
greedy identiques sur T2 et T7, chemin fusionné et repli ; les temps de prefill
ne bougent pas au bruit près.

Attention causale du prefill sur les accélérateurs neuronaux (23 octobre) : à
32k, le profil par sections montrait l'attention causale à 26,6 s sur 78,9
(34 %), soit ~8 TFLOPS pour le Steel f32, contre 5,3 s (40 TFLOPS) pour
`mx.fast.scaled_dot_product_attention` aux mêmes formes. Le kernel NAX « dsplit »
de MLX 0.32.4 (`bq64_bk32_bd256_wm4_wn2`, celui que MLX choisit pour d256) est
figé dans `assets/steel_attention_nax.metal`, Q/K/V convertis en bf16, sortie
f32. Lire Q/K/V en f32 (TF32) a été mesuré plus lent que le Steel (35,4 s).

| Prompt | Attention causale Steel | NAX | Prefill Steel | NAX |
|---|---|---|---|---|
| 8k | 1,38 s | 0,43 s | 635 tok/s | 676 tok/s |
| 32k | 26,6 s | 6,7 s | 405 tok/s | 578 tok/s |

Oracle CPU (entrées arrondies en bf16) à 2,5e-4 sur (16, 2) et (24, 4), lignes
de chaque groupe et tuiles partielles. Sur `serve` (ctxbench, 7 contextes 4k à
32k, greedy) : TTFT 32k 83 → 63 s, 16k 35 → 29 s, intégrité 4/4 partout, textes
identiques sur 4 contextes. Les 3 autres divergent sur un near-tie (marges
top1-top2 au premier écart 0,0001 à 0,011 contre 2,3 en médiane, mêmes deux
tokens inversés, logit top1 décalé d'au plus 0,16 avant l'écart). Défaut ON,
kill-switch `SARAGOSSA_RUST_PREFILL_ATTN_NAX=0`.

### Prefill résident par morceaux (24 oct. 2026)

Le prefill résident traite le prompt par morceaux égaux d'au plus 4096 tokens
(`SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK`, 0 pour désactiver). Les buffers
intermédiaires sont dimensionnés sur un morceau ; seuls le KV pleine longueur
des couches d'attention complète et l'état conv/SSM des couches linéaires
survivent d'un morceau au suivant. La RoPE reçoit la position absolue de la
première ligne, l'attention Steel/NAX lit les clés `[0, fin du morceau)` avec
`qL_off`. Les modèles hors périmètre Steel/NAX d256 restent en un morceau.

| 27B, `saragossa bench` | un morceau | morceaux 4096 |
|---|---|---|
| pic de footprint 32k | 46,0 Go | 30,4 Go |
| prefill 32k | 594 tok/s | 617 tok/s |
| pic de footprint 64k | 73 Go | 38,1 Go |

L'oracle synthétique (couche linéaire dk128 + attention gated 16/2 d256,
1100 tokens en 3 morceaux) est bit-exact ; les textes greedy AR serve sont
identiques sur 8k, 16k et 32k.

## Limites

- Une seule machine et trois prompts. Aucune garantie pour d'autres entrées.
- Température, fréquence et charge GPU non mesurées.
- Énergie par token, cycles matériels, temps GPU, RSS et pic mémoire exact
  restent inconnus. Ils ne sont pas notés zéro.
- Les valeurs par défaut de production ne changent pas. Le code expérimental
  des variantes n'est pas inclus dans ce document.
