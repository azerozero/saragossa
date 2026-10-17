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

## Limites

- Une seule machine et trois prompts. Aucune garantie pour d'autres entrées.
- Température, fréquence et charge GPU non mesurées.
- Énergie par token, cycles matériels, temps GPU, RSS et pic mémoire exact
  restent inconnus. Ils ne sont pas notés zéro.
- Les valeurs par défaut de production ne changent pas. Le code expérimental
  des variantes n'est pas inclus dans ce document.
