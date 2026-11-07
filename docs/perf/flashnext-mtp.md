# Qwen3.8-Flash-Next : tête MTP et décodage spéculatif exact

Étape 6 du portage `qwen4_exp`. Elle porte la tête MTP du checkpoint, la
branche sur la session résidente et en tire un décodage spéculatif glouton dont
la sortie est **identique au bit près** à l'AR glouton. Elle mesure ensuite ce
que coûterait une vérification en lignes (D) avant d'en écrire le code.

Bilan :
- la tête MTP, résidente, est validée contre une référence CPU et contre
  transformers (A, B) ;
- le spéculatif exact en pas mono (C) est vert : identité sur 8 prompts × 3
  profondeurs, retour arrière contrôlé par oracle d'état ;
- il est **plus lent que l'AR** (×0,53 à ×0,84), puisque la vérification de R
  lignes coûte R pas ;
- le profil de D chiffre le gain d'une vérification en lignes : ×0,88 au mieux
  en code à profondeur 3 dans l'hypothèse réaliste, sous le seuil de ×1,2 fixé
  avant la mesure. **D s'arrête.**

Le spéculatif reste **désactivé par défaut** sur Flash-Next. C sert de base
exacte : retour arrière, historique MTP et identité.

## Contrat

- **Historique MTP.** La ligne `j` contient la paire `(h_j, tok_{j+1})`, où
  `h_j` est le résiduel hc du tronc **vérifié**, pris avant le mixer final. Elle
  porte la position RoPE `j`, et ses blocs d'indexeur sont comptés en indices
  de ligne. Un décalage global est invariant (RoPE relative) ; seuls des
  décalages de requêtes seules ou de clés seules sont détectables, et les
  mutations les utilisent.
- **Entrée** (transcrite de `graph_mtp` de llama.cpp) :
  `e = rms(embed(tok))·(1+w_e)`, répété sur les 4 flux ;
  `h_s = rms_flux(h)·(1+w_h)` ;
  `res_s = fc_embedding·e + fc_hidden·h_s`.
  Viennent ensuite une couche QSA + MoE en hyper-connection, sans PLE, puis le
  mixer de tête et le `lm_head` partagé.
- **Normes `1+w`** appliquées une seule fois : le kernel de norme groupée ajoute
  déjà le `1`, la tête charge les poids bruts.
- **Les drafts peuvent être inexacts** : seule la sortie vérifiée compte.

## A. Référence CPU

`src/qwen4_exp/mtp.rs` porte `mtp_input`, `mtp_layer` et `mtp_head`.
`tools/flashnext/oracle_dump.py --mtp` assemble la couche MTP à partir des
blocs transformers du tronc et produit `mtp_golden.safetensors` (4095 paires,
dont 2044 en régime QSA creux).

Le test `n2_mtp_head_matches_transformers_blocks` contrôle la provenance du
golden, l'entrée et la sortie à 1e-4, quatre mutations (paires décalées, flux
permutés, gain absent, gain doublé), et l'accord du draft avec le tronc
(plancher ½, mesuré 49/64).

## B. Tête résidente

- Les poids `mtp.*` sont un sidecar indexé après validation ; une collision de
  noms est refusée.
- La couche MTP passe par le **même encodeur de couche** que le tronc
  (`LayerEncoder`). Le refactor a été prouvé au bit près : 8 pas, dont 2 en
  régime creux, ont donné le même dump binaire avant et après.
- Un pas MTP = un command buffer, avec la même gestion d'échec qu'un pas de
  tronc.

**N2, GPU contre CPU sur les entrées du golden** (ré-ancré ligne par ligne) :

| porte | seuil | mesuré |
|---|---|---|
| sortie hc | 1e-4 | 4,2e-5 |
| scores de l'indexeur | 1e-5 | 4,2e-6 |
| KL des logits de tête | 1e-3 | 2,3e-9 |
| sortie contre transformers | 1e-4 | 1,05e-5 |

Onze mutations rougissent toutes, dont :
- mauvais expert, mauvais bloc ;
- gain absent ou doublé ;
- requêtes QSA seules décalées ;
- requêtes d'indexeur seules décalées, jugées sur la porte des scores
  (≥ 2 × 1e-5 sur 256 lignes creuses) et en variante de 1024 sur la sélection ;
- historique figé.

Mémoire : la session à 256k alloue 14,64 Go d'état, égal au budget calculé.
Tronc inchangé : ABBA à 1k, −0,13 % [−0,52 ; +0,25].

## C. Spéculatif exact en pas mono

Un cycle, à profondeur d :
1. `D_1 = MTP(h_p, y)` sur la ligne vérifiée `p`, puis
   `D_i = MTP(h'_{i-1}, D_{i-1})` ;
2. `verify([y, D_1 … D_d])` : R = d + 1 pas mono dans **un** command buffer,
   avec captures par ligne (états DeltaNet conv et SSM, clés brutes de
   l'indexeur, anneau PLE, résiduel hc) ;
3. k = plus long préfixe où `D_i` égale l'argmax de la ligne précédente, puis
   `accept(k)` restaure l'état de la ligne k ;
4. l'historique MTP est tronqué à `p + 1`, puis reçoit les paires vérifiées ;
   le jeton en attente devient `a_k`.

Les drafts enchaînés écrivent dans l'anneau des clés brutes de l'indexeur
MTP : un instantané, pris après le premier pas MTP du cycle, est restauré par
la troncature.

En cas d'échec en cours de cycle, y compris pendant la reconstruction après
`accept`, l'historique MTP revient aux `p + 1` paires vérifiées, clés brutes
comprises. Le tronc garde ce qu'il a engagé, et toute reprise passe par un
`reset`.

Conventions alignées sur l'AR :
- **arrêt** : comme en AR ;
- **dernier jeton émis** : jamais consommé par le tronc. Un cycle émet au plus
  le budget restant moins un jeton.

### Identité (porte dure)

Test `speculative_greedy_matches_ar_bitwise`, commit 69eeb81 : 1977 s, pic
78,6 Go.

| prompt | d1 | d2 | d3 (jetons par cycle) |
|---|---|---|---|
| golden 64 | 1,92 | 2,63 | 3,28 |
| golden 512 | 1,81 | 2,34 | 2,55 |
| golden 2040 | 1,98 | 2,88 | 3,82 |
| golden 3000 (creux) | 1,77 | 2,36 | 2,71 |
| wikitext 1024 | 1,86 | 2,53 | 3,11 |
| wikitext 2600 (creux) | 1,59 | 1,82 | 1,93 |
| code4k-1 | 1,89 | 2,60 | 3,20 |
| code4k-2 | 1,85 | 2,53 | 2,94 |

Les 24 sorties sont identiques à l'AR. Cas limites couverts :
- rejet forcé : 0 draft accepté sur 63 cycles, sortie identique ;
- arrêt sur un draft accepté et sur un draft rejeté ;
- capacité limite ;
- profondeur 0 (délégation à l'AR) ;
- budgets 0 et 1 ; `usize::MAX` refusé ;
- échec injecté au cycle 3, suivi d'un historique MTP propre.

Le draft `D_1` est contrôlé contre un rejeu séquentiel des paires AR : 0 écart
sur 100 cycles (golden 512) et 80 cycles (code4k-1). Deux mutations rougissent
ce contrôle :

| mutation | golden 512 | code4k-1 |
|---|---|---|
| troncature oubliée | 3 écarts | 2 écarts |
| historique construit sur des `h` spéculatifs | 4 écarts | 6 écarts |

`resident_verify_rows_and_accept_match_ar_bitwise` vérifie de son côté que
`verify` puis `accept(k)` laissent, pour k = 0 à R−1, un état égal au bit près
à celui de l'AR (empreinte de tous les états), aux frontières 2046 à 2049, sous
six mutations.

### Débit (sans porte)

Test `bench_speculative_mono_vs_ar`, M3 Ultra, 256 jetons, 2 répétitions.

| | AR | d1 | d2 | d3 |
|---|---|---|---|---|
| prose (golden 512) | 45,8–46,5 tok/s | ×0,80 | ×0,66 | ×0,53–0,54 |
| code (code4k-1) | 45,2–46,2 tok/s | ×0,83–0,84 | ×0,73 | ×0,66–0,67 |

- Pas AR : 21,49 ms. Pas MTP : 3,60 ms. `T_mtp/T_ar = 0,168`, mesuré sur un
  historique MTP court (64 lignes), où l'attention de la tête est négligeable.
  L'argmax du `lm_head` pèse 2,93 ms de ce pas.
- Modèle d'un cycle : `gain = n / [ρ(R) + m·T_mtp/T_ar + h]`, où :
  - n : jetons par cycle ;
  - m = d + n − 1 : pas MTP du cycle (d drafts, puis n − 1 paires
    reconstruites), soit n + 2 à profondeur 3 ;
  - h : coût hôte, sans dimension.

  En pas mono, ρ(R) = R. Le modèle reproduit C avec h ≈ 0 : en code à
  profondeur 3, ×0,66 prédit pour ×0,665 mesuré.

## D. Vérification en lignes : profil avant code

D remplacerait les R pas mono de `verify` par un passage du tronc sur R lignes,
sans changer un bit. Le gain viendrait d'une lecture unique des poids pour R
lignes. Avant d'écrire ce code, chaque brique est profilée en lignes avec les
kernels existants.

### Méthode

- **Matmuls à poids**
  (`profile_rows_matmuls_flash_next_shapes`) : formes réelles, poids
  synthétiques, mode normal et strict, R = 1 à 4.
  - Chaque command buffer lit des copies distinctes du poids (≥ 512 Mo, une
    lecture par copie), pour mesurer depuis la DRAM comme le decode. Un premier
    jet relisait vingt fois le même poids depuis le SLC et donnait des ρ < 1.
  - Chauffe, puis minimum de 3 tours alternés mono et lots.
  - L'exactitude de chaque ligne contre le pas mono est vérifiée par
    `to_bits()`.
- **Reste du pas** (`profile_resident_decode_after_1k_context`, avec
  `SARAGOSSA_FLASH_NEXT_PROFILE=1` et `SARAGOSSA_RUST_DECODE_PROFILE=1`), sur
  les vrais poids après 1k de contexte :
  - temps GPU et dispatches par brique ;
  - latence d'un dispatch suivi d'une barrière, part qu'un kernel en lignes
    garde une seule fois pour R lignes ;
  - argmax de tête en lignes sur le vrai `lm_head` u6, contre R argmax mono.
- **Chevauchement des experts** entre jetons consécutifs : routage du golden
  long, 48 couches × 4096 positions.

### Hypothèses du modèle, fixées avant la re-mesure

Les relectures ont montré que la première mesure lisait un seul buffer GPU
pour toutes les « copies » (cache indexé par `weight_id`). Les chiffres des
sections suivantes viennent de la re-mesure (c96e469) et des mesures in situ (994d9aa). Les définitions ci-dessous ont été
écrites et commitées **avant** la re-mesure, pour ne pas être ajustées après
coup. La règle de décision, fixée par avance, ne change pas : D s'arrête si
aucune option ne dépasse ×1,2 en code à profondeur 3 dans l'hypothèse
réaliste.

**Formule.** `gain = n / [ρ(R) + m·T_mtp/T_ar + h]`, avec :
- n les jetons par cycle mesurés en C ;
- m = d + n − 1 pas MTP par cycle (d drafts, puis n − 1 paires reconstruites),
  soit n + 2 à profondeur 3 ;
- h = 0, calé sur C (code d3 : prédit ×0,66, mesuré ×0,665) ;
- `T_mtp/T_ar` mesuré en temps mural ;
- ρ(R) = temps GPU des R lignes / temps GPU d'un pas mono, sommés sur les
  briques du profil.

Décision sur le code à profondeur 3 : n = 3,20, R = 4.

**Hypothèse réaliste** (chaque facteur est mesuré, sauf le travail du reste, qui
est sans amortissement faute de kernel en lignes) :
- **matmuls à poids** : ρ(R) strict mesuré par forme, sur des poids distincts
  lus depuis la DRAM ;
- **hc Down** : temps mono du split-K de production, multiplié par le ρ(R) du
  matmul générique de même forme (témoin : il n'existe pas de split-K en
  lignes) ;
- **experts routés** : union des experts distincts, chacun lu une fois.
  Facteur = `Σ_m N_m · ρ_expert(m) / 10`, où :
  - N_m est la multiplicité mesurée sur le golden long
    (`tools/flashnext/expert_overlap.py`) : nombre moyen d'experts distincts
    servant exactement m des R lignes ;
  - ρ_expert(m) est le ρ strict mesuré d'un même expert lu pour m lignes ;
  - gate/up et down sont pondérés par leur temps mono ;
- **argmax de tête** : ρ(R) mesuré du kernel en lignes existant, sur le vrai
  `lm_head` u6 ;
- **reste de chaque brique** (hors matmuls à poids) :
  - la latence d'un dispatch suivi de sa barrière, mesurée sur la copie de
    10 240 mots, est comptée une fois pour R lignes ;
  - le travail restant est multiplié par R ;
- **PLE** : × R, séquentiel.

**Hypothèse optimiste** : comme la réaliste, avec deux exceptions supposées et
non mesurées :
- travail du reste × 1,2 ;
- argmax de tête par un qmv-T u6 exact, à écrire, au ρ(R) du `q_proj`.

**Borne basse** (kernels en lignes idéaux partout) :
- tout ce qui n'est pas un expert routé est à ρ = 1 : lecture unique et calcul
  gratuit ;
- experts routés à `Σ_m N_m / 10`, soit les experts distincts lus une fois.

**Variante** publiée à part, sans entrer dans la décision : le scan DeltaNet
lit son état une seule fois pour R lignes. L'état fait 36 couches × 48 têtes ×
128 × 128 × 4 octets = 113 Mo, lus et écrits ; c'est le temps soustrait au
travail × R.

**Options :**
- D complet ;
- draft sur sous-vocabulaire seul : 32k jetons de vocabulaire de draft, le
  `lm_head` du pas MTP mis à l'échelle 32 000 / 248 320, acceptation supposée
  inchangée ;
- les deux.

### Mesures complémentaires, combinaison fixée avant la mesure

La re-mesure sur poids distincts tombe au seuil (×1,19 à ×1,23 selon le
traitement du scan). Ce paragraphe a été écrit et commité **avant** les deux
mesures qui suivent ; il dit comment elles remplacent les postes concernés du
réaliste.

**(1) DeltaNet et MoE in situ.** Vrais poids de la session après 1k de
contexte : 36 et 48 couches distinctes, donc lues depuis la DRAM comme en
decode. Chaque mesure est faite sur toutes les couches dans un command buffer,
après une chauffe, au minimum de 3.
- **DeltaNet.**
  - `T_dn(1)` : les 36 couches résidentes
    (`encode_linear_attn_resident_dense_buffers`), sur leurs vrais états.
  - `M_dn(R)` : les matmuls de ces couches (`in_proj_qkv`, `in_proj_z`,
    `out_proj`, `in_proj_a/b`) sur R lignes, en vérification greedy stricte,
    avec l'exactitude au bit contre le mono.
  - Reste in situ = `T_dn(1) − M_dn(1)` : scan, conv, normes, portes. Un reste
    négatif invalide la mesure et est publié comme tel.
  - `DeltaNet(R) = M_dn(R) + R × reste` : pas de kernel en lignes pour l2/σ, scan
    séquentiel.
- **MoE.**
  - `T_moe(R)` : les 48 couches MoE (routeur, top-k, experts, expert partagé) en
    R lignes par le kernel existant (`encode_moe_shared_buffers_verify_rows`,
    routage par ligne, sans union), en vérification greedy stricte, avec
    l'exactitude au bit de chaque ligne contre le MoE mono.
  - Si les lignes sont exactes : `MoE(R) = T_moe(R)`.
  - Sinon : `MoE(R) = T_moe(1) × F_union(R)`, où `F_union` est le facteur des
    experts défini plus haut (`Σ_m N_m · ρ_expert(m) / 10`, pondéré). L'écart
    d'exactitude est publié.
- **Combinaison.** Chaque brique garde son temps mono du profil par brique et
  prend le rapport mesuré : DeltaNet × `DeltaNet(R) / T_dn(1)`, MoE ×
  `MoE(R) / T_moe(1)`. Hyper-connections, QSA, tête et PLE gardent les postes
  réalistes de la re-mesure (commit c96e469).

**(2) Acceptation d'un draft sur sous-vocabulaire.**
- **Sous-vocabulaire.** V_sub = les 32 000 jetons les plus fréquents, ou tous
  les jetons vus s'il y en a moins, plus EOS. Corpus : wikitext `corpus32k`,
  `code4k-2`, `code4k-3` et `chat-1` à `chat-6` (fixtures `rowsexact`), tous
  distincts des prompts d'évaluation.
- **Évaluation.** C à profondeur 3, 256 jetons :
  - prose : golden 512 ;
  - code : `code4k-1`, 1024 jetons.

  Les drafts sont l'argmax des logits MTP restreint à V_sub. L'identité à l'AR
  reste vérifiée.
- **Modèle.**
  - `n_sub` (jetons par cycle mesurés) remplace n dans les options à
    sous-vocabulaire, avec m = d + n_sub − 1.
  - `T_mtp_sub = T_mtp − T_tête × (1 − |V_sub| / V)`, où `T_tête` = 2,93 ms
    (argmax mono mesuré).

**Décision.** Réaliste, code à profondeur 3, seuil ×1,2, inchangé. Si la
meilleure option passe ×1,2 en code mais reste sous ×1 en prose, D ne sera
activable que par opt-in, jamais par défaut.

### Matmuls à poids (re-mesure, commit c96e469)

Mode strict, poids synthétiques aux formes réelles. Copies distinctes, adresses
GPU vérifiées, lues depuis la DRAM ; minimum de 3 après chauffe. Toutes les
lignes sont exactes au bit près.

| matmul | forme, bits | ×/pas | mono | Go/s | ρ(2) | ρ(3) | ρ(4) |
|---|---|---|---|---|---|---|---|
| q_proj (q + gate) | 12288×2560 u8 | 12 | 73,0 µs | 458 | 1,01 | 1,26 | 1,46 |
| k/v_proj | 512×2560 u8 | 24 | 8,8 µs | 159 | 0,98 | 1,01 | 1,05 |
| o_proj | 2560×6144 u8 | 12 | 38,5 µs | 434 | 1,00 | 1,36 | 1,63 |
| DeltaNet in_proj_qkv | 10240×2560 u8 | 36 | 63,2 µs | 440 | 1,01 | 1,20 | 1,46 |
| DeltaNet in_proj_z | 6144×2560 u8 | 36 | 40,0 µs | 418 | 0,99 | 1,21 | 1,43 |
| DeltaNet out_proj | 2560×6144 u8 | 36 | 38,8 µs | 431 | 1,01 | 1,37 | 1,66 |
| DeltaNet in_proj_a/b (f32) | 48×2560 | 72 | 7,2 µs | 68 | 1,01 | 1,02 | 1,03 |
| hc Up | 10240×320 u8 | 96 | 12,2 µs | 285 | 1,25 | 1,51 | 1,82 |
| hc Down, matmul générique | 320×10240 u8 | 96 | 25,9 µs | 134 | 0,97 | 0,98 | 1,00 |
| hc Down, split-K de production | 320×10240 u8 | 96 | 16,6 µs | 210 | — | — | — |
| expert gate/up | 640×2560 u4 | 960 | 5,4 µs | 170 | 1,85 | 1,70 | 1,77 |
| expert down | 2560×640 u4 | 480 | 5,7 µs | 162 | 1,08 | 1,28 | 1,51 |
| partagé gate/up | 640×2560 u8 | 96 | 9,1 µs | 191 | 0,97 | 1,03 | 1,07 |
| partagé down | 2560×640 u8 | 48 | 7,1 µs | 247 | 1,07 | 1,25 | 1,36 |
| `index_qk_proj` | 640×2560 u8 | 12 | 9,1 µs | 192 | 0,97 | 1,03 | 1,07 |
| routeur (f32) | 512×2560 | 48 | 13,0 µs | 403 | 1,08 | 1,13 | 1,32 |

Les ρ des experts portent sur un même expert lu pour 2, 3 ou 4 lignes : ce
sont les `ρ_expert(m)` du facteur des experts.

**Chevauchement des experts** (`tools/flashnext/expert_overlap.py`, golden long,
48 couches × 4096 positions). Experts distincts parmi les top-10 de R jetons
consécutifs :
- R = 2 : 16,84 sur 20 ;
- R = 3 : 22,67 sur 30 ;
- R = 4 : 27,93 sur 40, de multiplicité N_1 = 19,54, N_2 = 5,55, N_3 = 2,00,
  N_4 = 0,84.

Facteur des experts à R = 4 :
- gate/up 3,47, down 2,94, pondéré 3,29 ;
- union seule (Σ N_m / 10) : 2,79.

### Reste du pas

Profil par brique après 1k, sur 128 pas : 20,65 ms GPU, 1627 dispatches par
pas. Un dispatch suivi de sa barrière coûte 0,97 µs (copie de 10 240 mots ;
0,92 µs pour 1 mot, 1,16 µs pour 40 960 mots). Argmax de tête en lignes sur le
vrai `lm_head` u6 : 2,93 ms en mono, ρ = 1,94, 2,88 et 3,86 pour R = 2, 3 et
4, avec les mêmes indices. Il n'y a pas de qmv-T u6 : le kernel relit les poids
par ligne.

| brique | mono (ms) | matmuls (micro-bench) | reste | dont latence |
|---|---|---|---|---|
| hyper-connections | 3,41 | 2,77 | 0,65 | 0,47 (480 dispatches) |
| QSA (dense, 1k) | 2,09 | 1,66 | 0,43 | 0,10 (99 dispatches) |
| DeltaNet | 5,07 | 5,63 | **négatif** | — |
| MoE, routeur compris | 7,22 | 9,76 | **négatif** | — |
| `lm_head` (argmax) | 2,75 | — | — | — |
| PLE et mixer final | 0,12 | — | — | — |

Pour DeltaNet et le MoE, les matmuls isolés lus depuis la DRAM coûtent plus
que la brique entière dans le pas : les experts isolés coûtent plus que le
kernel MoE fusionné, et les dispatches isolés plus que les dispatches
enchaînés. La soustraction ne donne donc pas de reste. D'où les mesures in
situ.

### Mesures in situ (commit 994d9aa)

Vrais poids de la session après 1k, toutes les couches dans un command buffer,
vérification greedy stricte.

- **DeltaNet, 36 couches.**
  - Couche entière : 5,499 ms.
  - Matmuls de ces couches : 4,609 ms, d'où un **reste in situ de 0,890 ms**
    (scan, conv, normes, portes).
  - Matmuls en lignes : 4,735 ms (ρ = 1,03) à R = 2, 5,913 ms (ρ = 1,28) à
    R = 3, 7,059 ms (ρ = 1,53) à R = 4.
  - Lignes exactes.
- **MoE, 48 couches.**
  - Mono : 6,834 ms.
  - Kernel de vérification en lignes existant : 20,649 ms (ρ = 3,02) à R = 2,
    23,373 ms (ρ = 3,42) à R = 3, 26,223 ms (ρ = 3,84) à R = 4.
  - Lignes exactes au bit sur les 48 couches.
  - Ce kernel route chaque ligne sans union : dès deux lignes, il coûte plus
    que deux MoE mono.

### Acceptation d'un draft sur sous-vocabulaire (commit 994d9aa)

Le corpus distinct (41 464 jetons) ne contient que **5 623 jetons distincts**,
EOS compris : V_sub est donc l'ensemble des jetons vus, pas 32 000. La sortie
AR évaluée est couverte à 196/256 en code et à 134/256 en prose.

| | drafts pleins | drafts restreints |
|---|---|---|
| code (`code4k-1`), profondeur 3 | 3,19 jetons par cycle | **2,41** |
| prose (golden 512), profondeur 3 | 2,55 | 1,71 |

Les sorties restent identiques à l'AR. Le pas MTP restreint est estimé à
`T_mtp_sub = 3,60 − 2,93 × (1 − 5 623 / 248 320) = 0,74 ms`, soit
`T_mtp_sub/T_ar = 0,034`.

### Modèle à R = 4, combinaison e5ef5b6 appliquée telle quelle

| brique | mono (ms) | R = 4 réaliste (ms) | source |
|---|---|---|---|
| PLE | 0,09 | 0,37 | × R |
| hyper-connections | 3,41 | 4,91 | Up × 1,82, Down split-K × 1,00, latence une fois, travail × 4 |
| DeltaNet | 5,07 | 9,78 | × (7,059 + 4 × 0,890) / 5,499 = × 1,93 |
| QSA | 2,09 | 3,80 | matmuls aux ρ stricts, latence une fois, travail × 4 |
| MoE | 7,22 | 27,69 | × 26,223 / 6,834 = × 3,84 (lignes exactes) |
| mixer final | 0,03 | 0,04 | matmuls aux ρ stricts |
| `lm_head` (argmax) | 2,75 | 10,61 | × 3,86 |
| **total** | **20,65** | **57,20** | **ρ(4) = 2,77** |

**Gain modélisé**, réaliste. `T_mtp/T_ar = 0,168` et m = d + n − 1 pas MTP par
cycle.

| option | code d3 | prose d3 |
|---|---|---|
| D complet (n = 3,20 / 2,55) | **×0,88** | ×0,72 |
| draft sur sous-vocabulaire seul (n_sub = 2,41 / 1,71) | **×0,58** | ×0,41 |
| les deux | **×0,83** | ×0,59 |
| pour mémoire : C actuel, mesuré | ×0,665 | ×0,53 |

À titre indicatif, hors règle :
- une variante favorable (travail du reste × 1,2, qmv-T u6 pour la tête,
  union des experts au facteur 3,29 sur toute la brique MoE) donne ρ(4) =
  2,08, soit ×1,09 pour D complet et ×1,08 pour les deux en code ;
- même avec une acceptation inchangée sous sous-vocabulaire (n = 3,20), les
  deux options combinées donnent ×1,09 en réaliste et ×1,42 dans cette
  variante.

### Décision

Aucune option ne dépasse ×1,2 en code à profondeur 3 dans l'hypothèse
réaliste : **D s'arrête**, selon la règle fixée avant la mesure. La mention
« opt-in seulement » ne s'applique pas, aucune option ne passant en code.

- Le MoE domine : en lignes, il coûte ×3,84 à R = 4 avec le kernel exact
  existant, et l'union des experts ne ramènerait que vers ×3,3, car seuls
  30 % des experts sont partagés entre jetons consécutifs.
- L'argmax de tête relit le `lm_head` par ligne (×3,86).
- Le reste DeltaNet in situ, 0,89 ms par pas, est séquentiel.
- Le draft sur sous-vocabulaire perd plus en acceptation (3,19 → 2,41 jetons
  par cycle en code) qu'il ne gagne sur le pas MTP. Le corpus distinct n'a
  fourni que 5 623 jetons ; un corpus plus large réduirait cette perte sans
  dépasser le cas d'une acceptation inchangée (×1,09).
- C reste en place comme base exacte. Le spéculatif reste désactivé par défaut
  sur Flash-Next et n'est atteint par aucun point d'entrée (`serve`, `run`,
  CLI).

## Reproduire

```bash
export SARAGOSSA_QWEN4EXP_DIR=…/Qwen3.8-Flash-Next-MLX-q4e8
export SARAGOSSA_QWEN4EXP_GOLDEN=…/n2_golden_v3
export SARAGOSSA_FLASH_NEXT_CORPUS=…/corpus32k
T=decode_resident::flash_next::tests
cargo test --release --features metal,devtools --lib -- --ignored --nocapture \
  $T::speculative_greedy_matches_ar_bitwise
cargo test --release --features metal,devtools --lib -- --ignored --nocapture \
  $T::bench_speculative_mono_vs_ar
cargo test --release --features metal,devtools --lib -- --ignored --nocapture \
  $T::profile_rows_matmuls_flash_next_shapes
cargo test --release --features metal,devtools --lib -- --ignored --nocapture \
  $T::measure_subvocab_draft_acceptance
python -I tools/flashnext/expert_overlap.py …/n2_golden_v3/long.safetensors
SARAGOSSA_FLASH_NEXT_PROFILE=1 SARAGOSSA_RUST_DECODE_PROFILE=1 \
  cargo test --release --features metal,devtools --lib -- --ignored --nocapture \
  $T::profile_resident_decode_after_1k_context   # briques, reste, in situ
```

Les tests GPU tournent seuls sur la machine ; aucune mesure de débit n'est
prise pendant une autre charge GPU.
