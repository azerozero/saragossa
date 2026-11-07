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
- le profil de D chiffre le gain d'une vérification en lignes. La règle fixée
  avant la mesure : si aucune option ne dépasse ×1,2 en code à profondeur 3
  dans l'hypothèse réaliste, D s'arrête.

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

Le moindre échec en cours de cycle ramène l'historique MTP à son dernier état
propre. La convention d'arrêt est celle de l'AR.

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

- Pas AR : 21,49 ms. Pas MTP : 3,60 ms. `T_mtp/T_ar = 0,168`. Le `lm_head`
  pèse environ 2,8 ms du pas MTP.
- Modèle d'un cycle : `gain = n / [ρ(R) + (n+2)·T_mtp/T_ar + h]`, avec n les
  jetons par cycle et h le coût hôte. En pas mono, ρ(R) = R. Il reproduit C
  avec h ≈ 0 : code d3, 0,66 prédit et 0,665 mesuré.

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
sections suivantes sont donc à remplacer. Les définitions ci-dessous ont été
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

### Matmuls à poids

> Chiffres de la première mesure (poids aliasés), à remplacer.

Dernier passage (commit 267728a). Mono à ~550 Go/s. Toutes les lignes sont exactes au bit près, en normal comme
en strict.

Mode strict :

| matmul | forme, bits | ×/pas | mono | ρ(2) | ρ(3) | ρ(4) |
|---|---|---|---|---|---|---|
| q_proj (q + gate) | 12288×2560 u8 | 12 | 55,1 µs | 1,09 | 1,72 | 2,26 |
| k/v_proj | 512×2560 u8 | 24 | 5,1 µs | 1,20 | 1,16 | 1,24 |
| o_proj | 2560×6144 u8 | 12 | 23,3 µs | 1,03 | 1,53 | 1,92 |
| DeltaNet in_proj_qkv | 10240×2560 u8 | 36 | 42,1 µs | 1,06 | 1,31 | 1,68 |
| DeltaNet in_proj_z | 6144×2560 u8 | 36 | 24,8 µs | 1,06 | 1,40 | 1,76 |
| DeltaNet out_proj | 2560×6144 u8 | 36 | 24,7 µs | 1,07 | 1,48 | 1,81 |
| hc Up | 10240×320 u8 | 96 | 6,6 µs | 1,56 | 2,09 | 2,67 |
| expert gate/up | 640×2560 u4 | 960 | 4,2 µs | 1,56 | 1,83 | 1,96 |
| expert down | 2560×640 u4 | 480 | 3,6 µs | 1,33 | 1,67 | 1,96 |
| partagé gate/up | 640×2560 u8 | 96 | 5,3 µs | 1,19 | 1,22 | 1,28 |

Les ρ des experts portent sur **un même** expert lu pour R lignes. En
vérification, chaque ligne a ses propres experts.

**Chevauchement des experts** : 16,8 experts distincts sur 20 à R = 2, 22,7
sur 30 à R = 3, 27,9 sur 40 à R = 4. Même avec union, le coût des experts reste
×2,8 à R = 4.

Trois formes s'ajoutent à la table (mode strict) :

| matmul | forme | ×/pas | mono | ρ(4) |
|---|---|---|---|---|
| partagé down | 2560×640 u8 | 48 | 3,9 µs | 1,99 |
| hc Down | 320×10240 u8 | 97 | 15,1 µs | 1,08 |
| `index_qk_proj` | 640×2560 u8 | 12 | 5,3 µs | 1,30 |

Les linéaires bf16 sont mesurés en f32, soit deux fois leurs octets :

| matmul | forme | ×/pas | mono | ρ(4) |
|---|---|---|---|---|
| routeur | 512×2560 | 48 | 7,0 µs | 1,81 |
| `in_proj_a/b` | 48×2560 | 72 | 4,5 µs | 1,05 |

### Reste du pas

Profil par brique après 1k, sur 128 pas : 20,63 ms GPU et 1627 dispatches par
pas. Le temps mural profilé est de 23,34 ms, contre 21,49 ms hors profil. Le
« reste » d'une brique est son temps moins celui de ses matmuls à poids, au
temps mono de la table ci-dessus. Les dispatches du reste sont ceux de la
brique, moins ceux de ses matmuls.

| brique | mono | matmuls | dispatches du reste | latence | travail |
|---|---|---|---|---|---|
| hyper-connections | 3,40 | 2,10 | 480 | 0,47 | 0,84 |
| DeltaNet | 5,06 | 3,62 | 72 | 0,07 | 1,37 |
| QSA (dense, 1k) | 2,09 | 1,13 | 99 | 0,10 | 0,86 |
| MoE, routeur compris | 7,20 | 6,79 | 240 | 0,23 | 0,18 |
| `lm_head` (argmax) | 2,75 | — | 3 | — | — |
| PLE et mixer final | 0,12 | — | — | — | — |

Temps en ms par pas.

- **Latence de dispatch.** Un dispatch suivi de sa barrière coûte 0,93 µs.
  Une copie de 10 240 mots coûte 0,97 µs, une de 40 960 mots 1,16 µs : le
  travail élémentaire d'un kernel en lignes croît peu à cette taille.
- **Argmax de tête en lignes**, sur le vrai `lm_head` u6 : 2,93 ms en mono,
  ρ = 1,93, 2,88 et 3,85 pour R = 2, 3 et 4. Les indices sont égaux à ceux des
  argmax mono. Le kernel en lignes existant relit les poids par ligne : il
  n'y a pas de qmv-T u6.

### Modèle à R = 4 (profondeur 3)

Les hypothèses suivent la règle fixée avant la mesure.

**Hypothèse réaliste :**
- matmuls aux ρ(4) stricts mesurés ;
- experts à 2,79 fois leur temps mono, soit des experts distincts lus une fois
  (union, qui n'existe pas encore) ;
- argmax de tête au ρ mesuré (3,85) ;
- reste : latence partagée une fois, travail × R ;
- PLE séquentiel, × R.

**Hypothèse optimiste :** comme la réaliste, à trois différences près :
- travail du reste × 1,2, comme la copie mesurée ;
- argmax de tête par un qmv-T u6 exact, à écrire, au ρ(4) du `q_proj`
  (2,26) ;
- mixer partagé.

| brique | mono | R = 4 réaliste | R = 4 optimiste |
|---|---|---|---|
| hyper-connections | 3,40 | 7,09 | 4,74 |
| DeltaNet | 5,06 | 11,61 | 7,78 |
| QSA | 2,09 | 5,82 | 3,40 |
| MoE | 7,20 | 18,65 | 18,15 |
| `lm_head` | 2,75 | 10,59 | 6,22 |
| PLE et mixer | 0,12 | 0,49 | 0,41 |
| **total** | **20,63** | **54,25 (ρ = 2,63)** | **40,70 (ρ = 1,97)** |

Supposer que le scan DeltaNet ne lit son état qu'une fois pour R lignes
(0,41 ms partagés) ne baisse le ρ réaliste qu'à 2,57.

**Draft sur sous-vocabulaire.** Le `lm_head` coûte ~2,8 ms des 3,60 ms d'un pas
MTP. Avec 32k jetons de vocabulaire de draft, le pas MTP tomberait à ~1,16 ms,
soit `T_mtp/T_ar ≈ 0,054`. L'acceptation est supposée inchangée, ce qui est
favorable à cette option.

**Gain modélisé**, en code à profondeur 3 (n = 3,2) :

| option | réaliste | optimiste |
|---|---|---|
| C actuel (pas mono), pour référence | ×0,66 (mesuré ×0,665) | — |
| D complet | ×0,91 | ×1,13 |
| draft sous-vocabulaire seul | ×0,75 | ×0,75 |
| les deux | ×1,10 | ×1,42 |

En prose (n = 2,55), les deux options combinées donnent ×0,88 en réaliste.

### Décision

Aucune option ne dépasse ×1,2 dans l'hypothèse réaliste : **D s'arrête**.

- Pour atteindre ×1,2 avec les deux options, il faudrait un ρ(4) ≤ 2,39. Cela
  suppose d'écrire l'union MoE en lignes, un argmax de tête qmv-T u6 exact, le
  scan l2/σ en lignes, et la QSA et les hyper-connections en lignes. Même
  ainsi, le modèle s'arrête à ×1,18–1,21 en code et reste sous ×1 en prose.
- Le MoE domine : 512 experts top-10, avec seulement 30 % de chevauchement
  entre jetons consécutifs. Une vérification de R lignes y relit presque
  R fois les experts.
- C reste en place comme base exacte. Le spéculatif reste désactivé par défaut
  sur Flash-Next.

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
SARAGOSSA_FLASH_NEXT_PROFILE=1 SARAGOSSA_RUST_DECODE_PROFILE=1 \
  cargo test --release --features metal,devtools --lib -- --ignored --nocapture \
  $T::profile_resident_decode_after_1k_context
```

Les tests GPU tournent seuls sur la machine ; aucune mesure de débit n'est
prise pendant une autre charge GPU.
