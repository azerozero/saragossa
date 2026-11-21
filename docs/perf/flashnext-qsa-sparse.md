# Qwen3.8-Flash-Next : QSA creuse au-delà de 2051 tokens (étape 5)

Suite du decode résident de l'étape 4 ([flashnext-resident.md](flashnext-resident.md)).
Au-delà de `budget + compress − 1` tokens visibles (2051), l'indexeur de
chaque couche QSA écarte des blocs. La session accepte désormais jusqu'à
262 144 tokens. Le KV reste en f32 jusqu'à 256k ; le KV bf16 viendra plus
tard, en option mesurée.

## Chemin résident

Par couche QSA et par token, dans le même command buffer que le reste du
pas :

1. `index_qk_proj` (u8) donne 4 requêtes de 128 et une clé brute. La clé
   brute est rangée à la ligne `position mod 4` du bloc en cours.
2. Quand le token complète un bloc, `flash_next_index_pool_f32` calcule :
   - la moyenne des 4 clés brutes ;
   - `k_layernorm` (gains `1 + w`) ;
   - la RoPE à la position du premier token du bloc.

   Le résultat va dans le cache des clés poolées `[blocs, 128]`.
3. En régime creux (plus de 512 blocs complets) :
   - `q_layernorm` puis RoPE des requêtes, à la position du token ;
   - scores `s_b = Σ_h ReLU(q_h·k̄_b)/√128 + 0` sur les blocs complets
     (`+ 0` normalise les zéros signés) ;
   - top-512 exact (`flash_next_index_topk_f32`).
4. **Gather.** Les lignes K/V des blocs retenus, puis la queue
   `[4·complets, V)`, sont copiées dans un KV compact f32 d'au plus 2051
   lignes. Le kernel d'attention existant s'exécute sur ce KV.
   - Ce choix réutilise un kernel déjà validé, au lieu d'une attention
     indexée nouvelle.
   - Coût : au plus 2051 × 512 × 2 floats copiés par couche et par token.

### Top-512 exact

Le radix select travaille en 4 passes de 8 bits, depuis l'octet de poids fort
des scores f32 (positifs ou nuls).

- À égalité au seuil, le plus petit indice de bloc l'emporte.
- La compaction suit l'ordre des indices.
- Contrat : mêmes scores, même sélection que `select_for_query` du CPU,
  égalités massives comprises (scores tous nuls, 511/512/513 blocs, 65 536
  blocs).

### RoPE précise

Les q/k de l'attention QSA et les requêtes de l'indexeur passent par
`flash_next_rms_norm_rope_f32`.

- L'angle vaut `position·inv_freq` en f32, avec sin et cos précis, comme
  `rope_in_place` du CPU et transformers.
- Écart au CPU : porte à 1e-5, mesuré ≤ 1,2e-7 à toutes les positions
  jusqu'à 262 143, en d128 comme en d256.

Le kernel générique de l'arène (`rms_norm_rope_heads_decode_f32`) calcule
`position / pow(θ, e)` en fast-math. Son erreur croît avec la position :

| position | écart relatif (d128) |
|---|---|
| 2051 | 9e-6 |
| 32 767 | 1,4e-4 |
| 131 071 | 5,7e-4 |
| 262 143 | 1,1e-3 |

Ce kernel sert aussi aux chemins Qwen résidents. Cette branche n'y touche
pas ; un chantier séparé mesurera leur effet au-delà de 32k.
`flash_next_rope_decode_matches_cpu_at_long_positions` garde l'ancien kernel
comme témoin.

## Session

- **Validation à l'ouverture.** Elle refuse :
  - une tête d'attention ou d'indexeur au-delà de 256 dimensions ou non
    multiple de 32 ;
  - une RoPE plus large que la tête ;
  - un budget non multiple de la compression.
- **Mémoire.** `session_state_bytes` compte, par couche QSA, les clés poolées
  et brutes, plus le KV compact du gather, le scratch de l'indexeur et les
  scores.
- **Points de reprise.** `checkpoint` et `restore` couvrent :
  - les états DeltaNet ;
  - les longueurs KV ;
  - les clés brutes du bloc en cours ;
  - l'anneau et le contexte PLE.

  Chaque point de reprise porte un identifiant et une époque. La restauration
  n'est admise que sur la même branche, ou depuis la base de la branche
  courante. La session est invalidée pendant la restauration.
- **Rejeu sans tête** (`replay`). Il remplit le contexte sans lm_head.
- **Pas de rollback multi-tokens.** Il relève de l'étape MTP/DFlash.

## Oracles

### Briques (tests ordinaires)

| brique | oracle | écart |
|---|---|---|
| top-512 radix, cas d'égalités massives | `select_for_query` | identique |
| couche QSA creuse, budget 64, 200 tokens (133 pas creux) | `qsa` CPU | ≤ 1e-5 ; un pas à sélection justifiée par quasi-égalité saute la comparaison de sortie (aucun sur ce test) |
| RoPE à 5, 2051, 32k, 131k, 262k, d128 et d256 | `rms_norm_1p` + `rope_in_place` | porte 1e-5, mesuré ≤ 1,2e-7 ; le kernel de l'arène, témoin, dépasse 1e-4 |

### Règle des quasi-égalités QSA

Un échange de blocs entre deux sélections est justifié si chaque bloc échangé
a un score à τ du seuil du top-512, relativement à `max(|seuil|, 1)`. La même
unité sert à la marge `(s₅₁₂ − s₅₁₃) / max(|s₅₁₂|, 1)` et à l'écart des
scores.

### Torch ré-ancré (CPU, tests ignorés)

`oracle_dump.py --long` exécute transformers 5.20.2 en f32 sur CPU, sur les
4096 premiers tokens de `docs/perf/mtp-pairwise.md` (sha256 `cbbafb0f…`). Il
exporte :

- les routages et logits du routeur de chaque couche ;
- les masques et scores de l'indexeur des 12 couches QSA ;
- leurs résiduels d'entrée complets (`qsa_input`) et leurs sorties aux
  positions figées (`qsa_output`) ;
- les logits à ces positions (une sur 16, plus les 64 dernières).

Le golden fait 3,05 Go ; sa génération prend 21 min et culmine à 64,5 Go.

`n2_long_qsa_layers_anchored_on_transformers` repart, pour chaque couche QSA,
de l'entrée torch. Il n'y a donc aucune dérive accumulée. Experts torch
imposés aux quasi-égalités torch, sélection torch imposée pour la sortie.

| porte | tolérance | résultat (12 couches) |
|---|---|---|
| routage propre | égal, ou experts échangés à 1e-5 du seuil torch (règle de l'étape 1) | 1 échange justifié |
| scores de l'indexeur | 1e-4 relatif à `1 + \|torch\|` (tolérance de l'étape 1, maintenue par l'arbitrage) | vert |
| sélection propre | égale, ou justifiée à 1e-5 | 4 ex æquo justifiés |
| sortie de couche | 1e-4 | 4e-7 à 3,2e-6 |

Côté torch, les seuils restent ceux de l'étape 1. Les seuils GPU plus fins
(scores à 1e-5, échanges à 2e-5) valent pour l'oracle GPU ré-ancré, où les
deux côtés partagent les mêmes entrées.

`n2_long_cpu_chain_matches_transformers_on_4k_tokens` reste en contrôle de
dérive : la chaîne Rust des 48 couches n'est jamais ré-ancrée. La sélection
torch y est imposée pour suivre la même trajectoire (ablation de l'étape 1).

- Portes : KL ≤ 1e-3 et top-1 identique. Mesuré : KL moyen 2,8e-10, max
  3,4e-8, aucun top-1 différent sur 316 positions.
- Les scores QSA y dérivent jusqu'à 2,9e-3 (couche 47), et sont relevés sans
  porte. Exemple : à la couche 27, requête 2873, 8,87899 en Rust contre
  8,87795 en torch.

### N2 résident (GPU contre CPU, 4096 tokens)

1. **Passe native, contrôlée jusqu'à la première bascule.** Elle est verte
   jusqu'au token 313, encore en régime dense, où le routage de la couche 2
   bascule sur une marge de 3,4e-8 (paire au seuil).
2. **Oracle ré-ancré, porte stricte.** Chaque couche QSA CPU repart, sur toute
   la séquence, des entrées GPU de la passe native. La sélection et le
   routage GPU lui sont imposés, et ses choix naturels doivent les justifier.

   | porte | seuil | mesuré (12 couches) |
   |---|---|---|
   | scores de l'indexeur | 1e-5 | 3,0e-6 à 8,1e-6 |
   | échanges de sélection | justifiés à 2e-5 | 6, tous justifiés |
   | échanges de routage | paire au seuil, marge ≤ 1e-5 | 1, justifié |
   | sortie de couche | 1e-4 | 3,0e-6 à 8,6e-6 |

   - Les scores GPU et CPU, calculés aux mêmes entrées, ne diffèrent que par
     l'ordre d'accumulation f32 (qmv u8 sur 2560, puis 4 têtes × d128) :
     8,1e-6 au plus.
   - Une erreur ε par score peut échanger des blocs à 2ε du seuil, d'où la
     justification à 2e-5.
   - **Rapport bruit/seuil de la sortie.** À entrées identiques, la couche 0
     s'écarte déjà de 1,1e-5 : un seuil de 1e-5 serait sous le bruit des
     kernels. Les couches QSA ré-ancrées mesurent au plus 8,6e-6, soit au
     moins 11 fois sous le seuil de 1e-4, lui-même dix fois sous la porte de
     bout en bout (1e-3).
   - **Puissance.** Une mauvaise sélection imposée au GPU en passe native doit
     faire rougir la seule porte de sélection. Autotest du seuil de sortie :
     la sortie capturée dans le rapport, mise à l'échelle de 1 + 2e-4, doit
     faire rougir la seule porte de sortie.
3. **Bout en bout, contrôle de dérive.** Les experts du CPU sont imposés aux
   quasi-égalités ; la sélection QSA reste celle du GPU.
   - Portes :
     - régime dense vert, comme à l'étape 4 ;
     - aucun argmax différent au seuil de routage 1e-6, sur toute la séquence.
   - Au-delà de 2051 tokens, KL, argmax, part imposée et dérive des scores
     sont publiés sans porte de sélection.
   - Pourquoi : la dérive des scores atteint 1e-3 sur les tokens 2051 à 3072,
     comme entre le CPU et torch. Un seuil QSA calibré à 4 fois cette dérive
     (4,3e-3) dépasserait le plafond de 1e-4, et imposerait 91 % des points
     creux.

### Résultats du N2 résident (4096 tokens, 196 608 points token × couche)

Le test est vert en 2438 s, avec un pic mémoire de 93,5 Go.

- **Passe native.** Première bascule au token 313, couche 2, sur une marge de
  3,4e-8, avec un échange limité à la paire au seuil. Avant cette bascule,
  l'écart reste ≤ 6,8e-5 et le KL de 7,6e-9.
- **Oracle ré-ancré** : vert sur les 12 couches (tableau ci-dessus).
- **Mutations de l'oracle ré-ancré**, toutes deux rouges comme attendu :
  - sélection mutée (couche 3, dernier token) : 1 échange non justifié, et
    les autres portes restent vertes ;
  - autotest de sortie : écart 2,0e-4, au-dessus de 1e-4, sélections
    intactes.
- **Bout en bout**, sélection QSA propre du GPU :

  | routage imposé ≤ | régime dense | argmax ≠ (4096) | KL max | écart max | routage imposé | sélections QSA ≠ | dérive des scores |
  |---|---|---|---|---|---|---|---|
  | 1e-7 | rouge | 1 | 4,0e-3 | 1,6e-1 | 0,03 % | 1040 | 8,8e-2 |
  | 1e-6 | **vert** | **0** | 1,3e-3 | 9,7e-2 | 0,29 % | 178 | 6,0e-2 |
  | 1e-5 | vert | 0 | 4,0e-6 | 4,4e-3 | 3,0 % | 75 | 3,7e-3 |

  - Seuil retenu pour le régime dense : 1e-6, comme à l'étape 4.
  - Les sélections QSA différentes tombent toutes après 2051, avec des marges
    CPU de 6e-7 à 8e-4. Elles sont publiées, sans porte.
- **Mutations de routage** (tokens 38 et 316, en régime dense) : rouges,
  comme attendu.

## Mémoire jusqu'à 256k

`resident_session_allocates_max_capacity` ouvre une session de 262 144
tokens. Elle alloue 13,54 Go hors poids (dont 1,4 Mo au premier pas), égal au
budget calculé. Pic du processus : 89,1 Go, sous le working set de 115 Go.

| poste | à 256k |
|---|---|
| KV f32, 12 couches | 12,9 Go |
| clés poolées et brutes | 0,4 Go |
| KV compact, scores, sélection | < 0,1 Go |
| états DeltaNet et instantanés | 0,24 Go |

Un remplissage complet à 256k reste hors de portée sans prefill résident :
256k × 20 ms ≈ 85 min de rejeu.

## Débit à 32k

Porte : au moins 40 tok/s en AR après un contexte de 32 768 tokens, mesurés
sur la plus lente de 3 répétitions (256 pas gloutons, débit = 256 / durée
totale, `bench_resident_decode_after_32k_context`, M5 Max). Le test échoue
sous ce seuil.

- **Corpus figé.** wikitext-2-raw-v1, `wiki.test.raw` :
  - zip sha256 `ef7edb56…5a11`, texte `173c87a5…dd08` ;
  - tokenizer `0997f410…` ;
  - ids `1add1dbe…` (33 024 tokens).

  Il est passé par `SARAGOSSA_FLASH_NEXT_CORPUS`, avec `ids.u32` et
  `provenance.json`.
- **Rejeu sans lm_head.** Le contexte est rejoué une seule fois : 658 s, soit
  20 ms par token. Un point de reprise est pris ensuite, et chaque répétition
  repart de ce point.
- **Sondes, tous les 4096 tokens, sur la dernière couche QSA** :
  - requêtes recalculées sur CPU (norme et RoPE) : écart max 1,5e-7 ;
  - scores à 1e-5 ;
  - sélection identique à `select_for_query` ;
  - clé poolée du dernier bloc à 1e-5.

| répétition | débit | médiane par pas | p10 / p90 | max |
|---|---|---|---|---|
| 0 | 43,4 tok/s | 22,94 ms | 22,71 / 23,21 ms | 27,7 ms |
| 1 | 43,5 tok/s | 22,78 ms | 22,56 / 22,99 ms | 30,4 ms |
| 2 | 43,5 tok/s | 22,77 ms | 22,56 / 22,99 ms | 30,3 ms |

**Porte verte : 43,4 tok/s** pour la plus lente. Le pas médian passe de
21,4 ms à 1k (étape 4) à 22,8 ms à 32k. Le régime creux des 12 couches coûte
donc environ 1,4 ms par token : scores sur 8192 blocs, top-512 et gather.
Pic mémoire du bench : 78,1 Go.
