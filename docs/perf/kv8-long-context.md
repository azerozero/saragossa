# KV 8 bits par défaut au-delà d'un seuil de contexte ?

Question : le KV résident full-attn doit-il passer en q8 (`SARAGOSSA_RUST_KV_QUANT=8`)
au-delà d'un seuil de contexte, pour diviser par deux la mémoire là où le KV
domine ? Aujourd'hui le défaut est bf16 (PR #55 : q8 est opt-in, decode +2 à +3 %
à 115k/256k, −1 % à 32k, texte changé à 32k, pic RSS 256k non réduit).

Ce paragraphe et le critère ci-dessous sont écrits et commités AVANT toute
mesure. Les résultats viennent après, sous « Résultats ».

## Critère pré-enregistré

Modèle : Qwen3.8-27B 4 bits, M5 Max. Référence : le même binaire en KV bf16
(arène bf16 forcée, `SARAGOSSA_RUST_KV_QUANT=bf16`). Candidat de décision : q8 ;
k8v4 est mesuré et rapporté, il ne peut pas être le défaut.

Le défaut q8 au-delà d'un seuil N est adopté si TOUTES les conditions tiennent à
tous les contextes mesurés ≥ N :

| # | Condition | Seuil |
|---|---|---|
| Q1 | KL moyenne(bf16 ‖ q8), 2 prompts, 64 pas forcés | ≤ 1e-3 |
| Q2 | KL p99 sur les pas (agrégés sur les 2 prompts) | ≤ 5e-3 |
| Q3 | Accord top-1 avec le bf16 | ≥ 99 % |
| M1 | Taille de l'arène KV full-attn (analytique, relevée au chargement) | ≤ 0,60 × bf16 |
| M2 | Pic `peak memory footprint` du processus à 128k | au moins −5 % contre bf16 |
| V1 | Decode tok/s médian ABBA, AR et MTP | ≥ 0,98 × bf16 |
| V2 | TTFT médian ABBA | ≤ 1,02 × bf16 |
| I1 | md5 de la sortie MTP = md5 AR, KV q8 des deux côtés, 2 prompts | identiques |

Choix de N : 32768 si tout passe à 32k (Q, V, I) et à 64k/128k ; sinon 65536
si Q1-Q3 passent à 64k, V1/V2/I1 à 128k et M1/M2 ; sinon aucun défaut ne change
et seul ce rapport est publié. La KL n'est pas mesurable à 128k : l'outil
`--kl-record` plafonne le prompt à 65 536 tokens. 128k n'est donc couvert que
par la vitesse, la mémoire et l'identité ; la KL à 64k est prise comme borne.

### Justification des seuils

- Q1 = 1e-3 : porte KL des optimisations de kernels dans le dépôt
  (`flashnext-resident.md`, `flashnext-cpu-ref.md`, `flashnext-qsa-sparse.md`).
  La référence « bf16 contre f32 » (`prefill-bf16.md`) est à ≤ 3e-5 : q8 est une
  perte volontaire plus grosse, on tolère donc 30 fois plus, pas davantage. En
  comparaison, la KL des poids 4 bits contre bf16 est de 0,03 à 0,13 : 1e-3
  reste très en dessous du bruit de quantification des poids.
- Q2 = 5e-3 : 5 × Q1. La queue p99 est bornée (le gate Gemma 4 est exprimé en
  multiple du bruit bf16 sur KL moyenne ET p99, `gemma4-prefill.md`).
- Q3 = 99 % : `prefill-bf16.md` mesure 99,8 % (511/512) pour un changement de
  dtype du KV. 99 % laisse 1 pas divergent sur 100, sans plus.
- M1 = 0,60 : q8 stocke 1 octet par valeur plus échelle et biais par groupe de
  64 : le ratio attendu est ≈ 0,53 ; 0,60 absorbe les couches repliées en bf16.
- M2 = −5 % : sous ce gain, la complexité d'un défaut dépendant de la longueur
  ne se justifie pas ; le pic dépend aussi du prefill, pas seulement du KV.
- V1 = V2 : 0,98 tolère le bruit ABBA de la machine (2 %), sans accepter une
  régression réelle.
- I1 : règle du dépôt, la sortie spéculative greedy est identique au bit à l'AR.

### Protocole

- Prompts : concaténation déterministe des sources du dépôt (code `src/**/*.rs`
  et prose `docs/**/*.md`, ordre trié, commit de base de la branche) coupée au
  nombre de tokens visé, avec l'instruction de revue de code de
  `bench/competitors/prompts.py`. Le corpus Rust stdlib des campagnes n'est pas
  installé sur cette machine ; il est de même nature (code Rust, prose).
- KL : `saragossa run --kl-record` (bf16, référence) puis `--kl-compare` avec
  q8 et k8v4, 64 pas, 32k et 64k, prompt code et prompt prose.
- Vitesse : ABBA (bf16, q8, q8, bf16), 3 paires, `run --metrics`, 128 tokens,
  caches de préfixe coupés, médianes. Chaque passe n'est gardée que si
  `os.getloadavg()[0] < 2` et le Mac sur secteur ; sinon elle est rejouée.
- Mémoire : `/usr/bin/time -l` (`peak memory footprint`) sur un decode de 16
  tokens à 64k, 128k, 256k, pour bf16, q8, k8v4.
- Aucune tolérance n'est élargie, aucun golden n'est régénéré.

## Résultats

Machine : M5 Max, Qwen3.8-27B 4 bits, prompts code/prose du dépôt. Données brutes :
71 lignes de `results.jsonl` de la campagne (~/scratchpad/kv8). Seules les passes
valides sont enregistrées ; les passes jetées par la porte `loadavg < 2` sont
rejouées et n'apparaissent pas.

### Qualité (KL contre bf16, 64 pas forcés)

| Contexte | Prompt | KL moy. q8 | KL p99 q8 | Top-1 q8 | KL moy. k8v4 |
|---|---|---|---|---|---|
| 32k | code | 1,4e-5 | 1,2e-4 | 100 % | 1,3e-3 |
| 32k | prose | 1,2e-5 | 1,6e-4 | 100 % | 2,0e-3 |
| 64k | code | 1,4e-5 | 1,9e-4 | 98,4 % | 1,2e-3 |
| 64k | prose | 1,9e-5 | 2,9e-4 | 98,4 % | 1,0e-3 |

Q1 et Q2 passent largement. Q3 échoue à 64k (63/64 = 98,4 % < 99 %, les deux
prompts) ; il passe à 32k.

### Vitesse (ABBA, médiane des rapports q8/bf16 par bloc ; decode tok/s)

| Contexte | Mode | bf16 | q8 | Rapport (blocs) | TTFT q8/bf16 |
|---|---|---|---|---|---|
| 32k | AR | 28,39 | 28,64 | 1,011 (1,008-1,012, 3 blocs) | 1,046 |
| 32k | MTP | 43,28 | 39,99 | 0,923 (0,908-0,931, 3 blocs) | 1,012 |
| 128k | AR | 20,48 | 21,12 | 1,029 (1,028-1,031, 2 blocs) | 1,010 |
| 128k | MTP | 32,27 | 23,51 | 0,728 (0,728-0,729, 2 blocs) | 1,007 |

Pas d'IC formel : 2 à 3 blocs, on donne l'étendue min-max. Aucun bloc ne
chevauche le seuil 0,98 en MTP.

### Mémoire (pic `peak memory footprint`, decode de 16 tokens, Gio)

| Contexte | bf16 | q8 | q8/bf16 | k8v4 |
|---|---|---|---|---|
| 64k | 20,9 | 26,8 | 1,28 | 26,4 |
| 128k | 27,0 | 37,6 | 1,39 | 38,0 |
| 256k | 32,7 | 58,7 | 1,80 | 56,5 |

M2 échoue : le pic q8 est plus haut que bf16 (+39 % à 128k), pas plus bas. M1
(taille analytique de l'arène) n'a pas été relevé par la campagne : non évalué.

### Identité (md5 de la sortie, KV q8, 200 tokens)

MTP = AR sur code-32k, prose-32k et code-128k : I1 passe.

### Application du critère

| Condition | 32k | 64k | 128k |
|---|---|---|---|
| Q1, Q2 | passe | passe | non mesurable |
| Q3 | passe | échoue (98,4 %) | non mesurable |
| M1 | non relevé | non relevé | non relevé |
| M2 | n/a | échoue | échoue (+39 %) |
| V1 AR | passe | n/a | passe |
| V1 MTP | échoue (0,923) | n/a | échoue (0,728) |
| V2 | échoue en AR (1,046) | n/a | passe (1,010) |
| I1 | passe | n/a | passe |

Ni N = 32768 (V1 MTP, V2, Q3 à 64k) ni N = 65536 (Q3 à 64k, V1 MTP à 128k, M2)
ne sont atteints.

## Verdict

Aucun défaut ne change : q8 reste opt-in (`SARAGOSSA_RUST_KV_QUANT=8`), rapport
seul. Les causes sont mesurées : le decode MTP perd 8 % à 32k et 27 % à 128k, et
le pic mémoire monte au lieu de baisser (le prefill garde un KV bf16 en plus de
l'arène q8).

## Limites

- 128k : 2 blocs ABBA au lieu de 3 (la porte de charge a jeté des passes) ; KL
  non mesurée au-delà de 64k (plafond de `--kl-record`).
- Q3 repose sur 64 pas : un seul pas divergent donne 98,4 %.
- M1 non relevé ; le critère n'est pas rejoué avec ce seul point.
- Une seule machine, un seul modèle ; corpus du dépôt à la place du corpus Rust stdlib.
- Le pic mémoire inclut le prefill ; la réduction du KV seul n'est pas isolée.
