# Bench concurrents — nuit 2 (6 nov. 2026), Qwen3.8-27B, M5 Max 128 Gio, batch 1

Protocole v5, run dir `night1`, même gel des réglages (`matrix_sha256` identique). Lancée à 15 h 32, finie à 19 h 20, soit 3 h 48, contre ≈ 7 h prévues.

Binaires identiques à la nuit 1 (mêmes sha256) :
- **saragossa** : sha `7a12d6d8…`. Le commit de build n'est pas tracé : les `env-*.json` enregistrent le HEAD du dépôt au moment de la capture, pas celui du build. Le binaire date du 4 nov. Sur l'antériorité aux PR #109–#111, voir §3.
- **llama.cpp** : 0.8.0 build 11960 ;
- **llmd** : 20261029.0 ;
- **mlx-lm** : 0.32.2.

Sur secteur. 0 requête bridée détectée par le harnais : contrôle pmset avant et après chaque chargement, sans température du package.

**Ce qu'on compare** : des couples moteur–format, pas des moteurs seuls.
- saragossa et mlx-lm chargent le même checkpoint MLX u4 gs64 (≈ 4,5 bpw).
- llama.cpp charge Q4_K_M (4,92 bpw, **sans imatrix**) ou Q4_0 pur (4,50 bpw).
- llmd charge NVFP4, dont la qualité n'est pas mesurée ici.
- Drafters : DFlash 2 quantifié en 4 bits côté saragossa, BF16 côté llama.cpp.

Divulgation : l'auteur développe saragossa.

Sources : `tables.md`, `analysis.json`, `per_request.jsonl`, `results.jsonl`, `quality/`, `logs/`. Relu par deux relecteurs indépendants, toutes corrections intégrées.

## 1. Qualité des formats (wikitext-2 test)

Même découpage pour tous : 32 fenêtres de 512 tokens, 255 positions notées par fenêtre, soit **8 160 tokens évalués**. Les ids sont identiques entre llama.cpp et le tokenizer HF (297 193 tokens, fnv `0fbade12…`).

| format | bpw | PPL | PPL(Q)/PPL(base) | KL moyen / p99 | top-1 identique |
|---|---|---|---|---|---|
| BF16 (GGUF, llama.cpp) | 16 | 5,587 ± 0,147 | — | — | — |
| Q4_K_M (llama.cpp) | 4,92 | 5,595 ± 0,147 | 1,0017 ± 0,0026 | 0,0199 / 0,194 | 94,0 % |
| Q4_0 pur (llama.cpp) | 4,50 | 5,731 ± 0,151 | 1,0261 ± 0,0036 | 0,0385 / 0,327 | 91,1 % |
| MLX u4 gs64 (forward mlx-lm) | ≈ 4,5 | 5,752 | non comparable directement | non mesuré | non mesuré |

- Les ratios Q4 utilisent la `PPL(base)` du calcul KL, 5,5853. La passe BF16 seule donne 5,5871, soit +0,135 % pour Q4_K_M.
- **Q4_K_M** : aucun écart de PPL n'est mis en évidence (0,009 ± 0,015). Les distributions restent distinguables du BF16 (KL 0,020, top-1 94 %).
- **Q4_0** : la dégradation de PPL est nette (+2,6 % ± 0,4 %).
- **MLX** : 5,752 est la PPL observée la plus haute. Elle vient toutefois d'un autre calcul, un forward mlx-lm, alors que la référence BF16 est calculée par llama.cpp. On ne peut donc pas séparer l'effet du format de celui du moteur. Elle ne mesure pas non plus saragossa directement : saragossa partage le checkpoint mais pas le calcul, et ses textes greedy diffèrent de ceux de mlx-lm.
- **Conclusion prudente** : le checkpoint MLX u4 n'a pas la qualité démontrée de Q4_K_M. Pour le classer contre Q4_0, il faut une PPL BF16 calculée par le même forward MLX (§5).

`llama-bench` (moyenne de 3 essais ; pp4096 et tg400 sont deux tests séparés, tg400 sans prompt) :
- Q4_K_M : pp4096 709,5 tok/s, tg400 27,6 tok/s ;
- Q4_0 : pp4096 662,7 tok/s, tg400 30,7 tok/s.

## 2. Échantillonné exploratoire (sampled-S)

Paramètres : T = 0,7, top-k 20, top-p 1.
- 6 prompts chat et 3 prompts 4k par famille, soit 12, 6 et 6 observations par config ; 2 répétitions avec nonce distinct.
- Un seul chargement par config : le bootstrap ne mesure pas la variabilité entre chargements.
- Graines 11 et 22 pour saragossa et llama.cpp. Elles ne sont pas transmises à mlx-lm ni à llmd : pour ces deux moteurs, ce sont des répétitions, pas des graines.
- Q4_0 n'est pas joué dans cette phase.
- L'équivalence de distribution du spéculatif échantillonné **n'est pas validée**. Ces chiffres sont exploratoires (protocole §12).

Débit de livraison médian (tok/s), puis rapport à llkm-ar avec son IC à 95 % (bootstrap apparié). Le rapport est la moyenne géométrique des rapports par prompt, pas le quotient des médianes. « ≈ » marque les cellules que le harnais signale comme approximatives.

| config | chat | code | prose |
|---|---|---|---|
| llkm-ar | ≈ 28,3 | 27,9 | 27,9 |
| llkm-mtp | ≈ 39,0 · 1,34× [1,29–1,39] | 35,9 · 1,29× [1,28–1,31] | 36,7 · 1,31× [1,28–1,33] |
| llkm-dflash | ≈ 43,0 · 1,50× [1,37–1,62] | 41,0 · 1,48× [1,42–1,53] | 37,9 · 1,41× [1,35–1,53] |
| llmd-ar | ≈ 25,3 · 0,89× [0,89–0,90] | ≈ 24,0 · 0,86× [0,85–0,86] | 24,0 · 0,86× [0,86–0,87] |
| mlx-ar | ≈ 30,0 · 1,05× [1,04–1,07] | 29,4 · 1,05× [1,03–1,06] | 29,1 · 1,05× [1,04–1,06] |
| sar-ar | ≈ 31,3 · 1,11× [1,10–1,12] | 30,5 · 1,10× [1,09–1,10] | 30,6 · 1,10× [1,09–1,10] |
| sar-mtp | ≈ 33,6 · 1,18× [1,14–1,23] | 32,5 · 1,16× [1,14–1,17] | 31,1 · 1,12× [1,11–1,14] |
| sar-dflash | ≈ 50,2 · 1,72× [1,51–1,96] | 47,3 · 1,73× [1,64–1,87] | 39,6 · 1,44× [1,37–1,54] |

TTFT médian (s) :

| config | chat | code 4k | prose 4k |
|---|---|---|---|
| llkm-ar | 0,25 | 5,91 | 5,93 |
| llkm-dflash | 0,27 | 6,56 | 6,09 |
| mlx-ar | 0,24 | **4,53** | **4,49** |
| sar-ar | 0,16 | 4,95 | 5,27 |
| sar-dflash | 0,17 | 5,27 | 5,21 |
| sar-mtp | **0,15** | 5,08 | 5,32 |
| llmd-ar | 0,95 | 15,70 | 15,74 |

Lecture :
- **Débit** : sar-dflash a la plus haute médiane observée dans les trois familles. Les IC le comparent à llkm-ar, pas directement à llkm-dflash. En prose, les deux ne sont pas départagés (39,6 contre 37,9).
- **Gain contre l'AR du même moteur** :
  - sar-dflash ×1,55 / ×1,58 / ×1,31 ;
  - llkm-dflash ×1,50 / ×1,48 / ×1,41 ;
  - llkm-mtp ×1,29 à ×1,34 ;
  - **sar-mtp ×1,07 / ×1,06 / ×1,02 seulement** : objectif de performance à travailler, pas un bug démontré.
- **TTFT** : à 4k, mlx-lm est le plus rapide (4,5 s). En chat court, c'est saragossa.
- **Câblage de sar-dflash** : le drafter DFlash 2 est joint par le chemin spéculatif MTP de saragossa. Le sidecar MTP est chargé et le journal affiche `decode_path=mtp`. Dans le code de `saragossa serve` (`src/serve/state.rs`), le drafter n'est construit que si le MTP est actif.
- **llmd-native** : non supporté. Le drafter natif s'arrête au chargement sur « error: FileNotFound » (protocole, essai du 4 nov.).

## 3. Contexte long : 256k (eval-L, code256k-1)

**Exploratoire, non classé** (protocole §10.5) :
- un prompt de 260 999 tokens d'entrée, 400 tokens générés ;
- une observation par config, chauffe à 32k et non à 256k, aucun IC.

Mémoire :
- « max à vie » = `ri_lifetime_max_phys_footprint`, maximum depuis le lancement (chargement et chauffe compris), en Go décimaux ;
- les fichiers mappés de llama.cpp (ses poids) en sont exclus. La dernière colonne donne leur part résidente lue par vmmap en fin de chargement : un instantané, pas un pic simultané.
- Machine : 128 Gio = 137,4 Go.

| config | identité greedy | TTFT (s) | débit (tok/s) | temps jusqu'à 400 tokens (s) | max à vie (Go) | mappé résident (Go) |
|---|---|---|---|---|---|---|
| llkm-ar | référence | 1090,6 | 12,06 | 1123,7 | 18,6 | 16,3 |
| llq40-ar | référence | 1065,3 | 12,47 | 1097,3 | 18,6 | 14,9 |
| llkm-mtp | **oui** | 1145,1 | 15,65 | 1170,6 | 20,8 | 18,3 |
| llkm-dflash | **non** | 1093,9 | (17,0) | — | 19,9 | 20,4 |
| mlx-ar | référence | 722,0 | 14,25 | 750,0 | 84,6 (pic au decode) | ≈ 0 |
| sar-ar | référence | 732,1 | **14,73** | 759,2 | 43,2 | ≈ 0 |
| sar-mtp | **non** | 767,0 | (21,4) | — | 50,6 | ≈ 0 |
| sar-dflash | **non**, et mesuré sous pression mémoire | 785,9 | (17,2) | — | 130,1 | ≈ 0 |
| llmd-ar | **échec** : réponse vide | — | — | — | — | — |
| llmd-native | non supporté | — | — | — | — | — |

Les débits entre parenthèses ne sont pas admissibles : texte différent de l'AR du même moteur.

- **AR, sur cette seule observation** :
  - saragossa a le plus haut débit de decode : 14,7 tok/s, contre 14,3 pour mlx-lm et 12,1 à 12,5 pour llama.cpp ;
  - son TTFT est proche de celui de mlx-lm (732 contre 722 s, +1,4 %) et 1,46 à 1,49× plus court que celui de llama.cpp (1065 à 1091 s).
- **llkm-mtp** : seule config spéculative testée qui reste identique à son AR à 256k. Son débit de decode gagne ×1,30, mais son TTFT est plus long : le temps jusqu'aux 400 tokens est **+4,2 % plus lent** que llkm-ar.
- **Spéculatif saragossa non admissible à 256k** :
  - sar-mtp et sar-dflash divergent de sar-ar, tous deux au caractère 811. En nuit 1, ils divergeaient tous deux au caractère 779 de prose32k-1.
  - Une cause commune côté vérification est plausible, par exemple l'attention NAX bf16 en vérification au-delà de 16k, documentée dans `docs/perf/greedy-identity.md`. Elle n'est **pas démontrée** par ces mesures.
  - `SARAGOSSA_RUST_VERIFY_STRICT=1` n'était pas dans l'environnement. Ce flag n'existe que depuis la PR #111, fusionnée le 5 nov., et le binaire date du 4.
- **sar-dflash, mémoire** :
  - plateau ≈ 87,1 Go pendant le prefill ;
  - pointe d'environ 14 s au-dessus de 120 Go en fin de prefill, avec un max à vie de 130,05 Go (121,1 Gio), sous les 128 Gio physiques ;
  - le compteur système de swap augmente de 7,7 Go entre les captures qui encadrent ce chargement. L'effet chiffré sur le TTFT et le débit n'est pas isolé, donc ces deux valeurs ne sont pas exploitables ;
  - le maximum échantillonné au decode est de 55,8 Go.
- **llmd-ar** : le serveur évince la séquence (« evicting sequence … out of pages »), avec les fractions mémoire 0,5 × 0,5. Il rend un HTTP 200 vide (`completion_tokens 0`) après 708,8 s de requête. **Bug du harnais et du protocole** : la ligne est classée « admissible » malgré `n_usage = 0`, car le §11.3 n'exige pas de token rendu. Les tables l'excluent (n/d), mais le statut doit être « échec ». Obtenir une génération llmd réussie demanderait une nouvelle mesure.
- **Mémoire AR, lecture indicative** :
  - llkm-ar ≈ 18,6 + 16,3 ≈ 35 Go, en additionnant un maximum historique et un instantané résident ;
  - sar-ar : 43,2 Go, soit ≈ 8 Go de plus ;
  - mlx-ar : 84,6 Go, ≈ 2× sar-ar.

## 4. Ce que saragossa perd ou ne démontre pas, sans filtre

- **Qualité** : le checkpoint MLX u4 n'a pas la qualité démontrée de Q4_K_M. Sa PPL est la plus haute observée, avec la réserve du calcul différent.
- **Mémoire** :
  - au-dessus de llama.cpp dans les mesures de ces deux nuits ;
  - DFlash à 256k monte jusqu'à 130 Go avec du swap.
- **MTP échantillonné** : ×1,02 à ×1,07 sur son AR, contre ×1,3 pour le MTP de llama.cpp.
- **Spéculatif à 256k** : non admissible.
- **TTFT à 4k** : derrière mlx-lm (≈ 5,0–5,3 s contre 4,5 s).
- **256k** : une seule observation, aucun classement possible.

## 5. Suites proposées avant publication

1. **Harnais et protocole** : `n_usage = 0` ou `events_out = 0` doit donner un échec. Ré-analyser ensuite, sans re-mesurer.
2. **Spéculatif à 256k** : relancer eval-L sar-mtp et sar-dflash avec le binaire actuel de main (#109–#115) et `SARAGOSSA_RUST_VERIFY_STRICT=1`, en mesurant à la fois l'identité et le débit. Durée inconnue, ≥ 13 min par cellule d'après la nuit 2.
3. **DFlash** : diagnostiquer puis réduire le plateau et la pointe de prefill à 256k, puis remesurer.
4. **MTP échantillonné saragossa** : diagnostiquer l'acceptation en échantillonné contre greedy.
5. **Qualité MLX** :
   - PPL BF16 par le même forward mlx-lm, avec KL et top-1 contre elle ;
   - idéalement, PPL calculée par saragossa lui-même.
6. **Transparence** : publier le détail par prompt de sampled-S (protocole §10.5), ainsi que les flags gelés et les scripts.
7. **Optionnel** : un format saragossa plus précis (u4 imatrix ou mixte), à mesurer avant toute affirmation.

## 6. Suivi du 7 nov. : correctifs et remesures

1. **Harnais** (branche `feat/serve-bench-equite`, a92c70b) : preuve `output` obligatoire, une réponse vide n'est jamais admissible. Ré-analyse de la nuit 2 sans re-mesure : seule la ligne llmd-ar `code256k-1` change (admissible → `validation_failed`).
2. **Spéculatif à 256k** (nuit 3, `scratchpad/night/compet/night3b`) : binaire main + RoPE précise (#116) + captures DFlash bornées (#117) + commits serve du harnais, `SARAGOSSA_RUST_VERIFY_STRICT=1`. Les trois lignes sont admissibles, preuves comprises :

| config | TTFT (s) | livraison (tok/s) | max à vie (Go) | texte |
|---|---|---|---|---|
| sar-ar | 730,2 | 14,58 | 42,2 | référence |
| sar-mtp | 764,2 | 17,50 | 49,7 | identique à sar-ar |
| sar-dflash | 802,1 | 17,79 | 66,5 | identique à sar-ar |

   - Cause de la divergence de la nuit 2 : les textes de sar-mtp et sar-dflash sont ceux de la nuit 2 ; c'est le texte de sar-ar qui change, au caractère 811. L'ancien AR basculait un quasi-ex-aequo à cause de la dérive fast-math de la RoPE (1,1e-3 à 256k), corrigée par #116. La vérification spéculative n'était pas en cause.
   - Une seule observation par cellule : lecture indicative, pas d'IC.
3. **DFlash mémoire** : 130,1 Go + swap → 66,5 Go (#117 : captures du prefill bornées à la fenêtre de 2048 du drafter). Reste ≈ 10,7 Go de KV du drafter dimensionné sur toute la capacité alors que ses couches sont fenêtrées : chantier suivant.
4. **MTP échantillonné** : diagnostic par le harnais d'acceptance (code et prose 4k, profondeurs 1-3, 3 graines) : autant de tokens acceptés par vérification qu'en greedy, mais ×0,97-1,10 sur l'AR contre ×1,37-1,48 en greedy. Cause : avec top-k 20, la distribution triait les 248k probabilités par ligne sur CPU. Correctif PR #119 (sélection partielle, distribution identique au bit près) : MTP échantillonné **+30,6 %** (code, IC95 [+26,7 ; +34,9]) et **+25,6 %** (prose, [+24,4 ; +26,9]) en ABBA, soit ≈ ×1,35 sur l'AR échantillonné. À remesurer dans le harnais avant publication.
5. **Qualité MLX** : PPL du BF16 (Qwen/Qwen3.8-27B) par le même forward mlx-lm = **5,589**, contre 5,587 pour le BF16 de llama.cpp, mêmes ids de corpus (fnv `0fbade1299557298`). Les deux outils concordent : le MLX u4 (5,752) est à **+2,9 %** de son BF16, entre Q4_0 (+2,6 %) et loin de Q4_K_M (+0,2 %). C'est une perte de qualité réelle du format u4 utilisé par saragossa et mlx-lm.
6. **DFlash mémoire, suite** (nuits 3c et 3d) : #120 (KV du drafter à base fixe) retire 10,7 Go au decode (61,7 → 51,0 Go) ; PR #121 (prefill sans états finaux complets hors bascule MTP) retire le pic de prefill. sar-dflash à 256k : **49,7 Go** à vie (130,1 + swap en nuit 2), TTFT 732,8 s (= sar-ar), 18,15 tok/s, texte identique à sar-ar, toutes preuves admissibles. Comparaison llama.cpp (nuit 2, indicatif) : llkm-dflash ≈ 40 Go (19,9 hors poids + 20,4 mappés).

## 7. Nuit 4 (7 nov.) : saragossa seul, binaire main 8afdafc, réglages par défaut

Binaire main 8afdafc (#116–#122), **sans** `VERIFY_STRICT` (les nuits 3 l'activaient). Phases eval-S, eval-M, sampled-S et eval-L. Source : `scratchpad/night/compet/night4/{analysis.json,tables.md}`.

- **Admissibilité** : 273/273 requêtes admissibles, toutes preuves comprises. 20 sont marquées `≈` (approx). Aucune bridée.
- **Intégrité greedy** : 67/67 paires identiques à sar-ar, pour sar-dflash comme pour sar-mtp, 256k compris.

| phase / tâche | sar-ar (tok/s) | sar-dflash | sar-mtp |
|---|---|---|---|
| eval-S chat | ≈32,0 | ≈57,4 (×1,70 [1,51–1,94]) | ≈45,1 (×1,41 [1,32–1,51]) |
| eval-S code | 30,8 | 54,2 (×1,77 [1,70–1,84]) | 47,0 (×1,54 [1,49–1,60]) |
| eval-S prose | 30,9 | 46,9 (×1,50 [1,36–1,63]) | 45,8 (×1,48 [1,43–1,55]) |
| sampled-S chat | ≈31,6 | ≈50,1 (×1,56 [1,37–1,77]) | ≈43,2 (×1,36 [1,30–1,42]) |
| sampled-S code | ≈30,9 | ≈48,0 (×1,58 [1,53–1,66]) | 42,1 (×1,36 [1,33–1,40]) |
| sampled-S prose | 30,7 | 40,3 (×1,34 [1,25–1,45]) | 41,3 (×1,35 [1,32–1,39]) |
| eval-M code (3) | 27,9 | 51,6 (×1,85) | 39,3 (×1,41) |
| eval-M prose (3) | 27,9 | 45,1 (×1,62) | 40,1 (×1,44) |
| eval-L code 256k (1) | 14,5 | 36,0 (×2,49) | 21,6 (×1,49) |

Pour chaque config, le premier chiffre est le débit de livraison médian. Entre parenthèses, le rapport apparié à sar-ar, avec son IC 95 % bootstrap quand il y a au moins 6 paires.

**256k (une observation par cellule)** :

| config | TTFT (s) | latence 400 tokens (s) | max à vie (Go) |
|---|---|---|---|
| sar-ar | 731,9 | 759,4 | 42,2 |
| sar-dflash | 732,7 | 743,7 | 49,7 |
| sar-mtp | 764,4 | 782,8 | 49,3 |

**Lecture** :
- **MTP échantillonné.** Il passe de ×0,97–1,10 (nuit 2) à ×1,35–1,36 sur l'AR échantillonné : le correctif #119 se confirme dans le harnais.
- **Écart avec les nuits 3.** Sans `VERIFY_STRICT`, le décodage spéculatif à 256k va environ deux fois plus vite : sar-dflash livre 36 tok/s contre 18 en nuit 3d. Les textes restent identiques à l'AR.
- **Latence de bout en bout à 256k.** Le TTFT domine. DFlash fait gagner 2 % et le MTP coûte 3 % : son prefill est plus lent de 32 s à cause de l'historique MTP.
- **Comparaison aux concurrents.** Les lignes llama.cpp, mlx-lm et ZML viennent des nuits 1 et 2. Les comparer à ces chiffres suppose qu'il n'y a pas eu de dérive temporelle, ce qui n'est pas vérifié : rapports entre moteurs indicatifs, pas d'IC.

## 8. Nuit 5 (8 nov., 02 h 38 – 07 h 41) : eval-S et sampled-S, tous les moteurs dans la même nuit

Même matrice gelée que les nuits 1 et 4, réduite aux phases eval-S et sampled-S. Binaires identiques : llama.cpp et llmd ont le même sha que les nuits 1-2, saragossa est le binaire main 8afdafc de la nuit 4. Sources : `night5/{tables.md,analysis.json,pairs_sar.txt}`. `pairs_sar.txt` contient les rapports appariés saragossa contre concurrents, calculés par `paired_bootstrap` du harnais avec la même graine.

- **Admissibilité.** 0 requête bridée. Sonde de cache llmd : 6 sur 6 passées. Les spéculatifs saragossa sont tous admissibles : 84/84 pour sar-mtp et sar-dflash, textes identiques à sar-ar sur 60/60 paires.
- **Divergences greedy, aux mêmes caractères qu'en nuit 1, donc reproductibles :**
  - llkm-mtp : 1/60 ;
  - llq40-mtp et llq40-dflash : 1/60 chacun ;
  - llkm-dflash : 41/60, avec 41 requêtes `validation_failed`.
- **Stabilité entre nuits.** Les concurrents retrouvent leurs médianes de la nuit 1 à ±2 % près. Par exemple :
  - llq40-dflash : 49,35 → 49,33 en chat, 45,21 → 43,99 en prose ;
  - mlx-ar : 30,62 → 30,53 en chat.

**Rapports appariés de débit de livraison (saragossa / concurrent, IC 95 %) :**

| | chat | code 4k | prose 4k |
|---|---|---|---|
| sar-dflash / llq40-dflash | ×1,10 [1,02–1,18] | ×1,06 [0,99–1,11] | ×1,02 [0,94–1,10] |
| sar-mtp / llq40-mtp | ×1,13 [1,09–1,17] | ×1,18 [1,13–1,22] | ×1,18 [1,12–1,22] |
| sar-ar / llq40-ar | ×1,05 [1,05–1,06] | ×1,03 [1,02–1,03] | ×1,03 [1,02–1,04] |
| sar-ar / mlx-ar | ×1,05 [1,05–1,06] | ×1,04 [1,02–1,04] | ×1,03 [1,02–1,04] |
| sar-mtp / llq40-dflash | ×0,92 [0,88–0,96] | ×0,92 [0,89–0,97] | ×1,02 [0,96–1,06] |

Échantillonné (T 0,7, top-k 20, Q4_K_M seulement côté llama.cpp) :
- sar-dflash / llkm-dflash : ×1,15 en chat, ×1,20 en code, ×1,07 en prose ;
- sar-mtp / llkm-mtp : ×1,14 à ×1,17 ;
- sar-ar / mlx-ar : ×1,03 à ×1,04.

Tous ces IC excluent 1.

**TTFT appariés :**
- **Chat.** saragossa est plus rapide : ×0,73 contre llq40-ar et ×0,66 contre mlx-ar, IC serrés.
- **4k greedy.** Aucune différence n'est établie : sar-ar / mlx-ar ×1,03 [0,93–1,13] en code et ×1,02 [0,91–1,15] en prose ; sar-ar / llq40-ar ×0,92 [0,82–1,03] et ×0,94 [0,85–1,04].
- **4k échantillonné, prose.** mlx-lm est devant : sar-ar / mlx-ar ×1,16 [1,09–1,23].

**Lecture :**
- **Contre llama.cpp Q4_0 DFlash**, le meilleur concurrent, à format de qualité comparable : saragossa DFlash est devant en chat (+10 %). En code (+6 %) comme en prose (+2 %), l'écart n'est pas établi.
- **MTP saragossa** : +13 à +18 % sur le MTP de llama.cpp. Il reste derrière le DFlash de llama.cpp en chat et en code (−8 %).
- **AR** : +3 à +5 % sur llama.cpp Q4_0 et sur mlx-lm.
- **Réserve « nuits mélangées » levée** pour eval-S et sampled-S. Elle reste pour 256k et 32k, qui ont tourné en nuits 1-2 pour les concurrents et en nuit 4 pour saragossa.
