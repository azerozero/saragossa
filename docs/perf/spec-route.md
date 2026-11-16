# Routage spéculatif par classe de prompt

Actif par défaut dès qu'un drafter DFlash est chargé
(`SARAGOSSA_RUST_DFLASH_DRAFT_DIR`), coupé par `SARAGOSSA_RUST_SPEC_ROUTE=0`.
Sur `serve`, `run` et la CLI, le greedy non contraint ne passe par la tête MTP
que si la requête est nettement de la prose (`spec_route::prefers_dflash`) ;
tout le reste, et tout l'échantillonné, garde DFlash.

Règle prudente, retenue après relecture : en cas de doute, DFlash. MTP exige
à la fois

- aucun outil déclaré dans la requête ;
- aucune clôture ```` ``` ```` dans toute la conversation, messages assistant
  compris ;
- un dernier message utilisateur non vide, en écriture latine, sans syntaxe de
  code (accolades, `==`, `=>`, `def `, `fn `, `ident(`, indentation,
  `SELECT … FROM`…), sans nom de langage ou de format (ASCII, bornes non
  alphanumériques, donc aussi au milieu d'un texte CJK) et sans radical de mot
  de code (`cod`, `fonction`, `program`, `compil`…), lu en entier. La négation
  proche (« sans code ») et « uniquement de la prose » n'annulent que les
  radicaux.

Le prix de cette prudence est payé en prose : un « vue » ou un « go » suffit à
garder DFlash, et toute prose non latine reste en DFlash.

Alternance dans une conversation : quand MTP (ou l'AR) prolonge une entrée du
cache de préfixe qui porte des captures DFlash, il prolonge aussi ces captures,
et le prefill DFlash sème l'historique MTP quand le routage est actif. Un tour
MTP reprend donc l'entrée du tour DFlash, et le tour DFlash suivant reprend
l'entrée MTP, sans prefill complet (oracle
`spec_route_alternation_reuses_prefix_real_model`, sous et au-delà de 16k).

Classifieur sur embeddings (opt-in, `SARAGOSSA_RUST_SPEC_ROUTE_EMBED_DIR=<snapshot
local de multilingual-e5-small>`, jamais téléchargé) : il remplace la règle
« écriture latine » une fois les indices de code écartés, pour envoyer aussi la
prose non latine vers MTP. 160 exemples de référence (10 par classe et par
langue : fr, en, es, de, zh, ja, ko, ru, `src/spec_route/refs.json`), moyenne
des 5 meilleurs cosinus code moins prose, MTP seulement sous −0,01. Jeu tenu à
part de 64 consignes (`tools/fixtures/spec_route_embed_test.json`) : 61 justes,
3 proses gardées en DFlash, aucun code envoyé vers MTP. Latence médiane
35 ms, p95 44 ms sur CPU (charge 1 min 4,7) : loin de la cible de 2 ms, d'où
l'opt-in. Test : `embed_router_accuracy_and_latency` (ignoré sans poids).

## Pourquoi DFlash perd en prose

Les données de la campagne du 15 novembre (`mtp-dflash-fusion-results.json`
sur `perf/mtp-dflash-fusion`) suffisent. Un tour DFlash, soit l'arbre B = 8 et
la vérification du tronc en lignes, coûte 87 à 126 ms quelle que soit
l'acceptance. Le point mort contre MTP vaut donc ce coût divisé par le temps
par token de MTP :

| Cas | Tokens par tour DFlash | ms par tour | ms par token MTP | Point mort | Gagnant |
|---|---|---|---|---|---|
| C1 | 7,76 | 116 | 18,2 | 6,4 | DFlash |
| C2 | 5,12 | 96 | 22,1 | 4,3 | DFlash |
| C3 | 7,11 | 106 | 18,5 | 5,7 | DFlash |
| C32 | 7,76 | 126 | 22,1 | 5,7 | DFlash |
| P1 | 3,05 | 93 | 23,9 | 3,9 | MTP |
| P2 | 3,05 | 87 | 23,5 | 3,7 | MTP |
| P3 | 3,94 | 95 | 23,2 | 4,1 | MTP |
| P32 | 2,72 | 91 | 28,7 | 3,2 | MTP |

Les tours sont estimés par `proposed − accepted`, puisque le chemin parcouru
compte un échec par tour. Le calcul se rejoue avec
`python3 -I -B tools/spec_route_diag.py mtp-dflash-fusion-results.json`.
Le taux `accepted / proposed` sépare mal les deux
classes : 61-73 % en prose contre 79-86 % en code. Ce qui les sépare, c'est la
longueur du préfixe accepté. Le drafter, une diffusion par blocs, prolonge bien
du code mais rarement plus de 2 ou 3 tokens d'une phrase libre. La prose passe
alors sous le point mort d'environ 4 tokens par tour, et MTP, à 23-29 ms par
token, l'emporte.

## Signaux examinés

- **Acceptance des N premiers tours DFlash.** Ce signal a déjà été mesuré le
  19 octobre et le 20 octobre en version intégrée, sans re-prefill. Il se
  déclenche au premier tour éligible sur la prose (tokens 21-35) et jamais sur
  le code (10 runs sur 10). Il sépare bien, mais il paie toujours le départ
  DFlash lent, et il se trompe sur les consignes « code + explication » (T0,
  T2), dont l'introduction en prose fait basculer vers MTP alors que DFlash
  gagne sur le corps de la réponse (+22 % sur T0 à 500 tokens). La bascule
  intégrée n'a rejoint MTP sur aucune prose (verdict A). Ce signal reste donc
  écarté et nous ne l'avons pas refait.
- **Acceptance MTP du début.** 61-67 % en prose contre 76-88 % en code : les
  classes se recouvrent presque avec C2 (76 %). En plus, passer de MTP à DFlash
  demanderait les captures des couches cibles sur tout le texte généré. Signal
  écarté.
- **Classe de la consigne (retenu).** Elle est connue avant le prefill et ne
  coûte rien. Seul le dernier paragraphe compte, limité à ses 400 derniers
  caractères : un dossier technique à résumer (T7) contient du code sans en
  demander. Le test est une liste fixe de mots de code (langages, `code`,
  `fonction`, `implement*`, `program*`, `script`, une clôture ```` ``` ````),
  annulée par une négation dans les deux mots qui précèdent (« sans code »)
  ou par « uniquement de la prose ». Sur le corpus, la séparation est de
  11 sur 11 en classe de sortie : C1-C3, C32, T1, T6 et T0 sont classés code,
  P1-P3, P32 et T7 sont classés prose (test `spec_route::tests::corpus_classes`).
  Les cas mixtes partent tous en DFlash : M1 (= T0, code + explication),
  M2 (explication avec un court exemple Python) et M3 (code fourni, explication
  demandée). Ils sont mesurés ci-dessous.

## Mesure ABBA du 16 novembre 2026

Le protocole reprend celui du 15 novembre : 27B u4ls, sidecar MTP, drafter
DFlash 2 en u4, 256 tokens greedy sans EOS, préfixe froid, MTP sans lookup, une
chauffe par mode exclue, puis l'ordre AR, MTP, DFlash, routage suivi de l'ordre
inverse. Il y a six passages par cas courts et quatre par cas à 32k.
Tout passe par les API publiques de production : `generate_greedy_mtp_routed`
avec `allow_dflash` à faux (MTP), à vrai (DFlash) ou fixé par le classifieur
(routage). Le Mac était libre (nuit). La charge 1 min est relevée avant chaque
série et chaque passage, et les résultats la conservent.

Débit de decode en tok/s, médianes. « Routage / meilleur » est le gain apparié
du routage sur le meilleur des deux modes, médiane et IC 95 % bootstrap.

| Cas | Classe | AR | MTP | DFlash | Routage | Choix | Routage / meilleur |
|---|---|---|---|---|---|---|---|
| C1 | code | 30,9 | 54,7 | 68,1 | 68,0 | DFlash | −0,98 % [−2,12 ; 0,65] |
| C2 | code | 31,4 | 47,3 | 56,7 | 56,6 | DFlash | +0,14 % [−0,21 ; 0,47] |
| C3 | code | 31,4 | 54,6 | 69,1 | 69,4 | DFlash | +0,35 % [0,16 ; 0,42] |
| P1 | prose | 31,2 | 41,6 | 32,2 | 41,5 | MTP | −0,32 % [−0,70 ; 0,54] |
| P2 | prose | 31,4 | 41,9 | 35,5 | 41,7 | MTP | −0,32 % [−0,77 ; −0,13] |
| P3 | prose | 31,4 | 43,2 | 41,0 | 43,1 | MTP | −0,42 % [−0,90 ; −0,12] |
| M1 | mixte | 29,9 | 47,3 | 65,1 | 66,8 | DFlash | +1,60 % [0,44 ; 3,14] |
| M2 | mixte | 30,5 | 44,2 | 45,4 | 46,4 | DFlash | +1,75 % [0,27 ; 2,35] |
| M3 | mixte | 30,7 | 43,7 | 45,1 | 45,6 | DFlash | +0,96 % [0,38 ; 3,27] |
| C32 | code | 27,0 | 45,5 | 62,2 | 62,4 | DFlash | −0,35 % [−1,19 ; 0,41] |
| P32 | prose | 26,9 | 35,0 | 30,1 | 35,2 | MTP | +0,81 % [−1,74 ; 1,58] |

Charge pendant les séries courtes : charge 1 min au plus de 2,1 (code) et de
2,3 (prose), sans compilateur. La série mixte a croisé un pic à 20,7, avec au
plus un compilateur relevé. Les ratios restent appariés. Une première série à
32k a croisé jusqu'à 15 compilateurs (C32 routage −12,9 %, IC [−29,8 ; 0,5]).
Elle est écartée et rejouée avec `--calm-wait 90`, qui attend avant chaque
passage une charge sous 2 sans compilateur : charge 1 min de 1,1 à 2,0 sur les
passages mesurés, aucun compilateur.

Lecture :

- Par classe, le routage égale le meilleur des deux modes à ±1 %, à l'intérieur
  du critère de ±2 %. Il ne régresse sur aucun cas de code.
- Sur les cas mixtes de 256 tokens, DFlash reste devant MTP (+3 % à +38 %),
  même quand la réponse est surtout de la prose (M2, M3). Envoyer ces consignes
  vers DFlash est donc correct à ce budget.
- Gain du routage sur le défaut actuel quand le drafter est chargé (DFlash
  partout) : +29 % (P1), +18 % (P2) et +5 % (P3) sur la prose courte, +17 %
  sur la prose à 32k. Le code ne change pas.

## Identité greedy

Aucune divergence : chaque génération mesurée, chauffes comprises, produit
exactement les 256 IDs de l'AR du même cas. Les oracles réels
`dflash_resident_matches_cpu_loop_real_model` (complété par le routage vers
MTP) et `dflash_tree_resident_matches_chain_and_ar_real_model` passent.

## Reproduction

```sh
taskpolicy -b cargo build --release --bins --example spec_route_measure
python3 -I -B "$GPU_LOCK_TOOL" run --owner spec-route --prio 5 -- \
  taskpolicy -b python3 -I -B tools/bench_spec_route_pairwise.py \
    --binary target/release/examples/spec_route_measure \
    --model "$MODEL" --drafter "$DRAFTER" --output .tmp/spec-route \
    --repeats 6 --calm-wait 90 --ids C1 C2 C3
```

Le corpus se trouve dans `tools/fixtures/spec_route_prompts.json`. Sans `--ids`,
le script mesure les onze cas. Il refuse de tourner hors du verrou GPU, retire
les flags `SARAGOSSA_*` hérités et écrit le manifeste, `results.jsonl` (IDs,
choix du routage, charge) et `summary.json`. Les tests CPU du protocole se
lancent avec `python3 -I -B tools/test_bench_spec_route_pairwise.py`.

Les résumés, la charge relevée et les empreintes de chaque série sont dans
[spec-route-results.json](spec-route-results.json), sans les IDs ni les chemins locaux.

## Décision de défaut

Le critère (±2 % du meilleur sur chaque classe, aucune régression sur le code)
est rempli sur les onze cas, courts et à 32k. Le mainteneur a validé
l'activation par défaut après relecture ; kill-switch
`SARAGOSSA_RUST_SPEC_ROUTE=0`.

Limites de l'heuristique : sa liste de mots est fixe et le budget mesuré est de
256 tokens. Sur une prose longue (500 tokens et plus), DFlash a parfois
rattrapé MTP (T0, T2 en phase 7). Ces deux prompts sont pourtant des demandes
de code, que le routage envoie déjà vers DFlash.
