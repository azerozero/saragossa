# Bench concurrents — Qwen3.8-27B, batch 1, M5 Max

Harnais du bench saragossa / llama.cpp / mlx-lm / llmd. Le protocole (v4) fixe
les règles ; ce dossier les exécute. Python stdlib, plus le venv MLX
(`tokenizers`, `jinja2`, `transformers`) pour les prompts, le gel et l'analyse.

```sh
cp bench/competitors/local.env.example bench/competitors/local.env   # puis renseigner
VENV=models/mlxenv/bin/python
RUN=/chemin/du/run

bench/competitors/prepare_weights.sh                     # GGUF Q4_0 pur, Q4_K_M, tête MTP Q4_0
python3 -I bench/competitors/weights_report.py FICHIERS… > "$RUN/weights.json"
$VENV -I bench/competitors/prompts.py build "$RUN/prompts"

bench/competitors/night.sh validate "$RUN"   # essai court, puis relecture
bench/competitors/night.sh 1 "$RUN"          # réglage, freeze, eval-S, eval-M
bench/competitors/night.sh 2 "$RUN"          # qualité, échantillonné, 256k
```

Relancer une nuit interrompue la reprend : les plans déjà tirés sont
réutilisés, les entrées terminées sautées, un gel écrit n'est pas refait.
La reprise et l'analyse ignorent avec avertissement une unique dernière ligne
JSON tronquée de `results.jsonl`, sans fin de ligne. Toute corruption au milieu
du fichier reste fatale. La reprise retire ce fragment avant d'ajouter des
résultats ; l'analyse laisse le journal intact.

Avant la nuit 1, l'essai de validation fixe `usage_counts_eos` de chaque
moteur dans `matrix.json` : sans cette convention, aucun débit n'est exact
et le gel s'arrête.

Le gel ne conserve le défaut sans réglage que si toutes les répétitions de
tous les candidats divergent **de façon prouvée** de leur référence AR
admissible (`greedy_identity == false`), avec les autres preuves vraies et un
débit exact disponible. Une identité absente, une référence absente ou non
admissible, ou un débit approximatif arrête le gel ; ce n'est pas une divergence.

`eval-S` utilise trois blocs de douze prompts disjoints, soit 36 prompts vus
une seule fois par configuration : six chats, trois documents Rust 4k et trois
proses 4k par bloc. `prompt_sets[block]` remplace les prompts de la classe pour
toutes les configurations du bloc ; sa longueur doit égaler `blocks`. Le
premier ensemble reste `chat-1..6`, `code4k-1..3`, `prose4k-1..3`. Les autres
phases, dont `sampled-S` avec les douze prompts historiques de la classe S,
gardent leurs tirages. Le nombre de générations longues passe de 60 à 36 par
configuration (×0,6), hors sondes TTFT.

Le champ de phase `ttft_repeats` ajoute deux sondes de huit tokens pour
chaque document 4k des configurations `sar-ar`, `llq40-ar`, `llkm-ar`, `mlx-ar`
et `llmd-ar`, soit trois observations TTFT avec la requête longue. Ces requêtes
`ttft_only` passent après toutes les requêtes longues de l’entrée, mélangées
entre elles, avec des répétitions et nonces distincts, communs entre AR appariés.
La médiane TTFT AR mélange requêtes longues et sondes. Un échec de sonde ne
peut abandonner une génération longue déjà mesurée. Elles
suivent le même chemin SSE et les mêmes preuves d'entrée, de sortie non vide
et d'absence de cache. Leur budget propre est vérifié : huit tokens suffisent,
aucune preuve ne leur impose les 400 tokens des générations normales.
Elles alimentent uniquement le TTFT 4k, jamais le débit, les latences de
génération, l'identité greedy ou le gel ; elles ne raccourcissent pas L.
Les diagnostics d'admissibilité les comptent séparément (`ttft_only`), sans
les signaler comme générations anormalement courtes ou approximatives.

Le bootstrap apparié reconnaît les prompts imbriqués dans les blocs : il tire
les blocs avec remise, puis les prompts de chaque bloc tiré, puis leurs clés
avec remise. Le chemin croisé historique conserve exactement ses tirages.
Un mélange de prompts propres à un seul bloc et de prompts présents dans
plusieurs blocs parmi les paires disponibles est refusé explicitement.
En plan imbriqué, le verdict reste « indistinct » sauf si l’IC exclut 1 et si
l’écart au ratio 1 atteint 5 %. Avec 3 blocs, la couverture mesurée par Monte
Carlo est de 92–97 % sans effet bloc, et de 73–82 % si l’effet bloc est
comparable à l’effet prompt : les tableaux signalent « IC approximatif (3 blocs) ».
Les paires manquantes ne comptent que les clés prévues pour les deux configurations.

Les nouvelles proses utilisent les rangs admissibles wiki 4 à 9 : le rang 3
reste réservé au réglage, `wikicat` commence à 10. L'auto-test contrôle cette
séparation. Les six nouveaux fichiers Rust sont distincts des sources
existantes et hors de `core/src`, concaténé pour le contexte long ; `build`
refuse une source Rust trop courte avant de la découper.

Préparer un **nouveau dossier de run** pour ce plan : reconstruire
`prompts/prompts.json` avec `prompts.py build`, puis régénérer
`prompts/expected.json` avec `prompts.py expect` pour chaque plan publié,
sondes courtes comprises. Les empreintes de contenu et d'ids sont calculées
à partir des sources et des tokenizers locaux ; aucune valeur de remplacement
n'est fournie si le corpus wikitext est absent. Les SHA-256 publiés du corpus
restent inchangés. Un ancien run garde ses plans, sa matrice et ses empreintes.

| Fichier | Rôle |
|---|---|
| `client.py` | une requête SSE chronométrée, HTTP/1.1 sur socket brut, fragments bruts conservés |
| `run_matrix.py` | `env`, `plan` (graine publiée), `run` (reprise idempotente), `freeze` (règle de sélection), `reuse-freeze` (réglages sur protocole courant) |
| `matrix.json` | configurations, grille de réglage, phases, valeurs neutres et convention EOS par moteur |
| `prompts.py` | prompts neufs à longueur au plus la cible, empreintes attendues (bloquantes) |
| `launch_*.sh`, `paths.sh`, `config.py` | commande complète de chaque moteur, chemins locaux |
| `sensors.py` | un collecteur macmon continu, journal brut et agrégats par requête |
| `budget.py` | temps consommé par phase, journal brut ou gzip, en minutes |
| `memwatch.py` | `phys_footprint` et maximum à vie (`proc_pid_rusage`), `vmmap`, `vm_stat` |
| `analyze.py` | admissibilité, comptage par événement, T1/T2/T3, bootstrap apparié |
| `quality.py`, `quality_llama.sh` | ids du corpus, perplexité au protocole `llama-perplexity`, KL, `llama-bench` |

Auto-tests : `client.py`, `run_matrix.py`, `analyze.py`, `results.py`, `memwatch.py` et
`weights_report.py` lancés sans argument (`python3 -I -B …`) ; `prompts.py`
vérifie ses vecteurs et ses sources disjointes à chaque lancement, et sans
argument affiche le résultat de cet auto-test. Ces tests utilisent la stdlib
et des données synthétiques, sans moteur ni modèle. Les variables de chemins
de `paths.sh` doivent être définies, même si les fichiers ne sont pas ouverts.
Les tests de plan comparent les empreintes historiques des cinq autres phases ;
ceux d'analyse vérifient le résultat et l'état du générateur aléatoire croisés,
l'effet de bloc imbriqué, le refus du mélange et l'exclusion des sondes du gel.

La campagne refuse de démarrer ou s’arrête si `pmset` indique la batterie ou
le mode basse consommation ; elle se reprend après correction. L’alimentation
est contrôlée avant et après le chargement, avant chaque requête hors
chronométrage et avant `entry_done`. La reprise rejoue toute l’entrée interrompue
et marque ses lignes antérieures `superseded`. L’état est enregistré dans
l’environnement, chaque chargement et la ligne `unload`. Une erreur de lecture
de `pmset` est consignée sans bloquer. Les nuits `2` et `partial` contrôlent aussi
l’alimentation avant la qualité et entre ses sous-étapes (ids, PPL/KL,
llama-bench et PPL MLX). Une infrastructure indisponible (port occupé, échecs
de bind épuisés) arrête la nuit avec un code non nul. Un plan vide est refusé.

Un seul `macmon pipe -i 1000` accompagne tout `run`, y compris les pauses et
chargements. `sensors.jsonl` conserve ses lignes brutes ; les mesures et chauffes
reçoivent températures CPU/GPU début, maximum, fin et moyenne, fréquence GPU
médiane, activité GPU médiane, puissance système moyenne, usage CPU moyen
(`cpu_usage_pct` brut de macmon, fraction) et nombre d’échantillons. Seuls les
échantillons reçus pendant la requête sont agrégés. Capteur absent, arrêté ou
valeur zéro : champ nul, sans échec de la campagne. Les relevés ponctuels de
ventilateurs conservent toutefois `rpm: 0` (ventilateur arrêté) ; seule une
valeur absente devient nulle. Une requête sans échantillon
a un compteur nul et des métriques nulles. En plan imbriqué sans répétition
comparable, l’analyse publie la moyenne des températures GPU moyennes et la
médiane des fréquences GPU médianes des requêtes, par position de bloc.

Avant chaque chargement, le GPU doit présenter trois lectures distinctes sous
`busy_gpu_ratio` (0,2). L’attente est bornée par `busy_wait_max_s` (300 s),
consignée dans `busy_wait_s` ; à expiration, `machine_busy` vaut vrai et l’entrée
continue. Sans capteur d’activité utilisable, l’attente est omise.
`cooldown_mode: fixed` conserve l’attente historique. Le mode optionnel `sensor`
attend au moins `cooldown_min_s`, puis une température GPU réelle inférieure ou
égale à `cooldown_target_gpu_c` (45 °C), au plus `cooldown_max_s` depuis le début
de la pause ; sans température, il reprend le comportement `fixed`. Ces valeurs
sont dans `matrix.json`. Le mode `sensor` reste à valider sur secteur avant
activation. `warmup_max_tokens`, facultatif dans une classe, fixe son budget de
chauffe ; son absence conserve le budget habituel.

Pour un checkpoint changé, reprendre les réglages d’un ancien gel dans un
**nouveau** dossier de run, reconstruire ses prompts et lancer :

```sh
export BENCH_MLX_CKPT=/chemin/vers/le/checkpoint
python3 -I -B bench/competitors/run_matrix.py reuse-freeze "$RUN" "$ANCIEN_GEL"
# Reconstruire les prompts du nouveau run avant la nuit.
bench/competitors/night.sh partial "$RUN"
python3 -I -B bench/competitors/budget.py "$RUN"
```

`BENCH_MLX_CKPT` surcharge le chemin commun aux lanceurs saragossa et MLX, aux
prompts et à la qualité ; `environment()` enregistre le chemin effectif.
Sans cette variable, les chemins actuels restent utilisés. `partial` exige
`matrix.frozen.json` avec `phases.eval-S.prompt_sets` ; un ancien protocole
est refusé avec le code 2. `reuse-freeze` reprend uniquement les `configs.*.slots`
renseignés par `freeze`, sur le `matrix.json` courant, et consigne le SHA-256 de
l’ancien gel dans `reuse_freeze_sha256`. Une configuration ou un slot gelé
disparu est refusé, comme une destination déjà gelée. Les phases, prompts et
autres paramètres courants sont conservés. Aucun réglage n’est remesuré.
La nuit enchaîne qualité, `eval-S`, `sampled-S` et `eval-L-mlx` (sar-ar, sar-mtp, sar-dflash, mlx-ar). Un ancien gel
sans cette dernière phase reprend les paramètres de sa phase `eval-L` avec ces
quatre configurations. Les auto-tests de `run_matrix.py` et `analyze.py` précèdent
toute étape de `night.sh`.

`budget.py` additionne pauses (attente d’occupation comprise), chargements,
chauffes, sondes TTFT et de cache, prefill et génération des requêtes longues.
Une requête sans événement de sortie compte toute sa durée en prefill et zéro
en génération. Il compte toutes les tentatives réellement effectuées, même
remplacées à la reprise ; le total exclut les coûts sans durée enregistrée, notamment les arrêts
et les preuves HTTP.

Les outils `tools/quant/opt_quant.py` et `tools/quant/convert_ls.py` ajustent la
quantification u4 gs64 depuis un BF16 avec un checkpoint u4 comme template :
`VENV_PY -I -B tools/quant/convert_ls.py --src BF16 --template U4 --dst NOUVEAU`.
La destination doit être inexistante ; toutes les clés BF16 sont vérifiées
avant sa création. Sans argument, leurs auto-tests utilisent la stdlib ; l’oracle scalaire couvre l’ajustement partagé et les échelles
négatives, sans valider l’exécution tensorielle MLX ni lancer de conversion.
`sensors.py`, `budget.py` et `night_checks.py` ont également un auto-test sans
argument ; les lanceurs sont vérifiés sur des copies avec exécutables factices.
