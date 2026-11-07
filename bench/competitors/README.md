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

| Fichier | Rôle |
|---|---|
| `client.py` | une requête SSE chronométrée, HTTP/1.1 sur socket brut, fragments bruts conservés |
| `run_matrix.py` | `env`, `plan` (graine publiée), `run` (reprise idempotente), `freeze` (règle de sélection) |
| `matrix.json` | configurations, grille de réglage, phases, valeurs neutres et convention EOS par moteur |
| `prompts.py` | prompts neufs à longueur au plus la cible, empreintes attendues (bloquantes) |
| `launch_*.sh`, `paths.sh`, `config.py` | commande complète de chaque moteur, chemins locaux |
| `memwatch.py` | `phys_footprint` et maximum à vie (`proc_pid_rusage`), `vmmap`, `vm_stat` |
| `analyze.py` | admissibilité, comptage par événement, T1/T2/T3, bootstrap apparié |
| `quality.py`, `quality_llama.sh` | ids du corpus, perplexité au protocole `llama-perplexity`, KL, `llama-bench` |

Auto-tests : `client.py`, `run_matrix.py`, `analyze.py`, `results.py`, `memwatch.py` et
`weights_report.py` lancés sans argument (`python3 -I -B …`) ; `prompts.py`
vérifie ses vecteurs à chaque lancement.
