# Bench concurrents — Qwen3.8-27B, batch 1, M5 Max

Harnais du bench saragossa / llama.cpp / mlx-lm / llmd. Le protocole (v3) fixe
les règles ; ce dossier les exécute. Python stdlib, plus le venv MLX
(`tokenizers`, `jinja2`, `transformers`) pour les prompts et l'analyse.

```sh
VENV=models/mlxenv/bin/python
RUN=/chemin/du/run          # contient corpus/wikitext-2-raw/ (zip sha256 ef7edb56…)

bench/competitors/prepare_weights.sh                       # GGUF Q4_0 pur, Q4_K_M, tête MTP Q4_0
python3 -I bench/competitors/weights_report.py FICHIERS… > "$RUN/weights.json"
WIKITEXT_TRAIN="$RUN/corpus/wikitext-2-raw/wiki.train.raw" \
  $VENV -I bench/competitors/prompts.py build "$RUN/prompts"

bench/competitors/night.sh validate "$RUN"   # essai court, puis relecture
bench/competitors/night.sh 1 "$RUN"          # réglage, freeze, eval-S, eval-M
bench/competitors/night.sh 2 "$RUN"          # qualité, échantillonné, 256k
```

| Fichier | Rôle |
|---|---|
| `client.py` | une requête SSE chronométrée, HTTP/1.1 sur socket brut |
| `run_matrix.py` | `env`, `plan` (graine publiée), `run` (reprise), `freeze` (règle de sélection) |
| `matrix.json` | configurations, grille de réglage, phases, valeurs neutres par moteur |
| `prompts.py` | prompts neufs à longueur exacte, empreintes attendues des ids |
| `launch_*.sh`, `paths.sh` | commande complète de chaque moteur |
| `memwatch.py` | `phys_footprint` et maximum à vie (`proc_pid_rusage`), `vmmap`, `vm_stat` |
| `analyze.py` | contrôles, T1/T2/T3, bootstrap par prompt, intégrité greedy |
| `quality.py`, `quality_llama.sh` | perplexité au protocole `llama-perplexity`, KL, `llama-bench` |

Auto-tests : `client.py`, `run_matrix.py`, `analyze.py`, `memwatch.py` et
`weights_report.py` lancés sans argument (`python3 -I -B …`) ; `prompts.py`
vérifie ses vecteurs à chaque lancement.
