# Diagnostic couche par couche — Gemma 4 MoE

Ces outils capturent le **pas 0** : un prefill sur les IDs exacts du prompt,
les activations de toutes ses positions, puis les logits de sa dernière position
après softcap (avant log-softmax). Aucun tokenizer n'est appelé pour construire
le prompt. Ils ne génèrent pas de continuation.

Le JSON du golden est un index de provenance ; les `prompt_ids` sont dans le
payload UTF-8 `.bin` adjacent. L'extraction vérifie son SHA256. Le prompt 0 du
golden Gemma 4 contient `[53406, 236764, 2185, 520, 236789, 101652]`.

## Exécution locale

Définir les chemins locaux, hors dépôt public :

```sh
export REF_PYTHON=/chemin/vers/venv-ref/bin/python
export GOLD_WORKTREE=/chemin/vers/worktree/tool-gold
export SARAGOSSA_GEMMA4_DIR=/chemin/vers/checkpoint-local
export DUMP_ROOT="$(mktemp -d /tmp/gemma4-layers.XXXXXX)"
```

Depuis la racine du worktree de diagnostic, préparer les IDs et le binaire
(sans inférence ni GPU, sans téléchargement) :

```sh
"$REF_PYTHON" -I -B tools/goldens/dump_layers_mlx.py \
  --golden "$GOLD_WORKTREE/tests/fixtures/goldens/logits/gemma4.json" \
  --prompt-index 0 --write-ids "$DUMP_ROOT/prompt0.json"
taskpolicy -b cargo build --offline --release --example dump_layers
```

Lancer ensuite chaque capture sous le verrou de l'opérateur. Le lanceur fourni
par le dépôt peut exécuter les trois commandes séquentiellement :

```sh
python3 -I -B tools/gpulock.py run --owner gemma4-layers-mlx --prio 10 -- \
  "$REF_PYTHON" -I -B tools/goldens/dump_layers_mlx.py \
  --model "$SARAGOSSA_GEMMA4_DIR" --ids-file "$DUMP_ROOT/prompt0.json" \
  --out "$DUMP_ROOT/mlx"

python3 -I -B tools/gpulock.py run --owner gemma4-layers-cpu --prio 10 -- \
  taskpolicy -b target/release/examples/dump_layers \
  "$SARAGOSSA_GEMMA4_DIR" "$DUMP_ROOT/prompt0.json" "$DUMP_ROOT/cpu" cpu

python3 -I -B tools/gpulock.py run --owner gemma4-layers-metal --prio 10 -- \
  env SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4=0 \
  taskpolicy -b target/release/examples/dump_layers \
  "$SARAGOSSA_GEMMA4_DIR" "$DUMP_ROOT/prompt0.json" "$DUMP_ROOT/metal" metal
```

Le backend par défaut du binaire est `metal`. `cpu` n'instancie aucun executor
Metal. Le chemin Metal capturé est le prefill **per-op par défaut de Gemma 4**,
identique au choix de `prefill_cache` à cache froid. Le diagnostic court-circuite
uniquement le prefix cache et les snapshots disque ; il conserve les réglages
usuels de batching/chunking. Les captures sont agrégées par position si le
prefill est découpé ou tokenwise. Le chemin résident fusionné optionnel
`SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4=1` est explicitement refusé, car ses
sous-étapes ne sont pas instrumentées. Ce diagnostic ne qualifie pas le decode
entièrement résident.

Côté bibliothèque, `CausalDecoder::dump_gemma4_layers(prompt, directory)` est
exposé sous la feature existante `devtools`. Les observations sont limitées au
thread appelant et désactivées hors de cette portée. Les sorties sont copiées,
sans remplacement des calculs ni changement du dtype. Les modèles autres que
Gemma 4 MoE sont refusés. MLX utilise `load_model(Path)` sur les shards locaux,
sans téléchargement ni chargement du tokenizer. Versions exigées :
`mlx-lm==0.32.2`, `mlx==0.32.4` ; NumPy est utilisé pour le format et la comparaison.
Les méthodes originales de `gemma4_text.py` et de ses modules sont enveloppées
en mémoire puis restaurées, sans modification du paquet.

Comparer sur CPU, hors verrou :

```sh
"$REF_PYTHON" -I -B tools/goldens/compare_layers.py "$DUMP_ROOT/mlx" "$DUMP_ROOT/cpu" \
  --json "$DUMP_ROOT/mlx-cpu.json"
"$REF_PYTHON" -I -B tools/goldens/compare_layers.py "$DUMP_ROOT/mlx" "$DUMP_ROOT/metal" \
  --json "$DUMP_ROOT/mlx-metal.json"
"$REF_PYTHON" -I -B tools/goldens/compare_layers.py "$DUMP_ROOT/cpu" "$DUMP_ROOT/metal" \
  --json "$DUMP_ROOT/cpu-metal.json"
```

Codes de sortie : **0** accord selon les seuils, **1** divergence, **2** capture
invalide. Un code 1 est attendu lors du diagnostic : lancer les comparaisons
séparément si le shell utilise `set -e`. Les répertoires et rapports existants
ne sont jamais écrasés.

## Points de mesure et format

Chaque répertoire contient `index.json` et un fichier par point : float32 IEEE
754 little-endian, contigu en ordre C, forme `[tokens, largeur]`. Les logits et
`final_norm` ne gardent que la dernière position, forme `[1, largeur]`. Les
valeurs BF16 de MLX sont converties **après le calcul**, uniquement pour écrire
le fichier ; `source_dtype` conserve le dtype observé. `index.json` n'est écrit
qu'après une capture complète. Une interruption laisse un répertoire incomplet,
que le comparateur refuse.

| Point | Valeur observée |
|---|---|
| `embeddings` | Embeddings après multiplication par l'échelle Gemma |
| `input` | Entrée de couche |
| `attention_input` | Après `input_layernorm` |
| `attention_raw` | Projection O de l'attention, avant norme post-attention |
| `attention_out` | Après norme post-attention, avant addition résiduelle |
| `post_attention_residual` | Entrée de couche + `attention_out` ; source du routeur |
| `dense_input`, `dense_raw`, `dense_out` | Norme d'entrée dense, MLP dense, norme de sortie dense |
| `router_input`, `router_logits` | Entrée normalisée du routeur, scores avant top-k |
| `router_indices`, `router_weights` | Experts triés par ID, poids softmax après modulation par expert |
| `moe_input`, `moe_raw`, `moe_out` | Norme d'entrée experts, somme pondérée des experts, norme de sortie MoE |
| `ffn_sum` | Addition des branches dense et MoE normalisées |
| `ffn_out` | Après norme feed-forward commune, avant résiduel |
| `pre_scalar` | Résiduel post-attention + `ffn_out`, avant `layer_scalar` |
| `output` | Sortie de couche après `layer_scalar` |
| `final_norm`, `logits` | Norme finale et tête après softcap, dernière position |

`pre_scalar` côté MLX est la seule expression recalculée pour observation :
`residual + ffn_out`, dans le dtype original. Elle ne remplace aucune valeur
consommée par le modèle. Les indices d'experts sont des entiers exactement
représentables en f32. Le tri affecte uniquement les copies de diagnostic.

## Lecture des résultats

Le comparateur donne, par point et par couche : maximum absolu, erreur relative
L-inf (`max|diff| / max|référence|`), maximum relatif élémentaire (dénominateur
borné par `atol`), cosinus global, pire cosinus par position et maximum absolu
sur la dernière position. Le rapport JSON conserve aussi la position/coordonnée
du maximum, la première position hors seuil et le nombre d'éléments hors seuil.
Les NaN/Inf, formes incompatibles, points manquants et prompts différents sont
refusés, jamais assimilés à un accord.

Par défaut, une différence est significative dès qu'un élément vérifie
`abs(diff) > 1e-5 + 1e-3 * abs(référence)` ou qu'un cosinus par position est
inférieur à `0.99999`. Ces seuils **diagnostiques** sont configurables via
`--atol`, `--rtol`, `--min-cosine` ; ils ne sont pas un contrat d'équivalence BF16.
Les ensembles d'experts sont comparés exactement, quels que soient ces seuils.
Un changement d'experts affiche les IDs et la marge entre les scores classés
k et k+1 pour chaque backend. Le greedy final est affiché séparément.

La première divergence est cherchée dans l'ordre des couches puis dans l'ordre
des dépendances ci-dessus. Les branches dense, routeur et experts partagent une
entrée et certaines de leurs opérations sont indépendantes. Une divergence à
`embeddings` ou `input` peut se propager partout : cela ne démontre pas un défaut
dans chaque sous-étape suivante. Les sous-opérations internes de l'attention
(Q/K, RoPE, SDPA) ne sont pas séparées dans cette première instrumentation.

Pour tester la piste BF16, effectuer une capture de contrôle MLX distincte :

```sh
python3 -I -B tools/gpulock.py run --owner gemma4-layers-mlx-f32 --prio 10 -- \
  "$REF_PYTHON" -I -B tools/goldens/dump_layers_mlx.py \
  --model "$SARAGOSSA_GEMMA4_DIR" --ids-file "$DUMP_ROOT/prompt0.json" \
  --float32 --out "$DUMP_ROOT/mlx-f32"
"$REF_PYTHON" -I -B tools/goldens/compare_layers.py "$DUMP_ROOT/mlx-f32" "$DUMP_ROOT/cpu"
```

Ce contrôle convertit les paramètres flottants et les scales en F32, en gardant
les poids quantifiés empaquetés. Il modifie le dtype de tout le forward, y
compris les constantes d'échelle : il ne suffit pas à isoler le seul routeur.

## Vérification sans GPU ni modèle réel

```sh
"$REF_PYTHON" -I -B tools/goldens/test_layers.py
cargo fmt --all -- --check
taskpolicy -b cargo clippy --offline --all-targets --all-features -- -D warnings
taskpolicy -b cargo build --offline
taskpolicy -b cargo test --offline --lib --no-default-features --features devtools
```

Les tests Rust utilisent la fixture synthétique Gemma 4 : ils vérifient que la
capture laisse les logits inchangés, couvre tous les points et positions,
respecte les frontières résiduelles et refuse l'écrasement. Metal est désactivé
pour les tests bibliothèque car des tests GPU existants ne sont pas ignorés.
