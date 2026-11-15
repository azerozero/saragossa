# Modèles de décision Qwen3.5 (THX-02, d3-flash)

`saragossa::decision` expose le Decider joint de THX-02 (HAL-X-AI/THX-02,
Apache-2.0) et le format `d3-code-readout-v1` de d3-flash (vllm-sr/d3-flash,
Apache-2.0), en texte seul. Le format est détecté par `decision_config.json`
(`format_id`), sinon par `joint_config.json`. Vision, thinker (LoRA + tête MTP), TTA et le
découpage de haut niveau du runtime Python ne sont pas pris en charge et sont
refusés explicitement.

## Calcul

- Backbone Qwen3.5-4B : 24 couches gated-delta **causales** et 8 couches
  d'attention complète **non causales** (chaque requête voit toute la
  séquence, suffixe chat compris). Pas de cache de préfixe ni de découpage
  temporel : une couche reçoit toutes les positions avant la suivante.
- Prompt `joint-question-inline-readout-v1` : système fixe, `State:`, puis une
  section par question dans l'ordre JSON, terminée par `<|fim_suffix|>`. Les
  tokens spéciaux du contenu sont neutralisés par des chevrons pleine largeur.
- Lecture au **marqueur de chaque question** (et non au dernier token), après
  la RMSNorm finale. `readout.weight` `[255, hidden]` sans biais : seules les
  `n` premières lignes servent, puis `softmax(z / température)` en f32.
- Sorties : `choice` (probabilités, argmax, confiance), `noul` (p(true)),
  `score` (espérance, distribution, légende).

Limite : 2048 tokens, sans troncature.

## Interfaces

```sh
saragossa decide --model-dir DOSSIER < requetes.jsonl
```

Une requête JSON par document (`{"state": ..., "questions": {...}}`), une
réponse JSON par ligne. `--cpu` force le chemin de référence CPU.
`SARAGOSSA_DECISION_NAX=0` désactive l'attention NAX non causale.
`SARAGOSSA_DECISION_GEMM_BF16=0` remplace le GEMM matmul2d bf16 des
projections (activation arrondie en bf16, comme Torch bf16) par le GEMM NAX
hi + lo, plus précis et environ 1,3 fois plus lent. Les deux chemins passent
l'oracle ; la norme finale ne traite que les lignes des marqueurs.

Poids quantifiés (requantification min-max gs64 locale, mesurée sur l'oracle) :
u8 reste dans la tolérance (écart 0,0067, KL max 1,7e-4) mais est plus lent
que bf16 faute de GEMM dédié ; u4 sort de la tolérance (écart 0,10). Le
bf16 reste le format conseillé.

## Oracle

`tests/fixtures/decision/` contient la référence f32 (Torch MPS, couches
Transformers Qwen3.5 avec masque non causal explicite) : 32 questions courtes
et trois prompts de 128, 512 et 2048 tokens. La tolérance
(`tolerance.json`, 0,0122 en probabilité absolue) vaut deux fois le bruit bf16
mesuré sur la même référence plus 1e-6. Elle a été figée avant la première
comparaison Rust et ne s'élargit pas.

```sh
SARAGOSSA_DECISION_MODEL=DOSSIER cargo test --release --all-features \
  decision_real_weight_oracle -- --ignored
cargo run --release --example decision_bench -- DOSSIER tests/fixtures/decision
```

Les deux commandes lisent les poids réels : à lancer sous le verrou GPU.

## d3-flash

Backbone Qwen3.5-9B (hidden 4096, MLP 12288), mêmes couches et même masque
non causal que THX-02. Poids en shards (`model.safetensors.index.json`),
tenseurs `visual.*` ignorés, pas de tête de vocabulaire. Différences :

- **une séquence par question**, l'état répété dans chacune ; prompt
  `d3_format.user_prompt` : `State:`, `Question:`, `Options:` puis
  `Reply with only the code of the best option.`, système propre à d3 ;
  état `null` ou vide rendu `(empty)` ;
- lecture au **dernier token** ; `n` premières lignes du readout f32
  `[255, 4096]`, température 1,0 ; les `token_ids` valident seulement le
  tokenizer ;
- `score` limité à 2 à 10 niveaux distincts, légende en JSON compact trié ;
  confiance `choice`/`score` = 1 − entropie normalisée ;
- une question invalide ou trop longue reçoit `{type, error, message}`
  (`invalid_question`, `max_length_exceeded`) sans bloquer les autres.
  `permutation_average` n'est pas pris en charge.

Oracle : `tests/fixtures/decision/d3/`, 39 questions et prompts de 128, 512
et 2048 tokens, tolérance figée 0,0184 (deux fois le bruit bf16 mesuré
+ 1e-6).

```sh
SARAGOSSA_DECISION_D3_MODEL=DOSSIER cargo test --release --all-features \
  decision_d3_real_weight_oracle -- --ignored
cargo run --release --example decision_bench -- DOSSIER tests/fixtures/decision/d3
```

openjev (27B) devra fournir ses propres oracles ; sa configuration et son
protocole restent à vérifier avant tout support.

Écart assumé avec les références Python : saragossa neutralise tous les
`added_tokens` marqués `special` du tokenizer (21 pour THX-02), alors que
`all_special_tokens` de Transformers n'en liste que 9. Un texte utilisateur
qui contient par exemple `<|im_start|>` littéral ne peut donc pas injecter
de tour de conversation ; les prompts sans token spécial littéral sont
identiques. Le runtime d3 n'échappe rien ; saragossa applique la même
neutralisation aux 14 tokens spéciaux de d3, et sa référence aussi.
