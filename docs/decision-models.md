# Modèles de décision Qwen3.5 (THX-02, d3-flash)

`saragossa::decision` expose le Decider joint de THX-02 (HAL-X-AI/THX-02,
Apache-2.0) et le format `d3-code-readout-v1` de d3-flash (vllm-sr/d3-flash,
Apache-2.0), en texte seul. Le format est détecté par `decision_config.json`
(`format_id`), sinon par `joint_config.json`. Vision, thinker (LoRA + tête
MTP) et le découpage en morceaux de questions du runtime Python ne sont pas
pris en charge et sont refusés explicitement. THX-02 suit par défaut les
règles de lecture de `runtime.plan` (facettes jointes, sinon une lecture par
question) et sa moyenne sur plusieurs ordres d'options (`tta`, voir plus bas).

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

Le chargement d'un modèle de décision ne lit ni ne lie de tête de vocabulaire :
le readout la remplace. Une tête stockée dans les poids est jetée après
lecture. Footprint mesuré inchangé par rapport à la tête liée, qui partageait
déjà le buffer des embeddings (d3-flash 15,3 Gio, THX-02 8,1 Gio).

## Interfaces

```sh
saragossa decide --model-dir DOSSIER < requetes.jsonl
```

Une requête JSON par document (`{"state": ..., "questions": {...}}`, plus
`tta` et `joint` pour THX-02), une
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

## Moyenne sur les ordres d'options (TTA, THX-02)

THX-02 suit les règles de lecture de `runtime.plan`, celles des scores
publiés. Champ `tta` de la requête, défaut 1 comme
`THX02.load(default_tta=1)` de la référence, ramené à 1..=8 :

- **lecture jointe de facettes** (`engine.facet_joint`) : au moins deux
  questions, toutes à `instructions` textuelles, de même type et aux critères
  identiques (à l'ordre des clés près) ; une seule lecture jointe, `tta`
  ignoré ;
- sinon **chaque question est lue seule** (requête `{state, {clé: question}}`),
  `noul` et `score` une fois, une `choice` à au moins deux options dans les
  `k` premiers ordres distincts parmi identité, inverse, puis décalages
  circulaires `i ↦ (i + s) mod n` (`engine.orders`). À 4 ordres et 3 options :
  `012`, `210`, `120`, `201` ; à 2 options, seuls 2 ordres existent. Les codes
  `A`, `B`… restent dans l'ordre du prompt, seules les options bougent ;
- chaque distribution est remise dans l'ordre d'origine par clé d'option, puis
  `p_i ∝ exp(moyenne_j ln max(p_ji, 1e-9))`, renormalisée (`engine.geo_mean`).
  Choix, confiance et score sont recalculés sur cette distribution.

`"joint": true` force la lecture jointe de toutes les questions en une
séquence (`Decider.decide`, l'ancien comportement), sans `tta`. d3 refuse
`tta` > 1 et `joint` (son `permutation_average` n'est pas porté).

Calcul partagé. Les 8 couches d'attention complète sont non causales : les
lignes de l'état voient les options qui les suivent, donc aucun état de
préfixe n'est réutilisable au-delà de la première de ces couches (couche 3).
Les embeddings et les trois premières couches gated-delta du préfixe commun
seraient exacts à partager, mais ne pèsent qu'environ 9 % du calcul d'une
lecture ; ce n'est pas fait. Les lectures sont plutôt **empaquetées dans un
même prefill** (`CausalDecoder::decision_hidden_packed`) : normes, résiduel
et MLP (environ 63 % des poids) traitent toutes les lignes d'un coup, poids
lus une fois ; projections, RoPE, récurrence gated-delta et attention restent
séquence par séquence, chacune avec ses positions et son état neuf. Sur
l'oracle, les probabilités empaquetées égalent celles des lectures séparées. Un
paquet compte au plus 512 tokens (`SARAGOSSA_DECISION_PACK_TOKENS`, 0 :
lectures séparées) : au-delà, le GEMM est déjà saturé. Mesures provisoires
(machine utilisée, ABBA, 21 répétitions) : à 128 tokens, TTA=4 empaqueté
175 ms contre 209 ms en quatre prefills (−16 %) ; deux lectures de 512
empaquetées oscillent entre −8 % et +3 % selon le passage, d'où le seuil ;
tout empaqueter (8 192 lignes) est 11 % plus lent à 2048.

Oracle : `tests/fixtures/decision/tta/`, 12 requêtes TTA=4 passées par
`runtime.plan` avec la référence f32 MPS : choice à 2, 3, 4 et 6 options,
état de 567 tokens, requêtes mixtes non-facettes (choice/noul/score,
instructions absentes, critères différents) et facettes (noul, choice, score).
Tolérance figée 0,0165 (deux fois le bruit bf16 mesuré sur les probabilités
TTA=4, + 1e-6), inchangée à l'ajout des requêtes. Les lectures à `tta` = 1
sont vérifiées par leurs tokens (sous-ensemble à l'ordre d'origine), pas par
leurs probabilités : une lecture isolée a un bruit bf16 de 0,0088 sur ce
corpus, que ce seuil ne couvre pas et qui n'est pas élargi pour elle.

```sh
SARAGOSSA_DECISION_MODEL=DOSSIER cargo test --release --all-features \
  decision_tta_real_weight_oracle -- --ignored
cargo run --release --example decision_tta_bench -- DOSSIER
```

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
et 2048 tokens, tolérance figée 0,0184 (deux fois le bruit bf16 mesuré,
plus 1e-6).

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
