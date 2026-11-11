# Références externes

Ces scripts n'utilisent que des modèles locaux, en lecture seule. Ils forcent
le mode hors ligne et exigent `SARAGOSSA_GOLDEN_CAPTURE=1`. Installer les versions
indiquées en tête des scripts dans un environnement **hors dépôt**, puis définir
`REF_PYTHON` vers son interpréteur. Aucun script ne crée cet environnement.

Les fichiers dans `tests/fixtures/goldens/` reprennent le couple `.bin`/`.json`
de `src/golden.rs` : f32 little-endian pour les vecteurs, JSON UTF-8 exact dans
les `.bin` pour les structures et textes. Les headers contiennent dtype, shape,
tolérance, date, SHA256 du payload et provenance : modèle, chemin portable via
variable d'environnement, snapshot éventuel, SHA256 des configs/tokenizers et
versions réellement chargées. Les poids ne sont ni copiés ni téléchargés.
`mlx_sha=not-applicable` distingue ces références Python des anciens goldens mlx-rs.
Les hooks de normalisation de texte excluent ces `.bin` : ajouter un saut de
ligne à un payload UTF-8 invaliderait sa taille et son empreinte.

| Nom | Variable locale | Référence |
|---|---|---|
| qwen38 | `SARAGOSSA_QWEN38_DIR` | Qwen3.8-27B-u4ls |
| qwen36_moe | `SARAGOSSA_QWEN36_MOE_DIR` | Qwen3.6-35B-A3B-4bit |
| gemma3 | `SARAGOSSA_GEMMA3_DIR` | mlx-community/gemma-3-4b-it-4bit |
| gemma4 | `SARAGOSSA_GEMMA4_DIR` | mlx-community/gemma-4-26b-a4b-it-4bit |
| qwen3_tts | `SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR` | Qwen3-TTS VoiceDesign 1.7B 6bit |
| whisper_tiny | `SARAGOSSA_WHISPER_TINY_DIR` | openai/whisper-tiny |
| whisper_small | `SARAGOSSA_WHISPER_SMALL_DIR` | openai/whisper-small |
| embedder | `SARAGOSSA_EMBEDDER_DIR` | intfloat/multilingual-e5-small |

Les scripts résolvent sinon les snapshots uniques dans le cache HF local ; pour
les deux Qwen LLM, ils cherchent le dossier de même nom sous `~/workspace/ai_model`.
En cas de plusieurs snapshots, définir explicitement la variable. Les oracles
Rust utilisent les variables ci-dessus et vérifient les empreintes avant l'inférence.
Sans variable, ils indiquent `skip` ; `SARAGOSSA_REQUIRE_REAL_MODELS=1` rend cette
absence bloquante. Ils sont `#[ignore]`, comme les oracles existants du dépôt.

## Capture CPU

```sh
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I tools/goldens/chat_templates.py
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I tools/goldens/embedder.py
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I tools/goldens/whisper.py --model whisper_tiny
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I tools/goldens/whisper.py --model whisper_small
```

Templates : conversations multi-tours, outils déclarés, appel et résultat,
avec/sans prompt de génération et thinking si le template le prévoit. Les
rejets de rôle par transformers sont enregistrés comme tels. Qwen3-TTS n'a pas
de template de chat dans ce snapshot : le fichier enregistre cette absence.
Le test direct Gemma3 restitue le BOS fourni par le tokenizer ; le test de
`serve` appelle réellement `LoadedModel::prompt_encoding` avec un décodeur
sentinelle non exécuté, puis compare les IDs encodés à transformers. Les cas
sans prompt de génération et thinking=true ne concernent que le test direct,
car `serve` ne les expose pas.

Embedder : 20 phrases FR/EN/code, requêtes et passages alternés, préfixes E5
explicites, mean pooling masqué et normalisation L2. Le cache n'a pas
`modules.json` : ces modules sont construits explicitement avec
sentence-transformers. Tolérances fixées avant comparaison : cosinus ≥ 0,9999
et max |diff| ≤ 0,001. Le diagnostic donne les deux mesures pour chaque phrase.

Whisper : PCM 16 kHz de `tests/golden/whisper_samples16k.bin`, langue française
forcée. Les snapshots tiny/small n'ont pas de preprocessor/generation config :
frontend Whisper explicite (80 mels, FFT 400, hop 160, fenêtre 30 s), IDs de
langue/tâche issus du tokenizer local. Deux décodages indépendants sont capturés :
greedy sans timestamps, et greedy avec contraintes temporelles transformers.
Les deux utilisent les suppressions du config local. Vérification de la version
5.22.2 : les champs `suppress_tokens=None` et `begin_suppress_tokens=None` sont
remplacés par les valeurs du modèle (`begin_suppress_tokens=[220, 50257]`).
Rust ne possède pas ces processeurs. Le champ `decoding` des deux headers
initiaux mentionne à tort « raw greedy without suppression » : cette note
corrige l'interprétation du protocole ; les headers et sorties déjà capturés
sont conservés, sans recapture ni adaptation après les écarts. Le générateur
décrit désormais ce repli dans les prochains headers, avec les mêmes appels.
Les textes des segments sont le décodage brut des tokens, marqueurs temporels
inclus. Texte et tokens non temporels sont comparés exactement en Rust.
La tolérance temporelle cible est 0,02 s, un pas Whisper. **La parité temporelle
n'est pas vérifiable actuellement** : Rust force `notimestamps` et son API ne
renvoie pas d'horodatages. Le test actif valide la structure et la grille de la
référence, sans prétendre à une comparaison temporelle avec le moteur.

## Capture logits GPU — à lancer explicitement

Le générateur n'a pas été exécuté pendant l'ajout de ces tests. `pending.json`
enregistre cette absence ; aucun résultat synthétique n'est présenté comme une
sortie mlx-lm. Le test de lecture actif valide chaque capture dès sa présence.

```sh
for model in qwen38 qwen36_moe gemma3 gemma4; do
  SARAGOSSA_GOLDEN_CAPTURE=1 "$REF_PYTHON" -I tools/gpulock.py run \
    --owner goldens-reference --prio 10 -- \
    taskpolicy -b "$REF_PYTHON" -I tools/goldens/logits.py --model "$model" || break
done
```

Chaque fichier contient six prompts (FR, EN, arithmétique, code, traduction,
chat avec template), 16 pas greedy même après EOS, IDs de prompt et sortie,
top-20 IDs et log-probabilités normalisées sur le vocabulaire entier. Les tests
Rust alimentent le préfixe de référence après chaque pas pour isoler les écarts.
Ils exigent l'argmax exact et affichent sa marge log-p top1/top2 ; une marge
≤ 0,01 est signalée `near_tie`, sans masquer l'échec. La KL est calculée sur
21 catégories (top-20 référence + masse restante), seuil 0,02 nat ; le maximum
Δp sur cette partition doit être ≤ 0,02. Ce n'est pas la KL complète du vocabulaire.
Ces seuils sont provisoires, non validés par une exécution GPU.

```sh
# Lectures et vérifications sans poids.
taskpolicy -b cargo test --lib external_
taskpolicy -b cargo test --bin saragossa external_serve_template_fixtures_parse
# Comparaisons explicites ; définir les variables locales au préalable.
taskpolicy -b cargo test --lib external_chat_matches_transformers -- --ignored --nocapture
taskpolicy -b cargo test --bin saragossa external_serve_templates_match_transformers -- --ignored --nocapture
taskpolicy -b cargo test --release --lib external_embedder_matches_sentence_transformers -- --ignored --nocapture
taskpolicy -b cargo test --release --lib external_whisper_ -- --ignored --nocapture --test-threads=1
# GPU : sous le même verrou coopératif.
"$REF_PYTHON" -I tools/gpulock.py run --owner goldens-reference --prio 10 -- \
  taskpolicy -b cargo test --lib external_logits_ -- --ignored --nocapture --test-threads=1
```

Les oracles de parité restent stricts : un écart n'autorise pas à régénérer une
référence depuis Rust ni à modifier le moteur. Les tests de lecture demeurent
actifs, indépendamment de la disponibilité des poids.

## Résultats de capture et écarts

Capture CPU du 2026-11-11 : templates **55 651 octets** (28 cas, dont deux
rejets Gemma3, plus constat d'absence TTS), embedder **33 175 octets**, Whisper
**4 322 octets**, headers compris. Le manifeste d'attente logits fait 182 octets.
Total : **93 330 octets**. Aucune capture GPU n'a été exécutée.

Le test direct des sources Rust constate **21 écarts sur 28 cas** :

| Modèle | Écarts | Détail |
|---|---:|---|
| Qwen3.8 | 8/8 | L'historique assistant de référence inclut un bloc think vide ; thinking=true ajoute aussi un préambule système « Reasoning effort is set to xhigh » et ouvre `<think>`. Le rendu Rust direct omet ces éléments. Les quatre cas outils divergent aussi sur les déclarations et appels. |
| Qwen3.6 MoE | 5/8 | La génération thinking=true doit ouvrir `<think>\n`. Les quatre cas outils omettent côté Rust les déclarations et appels au format `<function=…>/<parameter=…>`. Les trois autres cas multi-tours correspondent. |
| Gemma3 | 2/4 | Transformers rejette la conversation avec rôle tool (« Conversation roles must alternate… »), que Rust rend sans validation. Les deux conversations multi-tours correspondent, BOS inclus. |
| Gemma4 | 6/8 | Thinking=true doit insérer `<|think|>` dans le système (deux cas multi-tours). Les quatre cas outils divergent sur les blocs declaration/call/response et l'enchaînement des tours. Les deux cas multi-tours thinking=false correspondent. |

Ces écarts sont des résultats conservés : aucun rendu du moteur ni texte de
référence n'a été adapté pour faire passer l'oracle. Les assertions indiquent
le premier octet divergent et les deux textes complets.

Le chemin réel `serve::LoadedModel::prompt_encoding` confirme **5 écarts sur
8 cas exposés par serve** : les quatre conversations outils divergent ; le
multi-tour Qwen3.6 thinking=false diverge aussi, car serve ajoute un bloc
think vide à l'historique assistant alors que ce tokenizer ne l'ajoute pas.
Les multi-tours Qwen3.8, Gemma3 et Gemma4 correspondent après encodage, BOS
compris. Le test de lecture des fixtures du binaire passe.

Embedder : **19/20 cas passent** (cosinus arrondi à 1,000000000, max |diff|
≤ 1,9 × 10⁻⁷). Le cas 8, requête vide avec préfixe `query: `, échoue :
cosinus **0,9545245355935004**, max |diff| **0,056011915**. Le vecteur de référence
est conservé, ainsi que les seuils initiaux 0,9999 / 0,001.
Le tokenizer chargé directement depuis `tokenizer.json` distingue `query: `
(IDs `[0, 41, 1294, 12, 6, 2]`) de `query:` (`[0, 41, 1294, 12, 2]`) :
l'espace final produit le token `▁` 6. En revanche, `AutoTokenizer`
(`XLMRobertaTokenizer`, transformers 5.22.2), utilisé par sentence-transformers,
renvoie `[0, 41, 1294, 12, 2]` pour les deux chaînes. Cette divergence de
tokenisation est vérifiée sur les fichiers locaux ; les 19 cas non vides
isolent une parité numérique bien plus serrée que le seuil initial.

Whisper-small : texte et **34 tokens identiques** entre le décodeur Rust CPU
et transformers. Texte obtenu par les deux implémentations :
« Elle voyait l'avenir non pas avec terreur, suit était un sentiment vise,
mais avec un dégoût bien rare à son âge. » La formulation est conservée telle
que transcrite ; il ne s'agit pas d'une correction humaine de l'audio.
Whisper-tiny : texte et **35 tokens identiques** également. Texte obtenu :
« Elle voyait l'avenir non pas avec terreur, suité t'es un sentiment visse,
mais avec un découpe bien rare à son âge. » Les deux oracles CPU passent
(2/2) ; aucun écart texte/tokens n'est observé sur cet extrait. Les horodatages
restent une référence externe sans équivalent retourné par l'API Rust.

Vérifications effectuées avec `taskpolicy -b` : `cargo fmt --all -- --check`,
`cargo clippy --all-targets --all-features -- -D warnings` et `cargo build`
passent. Les cinq tests actifs de lecture/métriques de la bibliothèque et
celui du chemin serve passent. Les oracles ignorés restent volontairement
stricts et échouent sur les écarts de parité décrits ci-dessus.
La suite `cargo test --lib` a aussi signalé un échec du test existant
`metal_backend::sparse::tests::missing_or_late_proof_never_recycles`, hors des
fichiers modifiés. Son rejeu isolé passe (1/1) ; l'échec collectif ne doit pas
être présenté comme une validation complète de la suite.
