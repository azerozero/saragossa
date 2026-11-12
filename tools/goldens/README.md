# Références externes

Ces scripts n'utilisent que des modèles locaux, en lecture seule. Ils forcent
le mode hors ligne et exigent `SARAGOSSA_GOLDEN_CAPTURE=1`. Installer les versions
épinglées dans `common.py` dans un environnement **hors dépôt**, puis définir
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
Les comparaisons dédiées échouent si une variable est absente, vide ou si le
répertoire est introuvable, même sans `SARAGOSSA_REQUIRE_REAL_MODELS=1`. Les
commandes dédiées imposent néanmoins ce garde-fou, commun aux tests réels.
Les comparaisons avec assets locaux sont `#[ignore]`. Le suivi direct des
templates reste actif et ne lit aucun modèle.

## Versions de capture

Avant tout import de moteur ou chargement local, chaque script vérifie ses
versions avec `importlib.metadata` et refuse toute version différente ou absente :

| Scripts | Distributions épinglées |
|---|---|
| Tous | `transformers==5.22.2`, `tokenizers==0.23.3`, `jinja2==3.1.6` |
| Logits | `mlx==0.32.4`, `mlx-lm==0.32.2` |
| Embedder et Whisper | `torch==2.15.0` |
| Embedder | `sentence-transformers==6.3.0` |

Ces versions sont enregistrées dans les prochains headers. Les captures déjà
produites restent intactes : leurs headers ne consignent pas tokenizers/Jinja2.
Les versions 0.23.3/3.1.6 ont été relevées dans les métadonnées de l'environnement
de capture existant ; elles ne constituent pas une preuve rétroactive de lancement.
Aucun script n'installe de paquet. Vérification stdlib sans modèle :
`python3 -I -B tools/goldens/test_common.py`.

## Capture CPU

```sh
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I -B tools/goldens/chat_templates.py
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I -B tools/goldens/embedder.py
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I -B tools/goldens/whisper.py --model whisper_tiny
SARAGOSSA_GOLDEN_CAPTURE=1 taskpolicy -b "$REF_PYTHON" -I -B tools/goldens/whisper.py --model whisper_small
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
Le chemin comparé appelle l’encodeur puis `generate_greedy`, reconstruit le
préfixe langue/tâche dans le test et limite la génération à 128 tokens.
**L’API publique `WhisperModel::transcribe` n’est pas couverte** : une régression
de sa construction du préfixe peut échapper à cet oracle.

### Provenance de l’audio Whisper

Le PCM historique vient du fichier embarqué `assets/voices/reti-fr.wav`, copie de
[`12080_11650_000047-0001_enhanced.wav`](https://huggingface.co/kyutai/tts-voices/blob/main/cml-tts/fr/12080_11650_000047-0001_enhanced.wav).
Attribution : **CML-TTS / Kyutai tts-voices**, locuteur identifié par `12080_11650`.
La [notice Kyutai](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/README.md#cml-ttsfr)
indique [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) pour ce sous-ensemble
et un débruitage « enhanced » par ai-coustics. L’identité civile du lecteur
n’est pas fournie dans cette notice. Voir aussi `assets/voices/README.md`.

Le SHA256 du WAV local est identique au SHA256 publié par Kyutai :
`266b428653f25d78dd5dc06d89afdaa26cb377ecd761c27b1e4e9da6639e46d5`.
Transformation vérifiée : WAV mono PCM i16 à 32 kHz, division par 32768 en f32,
puis conservation des échantillons 0, 2, 4… sans filtre. Les 138 880 échantillons
obtenus (8,68 s) sont identiques octet pour octet au PCM golden, SHA256
`1722dd7a13d6814fcd919d628c567f440929590393c49333d1b30842a7e0566e`.
Le test actif `external_whisper_audio_matches_attributed_voice` prouve cette
filiation sans modèle. Aucun audio, header capturé ni texte de référence n’a
été remplacé ou modifié.

## Inventaire logits GPU

Les quatre captures mlx-lm ont été fournies le 2026-11-11. L'inventaire
`tests/fixtures/goldens/logits/inventory.json` est exhaustif ; le test actif
exige chaque couple `.bin`/`.json` marqué `captured`, son SHA256 et son schéma.
Supprimer un fichier capturé est un échec. Les deux contrôles float32 marqués
`pending_capture` peuvent être absents jusqu'à leur première capture ; dès
qu'un des fichiers existe, le couple complet, son SHA256, son schéma et la
provenance `float32_control: true` sont vérifiés. Après capture, passer leur
statut à `captured` pour rendre leur présence obligatoire.

| Modèle | État | Octets du couple bin/json |
|---|---|---:|
| qwen38 | capturé | 51 989 |
| qwen36_moe | capturé | 52 581 |
| gemma3 | capturé | 52 885 |
| gemma4 | capturé | 53 490 |
| qwen36_moe_f32 | capture opérateur en attente | — |
| gemma4_f32 | capture opérateur en attente | — |

Total logits : **210 945 octets** hors inventaire. Les sorties n'ont pas été
recapturées ni modifiées pendant cette correction. La comparaison Rust avec
les nouvelles références float32 reste à lancer par l'opérateur.

Commande de capture pour une campagne ultérieure explicitement autorisée
(elle écrit des références ; ne pas la lancer pour valider les captures actuelles) :

```sh
(
  for model in qwen38 qwen36_moe gemma3 gemma4; do
    SARAGOSSA_GOLDEN_CAPTURE=1 "$REF_PYTHON" -I -B tools/gpulock.py run \
      --owner goldens-reference --prio 10 -- \
      taskpolicy -b "$REF_PYTHON" -I -B tools/goldens/logits.py --model "$model" || exit "$?"
  done
)
```

### Contrôles float32 de Gemma 4 et Qwen3.6 MoE

Les écarts observés face aux captures bf16 sont de **7/96 mismatches greedy
pour Gemma 4** et **2/96 pour Qwen3.6 MoE**. Le diagnostic couche par couche
fourni sur `g4rope` rattache ces écarts à la précision bf16 de mlx-lm et aux
quasi-égalités du routage MoE ; le CPU suit la référence mlx-lm float32.
Ce constat motive un oracle strict distinct, sans élargir les seuils.

`--float32` appelle `model.set_dtype(mx.float32)` puis évalue les paramètres :
poids flottants et échelles passent en f32, les poids quantifiés packés sont
conservés. La provenance contient `float32_control: true`. Les fichiers
`logits/<model>_f32.bin` et `.json` sont distincts des captures bf16 ; sans
l'option, le protocole et les noms de capture restent inchangés.

Capture à lancer par l'opérateur, avec les variables locales du tableau :

```sh
(
  for model in gemma4 qwen36_moe; do
    SARAGOSSA_GOLDEN_CAPTURE=1 "$REF_PYTHON" -I -B tools/gpulock.py run \
      --owner goldens-reference --prio 10 -- \
      taskpolicy -b "$REF_PYTHON" -I -B tools/goldens/logits.py --model "$model" --float32 || exit "$?"
  done
)
```

Chaque fichier contient six prompts (FR, EN, arithmétique, code, traduction,
chat avec template), 16 pas greedy même après EOS, IDs de prompt et sortie,
top-20 IDs et log-probabilités normalisées sur le vocabulaire entier. Les tests
Rust alimentent le préfixe de référence après chaque pas pour isoler les écarts.
Les oracles stricts (qwen38, gemma3 et les deux contrôles float32) exigent
l'argmax exact et affichent sa marge log-p top1/top2 ; une marge
≤ 0,01 est signalée `near_tie`, sans masquer l'échec. La KL est calculée sur
21 catégories (top-20 référence + masse restante), seuil 0,02 nat ; le maximum
Δp sur cette partition doit être ≤ 0,02. Ce n'est pas la KL complète du vocabulaire.
Ces seuils sont provisoires, non validés par une exécution GPU.

Les comparaisons bf16 de Gemma 4 et Qwen3.6 MoE deviennent des rapports
d'écarts connus : chaque pas affiche `mismatch`, `near_tie`, `KL21` et
`delta_p`, puis un bilan compte les mismatches et near-ties et donne les
maxima KL/Δp. Elles échouent si les mismatches dépassent respectivement
7/96 et 2/96. KL et Δp restent mesurées aux mêmes seuils, mais ne bloquent
pas ces deux rapports. Les erreurs de lecture, de provenance ou d'exécution
restent des échecs. Une amélioration du nombre de mismatches est acceptée.

## Exécuter les comparaisons

Définir les variables du tableau vers les snapshots locaux correspondants.
Le filtre logits ci-dessous sélectionne les six comparaisons ignorées
sur macOS avec la fonctionnalité `metal` explicite. Le verrou est commun aux
captures et aux comparaisons. `REF_PYTHON` désigne l'interpréteur existant ; le
script de verrou ne dépend que de la stdlib.

```sh
# Tests CPU sans poids ni GPU, même sur un hôte disposant de Metal.
taskpolicy -b cargo test --offline --locked --no-default-features --lib
taskpolicy -b cargo test --offline --locked --bin saragossa external_serve_template_fixtures_parse
taskpolicy -b cargo test --offline --locked --bin saragossa external_serve_rejection_requires_role_category
# Suivi des écarts avec tokenizers locaux / embedder CPU :
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --bin saragossa external_serve_templates_known_differences_are_stable -- --ignored --nocapture
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --release --lib external_embedder_known_differences_are_stable -- --ignored --nocapture
# Oracles stricts séparés : les écarts connus doivent toujours les faire échouer.
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --lib external_chat_strict_parity_matches_transformers -- --ignored --nocapture
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --bin saragossa external_serve_templates_strict_parity_matches_transformers -- --ignored --nocapture
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --release --lib external_embedder_strict_parity_matches_sentence_transformers -- --ignored --nocapture
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --release --lib external_whisper_ -- --ignored --nocapture --test-threads=1
# Comparaison Rust des logits : quatre oracles stricts et deux rapports bf16.
SARAGOSSA_REQUIRE_REAL_MODELS=1 "$REF_PYTHON" -I -B tools/gpulock.py run \
  --owner goldens-reference --prio 10 -- \
  taskpolicy -b cargo test --offline --locked --features metal --lib golden::logits::external_logits_ \
  -- --ignored --nocapture --test-threads=1
```

Pour comparer uniquement les nouveaux contrôles stricts float32 après capture :

```sh
(
  for model in gemma4 qwen36_moe; do
    SARAGOSSA_REQUIRE_REAL_MODELS=1 "$REF_PYTHON" -I -B tools/gpulock.py run \
      --owner goldens-reference --prio 10 -- \
      taskpolicy -b cargo test --offline --locked --features metal --lib \
      "golden::logits::external_logits_${model}_f32" \
      -- --exact --ignored --nocapture --test-threads=1 || exit "$?"
  done
)
```

Pour les rapports bf16, reprendre cette boucle avec
`"golden::logits::external_logits_${model}"` (sans `_f32`, conserver `--exact`).
Les comparaisons dédiées exigent les captures, même si l'inventaire indique
encore `pending_capture` ; seul le test de lecture autorise leur absence.

Aucun écart n'autorise à régénérer une référence depuis Rust ni à modifier les
seuils. Les tests de lecture demeurent actifs sans assets locaux.

## Suivi des ecarts

Les identifiants exacts sont figés dans `KNOWN_DIFFERENCES` de
`src/golden/chat.rs` (21/28) et `src/serve/state/reference_goldens.rs` (5/8).
Le test actif `external_chat_known_differences_are_stable` exécute chaque rendu,
y compris les deux cas que Gemma3 rejette dans la référence. L'API directe
actuelle retourne toujours une chaîne ; ce rendu est comparé à un résultat
« rendu » ou « rejet », sans présumer un échec avant l'appel Rust.

Le test dédié serve exige les mêmes cinq identifiants divergents, et l'embedder
exige uniquement `8/query` (requête vide). Un écart qui apparaît **ou disparaît**
fait échouer le suivi. Les erreurs techniques de serve ne sont pas des écarts
acceptables : un rejet de rôle n'est équivalent qu'avec la catégorie HTTP de
requête invalide et le message précis d'alternance des rôles. Un autre rejet
échoue immédiatement, même sur un cas déjà divergent.

Les oracles `strict_parity` séparés exigent zéro écart et restent ignorés avec
une raison explicite. Les seuils embedder restent 0,9999 / 0,001. Le suivi des
identifiants ne mesure pas une aggravation à l'intérieur d'un cas déjà divergent ;
il ne remplace donc pas l'oracle strict. Les corrections moteur sont séparées
de cette PR. L'adaptateur Anthropic et les réponses HTTP de tool calling ne
sont pas couverts par ces goldens.

## Résultats de capture et écarts historiques

Capture CPU du 2026-11-11 : templates **55 651 octets** (28 cas, dont deux
rejets Gemma3, plus constat d'absence TTS), embedder **33 175 octets**, Whisper
**4 322 octets**, headers compris. Ces résultats CPU historiques sont conservés ;
l’inventaire logits ci-dessus remplace l’ancien statut global d’attente.

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
fichiers modifiés. Son rejeu isolé passe (1/1) ; le passage debug n'est donc
pas validé.
Ce passage debug a été interrompu après près d'une heure d'exécution des
tests : les trois oracles CPU d'attention sink préexistants calculaient
encore leurs références. Les comparaisons des nouvelles fixtures étaient
toutes terminées.

La suite complète `cargo test --release --lib`, exécutée ensuite sans
changement de test ni de seuil, passe : **759 réussis, 184 ignorés, 0 échec**.
Les 184 tests ignorés incluent les oracles réels lancés séparément ci-dessus.
Les trois oracles d'attention sink et le test sparse passent dans ce profil.


## Validation du suivi (sans modèles, 2026-11-11)

Exécutés sous `taskpolicy -b`, en mode Cargo hors ligne : fmt, clippy
`--all-targets --all-features -- -D warnings`, build et
`cargo test --no-default-features --lib` : **374 réussis, 37 ignorés**.
La désactivation des fonctionnalités par défaut exclut Metal ; aucun GPU ni
poids réel n’a été utilisé. Le test CPU de quantification utilisait trois
accesseurs réservés à Metal : il consulte désormais les mêmes données internes
sans ces accesseurs, avec les mêmes assertions. Le profil sans fonctionnalités
émet encore des avertissements préexistants ; le clippy toutes fonctionnalités
passe avec les avertissements traités comme erreurs.

Les deux tests serve sans assets passent (lecture et catégorie de rejet), ainsi
que les quatre tests stdlib de versions Python. Les sept versions épinglées ont
été vérifiées dans l’environnement de capture existant, sans import de moteur.
Des lancements négatifs dédiés confirment l’échec sans assets pour serve,
embedder et Whisper ; l’oracle strict de chat échoue sur les 21 écarts attendus.
Les comparaisons numériques avec modèles restent à exécuter par l’opérateur.
Les 65 fichiers de captures/références présents au début de cette correction
sont identiques octet pour octet ; seul le manifeste d’attente est remplacé.
