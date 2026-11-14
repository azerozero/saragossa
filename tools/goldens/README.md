# Références externes

Les scripts de capture de références n'utilisent que des modèles locaux, en lecture seule. Ils forcent
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
Le helper unique `local()` saute la comparaison si une variable est absente,
vide ou si le répertoire est introuvable. Avec `SARAGOSSA_REQUIRE_REAL_MODELS=1`,
ces préconditions manquantes sont des erreurs explicites. Les commandes dédiées
imposent ce garde-fou, commun aux tests réels.
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
Supprimer un fichier capturé est un échec. Les deux contrôles float32 sont également marqués `captured` : leur présence,
le couple complet, son SHA256, son schéma et la provenance
`float32_control: true` sont obligatoires. Le lecteur conserve la prise en charge
de `pending_capture` pour un inventaire de campagne avant sa première capture.

| Modèle | État | Octets du couple bin/json |
|---|---|---:|
| qwen38 | capturé | 51 989 |
| qwen36_moe | capturé | 52 581 |
| gemma3 | capturé | 52 885 |
| gemma4 | capturé | 53 490 |
| qwen36_moe_f32 | capturé | 52 758 |
| gemma4_f32 | capturé | 53 841 |

Total logits : **317 544 octets** hors inventaire. Les sorties n'ont pas été
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
Δp sur cette partition doit être ≤ 0,05. Ce n'est pas la KL complète du vocabulaire.
Les seuils font foi dans `src/golden/logits.rs` (`MAX_KL21`, `MAX_DELTA_P`) ;
le champ `tolerance` des captures antérieures au 13 nov. 2026 garde l'ancien
texte provisoire (Δp ≤ 0,02).

### Calibration des seuils KL et Δp

Mesure du 13 nov. 2026. Le seuil Δp provisoire de 0,02 n'avait jamais été
validé sur GPU. Le gate est par pas, sans échec toléré sur 96 : seul le
maximum compte. La preuve directe vient des modèles denses conformes :

| comparaison | mismatches | KL21 max | Δp max |
|---|---|---|---|
| saragossa / mlx bf16, qwen38 | 0/96 | 0,0017 | 0,028 |
| saragossa / mlx bf16, gemma3 | 1/96 (near-tie) | 0,0033 | 0,039 |

Ordre de grandeur attendu : avec tous les logits arrondis à ±½ ulp bf16
(0,0625 pour |logit| ∈ [16, 32[), une probabilité bouge d'au plus δ/2 ≈ 0,031,
et la comparaison cumule deux chemins bf16 (saragossa et mlx). Le seuil
`MAX_DELTA_P = 0,05` garde environ 28 % de marge sur le pire mesuré.

Repère indirect : la capture mlx bf16 de Qwen3.6 MoE s'écarte de sa propre
capture float32 avec Δp p50 0,0045, p95 0,037, max 0,22 et KL21 p95 0,0077
(82 pas à préfixe commun, `reference_noise.py qwen36_moe`). Ce modèle est un
MoE à routage bruité. Le script borne par le haut la probabilité f32 des
70 ids (sur 1 640) absents du top-20 f32, ce qui biaise sa KL vers le bas.
Gemma 4 n'entre pas dans la calibration (p95 Δp 0,41, quasi-égalités de
routage). La KL reste à 0,02 : les modèles conformes sont 6 à 12 fois en
dessous. L'argmax exact reste exigé, near-tie compris.

Les comparaisons bf16 de Gemma 4 et Qwen3.6 MoE deviennent des rapports
d'écarts connus : chaque pas affiche `mismatch`, `near_tie`, `KL21` et
`delta_p`, puis un bilan compte les mismatches et near-ties et donne les
maxima KL/Δp. Elles échouent si les mismatches dépassent respectivement
7/96 et 2/96. KL et Δp restent mesurées aux mêmes seuils, mais ne bloquent
pas ces deux rapports. Les erreurs de lecture, de provenance ou d'exécution
restent des échecs. Une amélioration du nombre de mismatches est acceptée.

## KL sur corpus

Protocole inspiré de `llama-perplexity --kl-divergence` et du test par ratio
du guide de précision vLLM d'AWS Neuron. `kld.py` note 2 048 positions :
8 morceaux de 512 tokens de `kld/corpus.txt`, un extrait du domaine public
(Project Gutenberg n°1661, « A Scandal in Bohemia »), dont les 256 dernières
positions de chaque morceau sont notées. La référence est mlx-lm float32 ;
chaque position garde le top-64 et la masse restante. Le même script mesure
mlx-lm bf16 contre cette référence : c'est le bruit propre de la précision
bf16, stocké dans `kld/<modèle>_meta.json`.

L'oracle Rust (`golden::kld`, prefill de 256 tokens puis decode token par
token) échoue si la KL moyenne ou le Δp p99 de saragossa dépasse 1,5 fois
celui de mlx-lm bf16, ou si son taux de même top-1 est inférieur à celui de
mlx-lm bf16. Ce critère ne demande aucun seuil absolu choisi à la main.

Gemma 4 : la référence float32 applique l'échelle d'embedding canonique
(√hidden arrondi en bf16, comme transformers et mlx-lm Gemma 3 ; mlx-lm
Gemma 4 garde sinon la valeur exacte en float32). Même ainsi, mlx-lm bf16 et
float32 ne s'accordent que sur 80 % des top-1, contre 98 à 99 % ailleurs :
ce modèle est très sensible à la précision et les ratios y bornent peu.
Réarrondir en bf16 un seul composant de la référence float32 (entrée ou
poids du routeur, attention, MLP, experts ou résiduel) suffit à ramener
l'accord à 86-90 %.

**Gemma 4 sur texte brut (13 nov. 2026).** Sur le corpus, mlx-lm donne une
perplexité d'au moins 533 (Qwen3.8 : 2), et les greedy bruts des goldens
bouclent, alors que le chat répond juste. Arbitré contre les poids Google bf16
dans transformers 5.22.2 : perplexité d'environ 10⁵ sur le même corpus, sur
CPU comme sur MPS, et « The capital of France is » → « a bit is a bit is… » ;
le chat répond « Paris ». C'est donc le modèle instruct qui ne prédit pas le
texte brut, pas un bug commun à mlx-lm et saragossa.

**Gemma 4 en régime chat (14 nov. 2026).** L'oracle Gemma 4 n'utilise donc plus
le corpus : ses morceaux sont les 8 prompts de `tools/quant/chat_kl.py` suivis
de la réponse greedy (192 tokens au plus) de la référence float32, dont les 128
derniers tokens sont notés, soit 1 024 positions. mlx-lm bf16 y suit le float32
sur 98,8 % des top-1 (80 % sur le corpus). La perte de quantification contre
les poids Google se mesure à part (tools/quant/README.md, section Gemma 4).

Mesure du 13 nov. 2026 (M5 Max) :

| modèle | même top-1, saragossa | même top-1, mlx bf16 | KL moy. saragossa | KL moy. mlx bf16 | ratio KL | ratio Δp p99 |
|---|---|---|---|---|---|---|
| qwen38 (u4ls) | 99,95 % | 98,88 % | 0,000044 | 0,0016 | 0,03 | 0,18 |
| gemma3 | 100 % | 97,85 % | 0,0000077 | 0,0016 | 0,005 | 0,08 |
| qwen36_moe | 98,19 % | 97,71 % | 0,010 | 0,022 | 0,45 | 0,76 |
| gemma4 (chat, 14 nov.) | 99,61 % | 98,83 % | 0,00015 | 0,0014 | 0,11 | 0,36 |

```sh
SARAGOSSA_GOLDEN_CAPTURE=1 python -I tools/goldens/kld.py --model qwen38
cargo test --release --all-features --lib golden::kld::external_kld_ -- --ignored --test-threads=1 --nocapture
```

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
# Oracles stricts : templates conformes ; écart E5 encore suivi.
SARAGOSSA_REQUIRE_REAL_MODELS=1 taskpolicy -b cargo test --offline --locked --lib external_chat_strict_parity_matches_transformers -- --nocapture
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

Les corrections de main (PR #159) résolvent les **21 anciens écarts sur
28 templates** et les **5 anciens écarts sur 8 cas serve**. Les listes
`KNOWN_DIFFERENCES` de `src/golden/chat.rs` et
`src/serve/state/reference_goldens.rs` sont désormais vides. Les références
restent inchangées : l'ordre des clés JSON est conservé pour les outils et
appels, et les profils Qwen3.8/Qwen3.6 ainsi que le rendu Gemma 4 suivent main.
Les deux rejets Gemma3 sont comparés comme des erreurs de template attendues.
Le suivi direct et l'oracle strict sont actifs, sans poids ni tokenizer local.

Le chemin serve appelle `LoadedModel::prompt_encoding`, vérifie les huit cas
exposés et reste ignoré car il exige les tokenizers locaux. Les erreurs
techniques ne sont pas des écarts acceptables : un rejet de rôle exige la
catégorie HTTP et le message précis d'alternance des rôles. Sa parité est
attendue selon main ; cette fusion sans modèles ne la remesure pas.

L'embedder exige toujours uniquement `8/query` (requête vide). Un écart qui
apparaît **ou disparaît** fait échouer son suivi. Son oracle strict reste
ignoré et exige zéro écart, avec les seuils inchangés 0,9999 / 0,001.
L'adaptateur Anthropic et les réponses HTTP de tool calling ne sont pas
couverts par ces goldens.

## Résultats de capture et écarts historiques

Capture CPU du 2026-11-11 : templates **55 651 octets** (28 cas, dont deux
rejets Gemma3, plus constat d'absence TTS), embedder **33 175 octets**, Whisper
**4 322 octets**, headers compris. Ces résultats CPU historiques sont conservés ;
l’inventaire logits ci-dessus remplace l’ancien statut global d’attente.

Avant les corrections de main, le test direct constatait **21 écarts sur 28 cas** :

| Modèle | Écarts | Détail |
|---|---:|---|
| Qwen3.8 | 8/8 | L'historique assistant de référence inclut un bloc think vide ; thinking=true ajoute aussi un préambule système « Reasoning effort is set to xhigh » et ouvre `<think>`. Le rendu Rust direct omet ces éléments. Les quatre cas outils divergent aussi sur les déclarations et appels. |
| Qwen3.6 MoE | 5/8 | La génération thinking=true doit ouvrir `<think>\n`. Les quatre cas outils omettent côté Rust les déclarations et appels au format `<function=…>/<parameter=…>`. Les trois autres cas multi-tours correspondent. |
| Gemma3 | 2/4 | Transformers rejette la conversation avec rôle tool (« Conversation roles must alternate… »), que Rust rend sans validation. Les deux conversations multi-tours correspondent, BOS inclus. |
| Gemma4 | 6/8 | Thinking=true doit insérer `<|think|>` dans le système (deux cas multi-tours). Les quatre cas outils divergent sur les blocs declaration/call/response et l'enchaînement des tours. Les deux cas multi-tours thinking=false correspondent. |

Ce tableau conserve le diagnostic historique ; ces écarts de rendu sont
résolus par main. Aucun texte de référence n'a été modifié. Les assertions
indiquent le premier octet divergent et les deux textes complets.

Le chemin réel `serve::LoadedModel::prompt_encoding` constatait **5 écarts sur
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

Vérifications historiques avant fusion, effectuées avec `taskpolicy -b` : `cargo fmt --all -- --check`,
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
À cette date, des lancements négatifs dédiés confirmaient l’échec sans assets pour serve,
embedder et Whisper ; l’oracle strict de chat échouait sur les 21 écarts alors attendus.
Ces constats sont remplacés par le suivi actuel ci-dessus.
Les comparaisons numériques avec modèles restent à exécuter par l’opérateur.
Les 65 fichiers de captures/références présents au début de cette correction
sont identiques octet pour octet ; seul le manifeste d’attente est remplacé.

## Diagnostic couche par couche — Gemma 4 MoE

Ces outils capturent le **pas 0** : un prefill sur les IDs exacts du prompt,
les activations de toutes ses positions, puis les logits de sa dernière position
après softcap (avant log-softmax). Aucun tokenizer n'est appelé pour construire
le prompt. Ils ne génèrent pas de continuation.

Le JSON du golden est un index de provenance ; les `prompt_ids` sont dans le
payload UTF-8 `.bin` adjacent. L'extraction vérifie son SHA256. Le prompt 0 du
golden Gemma 4 contient `[53406, 236764, 2185, 520, 236789, 101652]`.

### Exécution locale

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

### Points de mesure et format

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

### Lecture des résultats

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

### Vérification sans GPU ni modèle réel

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
