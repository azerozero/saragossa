# Conformité de `saragossa serve`

Cette suite lance un serveur local et exerce ses interfaces avec les SDK officiels
OpenAI et Anthropic. Les contrats portent sur le JSON reçu avant toute coercition
du SDK, le décodage SDK, le contenu minimal et les flux SSE. Aucun téléchargement,
aucune dépendance Rust supplémentaire, aucune écriture dans les modèles.

Les commandes d'inférence ci-dessous sont à exécuter sous le verrou GPU de
l'opérateur. La suite ne prend pas elle-même ce verrou. `--list` et les tests des
validateurs n'exécutent ni serveur ni modèle, et n'utilisent pas le GPU.

## Fichiers

```text
tools/e2e/
├── .gitignore           # rapports locaux exclus du dépôt
├── README.md
├── contracts.py         # schémas, usage, outils et machines d'état SSE
├── launcher.py          # chemins locaux, flags et cycle de vie du serveur
├── matrix.py            # cas SDK et capacités explicites
├── run.py               # CLI, sélection et rapports JSON/Markdown
├── test_contracts.py    # unittest, fixtures en mémoire, aucun serveur
└── transport.py         # SDK officiels et sondes HTTP/SSE stdlib
```

Python 3.10 ou plus récent. Dans un environnement Python choisi par l'opérateur :

```sh
pip install openai==3.49.0 anthropic==2.8.0
```

Le runner ne lance jamais `pip`. Il refuse une autre version des SDK, désactive
leurs retries et les proxies hérités pour les connexions locales. `unittest`
fournit le runner minimal stdlib ; pytest n'est pas requis.

Le serveur exige `Authorization: Bearer` pour les deux API. Le client Anthropic
est donc construit avec `auth_token`, option officielle du SDK, plutôt qu'avec
`api_key` qui émettrait uniquement `x-api-key`. Le cas `auth_x_api_key_only`
vérifie explicitement le refus 401 de cette dernière méthode et le signale comme
non supporté attendu.

## Lancement par modèle

Depuis la racine du dépôt, définir `SARAGOSSA_BIN` avec le chemin du binaire local
déjà construit et les variables ci-dessous avec des chemins locaux existants.
Aucune variable ne doit contenir un identifiant de dépôt distant. Les LLM sont
enregistrés par un alias explicite `ID=DIR`, ce qui évite le résolveur Hugging Face.

| Cible | Modèle | Variable de chemin |
|---|---|---|
| `qwen38` | Qwen3.8-27B u4ls | `E2E_QWEN38_MODEL` |
| `qwen36_moe` | Qwen3.6-35B-A3B MoE | `E2E_QWEN36_MOE_MODEL` |
| `gemma3` | Gemma 3 4B | `E2E_GEMMA3_MODEL` |
| `gemma4` | Gemma 4 26B | `E2E_GEMMA4_MODEL` |
| `e5` | Embedder E5, dimension 384 | `E2E_E5_MODEL` |
| `whisper` | Whisper | `E2E_WHISPER_MODEL` |
| `qwen3_tts` | Qwen3-TTS | `E2E_QWEN3_TTS_MODEL` |
| `audio` | Qwen3-TTS + Whisper, aller-retour complet | les deux variables audio ci-dessus |

Les noms de cibles sont des profils de campagne ; ils ne prouvent ni la taille
ni la quantification des poids fournis. L'opérateur sélectionne les checkpoints.
Seul `config.json` du LLM est lu par le runner pour détecter `model_type=flint`,
condition actuelle du support des outils OpenAI.

```sh
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target qwen38 --report-dir tools/e2e/reports/qwen38
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target qwen36_moe --report-dir tools/e2e/reports/qwen36_moe
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target gemma3 --report-dir tools/e2e/reports/gemma3
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target gemma4 --report-dir tools/e2e/reports/gemma4
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target e5 --report-dir tools/e2e/reports/e5
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target whisper --report-dir tools/e2e/reports/whisper
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target qwen3_tts --report-dir tools/e2e/reports/qwen3_tts
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target audio --report-dir tools/e2e/reports/audio
```

Le profil Whisper seul exige aussi `E2E_AUDIO_WAV` (fichier WAV local) et
`E2E_AUDIO_TEXT` (transcription de référence). Le profil `audio` génère lui-même le
WAV à partir d'une phrase française fixe puis le transmet à Whisper, sans écrire
l'audio sur disque. Il vérifie les réponses JSON et texte ainsi que la similarité
des mots : ratio `SequenceMatcher` sur les mots Unicode minuscules, seuil 0,60,
configurable avec `--word-similarity`. Ce score est enregistré dans le JSON.

Un processus neuf est lancé sur un port libre de `127.0.0.1`, avec une socket Unix
temporaire et une clé API aléatoire de test. Le runner attend `/health`, confirme
l'accès authentifié à `/v1/models`, exécute les cas séquentiellement et arrête tout
le groupe de processus par TERM puis KILL après dix secondes si nécessaire.
Le nettoyage s'applique aussi aux erreurs et aux interruptions Ctrl-C/SIGTERM.
Le port est réservé puis libéré avant lancement ; une collision fait échouer le
lancement plutôt que de choisir un autre serveur. Le contrôle authentifié avec
la clé aléatoire évite de tester un serveur étranger.

Délais par défaut : 60 s pour le démarrage, 600 s par opération réseau, ajustables
avec `--startup-timeout` et `--request-timeout`. Le chargement est paresseux :
`/health` ne garantit pas que les poids sont déjà chargés. Le premier appel peut
donc utiliser tout le délai de requête. Le budget par cas est 256 tokens, sauf le
test `max_tokens=1`, et le plafond du serveur est 1024.

## AR, MTP et DFlash

Le mode par défaut est AR. Les variables héritées `SARAGOSSA_*` et `RETI_*` sont
retirées de l'environnement enfant pour éviter une activation implicite du MTP,
d'un drafter, d'un mode acceptance ou d'un preset expérimental. Les chemins des
modèles restent inchangés. Les variables offline Hugging Face sont activées.

| Mode | Flags transmis au serveur | Prérequis |
|---|---|---|
| AR | `SARAGOSSA_RUST_MTP_DECODE=0`, aucun drafter | tous profils |
| MTP | `SARAGOSSA_RUST_MTP_DECODE=1`, `SARAGOSSA_RUST_MTP_MAX_DRAFT=2`, `SARAGOSSA_RUST_MTP_SIDECAR` | tête locale explicite, backend Metal |
| DFlash | flags MTP + `SARAGOSSA_RUST_DFLASH_DRAFT_DIR` | tête MTP, drafter compatible, build `metal,devtools` |

Définir les variables correspondant au profil :

- `E2E_QWEN38_MTP_SIDECAR`, `E2E_QWEN38_DFLASH_DIR` ;
- `E2E_QWEN36_MOE_MTP_SIDECAR`, `E2E_QWEN36_MOE_DFLASH_DIR`.

Les sidecars sont des fichiers locaux, les drafters des répertoires locaux. Le
sidecar explicite est requis pour rendre les campagnes reproductibles ; il est
également une condition du MTP MoE dans `src/serve/mtp.rs`.

```sh
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target qwen38 --modes ar,mtp,dflash --report-dir tools/e2e/reports/qwen38-modes
python -I tools/e2e/run.py --binary "$SARAGOSSA_BIN" --target qwen36_moe --modes ar,mtp,dflash --report-dir tools/e2e/reports/qwen36-modes
```

La matrice complète tourne en AR. MTP et DFlash rejouent le sous-ensemble chat :
simple, stream, stream avec usage OpenAI, max_tokens, stop/stop_sequences et
seed + température zéro OpenAI. Les requêtes sont non contraintes, à température
zéro. Les autres profils portent un statut de mode non supporté attendu.
Un sidecar/drafter manquant pour un mode explicitement demandé est un **échec de
configuration**, pas un skip.

`mode_verified` contrôle les lignes d'activation MTP/DFlash et le `decode_path`
des complétions : AR accepte `ar`/`batch`, les modes spéculatifs exigent `mtp`.
DFlash utilise le chemin serveur nommé `mtp` : son activation est prouvée par le
journal de chargement du drafter, sans prétendre mesurer son taux d'acceptation
ni distinguer chaque pas DFlash d'un éventuel repli interne MTP.

## Cas et contrats

Inventaire exact, sans SDK ni poids ni exécution serveur :

```sh
python -I tools/e2e/run.py --target qwen38 --list
python -I tools/e2e/run.py --target audio --list
python -I tools/e2e/run.py --target qwen38 --modes ar,mtp,dflash --list
```

| Endpoint | Options/cas | Vérifications |
|---|---|---|
| `/v1/models` | `list` | object=list, registre exact, IDs uniques, object=model, created entier, owned_by non vide ; registre LLM vide attendu sur profils audio/E5 |
| `/v1/chat/completions` | `simple`, `system`, `multi_turn` | id, object, created, model, choice index=0, rôle, content, finish_reason ; Bonjour, instruction SAPHIR, mémoire ORCHIDEE |
| même endpoint | `max_tokens`, `stop`, `seed_temperature_zero` | usage entier cohérent, budget, finish_reason=length ; stop issu d'une réponse de référence et retiré ; deux réponses et usages identiques à seed=42/température 0 |
| même endpoint | `stream`, `stream_usage` | SSE UTF-8, chunks JSON, id/created stables, delta.role initial, contenu, finish_reason final, `[DONE]` unique/final ; usage final et choices vide si demandé |
| même endpoint | `n_rejected` | n=2 doit échouer HTTP 400 avec erreur explicite ; acceptation silencieuse = ÉCHEC |
| même endpoint | `json_object`, `json_schema` | objet JSON parsable avec greeting ; json_schema HTTP 501 avec erreur explicite |
| même endpoint | `tools_auto`, `tools_none`, `tools_required`, `tools_named` | sélection selon capacité ; required/nommé doivent être refusés sur flint |
| même endpoint | `tools_roundtrip`, `tools_stream` | id unique, type=function, nom, arguments JSON/Paris ; résultat corrélé par tool_call_id, réponse contenant 17 ; assemblage des fragments indexés |
| `/v1/messages` | `simple`, `system`, `multi_turn`, `max_tokens`, `stop_sequences` | type=message, rôle, modèle, blocs, input/output_tokens, stop_reason, stop_sequence corrélée ; mêmes contrôles de contenu |
| même endpoint | `stream` | message_start, blocs start/delta/stop indexés et ordonnés, message_delta, message_stop unique/final ; event SSE = type JSON ; usage final |
| même endpoint | `tools_auto`, `tools_any`, `tools_tool`, `tools_none` | auto via prompt ; choix contraints/none déclarés non supportés |
| même endpoint | `tools_roundtrip`, `tools_stream_order` | tool_use id/name/input, tool_result corrélé puis réponse finale ; fragments input_json_delta reconstitués et ordre des blocs vérifié |
| `/v1/embeddings` | `single_dimensions_l2_float`, `list_dimensions_l2_float` | liste/index/ordre/modèle, 384 nombres finis, norme L2 à 1 ± 0,001 |
| même endpoint | `dimensions_override`, `encoding_base64`, `usage` | limitations explicites du protocole actuel |
| `/v1/audio/speech` | `wav`, `format_mp3/opus/aac/flac/pcm` | MIME audio/wav, WAV mono PCM16 non vide, non silencieux, non tronqué ; autres formats marqués non supportés |
| `/v1/audio/transcriptions` | `roundtrip_json/text` ou `fixture_json/text` | texte minimal, MIME, décodage SDK, similarité de mots ; aller-retour TTS→STT sur profil audio |
| même endpoint | `format_verbose_json/srt/vtt` | limitations explicites |
| chat + messages | `auth_missing`, `auth_invalid`, `model_unknown`, `body_invalid`, `body_type`, `body_excessive` | HTTP 401/404/400, enveloppe et type d'erreur de chaque API, message non vide |
| `/v1/messages` | `auth_x_api_key_only` | x-api-key seul refusé 401 ; enveloppe Anthropic, non supporté attendu |
| embeddings/speech/transcriptions non configurés | `model_not_loaded` | HTTP 400 et message explicite, fichier WAV valide pour la sonde STT |
| processus serveur | `mode_verified`, `lifecycle`, `interrupted` | preuve du mode demandé, erreurs de lancement/arrêt, interruption |

Pour un stream, une requête stdlib contrôle les trames brutes, puis une seconde
requête identique contrôle l'itérateur officiel du SDK. Cela permet de vérifier
`[DONE]`, masqué par le SDK OpenAI. Les deux réponses sont validées séparément.
Les champs additionnels des réponses restent autorisés pour les extensions ;
les champs requis, leurs types exacts et les valeurs protocolaires sont contrôlés.

La sonde de taille annonce 16 MiB + 1 via `Content-Length` et attend le refus avant
envoi du corps. Le contrat actuel de `http.rs` est **400**, pas 413. Le schéma
Anthropic reste exigé sur `/v1/messages`, y compris pour ce refus précoce.

## Capacités et écarts connus

Les décisions suivantes viennent du code local, et doivent être revues lorsqu'une
PR modifie une capacité. Elles ne sont jamais inférées d'une exception réseau.

- [protocol.rs](../../src/serve/protocol.rs) : `json_object` disponible,
  `json_schema` refusé en 501. Le refus est exécuté et vérifié avant attribution
  du statut non supporté attendu.
- [http.rs](../../src/serve/http.rs) : le parser d'outils OpenAI est réservé à
  `model_type=flint`. Les familles Qwen et Gemma ignorent `tools/tool_choice`.
  Leurs cas d'outils OpenAI sont donc explicitement non exécutés/non supportés,
  avec la raison dans chaque ligne. Sur flint, auto/none, aller-retour et streaming
  sont exercés ; required/nommé sont testés comme refus HTTP 400.
- [anthropic.rs](../../src/serve/anthropic.rs) : outils rendus dans le prompt et
  parser XML/JSON disponibles ; `tool_choice` n'est pas désérialisé. Auto vérifie
  la sélection implicite, any/tool/none sont non supportés. L'absence de tool_use
  sur la consigne explicite du test auto est un échec de contenu.
- [embeddings.rs](../../src/serve/embeddings.rs) : sortie float, dimension fixe,
  pas de `usage`, pas de sélection du modèle par le champ `model`. Les contrôles
  de vecteurs restent actifs ; l'absence de la capacité usage a sa propre ligne.
- [audio.rs](../../src/serve/audio.rs) : WAV en synthèse, JSON/text en
  transcription. `response_format` TTS est ignoré. Le PCM existe seulement via
  l'extension locale `stream=true`, hors du contrat de format SDK testé ici.
- `n` n'est actuellement pas désérialisé dans la requête chat. Le cas
  `n_rejected` expose cette lacune par un échec si le serveur accepte la requête.
  Ce refus fait partie du contrat explicitement exigé par la campagne.
- Le refus précoce d'un corps excessif passe actuellement par `read_or_reject`
  et une enveloppe OpenAI, même pour `/v1/messages`. Le test Anthropic doit exposer
  cet écart ; il n'est pas converti en skip. Les timestamps des chunks doivent
  aussi rester stables même quand la génération traverse plusieurs secondes.

## Rapports et codes de sortie

Chaque répertoire de sortie reçoit `report.json` et `report.md`, réécrits après
chaque cas et à la fermeture. Le JSON versionné contient révision Git, versions
SDK exigées et vérifiées, dates, sélection, résultats et durées. Chaque ligne
indique endpoint × option × modèle × mode, statut, `executed` et détail.

- **OK** : assertions exécutées et satisfaites.
- **ÉCHEC** : assertion, SDK, transport, timeout, configuration ou processus en
  défaut. Un échec ne masque pas les suivants.
- **NON SUPPORTÉ attendu** : capacité absente ou endpoint hors du profil choisi.
  `executed=false` désigne une décision statique ; `executed=true` désigne un
  refus attendu effectivement vérifié. Un refus incorrect devient ÉCHEC.

Sortie 0 si aucun échec, 1 si au moins un échec/interruption, 2 pour une CLI
invalide. Un résultat vert avec des lignes non supportées ne signifie pas une
compatibilité complète. Les rapports ne contiennent ni chemins des poids, ni clé
API, ni texte généré. Le journal serveur temporaire, utilisé pour vérifier les
modes, est supprimé à la fermeture. Ne pas publier de diagnostics locaux bruts.

## Validation sans inférence et CI légère

```sh
python -I tools/e2e/test_contracts.py
python -I tools/e2e/run.py --target qwen38 --suite smoke --list
```

Les tests stdlib injectent des réponses en mémoire, y compris des flux tronqués,
des usages faux, des erreurs de type, des arguments outils invalides et des
événements désordonnés. Ils vérifient aussi le classement des résultats, la
suppression des flags hérités, l'arrêt forcé et la détection d'un repli de mode.
Si les SDK sont installés, leurs classes officielles décodent les fixtures et
leurs clients sont construits/fermés sans réseau. Sinon ce seul contrôle SDK est
signalé comme ignoré. Ces tests sont utilisables dans une CI CPU sans installer
de dépendance ; ils ne constituent pas une preuve de conformité du serveur.

`--suite smoke` sélectionne les mêmes objets de cas que la matrice complète :
models, simple/max_tokens/stream des deux API, stream_usage, JSON, n et erreurs,
ainsi que les endpoints non configurés. Elle peut être lancée sous verrou sur un
modèle réel, mais **aucun profil synthétique chargeable par `serve` n'est livré
actuellement**.

Le mini-modèle de [state/json_tests.rs](../../src/serve/state/json_tests.rs)
construit directement un `LoadedModel` avec `CausalDecoder::from_tensors` et un
catalogue safetensors vide ; son répertoire temporaire n'est pas un checkpoint
chargeable par la CLI. Les mocks Anthropic/audio sont également internes aux
tests Rust. Aucun de ces tests n'expose un serveur HTTP autonome.

Pour ajouter la véritable CI de bout en bout synthétique plus tard :

1. Exporter la fixture CPU existante vers un checkpoint local complet (config,
   tokenizer, safetensors contenant les tenseurs requis), dans un répertoire
   temporaire du runner CI, sans poids téléchargés.
2. Ajouter un profil `synthetic` explicitement `--backend cpu`, avec les capacités
   de ce tokenizer et des attentes de contenu déterministes adaptées ; réutiliser
   `Matrix` et les validateurs, sans serveur Python qui simulerait `serve`.
3. Exposer ce profil à `--suite smoke`, puis lancer le binaire de production avec
   cette fixture. Les tests de transport, schéma, usage, SSE et erreurs restent
   identiques ; les assertions sémantiques nécessitent des sorties synthétiques
   maîtrisées (le vocabulaire actuel `{}`, `<pad>`, `<eos>` ne peut produire Bonjour).
4. Publier les deux rapports comme artefacts de CI et faire échouer le job sur le
   code retour non nul. Conserver la campagne de modèles réels sous verrou GPU
   pour les outils, la sémantique, l'audio et les modes spéculatifs.
