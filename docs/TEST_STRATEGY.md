# Stratégie de tests

## Périmètre et niveaux de preuve

La suite valide le moteur et ses interfaces publiques. Elle sépare le calcul
CPU déterministe, le transport sans poids réels, la résistance des parseurs,
et la qualité utilisateur sur snapshots locaux. Une couverture CPU ne mesure
pas la qualité audio ni la parité Metal.

Les points 1–6 de l’audit sont traités sur la branche de correction TTS ; cette
branche ne modifie pas leurs oracles. Le point 7 est réservé et documenté dans
[tests/fixtures/wer](../tests/fixtures/wer/README.md), sans téléchargement.

## Matrice de traçabilité

| Point / contrat | Preuve ou cible | Exécution | Limite |
| --- | --- | --- | --- |
| 8 / STT-1 | `long_fixture_covers_every_sample`, `quality_whisper_long_audio_keeps_every_chunk` | Construction CPU ; transcription ignorée, nightly locale | Voix embarquée répétée, pas un corpus |
| 8 / STT-2 | `quality_whisper_five_seconds_silence_is_empty` | Whisper + Metal, ignoré | Peut révéler une hallucination à corriger |
| 9 / STT-4, TTS-1/2 | `tools/e2e/matrix.py` : `wav`, `wav_long`, `voice_custom`, transcriptions JSON/texte | Serveur et poids locaux | CustomVoice dépend des corrections TTS |
| 10 / EMB-1/2 | `golden_synthetic_embedder_cpu` | CPU, non ignoré | BERT synthétique, pas la qualité sémantique E5 |
| 10 / EMB-3 | `embeddings_http_cpu_matches_synthetic_golden` | HTTP sur socket Unix, CPU, non ignoré avec devtools | Pas de transport socket Unix sous Windows |
| 11 / WIRE-1/2/3, LLM-3 | `wav_decode`, `multipart`, `sse_parse`, `chat_template` | Nightly CI, 300 s par cible | Check local ≠ campagne fuzz exécutée |
| 12 / LLM-1/2, TTS-1 | `.cargo/mutants.toml`, workflow `mutants.yml` | CPU, hebdomadaire | Survivants à examiner, pas un taux supposé |
| 13 / RUN-3 | `tests/sampling_properties.rs`, `proptest-regressions/` | CPU, CI | Aucun historique de panne inventé |
| 14 / RUN-2 | `tools/nightly-real.sh` | Local, release, sous gpulock | Exige les modèles déjà présents |
| 15 / tous | `CONTRACTS.md`, ce document | Revue | Un contrat exigé peut rester non validé en réel |

## Gates ordinaires

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --locked
cargo test --locked --all-features
cargo test --locked --no-default-features --features devtools
python3 -I -B tools/e2e/test_audio_quality.py
python3 -I -B tools/e2e/test_contracts.py
bash -n tools/nightly-real.sh
shellcheck tools/nightly-real.sh
```

Sur macOS, préfixer les compilations par `taskpolicy -b`. `--all-features`
compile aussi Metal : la recette CPU stricte utilise `--no-default-features`
ou exécute les binaires all-features dans un environnement sans device Metal.
Les tests ignorés restent exclus de ces gates. Les tests synthétiques figurent
parmi les intégrations exécutées par le gate all-features existant.

Le mini-embedder est construit au runtime avec les dépendances déjà verrouillées
(safetensors et tempfile). Il n’utilise aucun poids ni golden capturé.

## Matrice audio réelle

Le plancher de durée est `max(1, mots / 5)` secondes, y compris pour la phrase
longue. Les transcriptions exigent similarité ≥ 0,8 et WER ≤ 0,20. Les mots sont
normalisés en Unicode NFC, en minuscules, avec séparation de la ponctuation.
Un seuil de similarité CLI supérieur est respecté ; un seuil inférieur ne
relâche pas le minimum de 0,8. Une référence vide est une erreur.

Pour `voice_custom`, fournir `E2E_CUSTOM_VOICE` et un snapshot local CustomVoice
via `E2E_QWEN3_TTS_MODEL`. Ce cas vérifie la requête et la sortie audio ; il ne
certifie pas l’identité perceptuelle de la voix. Les points 1–6 restent requis
pour le support effectif et la qualité TTS complète.

## Fuzzing

`fuzz/` est un crate autonome hors workspace, comme celui de grob. Seul ce
crate dépend de `libfuzzer-sys = 0.4.13`. Ses cibles appellent `wire_audio`,
`sse` et `chat_template` publics. Les parseurs partagés préservent la conversion
des erreurs en HTTP 400 dans le serveur.

Le workflow `fuzz.yml` installe uniquement en CI `cargo-fuzz 0.13.1` et
`nightly-2026-09-02`, puis exécute chaque cible pendant cinq minutes. Les actions
sont épinglées par SHA. En local, sans installation :

```sh
rustup toolchain list
taskpolicy -b rustup run nightly cargo check --offline --locked --manifest-path fuzz/Cargo.toml
```

Le lockfile du crate séparé est versionné. Un crash doit être réduit puis
reproduit par un test déterministe avant correction ; les corpus générés et
les sorties volumineuses restent hors Git.

## Mutations et propriétés

Le job hebdomadaire utilise `cargo-mutants 24.11.2` sur les trois modules
TTS `sampling`, `helpers`, `synth`, ainsi que `serve/audio.rs` et `sampling.rs`.
Il compile sans Metal, avec devtools pour inclure les tests du serveur.
Aucun survivant n’est automatiquement exclu.

Commande locale bornée aux deux fichiers autorisés :

```sh
TMPDIR=/private/tmp taskpolicy -b cargo mutants \
  --no-default-features --features devtools \
  -f src/sampling.rs -f src/serve/audio.rs \
  -C --offline -C --locked -- --lib --bins -- sampling:: serve::audio::
```

Placer les temporaires hors du checkout évite une copie récursive de celui-ci.
Conserver `caught`, `missed`, `timeout`, `unviable`, la commande et la révision.
Les assertions ajoutées après une campagne ne sont pas un résultat de seconde
campagne : distinguer leur test normal de la vérification d’un mutant rejoué.

Les graines sont expliquées dans
[proptest-regressions](../proptest-regressions/README.md). Rejouer les échecs
réduits avant d’augmenter le nombre de tirages aléatoires.

## Nightly réelle et verrou

Configurer un chemin de verrou commun aux consommateurs GPU et les chemins
de snapshots locaux. Ne pas lancer de téléchargement implicite :

```sh
export GPU_LOCK="${TMPDIR:-/tmp}/saragossa-gpu.lock"
export NIGHTLY_LOG_DIR="target/nightly-real"
bash tools/nightly-real.sh
```

`GPU_LOCK_TOOL`, `GPU_LOCK_OWNER` et `GPU_LOCK_PRIO` sont paramétrables.
Le script fixe `SARAGOSSA_REQUIRE_REAL_MODELS=1`, `SARAGOSSA_REQUIRE_METAL=1`,
`HF_HUB_OFFLINE=1` et `CARGO_NET_OFFLINE=true`. Il lance en release les tests
ignorés dont le nom contient `live_`, `golden_`, `quality_` ou `external_`, avec
un seul thread de test. Ces filtres sont une union, chaque test n’est joué
qu’une fois. Le journal garde la sortie et le code retour, y compris en échec.

Pour vérifier seulement le verrou et le refus d’un modèle absent, sélectionner
un unique test ignoré `external_whisper_small_matches_transformers` avec un
chemin de modèle inexistant. Un code 101 est attendu et ne prouve aucune
inférence réelle. Ne pas confondre cette sonde avec une nightly complète.

## Critères de sortie

- Gates format, clippy, build et tests concernés réussis ; écarts préexistants
  séparés des régressions introduites.
- Aucun poids téléchargé, golden régénéré ni tolérance élargie.
- Fuzz : compilation locale des quatre cibles, puis campagnes CI à réussir.
- Mutants : rapport conservé, survivants significatifs reliés à des assertions,
  inconnues et équivalences explicitement signalées.
- Recettes réelles : assets identifiés, verrou détenu, zéro saut de précondition,
  texte/durée/silence mesurés et seuils respectés avant de déclarer la qualité
  utilisateur validée.
- Huit commits séparés, sans publication automatique.
