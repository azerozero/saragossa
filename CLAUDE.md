# saragossa — Guide de développement

## Architecture

saragossa est un moteur d'inférence Rust pur pour Apple Silicon. Les kernels
Metal sont pilotés avec `metal-rs`; aucun runtime Python, MLX ou CUDA n'est
requis. Le crate fournit :

- les LLM Qwen dense/MoE et Gemma, avec quantification et decode GPU-résident ;
- Whisper STT, avec encodeur et décodeur résidents ;
- Qwen3-TTS, son codec GPU et le streaming audio ;
- un embedder texte CPU ;
- `saragossa serve`, serveur local compatible avec les endpoints OpenAI pour
  le chat, les transcriptions, la synthèse et les embeddings.

Le chemin Metal vise un command buffer par token en decode. Les buffers KV,
l'état récurrent, les activations et le sampling restent sur le GPU jusqu'au
retour de l'identifiant du token. Les variantes CPU et les kill-switches servent
de référence et de repli mesurable.

## Carte des modules

| Module | Responsabilité |
|---|---|
| `src/decoder/` | génération, cache d'attention, decode résident et vérification |
| `src/decode_resident/` | arène GPU, attention, tails dense/MoE et kernels de decode |
| `src/metal_backend/` | buffers, pipelines, matmuls, MoE, prefill et barrières Metal |
| `src/qwen_loader/` | chargement et validation des familles de modèles |
| `src/whisper/` | STT Whisper |
| `src/tts*.rs` | talker, codec, Mimi et voix Qwen3-TTS |
| `src/text_embedder/` | embeddings texte CPU |
| `src/serve/` | transport local et API compatible OpenAI |
| `src/runtime_flags.rs` | source unique des flags et diagnostics runtime |
| `src/kernels.metal` | kernels Metal généraux |
| `build.rs` | préparation des assets Metal à la compilation |

## Conventions Rust et Metal

- Pas de `unwrap()` ou `panic!` sur les chemins faillibles de production.
  Retourner une erreur typée et propager avec `?`.
- `expect()` documente un invariant prouvé avec le préfixe `invariant:`.
- Tout bloc `unsafe` est local, minimal et précédé de `// SAFETY:`.
- Les commentaires expliquent pourquoi. Utiliser `TODO:`, `FIXME:`, `HACK:`,
  `NOTE:` et `SAFETY:` avec une phrase complète.
- Les optimisations GPU conservent un oracle CPU ou un golden. Une dérive
  numérique est mesurée, documentée et testée avant activation par défaut.
- Les fichiers volumineux sont découpés par responsabilité métier; éviter les
  nouveaux modules fourre-tout.

## Vérifications

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo build
cargo test --all-features
cargo deny check
cargo audit
cargo machete
```

La compilation Metal exige macOS et la Metal Toolchain. Les tests d'inférence
qui nécessitent des poids restent ignorés ou activés explicitement; les gates
ordinaires ne téléchargent et n'exécutent aucun modèle.

Les hooks [`prek.toml`](prek.toml) lancent les contrôles rapides au commit et
les contrôles de dépendances au push.

## Contributions et releases

Le flux public est : fork du dépôt, branche courte, pull request vers `main`,
signature du [`CLA`](CLA.md), puis CI verte. Les commits suivent Conventional
Commits en français et n'ajoutent aucun trailer de paternité automatisé.

`release-plz` prépare les changements de version et le changelog. Seul le
workflow de release publie le crate; une contribution ordinaire ne modifie pas
manuellement les tags et ne lance pas `cargo publish`.
