# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Chargeur : les quants MLX `u6` (ex. `mlx-community/*-oQ6`) étaient refusés à la validation de config alors que les kernels existent ; les modes `mxfp4`/`nvfp4` et un `output_gate_type` autre que `swish`/`silu` (Qwen3.8) sont désormais refusés avec un message clair au lieu d'une erreur de poids manquant ou d'une sortie corrompue.
- Decode résident : la liste blanche `RESIDENT_SUPPORTED_BITS` ignorait u2/u6 alors que leurs kernels résidents (`FastQmvU2`/`FastQmvU6`) existent ; un 27B oQ6 tombait en per-op.
- Kernels Metal `affine_gather_*` (experts MoE, embeddings) : le décodage bitstream n'était appliqué qu'à u3 ; u6 passait en lanes alignées (5 x 6 bits != 32) et aurait produit des experts corrompus. La condition devient `32 % bits != 0`, identique pour u3.
- MTP : un sidecar quantifié (`mlx-community/*-MTP-4bit`, `fc.scales`/`fc.biases` sans préfixe) échouait en `dtype non supporté pour mtp.fc: U32` ; la normalisation des clés du chargeur s'aligne sur celle de la détection (`assets.rs`). Le gain dépend du prompt et reste à mesurer sur des runs comparables.
- CLI et `run` : détection automatique de la tête MTP locale pour les modèles denses sous Metal en greedy non contraint. `SARAGOSSA_RUST_MTP_DECODE=0` désactive MTP ; les alias `RETI_*` restent acceptés, avec priorité aux variables `SARAGOSSA_*`. La profondeur reste à 1 par défaut, sans activation automatique du contrôleur adaptatif.

## [0.3.0](https://github.com/azerozero/saragossa/releases/tag/v0.3.0) - 2026-07-31

### Added

- Stockage single-copy des poids quantifiés en mémoire unifiée (défaut ON) : une seule résidence dès le chargement (35B-A3B : 40,5 → 19,4 Go ; 27B : ~22 Go), pic de load ≈ steady ([#7](https://github.com/azerozero/saragossa/pull/7))
- Gathers MoE rapides u3 : un 35B mixed 3/4-bit décode à ~150 tok/s (+60 %, dépasse le 4-bit avec 18 % de mémoire en moins) ([#7](https://github.com/azerozero/saragossa/pull/7))
- `saragossa serve` prêt-à-l'emploi : `--model` par chemin nu ou id Hugging Face (auto-téléchargement), picker interactif sans argument, decode spéculatif MTP automatique (modèle dense + tête + requête greedy) ([#7](https://github.com/azerozero/saragossa/pull/7))

### Fixed

- Guard de bit-width du decode résident : les quants u6/u2 retombent sur le per-op correct au lieu de produire du texte corrompu ([#7](https://github.com/azerozero/saragossa/pull/7))
- Purge du prefix-cache après la chauffe de libération des poids (restaure le byte-id MoE sur prompt long) ([#7](https://github.com/azerozero/saragossa/pull/7))

## [0.2.0](https://github.com/azerozero/saragossa/releases/tag/v0.2.0) - 2026-07-30

### Added

- kernels affine u2/u3 + decode MTP spéculatif streaming (v0.2.0) ([#6](https://github.com/azerozero/saragossa/pull/6))
- gemma4 loader, saragossa run/list CLI, per-session cache isolation, guided-JSON polish
- *(serve)* structured output json_object (v1)

### Fixed

- *(tests)* chemins HF portables via $HOME (pas de chemin personnel en dur)

### Other

- *(release-plz)* mode git pur (pas de registre crates.io)
- *(release-plz)* bump l'action v0.5.50 → v0.5.131 (fix auth git push)
- *(brew)* bump la Formula vers v0.2.0
- licence Apache-2.0 seule + table de quantification
- *(brew)* activer le build stable (tag v0.1.0) ([#5](https://github.com/azerozero/saragossa/pull/5))
- *(readme)* réécrire en anglais et documenter le projet en autonome ([#4](https://github.com/azerozero/saragossa/pull/4))
- hygiène bon-marché (cache, binaires prébuiltés, harden-runner, MSRV, doc) ([#3](https://github.com/azerozero/saragossa/pull/3))
- *(brew)* ajouter une formule Homebrew (tap installable en --HEAD)
- sync du moteur saragossa depuis reti @aa3bec9
- sync du moteur saragossa depuis reti @2d7d437
- ajouter le contexte de contribution saragossa
- renforcer les contrôles et la publication crates.io
- sync du moteur saragossa depuis reti @523e0da
- Shared predictive memory guard + external-encoder transcription seam
- Make decode pacing hot-configurable
- True SSE streaming, /health endpoint, bench-serve subcommand
- Add /v1/embeddings, fix non-Metal build, standalone-first README
- Add OpenAI-compatible audio endpoints to serve
- Serve hardening, loader guard, and engine updates
- Add dual MIT/Apache-2.0 license files + gitleaks allowlist
- run push-time audit via cargo-audit CLI, not the check-run action
- relative tolerance off-M5 for kernel-variant asserts
- commit Cargo.lock (audit needs it) + widen portable ULP tolerance to 16
- Portable bitwise asserts + 35B OptiQ-4bit benchmark row
- bench rig, per-run GPU temperatures, verified HF links
- honest same-protocol A/B benchmarks + model links
- rust checks, daily audit, auto-merge on green (mirrors the parent repo)
- Sync engine to 2026-07-06 state
- Initial release: pure-Rust Metal LLM inference engine for Apple Silicon
