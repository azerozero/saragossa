# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/azerozero/saragossa/releases/tag/v0.2.0) - 2026-07-31

### Added

- single-copy unified-memory, gathers u3, serve prêt-à-l'emploi (v0.3.0) ([#7](https://github.com/azerozero/saragossa/pull/7))
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
