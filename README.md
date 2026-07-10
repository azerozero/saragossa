# saragossa

![licence](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-blue)
![plateforme](https://img.shields.io/badge/plateforme-Apple%20Silicon-black)
![rust](https://img.shields.io/badge/rust-1.85%2B-orange)
![python](https://img.shields.io/badge/python-0%20d%C3%A9pendance-success)

Moteur d'inférence **Rust pur** sur Apple Silicon (kernels `metal-rs` bruts,
zéro Python, zéro MLX). C'est le backend **prod par défaut** de
[reti](../../README.md) (`--backends rust-metal`) : il fait tourner in-process
le LLM, le STT et le TTS de la boucle vocale. `saragossa` est **open source**
(double licence MIT ou Apache-2.0) et s'utilise aussi seul, hors reti, via son
serveur `saragossa serve`.

## Capacités

| Domaine | Détail |
|---|---|
| LLM | Qwen3.x dense et MoE (27B/30B/35B-A3B), loader générique Llama/Mistral/Gemma 3 ; quantifs u4/u6/u8 gs32-128, scales/biases bf16 |
| STT | Whisper large-v3-turbo (encodeur + décodeur résidents, GEMM Neural-Accelerators bf16) |
| TTS | Qwen3-TTS : talker résident + codec GPU + streaming intra-phrase |
| Serveur | `saragossa serve` : HTTP multi-modèle, endpoints **OpenAI** (`/v1/chat/completions`, `/v1/models`) et **Anthropic** (`/v1/messages`, pour Claude Code) ; cache chaud par blocs, garde OOM, pool de modèles LRU |

## Principes de conception

- **Résidence GPU** : en decode, 1 token = 1 command buffer, zéro readback ni
  `commit_and_wait` par couche. Le per-op CPU-orchestré n'existe qu'en repli.
- **Byte-identité comme gate** : toute optimisation prouve qu'elle préserve la
  sortie (oracles md5 e2e, goldens STT/TTS) ; les dérives near-tie sont
  qualifiées (ids + top-5 + marge) et actées par dossier — jamais silencieuses.
- **Tout est débrayable** : chaque chemin optimisé a son kill-switch env ;
  les flags sont centralisés dans [`src/runtime_flags.rs`](src/runtime_flags.rs).
- Les dossiers de mesure et de décision vivent dans
  [`docs/reviews/`](../../docs/reviews/) ; l'architecture cible dans
  l'[ADR 0011](../../docs/adr/0011-saragossa-moteur-modulaire.md).

## Perfs (mesurées le 2026-07-06, M5 Max)

- Decode 35B-A3B greedy @1k : **145,7 tok/s** (`4bit`, défaut prod) ·
  **105,9 tok/s** (`oQ8`) ; chemin prod T>0 devant `mlx_lm` jusqu'à 32k
  (89,8 vs 88,1 tok/s @32k oQ8, KV bf16).
- Prefill 35B : 1,0 s @2k · 3,5 s @8k · 23,3 s @32k.
- STT Whisper turbo : rtf 0,107 · TTS : e2e 0,747, TTFA streaming ~1,2 s.

## Usage

```bash
# Bibliothèque : via reti (chemin nominal).
cargo run --release -- boss --backends rust-metal

# CLI de dev (le binaire requiert la feature devtools, activée par défaut) :
# smoke LLM direct.
cargo run --release -p saragossa -- \
  --model-dir models/Qwen3.6-35B-A3B-oQ8 --backend metal \
  --prompt "Bonjour" --max-tokens 64 --temperature 0 --metrics
```

## Serveur `saragossa serve`

Serveur HTTP local **mono-thread** (usage mono-utilisateur assumé), multi-modèle.
Transport socket Unix par défaut (`/tmp/saragossa-serve.sock`, chmod 0600) ; le
TCP loopback exige un bearer (`--api-key` ou `SARAGOSSA_API_KEY`). Read-timeout
par connexion (30 s) et plafond dur `max_tokens` (4096) débrayent les requêtes
qui dérapent.

```bash
# OpenAI-compatible sur socket Unix.
cargo run --release -p saragossa -- serve \
  --model qwen35=models/Qwen3.6-35B-A3B-oQ8

# TCP loopback + bearer, pour brancher Claude Code (shim Anthropic).
SARAGOSSA_API_KEY=local-dev cargo run --release -p saragossa -- serve \
  --port 8081 --model qwen35=models/Qwen3.6-35B-A3B-oQ8
```

```bash
# Claude Code parle au moteur local via /v1/messages.
ANTHROPIC_BASE_URL=http://127.0.0.1:8081 ANTHROPIC_API_KEY=local-dev claude
```

### Cache chaud (prefix-cache par blocs)

Les prompts sont découpés en blocs de 256 tokens (`RETI_SERVE_PREFIX_BLOCK_TOKENS`)
hachés en chaîne (SHA-256 de `hash_précédent ‖ tokens`) : un préfixe ne réutilise
un état que si **toute la chaîne amont** est identique. Chaque frontière de bloc
retient l'état de prompt CPU **et** son **snapshot Metal** (KV + état récurrent
linéaire GDN résident sur GPU), rechargé tel quel sur hit — donc seul le suffixe
est prérempli. Le cache est un LRU de 128 blocs (`RETI_SERVE_PREFIX_CACHE_BLOCKS`) ;
le header `x-saragossa-reused-prefix-tokens` rapporte la reprise.

### Garde OOM + pool de modèles LRU

- **Garde OOM prédictive** : projette l'empreinte process (`phys_footprint` Mach)
  plus le coût de la prochaine allocation contre le plus bas de trois plafonds —
  cap statique, mémoire hôte moins marge (2 Gio), working-set Metal recommandé.
  En cas de dépassement projeté, évince d'abord des blocs de cache, puis des
  modèles ; sinon refuse la requête (HTTP 503).
- **Pool de modèles LRU** : jusqu'à 2 modèles résidents simultanés
  (`RETI_SERVE_MODEL_POOL`) ; charger un modèle de plus évince le moins récemment
  utilisé.

| Variable | Défaut | Rôle |
|---|---|---|
| `RETI_SERVE_PREFIX_CACHE` | on | Cache chaud de préfixe par blocs |
| `RETI_SERVE_PREFIX_BLOCK_TOKENS` | 256 | Taille d'un bloc (tokens) |
| `RETI_SERVE_PREFIX_CACHE_BLOCKS` | 128 | Capacité LRU du cache (blocs) |
| `RETI_SERVE_LRU` | on | Pool LRU de modèles résidents |
| `RETI_SERVE_MODEL_POOL` | 2 | Modèles résidents simultanés |
| `RETI_SERVE_OOM_GUARD` | on | Garde mémoire prédictive |
| `RETI_SERVE_MEMORY_HEADROOM_BYTES` | 2 Gio | Marge hôte conservée hors process |
| `RETI_SERVE_MEMORY_CAP_BYTES` | auto | Plafond mémoire statique explicite |

### Shim Anthropic `/v1/messages` (Claude Code)

`POST /v1/messages` accepte le format Anthropic Messages en plus des routes OpenAI,
pour piloter le moteur local depuis Claude Code :

- Les `tools` déclarés sont rendus en bloc système (signatures `<tools>` + protocole
  d'appel) ; la sortie `<tool_call>{…}</tool_call>` du modèle est reparsée en blocs
  `tool_use` (`id`, `name`, `input` JSON). Les `tool_result` entrants deviennent des
  messages `tool`.
- Streaming SSE Anthropic (`message_start` → `content_block_*` → `message_delta` →
  `message_stop`) et non-streaming. `stop_reason` mappé sur
  `tool_use` / `max_tokens` / `stop_sequence` / `end_turn` ; erreurs au format Anthropic.

## Features

| Feature | Défaut | Rôle |
|---|---|---|
| `metal` | oui | Kernels GPU Metal (macOS ; `src/kernels.metal` embarqué au build, compilé au runtime) |
| `devtools` | oui (lib) / requis (bin) | Harnais bench/diagnostic (DFlash, MTP, doctor) — exclu du binaire reti prod |

Prérequis : Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`).

## Licence

Double licence, au choix : [MIT](https://spdx.org/licenses/MIT.html) ou
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) (SPDX `MIT OR Apache-2.0`).
