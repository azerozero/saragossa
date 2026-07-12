# saragossa

![licence](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-blue)
![plateforme](https://img.shields.io/badge/plateforme-Apple%20Silicon-black)
![rust](https://img.shields.io/badge/rust-1.85%2B-orange)
![python](https://img.shields.io/badge/python-0%20d%C3%A9pendance-success)

Moteur d'inférence **Rust pur** sur Apple Silicon : kernels `metal-rs` bruts,
zéro Python, zéro dépendance à MLX ou CoreML. LLM, STT et TTS dans un seul
binaire, utilisables en bibliothèque ou derrière le serveur HTTP
OpenAI-compatible `saragossa serve`. Né comme le moteur du projet reti (agent
vocal local), il s'utilise seul.

## Positionnement

Dans le paysage des backends d'inférence (Ollama, llama.cpp, vLLM, SGLang…),
saragossa occupe le quadrant **latence mono-utilisateur × Apple Silicon** :

- vLLM/SGLang/TGI/LMDeploy exigent un GPU NVIDIA ; ici tout est Metal natif.
- Ollama et llama.cpp sont multi-plateformes généralistes ; saragossa est
  optimisé pour UNE cible (GPU Apple Silicon, decode résident) et **dépasse
  `mlx_lm` sur les MoE** à quantification comparable.
- Optimisé latence locale (agent, copilote, boucle vocale), pas throughput
  multi-tenant : le serveur est mono-thread par choix.
- Cache chaud de préfixe par blocs avec snapshots GPU (même famille d'idées que
  le RadixAttention de SGLang) : le multi-turn ne repaye pas son historique.

## Capacités

| Domaine | Détail |
|---|---|
| LLM | Qwen3.x dense et MoE (27B/30B/35B-A3B), loader générique Llama/Mistral/Gemma 3 ; quantifs u4/u6/u8 gs32-128, scales/biases bf16 |
| STT | Whisper large-v3-turbo (encodeur + décodeur résidents, GEMM Neural-Accelerators bf16) |
| TTS | Qwen3-TTS : talker résident + codec GPU + streaming intra-phrase |
| Embeddings | e5-small pur Rust (CPU), pour la mémoire sémantique / le RAG |
| Serveur | HTTP multi-modèle mono-thread, pool LRU, cache chaud, garde OOM |

### Endpoints `saragossa serve`

| Endpoint | Rôle |
|---|---|
| `GET /v1/models` | Modèles servis |
| `POST /v1/chat/completions` | Chat OpenAI-compatible (SSE ou non), `response_format: {"type":"json_object"}` |
| `POST /v1/messages` | Shim Anthropic Messages + `tool_use` (pilotable par Claude Code) |
| `POST /v1/audio/transcriptions` | STT Whisper (multipart WAV, opt-in `--stt-model`) |
| `POST /v1/audio/speech` | TTS Qwen3 (JSON → WAV, opt-in `--tts-model`) |
| `POST /v1/embeddings` | Embeddings e5-small (opt-in `--embed-model`) |

## Principes de conception

- **Résidence GPU** : en decode, 1 token = 1 command buffer, zéro readback ni
  `commit_and_wait` par couche. Le per-op CPU-orchestré n'existe qu'en repli.
- **Byte-identité comme gate** : toute optimisation prouve qu'elle préserve la
  sortie (oracles md5 e2e, goldens STT/TTS) ; les dérives near-tie sont
  qualifiées (ids + top-5 + marge) et actées — jamais silencieuses.
- **Tout est débrayable** : chaque chemin optimisé a son kill-switch env ;
  les flags sont centralisés dans [`src/runtime_flags.rs`](src/runtime_flags.rs).

## Perfs (mesurées le 2026-07-06, M5 Max)

- Decode 35B-A3B greedy @1k : **145,7 tok/s** (`4bit`, défaut prod) ·
  **105,9 tok/s** (`oQ8`) ; chemin prod T>0 devant `mlx_lm` jusqu'à 32k
  (89,8 vs 88,1 tok/s @32k oQ8, KV bf16).
- Prefill 35B : 1,0 s @2k · 3,5 s @8k · 23,3 s @32k.
- STT Whisper turbo : rtf 0,107 · TTS : e2e 0,747, TTFA streaming ~1,2 s.

Ces chiffres valent pour leur contexte (matériel, modèle, longueur) — mesurez
sur votre machine avant de figer un choix.

## Usage

```bash
# CLI de dev (le binaire requiert la feature devtools, activée par défaut) :
# génération LLM directe.
cargo run --release -p saragossa -- \
  --model-dir models/Qwen3.6-35B-A3B-oQ8 --backend metal \
  --prompt "Bonjour" --max-tokens 64 --temperature 0 --metrics
```

En bibliothèque, les points d'entrée sont `qwen_loader` (LLM), `whisper` (STT),
`tts` (TTS) et `text_embedder` (embeddings).

## Serveur `saragossa serve`

Serveur HTTP local **mono-thread** (usage mono-utilisateur assumé), multi-modèle.
Transport socket Unix par défaut (`/tmp/saragossa-serve.sock`, chmod 0600) ; le
TCP loopback exige un bearer (`--api-key` ou `SARAGOSSA_API_KEY`). Deadline de
lecture par connexion (30 s) et plafond dur `max_tokens` (4096) débrayent les
requêtes qui dérapent.

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

### Structured output (v1 : `json_object`)

`POST /v1/chat/completions` accepte `response_format: {"type":"json_object"}`.
La sortie est contrainte par un automate JSON byte-level côté sampler : objet
racine obligatoire, chaînes/échappements/nombres/booléens/null et structures
imbriquées. L'EOT n'est admissible qu'après fermeture de l'objet racine.

En v1, seules ces requêtes guidées basculent sur le chemin de sampling CPU
(logits relus puis masqués avant sample). Les requêtes sans `response_format`,
ou avec `{"type":"text"}`, gardent le chemin résident/GPU existant. Le mode
`{"type":"json_schema"}` répond 501 : il est réservé à une version ultérieure.

Limites v1 : l'automate garantit la grammaire JSON (paires de surrogates
`𐀀` incluses) mais pas la magnitude d'un nombre au-delà de `f64` ;
en non-stream, un objet non fermé au budget `max_tokens` remonte une erreur
plutôt que du JSON tronqué ; en SSE, les deltas restent un préfixe JSON valide
(la garantie « objet complet » ne vaut qu'en fin de flux).

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

## Features

| Feature | Défaut | Rôle |
|---|---|---|
| `metal` | oui | Kernels GPU Metal (macOS ; `src/kernels.metal` embarqué au build, compilé au runtime) |
| `devtools` | oui (lib) / requis (bin) | Harnais bench/diagnostic (DFlash, MTP, doctor) — exclu des binaires de prod |

Prérequis : Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`).

## Licence

Double licence, au choix : [MIT](https://spdx.org/licenses/MIT.html) ou
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) (SPDX `MIT OR Apache-2.0`).
