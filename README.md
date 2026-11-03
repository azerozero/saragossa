# saragossa

![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![rust](https://img.shields.io/badge/rust-1.85%2B-orange)
![python](https://img.shields.io/badge/python-zero%20deps-success)

**Pure-Rust Metal inference engine for Apple Silicon.** Raw `metal-rs` kernels,
no Python, no MLX, no CoreML. LLM, speech-to-text and text-to-speech in a single
binary — usable as a library or behind the OpenAI-compatible HTTP server
`saragossa serve`.

## Where it fits

Among inference backends (Ollama, llama.cpp, vLLM, SGLang…), saragossa occupies
the **single-user latency × Apple Silicon** quadrant:

- vLLM / SGLang / TGI / LMDeploy require an NVIDIA GPU; here everything is native
  Metal.
- Ollama and llama.cpp are general-purpose and cross-platform; saragossa is tuned
  for **one** target (the Apple Silicon GPU, resident decode). On MoE **decode**
  at comparable quantization it is competitive with — and can exceed — `mlx_lm`
  (measured; benchmark on your own hardware).
- Optimized for local latency (agents, copilots, voice loops), not multi-tenant
  throughput: the server is single-threaded by design.
- A block-wise warm prefix cache with GPU snapshots (same family of ideas as
  SGLang's RadixAttention): multi-turn sessions don't re-pay for their history.

## Capabilities

| Domain | Details |
|---|---|
| LLM | Qwen3.5 / 3.6 / 3.8 dense and MoE (27B / 30B / 35B-A3B; MLX `affine` quants only, not `mxfp4`/`nvfp4`), Gemma 4 dense (`gemma4_unified`) and MoE (`gemma4`), plus a generic Llama / Mistral / Gemma 3 loader; u2–u8 (incl. mixed 3/4-bit) quantization at group sizes 32–128, bf16 scales/biases |
| STT | Whisper large-v3-turbo (resident encoder + decoder, bf16 Neural-Accelerator GEMM) |
| TTS | Qwen3-TTS: resident talker + GPU codec + intra-sentence streaming |
| Embeddings | e5-small, pure Rust (CPU) — for semantic memory / RAG |
| Server | Single-threaded multi-model HTTP, LRU model pool, warm cache, OOM guard |

### `saragossa serve` endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | Served models |
| `POST /v1/chat/completions` | OpenAI-compatible chat (SSE or not), `response_format: {"type":"json_object"}` |
| `POST /v1/messages` | Anthropic Messages shim + `tool_use` (drivable by Claude Code) |
| `POST /v1/audio/transcriptions` | Whisper STT (multipart WAV, opt-in `--stt-model`) |
| `POST /v1/audio/speech` | Qwen3 TTS (JSON → WAV, opt-in `--tts-model`) |
| `POST /v1/embeddings` | e5-small embeddings (opt-in `--embed-model`) |

## Design principles

- **GPU residency** — in decode, one token is one command buffer, with no
  readback and no per-layer `commit_and_wait`. The CPU-orchestrated per-op path
  exists only as a fallback.
- **Byte-identity as a gate** — every optimization proves it preserves the output
  (end-to-end md5 oracles, STT/TTS goldens); near-tie drifts are qualified (ids +
  top-5 + margin) and recorded — never silent.
- **Everything is switchable** — each optimized path has an environment
  kill-switch; the flags are centralized in
  [`src/runtime_flags.rs`](src/runtime_flags.rs).

## Performance

Measured 2026-07-06 on an M5 Max:

- Decode 35B-A3B, greedy @1k: **145.7 tok/s** (`4bit`, production default) ·
  **105.9 tok/s** (`oQ8`) · **89.8 tok/s** @32k (`oQ8`, KV bf16).
- Prefill 35B: 1.0 s @2k · 3.5 s @8k · 23.3 s @32k.
- Whisper turbo STT: rtf 0.107 · TTS: 0.747 end-to-end, streaming TTFA ~1.2 s.

These numbers hold for their context (hardware, model, length) — measure on your
own machine before committing to a choice.

## Quantization formats: footprint, throughput & quality

saragossa loads **MLX affine** integer weights `u2`/`u3`/`u4`/`u6`/`u8`
(group-size 64) and **fp8** (`e4m3`). Trade-offs on a **27B dense** model
(M5 Max, greedy decode T=0; footprint = resident weights; KL = `KL(bf16‖quant)`
averaged over 5 prompts):

| quant | bpw | weight footprint | decode | quality vs bf16 |
|-------|-----|------------------|--------|-----------------|
| `u8` | ~8.5 | ~29 GB | — | near-lossless |
| `u6` | ~6.5 | ~22 GB | — | near-lossless — **quality sweet-spot** |
| `u4` | ~4.5 | 18 GB¹ | 27 tok/s | good |
| `u3` mixed 3/4² | 3.85 | 12 GB | 35 tok/s | ≈ 4-bit (KL 0.07) |
| `u3` uniform | ~3.5 | 11 GB | 37 tok/s | degraded (KL 0.30) |
| `u2` mixed 2/3² | 2.9 | 9 GB | — | too degraded (KL 0.58) |

¹ A 27B `u4` fits a **24 GB** Mac: the single-copy unified-memory loader (v0.3)
stores weights once, in the shared Metal buffer → ~22 GB footprint (dense);
the 35B-A3B MoE drops from 40.5 to 19.4 GB.
² Sensitivity-based mixed quant: the most sensitive layers move up one bit.

**fp8** loads but a 27B (29 GB) saturates a 24-32 GB Mac — prefer `u6`/`u8` for
high fidelity (smaller, same quality).

**Speculative decode (MTP)**: on a **dense** model carrying a native MTP head,
in greedy (T=0), saragossa verifies the draft against the target model → +15
to +40% throughput (`u4` 27→~40, `u3` 35→~46 tok/s). The output matches the
autoregressive (AR) decode except near a tie between the top two logits. Two
default-on kernels of the batched verification are not bit-identical to the
single-row step: the skinny u4 matmul, from 5 rows (DFlash, lookup, MTP with
lookup drafts on long prompts), and the attention prefix, from 16k tokens of
context with a bf16 KV cache. A tiny drift can then flip such a tie (before the
other fixes: 1 request in 66 on a 32k prompt). `SARAGOSSA_RUST_VERIFY_STRICT=1`
(opt-in) makes greedy verification exact: the attention prefix stays on the
exact path, matmuls of 3 to 16 rows use an exact kernel instead of the skinny
one, and the DFlash tree defaults to 3 nodes. Cost on the 27B with DFlash: −4 %
at 4k and −12 % at 32k decode versus the default. Verified states then match
the AR bit for bit on the test corpus; see
[docs/perf/greedy-identity.md](docs/perf/greedy-identity.md). On the
**Qwen3.6-35B-A3B** MoE,
the official Qwen MTP head (not shipped in mlx-community weights) is opt-in:
point `SARAGOSSA_RUST_MTP_SIDECAR` at its `mtp.safetensors` (in a subdirectory,
not directly in the model root, e.g. `<model>/mtp-official/`). Greedy only, same
AR caveat on near ties: ×1.19 to ×1.61 decode (pairwise, 5 prompts), CLI 151→224-295 tok/s;
sampled requests stay on AR (the sampled path measured 0.94-0.99× AR). Cost:
+0.95 GB peak physical footprint in the CLI (+1.33 GB on a 424-token prompt),
+0.96 GB in serve after a 5,078-token request: the head is quantized straight
into shared Metal buffers, one copy. See
`generate_greedy_mtp_streaming_with_options` and `docs/perf/mtp-pairwise.md`.

## Install

### Homebrew (Apple Silicon)

```sh
brew install azerozero/tap/saragossa          # latest tagged release (stable)
# brew install --HEAD azerozero/tap/saragossa # build from the main branch
```

### From source

```sh
git clone https://github.com/azerozero/saragossa.git
cd saragossa
cargo install --path . --locked   # installs the `saragossa` binary
```

Requirements: **Apple Silicon** and the Metal Toolchain, which saragossa uses to
compile its GPU kernels on first launch:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Quick start

```sh
# Start an interactive chat, downloading the HF model if it is missing.
saragossa run mlx-community/Qwen3-4B-4bit

# List the models already present in the local Hugging Face cache.
saragossa list
```

For gated models (Gemma, for instance) accept the license on the model's Hugging
Face page first, then export `HF_TOKEN`.

## CLI usage

```sh
# Direct LLM generation (the binary requires the `devtools` feature, on by default):
saragossa \
  --model-dir models/Qwen3.6-35B-A3B-oQ8 --backend metal \
  --prompt "Hello" --max-tokens 64 --temperature 0 --metrics
```

## Library usage

Add the crate and drive the engine directly. The main entry points are
`qwen_loader` (LLM), `whisper` (STT), `tts` (TTS) and `text_embedder`
(embeddings); [`src/lib.rs`](src/lib.rs) lists the public API.

## Server: `saragossa serve`

A **single-threaded**, multi-model local HTTP server. The default behavior is
single-user: the warm state is global to a model until a session key is supplied.
The transport is a Unix socket by default (`/tmp/saragossa-serve.sock`, mode
0600); TCP loopback requires a bearer (`--api-key` or `SARAGOSSA_API_KEY`). A
per-connection read deadline (30 s) and a hard `max_tokens` ceiling (4096) shed
runaway requests.

```sh
# OpenAI-compatible, over a Unix socket.
saragossa serve --model qwen35=models/Qwen3.6-35B-A3B-oQ8

# TCP loopback + bearer, e.g. to wire up Claude Code (Anthropic shim).
SARAGOSSA_API_KEY=local-dev saragossa serve \
  --port 8081 --model qwen35=models/Qwen3.6-35B-A3B-oQ8
```

```sh
# Claude Code talks to the local engine via /v1/messages.
ANTHROPIC_BASE_URL=http://127.0.0.1:8081 ANTHROPIC_API_KEY=local-dev claude
```

### Structured output (v1: `json_object`)

`POST /v1/chat/completions` accepts `response_format: {"type":"json_object"}`.
The output is constrained by a byte-level JSON automaton in the sampler:
mandatory root object, strings/escapes/numbers/booleans/null and nested
structures. End-of-turn is only admissible after the root object closes.

In v1, only these guided requests switch to the CPU sampling path (logits read
back, then masked before sampling). Requests without `response_format`, or with
`{"type":"text"}`, keep the existing resident/GPU path. The
`{"type":"json_schema"}` mode returns 501: it is reserved for a later version.

v1 limits: the automaton guarantees JSON grammar (surrogate pairs `𐀀`
included) but not a number's magnitude beyond `f64`; in non-streaming mode, an
object left unclosed at the `max_tokens` budget raises an error rather than
returning truncated JSON; in SSE, the deltas stay a valid JSON prefix and an
unclosed object ends with an `error` event of type `incomplete_json`, without the
normal `[DONE]`.

### Sampling and request fields

`POST /v1/chat/completions` honours `temperature`, `top_p`, `top_k` (temperature
first, then top-k, then top-p on the scaled distribution), `seed` (unsigned; defaults to `0`, so an unseeded sampled
request stays reproducible), `stop` and `max_tokens` / `max_completion_tokens`.

- `chat_template_kwargs`: only `{"enable_thinking": bool}` is accepted. Qwen
  ChatML and Gemma 4 render it (thinking is **off** by default, which injects
  the empty `<think>` block); with thinking on, the reasoning stays in
  `content`. Flint and Gemma 3 have no switch and refuse an explicit value.
  Any other key is refused (HTTP 400) rather than silently ignored.
- `stream_options: {"include_usage": true}`: like the OpenAI API, a final
  chunk with `"choices": []` and `usage` precedes `data: [DONE]`.
- `min_p`, `presence_penalty`, `frequency_penalty` and `repetition_penalty`
  are not implemented: only their neutral value (`0`, `0`, `0`, `1`) is
  accepted, anything else returns 501.
- Each completion logs `prompt_tokens=…` and `prompt_fnv1a64=…` (FNV-1a 64 of
  the prompt ids as little-endian `u32`) on stderr, so a client can check that
  the ingested ids are the ones it tokenized.

### Warm cache (block-wise prefix cache)

Prompts are split into 256-token blocks, hashed as a chain (SHA-256 of
`previous_hash ‖ tokens`): a prefix only reuses state if the **entire upstream
chain** is identical. Each block boundary retains both the CPU prompt state
**and** its **Metal snapshot** (KV plus the resident linear-recurrent GDN state),
reloaded as-is on a hit — so only the suffix is prefilled. The cache is a
128-block LRU; the `x-saragossa-reused-prefix-tokens` header reports the reuse.

For a multi-user frontend, send `x-saragossa-session: <id>` on
`/v1/chat/completions`; otherwise the OpenAI `user` field serves as the session
key. On `/v1/messages`, the Anthropic shim uses the same header, then falls back
to `metadata.user_id`. The key derives the root of the hash chain: two distinct
sessions share no blocks, and `x-saragossa-reused-prefix-tokens` then reports only
intra-session reuse. Without a key, the historical root and the global namespace
are unchanged.

This isolation is **cache partitioning, not an authentication boundary**. On the
default Unix socket, with no bearer, any caller allowed by the file permissions
can claim any session id. For a real multi-tenant boundary, authenticate upstream
(a front gateway) or use the TCP bearer of `saragossa serve`.

### OOM guard + LRU model pool

- **Predictive OOM guard** — projects the process footprint (`phys_footprint`
  from Mach) plus the cost of the next allocation against the lowest of three
  ceilings: a static cap, host memory minus headroom, and the recommended Metal
  working-set. On a projected overshoot it first evicts cache blocks, then
  models; otherwise it refuses the request (HTTP 503).
- **LRU model pool** — up to two resident models at once; loading one more evicts
  the least recently used.

The warm cache, model pool and OOM guard are all tunable through environment
variables — see [`src/runtime_flags.rs`](src/runtime_flags.rs) for the exact
names and defaults (block size, LRU capacity, pool size, host headroom, static
memory cap, …).

## Features

| Feature | Default | Purpose |
|---|---|---|
| `metal` | yes | Metal GPU kernels (macOS; `src/kernels.metal` embedded at build time, compiled at runtime) |
| `devtools` | yes (lib) / required (bin) | Bench/diagnostic harness (DFlash, MTP, doctor) — excluded from production binaries |

Prerequisite: the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`).

## License

[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) (SPDX `Apache-2.0`).
