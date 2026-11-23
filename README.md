# saragossa

[![CI](https://github.com/azerozero/saragossa/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/azerozero/saragossa/actions/workflows/rust.yml)
[![release](https://img.shields.io/github/v/release/azerozero/saragossa)](https://github.com/azerozero/saragossa/releases)
[![crates.io](https://img.shields.io/crates/v/saragossa)](https://crates.io/crates/saragossa)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![rust](https://img.shields.io/badge/rust-1.85%2B-orange)
![python](https://img.shields.io/badge/python-zero%20deps-success)


**Pure-Rust Metal inference engine for Apple Silicon.** Raw `metal-rs` kernels,
no Python, no MLX, no CoreML. LLM, speech-to-text and text-to-speech in a single
binary — usable as a library or behind the OpenAI-compatible HTTP server
`saragossa serve`.


## Install

### Homebrew (Apple Silicon)

```sh
brew install azerozero/tap/saragossa # prebuilt binary from the latest stable release
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

## CPU builds on Linux and Windows

Linux and Windows support the CPU reference backend (correctness first, slow
for large models). No GPU backend is provided there. The default `metal`
feature stays enabled for compatibility with macOS, but its dependency and
implementation are compiled only on macOS. Existing Apple Silicon builds keep
their Metal path.

```sh
cargo build
cargo build --no-default-features                     # library only
cargo build --no-default-features --features devtools # library and CLI
cargo test --lib --no-default-features --features devtools
```

On Windows, use TCP explicitly; Unix sockets are unavailable. Start the server
with `SARAGOSSA_API_KEY=TOKEN saragossa serve --backend cpu --port 8090 --model ID=DIR`,
where `DIR` is the local weights directory. For `bench-serve`, use
`--url 127.0.0.1:8090 --api-key TOKEN`. TCP requires bearer authentication;
there is no automatic unauthenticated fallback from a Unix socket.

On Linux, the predictive memory guard reads total host RAM from
`/proc/meminfo` and process RSS from `/proc/self/status`. It does not account
for cgroup limits, swapped pages, or system-wide memory pressure; the pressure
watcher remains macOS-only. On Windows, the system memory guard is disabled
with a warning because no process footprint probe is available. CPU BF16
weights use the existing f32 fallback, which needs more memory than compact
Metal storage.

The checkpoint converter flushes output files before renaming them. On Unix,
it also synchronizes the parent directory. On Windows, directory synchronization
is unavailable through the standard library: file contents are flushed, but
the rename is not guaranteed to survive a power loss.

## Where it fits

Among inference backends (Ollama, llama.cpp, vLLM, SGLang…), saragossa occupies
the **single-user latency × Apple Silicon** quadrant:

- vLLM / SGLang / TGI / LMDeploy require an NVIDIA GPU; here everything is native
  Metal.
- Ollama and llama.cpp are general-purpose and cross-platform; saragossa is tuned
  for **one** target (the Apple Silicon GPU, resident decode). In the
  [November 2026 campaign](bench/competitors/data/campagne-2026-11/README.md) it
  is within a few percent of `mlx-vlm` 0.8.0 with MTP on the 27B 4-bit (ahead
  at 512, behind at 4096) and ahead on the 35B-A3B MoE and the 27B BF16 (see
  [Performance](#performance)). The fair
  comparison is against `mlx-vlm` with a drafter, not against `mlx_lm` in plain
  autoregressive mode, which is the weak baseline.
  (measured; benchmark on your own hardware).
- Optimized for local latency (agents, copilots, voice loops), not multi-tenant
  throughput: the server is single-threaded by design.
- A block-wise warm prefix cache with GPU snapshots (same family of ideas as
  SGLang's RadixAttention): multi-turn sessions don't re-pay for their history.

## Capabilities

| Domain | Details |
|---|---|
| LLM | Qwen3.5 / 3.6 / 3.8 dense and MoE (27B / 30B / 35B-A3B; MLX `affine` quants; `mxfp4`/`nvfp4`/`mxfp8` for rank-2 dense weights), Gemma 4 dense (`gemma4_unified`) and MoE (`gemma4`), plus a generic Llama / Mistral / Gemma 3 loader; u2–u8 (incl. mixed 3/4-bit) quantization at group sizes 32–128, bf16 scales/biases |
| STT | Whisper large-v3-turbo (resident encoder + decoder, bf16 Neural-Accelerator GEMM) |
| TTS | Qwen3-TTS: resident talker + GPU codec + intra-sentence streaming |
| Embeddings | e5-small, pure Rust (CPU) — for semantic memory / RAG |
| Server | Single-threaded multi-model HTTP, LRU model pool, warm cache, OOM guard |

### `saragossa serve` endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | Served models |
| `POST /v1/chat/completions` | OpenAI-compatible chat (SSE or not), `response_format`: `json_object`, `json_schema` |
| `POST /v1/responses` | Stateless text and function calls (JSON or SSE) |
| `POST /v1/messages` | Anthropic Messages shim + `tool_use` (drivable by Claude Code) |
| `POST /v1/audio/transcriptions` | Whisper STT (multipart WAV, `json`/`text`/`verbose_json`, SSE `stream=true`, opt-in `--stt-model`) |
| `POST /v1/audio/transcriptions/stream` | Whisper STT with incremental input (chunked PCM, see [docs/stt-streaming.md](docs/stt-streaming.md)) |
| `POST /v1/audio/speech` | Qwen3 TTS (JSON → WAV, or chunked PCM with `stream: true`; opt-in `--tts-model`; see [Details](#streaming-audio-over-http)) |
| `POST /v1/embeddings` | e5-small embeddings (opt-in `--embed-model`) |

## Design principles

- **GPU residency** — in decode, one token is one command buffer, with no
  readback and no per-layer `commit_and_wait`. The CPU-orchestrated per-op path
  exists only as a fallback.
- **Exactness as a gate** — every optimization is checked against its output
  (end-to-end md5 oracles, STT/TTS goldens); near-tie drifts are qualified (ids +
  top-5 + margin) and recorded — never silent. Greedy speculative decode is exact
  by default except for the cases listed under
  [speculative decode](#quantization-formats-footprint-throughput--quality); the
  strict mode is `SARAGOSSA_RUST_VERIFY_STRICT=1`.
- **Everything is switchable** — each optimized path has an environment
  kill-switch; the flags are centralized in
  [`src/runtime_flags.rs`](src/runtime_flags.rs).

## Performance

Decode throughput, tokens per second, on an M5 Max 128 GiB (AC power, greedy,
256 generated tokens). Each cell is the median of 3 repetitions of the best mode
of that engine (the mode is in parentheses), for prompts of 512, 4096 and 16384
tokens. Source: [`campagne-2026-11/21-final`](bench/competitors/data/campagne-2026-11/21-final)
(`summary.json`, `freeze.json`, full journal), described in the
[campaign README](bench/competitors/data/campagne-2026-11/README.md).

**27B, 4-bit**

| engine | 512 | 4096 | 16384 |
|---|---|---|---|
| saragossa | 55.0 (MTP) | 50.4 (DFLASH) | 45.1 (MTP) |
| mlx-vlm 0.8.0 | 54.7 (MTP) | 55.8 (MTP) | 46.4 (MTP) |
| mlx_lm (AR only) | 30.3 (AR) | 29.7 (AR) | 28.2 (AR) |
| llama.cpp (GGUF Q4_K_M) | 41.8 (DFLASH) | 43.8 (DFLASH) | 39.9 (DFLASH) |

**27B, BF16**

| engine | 512 | 4096 | 16384 |
|---|---|---|---|
| saragossa | 24.7 (DFLASH) | 24.7 (DFLASH) | 23.4 (DFLASH) |
| mlx-vlm 0.8.0 | 20.6 (DFLASH) | 21.1 (DFLASH) | 18.8 (MTP) |
| mlx_lm (AR only) | 9.0 (AR) | 8.9 (AR) | 8.9 (AR) |
| llama.cpp (GGUF Q4_K_M) | 20.0 (DFLASH) | 23.2 (DFLASH) | 19.1 (DFLASH) |

**35B-A3B, 4-bit**

| engine | 512 | 4096 | 16384 |
|---|---|---|---|
| saragossa | 169.7 (MTP) | 162.4 (MTP) | 138.0 (MTP) |
| mlx-vlm 0.8.0 | 153.9 (MTP) | 138.9 (MTP) | 120.6 (MTP) |
| mlx_lm (AR only) | 117.4 (AR) | 115.2 (AR) | 105.8 (AR) |
| llama.cpp (GGUF Q4_K_M) | n/a | n/a | n/a |

Caveats:

- n = 3 per cell, no statistical test; rates are computed from the stream
  delivery times in the journal (`(tokens - 1) / (last - first event)`).
- llama.cpp runs GGUF weights (Q4_K_M for the 4-bit row, a BF16 GGUF for the
  BF16 row), not the MLX weights the other engines load, so the weights are not
  identical. llama.cpp was not measured on the 35B-A3B in this run.
- The `mlx-vlm` 0.8.0 MTP and DFlash outputs are not bit-identical to its own
  AR output; saragossa's greedy outputs are (see below).
- Results hold for this hardware, these models and these prompts; measure on
  your own machine before choosing.

Other measurements, older and on other workloads (2026-07-06, M5 Max):
Whisper turbo STT rtf 0.107; Qwen3-TTS 0.747 end-to-end, streaming TTFA ~1.2 s.

## Quantization formats: footprint, throughput & quality

MLX micro-scaled `mxfp4`, `nvfp4` and `mxfp8` weights are supported for
rank-2 projections and embeddings, with compact QMV and NAX prefill kernels.
See [micro-scale formats and validation](docs/quantization/micro-scales.md)
for layouts, hardware requirements and current limitations.

saragossa loads **MLX affine** integer weights `u2`/`u3`/`u4`/`u6`/`u8`
(group-size 64) and **fp8** (`e4m3`). Trade-offs on a **27B dense** model
(M5 Max, greedy decode T=0; footprint = resident weights; KL = `KL(bf16‖quant)`
averaged over 5 prompts):

| quant | bpw | weight footprint | AR decode³ | quality vs bf16 |
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

³ Earlier single-run AR figures, kept for the relative cost of each format; current 27B 4-bit figures are in [Performance](#performance).

**fp8** loads but a 27B (29 GB) saturates a 24-32 GB Mac — prefer `u6`/`u8` for
high fidelity (smaller, same quality).

**Speculative decode (MTP, DFlash)**: in greedy mode (T=0), saragossa verifies
the draft against the target model, so the output matches the autoregressive
(AR) decode by default, except in the cases below. Current throughput is in
[Performance](#performance). Two default-on kernels of the batched verification
are not bit-identical to the single-row step: the skinny u4 matmul, from 5 rows
(DFlash, lookup, MTP with lookup drafts on long prompts), and the attention
prefix, from 16k tokens of context with a bf16 KV cache. A tiny drift can then
flip a near tie between the top two logits (before the other fixes: 1 request
in 66 on a 32k prompt). `SARAGOSSA_RUST_VERIFY_STRICT=1` (opt-in) makes greedy
verification exact: the attention prefix stays on the exact path, matmuls of 3
to 16 rows use an exact kernel instead of the skinny one, and the DFlash tree
defaults to 3 nodes. Cost on the 27B with DFlash: -4 % at 4k and -12 % at 32k
decode versus the default. Verified states then match the AR bit for bit on the
test corpus; see [docs/perf/greedy-identity.md](docs/perf/greedy-identity.md).
On the **Qwen3.6-35B-A3B** MoE, the official Qwen MTP head (not shipped in
mlx-community weights) is opt-in: point `SARAGOSSA_RUST_MTP_SIDECAR` at its
`mtp.safetensors` (in a subdirectory, not directly in the model root, e.g.
`<model>/mtp-official/`). Greedy only, same AR caveat on near ties; sampled
requests stay on AR. Cost: +0.95 GB peak physical footprint in the CLI (+1.33 GB
on a 424-token prompt), +0.96 GB in serve after a 5,078-token request: the head
is quantized straight into shared Metal buffers, one copy. See
`generate_greedy_mtp_streaming_with_options` and `docs/perf/mtp-pairwise.md`.

## CLI usage

```sh
# Direct LLM generation (the binary requires the `devtools` feature, on by default):
saragossa \
  --model-dir models/Qwen3.6-35B-A3B-oQ8 --backend metal \
  --prompt "Hello" --max-tokens 64 --temperature 0 --metrics

# Offline audio tools (features `denoise` and `diar`, on by default):
saragossa enhance --model dfn3.safetensors --input noisy.wav --output clean.wav
saragossa diarize --model-dir DIR [--format rttm|json] [--cpu] audio.wav
```

## Library usage

Add the crate and drive the engine directly. The main entry points are
`qwen_loader` (LLM), `whisper` (STT), `tts` (TTS) and `text_embedder`
(embeddings); [`src/lib.rs`](src/lib.rs) lists the public API.

## Server: `saragossa serve`

A **single-threaded**, multi-model local HTTP server. The default behavior is
single-user: the warm state is global to a model until a session key is supplied.
The transport is a Unix socket by default (`/tmp/saragossa-serve.sock`, mode
0600); TCP loopback requires a bearer (`SARAGOSSA_API_KEY`; `--api-key` is deprecated because `ps` shows it). A
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

### Responses: text and function calls

`POST /v1/responses` accepts client-provided history without server-side storage
(`store: false`). JSON and SSE preserve the same text/function-call order. Replay
all output items in `input`, then append `function_call_output` items associated
by `call_id`; assistant text between calls and results is accepted.

Tools run in non-strict mode when `strict` is omitted, `null`, or `false`.
Every tool in the returned response explicitly includes `"strict": false`.
`strict: true` returns HTTP 501 with `error.param: "tools[i].strict"` (the
zero-based tool index): constrained argument generation is not implemented.
The supplied schema guides the prompt and parser but is not a generation guarantee.

At the output token limit, the response becomes `incomplete`. Closed items keep
their status; only the text item still open at truncation becomes `incomplete`.

### Structured output (`json_object`, `json_schema`)

`POST /v1/chat/completions` accepts `response_format: {"type":"json_object"}`.
The output is constrained by a byte-level JSON automaton in the sampler:
mandatory root object, strings/escapes/numbers/booleans/null and nested
structures. End-of-turn is only admissible after the root object closes.

Only these guided requests switch to the CPU sampling path (logits read
back, then masked before sampling). Requests without `response_format`, or with
`{"type":"text"}`, keep the existing resident/GPU path.

For schema-constrained content, use:

```json
{"response_format":{"type":"json_schema","json_schema":{
  "name":"greeting", "strict":true,
  "schema":{"type":"object","properties":{"greeting":{"enum":["Bonjour"]}},
            "required":["greeting"],"additionalProperties":false}
}}}
```

`json_schema` uses llguidance CPU masks in both greedy and sampled generation,
including SSE. Anthropic Messages accepts the equivalent
`output_config: {"format":{"type":"json_schema","schema":{...}}}`.
Invalid or unsupported schemas return HTTP 400. `strict` may be omitted, `true`
or `false`; all accepted schemas are enforced. This does not enable strict tool
calling. `stop` is rejected for guided output.

Supported keywords: `type` (including nullable unions), `properties`, `required`,
scalar `enum`, `items`, `additionalProperties`; `title`, `description`, `default`
are annotations. The root must have `type: "object"`. Other keywords are rejected,
including `$ref`, `$defs`, combinators, bounds, `pattern` and `format`.
See [schema output and dependency decision](docs/serve-json-schema.md) for limits,
tokenizer handling and the Rust 1.85 version pin.

JSON limits: the automaton guarantees JSON grammar (surrogate pairs `𐀀`
included) but not a number's magnitude beyond `f64`; in non-streaming mode, an
object left unclosed at the `max_tokens` budget raises an error rather than
returning truncated JSON; in SSE, the deltas stay a valid JSON prefix and an
unclosed object ends with an `error` event of type `incomplete_json`, without the
normal `[DONE]`.

### Sampling and request fields

`POST /v1/chat/completions` honours `temperature`, `top_p`, `top_k` (temperature
first, then top-k, then top-p on the scaled distribution), `seed` (signed i64,
reinterpreted as u64 in two's complement; absent or null defaults to `0`),
`stop` and `max_tokens` / `max_completion_tokens`. Unseeded sampling stays
reproducible.

- `chat_template_kwargs`: only `{"enable_thinking": false}` is accepted, by
  Qwen ChatML and Gemma 4. Native rendering disables thinking; Qwen inserts an
  empty `<think>` block. `true` is refused (HTTP 400): native rendering does not
  reproduce the checkpoint template with thinking on.
  Flint and Gemma 3 have no switch and refuse any explicit value. Any other
  key is refused rather than silently ignored.
- `stream_options: {"include_usage": true}`: like the OpenAI API, a final
  chunk with `"choices": []` and `usage` precedes `data: [DONE]`; all preceding
  chunks carry `"usage": null`. Null or empty options are tolerated, including
  `include_usage: null`; options are ignored for non-streaming requests.
- `min_p`, `presence_penalty`, `frequency_penalty` and `repetition_penalty`
  are not implemented: only their neutral value (`0`, `0`, `0`, `1`) is
  accepted, anything else returns HTTP 400 `invalid_request_error` (permanent
  request error).
- Each completion logs `prompt_tokens=…` on stderr. Setting
  `SARAGOSSA_SERVE_LOG_PROMPT_FINGERPRINT=1` also logs `prompt_fnv1a64=…`
  (FNV-1a 64 of the prompt ids as little-endian `u32`) to verify ingested ids.
  This is **off by default**: the fingerprint can identify a known prompt.
  The competitor benchmark launcher enables it explicitly.

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

An opt-in [persistent SSD prefix cache](docs/serve-prefix-ssd.md) survives RAM
eviction and server restarts. Enable it with `--prefix-cache-ssd-dir DIR` and
`--prefix-cache-ssd-gib 32`; the guide covers integrity, priority eviction and
32k/128k restart measurements.

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

## Details

### Streaming audio over HTTP

CustomVoice accepts `voice` (default: `serena`). An unknown voice (`alloy`,
`nova`...) falls back to Serena and returns `X-Saragossa-TTS-Voice-Fallback`
with the requested name, in WAV and in streaming alike. The cause is logged on
stderr. VoiceDesign and CustomVoice use the snapshot's sampling with a
deterministic seed; `seed` selects it per request. See the
[TTS generation policy](docs/tts-generation.md) for the parameters, the greedy
flag of the oracles and the stop reasons.

`POST /v1/audio/speech` returns the full WAV by default (`stream` absent or
`false`). The explicit extension `{"input":"Hello.","stream":true}` emits each
batch as soon as it is decoded, over HTTP/1.1 `Transfer-Encoding: chunked`,
without `Content-Length`. The de-chunked body is **raw signed 16-bit
little-endian mono PCM, with no WAV header** (`Content-Type: audio/pcm`). The
codec's native rate is given by `X-Saragossa-Audio-Sample-Rate` (usually
24000 Hz); each sample uses the same conversion as the WAV. `model` is still
accepted and has no effect. `stream: true` alone selects this PCM format.

The existing audio cap and `SARAGOSSA_TTS_MAX_FRAMES` apply to both modes. In
streaming, `X-Saragossa-TTS-Frame-Limit-Reached` is an **HTTP trailer**,
announced by `Trailer` and sent after the zero chunk; in WAV it stays a header.
`X-Saragossa-TTS-Stop-Reason` follows the same transport and distinguishes
`eos`, `repeat` and `cap`. A client that ignores trailers does not get this
information. An error after the response has started closes the stream without a
terminal chunk: the client must treat that body as incomplete. A disconnect
stops synthesis at the next failed write. The engine stays serial, with the
existing socket backpressure and write timeout; no detached audio producer is
created.

Codec streaming uses fixed batches of **4 frames** (about 320 ms), adjustable
with `SARAGOSSA_TTS_STREAM_LOT`, only when its CPU and GPU contexts are truly
incremental (VoiceDesign and x-vector cloning). The CPU fallback, including
`SARAGOSSA_TTS_CODEC_GPU=0`, and ICL cloning re-decode the prefix: their
thresholds stay doubled after the first batch. Fixed batches avoid doubling the
generation wait between emissions; they guarantee neither a constant attention
cost over its history nor an RTF below 1.

### Full-weight NA dequantization limit

The NA matmul path that materializes a full bf16 weight is limited to 512 MiB
per weight by `SARAGOSSA_RUST_QMM_NA_DEQUANT_MAX_BYTES` (bytes). Above the
limit, selection falls back to qmv before allocating the scratch. Set it to `0`
to disable this dequantization path; an absent or invalid value uses the
default. Fused NA kernels do not materialize that weight and keep their existing
selection rules. This currently applies to `QmmNaU8`; u5/u6 have no full-weight
NA dequantization path in this revision.

### Modular compilation

All modalities are enabled by default. The [feature table](docs/features.md)
describes the minimal LLM and STT binaries, the server's 501 responses and the
compile-time checks. The [TOML profiles](docs/design/configuration.md) are a
design proposal with no implementation at this point.
