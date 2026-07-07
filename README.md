# Saragossa

**A fast, pure-Rust LLM inference engine for Apple Silicon — ahead of Apple's own `mlx_lm` on the flagship MoE model.**

Saragossa is a from-scratch transformer decoder written in Rust on top of `metal-rs`. No MLX, no PyTorch, no Python — a single self-contained crate that compiles its Metal kernels at runtime. It was extracted from a real-time local voice assistant, where decode latency is everything.

## Why

On Apple Silicon, `mlx_lm` (Apple's MLX-based generation library) is the reference. Same-session A/B, identical protocol both sides (decode tok/s, 2026-07-07):

**Rig**: MacBook Pro, Apple M5 Max (40-core GPU, 18-core CPU), 128 GB unified memory, macOS 26.5.1 · `mlx` 0.31.2 / `mlx_lm` 0.31.3 · 1k-token prompt, 512 generated tokens, greedy · runs GPU-serialized back-to-back (Saragossa then `mlx_lm` per pair) · GPU temperature sampled with [macmon](https://github.com/vladkens/macmon) at each run's start→end.

| Model | Saragossa (GPU °C) | `mlx_lm` (GPU °C) | Verdict |
|---|---|---|---|
| [Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) (MoE, default) | **146.0** (41→48) | 137.8 (48→58) | **+6 %, and cooler** |
| [Qwen3.6-35B-A3B-oQ8](https://huggingface.co/bearzi/Qwen3.6-35B-A3B-oQ8) (OptiQ 8-bit) @32k ctx, sampled T>0 | **89.8** | 88.1 | ahead at long context (bf16 KV; measured 2026-07-03) |
| [Qwen3.6-35B-A3B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit) (MoE, mixed-precision) | **116.3** (43→51) | 114.6 (51→64) | +1.5 % |
| [Qwen3.6-27B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-OptiQ-4bit) (dense) | 27.0 (65→81) | 26.9 (81→72) | parity |
| [Qwen3-30B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit) (MoE) | 112.7 (56→59) | 131.4 (59→68) | **−14 % — under investigation** (measured +11.5 % ahead in June; suspected routing/default drift on our side or `mlx_lm` progress) |

Numbers move with every engine and `mlx_lm` release — reproduce with `--prompt-tokens 1024 --max-tokens 512 --temperature 0 --metrics` vs `mlx_lm.benchmark -p 1024 -g 512`.

Output quality holds: on the 35B it produces fluent, literary prose where `mlx_lm` often drifts into meta-planning.

## How it's fast

- **Resident decode** — one Metal command buffer per token (no per-op host round-trips).
- **GPU sampler** — `top_k`/`top_p`/temperature sampling stays on-device; no full-vocab logits readback (a ~6× win at production temperatures vs. a CPU sampler).
- **Hand-written quantized kernels** — `affine` 4-bit (group-size 64) `qmv`/`qmm`, aligned fast-path, `bf16` scales/biases — the codegen levers that close the gap with Apple's metallib.
- **Quantized KV-cache** (optional) — u8 K+V Flash kernel for GQA, growing wins at long context.
- **MoE-aware** — routed + shared experts, resident.
- **Prefill kernels** — Neural-Accelerator GEMM paths plus dedicated causal-attention
  kernels (GQA-tiled and a Steel-derived d256 variant): 35B prefill runs 1.0 s @2k,
  3.5 s @8k, 23 s @32k.

## Format support

- **Quantization**: MLX `affine` 4-bit (incl. OptiQ mixed-precision), `bf16`, and FP8 (`e4m3`/`e5m2`, dequantized).
- **Models today**: the Qwen3.6 family (MoE + dense), plus a generic loader for the
  mainstream SwiGLU + RMSNorm + RoPE + GQA family — Llama, Mistral and Gemma 3 load
  and run at `mlx_lm` parity.
- **Beyond LLMs**: the same engine runs Whisper large-v3-turbo (STT, resident
  encoder+decoder) and Qwen3-TTS (talker + GPU codec, intra-sentence streaming).
- **Serving**: `saragossa serve` exposes an OpenAI-compatible HTTP endpoint
  (multi-model registry, Unix socket by default, bearer-gated TCP, read timeouts,
  `max_tokens` cap).
- Loads MLX-format `safetensors` (not GGUF).

## Build & run

Requires macOS + the Metal toolchain.

```sh
cargo build --release --features metal,devtools
./target/release/saragossa \
  --model-dir /path/to/mlx-model \
  --prompt "Explain why the sky is blue." \
  --max-tokens 128 --temperature 0.7 --top-k 20 --backend metal

# OpenAI-compatible server (multi-model).
./target/release/saragossa serve --model my-model=/path/to/mlx-model
```

## Status

Experimental, extracted from a production voice agent. The core is reference-grade (zero `unwrap`/`panic` in hot paths, colocated tests, `#![deny(unsafe_code)]` outside the audited Metal FFI). APIs may move.

## License

Dual-licensed under **MIT OR Apache-2.0**. The Metal kernels are an independent Rust reimplementation of algorithms from Apple's [MLX](https://github.com/ml-explore/mlx) (MIT) — see [`NOTICE`](NOTICE).
