# Saragossa

**A fast, pure-Rust LLM inference engine for Apple Silicon — and it beats Apple's own `mlx_lm` on Mixture-of-Experts models.**

Saragossa is a from-scratch transformer decoder written in Rust on top of `metal-rs`. No MLX, no PyTorch, no Python — a single self-contained crate that compiles its Metal kernels at runtime. It was extracted from a real-time local voice assistant, where decode latency is everything.

## Why

On Apple Silicon, `mlx_lm` (Apple's MLX-based generation library) is the reference. Saragossa **matches or beats it** on the models that matter, measured cold (60 s cooldown, decode tok/s):

| Model | Saragossa | `mlx_lm` | Verdict |
|---|---|---|---|
| **Qwen3.6-35B-A3B** (MoE, default) | ~133 tok/s | ~110 | **+13–21 %** |
| **Qwen3.6-30B-A3B** (MoE) | ~133 tok/s | ~119 | **+11.5 %** |
| **Qwen3.6-27B-OptiQ** (dense) | 24.3 tok/s | 24.9 | parity (0.97×) |

Output quality holds: on the 35B it produces fluent, literary prose where `mlx_lm` often drifts into meta-planning.

## How it's fast

- **Resident decode** — one Metal command buffer per token (no per-op host round-trips).
- **GPU sampler** — `top_k`/`top_p`/temperature sampling stays on-device; no full-vocab logits readback (a ~6× win at production temperatures vs. a CPU sampler).
- **Hand-written quantized kernels** — `affine` 4-bit (group-size 64) `qmv`/`qmm`, aligned fast-path, `bf16` scales/biases — the codegen levers that close the gap with Apple's metallib.
- **Quantized KV-cache** (optional) — u8 K+V Flash kernel for GQA, growing wins at long context.
- **MoE-aware** — routed + shared experts, resident.

## Format support

- **Quantization**: MLX `affine` 4-bit (incl. OptiQ mixed-precision), `bf16`, and FP8 (`e4m3`/`e5m2`, dequantized).
- **Models today**: the Qwen3.6 family (MoE + dense). A generic loader for the mainstream SwiGLU + RMSNorm + RoPE + GQA family (Llama, Mistral, Mixtral) is in progress — the building blocks and config are already architecture-agnostic.
- Loads MLX-format `safetensors` (not GGUF).

## Build & run

Requires macOS + the Metal toolchain.

```sh
cargo build --release --features metal
./target/release/saragossa \
  --model-dir /path/to/mlx-model \
  --prompt "Explain why the sky is blue." \
  --max-tokens 128 --temperature 0.7 --top-k 20 --backend metal
```

## Status

Experimental, extracted from a production voice agent. The core is reference-grade (zero `unwrap`/`panic` in hot paths, colocated tests, `#![deny(unsafe_code)]` outside the audited Metal FFI). APIs may move.

## License

Dual-licensed under **MIT OR Apache-2.0**. The Metal kernels are an independent Rust reimplementation of algorithms from Apple's [MLX](https://github.com/ml-explore/mlx) (MIT) — see [`NOTICE`](NOTICE).
