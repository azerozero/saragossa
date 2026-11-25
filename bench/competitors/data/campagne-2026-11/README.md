# Competitor benchmark, campaigns of 19, 20 and 21 November 2026, M5 Max 128 GiB

Raw data produced by the harness [`bench/competitors/campaign.py`](../../campaign.py)
(a self-contained campaign run with a single command), with methodological notes.

| folder | date | content |
|---|---|---|
| `19b/` | Nov 19 | six engines (saragossa, llama.cpp, mlx-lm, llmd, ollama, mistral.rs); 27B 4-bit, 35B-A3B 4-bit, 27B BF16; saragossa binary predating the fixes; 90 entries, 270 planned measurements of which 243 recorded (234 succeeded, 9 failed; 6 load failures, 3 skipped entries) |
| `20-sar/` | Nov 20 | saragossa only, `main` d894441; 18 entries, 54 measurements |
| `20-dfl/` | Nov 20 | saragossa 27B BF16 in AR, MTP and DFlash, `main` 0af2102, after PR #305; 9 entries, 27 measurements |
| `21-mlx/` | Nov 21 | mlx-lm and mlx-vlm in AR, MTP and DFlash; 27 entries, 81 measurements out of 81 |
| `21-final/` | Nov 21 | saragossa, mlx and llama.cpp in the same session, `main` 7cb2f3e (PR #309 and #311, build finished at 12:55:32 per `freeze.json` and the `saragossa_build` event); 72 entries, 216 measurements out of 216 |

Per folder:

- `journal.jsonl.gz`: full journal (loads, requests, probes, sensors);
- `freeze.json`: frozen settings, versions, sha256 of binaries and weights;
- `summary.json`, `DONE`, `campaign.out` (19b only): campaign summary and output;
- `logs.tar.gz`: server output, one file per entry and repetition;
- `prompts/*.json`: prompt fingerprints (see below).

Personal paths are replaced by `~`, the session temporary folder by `<scratch>`,
and the battery identifier is removed from `power.source_raw`.

## Protocol

M5 Max 128 GiB, on AC power, 3 planned repetitions per cell, greedy decoding
(temperature 0), 256 generated tokens, target prompt lengths of 512, 4096 and
16384 tokens (the longest actual prompt has 16383), context 20480, load gate
`loadavg < 2`, cooldown between entries (at least 45 s, then until GPU
≤ 50 °C, at most 300 s).

## Prompts

The prompt text is not published. `prompts/<model>.json` keeps, per target
length, `prompt_tokens` and the `sha256` of the content. `campaign.py` rebuilds
them with `build_prompts`: the sorted concatenation of the `*.rs` files in the
`--prompt-source` folder (default: the repository's `src/`; `freeze.json` gives
the path and `repo_commit`), special tokens removed, then the longest prefix that
fits the target once the model's chat template is applied (run with the mlx-lm
venv, `transformers`). Compare the sha256 values with the published ones; the
prompt depends on the commit of the sources.

## Caveats

- n = 3 planned per cell (fewer where runs failed), no statistical test.
- Different quantizations across engines (MLX 4-bit versus GGUF Q4_K_M, etc.):
  the throughputs do not compare identical weights.
- Apart from `21-mlx` (mlx) and `21-final` (mlx and llama.cpp), competitors were
  measured only on Nov 19; `20-sar` and `20-dfl` cover saragossa only. `21-final`
  re-measures saragossa, mlx and llama.cpp in the same session.
- mlx-mtp and mlx-dflash go through mlx-vlm 0.8.0; the mlx AR rows (`mlx-ar`) go
  through mlx_lm.server. mlx-vlm without a drafter was not measured, so the AR rows
  do not establish the best non-speculative mlx speed.
- This campaign does not test greedy identity between AR and speculative outputs:
  each request gets its own nonce (see Limits), so outputs of different
  configurations are not compared on identical inputs. saragossa's identity is
  covered by separate tests, listed in
  [`docs/perf/greedy-identity.md`](../../../../docs/perf/greedy-identity.md).
- llama.cpp was not measured on the 35B in `21-final`.
- `19b`, mistral.rs DFlash on the 27B 4-bit: throughput falls with prompt length
  (medians 28.7, 13.8 and 6.2 tok/s at 512, 4096 and 16384, computed as
  255 / (time of last non-empty chunk − time of first non-empty chunk)). The cause is not
  established; the nine measurements record `memory_pressure: normal` before and
  after.
- `19b`, llmd: the configuration with `--speculative-drafter-model=native` failed
  to load on the 27B 4-bit and the 27B BF16 (llmd AR also failed to load on the
  BF16), and the 35B has no local checkpoint readable by llmd per `freeze.json`.
  Only llmd AR on the 27B 4-bit was measured; this says nothing general about
  llmd's speculative support.
- `20-dfl/freeze.json` reports battery power, while the 9 `entry_start` events
  of the journal record `on_ac: true`. The freeze was probably written by an
  earlier launch; that launch is not in the published archives.

## Limits

- Order not counterbalanced: in `21-final`, all competitor entries ran first,
  then the saragossa entries, after its build.
- The nonce prefixed to each prompt is derived from the entry (model,
  configuration, repetition) and the length: inputs differ by a few tokens across
  configurations.
- One prompt corpus: a technical review of the repository's Rust sources. No
  dedicated conversational, multilingual, reasoning or tool-use evaluation; no
  sampling and no concurrent load.
- At most 3 observations per cell: medians and spreads are descriptive.
- The DFlash budget and MTP measurements of PR #314 and #316 are outside this
  campaign; their raw repetitions are not published here, so they cannot be
  audited from this folder.
