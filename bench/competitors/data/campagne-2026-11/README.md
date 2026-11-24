# Competitor benchmark, campaigns of 19, 20 and 21 November 2026, M5 Max 128 GiB

Raw data produced by the harness [`bench/competitors/campaign.py`](../../campaign.py)
(a self-contained campaign run with a single command). No analysis is published here.

| folder | date | content |
|---|---|---|
| `19b/` | Nov 19 | six engines (saragossa, llama.cpp, mlx-lm, llmd, ollama, mistral.rs); 27B 4-bit, 35B-A3B 4-bit, 27B BF16; saragossa binary predating the fixes; 90 entries, 270 planned measurements of which 243 recorded (234 succeeded, 9 failed; 6 load failures, 3 skipped entries) |
| `20-sar/` | Nov 20 | saragossa only, `main` d894441; 18 entries, 54 measurements |
| `20-dfl/` | Nov 20 | saragossa 27B BF16 in AR, MTP and DFlash, `main` 0af2102, after PR #305; 9 entries, 27 measurements |
| `21-mlx/` | Nov 21 | mlx-lm and mlx-vlm in AR, MTP and DFlash; 27 entries, 81 measurements out of 81 |
| `21-final/` | Nov 21 | saragossa, mlx and llama.cpp in the same session, `main` 7cb2f3e (PR #309 and #311, built at 12:31); 72 entries, 216 measurements out of 216 |

Per folder:

- `journal.jsonl.gz`: full journal (loads, requests, probes, sensors);
- `freeze.json`: frozen settings, versions, sha256 of binaries and weights;
- `summary.json`, `DONE`, `campaign.out` (19b only): campaign summary and output;
- `logs.tar.gz`: server output, one file per entry and repetition;
- `prompts/*.json`: prompt fingerprints (see below).

Personal paths are replaced by `~`, the session temporary folder by `<scratch>`,
and the battery identifier is removed from `power.source_raw`.

## Protocol

M5 Max 128 GiB, on AC power, 3 repetitions per cell, greedy decoding
(temperature 0), 256 generated tokens, prompts of 512, 4096 and 16384 tokens,
context 20480, load gate `loadavg < 2`, cooldown between entries (45 s, at most
5 min between models).

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

- n = 3 per cell, no statistical test.
- Different quantizations across engines (MLX 4-bit versus GGUF Q4_K_M, etc.):
  the throughputs do not compare identical weights.
- Apart from `21-mlx` (mlx) and `21-final` (mlx and llama.cpp), competitors were
  measured only on Nov 19; `20-sar` and `20-dfl` cover saragossa only. `21-final`
  re-measures saragossa, mlx and llama.cpp in the same session.
- mlx-mtp and mlx-dflash go through mlx-vlm 0.8.0, whose output is not
  bit-identical to its own AR output, whereas saragossa's is exact by default
  except for the cases listed in
  [`docs/perf/greedy-identity.md`](../../../../docs/perf/greedy-identity.md).
- mlx-vlm without a drafter is faster than mlx_lm.server.
- llama.cpp was not measured on the 35B in `21-final`.
- TTFT at 512 tokens in BF16 is bimodal.
- mistral.rs in speculative mode degrades at long context (memory pressure).
- llmd has neither MTP nor DFlash.
- `20-dfl/freeze.json` reports battery power: the freeze dates from a first launch
  stopped by the power gate before any measurement. The resumed run measured on AC
  power, as the 9 `on_ac: true` entries of the journal show.
