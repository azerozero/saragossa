"""CPU reference: transformers 5.22.2, torch 2.15.0.
pip install transformers==5.22.2 torch==2.15.0

The local snapshots lack preprocessor/generation configs. Use Whisper's default
80-bin, 16 kHz frontend; explicit French/task IDs come from the local tokenizer.
Capture non-timestamp greedy separately from timestamp generation. In 5.22.2,
None suppression settings fall back to the local model's generation defaults;
Rust's raw greedy decoder does not apply these processors.
"""
import argparse
import copy
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, local_model, provenance, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["whisper_tiny", "whisper_small"], required=True)
    args = parser.parse_args()
    import torch
    from transformers import (PreTrainedTokenizerFast, WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, GenerationConfig)
    from transformers.generation.utils import GenerationMixin
    torch.set_num_threads(4)
    path = local_model(args.model)
    audio_path = ROOT / "tests/golden/whisper_samples16k.bin"
    raw = audio_path.read_bytes()
    samples = struct.unpack(f"<{len(raw)//4}f", raw)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"))
    tok = tokenizer.convert_tokens_to_ids
    model = WhisperForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.float32, attn_implementation="eager").cpu().eval()
    frontend = WhisperFeatureExtractor(feature_size=80, sampling_rate=16000, hop_length=160,
                                       chunk_length=30, n_fft=400)
    inputs = frontend(list(samples), sampling_rate=16000, return_tensors="pt")
    prefix = [tok("<|startoftranscript|>"), tok("<|fr|>"), tok("<|transcribe|>"), tok("<|notimestamps|>")]
    cfg = GenerationConfig.from_model_config(model.config)
    cfg.forced_decoder_ids = None
    cfg.suppress_tokens = None
    cfg.begin_suppress_tokens = None
    cfg.do_sample = False
    cfg.num_beams = 1
    cfg.max_new_tokens = 128
    with torch.inference_mode():
        raw_ids = GenerationMixin.generate(model, inputs.input_features, generation_config=cfg,
                                          decoder_input_ids=torch.tensor([prefix]))[0].tolist()
        tokens = raw_ids[len(prefix):]
        if not tokens or tokens[-1] != cfg.eos_token_id:
            raise RuntimeError("incomplete raw greedy transcription (no EOS)")
        tokens = tokens[:-1]
        timestamp_cfg = copy.deepcopy(cfg)
        timestamp_cfg.no_timestamps_token_id = tok("<|notimestamps|>")
        timestamp_cfg.lang_to_id = {"<|fr|>": tok("<|fr|>")}
        timestamp_cfg.task_to_id = {"transcribe": tok("<|transcribe|>")}
        timestamp_cfg.is_multilingual = True
        timestamp_cfg.max_initial_timestamp_index = 50
        timestamp_cfg.suppress_tokens = model.config.suppress_tokens
        timestamp_cfg.begin_suppress_tokens = model.config.begin_suppress_tokens
        timed = model.generate(inputs.input_features, generation_config=timestamp_cfg,
                               language="fr", task="transcribe", return_timestamps=True,
                               return_segments=True)
    segments = []
    for segment in timed["segments"][0]:
        ids = segment["tokens"].tolist()
        segments.append(dict(start=float(segment["start"]), end=float(segment["end"]),
                             tokens=ids, text=tokenizer.decode(ids, skip_special_tokens=True).strip()))
    if not segments:
        raise RuntimeError("timestamp reference returned no segments")
    source = provenance(args.model, path, ["transformers", "torch"])
    source["audio"] = dict(path="tests/golden/whisper_samples16k.bin", sha256=sha(audio_path),
                           sample_rate=16000, samples=len(samples))
    write_json("whisper", args.model, dict(language="fr", prefix=prefix, tokens=tokens,
        text=tokenizer.decode(tokens, skip_special_tokens=True).strip(), segments=segments,
        timestamp_tokens=timed["sequences"][0].tolist()), source,
        "text/tokens exact; timestamps +/- 0.02s (one Whisper tick); Rust timestamp API unavailable",
        frontend=dict(feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400),
        decoding="float32 CPU eager; None suppression resolves to local model defaults in transformers 5.22.2; timestamps constrained; French forced")


if __name__ == "__main__":
    main()
