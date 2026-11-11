#!/usr/bin/env python3
"""Generate CPU-only references from local tokenizer files; never downloads.

Install: pip install tokenizers==0.22.2
Usage: python -I tools/tokenizer/make_goldens.py FAMILY LOCAL_PATH
LOCAL_PATH is tokenizer.json, or the vocab.json/merges.txt directory for qwen3-tts.
Only text, token IDs, a digest and public provenance are written to the repository.
decoded = decode(ids, skip_special_tokens=True), including for literal specials.
TTS digest = SHA256(vocab.json bytes followed by merges.txt bytes), in that order.

Rust tests accept SARAGOSSA_TOKENIZER_QWEN3, _GEMMA3, _GEMMA4, _WHISPER
and _QWEN3_TTS with the same path convention, or search the local HF cache.
An absent local fixture skips with a message; an explicit override takes priority.
"""

import argparse
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/tokenizer"
SOURCES = {
    "qwen3": "local:Qwen3.8-27B-u4ls",
    "gemma3": "mlx-community/gemma-3-4b-it-4bit@93724907d4ed1745d2fe50baadf3b0b01a65abf2",
    "gemma4": "mlx-community/gemma-4-26b-a4b-it-4bit@efbeee6e582ebfd06abc9d65e90839c4b5d2116b",
    "whisper": "openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d",
    "qwen3-tts": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit@ffc6545dc9cb086950aa46c6cd3db490e6ece3e1",
}


def load_tts(path):
    # Mirror src/tts/tokenizer.rs's fallback, including AddedToken defaults:
    # non-special AddedToken instances normalize, special ones do not.
    tokenizer = Tokenizer(models.BPE.from_file(str(path / "vocab.json"), str(path / "merges.txt")))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    special = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    ordinary = [
        "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
        "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
        "<tool_response>", "</tool_response>", "<think>", "</think>",
    ]
    audio = [
        "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>",
        "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
    ]
    # Python add_special_tokens forces special=True even on AddedToken objects.
    # Rust add_special_tokens delegates to add_tokens and preserves each flag.
    tokenizer.add_tokens(
        [AddedToken(s, special=True, normalized=False) for s in special]
        + [AddedToken(s, special=False, normalized=True) for s in ordinary]
        + [AddedToken(s, special=True, normalized=False) for s in audio]
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=SOURCES)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if tokenizers.__version__ != "0.22.2":
        parser.error("requires tokenizers==0.22.2")
    if args.family == "qwen3-tts":
        if (args.path / "tokenizer.json").exists():
            parser.error("TTS fallback requires a directory without tokenizer.json")
        files = [args.path / "vocab.json", args.path / "merges.txt"]
        tokenizer = load_tts(args.path)
    else:
        files = [args.path]
        tokenizer = Tokenizer.from_file(str(args.path))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    rows = [{"tokenizer_sha256": digest.hexdigest(), "source": SOURCES[args.family],
             "tokenizers_version": tokenizers.__version__}]
    corpus = [json.loads(line)["text"] for line in (FIXTURES / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    if not corpus:
        parser.error("empty corpus")
    for text in corpus:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        rows.append({"text": text, "ids": ids,
                     "ids_special": tokenizer.encode(text, add_special_tokens=True).ids,
                     "decoded": tokenizer.decode(ids, skip_special_tokens=True)})
    output = FIXTURES / f"{args.family}.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                              for row in rows), encoding="utf-8")
    print(f"{args.family}: {len(corpus)} entries, {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
