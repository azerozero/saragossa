#!/usr/bin/env python3
"""Tests de la recopie de `tokenizer.json`, stdlib seule.

    python3 -I -B test_tokenizer_sidecar.py
"""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tokenizer_sidecar import check_tokenizer, copy_tokenizer, pretokenizer_regexes  # noqa: E402

QWEN35 = r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
QWEN2 = r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"


def spec(regex):
    return {"pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated"},
        {"type": "ByteLevel", "add_prefix_space": False}]}, "model": {"vocab": {"a": 0}}}


def fails(call, needle):
    try:
        call()
    except SystemExit as error:
        assert needle in str(error), error
        return
    raise AssertionError(f"échec attendu : {needle}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = pathlib.Path(tmp, "src"), pathlib.Path(tmp, "dst")
        src.mkdir()
        dst.mkdir()
        (src / "tokenizer.json").write_text(json.dumps(spec(QWEN35), indent=2, ensure_ascii=False))
        assert pretokenizer_regexes(src / "tokenizer.json") == [QWEN35]
        copy_tokenizer(src, dst)
        assert (dst / "tokenizer.json").read_bytes() == (src / "tokenizer.json").read_bytes()
        # Réécriture façon `Qwen2Tokenizer.save_pretrained` : regex Qwen2.
        (dst / "tokenizer.json").write_text(json.dumps(spec(QWEN2)))
        fails(lambda: check_tokenizer(src, dst), "regex du pré-tokenizer réécrite")
        # Même regex mais fichier reformaté : refusé aussi.
        (dst / "tokenizer.json").write_text(json.dumps(spec(QWEN35)))
        fails(lambda: check_tokenizer(src, dst), "octet à octet")
        (dst / "tokenizer.json").unlink()
        fails(lambda: check_tokenizer(src, dst), "absent de la sortie")
        (src / "tokenizer.json").unlink()
        fails(lambda: check_tokenizer(src, dst), "absent de la source")
    print("tokenizer_sidecar: ok")


if __name__ == "__main__":
    main()
