"""Reference helpers: Python 3, stdlib only; no pip install required."""

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/fixtures/goldens"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = {
    "qwen38": ("SARAGOSSA_QWEN38_DIR", "Qwen3.8-27B-u4ls", None),
    "qwen36_moe": ("SARAGOSSA_QWEN36_MOE_DIR", "Qwen3.6-35B-A3B-4bit", None),
    "gemma3": ("SARAGOSSA_GEMMA3_DIR", None, "mlx-community/gemma-3-4b-it-4bit"),
    "gemma4": ("SARAGOSSA_GEMMA4_DIR", None, "mlx-community/gemma-4-26b-a4b-it-4bit"),
    "qwen3_tts": ("SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR", None,
                  "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit"),
    "whisper_tiny": ("SARAGOSSA_WHISPER_TINY_DIR", None, "openai/whisper-tiny"),
    "whisper_small": ("SARAGOSSA_WHISPER_SMALL_DIR", None, "openai/whisper-small"),
    "embedder": ("SARAGOSSA_EMBEDDER_DIR", None, "intfloat/multilingual-e5-small"),
}


def local_model(name):
    if os.environ.get("SARAGOSSA_GOLDEN_CAPTURE") != "1":
        raise RuntimeError("Set SARAGOSSA_GOLDEN_CAPTURE=1 before loading a reference model")
    env, folder, repo = MODELS[name]
    if env in os.environ:
        path = Path(os.environ[env]).expanduser()
    elif folder:
        path = Path.home() / "workspace/ai_model" / folder
    else:
        hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
        snapshots = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
        candidates = sorted(p for p in snapshots.glob("*") if (p / "config.json").is_file())
        if len(candidates) != 1:
            raise RuntimeError(f"Set {env}: expected one local snapshot, found {len(candidates)}")
        path = candidates[0]
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Set {env} to a local model directory")
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(name, path, packages):
    env, folder, repo = MODELS[name]
    files = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "chat_template.json", "generation_config.json"]
    return {
        "model": repo or folder,
        # Portable path, never a user's absolute path or a temporary interpreter.
        "path": f"${{{env}}}",
        "snapshot": path.name if path.parent.name == "snapshots" else None,
        "sha256": {f: sha(path / f) for f in files if (path / f).is_file()},
        "versions": {p: importlib.metadata.version(p) for p in packages},
    }


def write(name, stem, data, dtype, shape, source, tolerance, **metadata):
    if os.environ.get("SARAGOSSA_GOLDEN_CAPTURE") != "1":
        raise RuntimeError("Set SARAGOSSA_GOLDEN_CAPTURE=1 to capture external references")
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = data if isinstance(data, bytes) else struct.pack(f"<{len(data)}f", *data)
    header = dict(dtype=dtype, shape=shape, source=source, tolerance=tolerance,
                  mlx_sha="not-applicable", captured=datetime.date.today().isoformat(),
                  payload_sha256=hashlib.sha256(payload).hexdigest(), **metadata)
    (directory / f"{stem}.bin").write_bytes(payload)
    (directory / f"{stem}.json").write_text(
        json.dumps(header, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{name}/{stem}: {len(payload)} bytes")


def write_json(name, stem, data, source, tolerance, **metadata):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    write(name, stem, payload, "utf8", [len(payload)], source, tolerance, **metadata)
