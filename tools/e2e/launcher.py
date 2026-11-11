"""Lifecycle of one local serve process; never resolve remote model identifiers."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from contracts import require

TARGETS = {
    "qwen38": ("E2E_QWEN38_MODEL", "chat"),
    "qwen36_moe": ("E2E_QWEN36_MOE_MODEL", "chat"),
    "gemma3": ("E2E_GEMMA3_MODEL", "chat"),
    "gemma4": ("E2E_GEMMA4_MODEL", "chat"),
    "e5": ("E2E_E5_MODEL", "embed"),
    "whisper": ("E2E_WHISPER_MODEL", "stt"),
    "qwen3_tts": ("E2E_QWEN3_TTS_MODEL", "tts"),
    "audio": ("E2E_QWEN3_TTS_MODEL", "audio"),
}


def local_path(variable, directory=True):
    value = os.environ.get(variable)
    require(bool(value), f"variable requise: {variable}")
    path = Path(value).expanduser().resolve()
    require(path.is_dir() if directory else path.is_file(), f"chemin local invalide: {variable}")
    return path


def configuration(target, mode):
    variable, family = TARGETS[target]
    path = local_path(variable)
    args, paths = [], {variable: path}
    flint = False
    if family == "chat":
        config = json.loads((path / "config.json").read_text())
        flint = config.get("model_type") == "flint"
        args = ["--model", f"{target}={path}"]
    else:
        flag = {"embed": "--embed-model", "stt": "--stt-model",
                "tts": "--tts-model", "audio": "--tts-model"}[family]
        args = [flag, str(path)]
        if family == "audio":
            stt = local_path("E2E_WHISPER_MODEL")
            args += ["--stt-model", str(stt)]
            paths["E2E_WHISPER_MODEL"] = stt
    require(mode == "ar" or family == "chat", "MTP/DFlash réservés aux profils chat")
    require(mode == "ar" or target in {"qwen38", "qwen36_moe"},
            "MTP/DFlash non qualifiés pour ce profil")
    # Avoid inherited performance/acceptance flags changing the experiment.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SARAGOSSA_", "RETI_"))}
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_OFFLINE="1")
    env["SARAGOSSA_RUST_MTP_DECODE"] = "0" if mode == "ar" else "1"
    if mode != "ar":
        sidecar_var = "E2E_" + target.upper() + "_MTP_SIDECAR"
        sidecar = local_path(sidecar_var, directory=False)
        paths[sidecar_var] = sidecar
        env["SARAGOSSA_RUST_MTP_SIDECAR"] = str(sidecar)
        env["SARAGOSSA_RUST_MTP_MAX_DRAFT"] = "2"
    if mode == "dflash":
        draft_var = "E2E_" + target.upper() + "_DFLASH_DIR"
        draft = local_path(draft_var)
        paths[draft_var] = draft
        env["SARAGOSSA_RUST_DFLASH_DRAFT_DIR"] = str(draft)
    return family, flint, args, env, paths


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


@contextmanager
def server(binary, args, env, startup_timeout):
    # Unique socket + stdin=DEVNULL avoid the interactive model picker.
    with tempfile.TemporaryDirectory(prefix="saragossa-e2e-") as temporary:
        log_path = Path(temporary) / "serve.log"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        key = "e2e-" + os.urandom(16).hex()
        base = f"http://127.0.0.1:{port}"
        command = [str(binary), "serve", "--host", "127.0.0.1", "--port", str(port),
                   "--socket", str(Path(temporary) / "serve.sock"), "--api-key", key,
                   "--max-tokens-cap", "1024", *args]
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic() + startup_timeout
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                while True:
                    require(process.poll() is None, "serve arrêté avant /health")
                    try:
                        with opener.open(base + "/health", timeout=1) as response:
                            ready = response.status == 200 and json.load(response) == {"status": "ok"}
                        # A port race must not test an unrelated pre-existing server.
                        request = urllib.request.Request(base + "/v1/models",
                            headers={"Authorization": "Bearer " + key})
                        with opener.open(request, timeout=1) as response:
                            ready = ready and response.status == 200
                        if ready:
                            break
                    except (OSError, urllib.error.URLError, ValueError):
                        pass
                    require(time.monotonic() < deadline, "timeout attente /health")
                    time.sleep(0.1)
                yield base, key, log_path, process
            finally:
                stop(process)


def verify_mode(mode, log_path, model):
    log = log_path.read_text(errors="replace")
    mtp = f"saragossa serve MTP actif model={model} " in log
    dflash = f"saragossa serve drafter DFlash actif model={model} " in log
    require(mtp == (mode != "ar"), "mode MTP demandé non confirmé (ou actif en AR)")
    require(dflash == (mode == "dflash"), "mode DFlash demandé non confirmé")
    completions = [line for line in log.splitlines()
                   if f"saragossa serve completion model={model} " in line]
    require(bool(completions), "aucune trace de complétion pour vérifier le mode")
    paths = [re.search(r"decode_path=(\w+)", line) for line in completions]
    require(all(match is not None for match in paths), "trace decode_path absente")
    expected = {"ar", "batch"} if mode == "ar" else {"mtp"}
    require(all(match.group(1) in expected for match in paths), "repli de decode détecté")
