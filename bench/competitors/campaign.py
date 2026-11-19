"""Campagne de bench autonome : une commande, une nuit, des données brutes.

  python3 -I -B bench/competitors/campaign.py run RUN_DIR [options]   # campagne (reprise)
  python3 -I -B bench/competitors/campaign.py run RUN_DIR --dry-run   # tout vérifier sans mesurer
  python3 -I -B campaign.py                                            # auto-test stdlib

`run` fige d'abord le plan dans RUN_DIR/freeze.json (commits, versions et
sources des moteurs, binaires, SHA-256 des poids, matériel, macOS, prompts),
puis exécute la file d'entrées dans un ordre entrelacé : par modèle, les
configurations passent en round-robin, ordre inversé à chaque répétition
(ABBA). Une entrée = un chargement de serveur, une chauffe non comptée, puis
une requête mesurée par longueur de prompt, dans un ordre tourné par répétition.

Chaque entrée s'exécute sous `tools/gpulock.py` (verrou GPU), dans un
sous-processus de ce script. Avant chaque requête mesurée : loadavg < 2 et
aucun autre processus > 50 % CPU, sinon attente sans limite ; thermique et
pression mémoire sont notées. Watchdog : délai de chargement, délai absolu
par requête, SIGALRM de secours ; seuls les processus lancés ici sont arrêtés.

Reprise : `journal.jsonl` est append-only ; une entrée avec `entry_done` est
sautée, une entrée interrompue est rejouée (nouvel `attempt`). À la fin :
`summary.json` (comptes OK/échecs, aucune conclusion) et le fichier `DONE`.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import signal
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import client  # noqa: E402
import sensors  # noqa: E402
from run_matrix import port_free, wait_port_free  # noqa: E402

PORT = 18090
CTX = 20480                      # 16384 + sortie + marge
MAX_TOKENS = 256
WARMUP_TOKENS = 32
LENGTHS = (512, 4096, 16384)
LOAD_TIMEOUT_S = 2400            # ISQ mistral.rs et BF16 compris
REQUEST_DEADLINE_S = 1800
LOAD_MAX, CPU_MAX = 2.0, 50.0
COOLDOWN_MIN_S, COOLDOWN_MAX_S, COOLDOWN_GPU_C = 45, 600, 45
NONCE_LINE = "Request id: {nonce}\n\n"   # 16 chiffres = 16 tokens (prompts.py)
INSTRUCTION = ("\n\nAbove is Rust source code. Write a detailed technical review of it: "
               "explain the purpose of each main type and function, how they interact, and "
               "the invariants they rely on. Write at least 600 words.")


# --- Inventaire : modèles, moteurs, configurations ---------------------------

def inventory(ext, models, ai):
    """Poids par modèle et par moteur ; None = format absent localement."""
    gguf = ai / "Qwen3.8-27B-GGUF-bench"
    ggml = ai / "Qwen3.8-27B-GGUF-ggml"
    return {
        "27b-4bit": {
            "mlx": models / "Qwen3.8-27B-4bit", "bf16": ai / "Qwen3.8-27B-BF16",
            "dflash": models / "Qwen3.8-27B-DFlash2", "mtp_sidecar": None,
            "gguf": gguf / "Qwen3.8-27B-Q4_K_M.gguf", "gguf_mtp": gguf / "mtp-Qwen3.8-27B-Q4_0-pure.gguf",
            "gguf_dflash": ai / "Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf",
            "nvfp4": models / "nvfp4/Qwen3.8-27B-NVFP4",
        },
        "35b-a3b-4bit": {
            "mlx": ai / "Qwen3.6-35B-A3B-4bit", "bf16": None,
            "dflash": ai / "Qwen3.6-35B-A3B-DFlash2",
            "mtp_sidecar": ai / "Qwen3.6-35B-A3B-4bit/mtp-official/mtp.safetensors",
            "gguf": None, "gguf_mtp": None, "gguf_dflash": None, "nvfp4": None,
        },
        "27b-bf16": {
            "mlx": ai / "Qwen3.8-27B-BF16", "bf16": ai / "Qwen3.8-27B-BF16",
            "dflash": models / "Qwen3.8-27B-DFlash2", "mtp_sidecar": None,
            "gguf": ggml / "Qwen3.8-27B-BF16.gguf", "gguf_mtp": ggml / "mtp-Qwen3.8-27B-BF16.gguf",
            "gguf_dflash": ai / "Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf",
            "nvfp4": None,
        },
    }


def engines(ext, sar_bin):
    """Binaires et sources ; les versions sont relevées au gel."""
    return {
        "saragossa": {"bin": sar_bin, "source": "origin/main, cargo build --release"},
        "llama": {"bin": ext / "llama.cpp/v1.0.0/llama-b12359/llama-server",
                  "source": "github.com/ggml-org/llama.cpp release b12359 (= v1.0.0, b1b770d), "
                            "llama-b12359-bin-macos-arm64.tar.gz, sha256 82b00abc…"},
        "mlx": {"bin": ext / "mlx-lm/venv/bin/python",
                "source": "PyPI mlx-lm 0.32.2 (mlx 0.32.4), venv dédié"},
        "llmd": {"bin": ext / "llmd/llmd",
                 "source": "zml/homebrew-zml llmd 20261114.0, mirror.zml.ai tar.zst, sha256 9fd6d1c6…"},
        "ollama": {"bin": ext / "ollama/v0.40.16/ollama",
                   "source": "github.com/ollama/ollama release v0.40.16, ollama-darwin.tgz, sha256 a7437174…"},
        "mistralrs": {"bin": ext / "mistralrs/v0.9.5/mistralrs",
                      "source": "github.com/EricLBuehler/mistral.rs release v0.9.5, "
                                "mistralrs-metal-aarch64-apple-darwin.tar.gz, sha256 dcfb2e6c…"},
    }


# Réglages et leur source (« meilleure config documentée » de chaque moteur).
SOURCES = {
    "saragossa": "défauts de serve ; caches de préfixe coupés ; MTP profondeur 2 gelée en night1 "
                 "(matrix.frozen.json) ; DFlash arbre par défaut (PR #97-#104) ; tête MTP du 35B "
                 "par SARAGOSSA_RUST_MTP_SIDECAR (docs/perf/mtp-pairwise.md)",
    "llama": "night1 matrix.frozen.json : -fa on, -ngl all, KV f16, -b 2048 -ub 512, MTP "
             "--spec-draft-n-max 1, DFlash --spec-draft-n-max 3, cache de prompt coupé",
    "mlx": "night1 matrix.frozen.json : --prefill-step-size 2048, cache de prompt 0, un flux",
    "llmd": "night1 matrix.frozen.json : --token-batch-size=2048 --prefill-chunk-size=2048 "
            "--topk=1, fractions 0,5/0,5 (4 bits) ; BF16 : 0,85/0,3 pour loger 54 Go ; "
            "drafter natif retenté (en échec avec 20261029)",
    "ollama": "envconfig v0.40.16 : OLLAMA_FLASH_ATTENTION=1, KV f16, NUM_PARALLEL=1, "
              "KEEP_ALIVE=-1, num_gpu 999 ; GGUF locaux par Modelfile (pas de spéculatif GGUF)",
    "mistralrs": "docs v0.9.5 guides/perf/speculative-decoding : `--quant 4` depuis le BF16 "
                 "(ISQ), `--mtp` (tête intégrée), `--mtp-model` DFlash 2, paged attention auto "
                 "sur Metal ; --max-seqs 1, --prefix-cache-n 0",
}

SAR_ENV = {"SARAGOSSA_SERVE_PREFIX_CACHE": "0", "SARAGOSSA_RUST_PREFIX_CACHE": "0",
           "SARAGOSSA_API_KEY": "bench"}


def configs(model, w, eng, port=PORT):
    """Configurations d'un modèle : id -> spec, ou {"unsupported": raison}."""
    out = {}
    p = str(port)

    def sar(mode):
        env = dict(SAR_ENV, SARAGOSSA_RUST_MTP_DECODE="0" if mode == "ar" else "1")
        if mode == "mtp":
            env["SARAGOSSA_RUST_MTP_MAX_DRAFT"] = "2"
        if mode == "dflash":
            env["SARAGOSSA_RUST_DFLASH_DRAFT_DIR"] = str(w["dflash"])
        if mode != "ar" and w["mtp_sidecar"]:
            env["SARAGOSSA_RUST_MTP_SIDECAR"] = str(w["mtp_sidecar"])
        return {"engine": "saragossa", "env": env, "model_name": "qwen38",
                "argv": [str(eng["saragossa"]["bin"]), "serve", "--port", p,
                         "--model", f"qwen38={w['mlx']}", "--preload"],
                "files": [w["mlx"]] + ([w["dflash"]] if mode == "dflash" else [])
                + ([w["mtp_sidecar"]] if mode != "ar" and w["mtp_sidecar"] else [])}
    for mode in ("ar", "mtp", "dflash"):
        out[f"sar-{mode}"] = sar(mode)

    if w["gguf"]:
        def llama(mode):
            argv = [str(eng["llama"]["bin"]), "-m", str(w["gguf"]), "--host", "127.0.0.1",
                    "--port", p, "-np", "1", "-ngl", "all", "-fa", "on", "-c", str(CTX),
                    "-ctk", "f16", "-ctv", "f16", "--no-cache-prompt", "-cram", "0",
                    "--no-context-shift", "--jinja", "--chat-template-file",
                    str(w["mlx"] / "chat_template.jinja"), "--reasoning", "off",
                    "--reasoning-format", "none", "-b", "2048", "-ub", "512"]
            files = [w["gguf"]]
            if mode == "mtp":
                argv += ["--spec-type", "draft-mtp", "-md", str(w["gguf_mtp"]), "-ngld", "all",
                         "-ctkd", "f16", "-ctvd", "f16", "--spec-draft-n-max", "1"]
                files.append(w["gguf_mtp"])
            if mode == "dflash":
                argv += ["--spec-type", "draft-dflash", "-md", str(w["gguf_dflash"]), "-ngld", "all",
                         "-ctkd", "f16", "-ctvd", "f16", "--spec-draft-n-max", "3"]
                files.append(w["gguf_dflash"])
            return {"engine": "llama", "env": {}, "model_name": "qwen38", "argv": argv,
                    "files": files}
        for mode in ("ar", "mtp", "dflash"):
            out[f"llama-{mode}"] = llama(mode)
        out["ollama-ar"] = {"engine": "ollama", "model_name": f"bench-{model}", "gguf": str(w["gguf"]),
                            "argv": [str(eng["ollama"]["bin"]), "serve"], "files": [w["gguf"]],
                            "env": {"OLLAMA_HOST": f"127.0.0.1:{port}",
                                    "OLLAMA_MODELS": str(eng["ollama"]["bin"].parents[1] / "models"),
                                    "HOME": str(eng["ollama"]["bin"].parents[1] / "home"),
                                    "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "f16",
                                    "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1",
                                    "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_CONTEXT_LENGTH": str(CTX),
                                    "OLLAMA_NO_CLOUD": "1", "OLLAMA_NOPRUNE": "1",
                                    "OLLAMA_LOAD_TIMEOUT": "30m"}}
    else:
        for c in ("llama-ar", "llama-mtp", "llama-dflash", "ollama-ar"):
            out[c] = {"unsupported": "pas de GGUF local pour ce modèle (rien téléchargé)"}

    llmd_ckpt = w["nvfp4"] or w["bf16"]
    if llmd_ckpt:
        frac = ["--gpu-memory-fraction=0.5", "--cache-memory-fraction=0.5"] if w["nvfp4"] else \
            ["--gpu-memory-fraction=0.85", "--cache-memory-fraction=0.3"]
        for mode in ("ar", "native"):
            argv = [str(eng["llmd"]["bin"]), f"--model={llmd_ckpt}", f"--listen=127.0.0.1:{port}",
                    "--batch-size=1", f"--max-context-len={CTX}", "--model-name=qwen38", "--topk=1",
                    "--token-batch-size=2048", "--prefill-chunk-size=2048", *frac]
            if mode == "native":
                argv.append("--speculative-drafter-model=native")
            out[f"llmd-{mode}"] = {"engine": "llmd", "env": {}, "model_name": "qwen38",
                                   "argv": argv, "files": [llmd_ckpt]}
    else:
        out["llmd-ar"] = out["llmd-native"] = {
            "unsupported": "pas de checkpoint NVFP4 ou BF16 local lisible par llmd"}

    out["mlx-ar"] = {"engine": "mlx", "env": {}, "model_name": "default_model", "files": [w["mlx"]],
                     "argv": [str(eng["mlx"]["bin"]), "-I", "-m", "mlx_lm.server", "--model",
                              str(w["mlx"]), "--host", "127.0.0.1", "--port", p,
                              "--decode-concurrency", "1", "--prompt-concurrency", "1",
                              "--prompt-cache-size", "0", "--temp", "0", "--top-p", "1",
                              "--top-k", "0", "--min-p", "0", "--chat-template-args",
                              '{"enable_thinking":false}', "--prefill-step-size", "2048"]}

    base = [str(eng["mistralrs"]["bin"]), "serve", "--port", p, "--host", "127.0.0.1", "--no-ui",
            "--token-source", "none", "--max-seqs", "1", "--max-model-len", str(CTX),
            "--prefix-cache-n", "0"]
    if w["bf16"]:
        quant = ["--quant", "4"] if model.endswith("4bit") else []
        for mode, extra, files in (("ar", [], []), ("mtp", ["--mtp"], []),
                                   ("dflash", ["--mtp-model", str(w["dflash"])], [w["dflash"]])):
            out[f"mistralrs-{mode}"] = {"engine": "mistralrs", "env": {}, "model_name": None,
                                        "argv": base + ["-m", str(w["bf16"])] + quant + extra,
                                        "files": [w["bf16"]] + files}
    else:
        # Checkpoint MLX 4 bits (AFQ) : chargement documenté pour mlx-community, sans tête MTP.
        out["mistralrs-ar"] = {"engine": "mistralrs", "env": {}, "model_name": None,
                               "argv": base + ["-m", str(w["mlx"])], "files": [w["mlx"]]}
        out["mistralrs-mtp"] = out["mistralrs-dflash"] = {
            "unsupported": "pas de BF16 local ; MTP/DFlash mistral.rs documentés pour Qwen3.5/3.8 dense"}
    out["mlc-ar"] = {"unsupported": "MLC-LLM : pas de release GitHub ni de brew officiel "
                                    "(roues nightly mlc.ai seulement)"
                                    + (" ; qwen3_5_moe absent de mlc_llm/model" if "a3b" in model
                                       else " ; qwen3_5 présent, poids à recompiler (TVM)")}
    return out


DIALECT = {
    "saragossa": {"headers": {"Authorization": "Bearer bench"}, "health": ["/health"],
                  "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "llama": {"health": ["/health"], "extra": {"cache_prompt": False,
                                               "chat_template_kwargs": {"enable_thinking": False}}},
    "mlx": {"health": ["/v1/models"], "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "llmd": {"health": ["/v1/models", "/health"],
             "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "ollama": {"health": ["/api/version"], "extra": {"reasoning_effort": "none"}},
    "mistralrs": {"health": ["/health", "/v1/models"],
                  "extra": {"enable_thinking": False,
                            "chat_template_kwargs": {"enable_thinking": False}}},
}


# --- Plan -------------------------------------------------------------------

def build_plan(model_order, cfgs, reps, lengths):
    """Entrées entrelacées : par modèle, round-robin des configurations, ordre
    inversé à chaque répétition (A B C | C B A | A B C…), longueurs tournées.

    Les concurrents passent d'abord ; saragossa ferme la campagne, compilé
    depuis origin/main au début de son tour (décision du 19 nov.)."""
    entries = []
    for model, last in [(m, False) for m in model_order] + [(m, True) for m in model_order]:
        runnable = [c for c, spec in cfgs[model].items() if "unsupported" not in spec
                    and (spec.get("engine") == "saragossa") == last]
        for rep in range(reps):
            order = runnable if rep % 2 == 0 else runnable[::-1]
            for config in order:
                shift = rep % len(lengths)
                entries.append({"id": f"{model}|{config}|r{rep}", "model": model,
                                "config": config, "rep": rep,
                                "lengths": list(lengths[shift:] + lengths[:shift])})
    return entries


def nonce(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return f"{int(digest, 16) % 10**16:016d}"


# --- Prompts (venv : tokenizer du modèle) -----------------------------------

def build_prompts(model_dir, source_dir, out_path, lengths):
    """Prompt de longueur exacte (template, nonce compris) ; lancé avec le venv mlx-lm."""
    from transformers import AutoTokenizer  # noqa: PLC0415 (venv seulement)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    files = sorted(pathlib.Path(source_dir).rglob("*.rs"))
    doc = "".join(f"// {f.relative_to(source_dir)}\n{f.read_text(errors='replace')}\n"
                  for f in files)

    def count(chars):
        content = NONCE_LINE.format(nonce="0" * 16) + doc[:chars] + INSTRUCTION
        ids = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, "keys"):          # BatchEncoding selon la version de transformers
            ids = ids["input_ids"]
        return len(ids), content

    prompts = {}
    for target in lengths:
        lo, hi = 0, min(len(doc), target * 8)
        while lo < hi:                       # plus grand préfixe <= cible
            mid = (lo + hi + 1) // 2
            if count(mid)[0] <= target:
                lo = mid
            else:
                hi = mid - 1
        n, content = count(lo)
        prompts[str(target)] = {"content": content, "prompt_tokens": n,
                                "sha256": hashlib.sha256(content.encode()).hexdigest()}
    pathlib.Path(out_path).write_text(json.dumps(prompts))


def content_with_nonce(prompt, value):
    placeholder = NONCE_LINE.format(nonce="0" * 16)
    return NONCE_LINE.format(nonce=value) + prompt["content"][len(placeholder):]


# --- Environnement et gel ---------------------------------------------------

def sh(*argv, timeout=30):
    try:
        out = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"erreur: {error}"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_files(path):
    path = pathlib.Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")
                  and ".cache" not in p.parts)


def hash_weights(paths, cache_path):
    """SHA-256 de chaque fichier, en parallèle (hashlib libère le GIL), cache (taille, mtime)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    files = sorted({f for p in paths for f in weight_files(p)})
    todo = [f for f in files if cache.get(str(f), {}).get("key") != [f.stat().st_size,
                                                                       f.stat().st_mtime]]
    lock = threading.Lock()

    def work(chunk):
        for f in chunk:
            value = sha256_file(f)
            with lock:
                cache[str(f)] = {"key": [f.stat().st_size, f.stat().st_mtime], "sha256": value}
    threads = [threading.Thread(target=work, args=(todo[i::6],)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cache_path.write_text(json.dumps(cache))
    return {str(f): {"bytes": f.stat().st_size, "sha256": cache[str(f)]["sha256"]} for f in files}


def engine_version(name, spec):
    binary = spec["bin"]
    if name == "mlx":
        return sh(binary, "-I", "-c", "import importlib.metadata as m;"
                  "print('mlx-lm', m.version('mlx-lm'), 'mlx', m.version('mlx'))")
    if name == "llmd":
        return re.sub(r".*/llmd/v([^/]+)/.*", r"llmd \1", (binary.read_text()
                                                          if binary.exists() else "absent"),
                      flags=re.S)
    if name == "saragossa":
        return "commit noté à la compilation (tour de saragossa)"
    if not binary.exists():
        return "absent"
    lines = sh(binary, "--version").splitlines()
    return next((l for l in lines if "version" in l and "Warning: could not" not in l), lines[-1])


def machine():
    return {"model": sh("sysctl", "-n", "hw.model"),
            "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "memsize": sh("sysctl", "-n", "hw.memsize"),
            "macos": sh("sw_vers"), "kernel": platform.release(),
            "power": sensors.power_state()}


# --- Portes de mesure -------------------------------------------------------

def descendants(pid):
    found, frontier = [], [pid]
    while frontier:
        out = sh("pgrep", "-P", frontier.pop())
        kids = [int(x) for x in out.split() if x.isdigit()]
        found += kids
        frontier += kids
    return found


def busy_processes(own):
    out = sh("ps", "-A", "-o", "pid=,pcpu=,comm=")
    busy = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1].replace(".", "", 1).replace(",", "", 1).isdigit():
            pid, cpu = int(parts[0]), float(parts[1].replace(",", "."))
            if cpu > CPU_MAX and pid not in own:
                busy.append({"pid": pid, "cpu": cpu, "comm": parts[2][-60:]})
    return busy


def memory_pressure():
    level = sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
    return {"1": "normal", "2": "warn", "4": "critical"}.get(level, level)


def gate(own, log):
    """Attend loadavg < 2 et aucun processus étranger > 50 % CPU, sans limite."""
    start, last = time.monotonic(), 0
    while True:
        load = os.getloadavg()
        busy = busy_processes(own)
        if load[0] < LOAD_MAX and not busy:
            return {"gate_wait_s": round(time.monotonic() - start, 1), "loadavg": load,
                    "thermal": sensors.thermal(), "memory_pressure": memory_pressure()}
        if time.monotonic() - last > 600:
            log(f"attente machine calme : loadavg {load[0]:.2f}, occupés {busy[:3]}")
            last = time.monotonic()
        time.sleep(15)


def cooldown():
    """Plancher fixe, puis GPU <= 45 °C (macmon) au plus COOLDOWN_MAX_S."""
    start = time.monotonic()
    time.sleep(COOLDOWN_MIN_S)
    while time.monotonic() - start < COOLDOWN_MAX_S:
        temp = sensors.soc_sensors().get("gpu_temp_c")
        if temp is None or temp <= COOLDOWN_GPU_C:
            break
        time.sleep(5)
    return {"waited_s": round(time.monotonic() - start, 1), **sensors.soc_sensors()}


# --- Serveur ----------------------------------------------------------------

class Server:
    """Moteur lancé ici, dans le groupe du verrou GPU (pas de setsid)."""

    def __init__(self, spec, log_path):
        env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "USER") if k in os.environ}
        env.update({"LANG": "en_US.UTF-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        env.update(spec["env"])
        if "HOME" in spec["env"]:
            pathlib.Path(spec["env"]["HOME"]).mkdir(parents=True, exist_ok=True)
        self.spec, self.env = spec, env
        self.log = open(log_path, "ab")
        self.log.write(f"campaign-launch: {spec['argv']}\n".encode())
        self.log.flush()
        self.proc = subprocess.Popen(spec["argv"], env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def pids(self):
        return {self.proc.pid, *descendants(self.proc.pid)}

    def wait_ready(self, port, timeout_s):
        dialect = DIALECT[self.spec["engine"]]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False
            for path in dialect["health"]:
                try:
                    status, _ = client.http_json("127.0.0.1", port, "GET", path,
                                                 headers=dialect.get("headers"), timeout=5)
                    if status == 200:
                        return True
                except OSError:
                    pass
            time.sleep(1)
        return False

    def stop(self):
        """SIGTERM puis SIGKILL au moteur et à ses descendants ; vrai si tout a disparu."""
        tree = descendants(self.proc.pid)
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        for pid in tree:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.log.close()
        time.sleep(1)
        return all(sh("ps", "-p", pid, "-o", "pid=") == "" for pid in tree)


def prepare_ollama(spec, port, log):
    """Crée le modèle Ollama depuis le GGUF local (Modelfile, aucun téléchargement), une fois."""
    status, body = client.http_json("127.0.0.1", port, "GET", "/api/tags", timeout=30)
    if status == 200 and any(m.get("name", "").startswith(spec["model_name"])
                             for m in (body or {}).get("models", [])):
        return True
    modelfile = pathlib.Path(spec["env"]["OLLAMA_MODELS"]).parent / f"Modelfile-{spec['model_name']}"
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    modelfile.write_text(f"FROM {spec['gguf']}\nPARAMETER num_ctx {CTX}\nPARAMETER num_gpu 999\n"
                         "PARAMETER temperature 0\n")
    out = subprocess.run([spec["argv"][0], "create", spec["model_name"], "-f", str(modelfile)],
                         env={**os.environ, **spec["env"]}, capture_output=True, text=True,
                         timeout=7200)
    log(f"ollama create rc={out.returncode} {out.stdout[-300:]} {out.stderr[-500:]}")
    return out.returncode == 0


def request_body(spec, model_name, content, max_tokens):
    body = {"model": model_name, "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "stream_options": {"include_usage": True}}
    body.update(DIALECT[spec["engine"]]["extra"])
    return body


def integrity(record, expected_prompt_tokens):
    """Sortie exploitable : flux complet, tokens produits, aucune erreur."""
    usage = record.get("usage") or {}
    produced = usage.get("completion_tokens")
    reasons = []
    if record.get("status") != client.OK:
        reasons.append(f"statut {record.get('status')}")
    if record.get("stream_error") or record.get("error_body"):
        reasons.append("erreur dans la réponse")
    if not (produced or record.get("content") or record.get("reasoning")):
        reasons.append("aucun token produit")
    return {"ok": not reasons, "reasons": reasons, "completion_tokens": produced,
            "prompt_tokens_reported": usage.get("prompt_tokens"),
            "prompt_tokens_expected": expected_prompt_tokens,
            "output_chars": len(record.get("content", "")) + len(record.get("reasoning", ""))}


class Watchdog(Exception):
    pass


def run_entry(run_dir, entry_id):
    """Une entrée, sous verrou GPU : chargement, chauffe, mesures, arrêt."""
    freeze = json.loads((run_dir / "freeze.json").read_text())
    entry = next(e for e in freeze["plan"] if e["id"] == entry_id)
    spec = freeze["configs"][entry["model"]][entry["config"]]
    prompts = json.loads((run_dir / "prompts" / f"{entry['model']}.json").read_text())
    journal = Journal(run_dir)
    attempt = sum(1 for r in journal.rows() if r.get("entry") == entry_id
                  and r["kind"] == "entry_start")
    meta = {"entry": entry_id, "attempt": attempt, "model": entry["model"],
            "config": entry["config"], "engine": spec["engine"], "rep": entry["rep"]}
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", entry_id)
    server_log = log_dir / f"{safe}-a{attempt}.log"

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {entry_id}: {message}", flush=True)

    def on_alarm(*_):
        raise Watchdog("délai dépassé")

    def on_term(signum, _):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGALRM, on_alarm)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)

    journal.write({"kind": "entry_start", **meta, "time": time.time(),
                   "power": sensors.power_state()})
    sensors.validate_power(sensors.power_state())
    if not wait_port_free(PORT, 300):
        journal.write({"kind": "entry_done", **meta, "status": "port_busy"})
        return 1
    server, status = None, "ok"
    try:
        t0 = time.monotonic()
        signal.alarm(LOAD_TIMEOUT_S + 60)
        server = Server(spec, server_log)
        ready = server.wait_ready(PORT, LOAD_TIMEOUT_S)
        model_name = spec.get("model_name")
        if ready and spec["engine"] == "ollama":
            ready = prepare_ollama(spec, PORT, log)
            if ready:
                st, _ = client.http_json("127.0.0.1", PORT, "POST", "/api/generate",
                                         body={"model": model_name, "keep_alive": -1},
                                         timeout=LOAD_TIMEOUT_S)
                ready = st == 200
        if ready and model_name is None:
            st, body = client.http_json("127.0.0.1", PORT, "GET", "/v1/models", timeout=30)
            model_name = ((body or {}).get("data") or [{}])[0].get("id")
            ready = st == 200 and model_name is not None
        signal.alarm(0)
        journal.write({"kind": "load", **meta, "ready": ready, "load_s": round(time.monotonic() - t0, 1),
                       "model_name": model_name, "log": str(server_log.relative_to(run_dir))})
        if not ready:
            status = "load_failed"
            return 1
        headers = DIALECT[spec["engine"]].get("headers")
        signal.alarm(REQUEST_DEADLINE_S + 60)
        warm = client.chat_stream("127.0.0.1", PORT, request_body(
            spec, model_name, content_with_nonce(prompts["512"], nonce("warmup", entry_id, attempt)),
            WARMUP_TOKENS), headers=headers, deadline_s=REQUEST_DEADLINE_S)
        signal.alarm(0)
        journal.write({"kind": "warmup", **meta, "status": warm.get("status"),
                       "t_end_ns": warm.get("t_end_ns"), "usage": warm.get("usage"),
                       "error": warm.get("error") or warm.get("error_body")})
        for length in entry["lengths"]:
            prompt = prompts[str(length)]
            own = server.pids() | {os.getpid(), os.getppid()}
            state = gate(own, log)
            body = request_body(spec, model_name, content_with_nonce(
                prompt, nonce(entry_id, length)), MAX_TOKENS)
            signal.alarm(REQUEST_DEADLINE_S + 60)
            record = client.chat_stream("127.0.0.1", PORT, body, headers=headers,
                                        deadline_s=REQUEST_DEADLINE_S)
            signal.alarm(0)
            check = integrity(record, prompt["prompt_tokens"])
            journal.write({"kind": "measure", **meta, "prompt_len": length, "before": state,
                           "after": {"thermal": sensors.thermal(),
                                     "memory_pressure": memory_pressure(),
                                     "loadavg": os.getloadavg()},
                           "integrity": check, "record": record})
            log(f"{length} tokens : {'OK' if check['ok'] else check['reasons']} "
                f"({record['t_end_ns'] / 1e9:.1f} s)")
            if server.proc.poll() is not None:
                status = "crash"
                return 1
        return 0
    except Watchdog:
        status = "watchdog"
        return 1
    except BaseException:
        status = "interrupted"         # rejouée à la reprise
        raise
    finally:
        signal.alarm(0)
        gone = server.stop() if server else True
        journal.write({"kind": "entry_done", **meta, "status": status, "server_gone": gone,
                       "time": time.time()})


# --- Journal ----------------------------------------------------------------

class Journal:
    def __init__(self, run_dir):
        self.path = pathlib.Path(run_dir) / "journal.jsonl"

    def rows(self):
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # dernière ligne tronquée par une coupure
                pass
        return rows

    def write(self, row):
        prefix = b""
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as handle:
                handle.seek(-1, os.SEEK_END)
                prefix = b"" if handle.read(1) == b"\n" else b"\n"
        with open(self.path, "ab") as handle:
            handle.write(prefix + json.dumps(row).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


# --- Campagne ---------------------------------------------------------------

def build_saragossa(src, lock_env, log):
    """Worktree détaché sur origin/main, compilé sous build.lock ; rend (bin, commit)."""
    src = pathlib.Path(src)
    if not (src / ".git").exists():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(src),
                        "origin/main"], check=True)
    if sh("git", "-C", src, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"{src} modifié : refus de compiler un binaire non reproductible")
    subprocess.run(["git", "-C", str(src), "fetch", "-q", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", "--detach", "origin/main"], check=True)
    commit = sh("git", "-C", src, "rev-parse", "HEAD")
    log(f"compilation saragossa {commit[:10]} sous build.lock")
    subprocess.run([sys.executable, "-I", "-B", str(REPO / "tools/gpulock.py"), "run", "--owner",
                    "campagne-build", "--prio", "2", "--", "taskpolicy", "-b", "cargo", "build",
                    "--release", "--bin", "saragossa", "--manifest-path", str(src / "Cargo.toml")],
                   check=True, env={**os.environ, "GPU_LOCK": lock_env,
                                    "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS", "4")})
    return src / "target/release/saragossa", commit


def estimate(plan, freeze):
    """Durée prévue (minutes) : charge + chauffe + prefill + décodage + refroidissement.

    ponytail: débits forfaitaires par famille (night1, docs), à recaler sur budget réel.
    """
    total = 0.0
    for e in plan:
        bf16 = e["model"] == "27b-bf16"
        engine = freeze["configs"][e["model"]][e["config"]]["engine"]
        load = {"mistralrs": 240 if e["model"] != "35b-a3b-4bit" else 60,
                "ollama": 90}.get(engine, 40) * (2.5 if bf16 else 1)
        prefill_tps = (350 if bf16 else 600) * (0.35 if engine == "llmd" else 1)
        decode_tps = 11 if bf16 else (30 if e["model"] == "27b-4bit" else 70)
        total += load + 15 + sum(e["lengths"]) / prefill_tps + 3 * MAX_TOKENS / decode_tps
        total += COOLDOWN_MIN_S + 30
    return round(total / 60)


def main_run(args):
    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    ext = pathlib.Path(args.engines_dir).resolve()
    from config import path as cpath  # noqa: PLC0415  (local.env)
    inv = inventory(ext, cpath("MODELS"), cpath("AI_MODEL"))
    models = [m for m in inv if not args.models or m in args.models]
    lengths = tuple(args.lengths)
    freeze_path = run_dir / "freeze.json"

    if freeze_path.exists() and not args.dry_run:
        freeze = json.loads(freeze_path.read_text())
        log(f"reprise : freeze.json existant, plan de {len(freeze['plan'])} entrées")
    else:
        # Binaire compilé au tour de saragossa (fin de campagne), sauf --saragossa-bin.
        sar_bin = pathlib.Path(args.saragossa_bin) if args.saragossa_bin else \
            pathlib.Path(args.saragossa_src) / "target/release/saragossa"
        commit = sh("git", "-C", REPO, "rev-parse", "HEAD") if args.saragossa_bin else \
            "compilé depuis origin/main au tour de saragossa"
        eng = engines(ext, sar_bin)
        cfgs = {m: configs(m, inv[m], eng) for m in models}
        if args.configs:
            cfgs = {m: {c: s for c, s in v.items() if c in args.configs} for m, v in cfgs.items()}
        problems = []
        for m in models:
            for c, spec in cfgs[m].items():
                if "unsupported" in spec:
                    continue
                missing = [str(f) for f in spec["files"] if not pathlib.Path(f).exists()]
                if not missing and spec["engine"] == "saragossa" and not args.saragossa_bin:
                    continue      # binaire compilé au lancement réel
                if missing or not pathlib.Path(spec["argv"][0]).exists():
                    spec["unsupported"] = f"introuvable : {missing or spec['argv'][0]}"
                    problems.append(f"{m}/{c}: {spec['unsupported']}")
        venv = eng["mlx"]["bin"]
        for m in models:
            target = run_dir / "prompts" / f"{m}.json"
            if not target.exists():
                log(f"prompts {m} (tokenizer du checkpoint)")
                subprocess.run([str(venv), "-I", "-B", str(HERE / "campaign.py"), "_prompts",
                                str(inv[m]["mlx"]), str(args.prompt_source), str(target),
                                *map(str, lengths)], check=True,
                               env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
                                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        prompt_meta = {m: {k: {"prompt_tokens": v["prompt_tokens"], "sha256": v["sha256"]}
                           for k, v in json.loads((run_dir / "prompts" / f"{m}.json")
                                                  .read_text()).items()} for m in models}
        versions = {n: {"bin": str(s["bin"]), "source": s["source"],
                        "version": engine_version(n, s), "settings_source": SOURCES[n],
                        "sha256": sha256_file(s["bin"]) if pathlib.Path(s["bin"]).is_file() else None}
                    for n, s in eng.items()}
        versions["saragossa"]["commit"] = commit
        plan = build_plan(models, cfgs, args.reps, lengths)
        freeze = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo_commit": sh(
                      "git", "-C", REPO, "rev-parse", "HEAD"),
                  "engines": versions, "machine": machine(), "models": {
                      m: {k: (str(v) if v else None) for k, v in inv[m].items()} for m in models},
                  "configs": {m: {c: {k: v for k, v in s.items() if k != "files"} | {
                      "files": [str(f) for f in s.get("files", [])]} if "unsupported" not in s
                      else s for c, s in cfgs[m].items()} for m in models},
                  "prompts": prompt_meta, "prompt_source": str(args.prompt_source),
                  "protocol": {"lengths": lengths, "reps": args.reps, "max_tokens": MAX_TOKENS,
                               "warmup_tokens": WARMUP_TOKENS, "ctx": CTX, "port": PORT,
                               "load_max": LOAD_MAX, "cpu_max_pct": CPU_MAX,
                               "cooldown": [COOLDOWN_MIN_S, COOLDOWN_MAX_S, COOLDOWN_GPU_C],
                               "sampling": "greedy, temperature 0, top_p 1"},
                  "plan": plan}
        freeze["estimate_min"] = estimate(plan, freeze)
        if args.dry_run:
            for name, v in versions.items():
                state = "ok" if v["sha256"] else (
                    "compilé au lancement" if name == "saragossa" and not args.saragossa_bin
                    else "ABSENT")
                log(f"moteur {name}: {v['version']!s:.80} ({state})")
            for m in models:
                for c, s in cfgs[m].items():
                    log(f"{m:13} {c:16} {'non supporté : ' + s['unsupported'] if 'unsupported' in s else 'ok'}")
            log(f"port {PORT} libre : {port_free(PORT)} ; gpulock : "
                f"{(REPO / 'tools/gpulock.py').exists()} ; macmon : {bool(sh('which', 'macmon'))}")
            log(f"plan : {len(plan)} entrées, {len(plan) * len(lengths)} mesures, "
                f"durée estimée ~{freeze['estimate_min'] // 60} h {freeze['estimate_min'] % 60:02d} min")
            (run_dir / "freeze-dryrun.json").write_text(json.dumps(freeze, indent=1, default=str))
            return 0 if not problems else 2
        all_files = {f for m in models for s in cfgs[m].values() if "unsupported" not in s
                     for f in s["files"]}
        log(f"SHA-256 des poids ({len(all_files)} chemins)")
        freeze["weights_sha256"] = hash_weights(all_files, run_dir / ".sha-cache.json")
        freeze_path.write_text(json.dumps(freeze, indent=1, default=str))
        log(f"gel écrit : {len(plan)} entrées, ~{freeze['estimate_min']} min estimées")

    journal = Journal(run_dir)
    child = {"proc": None}
    flag = pathlib.Path(args.busy_flag)

    def stop_all(signum, _):
        # Pas de wait() ici : un handler qui attend dans un Popen.wait interrompu
        # se bloque (verrou de waitpid). On transmet, la boucle principale sort.
        child["stop"] = signum
        if child["proc"] is not None and child["proc"].returncode is None:
            try:
                os.kill(child["proc"].pid, signal.SIGTERM)   # gpulock le transmet au groupe
            except ProcessLookupError:
                pass
        if child["proc"] is None:
            raise SystemExit(f"interrompu (signal {signum}) ; relancer la même commande reprend")
    signal.signal(signal.SIGTERM, stop_all)
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGHUP, stop_all)

    flag.write_text(f"{os.getpid()} {run_dir}\n")   # les autres agents ne compilent pas
    try:
        return loop(args, run_dir, freeze, journal, child, log)
    finally:
        if flag.exists():
            flag.unlink()


def ensure_saragossa(args, run_dir, freeze, journal, log):
    """Compile origin/main au premier tour de saragossa ; commit et SHA dans freeze.json."""
    sar = freeze["engines"]["saragossa"]
    built = [r for r in journal.rows() if r["kind"] == "saragossa_build"]
    if built:
        if sha256_file(sar["bin"]) != built[-1]["sha256"]:
            raise SystemExit("binaire saragossa modifié depuis sa compilation : refus de reprendre")
        return
    if args.saragossa_bin:
        commit = sar["commit"]
    else:
        _, commit = build_saragossa(args.saragossa_src, args.build_lock, log)
    sar.update({"commit": commit, "sha256": sha256_file(sar["bin"]),
                "version": f"origin/main {commit}",
                "built": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (run_dir / "freeze.json").write_text(json.dumps(freeze, indent=1, default=str))
    journal.write({"kind": "saragossa_build", "commit": commit, "sha256": sar["sha256"],
                   "time": time.time()})


def loop(args, run_dir, freeze, journal, child, log):
    previous = None
    for entry in freeze["plan"]:
        rows = journal.rows()
        if any(r["kind"] == "entry_done" and r.get("status") != "interrupted"
               and r.get("entry") == entry["id"] for r in rows):
            continue
        failed = [r for r in rows if r["kind"] == "entry_done" and r.get("model") == entry["model"]
                  and r.get("config") == entry["config"] and r["status"] == "load_failed"]
        if len(failed) >= 2:
            journal.write({"kind": "entry_done", "entry": entry["id"], "model": entry["model"],
                           "config": entry["config"], "status": "skipped_after_load_failures"})
            continue
        if freeze["configs"][entry["model"]][entry["config"]]["engine"] == "saragossa":
            ensure_saragossa(args, run_dir, freeze, journal, log)
        if previous is not None:
            journal.write({"kind": "cooldown", "before": entry["id"], **cooldown()})
        log(f"entrée {entry['id']}")
        child["proc"] = subprocess.Popen(
            [sys.executable, "-I", "-B", str(REPO / "tools/gpulock.py"), "run", "--owner",
             "campagne", "--prio", str(args.prio), "--", sys.executable, "-I", "-B", str(HERE / "campaign.py"),
             "_entry", str(run_dir), entry["id"]], env={**os.environ, "GPU_LOCK": args.gpu_lock})
        child["proc"].wait()
        child["proc"] = None
        if child.get("stop"):
            raise SystemExit(f"interrompu (signal {child['stop']}) ; relancer la même commande reprend")
        previous = entry["id"]
        if not any(r["kind"] == "entry_done" and r.get("entry") == entry["id"]
                   for r in journal.rows()):
            raise SystemExit(f"entrée {entry['id']} sans fin enregistrée (rc "
                             "?) ; relancer reprend")

    rows = journal.rows()
    done = {r["entry"]: r for r in rows if r["kind"] == "entry_done"
            and r.get("status") != "interrupted"}
    final_attempt = {e: r.get("attempt") for e, r in done.items()}
    measures = [r for r in rows if r["kind"] == "measure"
                and final_attempt.get(r["entry"]) == r["attempt"]]
    summary = {"entries": len(freeze["plan"]),
               "entries_by_status": {s: sum(1 for r in done.values() if r["status"] == s)
                                     for s in sorted({r["status"] for r in done.values()})},
               "measures_planned": len(freeze["plan"]) * len(freeze["protocol"]["lengths"]),
               "measures_ok": sum(1 for r in measures if r["integrity"]["ok"]),
               "measures_failed": sum(1 for r in measures if not r["integrity"]["ok"]),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "DONE").write_text(summary["finished"] + "\n")
    log(f"terminé : {summary}")
    return 0


def _self_check():
    cfgs = {"m": {"a": {}, "b": {}, "c": {"unsupported": "x"}}}
    plan = build_plan(["m"], cfgs, 3, LENGTHS)
    assert [e["config"] for e in plan] == ["a", "b", "b", "a", "a", "b"]
    cfgs = {"m": {"s": {"engine": "saragossa"}, "l": {"engine": "llama"}},
            "n": {"s": {"engine": "saragossa"}, "l": {"engine": "llama"}}}
    order = [(e["model"], e["config"]) for e in build_plan(["m", "n"], cfgs, 1, LENGTHS)]
    assert order == [("m", "l"), ("n", "l"), ("m", "s"), ("n", "s")]
    assert plan[1]["lengths"] == [512, 4096, 16384] and plan[2]["lengths"] == [4096, 16384, 512]
    assert len(nonce("x")) == 16 and nonce("x") == nonce("x") != nonce("y")
    prompt = {"content": NONCE_LINE.format(nonce="0" * 16) + "body"}
    assert content_with_nonce(prompt, "1" * 16) == "Request id: 1111111111111111\n\nbody"
    good = {"status": client.OK, "usage": {"completion_tokens": 5, "prompt_tokens": 9},
            "content": "x"}
    assert integrity(good, 9)["ok"]
    assert not integrity({**good, "status": client.TIMEOUT}, 9)["ok"]
    assert not integrity({"status": client.OK, "usage": {"completion_tokens": 0}}, 9)["ok"]
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as tmp:
        journal = Journal(tmp)
        journal.write({"kind": "a"})
        with open(journal.path, "ab") as handle:
            handle.write(b'{"kind": "tronq')
        journal.write({"kind": "b"})
        assert [r["kind"] for r in journal.rows()] == ["a", "b"]
    ext, root = pathlib.Path("/e"), pathlib.Path("/m")
    inv = inventory(ext, root, root)
    eng = engines(ext, pathlib.Path("/bin/saragossa"))
    c35 = configs("35b-a3b-4bit", inv["35b-a3b-4bit"], eng)
    assert "unsupported" in c35["llama-ar"] and "unsupported" in c35["llmd-ar"]
    assert c35["sar-mtp"]["env"]["SARAGOSSA_RUST_MTP_SIDECAR"].endswith("mtp.safetensors")
    c4 = configs("27b-4bit", inv["27b-4bit"], eng)
    assert "--quant" in c4["mistralrs-ar"]["argv"]
    assert "--quant" not in configs("27b-bf16", inv["27b-bf16"], eng)["mistralrs-ar"]["argv"]
    assert c4["sar-ar"]["env"]["SARAGOSSA_RUST_MTP_DECODE"] == "0"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        print("campaign: auto-test ok")
        sys.exit(0)
    if sys.argv[1] == "_entry":
        sys.exit(run_entry(pathlib.Path(sys.argv[2]), sys.argv[3]))
    if sys.argv[1] == "_prompts":
        build_prompts(sys.argv[2], sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5:]])
        sys.exit(0)
    scratch = os.environ.get("BENCH_SCRATCH", str(pathlib.Path.home() / ".cache/saragossa"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run"])
    parser.add_argument("run_dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--models", nargs="+", help="sous-ensemble : 27b-4bit 35b-a3b-4bit 27b-bf16")
    parser.add_argument("--configs", nargs="+", help="sous-ensemble de configurations")
    parser.add_argument("--engines-dir", default=os.environ.get("EXT_ENGINES", f"{scratch}/ext-engines"))
    parser.add_argument("--saragossa-src", default=str(REPO / ".claude/worktrees/campagne-bin"),
                        help="worktree détaché sur origin/main, compilé au lancement")
    parser.add_argument("--saragossa-bin", help="binaire existant (sans compilation)")
    parser.add_argument("--prompt-source", default=str(REPO / "src"),
                        help="sources .rs découpées en prompts (figées dans RUN_DIR/prompts)")
    parser.add_argument("--gpu-lock", default=os.environ.get("GPU_LOCK", f"{scratch}/gpu.lock"))
    parser.add_argument("--prio", type=int, default=2, help="priorité gpulock (0 = la plus haute)")
    parser.add_argument("--busy-flag", default=f"{scratch}/bench-en-cours",
                        help="présent pendant la campagne : les autres agents ne compilent pas")
    parser.add_argument("--build-lock", default=os.environ.get("BUILD_LOCK", f"{scratch}/build.lock"))
    sys.exit(main_run(parser.parse_args()))
