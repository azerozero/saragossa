"""Ordonnanceur du bench concurrents (stdlib seule, sauf `freeze`).

  python3 -I run_matrix.py env RUN_DIR NAME        # versions, binaires, machine -> env-NAME.json
  python3 -I run_matrix.py plan RUN_DIR PHASE...   # tire et écrit RUN_DIR/plan.json
  python3 -I run_matrix.py run RUN_DIR             # exécute le plan, écrit results.jsonl
  VENV_PY -I run_matrix.py freeze RUN_DIR          # règle de sélection -> matrix.frozen.json

L'unité de randomisation est la configuration chargée : dans chaque bloc,
l'ordre des configurations est permuté, puis tous les prompts d'une
configuration passent à la suite dans un ordre tiré. Le tirage dépend de la
seule graine de `matrix.json`, publiée avec le plan avant toute mesure.

Identifiants : `campaign` (sha du plan), `run_seq` (n-ième lancement de
`run` sur ce dossier), `attempt` (chargement d'une entrée) et `request_id`
(phase, bloc, configuration, prompt, graine, répétition). Reprise : une
entrée terminée (`entry_done`) n'est pas rejouée ; une entrée interrompue
est rejouée en entier et ses lignes antérieures sont déclarées
`superseded`, que l'analyse écarte.

Le GPU est exclusif : le lanceur de la campagne prend le verrou GPU avant
`run` et le rend après ; ce script ne lance qu'un moteur à la fois.
"""

import hashlib
import json
import os
import pathlib
import random
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import client  # noqa: E402
import memwatch  # noqa: E402
from config import path as config_path, paths as config_paths, sanitize  # noqa: E402
from prompts import fnv1a64  # noqa: E402  (stdlib seule au niveau module)

HERE = pathlib.Path(__file__).resolve().parent

# Statuts d'échec d'une configuration ou d'une requête, en plus de ceux du client.
UNSUPPORTED = "unsupported"
INVALID_CONFIG = "invalid_config"
LOAD_FAILED = "load_failed"
OOM = "oom"
CRASH = "crash"
SKIPPED = "skipped_after_failure"
OOM_MARKERS = ("out of memory", "outofmemory", "insufficient memory", "failed to allocate",
               "kiogpucommandbuffercallbackerroroutofmemory", "ressource mémoire")
REFUSED_BUSY = "refused_busy"
ADMISSION_MARKERS = ("budget mémoire des requêtes en cours", "file d'inférence pleine")
BIND_MARKERS = ("address already in use", "bind tcp", "failed to bind", "eaddrinuse")
JETSAM_DIR = pathlib.Path("/Library/Logs/DiagnosticReports")
MAX_ATTEMPTS = 3


def load_matrix(run_dir=None):
    """Matrice gelée du run si elle existe (après `freeze`), sinon celle du dépôt."""
    frozen = pathlib.Path(run_dir) / "matrix.frozen.json" if run_dir else None
    path = frozen if frozen and frozen.exists() else HERE / "matrix.json"
    return json.loads(path.read_text()), path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(matrix, config_id, override=None):
    """Configuration effective : base, puis chaque réglage (`slots`) gelé ou par défaut.

    Un réglage non gelé (`{}`) prend le premier candidat de sa famille, le
    défaut documenté du moteur. `override` = (famille, candidat) pour le réglage.
    """
    base = matrix["configs"][config_id]
    args, env = list(base.get("args", [])), dict(base.get("env", {}))
    for slot, value in base.get("slots", {}).items():
        if override and override[0] == slot:
            value = override[1]
        elif not value:
            value = matrix["tuning"][slot]["candidates"][0]
        args += value.get("args", [])
        env.update(value.get("env", {}))
    return {"engine": base["engine"], "launch": base["launch"], "args": args, "env": env,
            "unsupported": base.get("unsupported")}


def tuning_entries(matrix):
    """(id, prompts, configuration) de chaque candidat de la grille de réglage."""
    for family, spec in matrix["tuning"].items():
        for index, candidate in enumerate(spec["candidates"]):
            yield (f"tune:{family}:{index}", spec["prompts"],
                   resolve(matrix, spec["probe"], (family, candidate)))


def tuning_references(matrix):
    """Références AR du réglage : une par configuration AR citée par une famille
    spéculative (`reference`), sur l'union des prompts de ces familles. Mêmes
    nonces que les candidats : l'identité greedy de chaque candidat se vérifie
    contre elle (analyze.pair_function)."""
    refs = {}
    for spec in matrix["tuning"].values():
        if spec.get("reference"):
            refs.setdefault(spec["reference"], set()).update(spec["prompts"])
    for config, prompts in sorted(refs.items()):
        yield (f"tune-ref:{config}", sorted(prompts), resolve(matrix, config))


def nonce(*parts):
    """16 chiffres dérivés de la graine et de la clé d'observation."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"{int(digest, 16) % 10**16:016d}"


def request_id(phase, block, config, prompt, seed, repeat):
    return f"{phase}|b{block}|{config}|{prompt}|s{seed}|r{repeat}"


def plan(run_dir, phases):
    matrix, matrix_path = load_matrix(run_dir)
    entries = []
    for phase in phases:
        spec = matrix["phases"][phase]
        rng = random.Random(f"{matrix['seed']}:{phase}")
        cls = matrix["classes"][spec["class"]]
        seeds = spec.get("seeds", [None])
        # Requêtes identiques réservées au diagnostic (validate) : ailleurs, le
        # nonce change à chaque répétition mesurée et reste commun aux
        # configurations appariées (même phase, bloc, prompt, graine, répétition).
        same = spec.get("identical_repeats", False)
        if spec["configs"] == "tuning":
            candidates = list(tuning_entries(matrix)) + list(tuning_references(matrix))
        else:
            candidates = [(c, spec.get("prompts", cls["prompts"]), resolve(matrix, c))
                          for c in spec["configs"]]
        for block in range(spec["blocks"]):
            rng.shuffle(candidates)
            for position, (config, prompts, config_spec) in enumerate(candidates):
                order = list(prompts)
                rng.shuffle(order)
                requests = [
                    {"prompt": p, "seed": s, "repeat": r,
                     "nonce": (nonce(matrix["seed"], phase, block, p) if same
                               else nonce(matrix["seed"], phase, block, p, s, r)),
                     "request_id": request_id(phase, block, config, p, s, r)}
                    for p in order for s in seeds for r in range(spec.get("repeats", 1))
                ]
                warmup = {"prompt": cls["warmup"], "seed": seeds[0], "repeat": 0,
                          "nonce": nonce(matrix["seed"], phase, block, "warmup", config),
                          "request_id": request_id(phase, block, config, "warmup", None, 0)}
                entries.append({"phase": phase, "block": block, "position": position,
                                "config": config, "config_spec": config_spec,
                                "class": spec["class"], "ctx": cls["ctx"],
                                "sampling": spec["sampling"], "warmup": warmup,
                                "requests": requests})
    out = {"seed": matrix["seed"], "phases": phases, "matrix": str(matrix_path.name),
           "matrix_sha256": sha256_file(matrix_path), "entries": entries,
           "requests": [r for e in entries for r in [e["warmup"], *e["requests"]]]}
    path = pathlib.Path(run_dir) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    path.write_text(text)
    # Copie nommée : `plan.json` est le plan courant, chaque plan publié reste.
    (path.parent / f"plan-{'+'.join(phases)}.json").write_text(text)
    real = [e for e in entries if not e["config_spec"].get("unsupported")]
    n = sum(len(e["requests"]) for e in real)
    print(f"{len(real)} chargements, {n} requêtes mesurées "
          f"(hors {len(entries) - len(real)} entrées non supportées), "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def request_body(matrix, engine, sampling, content, seed, max_tokens=None):
    dialect = matrix["engines"][engine]
    params = dict(matrix["sampling"][sampling])
    body = {"model": dialect["model"], "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": max_tokens or matrix["max_tokens"],
            "stream_options": {"include_usage": True}}
    for key in dialect["send"]:
        if key in params:
            body[key] = params[key]
    body.update(dialect.get("extra", {}))
    if seed is not None and dialect.get("seed", True):
        body["seed"] = seed
    return body


def content_for(prompts, prompt_id, nonce_value):
    placeholder = prompts["nonce_line"].format(nonce="0" * 16)
    content = prompts["by_id"][prompt_id]["content"]
    return prompts["nonce_line"].format(nonce=nonce_value) + content[len(placeholder):]


def thermal():
    """État de `pmset -g therm` : bridé si un niveau d'alerte est enregistré
    ou si une limite de vitesse CPU inférieure à 100 est annoncée."""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                         check=False).stdout
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    throttled = False
    for line in lines:
        if "warning level" in line.lower() and not line.startswith("Note: No"):
            throttled = True
        if line.startswith("CPU_Speed_Limit") and not line.rstrip().endswith("100"):
            throttled = True
    return {"raw": " | ".join(lines), "throttled": throttled}


def cooldown(matrix):
    """Repos minimal, puis attente d'un état thermique nominal (plafonnée)."""
    time.sleep(matrix["cooldown_min_s"])
    start = time.monotonic()
    state = thermal()
    while state["throttled"] and time.monotonic() - start < matrix["cooldown_max_s"]:
        time.sleep(15)
        state = thermal()
    state["waited_s"] = round(matrix["cooldown_min_s"] + time.monotonic() - start, 1)
    return state


def port_free(port):
    """Vrai si le port peut être lié sans SO_REUSEADDR (TIME_WAIT compris)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_port_free(port, timeout_s=180):
    deadline = time.monotonic() + timeout_s
    while not port_free(port):
        if time.monotonic() > deadline:
            return False
        time.sleep(2)
    return True


def engine_env(extra):
    """Environnement maîtrisé du moteur : rien n'est hérité hors de cette liste."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "USER")
           if key in os.environ}
    # Aucun téléchargement possible pendant une mesure.
    env.update({"LANG": "en_US.UTF-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if "BENCH_ENV" in os.environ:
        env["BENCH_ENV"] = os.environ["BENCH_ENV"]
    # Chemins déjà résolus par paths.sh : le lanceur n'a pas besoin de local.env.
    env.update(config_paths())
    env.update(extra)
    return env


class Engine:
    """Processus moteur lancé par ce script, dans son propre groupe de processus."""

    def __init__(self, matrix, entry, log_path):
        config = entry["config_spec"]
        self.matrix = matrix
        self.engine = config["engine"]
        self.port = matrix["port"]
        # Arguments par classe de contexte du moteur (fractions mémoire de llmd).
        class_args = matrix["engines"][self.engine].get("class_args", {}).get(entry["class"], [])
        self.argv = [str(HERE / config["launch"][0]), *config["launch"][1:],
                     str(self.port), str(entry["ctx"]), *config["args"], *class_args]
        self.env = engine_env(config["env"])
        self.log_path = log_path
        self.log = open(log_path, "xb")
        self.started = time.perf_counter_ns()
        self.started_wall = time.time()
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self):
        return self.proc.poll() is None

    def wait_ready(self, timeout_s):
        dialect = self.matrix["engines"][self.engine]
        paths = dialect["health"] if isinstance(dialect["health"], list) else [dialect["health"]]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                return False
            for path in paths:
                try:
                    status, _ = client.http_json("127.0.0.1", self.port, "GET", path,
                                                 headers=dialect.get("headers"), timeout=5)
                    if status == 200:
                        return True
                except OSError:
                    pass
            time.sleep(1)
        return False

    def log_text(self):
        self.log.flush()
        return pathlib.Path(self.log_path).read_text(errors="replace")

    def jetsam_report(self):
        """Rapport JetsamEvent postérieur au lancement qui cite ce PID, s'il existe."""
        for report in JETSAM_DIR.glob("JetsamEvent-*.ips"):
            try:
                if report.stat().st_mtime < self.started_wall:
                    continue
                if f'"pid" : {self.proc.pid},' in report.read_text(errors="replace"):
                    return str(report)
            except OSError:
                continue
        return None

    def failure_status(self):
        """OOM seulement sur preuve : marqueur du journal ou rapport jetsam du PID."""
        text = self.log_text().lower()
        if any(marker in text for marker in OOM_MARKERS) or self.jetsam_report():
            return OOM
        return CRASH

    def bind_failed(self):
        text = self.log_text().lower()
        return any(marker in text for marker in BIND_MARKERS)

    def stop(self):
        """SIGTERM au groupe, SIGKILL après 30 s ; vrai si le groupe a disparu."""
        try:
            if self.alive():
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    try:
                        self.proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        return False
            # Fils éventuels du même groupe : SIGKILL, puis vérification.
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                time.sleep(1)
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            return False
        finally:
            self.log.close()


def ingest_proof(matrix, engine, port, body, eng):
    """Preuve d'entrée, hors chronométrage ; sa nature est écrite dans `kind`.

    - `log` (saragossa) : ids ingérés lus dans le journal de la requête ;
    - `server_reconstruction` (llama.cpp) : `/apply-template` puis `/tokenize`
      sur le serveur chargé, reconstruction de la chaîne du chemin chat, pas
      lecture des ids de la requête ;
    - `offline` (mlx-lm, llmd) : ids calculés hors moteur, recoupés par
      `usage.prompt_tokens`.
    """
    kind = matrix["engines"][engine]["ingest"]
    if kind == "log":
        lines = [line for line in eng.log_text().splitlines()
                 if "saragossa serve completion" in line]
        if not lines:
            return {"kind": "log", "error": "ligne de complétion absente"}
        fields = dict(item.split("=", 1) for item in lines[-1].split() if "=" in item)
        return {"kind": "log", **fields}
    if kind == "server_reconstruction":
        status, rendered = client.http_json("127.0.0.1", port, "POST", "/apply-template", {
            "messages": body["messages"],
            "chat_template_kwargs": {"enable_thinking": False}})
        if status != 200 or not isinstance(rendered, dict):
            return {"kind": kind, "error": f"/apply-template {status}"}
        status, tokens = client.http_json("127.0.0.1", port, "POST", "/tokenize", {
            "content": rendered["prompt"], "add_special": True, "parse_special": True})
        if status != 200 or not isinstance(tokens, dict):
            return {"kind": kind, "error": f"/tokenize {status}"}
        ids = tokens["tokens"]
        return {"kind": kind, "prompt_tokens": len(ids), "prompt_fnv1a64": fnv1a64(ids)}
    return {"kind": "offline"}


class Campaign:
    """Contexte d'écriture d'un `run` : identifiants et flux de résultats."""

    def __init__(self, run_dir, matrix, prompts, out, campaign, run_seq):
        self.run_dir = run_dir
        self.matrix = matrix
        self.prompts = prompts
        self.out = out
        self.campaign = campaign
        self.run_seq = run_seq

    def write(self, line):
        self.last = line
        self.out.write(json.dumps({"campaign": self.campaign, "run_seq": self.run_seq,
                                   **sanitize(line)}, ensure_ascii=False) + "\n")
        self.out.flush()


def run_request(camp, eng, entry, request, kind, meta):
    matrix = camp.matrix
    engine = eng.engine
    content = content_for(camp.prompts, request["prompt"], request["nonce"])
    body = request_body(matrix, engine, entry["sampling"], content, request["seed"],
                        request.get("max_tokens"))
    headers = matrix["engines"][engine].get("headers")
    before = memwatch.rusage(eng.proc.pid)
    with memwatch.Sampler(eng.proc.pid, matrix["memory_period_s"]) as sampler:
        record = client.chat_stream("127.0.0.1", eng.port, body, headers,
                                    deadline_s=matrix["deadline_s"])
    after = memwatch.rusage(eng.proc.pid)
    if record["status"] in (client.CONNECTION_LOST, client.STREAM_INCOMPLETE,
                            client.STREAM_ERROR) and not eng.alive():
        record["status"] = eng.failure_status()
    if record["status"] == client.HTTP_ERROR and record.get("http_status") == 503:
        error_body = record.get("error_body", "")
        error_text = (error_body if isinstance(error_body, str)
                      else json.dumps(error_body, ensure_ascii=False)).lower()
        if any(m in error_text for m in ADMISSION_MARKERS):
            # Refus d'admission (octets en vol, file pleine) : pas un manque de mémoire.
            record["status"] = REFUSED_BUSY
        elif any(m in error_text for m in OOM_MARKERS):
            record["status"] = OOM
    t0 = record["t0_ns"]
    line = {
        "kind": kind, **meta, "engine": engine, **request,
        "body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "body_params": {k: v for k, v in body.items() if k != "messages"},
        "memory": {"before": before, "after": after,
                   # Échantillons conservés, en ns depuis t0 de la requête.
                   "samples": [(t - t0, v) for t, v in sampler.samples]},
        **record,
    }
    if record["status"] == client.OK:
        line["ingest"] = ingest_proof(matrix, engine, eng.port, body, eng)
    camp.write(line)
    print(f"  {kind:7} {request['prompt']:16} {record['status']:17} "
          f"{record['t_end_ns'] / 1e9:8.1f} s", flush=True)
    return record["status"]


def nonce_probe(camp, eng, entry, meta, measure_meta):
    """Sonde de cache par chargement (llmd, sans compteur de réutilisation).

    Même document, deux nonces : A est froid (document jamais envoyé à ce
    chargement), B ne diffère que par le nonce. Si le cache réutilisait le
    document au-delà du nonce, le TTFT de B s'effondrerait ; la sonde passe si
    TTFT(B) ≥ 0,8 × TTFT(A). Le document n'est pas celui de la chauffe.
    """
    warmup_doc = entry["warmup"]["prompt"]
    doc = "tune-prose4k-1" if warmup_doc != "tune-prose4k-1" else "tune-code4k-1"
    ttfts = []
    for index in range(2):
        request = {"prompt": doc, "seed": None, "repeat": index, "max_tokens": 1,
                   "nonce": nonce(camp.matrix["seed"], entry["phase"], entry["block"],
                                  "nonce-probe", entry["config"], index),
                   "request_id": request_id(entry["phase"], entry["block"], entry["config"],
                                            f"nonce-probe-{doc}", None, index)}
        status = run_request(camp, eng, entry, request, "probe", measure_meta)
        if status != client.OK:
            camp.write({"kind": "nonce_probe", **meta, "status": status, "passed": None})
            return status
        line = camp.last
        ttft = next((e["t_ns"] for e in line["events"]
                     if e.get("content_chars", 0) + e.get("reasoning_chars", 0) > 0), None)
        ttfts.append(ttft if ttft is not None else line["t_end_ns"])
    ratio = ttfts[1] / ttfts[0]
    camp.write({"kind": "nonce_probe", **meta, "doc": doc, "ttft_ns": ttfts,
                "ratio": round(ratio, 3), "passed": ratio >= 0.8})
    return client.OK


def argmax_check(camp, eng, entry):
    """Greedy effectif (llama.cpp, qui expose les logprobs) : chaque token choisi
    est le top-1 de sa distribution. Requête non streamée de 64 tokens, hors mesure."""
    request = entry["requests"][0]
    content = content_for(camp.prompts, request["prompt"], request["nonce"])
    body = request_body(camp.matrix, eng.engine, entry["sampling"], content, request["seed"])
    body.update({"stream": False, "max_tokens": 64, "logprobs": True, "top_logprobs": 1})
    body.pop("stream_options", None)
    status, resp = client.http_json("127.0.0.1", eng.port, "POST", "/v1/chat/completions", body)
    try:
        steps = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return {"http_status": status, "error": "logprobs absents"}
    non_argmax = [i for i, step in enumerate(steps)
                  if step.get("top_logprobs")
                  and step["logprob"] < step["top_logprobs"][0]["logprob"] - 1e-6]
    return {"http_status": status, "tokens": len(steps), "non_argmax": non_argmax[:10]}


def skip_rest(camp, meta, pending, cause):
    for request in pending:
        camp.write({"kind": "measure", **meta, **request, "status": SKIPPED, "cause": cause})


def entry_meta(entry, attempt=None):
    meta = {key: entry[key] for key in ("phase", "block", "position", "config", "class",
                                        "sampling")}
    meta["entry_key"] = entry_key(entry)
    if attempt is not None:
        meta["attempt"] = attempt
    return meta


def entry_key(entry):
    return f"{entry['phase']}|{entry['block']}|{entry['config']}"


class SurvivingEngine(SystemExit):
    """Un moteur dont la disparition n'est pas confirmée arrête la campagne."""


def run_entry(camp, entry):
    config = entry["config_spec"]
    if config.get("unsupported"):
        for request in entry["requests"]:
            camp.write({"kind": "measure", **entry_meta(entry), **request,
                        "status": UNSUPPORTED, "reason": config["unsupported"]})
        return
    pending = list(entry["requests"])
    for attempt in range(MAX_ATTEMPTS):
        meta = entry_meta(entry, attempt)
        state = cooldown(camp.matrix)
        if not wait_port_free(camp.matrix["port"]):
            skip_rest(camp, meta, pending, "port_busy")
            return
        name = (f"{entry['phase']}-b{entry['block']}-{entry['config']}"
                f"-r{camp.run_seq}-a{attempt}").replace(":", "_")
        eng = Engine(camp.matrix, entry, camp.run_dir / "logs" / f"{name}.log")
        status, retry = None, False
        try:
            ready = eng.wait_ready(camp.matrix["load_timeout_s"])
            load = {"kind": "load", **meta, "argv": eng.argv, "env": eng.env,
                    "thermal_before": state, "ready": ready,
                    "load_s": round((time.perf_counter_ns() - eng.started) / 1e9, 2),
                    "system_memory": memwatch.system_memory(), "log": str(eng.log_path)}
            if not ready:
                if eng.bind_failed():
                    load["status"], retry = "bind_failed", True
                elif not eng.alive() and eng.proc.returncode in (2, 3):
                    load["status"] = INVALID_CONFIG
                else:
                    load["status"] = eng.failure_status() if not eng.alive() else LOAD_FAILED
                camp.write(load)
                status = load["status"]
            else:
                camp.write(load)
                print(f"[{entry['phase']} b{entry['block']} p{entry['position']}] "
                      f"{entry['config']} prêt en {load['load_s']} s", flush=True)
                measure_meta = {**meta, "thermal_before_load": state["throttled"]}
                status = run_request(camp, eng, entry, entry["warmup"], "warmup", measure_meta)
                if status == client.OK and camp.matrix["engines"][eng.engine].get("nonce_probe"):
                    status = nonce_probe(camp, eng, entry, meta, measure_meta)
                while pending and status == client.OK:
                    status = run_request(camp, eng, entry, pending[0], "measure", measure_meta)
                    pending.pop(0)
                if entry["phase"] == "validate" and status == client.OK and eng.engine == "llama":
                    camp.write({"kind": "argmax_check", **meta, **argmax_check(camp, eng, entry)})
                    # Matière à prouver la nature de la reconstruction /tokenize.
                    camp.write({"kind": "slots_probe", **meta,
                                "slots": client.http_json("127.0.0.1", eng.port, "GET",
                                                          "/slots")})
                final = {"kind": "unload", **meta, "rusage": memwatch.rusage(eng.proc.pid),
                         "thermal_after": thermal()}
                # vmmap est lent : après les mesures, moteur encore chargé.
                if eng.alive():
                    final["file_backed"] = memwatch.file_backed(eng.proc.pid)
                camp.write(final)
        finally:
            gone = eng.stop()
            camp.write({"kind": "stopped", **meta, "group_gone": gone})
            if not gone:
                raise SurvivingEngine(f"moteur {entry['config']} encore vivant : campagne arrêtée")
        if not pending:
            return
        # Un timeout censure sa requête, un échec de bind se retente ; le
        # moteur est rechargé pour la suite. Tout autre échec clôt l'entrée.
        if status not in (client.TIMEOUT,) and not retry:
            skip_rest(camp, meta, pending, status)
            return
    skip_rest(camp, entry_meta(entry), pending, "max_attempts")


def run(run_dir):
    """Exécute le plan courant ; reprise idempotente (voir l'en-tête du module)."""
    run_dir = pathlib.Path(run_dir)
    plan_path = run_dir / "plan.json"
    plan_data = json.loads(plan_path.read_text())
    # La matrice est celle du tirage du plan : un plan de réglage repris après le
    # gel reste lié à matrix.json, pas à matrix.frozen.json.
    matrix_path = (run_dir / plan_data["matrix"] if plan_data["matrix"] == "matrix.frozen.json"
                   else HERE / "matrix.json")
    matrix = json.loads(matrix_path.read_text())
    prompts_path = run_dir / "prompts" / "prompts.json"
    manifest = json.loads(prompts_path.read_text())
    manifest["by_id"] = {p["id"]: p for p in manifest["prompts"]}
    hashes = {"plan_sha256": sha256_file(plan_path), "prompts_sha256": sha256_file(prompts_path),
              "matrix_sha256": sha256_file(matrix_path)}
    if hashes["matrix_sha256"] != plan_data["matrix_sha256"]:
        raise SystemExit("matrice modifiée depuis le tirage du plan")
    (run_dir / "logs").mkdir(exist_ok=True)
    results = run_dir / "results.jsonl"
    done, seen, run_seq = set(), {}, 1
    phases = set(plan_data["phases"])
    if results.exists():
        for raw in results.read_text().splitlines():
            line = json.loads(raw)
            if line.get("kind") == "run_start":
                run_seq = max(run_seq, line["run_seq"] + 1)
                overlap = phases & set(line.get("phases", []))
                if overlap and line["plan_sha256"] != hashes["plan_sha256"]:
                    raise SystemExit(f"plan différent pour {sorted(overlap)} : reprise refusée")
                if line["prompts_sha256"] != hashes["prompts_sha256"]:
                    raise SystemExit("prompts.json modifié depuis un run précédent")
            elif line.get("kind") == "entry_done":
                done.add(line["entry_key"])
            elif "entry_key" in line:
                seen.setdefault(line["entry_key"], set()).add(line["run_seq"])
    campaign = hashes["plan_sha256"][:12]
    with open(results, "a", encoding="utf-8") as out:
        camp = Campaign(run_dir, matrix, manifest, out, campaign, run_seq)
        camp.write({"kind": "run_start", "time": time.time(), "phases": sorted(phases),
                    **hashes, "resumed_entries": len(done & {entry_key(e) for e in
                                                               plan_data["entries"]}),
                    "collector_overhead_us": memwatch.collector_overhead(os.getpid()),
                    "system_memory": memwatch.system_memory()})
        for entry in plan_data["entries"]:
            key = entry_key(entry)
            if key in done:
                continue
            for old in sorted(seen.get(key, ())):
                camp.write({"kind": "superseded", "entry_key": key, "superseded_run_seq": old})
            run_entry(camp, entry)
            camp.write({"kind": "entry_done", "entry_key": key})


def freeze(run_dir):
    """Applique la règle de sélection aux mesures **admissibles** de la phase `tune`.

    Par famille : un candidat est retenu s'il a, pour chaque prompt de sa
    famille, toutes les répétitions du plan admissibles (preuves et, pour un
    spéculatif, identité greedy avec la référence AR `tune-ref:*`) et au moins
    une valeur de la métrique (exacte pour le débit). Score = moyenne géométrique, sur ces prompts, de la
    médiane de la métrique ; le meilleur remplace le défaut (premier candidat)
    seulement s'il le bat de plus de `tie_margin`. Une famille sans candidat
    admissible arrête le gel : aucune évaluation ne part d'un réglage non mesuré.
    """
    import analyze  # tokenizer HF : venv MLX requis

    run_dir = pathlib.Path(run_dir)
    matrix = json.loads((HERE / "matrix.json").read_text())
    plan_data = json.loads((run_dir / "plan-tune.json").read_text())
    expected_reps = {}
    for entry in plan_data["entries"]:
        for request in entry["requests"]:
            key = (entry["config"], request["prompt"])
            expected_reps[key] = expected_reps.get(key, 0) + 1
    tune_rows = [r for r in analyze.load(run_dir)[0] if r.get("phase") == "tune"]
    references = {(r["config"],) + analyze.request_key(r): r for r in tune_rows
                  if r["config"].startswith("tune-ref:")}
    rows = [r for r in tune_rows if r["config"].startswith("tune:")]
    values, admissible, only_identity = {}, {}, {}
    for row in rows:
        # Requête exclue par sa seule identité greedy (statut ok, toutes les
        # autres preuves vraies) : vraie divergence avec l'AR, pas un défaut.
        family = row["config"].split(":")[1]
        spec = matrix["tuning"][family]
        checks = row.get("proofs") or {}
        others_ok = all(checks.get(k) is True for k in
                        ("usage", "output", "prompt_tokens", "ids", "no_reuse"))
        m = row.get("m") or {}
        ref = references.get((f"tune-ref:{spec.get('reference')}",) + analyze.request_key(row))
        if (row.get("status") == client.OK and others_ok
                and checks.get("greedy_identity") is False
                and m.get("exact") is True and m.get("delivery_tok_s") is not None
                and ref is not None and ref.get("admissibility") == "admissible"):
            only_identity[row["config"]] = only_identity.get(row["config"], 0) + 1
        if row.get("admissibility") != "admissible":
            continue
        key = (row["config"], row["prompt"])
        admissible[key] = admissible.get(key, 0) + 1
        family = row["config"].split(":")[1]
        metric = matrix["tuning"][family]["metric"]
        m = row.get("m") or {}
        if m.get(metric) is None or (metric == "delivery_tok_s" and not m.get("exact")):
            continue
        values.setdefault(row["config"], {}).setdefault(row["prompt"], []).append(m[metric])
    report, failures = {}, []
    for family, spec in matrix["tuning"].items():
        lower = spec["metric"] != "delivery_tok_s"
        scores = {}
        for index in range(len(spec["candidates"])):
            config = f"tune:{family}:{index}"
            per_prompt = values.get(config, {})
            # Toutes les répétitions prévues admissibles (preuves et identité),
            # et au moins une valeur de la métrique par prompt.
            complete = all(admissible.get((config, p), 0) == expected_reps.get((config, p), -1)
                           and per_prompt.get(p) for p in spec["prompts"])
            if complete:
                scores[index] = analyze.geomean_of_medians(per_prompt)
        if not scores:
            # Tous les candidats mesurés en entier mais divergents de l'AR : le
            # réglage est impossible, pas la mesure. Le défaut documenté du
            # moteur est gardé sans réglage ; en évaluation, chaque requête
            # reste soumise à l'identité greedy. Tout autre cas arrête le gel.
            planned = {i: sum(n for (c, _), n in expected_reps.items()
                              if c == f"tune:{family}:{i}")
                       for i in range(len(spec["candidates"]))}
            divergent = all(only_identity.get(f"tune:{family}:{i}", 0) == planned[i] > 0
                            for i in planned)
            if not divergent:
                failures.append(family)
                report[family] = {"metric": spec["metric"], "scores": {}, "winner": None}
                continue
            report[family] = {"metric": spec["metric"], "scores": {}, "winner": 0,
                              "value": spec["candidates"][0], "default_admissible": False,
                              "reason": "aucun candidat identique à l'AR : défaut du moteur "
                                        "conservé sans réglage"}
            for config in matrix["configs"].values():
                if family in config.get("slots", {}):
                    config["slots"][family] = spec["candidates"][0]
            continue
        best = (min if lower else max)(scores, key=scores.get)
        default = scores.get(0)
        winner = best
        if default is not None and abs(scores[best] / default - 1) <= matrix["tie_margin"]:
            winner = 0
        report[family] = {"metric": spec["metric"], "scores": scores, "winner": winner,
                          "value": spec["candidates"][winner],
                          "default_admissible": default is not None}
        for config in matrix["configs"].values():
            if family in config.get("slots", {}):
                config["slots"][family] = spec["candidates"][winner]
    (run_dir / "freeze.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if failures:
        raise SystemExit(f"gel impossible, aucun candidat admissible : {failures}")
    (run_dir / "matrix.frozen.json").write_text(json.dumps(matrix, indent=1))


def capture(argv):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=60)
        # llama-server --version écrit sur stderr.
        return (out.stdout.strip() or out.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return repr(error)


def binary(name_or_path):
    """Chemin résolu et sha256 du binaire réellement exécuté."""
    found = shutil.which(str(name_or_path))
    if not found:
        return {"path": str(name_or_path), "missing": True}
    real = os.path.realpath(found)
    return {"path": real, "sha256": sha256_file(real)}


def environment(run_dir, name):
    venv = config_path("MLX_VENV")
    packages = sorted(p.name.removesuffix(".dist-info")
                      for p in (venv / "lib/python3.12/site-packages").glob("*.dist-info"))
    info = {
        "time": time.time(),
        "hw_model": capture(["sysctl", "-n", "hw.model"]),
        "cpu": capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memsize": capture(["sysctl", "-n", "hw.memsize"]),
        "sw_vers": capture(["sw_vers"]),
        "power_source": capture(["pmset", "-g", "batt"]).splitlines()[:1],
        "power_mode": [l.strip() for l in capture(["pmset", "-g"]).splitlines()
                       if "powermode" in l or "lowpowermode" in l],
        "thermal": thermal(),
        "llama_server": capture(["llama-server", "--version"]),
        "brew": capture(["brew", "list", "--versions", "llama.cpp", "llmd"]),
        "binaries": {"llama-server": binary("llama-server"), "llmd": binary("llmd"),
                     "saragossa": binary(config_path("SARAGOSSA_BIN")),
                     "mlx_python": binary(venv / "bin/python")},
        "mlx_venv_packages": packages,
        "saragossa_commit": capture(["/usr/bin/git", "-C", str(HERE), "rev-parse", "HEAD"]),
        "saragossa_dirty": capture(["/usr/bin/git", "-C", str(HERE), "status", "--porcelain"]),
        "rustc": capture(["rustc", "--version"]),
        "engine_env_keys": sorted(engine_env({})),
        "system_memory": memwatch.system_memory(),
    }
    path = pathlib.Path(run_dir) / f"env-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} existe déjà : un environnement n'est jamais écrasé")
    path.write_text(json.dumps(sanitize(info), indent=1))
    print(json.dumps(sanitize(info), indent=1))


def _self_check():
    _request_failure_check()
    a = nonce(1, "a", 0, "p", None, 0)
    assert a == nonce(1, "a", 0, "p", None, 0) and a != nonce(1, "a", 0, "p", None, 1)
    assert a != nonce(1, "a", 0, "p", 11, 0) and len(a) == 16 and a.isdigit()
    matrix, _ = load_matrix()
    for phase, spec in matrix["phases"].items():
        assert spec["class"] in matrix["classes"], phase
        for config in [] if spec["configs"] == "tuning" else spec["configs"]:
            assert config in matrix["configs"], (phase, config)
    for config_id, config in matrix["configs"].items():
        assert config["engine"] in matrix["engines"], config_id
        for slot in config.get("slots", {}):
            assert slot in matrix["tuning"], (config_id, slot)
    for family, spec in matrix["tuning"].items():
        assert family in matrix["configs"][spec["probe"]]["slots"], family
    spec = resolve(matrix, "llkm-mtp")
    assert spec["args"] == ["-b", "2048", "-ub", "512", "--spec-draft-n-max", "3"], spec
    spec = resolve(matrix, "sar-mtp", ("sar-mtp", {"env": {"SARAGOSSA_RUST_MTP_MAX_DRAFT": "3"}}))
    assert spec["env"] == {"SARAGOSSA_RUST_MTP_MAX_DRAFT": "3"}
    assert len(list(tuning_entries(matrix))) == sum(
        len(f["candidates"]) for f in matrix["tuning"].values())
    body = request_body(matrix, "mlx", "sampled", "x", 7)
    assert "seed" not in body and body["top_k"] == 20 and body["top_p"] == 1.0
    body = request_body(matrix, "saragossa", "greedy", "x", None)
    assert body["temperature"] == 0 and "seed" not in body
    env = engine_env({"X": "1"})
    assert env["X"] == "1" and env["HF_HUB_OFFLINE"] == "1"
    assert not set(env) - {"PATH", "HOME", "TMPDIR", "USER", "LANG", "HF_HUB_OFFLINE",
                           "TRANSFORMERS_OFFLINE", "BENCH_ENV", "X"} - set(config_paths())
    assert env["MODELS"] == config_paths()["MODELS"]
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        assert not port_free(busy.getsockname()[1])


def _request_failure_check():
    """Un refus HTTP conserve le corps de requête et reste enregistrable."""
    import contextlib
    import io
    from types import SimpleNamespace
    from unittest.mock import patch

    body = {"model": "m", "messages": [], "seed": 7}
    lines = []
    camp = SimpleNamespace(matrix={"engines": {"saragossa": {}}, "memory_period_s": 1,
                                   "deadline_s": 1}, prompts={}, write=lines.append)
    eng = SimpleNamespace(engine="saragossa", proc=SimpleNamespace(pid=0), port=0)
    request = {"prompt": "p", "nonce": "n", "seed": 7}
    cases = [(503, "file d'inférence pleine", REFUSED_BUSY),
             (503, {"error": "out of memory"}, OOM),
             (503, None, client.HTTP_ERROR), (503, [], client.HTTP_ERROR),
             (500, "plain text", client.HTTP_ERROR)]
    for status, error_body, expected in cases:
        record = {"status": client.HTTP_ERROR, "http_status": status,
                  "error_body": error_body, "t0_ns": 0, "t_end_ns": 0}
        with (patch(__name__ + ".content_for", return_value="x"),
              patch(__name__ + ".request_body", return_value=body),
              patch.object(memwatch, "rusage", return_value={}),
              patch.object(memwatch, "Sampler", return_value=contextlib.nullcontext(
                  SimpleNamespace(samples=[]))),
              patch.object(client, "chat_stream", return_value=record),
              contextlib.redirect_stdout(io.StringIO())):
            assert run_request(camp, eng, {"sampling": "greedy"}, request, "measure", {}) == expected
        assert lines[-1]["body_params"] == {"model": "m", "seed": 7}
        assert lines[-1]["body_sha256"] == hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()
        assert lines[-1]["error_body"] == error_body


def _freeze_check():
    """Gel sur données synthétiques : le candidat 2 bat le défaut et gagne ; le
    candidat 1, dont une répétition manque, est écarté ; une famille sans
    candidat admissible arrête le gel sans écrire de matrice gelée."""
    import contextlib
    import io
    import tempfile

    import analyze

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Out:
                ids = list(text)
            return Out()

    matrix, _ = load_matrix()
    prompts = matrix["tuning"]["sar-mtp"]["prompts"]

    def row(config, prompt, gap_ns, repeat):
        events = [{"t_ns": i * gap_ns, "content_chars": 1, "reasoning_chars": 0, "text": "x"}
                  for i in range(1, 11)]
        nonce_value = f"{repeat:016d}"
        return {"kind": "measure", "phase": "tune", "block": 0, "position": 0, "config": config,
                "prompt": prompt, "nonce": nonce_value, "seed": None, "repeat": repeat,
                "engine": "saragossa", "status": "ok", "events": events, "sampling": "greedy",
                "usage": {"completion_tokens": 10, "prompt_tokens": 5},
                "ingest": {"kind": "log", "prompt_fnv1a64": "f", "reused_prefix_tokens": "0"},
                "content": "x" * 10, "finish_reason": "length", "t0_wall": 0, "run_seq": 1,
                "entry_key": config}

    rows, plan_entries = [], []
    for index, gap in ((0, 100), (1, 50), (2, 40)):
        config = f"tune:sar-mtp:{index}"
        plan_entries.append({"config": config, "requests": [
            {"prompt": p, "repeat": r} for p in prompts for r in range(2)]})
        rows += [row(config, p, gap, r) for p in prompts for r in range(2)
                 if not (index == 1 and r == 1 and p == prompts[0])]
    # Référence AR du réglage : sans elle, aucun candidat spéculatif n'est admissible.
    rows += [row("tune-ref:sar-ar", p, 100, r) for p in prompts for r in range(2)]
    expected = {f"{p}|{r:016d}": {"prompt_tokens": 5, "fnv1a64": "f"}
                for p in prompts for r in range(2)}
    original = (analyze.tokenizer, analyze.load_matrix)
    eos_matrix = json.loads(json.dumps(matrix))
    eos_matrix["engines"]["saragossa"]["usage_counts_eos"] = False
    analyze.tokenizer = Tok
    analyze.load_matrix = lambda _run_dir: eos_matrix
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "prompts.json").write_text(json.dumps(
                {"prompts": [{"id": p, "kind": "code"} for p in prompts]}))
            (root / "prompts" / "expected.json").write_text(json.dumps(expected))
            (root / "plan-tune.json").write_text(json.dumps({"entries": plan_entries}))
            (root / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    freeze(tmp)
                    raise AssertionError("gel sans candidat admissible accepté")
                except SystemExit as error:
                    assert "aucun candidat admissible" in str(error)
            report = json.loads((root / "freeze.json").read_text())
            assert not (root / "matrix.frozen.json").exists()
    finally:
        analyze.tokenizer, analyze.load_matrix = original
    assert report["sar-mtp"]["winner"] == 2 and set(report["sar-mtp"]["scores"]) == {"0", "2"}
    assert report["llama-mtp"]["winner"] is None
    _freeze_divergent_check()


def _freeze_divergent_check():
    _freeze_identity_check("divergent")


def _freeze_missing_proof_check():
    for missing in ("reference", "reference_unverified", "identity", "exact_rate"):
        _freeze_identity_check(missing)


def _freeze_identity_check(case):
    """Une famille dont tous les candidats, mesurés en entier, divergent de l'AR
    garde le défaut sans arrêter le gel ; les autres familles restent soumises
    à la règle (ici, données absentes : arrêt)."""
    import contextlib
    import io
    import tempfile

    import analyze

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class Out:
                ids = list(text)
            return Out()

    matrix, _ = load_matrix()
    family = "llama-dflash"
    spec = matrix["tuning"][family]

    def row(config, prompt, repeat, text):
        events = [{"t_ns": (i + 1) * 100, "content_chars": 1, "reasoning_chars": 0, "text": ch}
                  for i, ch in enumerate(text)]
        return {"kind": "measure", "phase": "tune", "block": 0, "position": 0, "config": config,
                "prompt": prompt, "nonce": f"{repeat:016d}", "seed": None, "repeat": repeat,
                "engine": "llama", "status": "ok", "events": events, "sampling": "greedy",
                "usage": {"completion_tokens": len(text), "prompt_tokens": 5},
                "ingest": {"kind": "server_reconstruction", "prompt_fnv1a64": "f"},
                "timings": {"cache_n": 0}, "content": text, "finish_reason": "length",
                "t0_wall": 0, "run_seq": 1, "entry_key": config}

    rows, entries = [], []
    for index in range(len(spec["candidates"])):
        config = f"tune:{family}:{index}"
        entries.append({"config": config, "requests": [{"prompt": p, "repeat": r}
                                                       for p in spec["prompts"] for r in range(2)]})
        rows += [row(config, p, r, "xyw") for p in spec["prompts"]
                 for r in range(2)]
    rows += [row(f"tune-ref:{spec['reference']}", p, r, "xyz") for p in spec["prompts"]
             for r in range(2)]
    if case == "reference":
        rows = [r for r in rows if not r["config"].startswith("tune-ref:")]
    elif case == "reference_unverified":
        for r in rows:
            if r["config"].startswith("tune-ref:"):
                r["ingest"] = {}
    elif case == "exact_rate":
        for r in rows:
            if r["config"].startswith("tune:"):
                r["usage"]["completion_tokens"] += 1
    expected = {f"{p}|{r:016d}": {"prompt_tokens": 5, "fnv1a64": "f"}
                for p in spec["prompts"] for r in range(2)}
    original = (analyze.tokenizer, analyze.load_matrix)
    eos_matrix = json.loads(json.dumps(matrix))
    eos_matrix["engines"]["llama"]["usage_counts_eos"] = True
    eos_matrix["engines"]["llama"].pop("injected_content_prefix", None)
    analyze.tokenizer = Tok
    analyze.load_matrix = lambda _run_dir: eos_matrix
    original_identity = analyze.apply_identity
    if case == "identity":
        def missing_identity(measures, pair_of):
            original_identity(measures, pair_of)
            for measure in measures:
                if measure["config"].startswith("tune:"):
                    measure["proofs"].pop("greedy_identity", None)
                    measure["admissibility"] = "unverified"
        analyze.apply_identity = missing_identity
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "prompts").mkdir()
            (root / "prompts" / "prompts.json").write_text(json.dumps(
                {"prompts": [{"id": p, "kind": "code"} for p in spec["prompts"]]}))
            (root / "prompts" / "expected.json").write_text(json.dumps(expected))
            (root / "plan-tune.json").write_text(json.dumps({"entries": entries}))
            (root / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    freeze(tmp)
                except SystemExit as error:
                    assert (family not in str(error)) == (case == "divergent"), error
            report = json.loads((root / "freeze.json").read_text())
    finally:
        analyze.tokenizer, analyze.load_matrix = original
        analyze.apply_identity = original_identity
    if case == "divergent":
        assert report[family]["winner"] == 0 and "reason" in report[family]
    else:
        assert report[family]["winner"] is None and "reason" not in report[family]


if __name__ == "__main__":
    _self_check()
    _freeze_check()
    _freeze_missing_proof_check()
    if len(sys.argv) >= 3 and sys.argv[1] == "plan":
        plan(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "freeze":
        freeze(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "env":
        environment(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        print("run_matrix: auto-test ok")
    else:
        raise SystemExit(__doc__)
