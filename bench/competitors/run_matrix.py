"""Ordonnanceur du bench concurrents (stdlib seule).

  python3 -I run_matrix.py env RUN_DIR             # versions, machine, alimentation
  python3 -I run_matrix.py plan RUN_DIR PHASE...   # tire et écrit RUN_DIR/plan.json
  python3 -I run_matrix.py run RUN_DIR             # exécute le plan, écrit results.jsonl
  python3 -I run_matrix.py freeze RUN_DIR          # règle de sélection -> matrix.frozen.json

L'unité de randomisation est la configuration chargée : dans chaque bloc,
l'ordre des configurations est permuté, puis tous les prompts d'une
configuration passent à la suite dans un ordre tiré. Le tirage dépend de la
seule graine de `matrix.json`, publiée avec le plan avant toute mesure.

Le GPU est exclusif : le lanceur de la campagne prend le verrou GPU avant
`run` et le rend après ; ce script ne lance qu'un moteur à la fois.
"""

import hashlib
import json
import math
import os
import pathlib
import random
import signal
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import client  # noqa: E402
import memwatch  # noqa: E402
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


def load_matrix(run_dir=None):
    """Matrice gelée du run si elle existe (après `freeze`), sinon celle du dépôt."""
    frozen = pathlib.Path(run_dir) / "matrix.frozen.json" if run_dir else None
    path = frozen if frozen and frozen.exists() else HERE / "matrix.json"
    return json.loads(path.read_text())


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
    """(id, famille, configuration) de chaque candidat de la grille de réglage."""
    for family, spec in matrix["tuning"].items():
        for index, candidate in enumerate(spec["candidates"]):
            yield (f"tune:{family}:{index}", spec["prompts"],
                   resolve(matrix, spec["probe"], (family, candidate)))


def nonce(seed, phase, block, prompt):
    digest = hashlib.sha256(f"{seed}|{phase}|{block}|{prompt}".encode()).hexdigest()
    return f"{int(digest, 16) % 10**16:016d}"


def plan(run_dir, phases):
    matrix = load_matrix(run_dir)
    entries = []
    for phase in phases:
        spec = matrix["phases"][phase]
        rng = random.Random(f"{matrix['seed']}:{phase}")
        cls = matrix["classes"][spec["class"]]
        seeds = spec.get("seeds", [None])
        if spec["configs"] == "tuning":
            candidates = list(tuning_entries(matrix))
        else:
            candidates = [(c, spec.get("prompts", cls["prompts"]), resolve(matrix, c))
                          for c in spec["configs"]]
        for block in range(spec["blocks"]):
            rng.shuffle(candidates)
            for position, (config, prompts, config_spec) in enumerate(candidates):
                order = list(prompts)
                rng.shuffle(order)
                requests = [
                    {"prompt": p, "nonce": nonce(matrix["seed"], phase, block, p), "seed": s,
                     "repeat": r}
                    for p in order for s in seeds for r in range(spec.get("repeats", 1))
                ]
                warmup = {"prompt": cls["warmup"], "nonce": nonce(matrix["seed"], phase, block,
                                                                  "warmup"),
                          "seed": seeds[0], "repeat": 0}
                entries.append({"phase": phase, "block": block, "position": position,
                                "config": config, "config_spec": config_spec,
                                "class": spec["class"], "ctx": cls["ctx"],
                                "sampling": spec["sampling"], "warmup": warmup,
                                "requests": requests})
    out = {"seed": matrix["seed"], "phases": phases, "entries": entries,
           "requests": [r for e in entries for r in [e["warmup"], *e["requests"]]]}
    path = pathlib.Path(run_dir) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1)
    path.write_text(text)
    # Copie nommée : `plan.json` est le plan courant, chaque plan publié reste.
    (path.parent / f"plan-{'+'.join(phases)}.json").write_text(text)
    n = sum(len(e["requests"]) for e in entries)
    print(f"{len(entries)} chargements, {n} requêtes mesurées, "
          f"sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def request_body(matrix, engine, sampling, content, seed):
    dialect = matrix["engines"][engine]
    params = dict(matrix["sampling"][sampling])
    body = {"model": dialect["model"], "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": matrix["max_tokens"],
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


class Engine:
    """Processus moteur lancé par ce script, dans son propre groupe de processus."""

    def __init__(self, matrix, entry, log_path):
        config = entry["config_spec"]
        self.matrix = matrix
        self.config = config
        self.engine = config["engine"]
        self.port = matrix["port"]
        # Arguments par classe de contexte du moteur (fractions mémoire de llmd).
        class_args = matrix["engines"][self.engine].get("class_args", {}).get(entry["class"], [])
        argv = [str(HERE / config["launch"][0]), *config["launch"][1:],
                str(self.port), str(entry["ctx"]), *config["args"], *class_args]
        env = {**os.environ, **config["env"]}
        self.argv = argv
        self.log_path = log_path
        self.log = open(log_path, "ab")
        self.started = time.perf_counter_ns()
        self.proc = subprocess.Popen(argv, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

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

    def failure_status(self):
        text = self.log_text().lower()
        if any(marker in text for marker in OOM_MARKERS) or self.proc.returncode == -9:
            return OOM
        return CRASH

    def stop(self):
        """SIGTERM au groupe, SIGKILL après 30 s ; vérifie que le PID a disparu."""
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        self.log.close()
        return memwatch.rusage(self.proc.pid) is None


def ingest_proof(matrix, engine, port, body, eng):
    """Ids réellement ingérés quand le moteur permet de les lire, hors chronométrage."""
    kind = matrix["engines"][engine]["ingest"]
    if kind == "log":
        lines = [line for line in eng.log_text().splitlines()
                 if "saragossa serve completion" in line]
        if not lines:
            return {"kind": "log", "error": "ligne de complétion absente"}
        fields = dict(item.split("=", 1) for item in lines[-1].split() if "=" in item)
        return {"kind": "log", **fields}
    if kind == "tokenize":
        status, rendered = client.http_json("127.0.0.1", port, "POST", "/apply-template", {
            "messages": body["messages"],
            "chat_template_kwargs": {"enable_thinking": False}})
        if status != 200 or not isinstance(rendered, dict):
            return {"kind": "tokenize", "error": f"/apply-template {status}"}
        status, tokens = client.http_json("127.0.0.1", port, "POST", "/tokenize", {
            "content": rendered["prompt"], "add_special": True, "parse_special": True})
        if status != 200 or not isinstance(tokens, dict):
            return {"kind": "tokenize", "error": f"/tokenize {status}"}
        ids = tokens["tokens"]
        return {"kind": "tokenize", "prompt_tokens": len(ids), "prompt_fnv1a64": fnv1a64(ids)}
    return {"kind": "offline"}


def run_request(matrix, eng, entry, request, prompts, kind, out):
    engine = eng.engine
    content = content_for(prompts, request["prompt"], request["nonce"])
    body = request_body(matrix, engine, entry["sampling"], content, request["seed"])
    headers = matrix["engines"][engine].get("headers")
    before = memwatch.rusage(eng.proc.pid)
    with memwatch.Sampler(eng.proc.pid, matrix["memory_period_s"]) as sampler:
        record = client.chat_stream("127.0.0.1", eng.port, body, headers,
                                    deadline_s=matrix["deadline_s"])
    after = memwatch.rusage(eng.proc.pid)
    if record["status"] in (client.CONNECTION_LOST, client.STREAM_INCOMPLETE) and not eng.alive():
        record["status"] = eng.failure_status()
    line = {
        "kind": kind, "phase": entry["phase"], "block": entry["block"],
        "position": entry["position"], "config": entry["config"], "engine": engine,
        "class": entry["class"], "sampling": entry["sampling"], **request,
        "body_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "body_params": {k: v for k, v in body.items() if k != "messages"},
        "memory": {"before": before, "after": after,
                   "sampled_max_phys_footprint": max((v for _, v in sampler.samples),
                                                     default=None),
                   "samples": len(sampler.samples)},
        **record,
    }
    if record["status"] == client.OK:
        line["ingest"] = ingest_proof(matrix, engine, eng.port, body, eng)
    out.write(json.dumps(line, ensure_ascii=False) + "\n")
    out.flush()
    print(f"  {kind:7} {request['prompt']:16} {record['status']:17} "
          f"{record['t_end_ns'] / 1e9:8.1f} s", flush=True)
    return record["status"]


MAX_ATTEMPTS = 3


def argmax_check(matrix, eng, entry, prompts):
    """Greedy effectif (llama.cpp, qui expose les logprobs) : chaque token choisi
    est le top-1 de sa distribution. Requête non streamée de 64 tokens, hors mesure."""
    request = entry["requests"][0]
    content = content_for(prompts, request["prompt"], request["nonce"])
    body = request_body(matrix, eng.engine, entry["sampling"], content, request["seed"])
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


def skip_rest(out, entry, pending, cause):
    for request in pending:
        out.write(json.dumps({"kind": "measure", **entry_meta(entry), **request,
                              "status": SKIPPED, "cause": cause}) + "\n")


def run_entry(matrix, entry, prompts, run_dir, out):
    config = entry["config_spec"]
    if config.get("unsupported"):
        for request in entry["requests"]:
            out.write(json.dumps({"kind": "measure", **entry_meta(entry), **request,
                                  "status": UNSUPPORTED, "reason": config["unsupported"]}) + "\n")
        return
    pending = list(entry["requests"])
    for attempt in range(MAX_ATTEMPTS):
        state = cooldown(matrix)
        name = f"{entry['phase']}-b{entry['block']}-{entry['config']}-{attempt}".replace(":", "_")
        log_path = run_dir / "logs" / f"{name}.log"
        eng = Engine(matrix, entry, log_path)
        ready = eng.wait_ready(matrix["load_timeout_s"])
        load = {"kind": "load", **entry_meta(entry), "attempt": attempt, "argv": eng.argv,
                "env": config["env"], "thermal_before": state, "ready": ready,
                "load_s": round((time.perf_counter_ns() - eng.started) / 1e9, 2),
                "system_memory": memwatch.system_memory(), "log": str(log_path)}
        if not ready:
            load["status"] = eng.failure_status() if not eng.alive() else LOAD_FAILED
            if not eng.alive() and eng.proc.returncode in (2, 3):
                load["status"] = INVALID_CONFIG
            eng.stop()
            out.write(json.dumps(load) + "\n")
            skip_rest(out, entry, pending, load["status"])
            return
        out.write(json.dumps(load) + "\n")
        print(f"[{entry['phase']} b{entry['block']} p{entry['position']}] {entry['config']} "
              f"prêt en {load['load_s']} s", flush=True)
        status = run_request(matrix, eng, entry, entry["warmup"], prompts, "warmup", out)
        while pending and status == client.OK:
            status = run_request(matrix, eng, entry, pending[0], prompts, "measure", out)
            pending.pop(0)
        if entry["phase"] == "validate" and status == client.OK and eng.engine == "llama":
            out.write(json.dumps({"kind": "argmax_check", **entry_meta(entry),
                                  **argmax_check(matrix, eng, entry, prompts)}) + "\n")
        final = {"kind": "unload", **entry_meta(entry), "attempt": attempt,
                 "rusage": memwatch.rusage(eng.proc.pid), "thermal_after": thermal()}
        # vmmap est lent : après les mesures, moteur encore chargé.
        if eng.alive():
            final["file_backed"] = memwatch.file_backed(eng.proc.pid)
        final["stopped_cleanly"] = eng.stop()
        out.write(json.dumps(final) + "\n")
        out.flush()
        if not pending:
            return
        # Un timeout censure sa requête ; le moteur est rechargé pour la suite.
        if status != client.TIMEOUT:
            skip_rest(out, entry, pending, status)
            return
    skip_rest(out, entry, pending, "max_attempts")


def entry_meta(entry):
    return {key: entry[key] for key in ("phase", "block", "position", "config", "class",
                                        "sampling")}


def entry_key(entry):
    return f"{entry['phase']}|{entry['block']}|{entry['config']}"


def run(run_dir):
    """Exécute le plan ; reprend après le dernier chargement terminé si le run existe."""
    run_dir = pathlib.Path(run_dir)
    matrix = load_matrix(run_dir)
    plan_data = json.loads((run_dir / "plan.json").read_text())
    manifest = json.loads((run_dir / "prompts" / "prompts.json").read_text())
    manifest["by_id"] = {p["id"]: p for p in manifest["prompts"]}
    (run_dir / "logs").mkdir(exist_ok=True)
    results = run_dir / "results.jsonl"
    done = set()
    if results.exists():
        for raw in results.read_text().splitlines():
            line = json.loads(raw)
            if line.get("kind") == "entry_done":
                done.add(line["key"])
    with open(results, "a", encoding="utf-8") as out:
        out.write(json.dumps({"kind": "run_start", "time": time.time(),
                              "plan_sha256": hashlib.sha256(
                                  (run_dir / "plan.json").read_bytes()).hexdigest(),
                              "resumed_entries": len(done),
                              "collector_overhead_us": memwatch.collector_overhead(os.getpid()),
                              "system_memory": memwatch.system_memory()}) + "\n")
        for entry in plan_data["entries"]:
            if entry_key(entry) in done:
                continue
            run_entry(matrix, entry, manifest, run_dir, out)
            out.write(json.dumps({"kind": "entry_done", "key": entry_key(entry)}) + "\n")
            out.flush()


def freeze(run_dir):
    """Applique la règle de sélection aux résultats de la phase `tune`.

    Par famille : score de chaque candidat = moyenne géométrique, sur les
    prompts de la famille, de la médiane de la métrique ; le meilleur
    remplace le défaut (premier candidat) seulement s'il le bat de plus de
    `tie_margin`. Un candidat dont une requête a échoué est écarté.
    """
    from analyze import derive

    run_dir = pathlib.Path(run_dir)
    matrix = json.loads((HERE / "matrix.json").read_text())
    values, failed = {}, set()
    for raw in (run_dir / "results.jsonl").read_text().splitlines():
        line = json.loads(raw)
        if line.get("kind") != "measure" or line.get("phase") != "tune":
            continue
        if line.get("status") != client.OK:
            failed.add(line["config"])
            continue
        _, family, index = line["config"].split(":")
        metric = matrix["tuning"][family]["metric"]
        value = derive(line, {}, None).get(metric)
        if value is not None:
            values.setdefault((family, int(index)), {}).setdefault(line["prompt"], []).append(value)
    report = {}
    for family, spec in matrix["tuning"].items():
        lower = spec["metric"] != "delivery_tok_s"
        scores = {}
        for index in range(len(spec["candidates"])):
            per_prompt = values.get((family, index), {})
            if f"tune:{family}:{index}" in failed or set(per_prompt) != set(spec["prompts"]):
                continue
            logs = [math.log(statistics.median(v)) for v in per_prompt.values()]
            scores[index] = math.exp(sum(logs) / len(logs))
        winner = 0
        if scores:
            best = (min if lower else max)(scores, key=scores.get)
            default = scores.get(0)
            margin = matrix["tie_margin"]
            if default is None or (abs(scores[best] / default - 1) > margin and best != 0):
                winner = best
        report[family] = {"metric": spec["metric"], "scores": scores, "winner": winner,
                          "value": spec["candidates"][winner]}
        for config in matrix["configs"].values():
            if family in config.get("slots", {}):
                config["slots"][family] = spec["candidates"][winner]
    (run_dir / "matrix.frozen.json").write_text(json.dumps(matrix, indent=1))
    (run_dir / "freeze.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def capture(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return repr(error)


def environment(run_dir):
    venv = pathlib.Path("/Users/ludwig/workspace/saragossa-oss/models/mlxenv")
    packages = sorted(p.name.removesuffix(".dist-info")
                      for p in (venv / "lib/python3.12/site-packages").glob("*.dist-info"))
    info = {
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
        "mlx_venv_python": capture([str(venv / "bin/python"), "-I", "--version"]),
        "mlx_venv_packages": packages,
        "saragossa_commit": capture(["/usr/bin/git", "-C", str(HERE), "rev-parse", "HEAD"]),
        "saragossa_dirty": capture(["/usr/bin/git", "-C", str(HERE), "status", "--porcelain"]),
        "rustc": capture(["rustc", "--version"]),
        "system_memory": memwatch.system_memory(),
    }
    path = pathlib.Path(run_dir) / "env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def _self_check():
    assert nonce(1, "a", 0, "p") == nonce(1, "a", 0, "p")
    assert nonce(1, "a", 0, "p") != nonce(1, "a", 1, "p")
    assert len(nonce(1, "a", 0, "p")) == 16 and nonce(1, "a", 0, "p").isdigit()
    matrix = load_matrix()
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


def _freeze_check():
    """Le candidat 2 bat le défaut de plus de la marge : il est retenu ; un
    candidat en échec est écarté ; une famille sans mesure garde son défaut."""
    import contextlib
    import io
    import tempfile

    def line(config, prompt, gap_ns, status="ok"):
        events = [{"t_ns": i * gap_ns, "content_chars": 1, "reasoning_chars": 0}
                  for i in range(1, 11)]
        return {"kind": "measure", "phase": "tune", "config": config, "prompt": prompt,
                "nonce": "0", "engine": "saragossa", "status": status, "events": events,
                "usage": {"completion_tokens": 10}, "content": "x" * 10}

    prompts = load_matrix()["tuning"]["sar-mtp"]["prompts"]
    rows = [line(f"tune:sar-mtp:{i}", p, gap) for i, gap in ((0, 100), (1, 50), (2, 40))
            for p in prompts]
    rows.append(line("tune:sar-mtp:1", prompts[0], 50, status="timeout"))
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        with contextlib.redirect_stdout(io.StringIO()):
            freeze(tmp)
        report = json.loads(pathlib.Path(tmp, "freeze.json").read_text())
    assert report["sar-mtp"]["winner"] == 2 and set(report["sar-mtp"]["scores"]) == {"0", "2"}
    assert report["llama-mtp"]["winner"] == 0


if __name__ == "__main__":
    _self_check()
    _freeze_check()
    if len(sys.argv) >= 3 and sys.argv[1] == "plan":
        plan(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "freeze":
        freeze(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "env":
        environment(sys.argv[2])
    elif len(sys.argv) == 1:
        print("run_matrix: auto-test ok")
    else:
        raise SystemExit(__doc__)
