"""TTFT d'un tour d'agent : le cache de préfixe de `serve` évite-t-il le prefill ?

Conversation d'agent de type outil (system, outils, tours assistant/tool).
Le tour N+1 ajoute un court message tool. Quatre cas, un processus serveur
par configuration :
  a  cache froid : serveur neuf, sans SSD, tour N+1 d'emblée ;
  b  cache RAM chaud : tour N puis tour N+1 dans le même processus ;
  c  cache SSD : tour N, redémarrage du serveur (même répertoire SSD), tour N+1 ;
  d  contrôle : un message du milieu modifié, le cache ne doit pas servir
     (mesuré après b en RAM, et après c sur SSD).
Les tokens préremplis viennent de la ligne de journal `saragossa serve
completion` (prompt_tokens, reused_prefix_tokens, prefill_ms).

Usage (le pilote prend gpulock par passe, ne jamais l'appeler sous gpulock) :
  tour_agent_ttft.py all  OUT --bin BIN --model DIR --lock-tool tools/gpulock.py
  tour_agent_ttft.py pass OUT --bin BIN --model DIR --case cold|ram|ssd --tokens N
Une passe est jetée et rejouée si loadavg >= 2 ou si le Mac est sur batterie en fin
de passe ; après deux rejets, abandon : DONE porte `failed: ...` et le rc est non nul.
"""

import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "competitors"))
import client  # noqa: E402

HOST, PORT, KEY = "127.0.0.1", 18439, "bench"
TOOLS = [{"type": "function", "function": {
    "name": name, "description": f"Outil {name} de l'agent.",
    "parameters": {"type": "object", "properties": {"arg": {"type": "string"}},
                   "required": ["arg"]}}} for name in ("read_file", "grep", "run_tests", "edit")]
LOG_RE = re.compile(r"saragossa serve completion .*?prompt_tokens=(\d+) .*?prefill_ms=(\d+) "
                    r".*?reused_prefix_tokens=(\d+)")


def tool_output(turn, lines):
    # Texte déterministe et unique par tour, pour qu'aucun préfixe ne coïncide.
    return "\n".join(f"[{turn}.{i}] fn handler_{turn}_{i}(ctx: &Ctx, key: u64) -> Result<{(turn * 31 + i * 17) % 97}> "
                     f"{{ ctx.lookup(key ^ {turn * 7919 + i * 104729}) }}" for i in range(lines))


def conversation(turns, lines, edit_turn=None, tail="cargo test: 3 passed"):
    msgs = [{"role": "system", "content": "Tu es un agent de développement. Utilise les outils."},
            {"role": "user", "content": "Corrige le bug du module handler et lance les tests."}]
    for turn in range(turns):
        call = f"call_{turn}"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": call, "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"arg": f"src/h{turn}.rs"})}}]})
        body = tool_output(turn + (10_000 if turn == edit_turn else 0), lines)
        msgs.append({"role": "tool", "tool_call_id": call, "content": body})
    return msgs, tail


def with_new_tool_message(turns, lines, tail, edit_turn=None):
    msgs, _ = conversation(turns, lines, edit_turn)
    msgs.append({"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_last", "type": "function",
        "function": {"name": "run_tests", "arguments": "{\"arg\": \"all\"}"}}]})
    msgs.append({"role": "tool", "tool_call_id": "call_last", "content": tail})
    return msgs


def request(msgs):
    return {"model": "qwen38", "messages": msgs, "tools": TOOLS, "max_tokens": 8,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}


def ac_power():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    return "AC Power" in out


def quiet(deadline_s=1800):
    end = time.time() + deadline_s
    while time.time() < end:
        if os.getloadavg()[0] < 2 and ac_power():
            return
        time.sleep(15)
    raise SystemExit("loadavg >= 2 ou batterie : passe jetée")


class Server:
    def __init__(self, binary, model, log, ssd=None):
        env = {**os.environ, "SARAGOSSA_API_KEY": KEY}
        for key in [k for k in env if k.startswith("SARAGOSSA_") and k != "SARAGOSSA_API_KEY"]:
            del env[key]
        cmd = [binary, "serve", "--port", str(PORT), "--model", f"qwen38={model}", "--preload"]
        if ssd:
            cmd += ["--prefix-cache-ssd-dir", str(ssd), "--prefix-cache-ssd-gib", "64"]
        self.log = open(log, "ab")
        self.log_path = log
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=self.log, env=env)
        for _ in range(900):
            if self.proc.poll() is not None:
                raise SystemExit(f"serveur arrêté, voir {log}")
            try:
                if client.http_json(HOST, PORT, "GET", "/health", timeout=2)[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(1)
        raise SystemExit("serveur non prêt")

    def stop(self):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def turn(self, label, msgs):
        quiet()
        size = self.log_path.stat().st_size
        rec = client.chat_stream(HOST, PORT, request(msgs), {"Authorization": f"Bearer {KEY}"})
        first = next((e["t_ns"] for e in rec["events"]
                      if e.get("content_chars") or e.get("reasoning_chars")), None)
        time.sleep(1)
        with open(self.log_path, "rb") as fh:
            fh.seek(size)
            found = LOG_RE.findall(fh.read().decode("utf-8", "replace"))
        prompt, prefill, reused = map(int, found[-1]) if found else (None, None, None)
        out = {"label": label, "status": rec["status"],
               "ttft_s": None if first is None else first / 1e9, "prompt_tokens": prompt,
               "reused_prefix_tokens": reused, "prefilled_tokens": None if prompt is None else prompt - reused,
               "prefill_ms": prefill, "content": rec["content"] + rec["reasoning"]}
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return out


def calibrate(srv, lines):
    """Tokens par tour d'outil, mesurés sur deux conversations courtes."""
    sizes = []
    for turns in (2, 6):
        # Système différent : aucun préfixe partagé avec les cas mesurés.
        msgs = with_new_tool_message(turns, lines, "ok")
        msgs[0]["content"] = "Calibration."
        sizes.append(srv.turn(f"calib{turns}", msgs)["prompt_tokens"])
    return (sizes[1] - sizes[0]) / 4, sizes[0] - 2 * (sizes[1] - sizes[0]) / 4


def run_pass(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, tail = 40, "cargo test: 3 passed, 1 failed (handler_ok)"
    log = out / f"{args.case}-{args.tokens}.server.log"
    log.unlink(missing_ok=True)
    results = []
    ssd = pathlib.Path(tempfile.mkdtemp(prefix="tour-ssd-")) if args.case == "ssd" else None
    srv = Server(args.bin, args.model, log, ssd)
    try:
        per_turn, base = calibrate(srv, lines)
        turns = max(2, round((args.tokens - base) / per_turn))
        mid = turns // 2
        hist = lambda edit=None: with_new_tool_message(turns, lines, tail, edit)  # noqa: E731
        # Tour N : le même historique sans le message tool final.
        tour_n = hist()[:-2]
        if args.case == "cold":
            results.append(srv.turn("a_cold", hist()))
        elif args.case == "ram":
            results.append(srv.turn("n_warmup", tour_n))
            results.append(srv.turn("b_ram", hist()))
            results.append(srv.turn("d_ram_edit", hist(edit=mid)))
        else:
            results.append(srv.turn("n_warmup", tour_n))
            time.sleep(10)
            srv.stop()
            srv = Server(args.bin, args.model, log, ssd)
            results.append(srv.turn("c_ssd", hist()))
            results.append(srv.turn("d_ssd_edit", hist(edit=mid)))
    finally:
        srv.stop()
    doc = {"case": args.case, "target_tokens": args.tokens, "turns": turns, "results": results,
           "loadavg_end": os.getloadavg()[0], "ac": ac_power()}
    (out / f"{args.case}-{args.tokens}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))


MAX_REJECTS = 2


def pass_calm(path):
    """Charge et secteur relus en fin de passe : sinon la mesure est faussée."""
    doc = json.loads(path.read_text())
    return doc["loadavg_end"] < 2 and doc["ac"]


EXPECTED = {"cold": ["a_cold"], "ram": ["n_warmup", "b_ram", "d_ram_edit"],
            "ssd": ["n_warmup", "c_ssd", "d_ssd_edit"]}


def pass_failed(path):
    """Renvoie la cause d'échec (JSON illisible, tour manquant ou en échec), ou None."""
    try:
        doc = json.loads(path.read_text())
        labels = [r["label"] for r in doc["results"]]
        failed = next((r["label"] for r in doc["results"] if r.get("status") != "ok"), None)
        expected = EXPECTED[doc["case"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"JSON invalide ({exc})"
    if failed is not None:
        return f"tour {failed} en échec"
    if labels != expected:
        return f"tours {labels} au lieu de {expected}"
    return None


def run_case(args, out, case, tokens):
    """Joue une passe, la rejoue après au plus MAX_REJECTS rejets ; renvoie l'échec ou None."""
    path = out / f"{case}-{tokens}.json"
    # Le verrou est pris ici, par passe ; `pass` ne le rappelle jamais.
    cmd = [sys.executable, "-I", "-B", args.lock_tool, "run", "--owner", "parking", "--",
           sys.executable, "-I", "-B", __file__, "pass", str(out), "--bin", args.bin,
           "--model", args.model, "--case", case, "--tokens", str(tokens)]
    for attempt in range(MAX_REJECTS + 1):
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return f"{case}-{tokens}: passe rc={rc}"
        if (cause := pass_failed(path)) is not None:
            return f"{case}-{tokens}: {cause}"
        if pass_calm(path):
            return None
        path.rename(path.with_name(f"{case}-{tokens}.rejet{attempt}.json"))
        print(f"{case}-{tokens}: passe {attempt} rejetée (charge ou batterie)", file=sys.stderr)
    return f"{case}-{tokens}: {MAX_REJECTS + 1} passes rejetées (charge ou batterie), abandon"


def run_all(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Un ancien DONE ne doit pas survivre à une campagne interrompue.
    (out / "DONE").unlink(missing_ok=True)
    failure = None
    for tokens in (16000, 32000):
        for case in ("cold", "ram", "ssd"):
            path = out / f"{case}-{tokens}.json"
            # Une mesure reprise d'un dossier existant passe les mêmes contrôles.
            done = path.exists() and pass_failed(path) is None and pass_calm(path)
            if failure is None and not done:
                failure = run_case(args, out, case, tokens)
    (out / "DONE").write_text(f"failed: {failure}\n" if failure else "ok\n")
    if failure:
        sys.exit(failure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "pass"])
    ap.add_argument("out")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lock-tool")
    ap.add_argument("--case", choices=["cold", "ram", "ssd"])
    ap.add_argument("--tokens", type=int)
    args = ap.parse_args()
    (run_all if args.mode == "all" else run_pass)(args)


if __name__ == "__main__":
    main()
