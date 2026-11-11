#!/usr/bin/env python3
"""Run with python -I tools/e2e/run.py; --list never starts a server."""

import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# -I deliberately removes the script directory from sys.path. Only add our
# resolved sibling modules, never the caller's working directory or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contracts import require
from launcher import TARGETS, configuration, server, verify_mode
from matrix import CHAT, MESSAGES, Matrix
from transport import Clients, VERSIONS

OK, FAIL, UNSUPPORTED = "OK", "ÉCHEC", "NON SUPPORTÉ attendu"
MODE_SUBSET = {"simple", "stream", "stream_usage", "max_tokens", "stop", "stop_sequences", "seed_temperature_zero"}


def scrub(message, secrets):
    for value in sorted((str(s) for s in secrets if s), key=len, reverse=True):
        message = message.replace(value, "<local>")
    return message[:1500]


def execute_case(case, model, mode, secrets=()):
    row = {"endpoint": case.endpoint, "option": case.option, "model": model, "mode": mode,
           "status": OK, "executed": False, "detail": "", "seconds": 0.0}
    started = time.monotonic()
    try:
        if case.unsupported:
            row.update(status=UNSUPPORTED, detail=case.unsupported)
        if not case.unsupported or case.verify_rejection:
            require(callable(case.action), "cas exécutable sans action")
            row["executed"] = True
            result = case.action()
            if isinstance(result, dict):
                row["metrics"] = result
    except Exception as exc:
        row["status"] = FAIL
        # SDK errors may quote a request or a local model path. Keep their type
        # and status only; our assertion messages contain no generated content.
        if type(exc).__module__.startswith(("openai", "anthropic")):
            detail = f"{type(exc).__name__}: HTTP {getattr(exc, 'status_code', '?')}"
        else:
            detail = f"{type(exc).__name__}: {exc}"
        row["detail"] = scrub(detail, secrets)
    row["seconds"] = round(time.monotonic() - started, 3)
    return row


def markdown(report):
    lines = ["# Conformité saragossa serve", "", f"Révision : `{report['revision']}`. Suite : `{report['suite']}`.",
             "", "| Endpoint | Option | Modèle | Mode | Résultat | Exécuté | Détail |",
             "|---|---|---|---|---|---|---|"]
    for row in report["results"]:
        values = [row[k] for k in ("endpoint", "option", "model", "mode", "status")]
        values += ["oui" if row["executed"] else "non", row["detail"]]
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    lines += ["", ", ".join(f"{key}: {value}" for key, value in report["counts"].items()), ""]
    for diagnostic in report.get("server_diagnostics", []):
        lines += [f"## Diagnostic serveur — {diagnostic['mode']}", "",
                  f"Prêt : {diagnostic.get('ready', False)}. Code de sortie après arrêt : "
                  f"{diagnostic.get('returncode')}. Commande (arguments JSON) :", "",
                  "    " + json.dumps(diagnostic.get("command", []), ensure_ascii=False), "",
                  "Dernières lignes stdout/stderr (40 lignes, 8 Kio maximum, chemins masqués) :", ""]
        lines += ["    " + line for line in diagnostic.get("log_tail", "").splitlines()] or ["    (journal vide)"]
        lines.append("")
    return "\n".join(lines)


def write_report(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    report["counts"] = dict(Counter(row["status"] for row in report["results"]))
    for name, content in (("report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
                          ("report.md", markdown(report))):
        temporary = directory / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(directory / name)


def revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="binaire local saragossa")
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--modes", default="ar", help="ar,mtp,dflash; un processus neuf par mode")
    parser.add_argument("--suite", choices=("full", "smoke"), default="full")
    parser.add_argument("--report-dir", type=Path, default=Path("tools/e2e/reports/latest"))
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--word-similarity", type=float, default=0.6)
    parser.add_argument("--list", action="store_true", help="inventaire statique; aucun SDK, poids ou serveur")
    args = parser.parse_args(argv)
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or any(m not in {"ar", "mtp", "dflash"} for m in modes):
        parser.error("--modes attend une liste sans doublons parmi ar,mtp,dflash")
    if not 0 < args.word_similarity <= 1 or args.request_timeout <= 0 or args.startup_timeout <= 0:
        parser.error("seuil dans ]0,1], délais strictement positifs")
    if args.list:
        matrix = Matrix(None, args.target, TARGETS[args.target][1])
        for mode in modes:
            if mode != "ar" and args.target not in {"qwen38", "qwen36_moe"}:
                print(f"serve\tmode\t{args.target}\t{mode}\tprofil sans tête MTP qualifiée")
                continue
            for case in matrix.selected(args.suite):
                if mode == "ar" or (case.endpoint in {CHAT, MESSAGES} and case.option in MODE_SUBSET):
                    disposition = case.unsupported or "à exécuter"
                    print(f"{case.endpoint}\t{case.option}\t{args.target}\t{mode}\t{disposition}")
        return 0
    if args.binary is None or not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary doit désigner un exécutable local")
    report = {"schema_version": 1, "revision": revision(), "suite": args.suite,
              "sdk_versions_required": VERSIONS, "sdk_versions_verified": False,
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "results": [], "server_diagnostics": []}
    signal.signal(signal.SIGTERM, interrupted)
    secrets = [str(Path.home()), str(Path.cwd()), str(args.binary.resolve())]
    secrets.extend(str(Path(value).expanduser().resolve()) for key, value in os.environ.items()
                   if key.startswith("E2E_") and value and
                   (key.endswith(("_MODEL", "_SIDECAR", "_DIR")) or key == "E2E_AUDIO_WAV"))
    # Never permit the report destination to write inside any configured model.
    output = args.report_dir.resolve()
    model_roots = [Path(v).expanduser().resolve() for k, v in os.environ.items()
                   if k.startswith("E2E_") and (k.endswith("_MODEL") or k.endswith("_DFLASH_DIR")) and v]
    repo_models = Path(__file__).resolve().parents[2] / "models"
    require(not any(output == p or p in output.parents for p in [*model_roots, repo_models.resolve()]),
            "rapport interdit dans un répertoire de modèle")
    try:
        for mode in modes:
            if mode != "ar" and args.target not in {"qwen38", "qwen36_moe"}:
                report["results"].append({"endpoint": "serve", "option": "mode", "model": args.target,
                    "mode": mode, "status": UNSUPPORTED, "executed": False, "seconds": 0,
                    "detail": "src/serve/mtp.rs: profil sans tête MTP qualifiée; DFlash dépend du chemin MTP."})
                continue
            try:
                family, flint, server_args, env, paths = configuration(args.target, mode)
                secrets.extend(str(path) for path in paths.values())
                diagnostics = {"mode": mode}
                report["server_diagnostics"].append(diagnostics)
                with server(args.binary.resolve(), server_args, env, args.startup_timeout,
                            diagnostics, secrets) as (base, key, log, process):
                    secrets.append(key)
                    clients = Clients(base, key, args.request_timeout)
                    report["sdk_versions_verified"] = True
                    try:
                        matrix = Matrix(clients, args.target, family, flint, args.word_similarity)
                        selected = matrix.selected(args.suite)
                        if mode != "ar":
                            selected = [case for case in selected if case.endpoint in {CHAT, MESSAGES}
                                        and case.option in MODE_SUBSET]
                        for case in selected:
                            row = execute_case(case, args.target, mode, secrets)
                            report["results"].append(row)
                            print(f"{row['status']}: {mode} {case.endpoint} {case.option}", flush=True)
                            write_report(output, report)
                        if family == "chat":
                            from matrix import Case
                            row = execute_case(Case("serve", "mode_verified", lambda: verify_mode(mode, log, args.target)),
                                               args.target, mode, secrets)
                            report["results"].append(row)
                        require(process.poll() is None, "serve arrêté pendant la matrice")
                    finally:
                        clients.close()
            except Exception as exc:
                report["results"].append({"endpoint": "serve", "option": "lifecycle", "model": args.target,
                    "mode": mode, "status": FAIL, "executed": False, "seconds": 0,
                    "detail": scrub(f"{type(exc).__name__}: {exc}", secrets)})
            write_report(output, report)
    except KeyboardInterrupt:
        report["results"].append({"endpoint": "serve", "option": "interrupted", "model": args.target,
            "mode": mode, "status": FAIL, "executed": False, "seconds": 0, "detail": "Exécution interrompue; serveur arrêté."})
    finally:
        report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_report(output, report)
    print(f"Rapports : {args.report_dir / 'report.json'} et {args.report_dir / 'report.md'}")
    return int(any(row["status"] == FAIL for row in report["results"]))


if __name__ == "__main__":
    sys.exit(main())
