"""Temps consommé par phase, en minutes, y compris les tentatives remplacées.

  python3 -I -B budget.py RUN_DIR

Lit results.jsonl ou results.jsonl.gz. Le total additionne les postes mesurés ;
il ne prétend pas inclure les arrêts de moteur, preuves HTTP et interruptions.
"""

import collections
import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from results import read_results  # noqa: E402

POSTS = ("pause", "chargement", "chauffe", "sondes", "prefill", "génération")


def budget(rows):
    phases = collections.defaultdict(collections.Counter)
    for row in rows:
        phase = row.get("phase")
        if not phase:
            continue
        values = phases[phase]
        kind = row["kind"]
        if kind == "load":
            values["chargement"] += row.get("load_s", 0)
            values["pause"] += (row.get("thermal_before", {}).get("waited_s", 0)
                                + row.get("busy_wait_s", 0))
        if "t_end_ns" not in row:
            continue
        duration = row["t_end_ns"] / 1e9
        if kind == "warmup":
            values["chauffe"] += duration
        elif kind in ("probe", "nonce_probe") or (kind == "measure" and row.get("ttft_only")):
            values["sondes"] += duration
        elif kind == "measure":
            first = next((e["t_ns"] / 1e9 for e in row.get("events", [])
                          if e.get("content_chars") or e.get("reasoning_chars")), duration)
            values["prefill"] += first
            values["génération"] += duration - first
    return {phase: {**{p: v[p] / 60 for p in POSTS},
                    "total": sum(v[p] for p in POSTS) / 60}
            for phase, v in sorted(phases.items())}


def render(values):
    columns = (*POSTS, "total")
    lines = ["Temps en minutes (total des postes enregistrés).", "",
             "| phase | " + " | ".join(columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for phase, row in values.items():
        lines.append(f"| {phase} | " + " | ".join(f"{row[p]:.2f}" for p in columns) + " |")
    return "\n".join(lines)


def load(run_dir):
    path = pathlib.Path(run_dir) / "results.jsonl"
    if path.exists():
        return read_results(path)
    with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _self_check():
    import tempfile

    rows = [{"phase": "eval-S", "kind": "load", "load_s": 120,
             "thermal_before": {"waited_s": 55}, "busy_wait_s": 5},
            {"phase": "eval-S", "kind": "warmup", "t_end_ns": 180e9},
            {"phase": "eval-S", "kind": "probe", "t_end_ns": 30e9},
            {"phase": "eval-S", "kind": "measure", "ttft_only": True, "t_end_ns": 30e9},
            {"phase": "eval-S", "kind": "measure", "t_end_ns": 600e9,
             "events": [{"t_ns": 1, "content_chars": 0},
                        {"t_ns": 240e9, "reasoning_chars": 1}]},
            {"phase": "eval-S", "kind": "nonce_probe", "passed": True},
            {"phase": "eval-L", "kind": "measure", "t_end_ns": 60e9, "events": []}]
    expected = {"pause": 1, "chargement": 2, "chauffe": 3, "sondes": 1,
                "prefill": 4, "génération": 6, "total": 17}
    assert budget(rows)["eval-S"] == expected
    assert budget(rows)["eval-L"]["prefill"] == 1
    assert budget(rows)["eval-L"]["génération"] == 0
    assert "| eval-S | 1.00 | 2.00 | 3.00 | 1.00 | 4.00 | 6.00 | 17.00 |" in render(budget(rows))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "results.jsonl"
        data = "\n".join(json.dumps(row) for row in rows) + "\n"
        path.write_text(data)
        assert load(tmp) == rows
        with gzip.open(path.with_suffix(".jsonl.gz"), "wt") as out:
            out.write(data)
        path.unlink()
        assert load(tmp) == rows


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        print("budget: auto-test ok")
    elif len(sys.argv) == 2:
        print(render(budget(load(sys.argv[1]))))
    else:
        raise SystemExit(__doc__)
