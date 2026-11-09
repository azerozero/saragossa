"""Auto-tests des lanceurs sur copies et exécutables factices, sans moteur."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def check_power():
    # Même politique et même tolérance aux erreurs de pmset que les mesures.
    sys.path.insert(0, str(HERE))
    from run_matrix import require_power
    from types import SimpleNamespace

    require_power(SimpleNamespace(write=lambda row: print(json.dumps(row))))


def check_partial_freeze(run_dir):
    path = pathlib.Path(run_dir) / "matrix.frozen.json"
    try:
        matrix = json.loads(path.read_text())
        valid = bool(matrix.get("phases", {}).get("eval-S", {}).get("prompt_sets"))
    except (OSError, ValueError, AttributeError):
        valid = False
    if not valid:
        print("partial exige matrix.frozen.json avec phases.eval-S.prompt_sets ; "
              "utiliser run_matrix.py reuse-freeze RUN_DIR ANCIEN_GEL", file=sys.stderr)
        raise SystemExit(2)


def _self_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        trace = root / "calls"
        bindir = root / "bin"
        bindir.mkdir()
        fake = bindir / "python3"
        runner = ('#!/bin/bash\nprintf "%s\\n" "$*" >> ' + str(trace) + '\n'
                  'if [[ "$3" == */night_checks.py ]]; then\n'
                  f'  exec "{sys.executable}" -I -B "{HERE}/night_checks.py" "${{@:4}}"\nfi\n')
        fake.write_text(runner)
        fake.chmod(0o755)
        shutil.copy2(fake, bindir / "python")
        quality = root / "quality_llama.sh"
        shutil.copy2(HERE / "quality_llama.sh", quality)
        quality.chmod(0o755)
        for name in ("llama-server", "curl", "llama-perplexity", "llama-bench"):
            path = bindir / name
            path.write_text('#!/bin/bash\n' + (f'echo {name} >> "{trace}"\n'
                            if name in ("llama-perplexity", "llama-bench") else '') + 'exit 0\n')
            path.chmod(0o755)
        counter, failure = root / "power-count", root / "power-failure"
        pmset = bindir / "pmset"
        pmset.write_text(f'''#!/bin/bash
if [[ "$2" == batt ]]; then
  n=$(cat "{counter}"); n=$((n + 1)); echo "$n" > "{counter}"
  if ((n >= $(cat "{failure}"))); then echo 'Battery Power'; else echo 'AC Power'; fi
else
  echo 'lowpowermode 0'
fi
''')
        pmset.chmod(0o755)
        counter.write_text("0")
        failure.write_text("999")
        for name in ("night.sh", "paths.sh", "launch_saragossa.sh", "launch_mlx.sh"):
            shutil.copy2(HERE / name, root / name)
        envfile = root / "local.env"
        envfile.write_text(f'MODELS={root}\nAI_MODEL={root}\nRUST_SRC={root}\n'
                           f'CORPUS_DIR={root}\nMLX_VENV={root}\nMLX_CKPT={root}/default\n'
                           f'SARAGOSSA_BIN={fake}\n')
        env = {**os.environ, "BENCH_ENV": str(envfile), "PATH": f"{bindir}:{os.environ['PATH']}",
               "BENCH_MLX_CKPT": str(root / "changed"), "GPU_LOCK": ""}
        run = root / "run"
        run.mkdir()

        def night(which="partial"):
            return subprocess.run(["bash", str(root / "night.sh"), which, str(run)],
                                  env=env, capture_output=True, text=True, timeout=10)

        rejected = night()
        assert rejected.returncode == 2 and "matrix.frozen.json" in rejected.stderr
        assert trace.read_text().splitlines() == [f"-I -B {root}/run_matrix.py", f"-I -B {root}/analyze.py",
                                                 f"-I -B {root}/night_checks.py partial-freeze {run}"]
        trace.write_text("")
        frozen = run / "matrix.frozen.json"
        frozen.write_text(json.dumps({"phases": {"eval-S": {}}}))
        rejected = night()
        assert rejected.returncode == 2 and "phases.eval-S.prompt_sets" in rejected.stderr
        assert not any("quality.py" in c or " env " in c for c in trace.read_text().splitlines())
        frozen.write_text(json.dumps({"phases": {"eval-S": {"prompt_sets": [["p"]]}}}))
        original = frozen.read_bytes()
        trace.write_text("")
        completed = night()
        assert completed.returncode == 0, completed.stderr
        calls = trace.read_text().splitlines()
        assert calls[:2] == [f"-I -B {root}/run_matrix.py", f"-I -B {root}/analyze.py"]
        assert next(i for i, c in enumerate(calls) if " llama-ids " in c) < next(
            i for i, c in enumerate(calls) if " plan " in c)
        assert [c.rsplit(" ", 1)[-1] for c in calls if " plan " in c] == [
            "eval-S", "sampled-S", "eval-L-mlx"]
        assert not any(" freeze " in c or " tune" in c for c in calls)
        assert frozen.read_bytes() == original
        # Coupure à chaque contrôle : aucune sous-étape suivante ne démarre.
        for which in ("partial", "2"):
            trace.write_text("")
            counter.write_text("0")
            failure.write_text("999")
            completed = night(which)
            assert completed.returncode == 0, completed.stderr
            baseline = trace.read_text().splitlines()
            guards = [i for i, c in enumerate(baseline) if c.endswith("night_checks.py power")]
            assert len(guards) == 9, baseline
            # Avant serveur, ids, BF16, chaque Q4 et chaque bench, puis MLX.
            for i, call in enumerate(baseline):
                if " llama-ids " in call or " mlx-ppl " in call or call in ("llama-perplexity", "llama-bench"):
                    assert baseline[i - 1].endswith("night_checks.py power"), baseline
            for n, boundary in enumerate(guards, 1):
                trace.write_text("")
                counter.write_text("0")
                failure.write_text(str(n))
                stopped = night(which)
                assert stopped.returncode != 0 and "alimentation" in stopped.stderr, (which, n, stopped)
                def unstamped(calls):
                    return [c.rsplit(" ", 1)[0] if " env " in c else c for c in calls]
                assert unstamped(trace.read_text().splitlines()) == unstamped(baseline[:boundary + 1]), (which, n)
        failure.write_text("999")
        # Un auto-test en échec arrête la nuit avant environnement ou qualité.
        fake.write_text('#!/bin/bash\nexit 9\n')
        trace.write_text("")
        assert night().returncode == 9 and trace.read_text() == ""
        fake.write_text(runner)
        for override in (True, False):
            if not override:
                env.pop("BENCH_MLX_CKPT")
            trace.write_text("")
            for script in ("launch_saragossa.sh", "launch_mlx.sh"):
                result = subprocess.run(["bash", str(root / script), "ar", "18080", "8192"],
                                        env=env, capture_output=True, text=True, timeout=10)
                assert result.returncode == 0, result.stderr
            checkpoint = root / ("changed" if override else "default")
            calls = trace.read_text().splitlines()
            assert len(calls) == 2 and all(str(checkpoint) in c for c in calls)
            assert calls[1].startswith("-I -m mlx_lm.server ")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        print("night_checks: auto-test ok")
    elif sys.argv[1:] == ["power"]:
        check_power()
    elif len(sys.argv) == 3 and sys.argv[1] == "partial-freeze":
        check_partial_freeze(sys.argv[2])
    else:
        raise SystemExit("usage : night_checks.py [power|partial-freeze RUN_DIR]")
