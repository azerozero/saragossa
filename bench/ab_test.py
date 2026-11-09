"""Auto-tests de ab.py, exécutés par python3 -I -B bench/ab.py sans matériel."""

import contextlib
import copy
import io
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


real_wait_idle = ab.sensors.wait_idle
FAKE = r'''
import os, pathlib, sys
assert sys.flags.isolated and sys.dont_write_bytecode
assert 'SARAGOSSA_AMBIENT_POISON' not in os.environ
assert '--ignore-stop-tokens' in sys.argv and '--metrics' in sys.argv
assert '--model-dir' in sys.argv and '--max-tokens' in sys.argv
assert '--temperature' in sys.argv and '--seed' in sys.argv
counter = pathlib.Path(os.environ['AB_COUNTER'])
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
mode = os.environ.get('AB_MODE', '')
if mode == 'crash':
    sys.exit(7)
value = float(os.environ.get('AB_SPEED', '10')) + n * float(os.environ.get('AB_DRIFT', '0'))
prefill = os.environ.get('AB_PREFILL', '2000')
prefill_rate = os.environ.get('AB_PREFILL_RATE', str(8000 / float(prefill)))
print('été metrics decode_tok_s=999 — texte identique')
print('diagnostic metrics faux', file=sys.stderr)
if 'SARAGOSSA_RUST_ORACLE_DUMP_IDS' in os.environ and mode != 'noids':
    print('oracle_ids ' + ('1,2,4' if mode == 'ids' else '1,2,3'), file=sys.stderr)
if mode != 'missing':
    line = ('metrics prompt_tokens=' + ('9' if mode == 'prompt' else '8') +
            ' tokens=3 load_ms=2 warmup_ms=1 prefill_ms=' + prefill +
            ' prefill_tok_s=' + prefill_rate + ' decode_tokens=' + ('3' if mode == 'decode' else '2') +
            ' decode_ms=200 generate_ms=300 tok_s=' + str(value) +
            ' decode_tok_s=' + ('NaN' if mode == 'nan' else str(value)))
    print(line, file=sys.stderr, end='' if mode == 'partial' else '\n')
'''


class Sampler:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def snapshot(self, start, end):
        assert end > start
        return dict(n_samples=2, gpu_temp_c_start=40, gpu_temp_c_end=42, gpu_freq_mhz_median=900)


with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    binary = root / "faux moteur"
    binary.write_text(f"#!/usr/bin/env -S {sys.executable} -I -B\n" + FAKE)
    binary.chmod(0o700)
    counter = root / "counter"
    base = SimpleNamespace(a=str(binary), b=str(binary), env_a=[f"AB_COUNTER={counter}"],
                           env_b=[f"AB_COUNTER={counter}"], prompt_file=None, prompt_tokens=8,
                           model_dir=str(root / "aucun-modèle"), max_tokens=3, temperature=0,
                           seed=1, blocks=4, order="abba", random_seed=42, identity="ids",
                           primary="decode_tok_s", min_effect=0.01, allow_battery=False)
    ac = dict(on_ac=True, low_power=False)
    normal = dict(throttled=False, pressure=0)
    with (patch.object(ab.sensors, "Sampler", Sampler),
          patch.object(ab.sensors, "power_state", return_value=ac) as power,
          patch.object(ab.sensors, "thermal", return_value=normal) as thermal,
          patch.object(ab.sensors, "wait_idle", return_value=dict(machine_busy=False, idle_established=True)) as idle,
          patch.dict(os.environ, {"SARAGOSSA_AMBIENT_POISON": "interdit"})):
        serial = [0]

        def campaign(speed="10", mode="", **changes):
            args = copy.deepcopy(base)
            args.env_b += [f"AB_SPEED={speed}", f"AB_MODE={mode}"]
            for key, value in changes.items():
                setattr(args, key, value)
            config = ab.plan(args)
            path = root / str(serial[0])
            serial[0] += 1
            ab.run(path, config)
            return path, config, ab.report(path)

        def verdict(result, expected):
            assert result["metrics"]["decode_tok_s"]["verdict"] == expected, result

        def invalid(mode, reason):
            _, _, result = campaign(mode=mode)
            verdict(result, "invalide")
            assert any(reason in error for error in result["errors"]), result

        path, config, gain = campaign("12")
        verdict(gain, "gain")
        assert power.call_count == thermal.call_count == 32 and idle.call_count == 16
        assert all(call.kwargs == {"require_sensor": True} for call in idle.call_args_list)
        assert not gain["indicative"] and not gain["indicative_reasons"]
        assert gain["blocks"] == 4 and gain["identity"] == "ids" and not gain["excluded"]
        metric = gain["metrics"]["decode_tok_s"]
        assert abs(metric["ratio"] - 1.2) < 1e-12
        assert all(abs(x - 1.2) < 1e-12 for x in metric["interval"])
        rows, tail = ab.journal(path)
        assert not tail and len(rows) == 16 and rows[0]["env"]["SARAGOSSA_RUST_ORACLE_DUMP_IDS"] == "1"
        assert 'été metrics' in (path / rows[0]["stdout"]).read_text()
        assert rows[0]["sensors"]["gpu_freq_mhz_median"] == 900
        assert config["sides"]["a"]["sha256"] == ab.digest(binary)
        before = counter.read_text()
        ab.run(path, config)
        assert counter.read_text() == before  # Un bloc terminé ne se rejoue pas.
        bad_config = copy.deepcopy(config)
        bad_config["sides"]["b"]["env"]["AB_SPEED"] = "11"
        try:
            ab.run(path, bad_config)
            raise AssertionError("configuration modifiée acceptée")
        except ValueError as error:
            assert "reprise refusée" in str(error)
        loss_path, _, loss = campaign("8")
        verdict(loss, "perte")
        neutral_path, _, neutral = campaign()
        verdict(neutral, "indistinct")
        assert neutral["metrics"]["decode_tok_s"]["interval"] == [1.0, 1.0]
        _, _, reverse = campaign(env_a=base.env_a + ["AB_SPEED=12"])
        verdict(reverse, "perte")
        assert abs(reverse["metrics"]["decode_tok_s"]["ratio"] * metric["ratio"] - 1) < 1e-12
        secondary_path, _, secondary = campaign(primary="prefill_ms", env_b=base.env_b + ["AB_SPEED=12", "AB_PREFILL=4000"])
        verdict(secondary, "gain")
        assert secondary["metrics"]["prefill_ms"]["verdict"] == "perte"
        _, _, latency = campaign(env_b=base.env_b + ["AB_PREFILL=1000"])
        assert abs(latency["metrics"]["prefill_ms"]["ratio"] - 2) < 1e-12
        assert latency["metrics"]["prefill_ms"]["verdict"] == "gain"  # Exactement 1 s reste précis.
        for order in ("abba", "random"):
            _, _, drift = campaign(order=order, env_a=base.env_a + ["AB_DRIFT=3"],
                                    env_b=base.env_b + ["AB_DRIFT=3"])
            verdict(drift, "indistinct")
            assert drift["metrics"]["decode_tok_s"]["interval"] == [1, 1]
        for mode, reason in (("ids", "identité"), ("noids", "oracle_ids"), ("crash", "échec"),
                             ("prompt", "prompt_tokens"),
                             ("nan", "non finie"), ("missing", "metrics"), ("partial", "metrics")):
            invalid(mode, reason)
        failed_path, failed_config, _ = campaign(mode="crash")
        before = counter.read_text()
        ab.run(failed_path, failed_config)
        assert counter.read_text() == before
        verdict(ab.report(failed_path), "invalide")
        _, _, noids = campaign(mode="ids", identity="none")
        verdict(noids, "indistinct")
        for state in (dict(on_ac=False, low_power=False), dict(on_ac=None, low_power=False),
                      dict(on_ac=True, low_power=None), dict(on_ac=True, low_power=True)):
            power.return_value = state
            _, _, result = campaign()
            verdict(result, "invalide")
            assert any("alimentation" in e for e in result["errors"])
        _, _, allowed = campaign("12", allow_battery=True)
        verdict(allowed, "gain")
        assert allowed["indicative"]
        power.return_value = ac
        power.side_effect = [ac, dict(on_ac=False, low_power=False)] * 16
        _, _, unplugged = campaign()
        verdict(unplugged, "invalide")  # Relecture après, pas seulement avant.
        power.side_effect = None
        for state in (dict(throttled=True, pressure=0), dict(throttled=False, pressure=2)):
            thermal.return_value = state
            _, _, result = campaign()
            verdict(result, "invalide")
            assert any("thermique" in e for e in result["errors"])
        thermal.return_value = normal
        idle.return_value = dict(machine_busy=True, idle_established=False, reason="repos non confirmé au plafond")
        busy_path, _, busy = campaign()
        verdict(busy, "invalide")
        assert not busy["indicative"]
        assert any("occupée" in e for e in busy["errors"])
        idle.return_value = dict(machine_busy=False, idle_established=True)
        _, _, few = campaign("12", blocks=3)
        verdict(few, "indistinct")
        assert few["metrics"]["decode_tok_s"]["interval"] is None
        assert "insuffisant" in few["metrics"]["decode_tok_s"]["note"]
        args = copy.deepcopy(base)
        args.order = "random"
        args.blocks = 20
        one = ab.plan(args)
        assert one == ab.plan(args) and set(one["orders"]) == {"abba", "baab"}
        args.random_seed += 1
        assert one["orders"] != ab.plan(args)["orders"]
        prompts = [root / "été.txt", root / "code.txt"]
        for p in prompts:
            p.write_text("Texte français éè")
        _, multiple, result = campaign("12", prompt_file=list(map(str, prompts)))
        verdict(result, "gain")
        assert len(multiple["prompts"]) == 2 and len(result["sensors"]) == 32
        assert multiple["prompt_sha256"] == [ab.digest(p) for p in prompts]
        resume = root / "resume"
        original = ab.passage
        calls = [0]

        def interrupted(*args):
            calls[0] += 1
            if calls[0] == 6:
                raise KeyboardInterrupt
            return original(*args)

        with patch.object(ab, "passage", side_effect=interrupted):
            try:
                ab.run(resume, config)
                raise AssertionError("interruption perdue")
            except KeyboardInterrupt:
                pass
        partial = ab.report(resume)
        verdict(partial, "invalide")
        assert partial["blocks"] == 1 and len(partial["excluded"]) == 1
        with (resume / "results.jsonl").open("ab") as stream:
            stream.write(b'{"block":1')
        verdict(ab.report(resume), "invalide")
        ab.run(resume, config)
        resumed = ab.report(resume)
        verdict(resumed, "gain")
        assert resumed["blocks"] == 4 and len(resumed["excluded"]) == 1
        assert len(ab.journal(resume)[0]) == 17 and list((resume / "raw").glob("journal-interrompu-*"))
        assert len(list((resume / "raw").glob("*.stdout"))) == 17
        for saved, expected in ((path, 0), (resume, 0), (loss_path, 1), (neutral_path, 2),
                                (secondary_path, 1), (busy_path, 3)):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                assert ab.main(["report", str(saved), "--json"]) == expected
            parsed = json.loads(output.getvalue())
            assert ab.CODES[parsed["metrics"][parsed["primary"]]["verdict"]] == expected
        with contextlib.redirect_stdout(io.StringIO()) as output:
            assert ab.main(["run", str(root / "cli"), "--a", str(binary), "--b", str(binary),
                            "--env-a", f"AB_COUNTER={counter}", "--env-b", f"AB_COUNTER={counter}",
                            "--model-dir", str(root), "--prompt-file", *map(str, prompts),
                            "--blocks", "1", "--max-tokens", "3"]) == 2
        assert "insuffisant" in output.getvalue() and "40 / 42" in output.getvalue()
        # report est autonome et ne sollicite aucun capteur.
        cli = subprocess.run([sys.executable, "-I", "-B", ab.__file__, "report", str(path), "--json"],
                             capture_output=True, text=True)
        assert cli.returncode == 0 and json.loads(cli.stdout)["blocks"] == 4, cli.stderr

        def replay(change, **config_changes):
            """Rejoue seulement le rapport d'un journal factice, sans processus."""
            saved = root / str(serial[0])
            serial[0] += 1
            saved.mkdir()
            altered = copy.deepcopy(rows)
            for row in altered:
                change(row)
            (saved / "plan.json").write_text(json.dumps(dict(config, **config_changes)))
            (saved / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in altered))
            return saved, ab.report(saved)

        def report_code(saved, expected):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                assert ab.main(["report", str(saved)]) == expected, output.getvalue()
            return output.getvalue()

        # Une absence de preuve reste visible, même si les mesures favorisent B.
        unknown_cases = [
            (lambda r: r.pop("sensors"), "capteur absent"),
            (lambda r: r.update(sensors=None), "capteur absent"),
            (lambda r: r["sensors"].update(n_samples=0), "aucun échantillon"),
            (lambda r: r["sensors"].update(gpu_freq_mhz_median=None), "capteur GPU"),
            (lambda r: r["thermal_before"].update(pressure=None), "pression thermique inconnue"),
            (lambda r: r["thermal_after"].pop("pressure"), "pression thermique inconnue"),
            (lambda r: r.update(idle={"machine_busy": False}), "preuve de repos absente"),
            (lambda r: r.update(idle={"machine_busy": False, "idle_established": False,
                                      "reason": "capteur indisponible"}), "capteur indisponible"),
        ]
        for change, reason in unknown_cases:
            saved, result = replay(change)
            assert result["indicative"] and not result["errors"], result
            assert any(reason in item for item in result["indicative_reasons"]), result
            text = report_code(saved, 2)
            assert "INDICATIF" in text and reason in text
        for speed in (8, 10, 12):
            saved, result = replay(lambda r: r["metrics"].update(
                decode_tok_s=speed if r["side"] == "b" else 10), allow_battery=True)
            assert result["indicative"]
            report_code(saved, 2)
        saved, result = replay(lambda r: r["errors"].append("échec factice"), allow_battery=True)
        verdict(result, "invalide")
        report_code(saved, 3)  # Une mesure invalide prime sur le caractère indicatif.
        for ratios in ((0.1, 0.1), (0.1, 0.3)):
            with (patch.object(ab.sensors.time, "monotonic", side_effect=[0, 0, 1, 1]),
                  patch.object(ab.sensors.time, "sleep")):
                sensor = SimpleNamespace(latest=lambda: (next(samples), {"gpu_active_ratio": next(activity)}))
                samples, activity = iter((1, 2)), iter(ratios)
                state = real_wait_idle({"busy_wait_max_s": 1}, sensor, require_sensor=True)
            saved, result = replay(lambda r: r.update(idle=state))
            assert result["indicative"] == (ratios[-1] < 0.2)
            report_code(saved, 2 if ratios[-1] < 0.2 else 3)

        # Les ms entiers trompent : seul prompt_tokens / prefill_tok_s fait foi.
        saved, result = replay(lambda r: r["metrics"].update(
            prefill_ms=4000 if r["side"] == "b" else 1000,
            prefill_tok_s=8 if r["side"] == "b" else 4), primary="prefill_ms")
        assert result["metrics"]["prefill_ms"]["ratio"] == 2
        report_code(saved, 0)
        for fast_side in "ab":
            saved, result = replay(lambda r: r["metrics"].update(
                prefill_ms=0, prefill_tok_s=16000 if r["side"] == fast_side else 8000),
                primary="prefill_ms")
            prefill = result["metrics"]["prefill_ms"]
            assert math.isclose(prefill["ratio"], 2 if fast_side == "b" else 0.5, rel_tol=1e-12)
            assert prefill["verdict"] == "indistinct" and "imprécise" in prefill["note"]
            report_code(saved, 2)

        for key, value, reason in (("decode_tokens", 4, "decode_tokens doit valoir"),
                                   ("decode_tokens", 1, "decode_tokens doit valoir"),
                                   ("tokens", 2, "--max-tokens"),
                                   ("tokens", 4, "--max-tokens"),
                                   ("prefill_tok_s", 0, "prefill_tok_s"),
                                   ("prefill_tok_s", -1, "prefill_tok_s")):
            saved, result = replay(lambda r: r["metrics"].update(
                {key: value, **({"decode_tokens": value - 1} if key == "tokens" else {})}))
            verdict(result, "invalide")
            assert any(reason in e for e in result["errors"]), result
            report_code(saved, 3)
        saved, result = replay(lambda r: r["metrics"].update(tokens=4, decode_tokens=3),
                               args=[a for a in config["args"] if a != "--ignore-stop-tokens"])
        verdict(result, "gain")  # Le nombre demandé n'est imposé qu'avec ignore-stop.

        def single_token(row):
            row["metrics"].update(tokens=1, decode_tokens=0, decode_tok_s=0)
            row["ids"] = [1]

        one_args = ["1" if a == "3" else a for a in config["args"]]
        saved, result = replay(single_token, args=one_args, primary="tok_s")
        assert not result["errors"] and result["metrics"]["tok_s"]["verdict"] == "gain"
        decode = result["metrics"]["decode_tok_s"]
        assert decode["ratio"] is decode["interval"] is None and not decode["comparable"]
        assert "indisponible" in decode["note"]
        report_code(saved, 0)
        saved, result = replay(single_token, args=one_args)
        assert "decode_tok_s indisponible" in report_code(saved, 3)

        def mtp(row):
            if row["side"] == "b":
                row["metrics"]["decode_tokens"] = 3

        saved, result = replay(mtp, primary="tok_s")
        assert not result["errors"] and result["identity_status"] == "vérifiée"
        assert result["metrics"]["tok_s"]["verdict"] == "gain"
        for metric_name in ("decode_tok_s", "prefill_ms"):
            value = result["metrics"][metric_name]
            assert value["verdict"] == "indistinct" and not value["comparable"]
            assert value["ratio"] is value["interval"] is None
            assert "non comparables" in value["note"]
            bad, invalid_result = replay(mtp, primary=metric_name, allow_battery=True)
            assert invalid_result["metrics"][metric_name]["verdict"] == "invalide"
            assert "--primary tok_s" in report_code(bad, 3)
        report_code(saved, 0)

        def divergent_mtp(row):
            mtp(row)
            if row["side"] == "b":
                row["ids"] = [1, 2, 4]

        saved, result = replay(divergent_mtp, primary="tok_s")
        assert any("identité" in e for e in result["errors"])
        report_code(saved, 3)
        saved, result = replay(lambda r: r["metrics"].update(
            decode_tokens=3 if r["slot"] == 1 else 2), primary="tok_s")
        assert any("instables" in e for e in result["errors"])
        report_code(saved, 3)

        for argv in ([], ["inconnu"], ["report"], ["report", str(path), "--inconnu"],
                     ["run", str(root), "--blocks", "texte"], ["report", str(root / "absent")]):
            with contextlib.redirect_stderr(io.StringIO()):
                assert ab.main(argv) == 3, argv
            if argv:
                process = subprocess.run([sys.executable, "-I", "-B", ab.__file__, *argv],
                                         capture_output=True, text=True)
                assert process.returncode == 3, (argv, process.stderr)
        for error in (KeyError("clé"), TypeError("type"), OverflowError("exp"), RuntimeError("erreur"),
                      KeyboardInterrupt(), SystemExit(0), SystemExit(1), SystemExit(2)):
            with (patch.object(ab, "report", side_effect=error), contextlib.redirect_stderr(io.StringIO())):
                assert ab.main(["report", str(path)]) == 3, error
        with contextlib.redirect_stdout(io.StringIO()):
            assert ab.main(["--help"]) == 0
        broken_dir = root / "plan-invalide"
        broken_dir.mkdir()
        (broken_dir / "plan.json").write_text("{}")
        process = subprocess.run([sys.executable, "-I", "-B", ab.__file__, "report", str(broken_dir)],
                                 capture_output=True, text=True)
        assert process.returncode == 3 and "invalide" in process.stderr
        # La frontière sans argument protège aussi les échecs des auto-tests.
        entry = root / "frontière-cli"
        entry.mkdir()
        (entry / "ab.py").write_bytes(pathlib.Path(ab.__file__).read_bytes())
        (entry / "competitors").symlink_to(pathlib.Path(ab.sensors.__file__).resolve().parent,
                                           target_is_directory=True)
        for source in ("raise AssertionError('test')", "raise SystemExit(2)", "raise KeyboardInterrupt()", ""):
            (entry / "ab_test.py").write_text(source)
            process = subprocess.run([sys.executable, "-I", "-B", str(entry / "ab.py")],
                                     capture_output=True, text=True)
            assert process.returncode == 3 and "invalide" in process.stderr

    raw = (path / rows[0]["stderr"]).read_bytes()
    assert ab.parse(raw, "ids")[1] == [1, 2, 3]
    single = raw.replace(b"tokens=3", b"tokens=1").replace(b"decode_tokens=2", b"decode_tokens=0")
    single = single.replace(b"decode_tok_s=10.0", b"decode_tok_s=0").replace(b"1,2,3", b"1")
    assert ab.parse(single, "ids")[0]["decode_tok_s"] == 0
    for broken in (raw.rstrip(b"\n"), raw.replace(b"decode_tok_s=10.0", b"decode_tok_s=NaN"),
                   raw + b"metrics x=1\n", b"texte metrics x=1\n", raw.replace(b"1,2,3", b"1,2"),
                   raw.replace(b"prompt_tokens=8", b"prompt_tokens=8.5"),
                   raw.replace(b"decode_tokens=2", b"decode_tokens=4"),
                   raw.replace(b"decode_tokens=2", b"decode_tokens=1"),
                   raw.replace(b"prefill_tok_s=4.0", b"prefill_tok_s=0"),
                   raw.replace(b"prefill_tok_s=4.0", b"prefill_tok_s=-1"),
                   raw.replace(b"prefill_tok_s=4.0", b"prefill_tok_s=NaN"),
                   raw.replace(b"prefill_tok_s=4.0", b"prefill_tok_s=inf"),
                   raw.replace(b" prefill_tok_s=4.0", b""),
                   raw.replace(b"prefill_tok_s=4.0", b"prefill_tok_s=1e-320")):
        try:
            ab.parse(broken, "ids")
            raise AssertionError("parseur permissif")
        except ValueError:
            pass
    strata = [[0.0, 0.1, 0.2, 0.3], [-0.3, -0.2, -0.1, 0.0]]
    result = ab.estimate(strata, 0.01)
    assert result == ab.estimate(strata, 0.01)
    assert result["ratio"] == 1 and result["interval"][0] < 1 < result["interval"][1]
    # Calcul indépendant : quatre blocs, pas huit observations indépendantes.
    block_means = [-0.15, -0.05, 0.05, 0.15]
    margin = 3.1824463053 * statistics.stdev(block_means) / 2
    expected = [math.exp(-margin), math.exp(margin)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(result["interval"], expected))
    counterexample = ab.estimate([[-0.2, 0.1, 0.1, 0.2]] * 16, 0)
    assert counterexample["verdict"] == "indistinct" and counterexample["interval"][0] < 1
    for blocks in (0, 1, 2, 3):
        insufficient = ab.estimate([[0.2] * blocks] * 3, 0)
        assert insufficient["interval"] is None and insufficient["verdict"] == "indistinct"
        assert "insuffisant" in insufficient["note"]
    for df, expected in ((3, 3.1824463053), (5, 2.5705818356), (30, 2.0422724563),
                         (31, 2.0395134464), (32, 2.0369333435), (60, 2.0002978220),
                         (120, 1.9799304051)):
        assert abs(ab.student_critical(df) - expected) < 1e-9, df
    # A=A : effet commun à tous les prompts du bloc + bruit propre à chacun.
    # Un rééchantillonnage indépendant par prompt sous-couvre fortement ce cas.
    for blocks in (4, 6):
        rng = random.Random(20261109 + blocks)
        rejected, trials = 0, 4000
        for _ in range(trials):
            common = [rng.gauss(0, 0.2) for _ in range(blocks)]
            logs = [[effect + rng.gauss(0, 0.03) for effect in common] for _ in range(3)]
            rejected += ab.estimate(logs, 0)["verdict"] != "indistinct"
        assert rejected / trials <= 0.08, (blocks, rejected, trials)
        print(f"A=A : {blocks} blocs, effet commun, faux verdicts {rejected}/{trials} ({rejected / trials:.2%})")
    for log in (-0.005, 0.005):
        assert ab.estimate([[log] * 4], 0.01)["verdict"] == "indistinct"
    assert ab.CODES == {"gain": 0, "perte": 1, "indistinct": 2, "invalide": 3}

print("ab: auto-tests ok (Student, prefill, AR/MTP, comptes, capteurs, codes CLI, ABBA, identité, reprise)")
TESTS_COMPLETED = True
