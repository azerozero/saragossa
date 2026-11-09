"""Auto-tests de gpulock : processus réels, alimentation factice, aucun GPU."""

import fcntl
import importlib.util
import json
import os
import pathlib
import pty
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = pathlib.Path(__file__).with_name("gpulock.py")
PYTHON = [sys.executable, "-I", "-B"]


def check_local(root):
    spec = importlib.util.spec_from_file_location("gpulock", SCRIPT)
    gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu)
    with patch.object(gpu.os, "open", side_effect=PermissionError):
        assert gpu.held(root) is None
    lock = root / "partial.lock"
    args = SimpleNamespace(owner="test", prio=50, timeout=.3,
                           allow_battery=False, cmd=["/usr/bin/true"])
    power = dict(on_ac=True, low_power=False, errors=[])
    with patch.object(gpu.subprocess, "run", side_effect=[OSError(), SimpleNamespace(stdout="boot\n")]):
        assert gpu.boot_id() == "kern.boottime:boot"
    for error in (OSError(), subprocess.TimeoutExpired("sysctl", 3)):
        with patch.object(gpu.subprocess, "run", side_effect=error):
            assert gpu.boot_id() == ""
    write = pathlib.Path.write_text
    with patch.object(gpu.sensors, "power_state", return_value=power), \
            patch.object(gpu, "boot_id", return_value="kern.bootsessionuuid:test-boot"):
        assert gpu.occupied(lock, dict(pid=str(os.getpid()), child_pid="", pgid="",
                                       boot="kern.boottime:ancien"))
        for replace in (False, True):
            def fail(path, *a, **kw):
                if path == lock / "owner":
                    if replace:
                        lock.rename(root / "abandoned")
                        lock.mkdir()
                        write(lock / "owner", "autrui")
                        write(lock / "pid", str(os.getpid()))
                    raise OSError("écriture injectée")
                return write(path, *a, **kw)
            with patch.object(pathlib.Path, "write_text", fail):
                try:
                    gpu.run(lock, args)
                except OSError as error:
                    assert str(error) == "écriture injectée"
                else:
                    raise AssertionError("injection non déclenchée")
            assert lock.exists() == replace
            if replace:
                assert (lock / "owner").read_text() == "autrui"
                (lock / "owner").unlink()
                (lock / "pid").unlink()
                lock.rmdir()
            assert gpu.run(lock, args) == 0 and not lock.exists()
        # Le secteur peut disparaître entre le pré-contrôle et l'acquisition.
        with patch.object(gpu.sensors, "power_state", side_effect=[power, dict(power, on_ac=False)]):
            assert gpu.run(lock, args) == 3 and not lock.exists()
        # Disparition entre exists() et stat() dans le finally : garder le code enfant.
        original_stat = pathlib.Path.stat
        def vanished(path, *a, **kw):
            if path == lock and (lock / "pgid").exists():
                shutil.rmtree(lock)
            return original_stat(path, *a, **kw)
        def exists(path):
            try:
                original_stat(path)
                return True
            except FileNotFoundError:
                return False
        args.cmd = [*PYTHON, "-c", "raise SystemExit(37)"]
        with patch.object(pathlib.Path, "stat", vanished), patch.object(pathlib.Path, "exists", exists):
            assert gpu.run(lock, args) == 37
        assert not lock.exists()
    assert (gpu.SURVIVOR_GRACE, gpu.SURVIVOR_TERM_GRACE) == (30, 10)
    assert "--force est réservé à une intervention manuelle" in gpu.__doc__
    assert "ne retire pas le ticket du tenant toujours vivant" in gpu.__doc__
    assert "priorité stricte sans vieillissement (famine possible)" in gpu.__doc__
    assert "night.sh sonde à 60 s et ne participe pas à l'ordre des tickets" in gpu.__doc__
    print("gpulock: démarrage indisponible, écritures, inode étranger/disparu, secteur et docstring ok",
          flush=True)


def check_night(root):
    # Exécuter seulement le bloc de verrouillage, sans campagne ni moteur.
    script = SCRIPT.parents[1] / "bench/competitors/night.sh"
    block = script.read_text().split("\nfi\n", 1)[1].split("\nphase_run()", 1)[0]
    assert "sleep 60" in block and ".queue" not in block
    for value in (None, "", str(root / "custom/lock")):
        env = dict(os.environ, HOME=str(root / "home"), GPU_LOCK_HELD="autre")
        env.pop("GPU_LOCK", None)
        if value is not None:
            env["GPU_LOCK"] = value
        expected = value or str(root / "home/.cache/saragossa/gpu.lock")
        result = subprocess.run(["bash", "-eu", "-c", block + '\ntest -f "$1/owner"',
                                 "test", expected], env=env, capture_output=True, timeout=3)
        assert result.returncode == 0, result.stderr
        assert not pathlib.Path(expected).exists()
    lock = root / "custom/inherited.lock"
    lock.mkdir()
    (lock / "owner").write_text("tenant")
    alias = root / "alias"
    alias.symlink_to(lock, target_is_directory=True)
    for value in (str(lock), "./alias/../inherited.lock"):
        result = subprocess.run(["bash", "-eu", "-c", block + '\ntest -z "$(trap -p EXIT)"'],
                                cwd=root, env=dict(os.environ, GPU_LOCK=value,
                                                   GPU_LOCK_HELD=str(lock.resolve())),
                                capture_output=True, timeout=3)
        assert result.returncode == 0, result.stderr
        assert (lock / "owner").read_text() == "tenant"
    print("gpulock: night.sh défaut, vide, distinct ou hérité/résolu, trap et propriétaire ok", flush=True)


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"condition non satisfaite après {timeout}s")


def dead(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def check():
    with tempfile.TemporaryDirectory(prefix="gpulock-") as tmp:
        root = pathlib.Path(tmp)
        check_local(root)
        check_night(root)
        lock = root / "gpu.lock"
        queue = pathlib.Path(str(lock) + ".queue")
        journal = pathlib.Path(str(lock) + ".log.jsonl")
        fake = root / "pmset"
        fake.write_text("#!/bin/sh\ncase \"$POWER_TEST\" in\n"
                        " unknown) exit 1;;\n battery) echo 'Battery Power';;\n"
                        " *) echo 'AC Power';;\nesac\n"
                        "echo \"lowpowermode ${LOW_TEST:-0}\"\n")
        fake.chmod(0o700)
        sysctl = root / "sysctl"
        sysctl.write_text('#!/bin/sh\necho "${BOOT_TEST:-test-boot}"\n')
        sysctl.chmod(0o700)
        env = dict(os.environ, GPU_LOCK=str(lock), PATH=f"{root}:/usr/bin:/bin",
                   POWER_TEST="ac", LOW_TEST="0")
        processes, groups = [], set()

        def spawn(*args, extra=None, inject=None, **kwargs):
            command = (["-c", "import runpy,sys; from unittest.mock import patch; "
                        "g=runpy.run_path(sys.argv.pop(1));\n" + inject] if inject else [])
            process = subprocess.Popen([*PYTHON, *command, str(SCRIPT), *args],
                                       env=dict(env, **(extra or {})),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
            processes.append(process)
            return process

        def finish(process, expected=0):
            out, err = process.communicate(timeout=12)
            assert process.returncode == expected, (process.args, process.returncode, out, err)
            return out.decode(), err.decode()

        def rows(event=None):
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            required = {"ts", "event", "owner", "pid", "prio", "cmd", "wait_s", "duration_s", "exit_code"}
            assert all(required <= row.keys() for row in records)
            return [r for r in records if event is None or r["event"] == event]

        def waiting(process):
            eventually(lambda: queue.exists() and any(queue.glob(f"*-{process.pid}")))

        def suspend(process):
            with open(str(lock) + ".guard", "a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                os.kill(process.pid, signal.SIGSTOP)

        def holder():
            ready = root / "ready"
            ready.unlink(missing_ok=True)
            process = spawn("run", "--owner", "tenant", "--", *PYTHON, "-c",
                            "import pathlib,time,os; pathlib.Path('ready').touch(); time.sleep(60)",
                            cwd=root)
            eventually(lambda: ready.exists())
            assert (lock / "owner").read_text() == "tenant\n"
            assert int((lock / "pid").read_text()) == process.pid
            assert (lock / "holder").exists()
            assert (lock / "boot").read_text().strip() == "kern.bootsessionuuid:test-boot"
            pid = int((lock / "child_pid").read_text())
            assert int((lock / "pgid").read_text()) == pid
            groups.add(pid)
            return process, pid

        def attendant(name, priority):
            process = spawn("run", "--owner", name, "--prio", str(priority), "--timeout", "10",
                            "--", *PYTHON, "-c",
                            "import os,time,pathlib; "
                            "f=os.open('exclusive',os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(f); "
                            f"open('order','a').write('{name}\\n'); "
                            "time.sleep(.15); pathlib.Path('exclusive').unlink()", cwd=root)
            waiting(process)
            return process

        def orphan():
            # Un vrai lanceur, tué brutalement ; son enfant est encore protégé.
            process, pid = holder()
            process.kill()
            # L'enfant vivant conserve stdout/stderr : communicate attendrait son EOF.
            assert process.wait(timeout=3) == -signal.SIGKILL
            return pid

        try:
            for value, expected in (("", root / "home/.cache/saragossa/gpu.lock"), ("./gpu.lock", lock)):
                finish(spawn("run", "--", *PYTHON, "-c", "import os,pathlib; "
                             f"p=pathlib.Path({str(expected)!r}); "
                             "assert p.is_dir() and os.environ['GPU_LOCK_HELD']==str(p.resolve())",
                             extra=dict(GPU_LOCK=value, GPU_LOCK_HELD="incorrect", HOME=str(root / "home")),
                             cwd=root))
                assert not expected.exists()
            print("gpulock: GPU_LOCK vide et GPU_LOCK_HELD absolu exporté ok", flush=True)
            for priorities, expected in (((50, 50), ["premier", "second"]),
                                         ((80, 10), ["second", "premier"])):
                (root / "order").unlink(missing_ok=True)
                process, pid = holder()
                first = attendant("premier", priorities[0])
                second = attendant("second", priorities[1])
                head = first if priorities[0] == priorities[1] else second
                suspend(head)
                assert not (root / "order").exists()
                out, _ = finish(spawn("status"))
                assert json.loads(out)["state"] == "occupé"
                finish(spawn("release"), 1)
                assert lock.exists() and not dead(pid)
                process.terminate()
                finish(process, 143)
                eventually(lambda: dead(pid))
                time.sleep(2.2)
                assert not (root / "order").exists(), "la tête vivante suspendue doit garder son rang"
                os.kill(head.pid, signal.SIGCONT)
                finish(first)
                finish(second)
                assert (root / "order").read_text().splitlines() == expected
                assert not lock.exists() and not list(queue.iterdir())
            print("gpulock: FIFO, priorité, exclusion et SIGTERM ok", flush=True)

            lock.mkdir()
            (lock / "owner").write_text("bench-competitors\n")
            directory, fifo, link = [queue / f"00-0-{n}" for n in range(3)]
            directory.mkdir()
            os.mkfifo(fifo)
            link.symlink_to(lock / "owner")
            before = len(rows("reclaim"))
            out, _ = finish(spawn("status"))
            assert json.loads(out)["state"] == "propriétaire inconnu"
            started = time.monotonic()
            finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"), 124)
            assert time.monotonic() - started < 1.8
            assert lock.exists() and not (lock / "pid").exists()
            assert len(rows("reclaim")) == before and rows("timeout")[-1]["exit_code"] == 124
            finish(spawn("release"), 1)
            (lock / "owner").unlink()
            lock.rmdir()
            finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"))
            assert directory.is_dir() and fifo.exists() and link.is_symlink()
            directory.rmdir()
            fifo.unlink()
            link.unlink()
            print("gpulock: tickets répertoire/FIFO/lien ignorés, verrou inconnu et timeout ok", flush=True)

            with open(str(lock) + ".guard", "a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                started = time.monotonic()
                finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"), 124)
                assert time.monotonic() - started < 1.8 and not lock.exists()

            pid = orphan()
            before = len(rows("reclaim"))
            launcher = int((lock / "pid").read_text())
            # Même un démarrage différent et des PID morts ne battent pas holder.
            original = {key: (lock / key).read_text() for key in ("boot", "child_pid", "pgid")}
            for key in ("child_pid", "pgid"):
                (lock / key).write_text(str(launcher))
            (lock / "boot").write_text("kern.bootsessionuuid:ancien")
            finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"), 124)
            finish(spawn("release"), 1)
            finish(spawn("release", inject=f"with patch('os.getpid', return_value={launcher}):\n"
                         "    raise SystemExit(g['main']())"), 1)
            assert len(rows("reclaim")) == before and not dead(pid)
            for key, value in original.items():
                (lock / key).write_text(value)
            os.killpg(pid, signal.SIGTERM)
            eventually(lambda: dead(pid))
            # L'enfant est mort mais un autre groupe vivant interdit encore la reprise.
            survivor = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
            processes.append(survivor)
            groups.add(survivor.pid)
            (lock / "pgid").write_text(str(survivor.pid))
            finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"), 124)
            survivor.terminate()
            survivor.wait(timeout=3)
            finish(spawn("run", "--", "/usr/bin/true"))
            assert len(rows("reclaim")) == before + 1
            assert list(root.glob("gpu.lock.stale-*")) and not lock.exists()
            print("gpulock: holder hérité, release avec PID réutilisé, groupe survivant et reprise ok", flush=True)

            survivor = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
            processes.append(survivor)
            groups.add(survivor.pid)
            lock.mkdir()
            for key in ("pid", "child_pid", "pgid"):
                (lock / key).write_text(str(survivor.pid))
            (lock / "boot").write_text("kern.bootsessionuuid:ancien")
            (lock / "holder").touch()
            finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"))
            assert not lock.exists()
            survivor.terminate()
            survivor.wait(timeout=3)
            print("gpulock: redémarrage simulé et PID réutilisés, reprise ok", flush=True)

            pid = orphan()
            # Deux lanceurs prêts en même temps devant le même verrou orphelin.
            (root / "order").unlink(missing_ok=True)
            before = len(rows("reclaim"))
            first = attendant("premier", 50)
            second = attendant("second", 50)
            os.killpg(pid, signal.SIGTERM)
            eventually(lambda: dead(pid))
            finish(first)
            finish(second)
            assert len(rows("reclaim")) == before + 1
            assert sorted((root / "order").read_text().splitlines()) == ["premier", "second"]
            assert not (root / "exclusive").exists()
            print("gpulock: deux réclamants, une seule reprise ok", flush=True)

            # Un processus meurt en laissant un ticket dont le PID désigne un vivant.
            stale_ticket = queue / f"00-0-{os.getpid()}"
            maker = subprocess.Popen([*PYTHON, "-c", "import fcntl,pathlib,sys; "
                                      "f=pathlib.Path(sys.argv[1]).open('x'); "
                                      "fcntl.flock(f,fcntl.LOCK_EX)", str(stale_ticket)])
            assert maker.wait(timeout=3) == 0
            foreign = queue / ".DS_Store"
            foreign.touch()
            finish(spawn("run", "--timeout", "0.3", "--", "/usr/bin/true"))
            assert not stale_ticket.exists() and foreign.exists()
            foreign.unlink()

            # Même groupe pour le petit-enfant ; son père attend son arrêt propre.
            ready, ended = root / "group-ready", root / "group-ended"
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                ready.unlink(missing_ok=True)
                ended.unlink(missing_ok=True)
                grandchild = ("import signal,time,pathlib,sys; "
                              f"done=lambda *a: (pathlib.Path({str(ended)!r}).touch(),sys.exit(0)); "
                              "signal.signal(signal.SIGTERM,done); signal.signal(signal.SIGINT,done); "
                              "signal.signal(signal.SIGHUP,done); "
                              f"pathlib.Path({str(ready)!r}).touch(); time.sleep(60)")
                parent = ("import signal,subprocess,sys,time; "
                          f"p=subprocess.Popen({[*PYTHON, '-c', grandchild]!r}); "
                          "done=lambda *a: (p.wait(),sys.exit(0)); "
                          "signal.signal(signal.SIGTERM,done); signal.signal(signal.SIGINT,done); "
                          "signal.signal(signal.SIGHUP,done); time.sleep(60)")
                process = spawn("run", "--", *PYTHON, "-c", parent)
                eventually(ready.exists)
                groups.add(int((lock / "pgid").read_text()))
                process.send_signal(signum)
                finish(process)
                assert ended.exists() and not lock.exists()
            print("gpulock: ticket mort avec PID réutilisé, .DS_Store, SIGTERM/SIGINT/SIGHUP au groupe ok", flush=True)

            ready.unlink()
            ended.unlink()
            process = spawn("run", "--", *PYTHON, "-c",
                            "import signal,time,pathlib; "
                            f"signal.signal(signal.SIGTERM,lambda *a: pathlib.Path({str(ended)!r}).touch()); "
                            f"pathlib.Path({str(ready)!r}).touch(); time.sleep(60)")
            eventually(ready.exists)
            pid = int((lock / "pgid").read_text())
            groups.add(pid)
            process.terminate()
            eventually(ended.exists)
            assert process.poll() is None and lock.exists() and not dead(pid)
            process.terminate()
            finish(process, 137)
            assert dead(pid) and not lock.exists()
            print("gpulock: enfant résistant, deuxième signal converti en SIGKILL ok", flush=True)

            # Le père sort normalement ; seul son descendant reste dans le groupe.
            for mode in ("naturel", "term", "kill"):
                ready.unlink(missing_ok=True)
                ended.unlink(missing_ok=True)
                before = len(rows("killed_survivors"))
                parent = ("import os,signal,time,pathlib,sys; r,w=os.pipe(); pid=os.fork();\n"
                          "if pid:\n os.close(w); os.read(r,1); sys.exit(37)\n"
                          "os.close(r)\n"
                          f"def stop(*a):\n pathlib.Path({str(ended)!r}).touch()\n"
                          + (" sys.exit(0)\n" if mode == "term" else "") +
                          "signal.signal(signal.SIGTERM,stop)\n"
                          f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
                          "os.write(w,b'1'); os.close(w)\n"
                          f"time.sleep({0.2 if mode == 'naturel' else 60})\n")
                process = spawn("run", "--", *PYTHON, "-c", parent,
                                extra={"_GPU_LOCK_SURVIVOR_GRACE": "0.5",
                                       "_GPU_LOCK_SURVIVOR_TERM_GRACE": "0.5"})
                eventually(ready.exists)
                pid = int((lock / "pgid").read_text())
                groups.add(pid)
                eventually(lambda: dead(pid))
                assert lock.exists() and process.poll() is None
                started = time.monotonic()
                finish(process, 37)
                elapsed = time.monotonic() - started
                events = rows("killed_survivors")[before:]
                expected = [] if mode == "naturel" else [signal.SIGTERM]
                if mode == "kill":
                    expected.append(signal.SIGKILL)
                assert [r["signal"] for r in events] == expected
                assert all(r["exit_code"] == 37 for r in events)
                assert elapsed < 3 and (mode == "naturel" or elapsed >= .4 * len(expected))
                assert ended.exists() == (mode != "naturel")
                assert dead(int(ready.read_text())) and dead(-pid) and not lock.exists()
            print("gpulock: descendants après sortie 37, grâce, SIGTERM/SIGKILL et journal ok", flush=True)

            # La libération explicite d'un mort, et la confirmation sur un vrai PTY.
            pid = orphan()
            os.killpg(pid, signal.SIGTERM)
            eventually(lambda: dead(pid))
            finish(spawn("release"))
            assert not lock.exists() and rows("release")[-1]["force"] is False
            lock.mkdir()
            (lock / "owner").write_text("ancien\n")
            finish(spawn("release", "--force", stdin=subprocess.DEVNULL), 2)
            assert lock.exists()
            for answer, expected in ((b"NON\n", 1), (b"OUI\n", 0)):
                master, slave = pty.openpty()
                try:
                    process = spawn("release", "--force", stdin=slave)
                    os.write(master, answer)
                    finish(process, expected)
                    assert lock.exists() == (expected == 1)
                finally:
                    os.close(master)
                    os.close(slave)
            assert rows("release")[-1]["force"] is True
            print("gpulock: release prudent et confirmation terminal journalisée ok", flush=True)

            marker = root / "executed"
            command = [*PYTHON, "-c", f"import pathlib; pathlib.Path({str(marker)!r}).touch()"]
            for extra in ({"POWER_TEST": "battery"}, {"LOW_TEST": "1"}):
                # Une file bloquée ne doit pas retarder le refus d'alimentation.
                lock.mkdir()
                with open(str(lock) + ".guard", "a") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    refused = spawn("run", "--timeout", "0.3", "--", *command, extra=extra)
                    finish(refused, 3)
                assert not list(queue.glob(f"*-{refused.pid}"))
                lock.rmdir()
                assert not marker.exists() and not lock.exists()
                assert rows("refused_power")[-1]["exit_code"] == 3
                finish(spawn("run", "--allow-battery", "--", *command, extra=extra))
                assert marker.exists()
                marker.unlink()
            _, err = finish(spawn("run", "--", *command, extra={"POWER_TEST": "unknown"}))
            assert marker.exists() and "alimentation inconnue" in err
            finish(spawn("run", "--", *PYTHON, "-c", "raise SystemExit(37)"), 37)
            assert rows("release")[-1]["exit_code"] == 37
            process = spawn("run", "--", *PYTHON, "-c",
                            "import sys; print(sys.stdin.read().strip()); print('erreur',file=sys.stderr)",
                            stdin=subprocess.PIPE)
            out, err = process.communicate(b"entree\n", timeout=5)
            assert process.returncode == 0 and out == b"entree\n" and err == b"erreur\n"
            assert not lock.exists()
            out, _ = finish(spawn("status"))
            assert json.loads(out)["state"] == "libre"
            print("gpulock: batterie, mode éco, erreur pmset et code de sortie ok", flush=True)
        finally:
            if (lock / "pgid").exists():
                groups.add(int((lock / "pgid").read_text()))
            for pid in groups:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)


if __name__ == "__main__":
    check()
    print("gpulock: auto-test ok")
