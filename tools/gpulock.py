"""Verrou GPU coopératif, sans démon ni dépendance ; sans argument : auto-tests.

python3 -I -B tools/gpulock.py run --owner bench --prio 10 -- commande arguments
python3 -I -B tools/gpulock.py status   # release [--force] : confirmation au terminal
GPU_LOCK choisit le chemin, sinon ~/.cache/saragossa/gpu.lock. Priorités 0..99,
FIFO à priorité égale, priorité stricte sans vieillissement (famine possible),
sans préemption ; --timeout borne seulement l'attente.
--allow-battery autorise les travaux CPU. Un verrou sans pid reste intouchable.
Le journal .log.jsonl et les archives .stale-* restent sur disque.

SIGTERM/SIGINT/SIGHUP sont transmis au groupe ; au deuxième signal, SIGKILL.
Le fichier holder reste verrouillé par le lanceur et hérité par l'enfant.
Après sa fermeture, un autre démarrage ou la mort de pid, child_pid et pgid
autorise la reprise. L'enfant attend leur publication avant exec. Les commandes doivent
rester dans ce groupe (pas de setsid/démonisation). Un PID réutilisé ou zombie
peut retarder la reprise ; --force est réservé à une intervention manuelle
et ne retire pas le ticket du tenant toujours vivant.
night.sh sonde à 60 s et ne participe pas à l'ordre des tickets.
"""

import argparse
import contextlib
import errno
import fcntl
import json
import math
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench/competitors"))
import sensors  # noqa: E402

SURVIVOR_GRACE = float(os.environ.get("_GPU_LOCK_SURVIVOR_GRACE", "30"))
SURVIVOR_TERM_GRACE = float(os.environ.get("_GPU_LOCK_SURVIVOR_TERM_GRACE", "10"))


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:  # Seul ESRCH prouve la mort ; EPERM ne suffit pas.
        return False
    except PermissionError:
        pass
    return True


def metadata(lock):
    def read(name):
        try:
            return (lock / name).read_text().strip()
        except FileNotFoundError:
            return ""
    return {name: read(name) for name in
            ("owner", "pid", "child_pid", "pgid", "cmd", "since", "prio", "boot")}


def boot_id():
    for key in ("kern.bootsessionuuid", "kern.boottime"):
        try:
            value = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                   text=True, check=True, timeout=3).stdout.strip()
            if value:
                return key + ":" + value
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def held(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ELOOP) or not path.is_file():
            return None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)


def occupied(lock, info):
    # Métadonnées incomplètes ou étrangères : rester conservateur.
    if not info["pid"].isdigit() or int(info["pid"]) <= 0:
        return True
    if held(lock / "holder"):
        return True
    boot = boot_id()
    # Un changement de source sysctl ne prouve pas un redémarrage.
    if (boot and info["boot"].partition(":")[0] == boot.partition(":")[0]
            and info["boot"] != boot):
        return False
    for key in ("pid", "child_pid", "pgid"):
        value = info[key]
        if value and (not value.isdigit() or int(value) <= 0 or
                      alive(-int(value) if key == "pgid" else int(value))):
            return True
    return False


@contextlib.contextmanager
def guard(lock, deadline=float("inf"), cancelled=lambda: False):
    # Le rename seul ne protège pas d'un réclamant retardé visant un nouveau tenant.
    # Ce verrou noyau ne couvre que les mutations et disparaît même après SIGKILL.
    with open(str(lock) + ".guard", "a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancelled():
                    raise InterruptedError
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                time.sleep(min(2, remaining))
                if time.monotonic() >= deadline:
                    raise TimeoutError
        yield


def log(lock, event, args, start, acquired=None, code=None, **extra):
    now = time.monotonic()
    row = dict(ts=time.time(), event=event, owner=args.owner, pid=os.getpid(),
               prio=args.prio, cmd=args.cmd, wait_s=(acquired or now) - start,
               duration_s=now - acquired if acquired is not None else 0,
               exit_code=code, **extra)
    with open(str(lock) + ".log.jsonl", "ab", buffering=0) as handle:
        handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode())


def run(lock, args):
    start, acquired, child, stopped = time.monotonic(), None, None, 0
    code, owned, ticket = 1, None, None
    handles = contextlib.ExitStack()

    def send(signum):
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def forward(signum, _frame):
        nonlocal stopped
        repeated = bool(stopped)
        stopped = signum
        send(signal.SIGKILL if repeated else signum)

    def power_allowed():
        power = sensors.power_state()
        if power["errors"] or power["on_ac"] is None or power["low_power"] is None:
            print("gpulock: alimentation inconnue, poursuite : " + json.dumps(power), file=sys.stderr)
        if not args.allow_battery:
            try:
                sensors.validate_power(power)
            except SystemExit as error:
                print(error, file=sys.stderr)
                log(lock, "refused_power", args, start, acquired, 3, power=power)
                return False
        return True

    previous = {s: signal.signal(s, forward)
                for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        if not power_allowed():
            code = 3
            return code
        queue = pathlib.Path(str(lock) + ".queue")
        ticket = queue / f"{args.prio:02d}-{time.time_ns()}-{os.getpid()}"
        with guard(lock, start + args.timeout, lambda: stopped):
            queue.mkdir(exist_ok=True)
            handle = handles.enter_context(ticket.open("x"))
            fcntl.flock(handle, fcntl.LOCK_EX)
        first = True
        while not stopped:
            if not first and time.monotonic() - start >= args.timeout:
                log(lock, "timeout", args, start, code=124)
                return 124
            first = False
            with guard(lock, start + args.timeout, lambda: stopped):
                tickets = [ticket.name]
                for path in queue.iterdir():
                    if path == ticket or not re.fullmatch(r"\d{2}-\d+-\d+", path.name):
                        continue
                    active = held(path)
                    if active:
                        tickets.append(path.name)
                    elif active is False:
                        path.unlink(missing_ok=True)
                if ticket.name == min(tickets):
                    info = metadata(lock)
                    if lock.exists() and not occupied(lock, info):
                        os.rename(lock, str(lock) + f".stale-{time.time_ns()}")
                        log(lock, "reclaim", args, start, previous=info)
                    try:
                        os.mkdir(lock)
                    except FileExistsError:
                        pass
                    else:
                        owned, acquired = lock.stat(), time.monotonic()
                        holder = handles.enter_context((lock / "holder").open("x"))
                        fcntl.flock(holder, fcntl.LOCK_EX)
                        for key, value in dict(owner=args.owner, cmd=json.dumps(args.cmd),
                                               since=time.time(), prio=args.prio,
                                               boot=boot_id(),
                                               pid=os.getpid()).items():
                            (lock / key).write_text(str(value) + "\n")
                        log(lock, "acquire", args, start, acquired)
                        break
            remaining = args.timeout - (time.monotonic() - start)
            if remaining <= 0:
                log(lock, "timeout", args, start, code=124)
                return 124
            # ponytail: sondage à 2 s, latence et famine possibles sous priorité stricte.
            time.sleep(min(2, remaining))
        if stopped:
            code = 128 + stopped
            return code
        if not power_allowed():
            code = 3
            return code
        # Pipe-barrière : si le lanceur meurt avant publication, EOF interdit exec.
        reader, writer = os.pipe()
        try:
            child = subprocess.Popen([sys.executable, "-I", "-B", "-c",
                                      "import os,sys; f=int(sys.argv[1]); b=os.read(f,1); "
                                      "os.close(f); b == b'1' or sys.exit(125); "
                                      "os.execvp(sys.argv[2],sys.argv[2:])", str(reader), *args.cmd],
                                     pass_fds=(reader, holder.fileno()), start_new_session=True,
                                     env=dict(os.environ, GPU_LOCK_HELD=str(lock.resolve())))
            with guard(lock):
                if metadata(lock)["pid"] != str(os.getpid()):
                    raise RuntimeError("verrou retiré avant lancement")
                (lock / "child_pid").write_text(f"{child.pid}\n")
                (lock / "pgid").write_text(f"{child.pid}\n")
                if not stopped:
                    os.write(writer, b"1")
        finally:
            os.close(reader)
            os.close(writer)
        if stopped:
            send(stopped)
        result = child.wait()
        code = result if result >= 0 else 128 - result
        return code
    except (TimeoutError, InterruptedError) as error:
        code = 124 if isinstance(error, TimeoutError) else 128 + stopped
        if code == 124:
            log(lock, "timeout", args, start, code=code)
        return code
    finally:
        if child is not None:
            if child.poll() is None:
                send(signal.SIGTERM)
            child.wait()
            for delay, signum in ((SURVIVOR_GRACE, signal.SIGTERM),
                                  (SURVIVOR_TERM_GRACE, signal.SIGKILL)):
                deadline = time.monotonic() + delay
                while alive(-child.pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if not alive(-child.pid):
                    break
                send(signum)
                log(lock, "killed_survivors", args, start, acquired, code, signal=signum)
            while alive(-child.pid):
                time.sleep(0.1)
        if owned is not None:
            with guard(lock):
                try:
                    current = lock.stat()
                except FileNotFoundError:
                    current = None
                if current is not None and os.path.samestat(owned, current):
                    shutil.rmtree(lock)
                    log(lock, "release", args, start, acquired, code)
        if ticket is not None:
            ticket.unlink(missing_ok=True)
        handles.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    if len(sys.argv) == 1:
        import runpy
        runpy.run_path(str(pathlib.Path(__file__).with_name("gpulock_checks.py")), run_name="__main__")
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    job = sub.add_parser("run")
    job.add_argument("--owner", default="gpulock")
    job.add_argument("--prio", type=int, choices=range(100), default=50)
    job.add_argument("--allow-battery", action="store_true")
    job.add_argument("--timeout", type=float, default=float("inf"))
    job.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    sub.add_parser("release").add_argument("--force", action="store_true")
    args = parser.parse_args()
    lock = pathlib.Path(os.environ.get("GPU_LOCK") or "~/.cache/saragossa/gpu.lock").expanduser().resolve()
    lock.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "run":
        args.cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
        if not args.cmd or math.isnan(args.timeout) or args.timeout < 0:
            parser.error("commande requise et timeout positif ou nul")
        return run(lock, args)
    confirmed = None
    if args.action == "release" and args.force and lock.exists():
        with guard(lock):
            confirmed = metadata(lock)
        if not sys.stdin.isatty():
            print("gpulock: --force exige un terminal", file=sys.stderr)
            return 2
        print(f"Forcer la libération de {lock} ? Saisir OUI : ", end="", file=sys.stderr, flush=True)
        if sys.stdin.readline().strip() != "OUI":
            return 1
    with guard(lock):
        info = metadata(lock)
        if args.action == "status":
            state = ("libre" if not lock.exists() else "propriétaire inconnu" if not info["pid"]
                     else "occupé" if occupied(lock, info) else "orphelin")
            print(json.dumps(dict(state=state, **info), ensure_ascii=False))
        elif lock.exists():
            if args.force:
                if info != confirmed:
                    print("gpulock: tenant changé depuis la confirmation", file=sys.stderr)
                    return 1
            elif occupied(lock, info):
                print("gpulock: tenant vivant ou inconnu, libération refusée", file=sys.stderr)
                return 1
            os.rename(lock, str(lock) + f".stale-{time.time_ns()}")
            args.owner, args.prio, args.cmd = info["owner"], info["prio"], info["cmd"]
            log(lock, "release", args, time.monotonic(), code=0, force=args.force, previous=info)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"gpulock: {error}", file=sys.stderr)
        raise SystemExit(1)
