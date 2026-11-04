"""Mémoire d'un processus macOS par `proc_pid_rusage` (libproc), stdlib seule.

`phys_footprint` est le champ que rapportent `footprint` et le moniteur
d'activité : mémoire anonyme et compressée du processus, buffers Metal
privés compris, pages de fichiers mappées propres exclues. Le noyau tient
aussi `lifetime_max_phys_footprint`, le vrai maximum depuis le lancement :
pas de pic manqué entre deux échantillons.

La mémoire résidente adossée à un fichier (poids mmapés par llama.cpp et
MLX) se lit à part, une fois, par `vmmap -summary` (voir `file_backed`).
"""

import ctypes
import errno
import re
import subprocess
import threading
import time

RUSAGE_INFO_V4 = 4


class RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(RusageInfoV4)]
_libproc.proc_pid_rusage.restype = ctypes.c_int


def rusage(pid):
    """phys_footprint courant, maximum à vie et résident, en octets.

    None si le processus n'existe plus (ESRCH) ; `{"error": "EPERM"}` (ou autre
    code) si la lecture est refusée : les deux cas ne se confondent pas.
    """
    info = RusageInfoV4()
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        if code == errno.ESRCH:
            return None
        return {"error": errno.errorcode.get(code, str(code))}
    return {
        "phys_footprint": info.ri_phys_footprint,
        "lifetime_max_phys_footprint": info.ri_lifetime_max_phys_footprint,
        "resident_size": info.ri_resident_size,
    }


class Sampler:
    """Échantillonne phys_footprint toutes les `period` s dans un thread."""

    def __init__(self, pid, period=0.2):
        self.pid = pid
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            sample = rusage(self.pid)
            if sample is not None and "phys_footprint" in sample:
                self.samples.append((time.perf_counter_ns(), sample["phys_footprint"]))
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


def file_backed(pid):
    """Résident adossé à des fichiers selon `vmmap -summary` (lent : hors mesure de débit)."""
    out = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True,
                         check=False).stdout
    result = {"raw_lines": []}
    for line in out.splitlines():
        if line.startswith(("mapped file", "TOTAL", "IOAccelerator", "Physical footprint")):
            result["raw_lines"].append(line.strip())
    return result


def system_memory():
    """Pression, compression et swap du système (vm_stat, sysctl)."""
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1)) if "page size" in vm else 16384
    fields = {}
    for line in vm.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = int(value.strip().rstrip(".")) * page
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                          check=False).stdout.strip()
    level = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return {
        "compressed_bytes": fields.get("Pages occupied by compressor"),
        "free_bytes": fields.get("Pages free"),
        "swapouts_bytes": fields.get("Swapouts"),
        "swapusage": swap,
        "pressure_level": level,
    }


def collector_overhead(pid, n=1000):
    """Coût moyen d'un appel `rusage`, en microsecondes (mesuré une fois)."""
    start = time.perf_counter_ns()
    for _ in range(n):
        rusage(pid)
    return (time.perf_counter_ns() - start) / n / 1000


if __name__ == "__main__":
    import os

    me = rusage(os.getpid())
    assert me is not None and me["phys_footprint"] > 0
    # PID inexistant (au-delà de kern.maxproc) : disparition, pas refus.
    assert rusage(99_999_999) is None
    assert me["lifetime_max_phys_footprint"] >= me["phys_footprint"]
    print(me, f"{collector_overhead(os.getpid()):.1f} µs/appel", system_memory())
