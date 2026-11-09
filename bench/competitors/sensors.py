"""Collecte macmon continue, facultative, hors du chemin chronométré (stdlib).

Les lignes brutes restent dans sensors.jsonl. Les horodatages monotones de
réception servent à sélectionner les échantillons de chaque requête.
"""

import ctypes
import json
import math
import pathlib
import statistics
import subprocess
import threading
import time


def pmset(*args):
    """Lecture bornée ; les erreurs sont des informations, jamais une panne du run."""
    try:
        return subprocess.run(["pmset", "-g", *args], capture_output=True, text=True,
                              check=True, timeout=3).stdout, None
    except (OSError, subprocess.SubprocessError) as error:
        return "", str(error)


def power_state():
    batt, batt_error = pmset("batt")
    mode, mode_error = pmset()
    modes = [line.split() for line in mode.splitlines()
             if line.split() and line.split()[0] in ("powermode", "lowpowermode")]
    values = [parts[-1] for parts in modes if len(parts) >= 2]
    return {"on_ac": (True if "AC Power" in batt else
                       False if "Battery Power" in batt else None),
            "low_power": any(v == "1" for v in values) if values else None,
            "source_raw": batt.strip(), "mode_raw": [" ".join(p) for p in modes],
            "errors": [e for e in (batt_error, mode_error) if e]}


def validate_power(state):
    if state["on_ac"] is False or state["low_power"] is True:
        raise SystemExit("alimentation : secteur requis et mode basse consommation désactivé ; "
                         "campagne arrêtée, reprise possible")


def thermal_pressure():
    """Niveau `OSThermalPressureLevel` par notify(3), nul si illisible."""
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        token, level = ctypes.c_int(), ctypes.c_uint64()
        if lib.notify_register_check(b"com.apple.system.thermalpressurelevel",
                                     ctypes.byref(token)) != 0:
            return None
        try:
            status = lib.notify_get_state(token.value, ctypes.byref(level))
            return None if status != 0 else level.value
        finally:
            lib.notify_cancel(token.value)
    except (OSError, AttributeError):
        return None


def soc_sensors():
    """Capteurs macmon informatifs ; zéro reste une mesure pour les ventilateurs."""
    empty = {"cpu_temp_c": None, "gpu_temp_c": None, "fan_rpm": None,
             "sys_power_w": None}

    def value(number):
        return (number if type(number) in (int, float)
                and math.isfinite(number) and number > 0 else None)

    try:
        out = subprocess.run(["macmon", "pipe", "-s", "1", "-i", "200"],
                             capture_output=True, text=True, timeout=2, check=True).stdout
        sample = json.loads(out)
        temp = sample.get("temp") or {}
        return {"cpu_temp_c": value(temp.get("cpu_temp_avg")),
                "gpu_temp_c": value(temp.get("gpu_temp_avg")),
                "fan_rpm": [fan.get("rpm") for fan in sample.get("fans", [])] or None,
                "sys_power_w": value(sample.get("sys_power"))}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError):
        return empty


def thermal(sampler=None):
    """État de `pmset -g therm` : bridé si un niveau d'alerte est enregistré
    ou si une limite de vitesse CPU inférieure à 100 est annoncée."""
    out, error = pmset("therm")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    throttled = False
    for line in lines:
        if "warning level" in line.lower() and not line.startswith("Note: No"):
            throttled = True
        if line.startswith("CPU_Speed_Limit") and not line.rstrip().endswith("100"):
            throttled = True
    state = {"raw": " | ".join(lines), "throttled": throttled,
             "pressure": thermal_pressure(),
             **(sampler.latest()[1] if sampler is not None else soc_sensors())}
    if error:
        state["error"] = error
    return state


def wait_idle(matrix, sampler, *, require_sensor=False):
    """Trois lectures distinctes sous le seuil.

    Par défaut, capteur absent : pas d'attente (matrice et nuit inchangées).
    require_sensor attend son rétablissement jusqu'au plafond et trace la preuve.
    """
    start = time.monotonic()
    deadline = start + matrix.get("busy_wait_max_s", 300)
    consecutive, previous = 0, None
    while sampler is not None or require_sensor:
        stamp, sample = sampler.latest() if sampler is not None else (None, {})
        ratio = sample.get("gpu_active_ratio")
        if ratio is None:
            if not require_sensor:
                break
            consecutive, previous = 0, None
        elif stamp != previous:
            consecutive = consecutive + 1 if ratio < matrix.get("busy_gpu_ratio", 0.2) else 0
            previous = stamp
        if consecutive >= 3:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if require_sensor:
                return {"busy_wait_s": round(time.monotonic() - start, 1),
                        "machine_busy": ratio is not None and ratio >= matrix.get("busy_gpu_ratio", 0.2), "idle_established": False,
                        "reason": "capteur indisponible" if ratio is None else "repos non confirmé au plafond"}
            return {"busy_wait_s": round(time.monotonic() - start, 1), "machine_busy": True}
        time.sleep(min(1, remaining))
    result = {"busy_wait_s": round(time.monotonic() - start, 1), "machine_busy": False}
    if require_sensor:
        result["idle_established"] = True
    return result


def positive(value):
    return (value if type(value) in (int, float) and math.isfinite(value)
            and value > 0 else None)


def rpm(value):
    """Vitesse de ventilateur : 0 est une vraie mesure, ventilateur arrêté."""
    return (value if type(value) in (int, float) and math.isfinite(value)
            and value >= 0 else None)


def decode(sample):
    temp = sample.get("temp") or {}
    usage = sample.get("gpu_usage") or [None]
    fans = sample.get("fans")
    return {"cpu_temp_c": positive(temp.get("cpu_temp_avg")),
            "gpu_temp_c": positive(temp.get("gpu_temp_avg")),
            "gpu_freq_mhz": positive(sample.get("gpu_freq_mhz", usage[0])),
            "gpu_active_ratio": positive(sample.get("gpu_active_ratio")),
            "sys_power_w": positive(sample.get("sys_power")),
            "cpu_usage": positive(sample.get("cpu_usage_pct")),
            "fan_rpm": [rpm(fan.get("rpm")) for fan in fans if isinstance(fan, dict)] or None
            if isinstance(fans, list) else None}


def summarize(samples):
    result = {"n_samples": len(samples)}
    for key in ("cpu_temp_c", "gpu_temp_c"):
        values = [s[key] for s in samples if s.get(key) is not None]
        for suffix, value in (("start", samples[0].get(key) if samples else None),
                              ("max", max(values) if values else None),
                              ("end", samples[-1].get(key) if samples else None),
                              ("mean", statistics.fmean(values) if values else None)):
            result[f"{key}_{suffix}"] = value
    for key, suffix, fn in (("gpu_freq_mhz", "median", statistics.median),
                            ("gpu_active_ratio", "median", statistics.median),
                            ("sys_power_w", "mean", statistics.fmean),
                            ("cpu_usage", "mean", statistics.fmean)):
        values = [s[key] for s in samples if s.get(key) is not None]
        result[f"{key}_{suffix}"] = fn(values) if values else None
    return result


class Sampler:
    """Un processus et un lecteur par run ; aucune E/S dans snapshot/latest."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.proc = self.thread = self.out = None
        self.samples = []
        self.lock = threading.Lock()

    def __enter__(self):
        try:
            self.out = self.path.open("ab")
            self.proc = subprocess.Popen(["macmon", "pipe", "-i", "1000"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except OSError:
            self.close()
        except BaseException:
            self.close()
            raise
        return self

    def _read(self):
        try:
            for raw in self.proc.stdout:
                stamp = time.perf_counter_ns()
                self.out.write(raw)
                self.out.flush()
                try:
                    sample = decode(json.loads(raw))
                except (ValueError, TypeError, AttributeError, IndexError):
                    continue
                with self.lock:
                    self.samples.append((stamp, sample))
        except (OSError, ValueError):
            pass

    def latest(self):
        if self.proc is None or self.proc.poll() is not None:
            return None, {}
        with self.lock:
            stamp, sample = self.samples[-1] if self.samples else (None, {})
        if stamp is None or time.perf_counter_ns() - stamp > 3_000_000_000:
            return None, {}
        return stamp, sample

    def snapshot(self, start, end):
        if self.proc is None or self.proc.poll() is not None:
            return summarize([])
        with self.lock:
            samples = [s for t, s in self.samples if start <= t <= end]
        return summarize(samples)

    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.proc is not None and self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.out is not None:
            self.out.close()

    def __exit__(self, *_):
        self.close()


def _thermal_check():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    lib = Mock()
    lib.notify_register_check.return_value = 0

    def read_level(token, level):
        level._obj.value = 3
        return 0

    lib.notify_get_state.side_effect = read_level
    with patch.object(ctypes, "CDLL", return_value=lib):
        assert thermal_pressure() == 3
        lib.notify_cancel.assert_called_once()
        lib.notify_register_check.return_value = 1
        assert thermal_pressure() is None
        assert lib.notify_cancel.call_count == 1
        lib.notify_register_check.return_value = 0
        lib.notify_get_state.side_effect = None
        lib.notify_get_state.return_value = 1
        assert thermal_pressure() is None
        assert lib.notify_cancel.call_count == 2
    with patch.object(ctypes, "CDLL", side_effect=OSError("absent")):
        assert thermal_pressure() is None
    sample = {"temp": {"cpu_temp_avg": 62.5, "gpu_temp_avg": 51.0},
              "fans": [{"rpm": 1200}, {"rpm": 0.0}, {}], "sys_power": 42.0}
    sensors = {"cpu_temp_c": 62.5, "gpu_temp_c": 51.0, "fan_rpm": [1200, 0.0, None],
               "sys_power_w": 42.0}
    with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(sample))) as call:
        assert soc_sensors() == sensors
    assert call.call_args.args[0] == ["macmon", "pipe", "-s", "1", "-i", "200"]
    assert call.call_args.kwargs["timeout"] == 2 and call.call_args.kwargs["check"]
    empty = dict.fromkeys(sensors)
    for sample in ({}, {"temp": {"cpu_temp_avg": 0.0, "gpu_temp_avg": 0.0}},
                   None, [], {"temp": []}, {"fans": None}):
        with patch.object(subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(sample))):
            assert soc_sensors() == empty
    for error in (FileNotFoundError("macmon"), subprocess.TimeoutExpired("macmon", 2),
                  subprocess.CalledProcessError(1, "macmon"), ValueError("JSON")):
        with patch.object(subprocess, "run", side_effect=error):
            assert soc_sensors() == empty
    for raw, throttled in (("Note: No thermal warning level has been recorded", False),
                           ("CPU_Speed_Limit = 100", False),
                           ("CPU_Speed_Limit = 80", True), ("Thermal warning level: 1", True)):
        with (patch.object(subprocess, "run", return_value=SimpleNamespace(stdout=raw)),
              patch(__name__ + ".thermal_pressure", return_value=4),
              patch(__name__ + ".soc_sensors", return_value=sensors)):
            state = thermal()
        assert state == {"raw": raw, "throttled": throttled, "pressure": 4, **sensors}


def _idle_check():
    from unittest.mock import Mock, patch

    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    with (patch.object(time, "monotonic", side_effect=lambda: clock[0]),
          patch.object(time, "sleep", side_effect=sleep)):
        sensor = Mock()
        sensor.latest.side_effect = [(i, {"gpu_active_ratio": r}) for i, r in enumerate((0.1, 0.3, 0.1, 0.1, 0.1))]
        assert wait_idle({}, sensor) == {"busy_wait_s": 4, "machine_busy": False}
        sensor.latest.side_effect = None
        sensor.latest.return_value = (1, {"gpu_active_ratio": 0.1})
        assert wait_idle({"busy_wait_max_s": 3}, sensor) == {"busy_wait_s": 3, "machine_busy": True}
        sensor.latest.side_effect = [(i, {"gpu_active_ratio": 0.3}) for i in range(3)]
        assert wait_idle({"busy_gpu_ratio": 0.4}, sensor) == {"busy_wait_s": 2, "machine_busy": False}
        sensor.latest.side_effect = None
        sensor.latest.return_value = (None, {})
        assert wait_idle({}, sensor) == {"busy_wait_s": 0, "machine_busy": False}
        assert wait_idle({}, None) == {"busy_wait_s": 0, "machine_busy": False}
        for missing in (sensor, None):
            assert wait_idle({"busy_wait_max_s": 3}, missing, require_sensor=True) == {
                "busy_wait_s": 3, "machine_busy": False, "idle_established": False,
                "reason": "capteur indisponible"}
        sensor.latest.side_effect = [(None, {}), (1, {"gpu_active_ratio": 0.1}),
                                     (2, {"gpu_active_ratio": 0.1}), (None, {}),
                                     (3, {"gpu_active_ratio": 0.1}),
                                     (4, {"gpu_active_ratio": 0.1}),
                                     (5, {"gpu_active_ratio": 0.1})]
        assert wait_idle({"busy_wait_max_s": 7}, sensor, require_sensor=True) == {
            "busy_wait_s": 6, "machine_busy": False, "idle_established": True}
        sensor.latest.side_effect = None
        for ratio in (0.1, 0.2, 0.3):
            sensor.latest.return_value = (1, {"gpu_active_ratio": ratio})
            assert wait_idle({"busy_wait_max_s": 2}, sensor, require_sensor=True) == {
                "busy_wait_s": 2, "machine_busy": ratio >= 0.2, "idle_established": False,
                "reason": "repos non confirmé au plafond"}

def _self_check():
    _thermal_check()
    _idle_check()
    import io
    import tempfile
    from unittest.mock import Mock, patch

    a = decode({"temp": {"cpu_temp_avg": 50, "gpu_temp_avg": 40},
                "gpu_usage": [800, 0.5], "gpu_active_ratio": 0.8,
                "sys_power": 30, "cpu_usage_pct": 0.2})
    b = dict(a, gpu_temp_c=60, gpu_freq_mhz=1000, sys_power_w=50)
    summary = summarize([a, b])
    assert summary == {"n_samples": 2, "cpu_temp_c_start": 50, "cpu_temp_c_max": 50,
                       "cpu_temp_c_end": 50, "cpu_temp_c_mean": 50,
                       "gpu_temp_c_start": 40, "gpu_temp_c_max": 60,
                       "gpu_temp_c_end": 60, "gpu_temp_c_mean": 50,
                       "gpu_freq_mhz_median": 900, "gpu_active_ratio_median": 0.8,
                       "sys_power_w_mean": 40, "cpu_usage_mean": 0.2}
    assert all(v is None for v in decode({"gpu_active_ratio": 0.0}).values())
    assert decode({"fans": [{"rpm": 0}, {"rpm": 1200}, {}]})["fan_rpm"] == [0, 1200, None]
    assert decode({"fans": "absent"})["fan_rpm"] is None
    partial = summarize([decode({}), a, decode({})])
    assert partial["gpu_temp_c_start"] is partial["gpu_temp_c_end"] is None
    assert partial["gpu_temp_c_max"] == partial["gpu_temp_c_mean"] == 40
    for value in (0, -1, float("nan"), float("inf"), "0", True):
        assert positive(value) is None
    raw = b'{"temp":{"gpu_temp_avg":45}}\ninvalid\n'
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sensors.jsonl"
        proc = Mock(stdout=io.BytesIO(raw))
        proc.poll.return_value = None
        with patch.object(subprocess, "Popen", return_value=proc) as launch:
            try:
                with Sampler(path) as sampler:
                    sampler.thread.join(timeout=1)
                    assert not sampler.thread.is_alive()
                    assert sampler.snapshot(0, time.perf_counter_ns())["gpu_temp_c_mean"] == 45
                    assert sampler.snapshot(0, 0) == summarize([])
                    assert sampler.latest()[1]["gpu_temp_c"] == 45
                    proc.poll.return_value = 1
                    assert sampler.latest() == (None, {})
                    assert sampler.snapshot(0, time.perf_counter_ns()) == summarize([])
                    raise SystemExit("arrêt")
            except SystemExit:
                pass
        launch.assert_called_once_with(["macmon", "pipe", "-i", "1000"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.terminate.assert_called_once()
        assert path.read_bytes() == raw and sampler.out.closed
        with patch.object(subprocess, "Popen", side_effect=FileNotFoundError):
            with Sampler(path) as sampler:
                assert sampler.snapshot(0, 1) == summarize([])
        assert sampler.out.closed
        proc = Mock(stdout=io.BytesIO(b""))
        proc.wait.side_effect = [subprocess.TimeoutExpired("macmon", 2), 0]
        with patch.object(subprocess, "Popen", return_value=proc):
            with Sampler(path):
                pass
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2 and proc.stdout.closed


if __name__ == "__main__":
    _self_check()
    print("sensors: auto-test ok")
