"""Collecte macmon continue, facultative, hors du chemin chronométré (stdlib).

Les lignes brutes restent dans sensors.jsonl. Les horodatages monotones de
réception servent à sélectionner les échantillons de chaque requête.
"""

import json
import math
import pathlib
import statistics
import subprocess
import threading
import time


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


def _self_check():
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
