"""Chemins du bench, lus dans `paths.sh` (seule source, local.env compris)."""

import functools
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def paths():
    script = 'source "$1"; for k in "${BENCH_PATH_KEYS[@]}"; do printf "%s=%s\\n" "$k" "${!k}"; done'
    out = subprocess.run(["bash", "-c", script, "bash", str(HERE / "paths.sh")],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"paths.sh: {out.stderr.strip()}")
    return {key: value for key, _, value in
            (line.partition("=") for line in out.stdout.splitlines())}


def path(key):
    return pathlib.Path(paths()[key])


def sanitize(value):
    """Remplace le dossier personnel par `~` dans tout ce qui sera publié."""
    home = str(pathlib.Path.home())
    if isinstance(value, str):
        return value.replace(home, "~")
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value
