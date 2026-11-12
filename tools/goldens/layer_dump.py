"""Shared layer-dump contract; no MLX import or device initialization."""
import hashlib
import json
from pathlib import Path

FORMAT = "saragossa-layer-dump-v1"
POINTS = (
    "input", "attention_input", "attention_raw", "attention_out",
    "post_attention_residual", "dense_input", "dense_raw", "dense_out",
    "router_input", "router_logits", "router_indices", "router_weights",
    "moe_input", "moe_raw", "moe_out", "ffn_sum", "ffn_out", "pre_scalar", "output",
)


def key(layer, point):
    return point if layer is None else f"layers.{layer:03}.{point}"


def expected_points(layers):
    return [(None, "embeddings")] + [
        (i, point) for i in range(layers) for point in POINTS
    ] + [(None, "final_norm"), (None, "logits")]


def validate_ids(ids):
    if not isinstance(ids, list) or not ids or any(
        type(i) is not int or i < 0 for i in ids
    ):
        raise ValueError("prompt_ids must be a nonempty list of nonnegative integers")
    return ids


def golden_ids(path, prompt_index=0):
    """The golden JSON is metadata; IDs live in its SHA256-checked UTF-8 payload."""
    path = Path(path)
    metadata = json.loads(path.read_text())
    payload = path.with_suffix(".bin").read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["payload_sha256"]:
        raise ValueError("golden payload SHA256 mismatch")
    fixture = json.loads(payload)
    if fixture["model"] != "gemma4" or not 0 <= prompt_index < len(fixture["cases"]):
        raise ValueError("expected Gemma 4 golden and a valid prompt index")
    return validate_ids(fixture["cases"][prompt_index]["prompt_ids"])


class Writer:
    def __init__(self, directory, ids, layers, backend, **metadata):
        self.directory = Path(directory)
        self.directory.mkdir()  # Deliberately refuse existing output, including symlinks.
        self.index = dict(format=FORMAT, backend=backend, prompt_ids=validate_ids(ids),
                          num_layers=layers, logits="last-token-after-softcap", **metadata)
        self.entries = {}

    def save(self, layer, point, value, source_dtype=None):
        import numpy as np
        value = np.asarray(value)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or not all(value.shape):
            raise ValueError(f"invalid tensor shape for {key(layer, point)}: {value.shape}")
        name = key(layer, point)
        if name in self.entries:
            raise ValueError(f"duplicate capture: {name}")
        filename = name + ".f32"
        with (self.directory / filename).open("xb") as f:
            f.write(np.asarray(value, dtype="<f4").tobytes(order="C"))
        self.entries[name] = dict(layer=layer, point=point, shape=list(value.shape),
                                  file=filename, dtype="<f4",
                                  source_dtype=source_dtype or str(value.dtype))

    def finish(self):
        expected = expected_points(self.index["num_layers"])
        if set(self.entries) != {key(*p) for p in expected}:
            raise ValueError("incomplete layer capture")
        entries = [self.entries[key(*p)] for p in expected]
        for entry in entries:
            rows = 1 if entry["point"] in ("final_norm", "logits") else len(self.index["prompt_ids"])
            if entry["shape"][0] != rows:
                raise ValueError(f"incomplete rows: {entry}")
        self.index["entries"] = entries
        with (self.directory / "index.json").open("x") as f:
            json.dump(self.index, f, indent=2)
            f.write("\n")


class CallHooks:
    """Patch class methods in memory: Python special calls bypass instance attributes."""
    def __init__(self):
        self.callbacks = {}
        self.originals = {}

    def add(self, module, before=None, after=None):
        cls = type(module)
        if id(module) in self.callbacks:
            raise ValueError("module hooked twice")
        self.callbacks[id(module)] = (before, after)
        if cls not in self.originals:
            original = cls.__call__
            self.originals[cls] = original
            callbacks = self.callbacks

            def call(instance, *args, **kwargs):
                pre, post = callbacks.get(id(instance), (None, None))
                if pre:
                    pre(*args, **kwargs)
                output = original(instance, *args, **kwargs)
                if post:
                    post(output)
                return output

            cls.__call__ = call

    def close(self):
        for cls, original in reversed(list(self.originals.items())):
            cls.__call__ = original
        self.originals.clear()
        self.callbacks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
