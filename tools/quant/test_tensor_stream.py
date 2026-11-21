"""Régressions du flux safetensors ; --mlx-cpu vérifie les payloads bit à bit."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_ls
import opt_quant
import tensor_stream


def payload_hash(tensor):
    with tensor.path.open("rb") as handle:
        handle.seek(tensor.offset)
        return hashlib.sha256(handle.read(tensor.nbytes)).hexdigest()


def stdlib_checks():
    # Une longue entrée ne doit pas créer une longue file de blocs en mémoire.
    class Future:
        def __init__(self, pool, value):
            self.pool, self.value = pool, value

        def result(self):
            self.pool.outstanding -= 1
            return self.value

        def cancel(self):
            self.pool.outstanding -= 1

    class Pool:
        outstanding = 0
        peak = 0

        def submit(self, function, value):
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
            return Future(self, value)

    pool = Pool()
    assert list(tensor_stream.fit_blocks(pool, iter(range(1000)), 4)) == list(range(1000))
    assert pool.peak == 4 and pool.outstanding == 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "model.safetensors"
        header = {"x": {"dtype": "BF16", "shape": [2, 64], "data_offsets": [0, 256]}}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(256))
        assert tensor_stream.catalog([path])["x"].shape == (2, 64)
        for paths in ([path, path],):
            try:
                tensor_stream.catalog(paths)
                raise AssertionError("doublon accepté")
            except ValueError:
                pass
        path.write_bytes(path.read_bytes()[:-1])
        try:
            tensor_stream.catalog([path])
            raise AssertionError("payload tronqué accepté")
        except ValueError:
            pass
        with patch.object(tensor_stream.shutil, "disk_usage") as usage:
            usage.return_value.free = 100 * 2**30
            tensor_stream.check_space(root / "absent" / "nested", 30)
            usage.assert_called_with(root)
            assert not (root / "absent").exists()
            usage.return_value.free = 29 * 2**30
            try:
                tensor_stream.copy_shard(path, root / "out", 30)
                raise AssertionError("réserve disque ignorée")
            except OSError:
                assert (root / "out").stat().st_size == 0
    print("tensor_stream: en-têtes, doublons, troncature et réserve disque ok")


def mlx_checks():
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, template = root / "source", root / "template"
        source.mkdir()
        template.mkdir()
        # Plusieurs experts et un bloc traversant leur frontière.
        x = mx.array(np.random.default_rng(17).normal(size=(2, 14, 128)), mx.bfloat16)
        fused = "model.language_model.layers.0.experts.gate_up_proj"
        mx.save_safetensors(str(source / "model.safetensors"), {fused: x})
        # convert_ls exige le tokenizer source et le recopie à l'identique.
        (source / "tokenizer.json").write_text('{"fixture": true}')
        bases = ["language_model.model.layers.0.experts.switch_glu." + p
                 for p in ("gate_proj", "up_proj")]
        matrix = {base: mx.array([0 if i % 13 == 0 else i + 1 for i in range(128)], mx.float32)
                  for base in bases}
        matrix_path = root / "imatrix.safetensors"
        mx.save_safetensors(str(matrix_path), matrix)
        checked = 0
        for bits in (3, 4, 5, 6, 8):
            for blend in (False, True):
                shards = [{}, {}]
                expected = {}
                for base, half in zip(bases, mx.split(x, 2, axis=-2)):
                    initial = mx.quantize(half, group_size=64, bits=bits)
                    fitted = opt_quant.fit(half, bits=bits, weight=matrix[base], imatrix_blend=blend)
                    for i, suffix in enumerate((".weight", ".scales", ".biases")):
                        shards[i % 2][base + suffix] = initial[i]
                        expected[base + suffix] = fitted[i]
                shards[0]["unchanged"] = mx.array([1, 2, 3], mx.int32)
                expected["unchanged"] = shards[0]["unchanged"]
                for i, shard in enumerate(shards):
                    mx.save_safetensors(str(template / f"model-{i}.safetensors"), shard)
                (template / "config.json").write_text(json.dumps({
                    "quantization": {"bits": bits, "group_size": 64}}))
                oracle = root / "oracle.safetensors"
                mx.save_safetensors(str(oracle), expected)
                destination = root / f"out-{bits}-{blend}"
                convert_ls.convert(source, template, destination, matrix_path, blend,
                                   block_rows=3, min_free_gib=0,
                                   workers=2 if bits in (3, 8) else 1)
                actual = tensor_stream.catalog(sorted(destination.glob("model*.safetensors")))
                reference = tensor_stream.catalog([oracle])
                assert actual.keys() == reference.keys()
                for key in actual:
                    assert payload_hash(actual[key]) == payload_hash(reference[key]), (bits, blend, key)
                assert not list(destination.glob("*.partial"))
                assert (destination / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
                checked += 1
    assert checked == 10, checked
    print("tensor_stream: SHA-256 identiques 3/4/5/6/8 bits, blend, experts et shards croisés")


if __name__ == "__main__":
    stdlib_checks()
    if sys.argv[1:] == ["--mlx-cpu"]:
        mlx_checks()
    elif sys.argv[1:]:
        raise SystemExit("seul argument accepté : --mlx-cpu")
