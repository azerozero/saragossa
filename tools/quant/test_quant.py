"""Auto-tests stdlib ; --mlx-cpu ajoute de petits tenseurs MLX, sans modèle externe."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_ls
import imatrix
import mixed_quant
import opt_quant
import sensitivity


def stdlib_checks():
    for module in (opt_quant, imatrix, convert_ls, sensitivity):
        module._self_check()
    mixed_quant.main([])
    # Import et aide restent accessibles sans MLX, y compris sous python -I.
    assert "mlx.core" not in sys.modules
    dense = mixed_quant.parse_args(["--input", "src", "--output", "dst", "--target-bpw", "4.8"])
    assert dense.high_bits == 6 and dense.force_high == mixed_quant.DENSE_FORCE_HIGH
    assert mixed_quant.parse_args(["--input", "src", "--output", "dst"]).high_bits == 4
    # Des scores identiques gardent une allocation reproductible quel que soit l'ordre d'entrée.
    scores = {n: {4: 20, 6: 3, 8: 1} for n in ("b", "a", "c")}
    sizes = {"b": 128, "a": 64, "c": 256}
    a = sensitivity.allocate(scores, sizes, 448, 0, 5.1)
    b = sensitivity.allocate(dict(reversed(list(scores.items()))), sizes, 448, 0, 5.1)
    assert a == b and a[1] <= 5.1
    # Les petits modules peuvent utiliser le budget restant après un gros module trop coûteux.
    assert a[0]["a"] == 8
    # Une importance mal dimensionnée ne doit pas devenir silencieusement uniforme.
    class Vector:
        shape = (1, 64)
    try:
        imatrix.importance_for({"projection": Vector()}, "projection", 64)
        raise AssertionError("forme 2D acceptée")
    except ValueError:
        pass
    print("test_quant: auto-tests stdlib ok")


def mlx_checks():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    x = mx.array([[((i * 19 + r * 23) % 113 - 56) / 37 for i in range(128)]
                  for r in range(6)], mx.bfloat16).reshape(2, 3, 128)
    importance = mx.array([0] * 64 + [0.001 if i % 7 else 10 for i in range(64)], mx.float32)
    for bits in sensitivity.WIDTHS:
        q = mx.array([i % (1 << bits) for i in range(128)], mx.uint32)
        packed = opt_quant.pack_codes(mx, q, (1, 128), bits)
        back = mx.dequantize(packed, mx.ones((1, 2)), mx.zeros((1, 2)), group_size=64, bits=bits)
        assert mx.array_equal(back.reshape(-1), q).item()
        plain = opt_quant.fit(x, bits=bits)
        uniform = opt_quant.fit(x, bits=bits, weight=mx.ones((128,)))
        for a, b in zip(plain[:3], uniform[:3]):
            assert a.dtype == b.dtype and mx.array_equal(a, b).item()
        for a, b in zip(plain[3:], uniform[3:]):
            assert mx.allclose(a, b).item()
        p, s, b, old, new = opt_quant.fit(x, bits=bits, weight=importance)
        restored = mx.dequantize(p, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)
        error = (importance * (restored - x.astype(mx.float32)) ** 2).sum()
        expected = mx.sqrt(error / (6 * importance.sum())).item()
        assert abs(new.item() - expected) < 1e-6 and new.item() <= old.item() * 1.001

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) % 16 for c in text]
        def decode(self, ids):
            return "".join(chr(97 + i) for i in ids)

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(16, 64)
            self.lm_head = nn.Linear(64, 16, bias=False)
            self.model.embed_tokens.weight = mx.array(
                [[(i + j) / 64 for j in range(64)] for i in range(16)], mx.bfloat16)
            self.lm_head.weight = self.model.embed_tokens.weight
        def __call__(self, ids):
            return self.lm_head(self.model.embed_tokens(ids))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Language()
        def __call__(self, ids):
            return self.language_model(ids)

    # Capture réelle et conversion complète : mêmes fichiers/config que la recette 27B.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = Tiny()
        corpus, matrix_file = root / "corpus.txt", root / "imatrix.safetensors"
        corpus.write_text("abcdefgh")
        with patch.object(imatrix, "load_bf16", return_value=(model, Tokenizer())):
            imatrix.capture(root, corpus, matrix_file, samples=2, length=4)
        matrix = mx.load(str(matrix_file))
        assert set(matrix) == {"language_model.lm_head"}
        ids = mx.array([Tokenizer().encode("abcdefgh")])
        activations = model.language_model.model.embed_tokens(ids).astype(mx.float32)
        expected = (activations * activations).mean(axis=(0, 1))
        assert mx.allclose(matrix["language_model.lm_head"], expected).item()
        assignment, report = sensitivity.scan(model, matrix, "", 6.5, 6)
        source, template, destination = (root / p for p in ("source", "template", "destination"))
        source.mkdir()
        template.mkdir()
        original = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(source / "model.safetensors"), original)
        quantized = {}
        config = {"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}
        for name, bits in assignment.items():
            config["quantization"][name] = {"bits": bits, "group_size": 64, "mode": "affine"}
            values = mx.quantize(original[name + ".weight"], group_size=64, bits=bits)
            quantized.update((name + suffix, v) for suffix, v in zip((".weight", ".scales", ".biases"), values))
        mx.save_safetensors(str(template / "model.safetensors"), quantized)
        (template / "config.json").write_text(json.dumps(config))
        convert_ls.convert(source, template, destination, matrix_file)
        result = mx.load(str(destination / "model.safetensors"))
        bpw = sum(v.nbytes * 8 for v in result.values()) / sum(v.size for v in original.values())
        assert abs(bpw - report["actual_bpw"]) < 1e-9 and bpw <= 6.5
    print("test_quant: MLX CPU, capture, fit 4/6/8, template et conversion ok")


if __name__ == "__main__":
    stdlib_checks()
    if sys.argv[1:] == ["--mlx-cpu"]:
        mlx_checks()
    elif sys.argv[1:]:
        raise SystemExit("seul argument accepté : --mlx-cpu")
