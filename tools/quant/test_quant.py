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
    assert not dense.imatrix_blend
    blended = mixed_quant.parse_args(["--input", "src", "--output", "dst",
                                      "--target-bpw", "4.8", "--imatrix-blend"])
    assert blended.imatrix_blend
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text("{}")
        args = mixed_quant.parse_args(["--input", str(root), "--output", str(root / "out"),
                                       "--imatrix-blend"])
        try:
            mixed_quant.validate(args)
            raise AssertionError("blend sans allocation calibrée accepté")
        except SystemExit as error:
            assert "exigent --target-bpw" in str(error)
    # Des scores identiques gardent une allocation reproductible quel que soit l'ordre d'entrée.
    scores = {n: {4: 20, 5: 19, 6: 3, 8: 1} for n in ("b", "a", "c")}
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
        # Les mots doivent aussi être identiques à ceux du quantificateur MLX.
        q0, s0, b0 = mx.quantize(x, group_size=64, bits=bits)
        # Les codes MLX précèdent l'arrondi BF16 des paramètres : les relire
        # depuis la grille restaurée, sans refaire l'arrondi des poids source.
        wf = mx.dequantize(q0, s0.astype(mx.float32), b0.astype(mx.float32),
                           group_size=64, bits=bits).reshape(6, 2, 64)
        codes = opt_quant._codes(mx, wf, s0.astype(mx.float32).reshape(6, 2, 1),
                                 b0.astype(mx.float32).reshape(6, 2, 1), (1 << bits) - 1)
        assert mx.array_equal(opt_quant.pack_codes(mx, codes, x.shape, bits), q0).item(), bits
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
        # Deux groupes par ligne : sigma² doit utiliser la ligne entière,
        # et chaque expert doit garder sa propre moyenne.
        p, s, b, old, new = opt_quant.fit(x, bits=bits, weight=importance, imatrix_blend=True)
        restored = mx.dequantize(p, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)
        xf = x.astype(mx.float32)
        element_weight = importance * mx.sqrt((xf * xf).mean(-1, keepdims=True) + xf * xf)
        error = (element_weight * (restored - xf) ** 2).sum().item()
        mass = opt_quant.error_mass(x, importance, True)
        assert abs(mass - element_weight.sum().item()) < 1e-4
        assert abs(new.item() ** 2 * mass - error) < 1e-5
        assert new.item() <= old.item() * 1.001
        null = opt_quant.fit(mx.zeros((2, 128), mx.bfloat16), bits=bits,
                             weight=importance, imatrix_blend=True)
        assert null[-1].item() == null[-2].item() == 0

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
        original = dict(tree_flatten(model.parameters()))
        embed = "language_model.model.embed_tokens"
        head = "language_model.lm_head"
        source = root / "source"
        source.mkdir()
        mx.save_safetensors(str(source / "model.safetensors"), original)
        for blend in (False, True):
            assignment, report = sensitivity.scan(model, matrix, "lm_head", 6.0, 6, blend)
            assert assignment == {embed: 5, head: 6}
            assert report["imatrix_blend"] == blend and report["widths"] == [4, 5, 6, 8]
            for name, module in imatrix.language_modules(model).items():
                w = module.weight.astype(mx.float32)
                for bits in sensitivity.WIDTHS:
                    importance = None if name == embed else matrix[name]
                    p, s, b, _, _ = opt_quant.fit(module.weight, bits=bits, weight=importance,
                                                 imatrix_blend=blend and name != embed)
                    back = mx.dequantize(p, s.astype(mx.float32), b.astype(mx.float32),
                                         group_size=64, bits=bits)
                    if name == embed:
                        expected = ((back - w) ** 2).sum().item() / w.shape[0]
                    else:
                        a = importance
                        if blend:
                            a = a * mx.sqrt((w * w).mean(-1, keepdims=True) + w * w)
                        expected = (a * (back - w) ** 2).sum().item()
                    assert abs(report["scores"][name][bits] - expected) < 1e-5
            template, destination = (root / (p + str(blend)) for p in ("template", "destination"))
            template.mkdir()
            quantized = {}
            config = {"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}
            for name, bits in assignment.items():
                config["quantization"][name] = {"bits": bits, "group_size": 64, "mode": "affine"}
                values = mx.quantize(original[name + ".weight"], group_size=64, bits=bits)
                quantized.update((name + suffix, v) for suffix, v in zip((".weight", ".scales", ".biases"), values))
            mx.save_safetensors(str(template / "model.safetensors"), quantized)
            (template / "config.json").write_text(json.dumps(config))
            (template / "allocation.json").write_text(json.dumps(report))
            try:
                convert_ls.convert(source, template, destination)
                raise AssertionError("template calibré sans imatrix accepté")
            except ValueError as error:
                assert "exige --imatrix" in str(error) and not destination.exists()
            try:
                convert_ls.convert(source, template, destination, matrix_file, not blend)
                raise AssertionError("mode différent du rapport accepté")
            except ValueError as error:
                assert "mode du template" in str(error) and not destination.exists()
            convert_ls.convert(source, template, destination, matrix_file, blend)
            result = mx.load(str(destination / "model.safetensors"))
            bpw = sum(v.nbytes * 8 for v in result.values()) / sum(v.size for v in original.values())
            assert abs(bpw - report["actual_bpw"]) < 1e-9 and bpw <= 6.0
            assert json.loads((destination / "allocation.json").read_text()) == json.loads(json.dumps(report))
            for name in assignment:
                expected = opt_quant.fit(original[name + ".weight"], bits=assignment[name],
                                         weight=None if name == embed else matrix[name],
                                         imatrix_blend=blend and name != embed)
                for suffix, value in zip((".weight", ".scales", ".biases"), expected[:3]):
                    assert mx.array_equal(result[name + suffix], value).item()
        # Répéter le vocabulaire conserve son erreur moyenne ; 272 lignes
        # franchissent la frontière des blocs de 256 du scan.
        model.language_model.model.embed_tokens.weight = mx.tile(original[embed + ".weight"], (17, 1))
        _, repeated = sensitivity.scan(model, matrix, "", 4.5, 6)
        for bits in sensitivity.WIDTHS:
            assert abs(repeated["scores"][embed][bits] - report["scores"][embed][bits]) < 1e-5
    print("test_quant: MLX CPU, capture, fit 4/5/6/8 brut et blend, embedding, template et conversion ok")


if __name__ == "__main__":
    stdlib_checks()
    if sys.argv[1:] == ["--mlx-cpu"]:
        mlx_checks()
    elif sys.argv[1:]:
        raise SystemExit("seul argument accepté : --mlx-cpu")
