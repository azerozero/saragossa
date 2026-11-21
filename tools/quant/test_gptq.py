"""Auto-tests sans poids externes : stdlib, puis --mlx-cpu facultatif."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gptq
import imatrix
import opt_quant
from gptq_capture import collect_hessians, hessian_batches, routed_rows


def raises(kind, fn, contains=""):
    try:
        fn()
    except kind as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError(f"{kind.__name__} attendu")


def stdlib_checks():
    for damp in (0, -1, math.inf, math.nan):
        raises(ValueError, lambda: gptq.validate_options(damp, 128))
    for block in (0, -1, 1.5):
        raises(ValueError, lambda: gptq.validate_options(0.01, block))
    modules = {"dense": SimpleNamespace(weight=SimpleNamespace(shape=(4, 64))),
               "experts": SimpleNamespace(weight=SimpleNamespace(shape=(3, 4, 64)))}
    assert hessian_batches(modules, 1 / 32) == [
        [("dense", None), ("experts", 0)], [("experts", 1), ("experts", 2)]]
    for budget in (0, -1, math.inf, math.nan, 1 / 128):
        raises(ValueError, lambda: hessian_batches(modules, budget))

    class Switch:
        def __call__(self, x, indices, sorted_indices=False):
            return x + len(indices)
    module = Switch()
    original = Switch.__call__
    seen = []
    def observe(name, x, *args, **kwargs):
        seen.append((name, x, args, kwargs))
    with imatrix.capture_calls({"experts": module}, observe, with_call_args=True):
        assert module(3, [0, 1], sorted_indices=True) == 5
        assert module(4, indices=[1]) == 5
    assert Switch.__call__ is original
    assert seen == [("experts", 3, ([0, 1],), {"sorted_indices": True}),
                    ("experts", 4, (), {"indices": [1]})]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, template, dst = (root / name for name in ("src", "template", "dst"))
        for directory in (src, template):
            directory.mkdir()
            (directory / "model.safetensors").touch()
            (directory / "tokenizer.json").write_text('{"fixture": true}')
        (template / "config.json").write_text("{}")
        corpus = root / "corpus.txt"
        corpus.write_text(imatrix.PROMPTS[0])
        raises(ValueError, lambda: gptq.convert(src, template, dst, corpus), "PROMPT")
        assert not dst.exists()
        dst.mkdir()
        raises(FileExistsError, lambda: gptq.convert(src, template, dst, corpus))
    assert "mlx.core" not in sys.modules
    print("gptq: auto-tests stdlib ok")


def mlx_checks():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import SwitchGLU

    mx.set_default_device(mx.cpu)
    mx.random.seed(37)
    latent = mx.random.normal((256, 8))
    activations = latent @ mx.random.normal((8, 128)) + 0.15 * mx.random.normal((256, 128))
    # Canal mort et corrélations entre groupes ; act-order doit traverser la frontière gs64.
    activations[:, 7] = 0
    h = activations.T @ activations / activations.shape[0]
    w = (mx.random.normal((6, 128)) * 0.2).astype(mx.bfloat16)
    improvements = []
    for bits in gptq.WIDTHS:
        for act_order in (False, True):
            p, s, b, report = gptq.fit(w, h, bits, act_order=act_order, block_size=48)
            assert p.shape == (6, 128 * bits // 32) and p.dtype == mx.uint32
            assert s.shape == b.shape == (6, 2) and s.dtype == b.dtype == mx.bfloat16
            back = mx.dequantize(p, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)
            actual = ((activations @ (back - w.astype(mx.float32)).T) ** 2).sum().item() / 256
            assert abs(actual - report["output_error"]) < max(1e-5, actual * 2e-4)
            for importance in (None, mx.diag(h)):
                ls = opt_quant.fit(w, bits=bits, weight=importance)[:3]
                restored = mx.dequantize(ls[0], ls[1].astype(mx.float32), ls[2].astype(mx.float32),
                                        group_size=64, bits=bits)
                expected = ((activations @ (restored - w.astype(mx.float32)).T) ** 2).sum().item() / 256
                assert actual <= expected * 1.0001, (bits, act_order, actual, expected)
            improvements.append(report["gptq_error"] < report["ls_error"] * 0.9)
            print(f"  {bits} bits act_order={act_order}: {report}")
    # Le test doit prouver une amélioration GPTQ réelle, pas seulement le repli LS.
    assert all(improvements)
    # Le découpage en blocs n'affecte pas la quantification ; identité des codes.
    a = gptq.fit(w, h, act_order=True, block_size=128)
    b = gptq.fit(w, h, act_order=True, block_size=32)
    assert all(mx.array_equal(x, y).item() for x, y in zip(a[:3], b[:3]))
    for bad in (mx.zeros((128, 128)), mx.ones((64, 64)), h * math.nan, -mx.eye(128)):
        raises(ValueError, lambda: gptq.fit(w, bad))
    asymmetric = mx.eye(128)
    asymmetric[0, 1] = 1
    raises(ValueError, lambda: gptq.fit(w, asymmetric), "symétrique")
    constant = gptq.fit(mx.full((2, 128), 2, mx.bfloat16), mx.eye(128))
    assert constant[-1]["output_error"] == 0

    # Diffusion des entrées gate/up puis entrées down propres à chaque route.
    x = mx.arange(4 * 64).reshape(1, 4, 1, 1, 64).astype(mx.float32)
    indices = mx.array([[[0, 1], [1, 2], [2, 0], [0, 2]]])
    values, positions = routed_rows(mx, x, indices, 3)
    assert positions == {0: [0, 5, 6], 1: [1, 2], 2: [3, 4, 7]}
    assert mx.array_equal(values[:, 0], mx.repeat(x.reshape(4, 64), 2, axis=0)).item()
    down = mx.arange(8 * 64).reshape(1, 4, 2, 1, 64).astype(mx.float32)
    values, _ = routed_rows(mx, down, indices, 3)
    assert mx.array_equal(values, down.reshape(8, 1, 64)).item()
    raises(ValueError, lambda: routed_rows(mx, x, None, 3))
    raises(ValueError, lambda: routed_rows(mx, x, indices.astype(mx.float32), 3))
    raises(ValueError, lambda: routed_rows(mx, x, indices, 2))
    raises(ValueError, lambda: routed_rows(mx, x, mx.zeros((5, 2), mx.int32), 3))

    # Vrai SwitchGLU mlx-lm, avant ET après son tri des tokens par expert.
    switch = SwitchGLU(64, 64, 3)
    for tokens in (8, 40):
        inputs = mx.random.normal((1, tokens, 64)).astype(mx.bfloat16)
        routes = mx.array([[[i % 3, (i + 1) % 3] for i in range(tokens)]])
        class Routed:
            def __call__(self, ids):
                return switch(inputs, routes)
        modules = {"gate": switch.gate_proj, "down": switch.down_proj}
        targets = [(name, e) for name in modules for e in range(3)]
        matrices, counts = collect_hessians(Routed(), modules, [[0]], targets)
        # Oracle indépendant, un expert à la fois sur les tokens réellement routés.
        for e in range(3):
            selected = [i for i in range(tokens) if e in routes[0, i].tolist()]
            inp = inputs[0, mx.array(selected)].astype(mx.float32)
            assert counts[("gate", e)] == len(selected)
            assert mx.allclose(matrices[("gate", e)], inp.T @ inp / len(selected), atol=2e-5).item()
            # Même dtype et même activation que SwitchGLU, sans gather_mm.
            inp_bf = inp.astype(mx.bfloat16)
            up = inp_bf @ switch.up_proj.weight[e].T
            gate = inp_bf @ switch.gate_proj.weight[e].T
            hidden = switch.activation(up, gate).astype(mx.float32)
            assert mx.allclose(matrices[("down", e)], hidden.T @ hidden / len(selected),
                               atol=2e-5, rtol=2e-4).item()
        assert not mx.allclose(matrices[("gate", 0)], matrices[("gate", 1)]).item()
        routes = mx.zeros_like(routes)
        raises(ValueError, lambda: collect_hessians(Routed(), modules, [[0]], targets), "non observés")

    gated_delta_checks(mx)
    checkpoint_checks(mx, nn, tree_flatten, SwitchGLU)
    print("gptq: MLX CPU, GPTQ/LS 3/4/5/6/8, act-order, MoE, GatedDeltaNet et checkpoint ok")


def gated_delta_checks(mx):
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    layer = GatedDeltaNet(TextModelArgs(
        hidden_size=64, linear_num_value_heads=1, linear_num_key_heads=1,
        linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=3))
    layer.eval()
    inputs = mx.random.normal((1, 4, 64))
    class Recurrent:
        def __call__(self, ids):
            return layer(inputs)
    modules = {name: module for name, module in layer.named_modules()
               if name.startswith("in_proj_") or name == "out_proj"}
    targets = [(name, None) for name in modules]
    with (patch.object(mx, "grad", side_effect=AssertionError("rétropropagation interdite")),
          patch.object(mx, "value_and_grad", side_effect=AssertionError("rétropropagation interdite"))):
        single, _ = collect_hessians(Recurrent(), modules, [[0]], targets)
        repeated, counts = collect_hessians(Recurrent(), modules, [[0], [0]], targets)
        for key in targets:
            assert counts[key] == 8 and mx.array_equal(single[key], repeated[key]).item()
        result = gptq.fit(layer.out_proj.weight.astype(mx.bfloat16), repeated[("out_proj", None)])
        assert result[-1]["output_error"] <= result[-1]["ls_error"]


def checkpoint_checks(mx, nn, tree_flatten, SwitchGLU):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(c) % 16 for c in text]
        def decode(self, ids):
            return "".join(chr(97 + i) for i in ids)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            lang = self.language_model
            lang.embed_tokens = nn.Embedding(16, 64)
            lang.switch_mlp = SwitchGLU(64, 64, 2)
            lang.lm_head = nn.Linear(64, 16, bias=False)
            for _, module in self.named_modules():
                if hasattr(module, "weight"):
                    module.weight = module.weight.astype(mx.bfloat16)
        def __call__(self, ids):
            lang = self.language_model
            x = lang.embed_tokens(ids)
            routes = mx.stack([ids % 2, (ids + 1) % 2], axis=-1)
            x = lang.switch_mlp(x, routes).sum(-2)
            return lang.lm_head(x)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, template, dst = (root / n for n in ("src", "template", "dst"))
        src.mkdir()
        template.mkdir()
        model = Tiny()
        original = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(src / "model.safetensors"), original)
        (src / "config.json").write_text("{}")
        (src / "tokenizer.json").write_text('{"fixture": true}')
        corpus = root / "corpus.txt"
        corpus.write_text("abcdefghijklmnop")
        config = {"quantization": {"bits": 3, "group_size": 64}}
        template_weights = dict(original)
        names = list(imatrix.language_modules(model))
        for name, bits in zip(names, gptq.WIDTHS):
            p, s, b = mx.quantize(original[name + ".weight"], group_size=64, bits=bits)
            template_weights.update({name + suffix: value for suffix, value in zip(gptq.SUFFIXES, (p, s, b))})
            config["quantization"][name] = {"bits": bits, "group_size": 64}
        # Les trois tenseurs d'un module n'ont pas besoin d'être dans le même shard.
        split = [{k: v for k, v in template_weights.items() if k.endswith(suffix)} for suffix in gptq.SUFFIXES]
        weight_map = {}
        for i, tensors in enumerate(split):
            shard = f"model-{i + 1}.safetensors"
            mx.save_safetensors(str(template / shard), tensors)
            weight_map.update({k: shard for k in tensors})
        (template / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (template / "config.json").write_text(json.dumps(config))
        (template / "allocation.json").write_text('{"base_bits": 3, "target_bpw": 4.2}')
        (template / "tokenizer.json").write_text('{"fixture": true}')
        gather_mm = mx.gather_mm
        def cpu_gather(x, weights, **kwargs):
            # Le kernel CPU GatherMM de MLX 0.32 ne prend que f32 ; conserver
            # les poids BF16 du fixture et arrondir la sortie comme le forward.
            return gather_mm(x.astype(mx.float32), weights.astype(mx.float32),
                             **kwargs).astype(x.dtype)
        with (patch.object(imatrix, "load_bf16", return_value=(model, Tokenizer())),
              patch.object(mx, "gather_mm", cpu_gather)):
            gptq.convert(src, template, dst, corpus, samples=2, length=8,
                         act_order=True, hessian_memory_mib=1 / 32)
        output = {}
        for path in dst.glob("model-*.safetensors"):
            output.update(mx.load(str(path)))
        assert set(output) == set(template_weights)
        for key, value in output.items():
            assert value.shape == template_weights[key].shape and value.dtype == template_weights[key].dtype
        for file in ("config.json", "allocation.json", "model.safetensors.index.json", "tokenizer.json"):
            assert (dst / file).read_bytes() == (template / file).read_bytes()
        # Modules MLX standards rechargés avec les tenseurs du checkpoint écrit.
        restored = Tiny()
        nn.quantize(restored, group_size=64, bits=3,
                    class_predicate=lambda name, module: config["quantization"].get(name, False))
        restored.load_weights(list(output.items()), strict=True)
        y = restored(mx.array([[1, 2, 3]]))
        assert y.shape == (1, 3, 16) and mx.all(mx.isfinite(y)).item()
        report = json.loads((dst / "gptq.json").read_text())
        assert report["calibration_passes"] == 4
        for name, info in report["modules"].items():
            if "switch_mlp" in name:
                assert set(info["experts"]) == {"0", "1"}
        # Échec pendant la capture : restauration des hooks, nettoyage, dst absent.
        failed = root / "failed"
        original_call = type(model.language_model.lm_head).__call__
        with (patch.object(imatrix, "load_bf16", return_value=(model, Tokenizer())),
              patch.object(gptq, "collect_hessians", side_effect=ValueError("expert non observé"))):
            raises(ValueError, lambda: gptq.convert(src, template, failed, corpus, samples=2, length=8))
        assert not failed.exists() and not list(root.glob(".gptq-*"))
        assert type(model.language_model.lm_head).__call__ is original_call


if __name__ == "__main__":
    stdlib_checks()
    if sys.argv[1:] == ["--mlx-cpu"]:
        mlx_checks()
    elif sys.argv[1:]:
        raise SystemExit("seul argument accepté : --mlx-cpu")
