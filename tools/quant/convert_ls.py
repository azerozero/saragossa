"""Re-quantification affine gs64 par moindres carrés depuis les poids BF16.

  VENV_PY -I -B convert_ls.py --src BF16 --template U4 --dst NOUVEAU_DOSSIER

Recalcule les tenseurs quantifiés language_model du modèle de base ; recopie
les autres tenseurs et fichiers du checkpoint MLX u4 fourni comme template.
La largeur (4, 6 ou 8 bits) de chaque tenseur suit le template ; les experts
fusionnés gate_up_proj (Gemma 4) sont découpés comme mlx-lm.
L'échelle peut être négative. Le meilleur groupe entre min-max et ajustement
est conservé après arrondi BF16, sans changer les kernels ni le format.
Résultat mesuré : perplexité +1,7 % contre +2,9 % par rapport au BF16,
forward mlx-lm, 8160 tokens. Cette mesure ne vaut que pour le checkpoint testé.
Sans argument : auto-test stdlib des chemins et du mappage, sans MLX.
"""

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import opt_quant  # noqa: E402
from imatrix import importance_for  # noqa: E402


def bf16_key(base):
    if base == "language_model.lm_head":
        return "lm_head.weight"
    if not base.startswith("language_model.model."):
        raise ValueError(f"tenseur BF16 non reconnu : {base}")
    return "model.language_model." + base[len("language_model.model."):] + ".weight"


def bf16_tensor(mx, weights, base):
    """Tenseur BF16 du module quantifié base ; experts fusionnés découpés comme mlx-lm."""
    key = bf16_key(base)
    if base + ".weight" in weights:
        return weights[base + ".weight"]
    if key in weights:
        return weights[key]
    marker = ".switch_mlp." if ".switch_mlp." in base else ".switch_glu."
    stem, _, proj = base.rpartition(marker)
    fused = "model.language_model." + stem[len("language_model.model."):]
    fused += ".experts." if marker == ".switch_mlp." else "."
    if proj == "down_proj" and fused + "down_proj" in weights:
        return weights[fused + "down_proj"]
    if proj in ("gate_proj", "up_proj") and fused + "gate_up_proj" in weights:
        gate, up = mx.split(weights[fused + "gate_up_proj"], 2, axis=-2)
        return gate if proj == "gate_proj" else up
    raise ValueError(f"clé BF16 absente : {key}")


def validate(src, template, dst):
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"destination existante refusée : {dst}")
    sources = sorted(src.glob("model*.safetensors"))
    shards = sorted(template.glob("model*.safetensors"))
    if not sources or not shards:
        raise ValueError("src et template doivent contenir des shards model*.safetensors")
    if not (template / "config.json").is_file():
        raise ValueError("template : config.json absent")
    return sources, shards


def quantized_names(shard):
    return sorted(k.removesuffix(".scales") for k in shard
                  if k.endswith(".scales") and k.startswith("language_model."))


def quantization_spec(config, base):
    spec = dict(config.get("quantization", {}))
    override = spec.get(base, {})
    if not isinstance(override, dict):
        raise ValueError(f"configuration quantifiée invalide : {base}")
    spec.update(override)
    if (spec.get("bits") not in (4, 6, 8) or spec.get("group_size") != 64
            or spec.get("mode", "affine") != "affine"):
        raise ValueError(f"format non pris en charge : {base} (affine gs64 4/6/8 requis)")
    return spec["bits"]


def validate_shapes(shard, base, shape, bits):
    if not shape or shape[-1] % 64:
        raise ValueError(f"forme BF16 incompatible : {base}")
    for suffix, columns in ((".weight", shape[-1] * bits // 32),
                            (".scales", shape[-1] // 64), (".biases", shape[-1] // 64)):
        tensor = shard.get(base + suffix)
        if tensor is None or tuple(tensor.shape) != (*shape[:-1], columns):
            raise ValueError(f"forme du template incompatible : {base}{suffix}")


def convert(src, template, dst, imatrix=None):
    sources, shards = validate(src, template, dst)
    config = json.loads((template / "config.json").read_text())
    # Le refus d'une destination existante précède même l'import de MLX.
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    matrix = mx.load(str(imatrix)) if imatrix is not None else None
    weights = {}
    for path in sources:
        weights.update(mx.load(str(path)))
    # Tous les shards sont contrôlés avant toute création ou conversion.
    for path in shards:
        shard = mx.load(str(path))
        for base in quantized_names(shard):
            try:
                x = bf16_tensor(mx, weights, base)
                bits = quantization_spec(config, base)
                validate_shapes(shard, base, x.shape, bits)
                importance_for(matrix, base, x.shape[-1])
            except ValueError as error:
                raise ValueError(f"{error} ({path.name})") from None
    dst.mkdir(parents=True, exist_ok=False)
    count, err0, err1, total = 0, 0.0, 0.0, 0
    for path in shards:
        shard = mx.load(str(path))
        names = quantized_names(shard)
        for base in names:
            x = bf16_tensor(mx, weights, base)
            bits = quantization_spec(config, base)
            weight = importance_for(matrix, base, x.shape[-1])
            packed, scale, bias, rmse0, rmse1 = opt_quant.fit(x, bits=bits, weight=weight)
            mx.eval(packed, scale, bias, rmse0, rmse1)
            for suffix, new in ((".weight", packed), (".scales", scale), (".biases", bias)):
                old = shard[base + suffix]
                if new.shape != old.shape or new.dtype != old.dtype:
                    raise ValueError(f"format incompatible : {base}{suffix}")
                shard[base + suffix] = new
            count += 1
            mass = sum(weight.tolist()) * (x.size // x.shape[-1]) if weight is not None else x.size
            err0 += rmse0.item() ** 2 * mass
            err1 += rmse1.item() ** 2 * mass
        out = dst / path.name
        mx.save_safetensors(str(out), shard, metadata={"format": "mlx"})
        total += sum(v.nbytes for v in shard.values())
        ratio = err1 / err0 if err0 else 0
        print("écrit", out.name, count, f"MSE ls/minmax {ratio:.4f}", flush=True)
    for path in template.iterdir():
        if path.is_file() and path not in shards and path.name != "README.md":
            shutil.copy2(path, dst / path.name)
    (dst / "README.md").write_text(
        "Checkpoint MLX affine gs64 : échelles et biais ajustés aux moindres carrés "
        "depuis les tenseurs BF16, format du template conservé. "
        + ("Norme pondérée par E[x²].\n" if matrix is not None else "Norme uniforme.\n"))
    print("fini", count, total / 2**30, "Gio", flush=True)


def _self_check():
    import tempfile
    from types import ModuleType
    from unittest.mock import Mock, patch

    assert bf16_key("language_model.lm_head") == "lm_head.weight"
    assert bf16_key("language_model.model.layers.3.mlp.up_proj") == (
        "model.language_model.layers.3.mlp.up_proj.weight")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for name in ("src", "template", "existing"):
            (root / name).mkdir()
        src, template, dst = (root / name for name in ("src", "template", "new"))
        for path in (src / "model-1.safetensors", template / "model-1.safetensors",
                     template / "config.json"):
            path.write_bytes(b"")
        (template / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "group_size": 64}}))
        assert validate(src, template, dst) == ([src / "model-1.safetensors"],
                                               [template / "model-1.safetensors"])
        for name in ("existing", "template", "src"):
            try:
                convert(src, template, root / name)
                raise AssertionError("destination existante acceptée")
            except FileExistsError:
                pass
        assert not dst.exists() and "mlx.core" not in sys.modules
        # La clé absente est dans le dernier shard : aucun début de sortie,
        # même si le premier shard et sa clé BF16 sont valides.
        second = template / "model-2.safetensors"
        second.write_bytes(b"")
        base = "language_model.model.layers.0.mlp.up_proj"
        mx = ModuleType("mlx.core")
        mx.cpu = "cpu"
        mx.set_default_device = Mock()
        mx.load = Mock(side_effect=lambda path: {
            str(src / "model-1.safetensors"): {bf16_key(base): Mock(shape=(64, 64))},
            str(template / "model-1.safetensors"): {
                base + ".weight": Mock(shape=(64, 8)),
                base + ".scales": Mock(shape=(64, 1)),
                base + ".biases": Mock(shape=(64, 1))},
            str(second): {"language_model.lm_head.scales": object()},
            str(root / "imatrix.safetensors"): {},
        }[path])
        mlx = ModuleType("mlx")
        mlx.core = mx
        with (patch.dict(sys.modules, {"mlx": mlx, "mlx.core": mx}),
              patch.object(opt_quant, "fit") as fit):
            try:
                convert(src, template, dst)
                raise AssertionError("clé BF16 absente acceptée")
            except ValueError as error:
                assert "lm_head.weight" in str(error)
            assert not dst.exists()
            fit.assert_not_called()
            try:
                convert(src, template, dst, root / "imatrix.safetensors")
                raise AssertionError("imatrix incomplète acceptée")
            except ValueError as error:
                assert "imatrix absente" in str(error)
            assert not dst.exists()
            fit.assert_not_called()
    # Experts fusionnés Gemma 4 : moitié haute de gate_up_proj = gate, basse = up.
    mx = ModuleType("mlx.core")
    mx.split = Mock(return_value=("G", "U"))
    stem = "language_model.model.layers.2.experts.switch_glu."
    fused = {"model.language_model.layers.2.experts.gate_up_proj": "GU",
             "model.language_model.layers.2.experts.down_proj": "D"}
    assert [bf16_tensor(mx, fused, stem + p) for p in ("gate_proj", "up_proj", "down_proj")] == [
        "G", "U", "D"]
    mx.split.assert_called_with("GU", 2, axis=-2)
    # Qwen MoE place les tenseurs fusionnés sous mlp.experts, puis crée switch_mlp.
    stem = "language_model.model.layers.2.mlp.switch_mlp."
    fused_qwen = {"model.language_model.layers.2.mlp.experts.gate_up_proj": "GU",
                  "model.language_model.layers.2.mlp.experts.down_proj": "D"}
    assert [bf16_tensor(mx, fused_qwen, stem + p) for p in ("gate_proj", "up_proj", "down_proj")] == [
        "G", "U", "D"]
    try:
        bf16_tensor(mx, fused, "language_model.model.layers.2.mlp.up_proj")
        raise AssertionError("clé absente acceptée")
    except ValueError:
        pass
    opt_quant._self_check()
    config = {"quantization": {"bits": 4, "group_size": 64, base: {"bits": 6}}}
    assert quantization_spec(config, base) == 6
    for bits in (4, 6, 8):
        shard = {base + suffix: Mock(shape=(2, columns)) for suffix, columns in (
            (".weight", 128 * bits // 32), (".scales", 2), (".biases", 2))}
        validate_shapes(shard, base, (2, 128), bits)
        try:
            validate_shapes(shard, base, (2, 128), 8 if bits != 8 else 4)
            raise AssertionError("largeur du template incohérente acceptée")
        except ValueError:
            pass
    for field, value in (("bits", 5), ("group_size", 32), ("mode", "mxfp4")):
        bad = {"quantization": {"bits": 4, "group_size": 64, field: value}}
        try:
            quantization_spec(bad, base)
            raise AssertionError("format invalide accepté")
        except ValueError:
            pass


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("convert_ls: auto-test ok (aucune conversion)")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="checkpoint BF16")
    parser.add_argument("--template", type=pathlib.Path, required=True, help="checkpoint MLX affine gs64 (4/6/8 bits)")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="destination inexistante")
    parser.add_argument("--imatrix", type=pathlib.Path, help="importance E[x²] par colonne ; linéaires manquants refusés")
    args = parser.parse_args()
    convert(args.src, args.template, args.dst, args.imatrix)


if __name__ == "__main__":
    main()
