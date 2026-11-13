"""Re-quantification affine gs64 par moindres carrés depuis les poids BF16.

  VENV_PY -I -B convert_ls.py --src BF16 --template U4 --dst NOUVEAU_DOSSIER

Recalcule les tenseurs quantifiés language_model du modèle de base ; recopie
les autres tenseurs et fichiers du checkpoint MLX u4 fourni comme template.
La largeur (4 ou 8 bits) de chaque tenseur suit le template ; les experts
fusionnés gate_up_proj (Gemma 4) sont découpés comme mlx-lm. L'échelle peut être négative. Le meilleur groupe entre min-max et ajustement
est conservé après arrondi BF16, sans changer les kernels ni le format.
Résultat mesuré : perplexité +1,7 % contre +2,9 % par rapport au BF16,
forward mlx-lm, 8160 tokens. Cette mesure ne vaut que pour le checkpoint testé.
Sans argument : auto-test stdlib des chemins et du mappage, sans MLX.
"""

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import opt_quant  # noqa: E402


def bf16_key(base):
    if base == "language_model.lm_head":
        return "lm_head.weight"
    if not base.startswith("language_model.model."):
        raise ValueError(f"tenseur BF16 non reconnu : {base}")
    return "model.language_model." + base[len("language_model.model."):] + ".weight"


def bf16_tensor(mx, weights, base):
    """Tenseur BF16 du module quantifié base ; experts fusionnés découpés comme mlx-lm."""
    key = bf16_key(base)
    if key in weights:
        return weights[key]
    stem, _, proj = base.rpartition(".switch_glu.")
    fused = "model.language_model." + stem[len("language_model.model."):] + "."
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


def convert(src, template, dst):
    sources, shards = validate(src, template, dst)
    # Le refus d'une destination existante précède même l'import de MLX.
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    weights = {}
    for path in sources:
        weights.update(mx.load(str(path)))
    # Tous les shards sont contrôlés avant toute création ou conversion.
    for path in shards:
        for base in quantized_names(mx.load(str(path))):
            try:
                bf16_tensor(mx, weights, base)
            except ValueError as error:
                raise ValueError(f"{error} ({path.name})") from None
    dst.mkdir(parents=True, exist_ok=False)
    count, err0, err1, total = 0, 0.0, 0.0, 0
    for path in shards:
        shard = mx.load(str(path))
        names = quantized_names(shard)
        for base in names:
            x = bf16_tensor(mx, weights, base)
            bits = shard[base + ".weight"].shape[-1] * 32 // x.shape[-1]
            packed, scale, bias, rmse0, rmse1 = opt_quant.fit(x, bits=bits)
            mx.eval(packed, scale, bias, rmse0, rmse1)
            for suffix, new in ((".weight", packed), (".scales", scale), (".biases", bias)):
                old = shard[base + suffix]
                if new.shape != old.shape or new.dtype != old.dtype:
                    raise ValueError(f"format incompatible : {base}{suffix}")
                shard[base + suffix] = new
            count += 1
            err0 += rmse0.item() ** 2 * x.size
            err1 += rmse1.item() ** 2 * x.size
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
        "depuis les tenseurs BF16, format du template conservé.\n")
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
            str(src / "model-1.safetensors"): {bf16_key(base): object()},
            str(template / "model-1.safetensors"): {base + ".scales": object()},
            str(second): {"language_model.lm_head.scales": object()},
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
    opt_quant._self_check()


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("convert_ls: auto-test ok (aucune conversion)")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, required=True, help="checkpoint BF16")
    parser.add_argument("--template", type=pathlib.Path, required=True, help="checkpoint MLX affine gs64 (4/8 bits)")
    parser.add_argument("--dst", type=pathlib.Path, required=True, help="destination inexistante")
    args = parser.parse_args()
    convert(args.src, args.template, args.dst)


if __name__ == "__main__":
    main()
