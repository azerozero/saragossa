"""GPTQ forward seul, affine MLX gs64 à la largeur du template (3/4/5/6/8).

Sans argument : auto-tests stdlib. --mlx-cpu : petits tests numériques CPU.
Calibration identique à imatrix.py, Hessienne distincte pour chaque expert.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_ls
from tokenizer_sidecar import copy_tokenizer  # noqa: E402
import imatrix
import opt_quant
from gptq_capture import collect_hessians, hessian_batches


WIDTHS = (3, 4, 5, 6, 8)
SUFFIXES = (".weight", ".scales", ".biases")


def validate_options(dampening, block_size):
    if not math.isfinite(dampening) or dampening <= 0:
        raise ValueError("dampening doit être fini et strictement positif")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block-size doit être un entier positif")


def fit(x, hessian, bits=4, dampening=0.01, act_order=False, block_size=128):
    """GPTQ d'une matrice [sorties, entrées], avec groupes statiques d'origine.

    La grille LS BF16 est figée avant la permutation. U = chol(H⁻¹), supérieur :
    e_j = (w_j - q_j) / U_jj, puis W[:,j:] -= e_j U[j,j:]. Les mises à jour
    hors bloc sont regroupées en GEMM. Aucun g_idx ni permutation à l'inférence.
    Le meilleur résultat par ligne entre GPTQ et LS (uniforme/diag(H)) est
    conservé selon l'erreur complète eᵀHe, après emballage et arrondi BF16.
    """
    validate_options(dampening, block_size)
    if bits not in WIDTHS or x.ndim != 2 or x.shape[-1] % 64 or x.size == 0:
        raise ValueError("GPTQ exige une matrice non vide affine gs64, 3/4/5/6/8 bits")
    import mlx.core as mx

    rows, columns = x.shape
    h = hessian.astype(mx.float32)
    if h.shape != (columns, columns):
        raise ValueError("forme de Hessienne incompatible")
    if not mx.all(mx.isfinite(h)).item() or not mx.all(mx.isfinite(x)).item():
        raise ValueError("poids ou Hessienne non finis")
    if not mx.allclose(h, h.T, rtol=1e-4, atol=1e-6).item():
        raise ValueError("Hessienne non symétrique")
    diagonal = mx.diag(h)
    if mx.any(diagonal < 0).item() or not mx.any(diagonal > 0).item():
        raise ValueError("Hessienne sans énergie positive ou diagonale négative")
    original = x.astype(mx.float32)
    baseline = opt_quant.fit(x, bits=bits, weight=diagonal)[:3]

    def losses(quantized):
        p, s, b = quantized
        restored = mx.dequantize(p, s.astype(mx.float32), b.astype(mx.float32),
                                group_size=64, bits=bits)
        delta = restored - original
        return mx.maximum(((delta @ h) * delta).sum(-1), 0)

    def choose(candidate, reference):
        candidate_loss, reference_loss = losses(candidate), losses(reference)
        use = candidate_loss < reference_loss
        result = tuple(mx.where(use[:, None], a, b) for a, b in zip(candidate, reference))
        mx.eval(*result)
        return result

    # Même garde contre convert_ls sans imatrix ; la norme de comparaison reste H.
    baseline = choose(baseline, opt_quant.fit(x, bits=bits)[:3])
    _, scales, biases = baseline
    sf, bf = scales.astype(mx.float32), biases.astype(mx.float32)
    perm = mx.argsort(-diagonal) if act_order else mx.arange(columns)
    order = perm.tolist()
    hp = h[perm][:, perm]
    # Les canaux jamais activés ont une ligne/colonne nulle dans XᵀX.
    dead = mx.diag(hp) == 0
    hp = hp + mx.diag(dead.astype(mx.float32))
    hp = hp + mx.eye(columns) * (dampening * diagonal.mean())
    with mx.stream(mx.cpu):
        try:
            lower = mx.linalg.cholesky(hp)
            inverse = mx.linalg.cholesky_inv(lower)
            upper = mx.linalg.cholesky(inverse, upper=True)
            mx.eval(upper)
        except (ValueError, RuntimeError) as error:
            raise ValueError("factorisation de H impossible ; augmenter --dampening") from error
    if not mx.all(mx.isfinite(upper)).item() or not mx.all(mx.diag(upper) > 0).item():
        raise ValueError("factorisation de H non finie ; augmenter --dampening")
    del lower, inverse, hp
    work = original[:, perm]
    codes = mx.zeros((rows, columns), mx.uint32)
    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        local = work[:, start:end]
        errors = mx.zeros_like(local)
        for column in range(start, end):
            j = column - start
            group = order[column] // 64
            s, b = sf[:, group], bf[:, group]
            w = local[:, j]
            q = opt_quant._codes(mx, w, s, b, (1 << bits) - 1)
            codes[:, column] = q.astype(mx.uint32)
            error = (w - (s * q + b)) / upper[column, column]
            local[:, j:] = local[:, j:] - error[:, None] * upper[column, column:end]
            errors[:, j] = error
            mx.eval(local, errors, codes)
        if end < columns:
            work[:, end:] = work[:, end:] - errors @ upper[start:end, end:]
            mx.eval(work)
    # Rétablir l'ordre physique des colonnes avant l'emballage MLX existant.
    codes = codes[:, mx.argsort(perm)]
    packed = opt_quant.pack_codes(mx, codes, x.shape, bits)
    candidate = (packed, scales, biases)
    raw_loss, ls_loss = losses(candidate), losses(baseline)
    result = choose(candidate, baseline)
    final_loss = losses(result)
    mx.eval(raw_loss, ls_loss, final_loss)
    report = {"ls_error": ls_loss.sum().item(), "gptq_error": raw_loss.sum().item(),
              "output_error": final_loss.sum().item(),
              "fallback_rows": int((raw_loss > ls_loss).sum().item())}
    return (*result, report)


def convert(src, template, dst, calib, samples=128, length=512, chat=False,
            dampening=0.01, act_order=False, block_size=128, hessian_memory_mib=2048):
    _, shards = convert_ls.validate(src, template, dst)
    validate_options(dampening, block_size)
    if samples <= 0 or length <= 0:
        raise ValueError("samples et length doivent être positifs")
    if not math.isfinite(hessian_memory_mib) or hessian_memory_mib <= 0:
        raise ValueError("hessian-memory-mib doit être fini et positif")
    text = calib.read_text(encoding="utf-8")
    imatrix.reject_eval_text(text)
    config = json.loads((template / "config.json").read_text())
    import mlx.core as mx

    model, tokenizer = imatrix.load_bf16(src)
    chunks = list(imatrix.calibration_chunks(tokenizer, text, samples, length, chat))
    available = imatrix.language_modules(model)
    merged = {}
    for path in shards:
        merged.update(mx.load(str(path)))
    names = convert_ls.quantized_names(merged)
    if not names:
        raise ValueError("template sans module language_model quantifié")
    modules, widths, embeddings = {}, {}, set()
    for name in names:
        if name not in available:
            raise ValueError(f"module BF16 absent : {name}")
        module = available[name]
        bits = convert_ls.quantization_spec(config, name)
        convert_ls.validate_shapes(merged, name, module.weight.shape, bits)
        for suffix, dtype in zip(SUFFIXES, (mx.uint32, mx.bfloat16, mx.bfloat16)):
            if merged[name + suffix].dtype != dtype:
                raise ValueError(f"dtype du template incompatible : {name}{suffix}")
        widths[name] = bits
        if imatrix.is_embedding(module):
            embeddings.add(name)
        else:
            modules[name] = module
    del merged
    batches = hessian_batches(modules, hessian_memory_mib)
    report = {"method": "gptq", "statistic": "mean_input_outer_product_per_expert",
              "calib_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "samples": samples, "length": length, "chat": chat,
              "dampening": dampening, "act_order": act_order, "static_groups": True,
              "block_size": block_size, "hessian_memory_mib": hessian_memory_mib,
              "calibration_passes": len(batches), "modules": {}}
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Une calibration incomplète ne laisse jamais un checkpoint final partiel.
    with tempfile.TemporaryDirectory(prefix=".gptq-", dir=dst.parent) as tmp:
        scratch = Path(tmp)
        output = scratch / "output"
        output.mkdir()
        fitted = {}

        def save_piece(key, values):
            path = scratch / f"fit-{len(fitted)}.safetensors"
            mx.save_safetensors(str(path), dict(zip(SUFFIXES, values)))
            fitted[key] = path

        for i, targets in enumerate(batches):
            print(f"Hessiennes {i + 1}/{len(batches)} : {len(targets)} matrices", flush=True)
            matrices, counts = collect_hessians(model, modules, chunks, targets)
            for name, expert in targets:
                key = (name, expert)
                x = modules[name].weight
                if expert is not None:
                    x = x[expert]
                p, s, b, metrics = fit(x, matrices.pop(key), widths[name], dampening,
                                       act_order, block_size)
                save_piece(key, (p, s, b))
                info = report["modules"].setdefault(name, {"bits": widths[name], "experts": {}})
                info["experts"][str(expert) if expert is not None else "dense"] = {
                    "rows_observed": counts[key], **metrics}
                print(f"  quantifié {name} expert={expert}, erreur {metrics['output_error']:.6g}", flush=True)
                del p, s, b
            del matrices
            mx.clear_cache()
        for name in sorted(embeddings):
            values = opt_quant.fit(available[name].weight, bits=widths[name])[:3]
            save_piece((name, None), values)
            report["modules"][name] = {"bits": widths[name], "method": "ls_embedding"}
        # Fusion par module, indépendante de la répartition poids/échelles/biais en shards.
        files = {}
        for name in names:
            shape = available[name].weight.shape
            keys = [(name, e) for e in range(shape[0])] if len(shape) == 3 else [(name, None)]
            if len(shape) == 2:
                files[name] = fitted[keys[0]]
                continue
            parts = [mx.load(str(fitted[key])) for key in keys]
            values = {suffix: mx.stack([part[suffix] for part in parts]) for suffix in SUFFIXES}
            path = scratch / f"module-{len(files)}.safetensors"
            mx.save_safetensors(str(path), values)
            files[name] = path
            del parts, values
        # Libérer les poids BF16 avant la réécriture des shards du template.
        del model, available, modules
        mx.clear_cache()
        for path in shards:
            shard = mx.load(str(path))
            for name in names:
                present = [suffix for suffix in SUFFIXES if name + suffix in shard]
                if not present:
                    continue
                values = mx.load(str(files[name]))
                for suffix in present:
                    old, new = shard[name + suffix], values[suffix]
                    if old.shape != new.shape or old.dtype != new.dtype:
                        raise ValueError(f"format incompatible : {name}{suffix}")
                    shard[name + suffix] = new
            mx.save_safetensors(str(output / path.name), shard, metadata={"format": "mlx"})
            print(f"écrit {path.name}", flush=True)
        for path in template.iterdir():
            if path.is_file() and path not in shards and path.name not in ("README.md", "gptq.json", "tokenizer.json"):
                shutil.copy2(path, output / path.name)
        # Même règle que convert_ls : le tokenizer vient de la source BF16.
        copy_tokenizer(src, output)
        (output / "gptq.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "README.md").write_text(
            "Checkpoint MLX affine gs64 GPTQ, groupes statiques, largeur du template conservée.\n"
            "Calibration BF16 forward seul ; Hessienne séparée par expert.\n"
            "Repli LS par ligne selon l'erreur complète sur la calibration ; embeddings LS.\n"
            "allocation.json décrit l'allocation LS du template ; gptq.json décrit cette conversion.\n")
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(f"destination existante refusée : {dst}")
        output.rename(dst)
    print(f"fini : {len(names)} modules, {dst}", flush=True)


def main():
    if not sys.argv[1:] or sys.argv[1:] == ["--mlx-cpu"]:
        from test_gptq import stdlib_checks, mlx_checks
        stdlib_checks()
        if sys.argv[1:]:
            mlx_checks()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for option, help_text in (("src", "checkpoint BF16"), ("template", "checkpoint MLX affine gs64"),
                              ("dst", "destination inexistante"), ("calib", "corpus indépendant")):
        parser.add_argument("--" + option, type=Path, required=True, help=help_text)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--dampening", type=float, default=0.01, help="fraction de moyenne(diag(H)), > 0")
    parser.add_argument("--act-order", action="store_true", help="ordre décroissant de diag(H), groupes statiques")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--hessian-memory-mib", type=float, default=2048,
                        help="budget des accumulateurs H ; hors poids et factorisation")
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
