"""Capture E[x²] par canal d'entrée des linéaires de langage via mlx-lm BF16.

Sans argument : auto-tests stdlib. --chat enveloppe un extrait sur deux dans
le chat template ; les autres restent du texte brut. Aucun prompt d'évaluation.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_kl import PROMPTS
from opt_quant import validate_weight


def reserved_prompts():
    """PROMPTS de chat_kl.py et ensembles figés sélection/validation (sets/*.json)."""
    reserved = list(PROMPTS)
    for path in sorted((Path(__file__).resolve().parent / "sets").glob("*.json")):
        reserved += [p["prompt"] for p in json.loads(path.read_text(encoding="utf-8"))["prompts"]]
    return reserved


def reject_eval_text(text):
    """Refuse les prompts réservés, y compris casse, espaces et ponctuation modifiés."""
    def normalize(value):
        return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())
    normalized = normalize(text)
    if any(normalize(prompt) in normalized for prompt in reserved_prompts()):
        raise ValueError("calibration contaminée par un PROMPT réservé (chat_kl.py ou sets/)")


def calibration_chunks(tokenizer, text, samples=128, length=512, chat=False):
    if samples <= 0 or length <= 0:
        raise ValueError("samples et length doivent être positifs")
    reject_eval_text(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < samples * length:
        raise ValueError(f"corpus trop court : {len(ids)} tokens, {samples * length} requis")
    # Extraits disjoints répartis dans tout le corpus, sans tirage ni répétition.
    for i in range(samples):
        start = i * (len(ids) - length) // max(samples - 1, 1)
        chunk = ids[start:start + length]
        excerpt = tokenizer.decode(chunk)
        reject_eval_text(excerpt)
        if chat and i % 2 == 1:
            chunk = tokenizer.apply_chat_template(
                [{"role": "user", "content": excerpt}], tokenize=True, add_generation_prompt=True)
            reject_eval_text(tokenizer.decode(chunk))
        yield chunk


def language_modules(model):
    """Modules quantifiables gs64 ; noms issus de named_modules, sans renommage."""
    result = {}
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if (name.startswith("language_model.") and w is not None and w.ndim in (2, 3)
                and (hasattr(module, "to_quantized") or hasattr(module, "scales"))):
            columns = (module.scales.shape[-1] * module.group_size
                       if hasattr(module, "scales") else w.shape[-1])
            if columns % 64 == 0:
                result[name] = module
    if not result:
        raise ValueError("aucun module language_model quantifiable ; vérifier la version de mlx-lm")
    return result


def is_embedding(module):
    return "Embedding" in type(module).__name__


def importance_for(matrix, name, columns, embedding=False):
    if matrix is None:
        return None
    if name not in matrix:
        # L'embedding reçoit des ids, pas un vecteur d'activations par colonne.
        if embedding or name.endswith(".embed_tokens"):
            return None
        raise ValueError(f"imatrix absente : {name}")
    vector = matrix[name]
    if tuple(vector.shape) != (columns,):
        raise ValueError(f"forme imatrix incorrecte : {name}, attendu ({columns},)")
    validate_weight(vector.tolist(), columns)
    return vector


def load_bf16(path):
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
    from mlx_lm import load

    config = json.loads((path / "config.json").read_text())
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("source quantifiée refusée : utiliser les poids BF16 officiels")
    model, tokenizer = load(str(path))
    model.eval()
    cast = getattr(model, "cast_predicate", lambda _: True)
    # Les états sensibles (A_log chez Qwen) gardent le dtype prévu par mlx-lm.
    model.update(tree_map_with_path(
        lambda k, v: v.astype(mx.bfloat16) if cast(k) and mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters()))
    return model, tokenizer


@contextmanager
def capture_calls(modules, observe, with_call_args=False):
    """Patch temporaire de __call__ sur les classes : Python ignore celui d'une instance."""
    names = {id(module): name for name, module in modules.items()}
    def hook(original):
        def call(module, x, *args, **kwargs):
            if id(module) in names:
                if with_call_args:
                    observe(names[id(module)], x, *args, **kwargs)
                else:
                    observe(names[id(module)], x)
            return original(module, x, *args, **kwargs)
        return call
    with ExitStack() as stack:
        originals = {type(m): type(m).__call__ for m in modules.values()}
        for cls, original in originals.items():
            stack.enter_context(patch.object(cls, "__call__", hook(original)))
        yield


def capture(model_path, calib, out, samples=128, length=512, chat=False):
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"destination existante refusée : {out}")
    text = calib.read_text(encoding="utf-8")
    reject_eval_text(text)  # Avant même l'import MLX et le chargement des poids.
    import mlx.core as mx

    model, tokenizer = load_bf16(model_path)
    chunks = list(calibration_chunks(tokenizer, text, samples, length, chat))
    modules = {name: m for name, m in language_modules(model).items() if not is_embedding(m)}
    sums, counts = {}, {}

    def observe(name, x):
        columns = modules[name].weight.shape[-1]
        if hasattr(modules[name], "scales"):
            columns = modules[name].scales.shape[-1] * modules[name].group_size
        if x.shape[-1] != columns or x.size == 0:
            raise ValueError(f"entrée incompatible : {name}, {x.shape}")
        # Réduction f32 avant cumul, sans conserver les activations ni le graphe du forward.
        flat = x.astype(mx.float32).reshape(-1, columns)
        value = (flat * flat).sum(0)
        sums[name] = sums[name] + value if name in sums else value
        counts[name] = counts.get(name, 0) + flat.shape[0]
        mx.eval(sums[name])

    print(f"capture de {len(modules)} modules, {len(chunks)} extraits", flush=True)
    with capture_calls(modules, observe):
        for i, chunk in enumerate(chunks):
            # Aucun cache partagé : chaque extrait repart d'un état récurrent vierge.
            logits = model(mx.array([chunk]))
            mx.eval(logits)
            del logits
            mx.clear_cache()
            print(f"calibration {i + 1}/{samples}", flush=True)
    missing = sorted(set(modules) - sums.keys())
    if missing:
        raise ValueError(f"modules non observés, capture refusée : {missing}")
    result = {name: value / counts[name] for name, value in sums.items()}
    for name, vector in result.items():
        validate_weight(vector.tolist(), vector.size)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), result, metadata={
        "format": "mlx", "statistic": "mean_square_input", "samples": str(samples),
        "length": str(length), "chat": str(chat),
        "calib_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })
    print(f"écrit {out} : {len(result)} vecteurs f32", flush=True)


def _self_check():
    for prompt in PROMPTS:
        try:
            reject_eval_text("avant\n" + prompt.upper().replace(" ", "\n  ") + "\naprès")
            raise AssertionError("prompt réservé accepté")
        except ValueError:
            pass
    reject_eval_text("Un corpus indépendant sur la botanique et la mécanique.")
    for prompt in reserved_prompts()[len(PROMPTS):]:
        try:
            reject_eval_text("x " + prompt + " y")
            raise AssertionError("prompt d'un ensemble figé accepté")
        except ValueError:
            pass
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)
        def decode(self, ids):
            return "".join(ids)
        def apply_chat_template(self, messages, **kwargs):
            return list("<user>" + messages[0]["content"] + "</user><assistant>")
    tok = Tokenizer()
    assert list(calibration_chunks(tok, "abcdefghijkl", 3, 4)) == [list(s) for s in ("abcd", "efgh", "ijkl")]
    mixed = list(calibration_chunks(tok, "abcdefghijkl", 3, 4, True))
    assert mixed[0] == list("abcd") and "<user>efgh</user>" in "".join(mixed[1])
    try:
        list(calibration_chunks(tok, "abc", 2, 4))
        raise AssertionError("corpus court accepté")
    except ValueError:
        pass
    class Linear:
        def __call__(self, x, offset=0):
            return x + offset
    a, b = Linear(), Linear()
    seen = []
    original = Linear.__call__
    try:
        with capture_calls({"layer": a}, lambda name, x: seen.append((name, x))):
            assert a(3, offset=4) == 7 and b(9) == 9
            raise RuntimeError("test restauration")
    except RuntimeError:
        pass
    assert Linear.__call__ is original and seen == [("layer", 3)]
    class Derived(Linear):
        pass
    seen.clear()
    derived = Derived()
    with capture_calls({"base": a, "derived": derived}, lambda name, x: seen.append((name, x))):
        a(1)
        derived(2)
    assert seen == [("base", 1), ("derived", 2)]
    assert importance_for({}, "language_model.model.embed_tokens", 64) is None
    try:
        importance_for({}, "language_model.lm_head", 64)
        raise AssertionError("module absent accepté")
    except ValueError:
        pass
    assert "mlx.core" not in sys.modules


def main():
    if len(sys.argv) == 1:
        _self_check()
        print("imatrix: auto-tests stdlib ok")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True, help="texte indépendant de chat_kl.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--chat", action="store_true", help="un extrait sur deux en chat (L + tokens du template)")
    args = parser.parse_args()
    if args.samples <= 0 or args.length <= 0:
        parser.error("samples et length doivent être positifs")
    capture(args.model, args.calib, args.out, args.samples, args.length, args.chat)


if __name__ == "__main__":
    main()
