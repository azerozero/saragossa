"""Prompts neufs du bench concurrents et empreintes attendues des ids.

Lancer avec le venv MLX (tokenizers, jinja2, transformers y sont déjà) :

  models/mlxenv/bin/python -I bench/competitors/prompts.py build OUT_DIR
  models/mlxenv/bin/python -I bench/competitors/prompts.py expect OUT_DIR PLAN.json

`build` découpe les sources publiques figées (rust-src de la toolchain
épinglée, wikitext-2 raw train) au nombre exact de tokens visé, template et
nonce compris, et écrit `OUT_DIR/prompts.json` (textes, sha, longueurs).
`expect` calcule, pour chaque couple (prompt, nonce) du plan, l'empreinte des
ids par deux chemins indépendants : `tokenizers` + rendu jinja2 du template
du checkpoint, et `transformers.apply_chat_template` (le chemin de mlx-lm).
"""

import hashlib
import json
import os
import pathlib
import re
import sys
import unicodedata

CHECKPOINT = pathlib.Path("/Users/ludwig/workspace/saragossa-oss/models/Qwen3.8-27B-4bit")
NVFP4 = pathlib.Path("/Users/ludwig/workspace/saragossa-oss/models/nvfp4/Qwen3.8-27B-NVFP4")
RUST_SRC = pathlib.Path("/opt/homebrew/Cellar/rust/1.98.1/lib/rustlib/src/rust/library")
# wiki.train.raw extrait de wikitext-2-raw-v1.zip (source unique :
# https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip).
WIKITEXT = pathlib.Path(os.environ.get("WIKITEXT_TRAIN", "corpus/wikitext-2-raw/wiki.train.raw"))
# sha256 publiés des sources : toute dérive arrête la construction.
WIKITEXT_ZIP_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
WIKITEXT_TRAIN_SHA256 = "6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57"

# Nonce : 16 chiffres, un token par chiffre (pré-tokeniseur \p{N}), donc une
# longueur de prompt indépendante de sa valeur.
NONCE_PLACEHOLDER = "0" * 16
NONCE_LINE = "Request id: {nonce}\n\n"

CODE_INSTRUCTION = (
    "\n\nAbove is source code from the Rust standard library. Write a detailed "
    "technical review of it: explain the purpose of each main type and function, "
    "how they interact, and the invariants they rely on, then propose three "
    "concrete improvements with example code. Write at least 600 words."
)
PROSE_INSTRUCTION = (
    "\n\nAbove is an encyclopedia text. Write a detailed summary and analysis of "
    "it, section by section, then discuss what it leaves out and which sources a "
    "reader should consult next. Write at least 600 words."
)
LONG_CODE_INSTRUCTION = (
    "\n\nAbove are source files from the Rust standard library. Choose the five "
    "most important types defined in them and, for each one, explain in detail "
    "its purpose, its invariants and how the other files use it. Write at least "
    "600 words."
)

CHAT = {
    "chat-1": "Write a detailed tutorial on implementing a thread-safe LRU cache in "
              "Rust, with complete code and an explanation of every design choice.",
    "chat-2": "Explain how the TCP congestion control algorithms Reno, CUBIC and BBR "
              "differ. Give a thorough comparison with concrete examples and numbers.",
    "chat-3": "Write a Python script that parses web server access logs and reports "
              "hourly statistics per status code, then explain it line by line.",
    "chat-4": "Write a short story of at least 800 words about a lighthouse keeper "
              "who finds a message in a bottle.",
    "chat-5": "Write a detailed essay on the causes, main events and consequences of "
              "the French Revolution.",
    "chat-6": "Give a comprehensive guide to brewing coffee with a French press: "
              "beans, grind, water, timing, and the most common mistakes.",
    "tune-chat-1": "Explain in depth how a garbage collector based on generational "
                   "copying works, with diagrams described in text and pseudo-code.",
}

# (id, split, target_tokens, source). Sources de code : chemin relatif à
# RUST_SRC. Sources de prose : rang parmi les articles admissibles.
DOCS = [
    ("code4k-1", "eval", 4096, ("rust", "alloc/src/collections/binary_heap/mod.rs")),
    ("code4k-2", "eval", 4096, ("rust", "core/src/str/pattern.rs")),
    ("code4k-3", "eval", 4096, ("rust", "std/src/io/stdio.rs")),
    ("prose4k-1", "eval", 4096, ("wiki", 0)),
    ("prose4k-2", "eval", 4096, ("wiki", 1)),
    ("prose4k-3", "eval", 4096, ("wiki", 2)),
    ("code32k-1", "eval", 32768, ("rust", "alloc/src/collections/vec_deque/mod.rs")),
    ("prose32k-1", "eval", 32768, ("wikicat", 10)),
    ("code256k-1", "eval", 261000, ("rustcat", "core/src")),
    ("tune-code4k-1", "tune", 4096, ("rust", "std/src/sync/mpsc.rs")),
    ("tune-prose4k-1", "tune", 4096, ("wiki", 3)),
    ("tune-code32k-1", "tune", 32768, ("rust", "alloc/src/rc.rs")),
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def fnv1a64(ids):
    """Même empreinte que `prompt_fnv1a64` du journal saragossa serve."""
    value = 0xCBF29CE484222325
    for token in ids:
        for byte in int(token).to_bytes(4, "little"):
            value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def ids_sha256(ids):
    return sha256_bytes(b"".join(int(t).to_bytes(4, "little") for t in ids))


def has_combining_mark(text):
    """Les deux tokenizers du checkpoint divergent sur \\p{M} : on les exclut."""
    return any(unicodedata.category(char).startswith("M") for char in text)


class Renderer:
    """Template jinja du checkpoint + `tokenizers`, sans transformers."""

    def __init__(self, checkpoint):
        import jinja2
        import tokenizers

        self.tokenizer = tokenizers.Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(message):
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        self.template_text = (checkpoint / "chat_template.jinja").read_text()
        self.template = env.from_string(self.template_text)

    def render(self, content):
        return self.template.render(
            messages=[{"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def ids(self, content):
        return self.tokenizer.encode(self.render(content), add_special_tokens=False).ids

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def wiki_articles():
    data = WIKITEXT.read_bytes()
    if sha256_bytes(data) != WIKITEXT_TRAIN_SHA256:
        raise SystemExit(f"{WIKITEXT}: sha256 inattendu")
    text = data.decode("utf-8")
    # Titre d'article : « = Titre = » seul sur sa ligne (pas « = = Section = = »).
    parts = re.split(r"\n (= [^=\n][^\n]*? =) \n", text)
    articles = []
    for index in range(1, len(parts) - 1, 2):
        title = parts[index].strip("= ").strip()
        body = parts[index + 1]
        # Détokenisation wikitext standard des nombres ; le reste est laissé tel quel.
        body = body.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        articles.append((title, f"{title}\n{body.strip()}\n"))
    return articles


def admissible(articles, renderer, min_tokens):
    return [
        (title, body) for title, body in articles
        if not has_combining_mark(body)
        and unicodedata.normalize("NFC", body) == body
        and len(renderer.encode(body)) >= min_tokens
    ]


def rust_file(rel):
    path = RUST_SRC / rel
    data = path.read_bytes()
    return data.decode("utf-8"), {"path": rel, "sha256": sha256_bytes(data)}


def rust_concat(root, exclude, renderer, min_tokens):
    """Fichiers .rs triés de `root`, tests exclus, jusqu'à `min_tokens`."""
    pieces, files, total = [], [], 0
    for path in sorted((RUST_SRC / root).rglob("*.rs")):
        rel = str(path.relative_to(RUST_SRC))
        if rel in exclude or "/tests" in rel or rel.endswith("tests.rs"):
            continue
        data = path.read_bytes()
        text = f"// file: {rel}\n{data.decode('utf-8')}\n"
        if has_combining_mark(text):
            continue
        pieces.append(text)
        files.append({"path": rel, "sha256": sha256_bytes(data)})
        total += len(renderer.encode(text))
        if total >= min_tokens:
            break
    return "".join(pieces), files


def fit(renderer, doc, instruction, target):
    """Plus long préfixe de `doc` (en tokens) tel que le prompt complet tienne en `target`."""
    doc_ids = renderer.encode(doc)

    def content(n):
        text = renderer.decode(doc_ids[:n]).rstrip("�")
        return NONCE_LINE.format(nonce=NONCE_PLACEHOLDER) + text + instruction

    low, high = 0, len(doc_ids)
    while low < high:
        mid = (low + high + 1) // 2
        if len(renderer.ids(content(mid))) <= target:
            low = mid
        else:
            high = mid - 1
    best = content(low)
    return best, len(renderer.ids(best))


def build(out_dir):
    renderer = Renderer(CHECKPOINT)
    articles = wiki_articles()
    wiki4k = admissible(articles, renderer, 4600)
    used = {source[1] for _, _, _, source in DOCS if source[0] == "rust"}
    prompts = []
    for prompt_id, text in CHAT.items():
        content = NONCE_LINE.format(nonce=NONCE_PLACEHOLDER) + text
        prompts.append({
            "id": prompt_id, "kind": "chat", "split": "tune" if prompt_id.startswith("tune") else "eval",
            "target_tokens": None, "content": content, "source": {"origin": "harness"},
        })
    for prompt_id, split, target, source in DOCS:
        kind = "code" if source[0].startswith("rust") else "prose"
        if source[0] == "rust":
            doc, meta = rust_file(source[1])
            meta = {"origin": "rust-src 1.98.1", "files": [meta]}
        elif source[0] == "rustcat":
            doc, files = rust_concat(source[1], used, renderer, target + 2000)
            meta = {"origin": "rust-src 1.98.1", "files": files}
        elif source[0] == "wiki":
            title, doc = wiki4k[source[1]]
            meta = {"origin": "wikitext-2-raw-v1 train", "articles": [title]}
        else:
            # Articles admissibles consécutifs, à partir du rang donné.
            chosen, total = [], 0
            for title, body in wiki4k[source[1]:]:
                chosen.append((title, body))
                total += len(renderer.encode(body))
                if total >= target + 2000:
                    break
            doc = "\n".join(body for _, body in chosen)
            meta = {"origin": "wikitext-2-raw-v1 train", "articles": [t for t, _ in chosen]}
        if source[0] in ("wiki", "wikicat"):
            meta["zip_sha256"] = WIKITEXT_ZIP_SHA256
            meta["train_sha256"] = WIKITEXT_TRAIN_SHA256
        instruction = {"code": CODE_INSTRUCTION, "prose": PROSE_INSTRUCTION}[kind]
        if source[0] == "rustcat":
            instruction = LONG_CODE_INSTRUCTION
        content, actual = fit(renderer, doc, instruction, target)
        prompts.append({
            "id": prompt_id, "kind": kind, "split": split, "target_tokens": target,
            "content": content, "source": meta,
        })
    for prompt in prompts:
        # Le rendu jinja du checkpoint doit égaler le ChatML natif de saragossa
        # (`render_qwen_chatml`, thinking désactivé) octet pour octet.
        chatml = (f"<|im_start|>user\n{prompt['content']}<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        if renderer.render(prompt["content"]) != chatml:
            raise SystemExit(f"{prompt['id']}: rendu jinja différent du ChatML saragossa")
        ids = renderer.ids(prompt["content"])
        prompt["prompt_tokens"] = len(ids)
        prompt["content_sha256"] = sha256_bytes(prompt["content"].encode("utf-8"))
        prompt["ids_sha256_placeholder"] = ids_sha256(ids)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(CHECKPOINT),
        "tokenizer_sha256": sha256_bytes((CHECKPOINT / "tokenizer.json").read_bytes()),
        "chat_template_sha256": sha256_bytes(renderer.template_text.encode("utf-8")),
        "nonce_line": NONCE_LINE,
        "prompts": prompts,
    }
    (out / "prompts.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for prompt in prompts:
        print(f"{prompt['id']:16} {prompt['split']:5} {prompt['prompt_tokens']:7} {prompt['content_sha256'][:12]}")


def with_nonce(content, nonce):
    placeholder = NONCE_LINE.format(nonce=NONCE_PLACEHOLDER)
    if not content.startswith(placeholder) or len(nonce) != 16 or not nonce.isdigit():
        raise ValueError("nonce invalide ou contenu sans ligne de nonce")
    return NONCE_LINE.format(nonce=nonce) + content[len(placeholder):]


def expect(out_dir, plan_path):
    """Empreintes attendues de chaque couple (prompt, nonce) du plan."""
    from transformers import AutoTokenizer

    out = pathlib.Path(out_dir)
    prompts = {p["id"]: p for p in json.loads((out / "prompts.json").read_text())["prompts"]}
    plan = json.loads(pathlib.Path(plan_path).read_text())
    renderer = Renderer(CHECKPOINT)
    llmd_renderer = Renderer(NVFP4)
    hf = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    # Les plans successifs d'un même run complètent le même fichier.
    path = out / "expected.json"
    expected = json.loads(path.read_text()) if path.exists() else {}
    pairs = sorted({(r["prompt"], r["nonce"]) for r in plan["requests"]})
    for prompt_id, nonce in pairs:
        content = with_nonce(prompts[prompt_id]["content"], nonce)
        ids = renderer.ids(content)
        hf_ids = list(hf.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, enable_thinking=False, return_dict=False,
        ))
        llmd_ids = llmd_renderer.ids(content)
        expected[f"{prompt_id}|{nonce}"] = {
            "prompt_tokens": len(ids),
            "fnv1a64": fnv1a64(ids),
            "ids_sha256": ids_sha256(ids),
            "transformers_fnv1a64": fnv1a64(hf_ids),
            "llmd_offline_fnv1a64": fnv1a64(llmd_ids),
            "content_sha256": sha256_bytes(content.encode("utf-8")),
        }
        if len(ids) != prompts[prompt_id]["prompt_tokens"]:
            raise SystemExit(f"{prompt_id}: la longueur dépend du nonce")
    (out / "expected.json").write_text(json.dumps(expected, indent=1))
    agree = sum(e["fnv1a64"] == e["transformers_fnv1a64"] == e["llmd_offline_fnv1a64"]
                for e in expected.values())
    print(f"{len(expected)} couples, {agree} identiques par les trois chemins")


def _self_check():
    # Vecteur partagé avec le test Rust `prompt_fingerprint_matches_reference_fnv1a`.
    assert fnv1a64([]) == "cbf29ce484222325"
    assert fnv1a64([248045, 872, 198]) == "a5c9734054dc81d8"
    assert with_nonce(NONCE_LINE.format(nonce=NONCE_PLACEHOLDER) + "x", "1" * 16).startswith(
        "Request id: 1111111111111111\n\nx")
    assert has_combining_mark("é") and not has_combining_mark("é")


if __name__ == "__main__":
    _self_check()
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "expect":
        expect(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
