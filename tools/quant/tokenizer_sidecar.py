"""Recopie de `tokenizer.json` à l'identique depuis le modèle source.

transformers 5.x (avant 5.18) charge les Qwen3.5 via `Qwen2Tokenizer`, qui
impose sa regex `\\p{L}+` ; un `save_pretrained` réécrit alors le fichier avec
cette regex et casse les écritures à marques combinantes (dévanagari, arabe
vocalisé, thaï). Un outil de quantification ne doit donc jamais laisser
transformers produire le `tokenizer.json` de sortie. Stdlib seule.
"""

import json
import pathlib
import shutil

NAME = "tokenizer.json"


def pretokenizer_regexes(path):
    """Regex du pré-tokenizer, dans l'ordre du fichier."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            pattern = node.get("pattern")
            if isinstance(pattern, dict) and isinstance(pattern.get("Regex"), str):
                found.append(pattern["Regex"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(pathlib.Path(path).read_text(encoding="utf-8")).get("pre_tokenizer"))
    return found


def check_tokenizer(source, destination):
    """Échoue si `tokenizer.json` de sortie diffère de celui de la source."""
    src, dst = pathlib.Path(source) / NAME, pathlib.Path(destination) / NAME
    if not src.is_file():
        raise SystemExit(f"{NAME} absent de la source : {src}")
    if not dst.is_file():
        raise SystemExit(f"{NAME} absent de la sortie : {dst}")
    expected, actual = pretokenizer_regexes(src), pretokenizer_regexes(dst)
    if actual != expected:
        raise SystemExit(
            f"{dst} : regex du pré-tokenizer réécrite ({actual!r}) au lieu de celle "
            f"de {src} ({expected!r}) ; transformers l'a sans doute sauvegardé")
    if src.read_bytes() != dst.read_bytes():
        raise SystemExit(f"{dst} diffère octet à octet de {src}")


def copy_tokenizer(source, destination):
    """Recopie `tokenizer.json` de la source à l'identique, puis le vérifie."""
    shutil.copyfile(pathlib.Path(source) / NAME, pathlib.Path(destination) / NAME)
    check_tokenizer(source, destination)
