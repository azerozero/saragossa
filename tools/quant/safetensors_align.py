"""Recopie un modèle avec des shards safetensors à données alignées sur 16 Ko.

MLX écrit un en-tête JSON de longueur quelconque : les données commencent à un
offset impair et `SARAGOSSA_RUST_MMAP_WEIGHTS=1` ne peut pas les envelopper
sans copie. L'en-tête est complété d'espaces (permis par le format) pour que
les données commencent sur une page ; les offsets relatifs, déjà alignés sur
16 octets par MLX, restent inchangés. Les autres fichiers, hors fichiers
cachés, sont des liens symboliques vers la source.

Usage : python3 -I tools/quant/safetensors_align.py SOURCE DESTINATION
"""
import json
import shutil
import struct
import sys
from pathlib import Path

PAGE = 16384


def align(source: Path, destination: Path) -> None:
    with source.open("rb") as src:
        (length,) = struct.unpack("<Q", src.read(8))
        header = src.read(length)
        json.loads(header)
        padded = header.rstrip(b" ")
        padded += b" " * (-(8 + len(padded)) % PAGE)
        with destination.open("xb") as dst:
            dst.write(struct.pack("<Q", len(padded)))
            dst.write(padded)
            shutil.copyfileobj(src, dst, 64 << 20)


def main() -> None:
    source, destination = (Path(arg).resolve() for arg in sys.argv[1:3])
    if destination.is_relative_to(source):
        sys.exit(f"destination {destination} sous la source {source}")
    destination.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        target = destination / path.name
        if path.name.startswith("."):
            # Caches et métadonnées git restent propres à la source.
            continue
        if path.suffix == ".safetensors":
            align(path, target)
        else:
            target.symlink_to(path.resolve())


if __name__ == "__main__":
    main()
