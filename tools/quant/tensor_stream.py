"""Accès borné aux payloads safetensors, sans charger les shards dans MLX."""
from dataclasses import dataclass, replace
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import json
import math
import multiprocessing
import shutil
import struct


@dataclass(frozen=True)
class Tensor:
    path: object
    shape: tuple
    dtype: str
    offset: int
    nbytes: int
    source_shape: tuple
    half: object = None

    def read_rows(self, mx, start, stop):
        """Copie uniquement le bloc demandé ; le mapping est fermé avant le fit."""
        import numpy as np

        types = {"BF16": ("<u2", mx.bfloat16), "F16": ("<f2", mx.float16),
                 "F32": ("<f4", mx.float32)}
        if self.dtype not in types:
            raise ValueError(f"type source non pris en charge : {self.dtype}")
        dtype, target = types[self.dtype]
        mapped = np.memmap(self.path, mode="r", dtype=dtype, offset=self.offset,
                           shape=self.source_shape)
        try:
            view = mapped
            if self.half is not None:
                middle = self.source_shape[-2] // 2
                view = view[..., self.half * middle:(self.half + 1) * middle, :]
            indices = np.unravel_index(np.arange(start, stop), self.shape[:-1])
            block = np.array(view[indices], copy=True)
            result = mx.array(block)
            if self.dtype == "BF16":
                result = result.view(target)
            mx.eval(result)
            return result
        finally:
            mapped._mmap.close()


def split(tensor, sections, axis):
    """Vue des experts fusionnés ; les données ne sont pas encore lues."""
    if sections != 2 or axis != -2 or tensor.shape[-2] % 2:
        raise ValueError("forme des experts fusionnés incompatible")
    shape = (*tensor.shape[:-2], tensor.shape[-2] // 2, tensor.shape[-1])
    return tuple(replace(tensor, shape=shape, half=i) for i in range(2))


def catalog(paths):
    """Ne lit que les en-têtes ; refuse doublons, trous et payloads tronqués."""
    tensors = {}
    sizes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
             "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4,
             "U64": 8, "I64": 8, "F64": 8}
    for path in paths:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"en-tête safetensors tronqué : {path.name}")
            length = struct.unpack("<Q", prefix)[0]
            if length > min(size - 8, 100_000_000):
                raise ValueError(f"taille d'en-tête invalide : {path.name}")
            header = json.loads(handle.read(length))
        cursor = 0
        entries = [(k, v) for k, v in header.items() if k != "__metadata__"]
        for key, info in sorted(entries, key=lambda item: item[1]["data_offsets"]):
            shape, dtype = tuple(info["shape"]), info["dtype"]
            start, end = info["data_offsets"]
            if (key in tensors or dtype not in sizes or any(n < 0 for n in shape)
                    or start != cursor or end - start != math.prod(shape) * sizes[dtype]
                    or 8 + length + end > size):
                raise ValueError(f"payload safetensors invalide : {key} ({path.name})")
            tensors[key] = Tensor(path, shape, dtype, 8 + length + start,
                                  end - start, shape)
            cursor = end
        if 8 + length + cursor != size:
            raise ValueError(f"fin safetensors invalide : {path.name}")
    return tensors


def check_space(path, minimum):
    free = shutil.disk_usage(path).free
    if free < minimum * 2**30:
        raise OSError(f"arrêt disque : {free / 2**30:.2f} Gio libres < {minimum} Gio")


def copy_shard(source, destination, minimum):
    # Pas de copie globale en mémoire, ni de clone APFS dont le COW serait différé.
    with source.open("rb") as reader, destination.open("xb") as writer:
        while block := reader.read(8 * 2**20):
            check_space(destination.parent, minimum + len(block) / 2**30)
            writer.write(block)


def write_rows(mx, tensor, destination, start, value):
    """Écrit dans les offsets du template, même si un module traverse les shards."""
    import numpy as np

    types = {"U32": (mx.uint32, "<u4"), "BF16": (mx.bfloat16, "<u2"),
             "F16": (mx.float16, "<f2"), "F32": (mx.float32, "<f4")}
    expected, dtype = types[tensor.dtype]
    rows = math.prod(tensor.shape[:-1])
    if (value.dtype != expected or value.ndim != 2
            or value.shape[-1] != tensor.shape[-1]
            or start < 0 or start + value.shape[0] > rows):
        raise ValueError(f"format de sortie incompatible : {tensor.path.name}")
    raw = value.view(mx.uint16) if tensor.dtype == "BF16" else value
    block = np.asarray(raw).astype(dtype, copy=False).tobytes()
    offset = tensor.offset + start * tensor.shape[-1] * np.dtype(dtype).itemsize
    with destination.open("r+b") as handle:
        handle.seek(offset)
        handle.write(block)


def _init_cpu():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)


def fit_block(job):
    """Un bloc indépendant, écrit sur des plages disjointes des autres blocs."""
    import mlx.core as mx
    import opt_quant

    source, start, stop, bits, weight, blend, targets, minimum = job
    check_space(targets[0][1].parent, minimum)
    x = source.read_rows(mx, start, stop)
    packed, scale, bias, rmse0, rmse1 = opt_quant.fit(
        x, bits=bits, weight=weight, imatrix_blend=blend)
    mx.eval(packed, scale, bias, rmse0, rmse1)
    for (target, destination), value in zip(targets, (packed, scale, bias)):
        write_rows(mx, target, destination, start, value)
    mass = opt_quant.error_mass(x, weight, blend)
    result = rmse0.item() ** 2 * mass, rmse1.item() ** 2 * mass
    del x, packed, scale, bias, rmse0, rmse1, value
    mx.clear_cache()
    return result


@contextmanager
def cpu_pool(workers):
    if workers == 1:
        yield None
        return
    # Spawn évite d'hériter d'un runtime MLX déjà initialisé par fork.
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_cpu,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        yield pool


def fit_blocks(pool, jobs, workers):
    if pool is None:
        yield from map(fit_block, jobs)
        return
    pending = deque()
    try:
        for job in jobs:
            pending.append(pool.submit(fit_block, job))
            if len(pending) == workers:
                # Consommer dans l'ordre garde le diagnostic reproductible.
                # Le map d'Executor soumettrait tout le modèle sans cette borne.
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
