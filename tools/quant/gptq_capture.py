"""Collecte bornée de XᵀX, avec routage distinct de chaque expert MLX."""
from collections import defaultdict
import math

from imatrix import capture_calls


def hessian_batches(modules, memory_mib):
    """Budget des accumulateurs f32 seulement, hors modèle et factorisation."""
    if not math.isfinite(memory_mib) or memory_mib <= 0:
        raise ValueError("hessian-memory-mib doit être fini et positif")
    limit = int(memory_mib * 2**20)
    batches, batch, size = [], [], 0
    for name, module in modules.items():
        shape = module.weight.shape
        cost = 4 * shape[-1] ** 2
        if cost > limit:
            raise ValueError(f"H trop grande : {name}, {cost / 2**20:.1f} Mio requis")
        for expert in (range(shape[0]) if len(shape) == 3 else (None,)):
            if batch and size + cost > limit:
                batches.append(batch)
                batch, size = [], 0
            batch.append((name, expert))
            size += cost
    if batch:
        batches.append(batch)
    return batches


def routed_rows(mx, x, indices, experts):
    """Même diffusion des dimensions batch que gather_mm (axe -2 = lignes).

    gate/up : [B,T,1,1,C] et indices [B,T,K] ; down : [B,T,K,1,C].
    Après tri mlx-lm : [T*K,1,C] et indices [T*K]. Aucune moyenne entre experts.
    """
    if indices is None or x.ndim < 2:
        raise ValueError("projection experte sans indices de routage")
    if not mx.issubdtype(indices.dtype, mx.integer):
        raise ValueError("indices de routage non entiers")
    try:
        batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
        routed = mx.broadcast_to(indices, batch).reshape(-1).tolist()
        values = mx.broadcast_to(x, (*batch, *x.shape[-2:]))
    except (ValueError, RuntimeError) as error:
        raise ValueError("formes d'activations et de routage incompatibles") from error
    positions = defaultdict(list)
    for i, expert in enumerate(routed):
        if not 0 <= expert < experts:
            raise ValueError(f"indice expert hors limites : {expert}")
        positions[expert].append(i)
    return values.reshape(-1, *x.shape[-2:]), positions


def collect_hessians(model, modules, chunks, targets):
    """Un forward BF16 par extrait, sans cache récurrent partagé ni rétropropagation."""
    import mlx.core as mx

    wanted = defaultdict(set)
    for name, expert in targets:
        wanted[name].add(expert)
    sums, counts = {}, {}

    def add(key, flat):
        if flat.shape[0] == 0:
            return
        flat = flat.astype(mx.float32)
        value = flat.T @ flat
        sums[key] = sums[key] + value if key in sums else value
        counts[key] = counts.get(key, 0) + flat.shape[0]
        mx.eval(sums[key])

    def observe(name, x, *args, **kwargs):
        shape = modules[name].weight.shape
        if x.shape[-1] != shape[-1] or x.size == 0:
            raise ValueError(f"entrée incompatible : {name}, {x.shape}")
        if len(shape) == 2:
            add((name, None), x.reshape(-1, shape[-1]))
            return
        indices = args[0] if args else kwargs.get("indices")
        values, positions = routed_rows(mx, x, indices, shape[0])
        for expert in wanted[name]:
            selected = positions.get(expert, [])
            if selected:
                flat = values[mx.array(selected, mx.int32)].reshape(-1, shape[-1])
                add((name, expert), flat)

    with capture_calls({n: modules[n] for n in wanted}, observe, with_call_args=True):
        for i, chunk in enumerate(chunks):
            logits = model(mx.array([chunk]))
            mx.eval(logits)
            del logits
            mx.clear_cache()
            print(f"  calibration {i + 1}/{len(chunks)}", flush=True)
    missing = sorted(set(targets) - sums.keys(), key=str)
    if missing:
        raise ValueError(f"modules/experts non observés : {missing} ; élargir le corpus")
    # Normaliser ne change pas GPTQ ; facilite le choix relatif de l'amortissement.
    result = {key: value / counts[key] for key, value in sums.items()}
    mx.eval(*result.values())
    return result, counts
