"""Échelle et biais affines ajustés aux moindres carrés, même format affine gs64 que MLX (4 ou 8 bits).

Le min-max de MLX cale la grille sur les extrêmes du groupe. Ici, q et (s, b)
sont réajustés en alternance, puis chaque groupe garde la meilleure des deux
solutions : l'erreur quadratique ne peut pas augmenter, et les kernels ne changent pas.
L'échelle s peut être négative : seul zéro est dégénéré.

Résultat mesuré : perplexité +1,7 % contre +2,9 % pour le min-max par rapport
au BF16, forward mlx-lm, 8160 tokens. Ce résultat décrit le checkpoint évalué,
pas une garantie pour d'autres poids. Conversion : convert_ls.py --help.
Sans argument, auto-test stdlib de l'ajustement scalaire partagé ; aucun import MLX.
"""
import math
import struct

BITS, GROUP = 4, 64
LEVELS = 2**BITS - 1

def _codes(mx, w, s, b, levels=LEVELS):
    safe = mx.where(s != 0, s, mx.array(1.0, mx.float32))
    return mx.where(s != 0, mx.clip(mx.round((w - b) / safe), 0, levels), mx.zeros_like(w))

def _bf16(mx, x):
    return x.astype(mx.bfloat16).astype(mx.float32)

def fit(x, iters=8, bits=BITS):
    """x: BF16 [..., cols]. Retourne (poids u32 emballés, échelles bf16, biais bf16, rmse0, rmse1).

    rmse1 est recalculé par mx.dequantize sur la sortie emballée : il valide aussi l'emballage."""
    import mlx.core as mx

    shape = x.shape
    if not shape or shape[-1] % GROUP:
        raise ValueError("la dernière dimension doit être un multiple de 64")
    rows = x.size // shape[-1]
    groups = shape[-1] // GROUP
    w = x.astype(mx.float32).reshape(rows, groups, GROUP)
    levels = 2**bits - 1
    w0, s0, b0 = mx.quantize(x, group_size=GROUP, bits=bits, mode="affine")
    s0f, b0f = s0.astype(mx.float32).reshape(rows, groups, 1), b0.astype(mx.float32).reshape(rows, groups, 1)
    s, b = s0f, b0f
    for _ in range(iters):
        q = _codes(mx, w, s, b, levels)
        n = float(GROUP)
        sq, sqq = q.sum(-1, keepdims=True), (q * q).sum(-1, keepdims=True)
        sw, sqw = w.sum(-1, keepdims=True), (q * w).sum(-1, keepdims=True)
        det = n * sqq - sq * sq
        safe = mx.where(det > 0, det, mx.array(1.0, mx.float32))
        s1, b1 = least_squares(n, sq, sqq, sw, sqw, safe)
        ok = (det > 0) & (s1 != 0)
        s, b = mx.where(ok, s1, s), mx.where(ok, b1, b)
        mx.eval(s, b)
    s, b = _bf16(mx, s), _bf16(mx, b)
    q = _codes(mx, w, s, b, levels)
    q_ref = _codes(mx, w, s0f, b0f, levels)
    e1 = ((s * q + b - w) ** 2).sum(-1, keepdims=True)
    e0 = ((s0f * q_ref + b0f - w) ** 2).sum(-1, keepdims=True)
    use = e1 < e0
    s, b, q = mx.where(use, s, s0f), mx.where(use, b, b0f), mx.where(use, q, q_ref)
    per = 32 // bits
    q = q.astype(mx.uint32).reshape(rows, groups * GROUP // per, per)
    shifts = mx.array([bits * i for i in range(per)], mx.uint32)
    packed = (q << shifts).sum(-1).astype(mx.uint32).reshape(*shape[:-1], shape[-1] // per)
    scales, biases = s.astype(mx.bfloat16).reshape(*s0.shape), b.astype(mx.bfloat16).reshape(*b0.shape)
    rmse0 = mx.sqrt(e0.mean() / GROUP)
    # Échelles en f32 : un résultat bf16 ajouterait son propre arrondi à l'erreur mesurée.
    back = mx.dequantize(packed, scales.astype(mx.float32), biases.astype(mx.float32),
                         group_size=GROUP, bits=bits, mode="affine")
    rmse1 = mx.sqrt(((back.reshape(w.shape) - w) ** 2).mean())
    return packed, scales, biases, rmse0, rmse1


def least_squares(n, sq, sqq, sw, sqw, denominator=None):
    """Équations normales scalaires ou MLX ; le signe de l'échelle est conservé."""
    det = n * sqq - sq * sq if denominator is None else denominator
    if denominator is None and det == 0:
        return None, None
    scale = (n * sqw - sq * sw) / det
    return scale, (sw - scale * sq) / n


def fit_group(weights, scale, bias, iters=8):
    """Oracle scalaire stdlib d'un groupe, à partir de sa grille affine initiale.

    Le fit tensoriel utilise les mêmes équations normales. Cet oracle teste
    aussi les échelles négatives, le retour BF16 et le repli vers le min-max.
    """
    def codes(s, b):
        return [min(LEVELS, max(0, round((w - b) / s))) if s else 0 for w in weights]

    def bf16(value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    s, b = scale, bias
    for _ in range(iters):
        q = codes(s, b)
        candidate, offset = least_squares(len(q), sum(q), sum(v*v for v in q),
                                          sum(weights), sum(v*w for v, w in zip(q, weights)))
        if candidate is not None and candidate != 0:
            s, b = candidate, offset
    s, b = bf16(s), bf16(b)
    q, q0 = codes(s, b), codes(scale, bias)
    e1 = sum((s*v + b - w)**2 for v, w in zip(q, weights))
    e0 = sum((scale*v + bias - w)**2 for v, w in zip(q0, weights))
    if e1 >= e0:
        s, b, q, e1 = scale, bias, q0, e0
    packed = [sum(int(v) << (4*i) for i, v in enumerate(q[j:j+8]))
              for j in range(0, len(q), 8)]
    return packed, s, b, math.sqrt(e0/len(q)), math.sqrt(e1/len(q))


def _self_check():
    q = list(range(16)) * 4
    weights = [-0.5 * v + 4 for v in q]
    assert least_squares(64, sum(q), sum(v*v for v in q), sum(weights),
                         sum(v*w for v, w in zip(q, weights))) == (-0.5, 4)
    assert least_squares(64, 0, 0, 64, 0) == (None, None)
    packed, scale, bias, old, new = fit_group(weights, -0.52, 4.05)
    assert scale == -0.5 and bias == 4 and new == 0 and old > 0
    assert packed == [0x76543210, 0xfedcba98] * 4
    assert fit_group(weights, -0.5, 4)[1:] == (-0.5, 4, 0, 0)
    assert fit_group([2.0]*64, 0, 2) == ([0]*8, 0, 2, 0, 0)
    assert fit_group(weights, -0.5, 4, iters=0)[-1] == 0


if __name__ == "__main__":
    import pathlib
    import sys

    if len(sys.argv) == 1:
        _self_check()
        print("opt_quant: auto-test scalaire ok (fit MLX non exécuté)")
    else:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from convert_ls import main
        main()
