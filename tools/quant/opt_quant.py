"""Échelle et biais affines ajustés aux moindres carrés, format MLX affine gs64 (4/5/6/8 bits).

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

def validate_weight(weight, columns):
    """Validation stdlib d'une importance par colonne ; zéro signifie canal ignoré."""
    try:
        valid = len(weight) == columns and all(math.isfinite(v) and v >= 0 for v in weight)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("importance : longueur incorrecte, valeur négative ou non finie")
    if not any(weight):
        raise ValueError("importance entièrement nulle")


def pack_codes(mx, q, shape, bits):
    """Flux little-endian MLX ; 32 codes 5 bits ou 16 codes 6 bits par bloc."""
    rows = math.prod(shape[:-1])
    if bits in (5, 6):
        count = 32 // math.gcd(bits, 32)
        words_per_block = count * bits // 32
        q = q.astype(mx.uint32).reshape(rows, -1, count)
        words = []
        for word in range(words_per_block):
            value = mx.zeros(q.shape[:-1], mx.uint32)
            for i, shift in pack_terms(word, bits):
                value = value | (q[..., i] << shift if shift >= 0 else q[..., i] >> -shift)
            words.append(value)
        return mx.stack(words, axis=-1).reshape(*shape[:-1], shape[-1] * bits // 32)
    # Chemin historique 4/8 bits conservé, opérations et ordre inchangés.
    per = 32 // bits
    q = q.astype(mx.uint32).reshape(rows, shape[-1] // per, per)
    shifts = mx.array([bits * i for i in range(per)], mx.uint32)
    return (q << shifts).sum(-1).astype(mx.uint32).reshape(*shape[:-1], shape[-1] // per)


def pack6_terms(word):
    return pack_terms(word, 6)


def pack_terms(word, bits):
    count = 32 // math.gcd(bits, 32)
    return [(i, bits * i - 32 * word) for i in range(count)
            if -bits < bits * i - 32 * word < 32]


def blend_importance(mx, w, importance):
    """w: [lignes, groupes, 64] ; sigma² porte sur la ligne entière."""
    sigma2 = (w * w).mean(axis=(-2, -1), keepdims=True)
    return importance * mx.sqrt(sigma2 + w * w)


def error_mass(x, weight=None, imatrix_blend=False):
    """Masse non normalisée pour reconvertir RMSE² en erreur totale du fit."""
    if weight is None:
        if imatrix_blend:
            raise ValueError("imatrix_blend exige une importance par colonne")
        return x.size
    values = weight.tolist() if hasattr(weight, "tolist") else list(weight)
    validate_weight(values, x.shape[-1])
    if not imatrix_blend:
        return sum(values) * (x.size // x.shape[-1])
    import mlx.core as mx
    w = x.astype(mx.float32).reshape(-1, x.shape[-1] // GROUP, GROUP)
    importance = mx.array(values, mx.float32).reshape(1, -1, GROUP)
    return blend_importance(mx, w, importance).sum().item()


def fit(x, iters=8, bits=BITS, weight=None, imatrix_blend=False):
    """x: BF16 [..., cols]. Retourne (poids u32 emballés, échelles bf16, biais bf16, rmse0, rmse1).

    weight: E[x²] par colonne, partagé entre lignes et experts. Les RMSE sont
    pondérées et normalisées par la somme des importances si weight est fourni.
    imatrix_blend: multiplie l'importance par sqrt(moyenne_ligne(w²) + w_ij²).
    Les RMSE utilisent alors la somme des importances par élément.
    rmse1 est recalculé par mx.dequantize sur la sortie emballée : il valide aussi l'emballage."""
    import mlx.core as mx

    shape = x.shape
    if not shape or shape[-1] % GROUP:
        raise ValueError("la dernière dimension doit être un multiple de 64")
    rows = x.size // shape[-1]
    groups = shape[-1] // GROUP
    w = x.astype(mx.float32).reshape(rows, groups, GROUP)
    if bits not in (4, 5, 6, 8):
        raise ValueError(f"largeur non prise en charge : {bits} bits (4, 5, 6 ou 8)")
    if imatrix_blend and weight is None:
        raise ValueError("imatrix_blend exige une importance par colonne")
    importance = None
    if weight is not None:
        values = weight.tolist() if hasattr(weight, "tolist") else list(weight)
        validate_weight(values, shape[-1])
        # Une normalisation globale améliore le conditionnement sans changer la norme.
        peak = max(values)
        importance = mx.array([v / peak for v in values], mx.float32).reshape(1, groups, GROUP)
        if imatrix_blend:
            importance = blend_importance(mx, w, importance)
    levels = 2**bits - 1
    w0, s0, b0 = mx.quantize(x, group_size=GROUP, bits=bits, mode="affine")
    s0f, b0f = s0.astype(mx.float32).reshape(rows, groups, 1), b0.astype(mx.float32).reshape(rows, groups, 1)
    s, b = s0f, b0f
    for _ in range(iters):
        q = _codes(mx, w, s, b, levels)
        if importance is None:
            n = float(GROUP)
            sq, sqq = q.sum(-1, keepdims=True), (q * q).sum(-1, keepdims=True)
            sw, sqw = w.sum(-1, keepdims=True), (q * w).sum(-1, keepdims=True)
        else:
            n = mx.maximum(importance.sum(-1, keepdims=True), 1e-30)
            sq = (importance * q).sum(-1, keepdims=True)
            sqq = (importance * q * q).sum(-1, keepdims=True)
            sw = (importance * w).sum(-1, keepdims=True)
            sqw = (importance * q * w).sum(-1, keepdims=True)
        det = n * sqq - sq * sq
        safe = mx.where(det > 0, det, mx.array(1.0, mx.float32))
        s1, b1 = least_squares(n, sq, sqq, sw, sqw, safe)
        ok = (det > 0) & (s1 != 0)
        s, b = mx.where(ok, s1, s), mx.where(ok, b1, b)
        mx.eval(s, b)
    s, b = _bf16(mx, s), _bf16(mx, b)
    q = _codes(mx, w, s, b, levels)
    q_ref = _codes(mx, w, s0f, b0f, levels)
    if importance is None:
        e1 = ((s * q + b - w) ** 2).sum(-1, keepdims=True)
        e0 = ((s0f * q_ref + b0f - w) ** 2).sum(-1, keepdims=True)
    else:
        e1 = (importance * (s * q + b - w) ** 2).sum(-1, keepdims=True)
        e0 = (importance * (s0f * q_ref + b0f - w) ** 2).sum(-1, keepdims=True)
    use = e1 < e0
    s, b, q = mx.where(use, s, s0f), mx.where(use, b, b0f), mx.where(use, q, q_ref)
    packed = pack_codes(mx, q, shape, bits)
    scales, biases = s.astype(mx.bfloat16).reshape(*s0.shape), b.astype(mx.bfloat16).reshape(*b0.shape)
    rmse0 = mx.sqrt(e0.mean() / GROUP)
    # Échelles en f32 : un résultat bf16 ajouterait son propre arrondi à l'erreur mesurée.
    back = mx.dequantize(packed, scales.astype(mx.float32), biases.astype(mx.float32),
                         group_size=GROUP, bits=bits, mode="affine")
    rmse1 = mx.sqrt(((back.reshape(w.shape) - w) ** 2).mean())
    if importance is not None:
        # Le mélange contient déjà toutes les lignes ; les lignes nulles ont
        # une masse nulle et une erreur nulle, y compris pour un tenseur nul.
        denominator = mx.maximum(importance.sum() * (1 if imatrix_blend else rows), 1e-30)
        rmse0 = mx.sqrt(e0.sum() / denominator)
        rmse1 = mx.sqrt((importance * (back.reshape(w.shape) - w) ** 2).sum() / denominator)
    # Le meilleur groupe ne peut pas faire pire que le min-max : sinon l'emballage est faux.
    if rmse1.item() > rmse0.item() * (1 + 1e-3):
        raise ValueError(f"emballage {bits} bits incohérent : rmse {rmse1.item()} > {rmse0.item()}")
    return packed, scales, biases, rmse0, rmse1


def least_squares(n, sq, sqq, sw, sqw, denominator=None):
    """Équations normales scalaires ou MLX ; le signe de l'échelle est conservé."""
    det = n * sqq - sq * sq if denominator is None else denominator
    if denominator is None and det == 0:
        return None, None
    scale = (n * sqw - sq * sw) / det
    return scale, (sw - scale * sq) / n


def fit_group(weights, scale, bias, iters=8, weight=None, bits=BITS, imatrix_blend=False):
    """Oracle scalaire stdlib d'un groupe, à partir de sa grille affine initiale.

    Le fit tensoriel utilise les mêmes équations normales. Cet oracle teste
    aussi les échelles négatives, le retour BF16 et le repli vers le min-max.
    """
    def codes(s, b):
        return [min(2**bits - 1, max(0, round((w - b) / s))) if s else 0 for w in weights]

    def bf16(value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    if weight is not None:
        validate_weight(weight, len(weights))
    importance = [1.0] * len(weights) if weight is None else weight
    if imatrix_blend:
        if weight is None:
            raise ValueError("imatrix_blend exige une importance par colonne")
        sigma2 = sum(w*w for w in weights) / len(weights)
        importance = [a * math.sqrt(sigma2 + w*w) for a, w in zip(importance, weights)]
    s, b = scale, bias
    for _ in range(iters):
        q = codes(s, b)
        candidate, offset = least_squares(sum(importance),
            sum(a*v for a, v in zip(importance, q)),
            sum(a*v*v for a, v in zip(importance, q)),
            sum(a*w for a, w in zip(importance, weights)),
            sum(a*v*w for a, v, w in zip(importance, q, weights)))
        if candidate is not None and candidate != 0:
            s, b = candidate, offset
    s, b = bf16(s), bf16(b)
    q, q0 = codes(s, b), codes(scale, bias)
    e1 = sum(a*(s*v + b - w)**2 for a, v, w in zip(importance, q, weights))
    e0 = sum(a*(scale*v + bias - w)**2 for a, v, w in zip(importance, q0, weights))
    if e1 >= e0:
        s, b, q, e1 = scale, bias, q0, e0
    stream = sum(int(v) << (bits*i) for i, v in enumerate(q))
    packed = [(stream >> offset) & 0xffffffff for offset in range(0, len(q)*bits, 32)]
    mass = sum(importance) or 1.0
    return packed, s, b, math.sqrt(e0/mass), math.sqrt(e1/mass)


def _self_check():
    # Oracle indépendant : sérialisation LE, y compris les codes à cheval sur u32.
    for bits, count in ((5, 32), (6, 16)):
        maximum = (1 << bits) - 1
        for values in ([maximum]*count, list(range(count)),
                       [(i*37+11) & maximum for i in range(count)]):
            nwords = bits * count // 32
            blob = sum(v << (bits*i) for i, v in enumerate(values)).to_bytes(4*nwords, "little")
            expected = struct.unpack("<" + "I"*nwords, blob)
            actual = []
            for word in range(nwords):
                value = 0
                for i, shift in pack_terms(word, bits):
                    value |= values[i] << shift if shift >= 0 else values[i] >> -shift
                actual.append(value & 0xffffffff)
            assert tuple(actual) == expected
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
    assert fit_group(weights, -0.52, 4.05) == fit_group(weights, -0.52, 4.05, weight=[1]*64)
    noisy = [math.sin(i) * 3 for i in range(63)] + [20]
    importance = [1]*63 + [0.0001]
    weighted = fit_group(noisy, 1.5, -3, weight=importance)
    assert weighted[-1] < weighted[-2]
    assert weighted[:3] != fit_group(noisy, 1.5, -3)[:3]
    assert fit_group([2.0]*64, 0, 2, weight=[1]+[0]*63)[-1] == 0
    for bits in (4, 5, 6, 8):
        blended = fit_group(noisy, 23 / (2**bits - 1), -3, weight=importance,
                            bits=bits, imatrix_blend=True)
        sigma2 = sum(w*w for w in noisy) / 64
        expected_weight = [a * math.sqrt(sigma2 + w*w) for a, w in zip(importance, noisy)]
        assert blended == fit_group(noisy, 23 / (2**bits - 1), -3,
                                    weight=expected_weight, bits=bits)
        assert blended[-1] <= blended[-2]
    assert fit_group([0.0]*64, 0, 0, weight=[1]*64, imatrix_blend=True)[-1] == 0
    for bad in ([], [0]*64, [-1]*64, [math.nan]*64, [math.inf]*64, [[1]]*64):
        try:
            validate_weight(bad, 64)
            raise AssertionError("importance invalide acceptée")
        except ValueError:
            pass


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
