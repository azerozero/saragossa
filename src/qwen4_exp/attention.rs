//! Attention QSA (Qwen Sparse Attention) de `qwen4_exp`, référence sans cache.
//!
//! Transcription de `Qwen4ExpTextQSAIndexer` et `Qwen4ExpTextAttention`
//! (transformers 5.20.2, modeling l. 664-893) :
//! - requêtes de l'indexeur : `q_layernorm` puis RoPE à la position courante ;
//! - clés de bloc : moyenne des clés brutes de 4 tokens visibles consécutifs,
//!   puis `k_layernorm`, puis RoPE à la position du premier token du bloc ;
//! - score `Σ_têtes ReLU(q·k)/√d_idx`, top-`min(budget/4, blocs)`, plus la
//!   queue incomplète : `4·min(⌊V/4⌋, budget/4) + V mod 4` tokens ;
//! - attention GQA sur ces tokens, `q_proj` entrelacé par tête
//!   `(d query | d gate)`, `σ(gate)` avant `o_proj`.
//!
//! Le masque est causal et contigu : les tokens visibles de la requête `i`
//! sont `0..=i`, les blocs sont donc `[4b, 4b+4)` pour toute requête.

use rayon::prelude::*;

use super::math::{inv_freq, linear, rms_norm_1p, rope_in_place, sigmoid};
use crate::{InferError, Result, Tensor};

/// Poids d'une couche QSA (déquantifiés en f32).
pub(crate) struct QsaWeights {
    pub(crate) q_proj: Tensor,
    pub(crate) k_proj: Tensor,
    pub(crate) v_proj: Tensor,
    pub(crate) o_proj: Tensor,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) index_qk: Tensor,
    pub(crate) index_q_norm: Vec<f32>,
    pub(crate) index_k_norm: Vec<f32>,
}

/// Dimensions QSA.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QsaDims {
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) rope_theta: f64,
    pub(crate) index_heads: usize,
    pub(crate) index_head_dim: usize,
    pub(crate) budget: usize,
    pub(crate) compress: usize,
    pub(crate) eps: f32,
}

/// Sortie de couche, sélection propre de l'indexeur (tokens triés) et scores
/// de blocs.
pub(crate) struct QsaOutput {
    pub(crate) output: Tensor,
    pub(crate) selected: Vec<Vec<usize>>,
    pub(crate) scores: Vec<Vec<f32>>,
}

fn per_head_norm_rope(
    rows: &[f32],
    width: usize,
    head_dim: usize,
    norm: &[f32],
    eps: f32,
    freqs: &[f32],
) -> Vec<f32> {
    let mut out = rms_norm_1p(rows, head_dim, head_dim, norm, eps);
    let heads = width / head_dim;
    for (t, row) in out.chunks_exact_mut(width).enumerate() {
        for h in 0..heads {
            rope_in_place(&mut row[h * head_dim..(h + 1) * head_dim], freqs, t);
        }
    }
    out
}

/// Scores de l'indexeur `Σ_têtes ReLU(q·k)/√d_idx` de chaque requête sur ses
/// blocs complets visibles (`⌊(i+1)/compress⌋` valeurs pour la requête `i`).
pub(crate) fn block_scores(
    x: &Tensor,
    w: &QsaWeights,
    d: QsaDims,
    freqs: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let (tokens, _) = x.as_matrix()?;
    let ihd = d.index_head_dim;
    let qk = linear(x, &w.index_qk)?;
    let width = (d.index_heads + 1) * ihd;
    let mut q_raw = Vec::with_capacity(tokens * d.index_heads * ihd);
    let mut k_raw = Vec::with_capacity(tokens * ihd);
    for row in qk.data().chunks_exact(width) {
        q_raw.extend_from_slice(&row[..d.index_heads * ihd]);
        k_raw.extend_from_slice(&row[d.index_heads * ihd..]);
    }
    let q = per_head_norm_rope(
        &q_raw,
        d.index_heads * ihd,
        ihd,
        &w.index_q_norm,
        d.eps,
        freqs,
    );
    // Clés de bloc : identiques pour toute requête qui voit le bloc entier.
    let blocks = tokens / d.compress;
    let mut block_keys = vec![0.0_f32; blocks * ihd];
    for b in 0..blocks {
        let key = &mut block_keys[b * ihd..(b + 1) * ihd];
        for t in b * d.compress..(b + 1) * d.compress {
            for (k, v) in key.iter_mut().zip(&k_raw[t * ihd..(t + 1) * ihd]) {
                *k += v;
            }
        }
        key.iter_mut().for_each(|k| *k /= d.compress as f32);
    }
    let mut block_keys = rms_norm_1p(&block_keys, ihd, ihd, &w.index_k_norm, d.eps);
    for (b, key) in block_keys.chunks_exact_mut(ihd).enumerate() {
        rope_in_place(key, freqs, b * d.compress);
    }
    let scale = (ihd as f32).sqrt();
    Ok((0..tokens)
        .into_par_iter()
        .map(|i| {
            let qi = &q[i * d.index_heads * ihd..(i + 1) * d.index_heads * ihd];
            (0..(i + 1) / d.compress)
                .map(|b| {
                    let key = &block_keys[b * ihd..(b + 1) * ihd];
                    let s: f32 = qi
                        .chunks_exact(ihd)
                        .map(|qh| qh.iter().zip(key).map(|(a, k)| a * k).sum::<f32>().max(0.0))
                        .sum();
                    s / scale
                })
                .collect()
        })
        .collect())
}

/// Tokens retenus (triés) à partir des scores de blocs : top-`budget/compress`
/// blocs complets plus la queue incomplète.
pub(crate) fn select_from_scores(scores: &[Vec<f32>], d: QsaDims) -> Vec<Vec<usize>> {
    let keep = d.budget / d.compress;
    scores
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let visible = i + 1;
            let complete = row.len();
            let mut chosen: Vec<usize> = if complete <= keep {
                (0..complete).collect()
            } else {
                let mut scored: Vec<(f32, usize)> = row
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(b, s)| (s, b))
                    .collect();
                // Ex æquo départagés par l'indice de bloc : torch.topk ne fixe pas
                // cet ordre, les comparaisons d'oracle les traitent à part.
                scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                scored.truncate(keep);
                scored.into_iter().map(|(_, b)| b).collect()
            };
            chosen.sort_unstable();
            let mut selected: Vec<usize> = chosen
                .iter()
                .flat_map(|b| b * d.compress..(b + 1) * d.compress)
                .collect();
            selected.extend(complete * d.compress..visible);
            selected
        })
        .collect()
}

/// Blocs retenus par l'indexeur pour chaque requête (tokens triés).
#[cfg(test)]
pub(crate) fn select_tokens(
    x: &Tensor,
    w: &QsaWeights,
    d: QsaDims,
    freqs: &[f32],
) -> Result<Vec<Vec<usize>>> {
    Ok(select_from_scores(&block_scores(x, w, d, freqs)?, d))
}

/// Couche QSA complète sur `x` `[T, hidden]`. `forced` impose une sélection
/// (oracle : sélection torch en cas d'ex æquo), sinon celle de l'indexeur.
pub(crate) fn qsa(
    x: &Tensor,
    w: &QsaWeights,
    d: QsaDims,
    forced: Option<&[Vec<usize>]>,
) -> Result<QsaOutput> {
    let (tokens, _) = x.as_matrix()?;
    let (hd, heads, kvh) = (d.head_dim, d.heads, d.kv_heads);
    if heads % kvh != 0 || d.rotary_dim > hd || d.rotary_dim > d.index_head_dim {
        return Err(InferError::Config(
            "qsa: dimensions incohérentes".to_string(),
        ));
    }
    let freqs = inv_freq(d.rotary_dim, d.rope_theta);
    let scores = block_scores(x, w, d, &freqs)?;
    let selected = select_from_scores(&scores, d);
    let used = match forced {
        Some(forced) => {
            if forced.len() != tokens
                || forced
                    .iter()
                    .enumerate()
                    .any(|(i, sel)| sel.iter().any(|&j| j > i))
            {
                return Err(InferError::Config(
                    "qsa: sélection imposée non causale".to_string(),
                ));
            }
            forced
        }
        None => &selected[..],
    };

    let qp = linear(x, &w.q_proj)?;
    let mut q_raw = Vec::with_capacity(tokens * heads * hd);
    let mut gate = Vec::with_capacity(tokens * heads * hd);
    for row in qp.data().chunks_exact(2 * heads * hd) {
        for head in row.chunks_exact(2 * hd) {
            q_raw.extend_from_slice(&head[..hd]);
            gate.extend_from_slice(&head[hd..]);
        }
    }
    let q = per_head_norm_rope(&q_raw, heads * hd, hd, &w.q_norm, d.eps, &freqs);
    let k_raw = linear(x, &w.k_proj)?;
    let k = per_head_norm_rope(k_raw.data(), kvh * hd, hd, &w.k_norm, d.eps, &freqs);
    let v = linear(x, &w.v_proj)?;
    let v = v.data();
    let scale = (hd as f32).powf(-0.5);
    let group = heads / kvh;
    let mut attn = vec![0.0_f32; tokens * heads * hd];
    attn.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(i, out)| {
            let sel = &used[i];
            let mut probs = vec![0.0_f32; sel.len()];
            for h in 0..heads {
                let g = h / group;
                let qh = &q[(i * heads + h) * hd..(i * heads + h + 1) * hd];
                let mut max = f32::NEG_INFINITY;
                for (p, &j) in probs.iter_mut().zip(sel) {
                    let kj = &k[(j * kvh + g) * hd..(j * kvh + g + 1) * hd];
                    *p = qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                    max = max.max(*p);
                }
                let mut sum = 0.0_f32;
                for p in probs.iter_mut() {
                    *p = (*p - max).exp();
                    sum += *p;
                }
                let oh = &mut out[h * hd..(h + 1) * hd];
                for (p, &j) in probs.iter().zip(sel) {
                    let vj = &v[(j * kvh + g) * hd..(j * kvh + g + 1) * hd];
                    for (o, x) in oh.iter_mut().zip(vj) {
                        *o += p / sum * x;
                    }
                }
            }
        });
    for (a, g) in attn.iter_mut().zip(&gate) {
        *a *= sigmoid(*g);
    }
    let attn = Tensor::from_vec(vec![tokens, heads * hd], attn)?;
    Ok(QsaOutput {
        output: linear(&attn, &w.o_proj)?,
        selected,
        scores,
    })
}
