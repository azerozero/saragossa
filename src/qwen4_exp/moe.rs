//! MoE `qwen4_exp` : softmax f32 sur tous les experts, top-k, renormalisation
//! (`norm_topk_prob`), expert partagé pondéré par `σ(shared_expert_gate)`
//! hors renormalisation (modeling l. 912-992). Les experts sont déquantifiés un
//! par un, seulement s'ils sont routés.

use super::math::{linear, sigmoid, silu};
use crate::{Result, Tensor};

/// Poids du MoE hors experts routés.
pub(crate) struct MoeWeights {
    pub(crate) router: Tensor,
    pub(crate) shared_gate: Tensor,
    pub(crate) shared_up: Tensor,
    pub(crate) shared_down: Tensor,
    pub(crate) shared_expert_gate: Tensor,
    pub(crate) top_k: usize,
    pub(crate) norm_topk_prob: bool,
}

/// Sortie du MoE et experts routés par token (ensembles triés).
pub(crate) struct MoeOutput {
    pub(crate) output: Tensor,
    pub(crate) routing: Vec<Vec<usize>>,
}

fn gated_mlp(x: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> Result<Tensor> {
    let g = linear(x, gate)?;
    let u = linear(x, up)?;
    let hidden = g
        .data()
        .iter()
        .zip(u.data())
        .map(|(a, b)| silu(*a) * b)
        .collect();
    let hidden = Tensor::from_vec(g.shape().to_vec(), hidden)?;
    linear(&hidden, down)
}

/// MoE sur `x` `[T, hidden]` ; `expert(e)` renvoie `(gate, up, down)` déquantifiés.
pub(crate) fn moe(
    x: &Tensor,
    w: &MoeWeights,
    mut expert: impl FnMut(usize) -> Result<(Tensor, Tensor, Tensor)>,
) -> Result<MoeOutput> {
    let (tokens, hidden) = x.as_matrix()?;
    let logits = linear(x, &w.router)?;
    let experts = logits.as_matrix()?.1;
    let mut chosen: Vec<Vec<(usize, f32)>> = Vec::with_capacity(tokens);
    for row in logits.data().chunks_exact(experts) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut probs: Vec<(usize, f32)> = exp.iter().map(|e| e / sum).enumerate().collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        probs.truncate(w.top_k);
        if w.norm_topk_prob {
            let total: f32 = probs.iter().map(|p| p.1).sum();
            probs.iter_mut().for_each(|p| p.1 /= total);
        }
        chosen.push(probs);
    }
    let mut out = vec![0.0_f32; tokens * hidden];
    for e in 0..experts {
        let rows: Vec<(usize, f32)> = chosen
            .iter()
            .enumerate()
            .filter_map(|(t, picks)| picks.iter().find(|p| p.0 == e).map(|p| (t, p.1)))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let (gate, up, down) = expert(e)?;
        let mut input = Vec::with_capacity(rows.len() * hidden);
        for (t, _) in &rows {
            input.extend_from_slice(&x.data()[t * hidden..(t + 1) * hidden]);
        }
        let y = gated_mlp(
            &Tensor::from_vec(vec![rows.len(), hidden], input)?,
            &gate,
            &up,
            &down,
        )?;
        for (r, (t, weight)) in rows.iter().enumerate() {
            for (o, v) in out[t * hidden..(t + 1) * hidden]
                .iter_mut()
                .zip(&y.data()[r * hidden..(r + 1) * hidden])
            {
                *o += v * weight;
            }
        }
    }
    let shared = gated_mlp(x, &w.shared_gate, &w.shared_up, &w.shared_down)?;
    let shared_gate = linear(x, &w.shared_expert_gate)?;
    for t in 0..tokens {
        let g = sigmoid(shared_gate.data()[t]);
        for (o, s) in out[t * hidden..(t + 1) * hidden]
            .iter_mut()
            .zip(&shared.data()[t * hidden..(t + 1) * hidden])
        {
            *o += g * s;
        }
    }
    let routing = chosen
        .into_iter()
        .map(|picks| {
            let mut ids: Vec<usize> = picks.into_iter().map(|p| p.0).collect();
            ids.sort_unstable();
            ids
        })
        .collect();
    Ok(MoeOutput {
        output: Tensor::from_vec(vec![tokens, hidden], out)?,
        routing,
    })
}
