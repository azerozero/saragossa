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
    /// Écart de probabilité entre le k-ième et le (k+1)-ième expert, par token.
    pub(crate) margins: Vec<f32>,
    /// `(k-ième, (k+1)-ième)` experts au seuil du top-k, par token.
    pub(crate) boundary: Vec<(usize, usize)>,
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
    forced: Option<&[Option<Vec<usize>>]>,
    mut expert: impl FnMut(usize) -> Result<(Tensor, Tensor, Tensor)>,
) -> Result<MoeOutput> {
    let (tokens, hidden) = x.as_matrix()?;
    let logits = linear(x, &w.router)?;
    let experts = logits.as_matrix()?.1;
    let mut chosen: Vec<Vec<(usize, f32)>> = Vec::with_capacity(tokens);
    let mut margins = Vec::with_capacity(tokens);
    let mut boundary = Vec::with_capacity(tokens);
    // Sélection propre du routeur, conservée même quand une sélection est imposée.
    let mut natural: Vec<Vec<usize>> = Vec::with_capacity(tokens);
    for row in logits.data().chunks_exact(experts) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut probs: Vec<(usize, f32)> = exp.iter().map(|e| e / sum).enumerate().collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        boundary.push(match (probs.get(w.top_k - 1), probs.get(w.top_k)) {
            (Some(kth), Some(next)) => (kth.0, next.0),
            _ => (usize::MAX, usize::MAX),
        });
        margins.push(match (probs.get(w.top_k - 1), probs.get(w.top_k)) {
            (Some(kth), Some(next)) => kth.1 - next.1,
            _ => f32::INFINITY,
        });
        probs.truncate(w.top_k);
        let mut own: Vec<usize> = probs.iter().map(|p| p.0).collect();
        own.sort_unstable();
        natural.push(own);
        // Oracles : experts imposés (quasi-égalité de l'oracle), poids des
        // probabilités propres renormalisées sur l'ensemble imposé.
        if let Some(set) = forced
            .and_then(|f| f.get(chosen.len()))
            .and_then(Option::as_ref)
        {
            let all: Vec<f32> = exp.iter().map(|e| e / sum).collect();
            probs = set
                .iter()
                .map(|&e| {
                    all.get(e).map(|&p| (e, p)).ok_or_else(|| {
                        crate::InferError::Shape(format!("expert imposé {e} hors des {experts}"))
                    })
                })
                .collect::<Result<_>>()?;
        }
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
    // Routage propre (avant imposition) : les oracles comparent le choix du
    // routeur, pas l'ensemble imposé.
    let routing = natural;
    Ok(MoeOutput {
        output: Tensor::from_vec(vec![tokens, hidden], out)?,
        routing,
        margins,
        boundary,
    })
}
