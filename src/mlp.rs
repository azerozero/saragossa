//! MLP dense, MoE et routage des experts du modèle.

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::decoder::flags::trace_moe_enabled;
use crate::{silu, softmax, ForwardRuntime, InferError, Linear, Result, Tensor};

#[derive(Clone, Debug)]
/// Sélectionne le type de bloc feed-forward d'une couche.
pub enum FeedForward {
    /// Utilise un MLP dense classique.
    Dense(Box<GatedMlp>),
    /// Utilise un MLP à experts routés.
    Moe(Box<MoeMlp>),
}

impl FeedForward {
    /// Exécute le bloc avec le runtime CPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un sous-bloc échoue.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_runtime(x, ForwardRuntime::cpu())
    }

    /// Exécute le bloc avec le runtime demandé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un sous-bloc ou le runtime échoue.
    pub fn forward_with_runtime(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward_with_runtime(x, runtime),
            Self::Moe(mlp) => mlp.forward_with_runtime(x, runtime),
        }
    }
}

#[derive(Clone, Debug)]
/// Représente un MLP SwiGLU dense.
pub struct GatedMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GatedMlp {
    /// Crée un MLP SwiGLU depuis ses trois projections.
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Exécute le MLP avec le runtime CPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection échoue.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_runtime(x, ForwardRuntime::cpu())
    }

    /// Exécute le MLP avec le runtime demandé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une projection ou le runtime échoue.
    pub fn forward_with_runtime(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        let gate = silu(&self.gate_proj.forward_with_runtime(x, runtime)?);
        let up = self.up_proj.forward_with_runtime(x, runtime)?;
        let hidden = gate.mul_elementwise(&up)?;
        self.down_proj.forward_with_runtime(&hidden, runtime)
    }

    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        expect(
            dead_code,
            reason = "utilisé par les buffers MoE Metal, absent du build CPU pur"
        )
    )]
    pub(crate) fn projections(&self) -> (&Linear, &Linear, &Linear) {
        (&self.gate_proj, &self.up_proj, &self.down_proj)
    }
}

#[derive(Clone, Debug)]
/// Représente un MLP MoE avec experts routés.
pub struct MoeMlp {
    router: Linear,
    experts: Vec<GatedMlp>,
    shared_expert: Option<GatedMlp>,
    shared_expert_gate: Option<Linear>,
    top_k: usize,
}

impl MoeMlp {
    /// Crée un MLP MoE validé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la configuration MoE est incohérente.
    pub fn new(
        router: Linear,
        experts: Vec<GatedMlp>,
        shared_expert: Option<GatedMlp>,
        shared_expert_gate: Option<Linear>,
        top_k: usize,
    ) -> Result<Self> {
        if experts.is_empty() {
            return Err(InferError::Config("MoE sans expert".to_string()));
        }
        if top_k == 0 {
            return Err(InferError::Config("MoE top_k nul".to_string()));
        }
        if shared_expert.is_some() != shared_expert_gate.is_some() {
            return Err(InferError::Config(
                "MoE shared expert partiellement initialisé".to_string(),
            ));
        }
        Ok(Self {
            router,
            experts,
            shared_expert,
            shared_expert_gate,
            top_k,
        })
    }

    /// Exécute le MoE avec le runtime CPU.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le routage ou un expert échoue.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_runtime(x, ForwardRuntime::cpu())
    }

    /// Exécute le MoE avec le runtime demandé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le routage, un expert ou le runtime échoue.
    pub fn forward_with_runtime(&self, x: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        let (batch, hidden) = x.as_matrix()?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            if batch == 1 {
                if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                    self.shared_metal_parts()
                {
                    match metal.moe_gated_router_topk_shared(
                        x,
                        router,
                        experts,
                        top_k,
                        shared_expert,
                        shared_gate,
                    ) {
                        Ok(output) => return Ok(output),
                        Err(error) => {
                            if trace_moe_enabled() {
                                eprintln!("moe shared router gpu fallback: {error}");
                            }
                        }
                    }
                } else {
                    match metal.moe_gated_router_topk(x, &self.router, &self.experts, self.top_k) {
                        Ok(output) => return Ok(output),
                        Err(error) => {
                            if trace_moe_enabled() {
                                eprintln!("moe router gpu fallback: {error}");
                            }
                        }
                    }
                }
            }
        }
        let router_logits = self.router.forward_with_runtime(x, runtime)?;
        let (_, expert_count) = router_logits.as_matrix()?;
        if expert_count != self.experts.len() {
            return Err(InferError::Dimension(format!(
                "routeur MoE experts={expert_count}, poids experts={}",
                self.experts.len()
            )));
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            if batch > 1 && self.shared_expert.is_none() {
                let mut weighted_rows = Vec::with_capacity(batch);
                for row in 0..batch {
                    let probs = softmax(router_logits.row_slice(row)?, 1.0);
                    let top = top_k_indices(&probs, self.top_k);
                    let denom = top
                        .iter()
                        .map(|idx| probs[*idx])
                        .sum::<f32>()
                        .max(f32::EPSILON);
                    weighted_rows.push(
                        top.iter()
                            .map(|idx| (*idx, probs[*idx] / denom))
                            .collect::<Vec<_>>(),
                    );
                }
                match metal.moe_gated_topk_batch(x, &self.experts, &weighted_rows) {
                    Ok(output) => return Ok(output),
                    Err(error) => {
                        if trace_moe_enabled() {
                            eprintln!("moe batch gpu fallback: {error}");
                        }
                    }
                }
            }
        }

        let mut rows = Vec::new();
        let mut out_dim = None;
        for row in 0..batch {
            let input = Tensor::row(x.row_slice(row)?.to_vec())?;
            let probs = softmax(router_logits.row_slice(row)?, 1.0);
            let top = top_k_indices(&probs, self.top_k);
            let denom = top
                .iter()
                .map(|idx| probs[*idx])
                .sum::<f32>()
                .max(f32::EPSILON);
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if let Some(metal) = runtime.metal_executor() {
                if self.shared_expert.is_none() {
                    let weighted_top = top
                        .iter()
                        .map(|idx| (*idx, probs[*idx] / denom))
                        .collect::<Vec<_>>();
                    let expert_out = metal.moe_gated_topk(&input, &self.experts, &weighted_top)?;
                    match out_dim {
                        Some(dim) if dim != expert_out.as_row()?.len() => {
                            return Err(InferError::Dimension(format!(
                                "MoE out_dim incohérent: {dim} puis {}",
                                expert_out.as_row()?.len()
                            )))
                        }
                        Some(_) => {}
                        None => out_dim = Some(expert_out.as_row()?.len()),
                    }
                    rows.extend_from_slice(expert_out.as_row()?);
                    continue;
                }
            }
            let mut acc = None;

            for expert_idx in top {
                let weight = probs[expert_idx] / denom;
                let expert_out = self.experts[expert_idx].forward_with_runtime(&input, runtime)?;
                add_scaled_row(&mut acc, expert_out.as_row()?, weight)?;
            }

            if let (Some(shared), Some(shared_gate)) =
                (&self.shared_expert, &self.shared_expert_gate)
            {
                let gate = sigmoid(
                    shared_gate
                        .forward_with_runtime(&input, runtime)?
                        .as_row()?[0],
                );
                let shared_out = shared.forward_with_runtime(&input, runtime)?;
                add_scaled_row(&mut acc, shared_out.as_row()?, gate)?;
            }

            let row_out = acc.ok_or_else(|| InferError::Config("MoE sans sortie".to_string()))?;
            match out_dim {
                Some(dim) if dim != row_out.len() => {
                    return Err(InferError::Dimension(format!(
                        "MoE out_dim incohérent: {dim} puis {}",
                        row_out.len()
                    )))
                }
                Some(_) => {}
                None => out_dim = Some(row_out.len()),
            }
            rows.extend(row_out);
        }

        Tensor::from_vec(vec![batch, out_dim.unwrap_or(hidden)], rows)
    }

    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        expect(
            dead_code,
            reason = "utilisé par les chemins MoE Metal, absent du build CPU pur"
        )
    )]
    pub(crate) fn metal_parts(&self) -> Option<(&Linear, &[GatedMlp], usize)> {
        if self.shared_expert.is_some() {
            return None;
        }
        Some((&self.router, &self.experts, self.top_k))
    }

    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        expect(
            dead_code,
            reason = "utilisé par les chemins MoE shared Metal, absent du build CPU pur"
        )
    )]
    pub(crate) fn shared_metal_parts(
        &self,
    ) -> Option<(&Linear, &[GatedMlp], usize, &GatedMlp, &Linear)> {
        self.shared_expert
            .as_ref()
            .zip(self.shared_expert_gate.as_ref())
            .map(|(shared_expert, shared_gate)| {
                (
                    &self.router,
                    self.experts.as_slice(),
                    self.top_k,
                    shared_expert,
                    shared_gate,
                )
            })
    }
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
    indices.truncate(k.min(indices.len()));
    indices
}

fn add_scaled_row(acc: &mut Option<Vec<f32>>, row: &[f32], scale: f32) -> Result<()> {
    if let Some(acc) = acc {
        if acc.len() != row.len() {
            return Err(InferError::Dimension(format!(
                "MoE somme rows incompatibles: {} vs {}",
                acc.len(),
                row.len()
            )));
        }
        for (dst, value) in acc.iter_mut().zip(row.iter()) {
            *dst += value * scale;
        }
    } else {
        *acc = Some(row.iter().map(|value| value * scale).collect());
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_mlp_applies_silu_gate() {
        let gate = Linear::new(
            Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("invariant: poids valides"),
            None,
        )
        .expect("invariant: linear valide");
        let up = Linear::new(
            Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("invariant: poids valides"),
            None,
        )
        .expect("invariant: linear valide");
        let down = Linear::new(
            Tensor::from_vec(vec![1, 2], vec![1.0, 1.0]).expect("invariant: poids valides"),
            None,
        )
        .expect("invariant: linear valide");
        let mlp = GatedMlp::new(gate, up, down);
        let x = Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).expect("invariant: entrée valide");
        let out = mlp.forward(&x).expect("invariant: forward valide");
        let expected = 1.0 * 0.731_058_6 + 2.0 * 1.761_594_2;
        assert!((out.data()[0] - expected).abs() < 1.0e-5);
    }

    #[test]
    fn moe_routes_to_top_expert() {
        let router = Linear::new(
            Tensor::from_vec(vec![2, 2], vec![4.0, 0.0, 0.0, 4.0])
                .expect("invariant: routeur valide"),
            None,
        )
        .expect("invariant: routeur linear valide");
        let expert_a = constant_expert(1.0);
        let expert_b = constant_expert(2.0);
        let moe = MoeMlp::new(router, vec![expert_a, expert_b], None, None, 1)
            .expect("invariant: MoE valide");
        let input =
            Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("invariant: input");

        let out = moe.forward(&input).expect("invariant: MoE forward valide");

        assert_eq!(out.shape(), &[2, 1]);
        assert!((out.data()[0] - silu_scalar(1.0)).abs() < 1.0e-5);
        assert!((out.data()[1] - 2.0 * silu_scalar(2.0)).abs() < 1.0e-5);
    }

    fn constant_expert(scale: f32) -> GatedMlp {
        let gate = Linear::new(
            Tensor::from_vec(vec![1, 2], vec![scale, scale]).expect("invariant: gate"),
            None,
        )
        .expect("invariant: gate linear");
        let up = Linear::new(
            Tensor::from_vec(vec![1, 2], vec![scale, scale]).expect("invariant: up"),
            None,
        )
        .expect("invariant: up linear");
        let down = Linear::new(
            Tensor::from_vec(vec![1, 1], vec![1.0]).expect("invariant: down"),
            None,
        )
        .expect("invariant: down linear");
        GatedMlp::new(gate, up, down)
    }

    fn silu_scalar(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }
}
