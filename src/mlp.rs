//! MLP dense, MoE et routage des experts du modèle.

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::runtime_flags::trace_moe_enabled;
use crate::{
    gelu_tanh, rms_norm, silu, softmax, ForwardRuntime, InferError, Linear, Result, Tensor,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Sélectionne l'activation de la porte d'un MLP gated.
pub enum Activation {
    /// SwiGLU : porte `silu` (Qwen, Llama, Mistral).
    #[default]
    Silu,
    /// GeGLU : porte `gelu_pytorch_tanh` (Gemma).
    GeluTanh,
}

impl Activation {
    fn apply(self, x: &Tensor) -> Tensor {
        match self {
            Self::Silu => silu(x),
            Self::GeluTanh => gelu_tanh(x),
        }
    }
}

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
/// Représente un MLP gated dense (SwiGLU ou GeGLU).
pub struct GatedMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl GatedMlp {
    /// Crée un MLP SwiGLU depuis ses trois projections.
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            activation: Activation::Silu,
        }
    }

    /// Fixe l'activation de la porte (GeGLU pour Gemma).
    #[must_use]
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
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
        let gate = self
            .activation
            .apply(&self.gate_proj.forward_with_runtime(x, runtime)?);
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
    router_norm: Option<(Tensor, f32)>,
    per_expert_scale: Option<Tensor>,
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
            router_norm: None,
            per_expert_scale: None,
        })
    }

    /// Ajoute une normalisation avant le routeur.
    #[must_use]
    pub fn with_router_norm(mut self, weight: Tensor, eps: f32) -> Self {
        self.router_norm = Some((weight, eps));
        self
    }

    /// Ajoute une échelle multiplicative par expert après le softmax top-k.
    #[must_use]
    pub fn with_per_expert_scale(mut self, scale: Tensor) -> Self {
        self.per_expert_scale = Some(scale);
        self
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
            if batch == 2 {
                if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                    self.shared_metal_parts()
                {
                    match metal.moe_gated_router_topk_shared_batch2(
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
                                eprintln!("moe shared batch2 gpu fallback: {error}");
                            }
                        }
                    }
                }
            }
            if batch > 1 && crate::runtime_flags::prefill_moe_rows_enabled() {
                if let Some((router, experts, top_k, shared_expert, shared_gate)) =
                    self.shared_metal_parts()
                {
                    match metal.moe_gated_router_topk_shared_rows(
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
                                eprintln!("moe shared rows gpu fallback: {error}");
                            }
                        }
                    }
                }
            }
        }
        let router_input = match &self.router_norm {
            Some((weight, eps)) => rms_norm(x, weight, *eps)?,
            None => x.clone(),
        };
        let router_logits = self.router.forward_with_runtime(&router_input, runtime)?;
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
                    weighted_rows.push(self.top_weights(router_logits.row_slice(row)?)?);
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
            let top_weights = self.top_weights(router_logits.row_slice(row)?)?;
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if let Some(metal) = runtime.metal_executor() {
                if self.shared_expert.is_none() {
                    let expert_out = metal.moe_gated_topk(&input, &self.experts, &top_weights)?;
                    match out_dim {
                        Some(dim) if dim != expert_out.as_row()?.len() => {
                            return Err(InferError::Dimension(format!(
                                "MoE out_dim incohérent: {dim} puis {}",
                                expert_out.as_row()?.len()
                            )));
                        }
                        Some(_) => {}
                        None => out_dim = Some(expert_out.as_row()?.len()),
                    }
                    rows.extend_from_slice(expert_out.as_row()?);
                    continue;
                }
            }
            let mut acc = None;

            for (expert_idx, weight) in top_weights {
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
                    )));
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
        if self.router_norm.is_some() || self.per_expert_scale.is_some() {
            return None;
        }
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
            .filter(|_| self.router_norm.is_none() && self.per_expert_scale.is_none())
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

    fn top_weights(&self, logits: &[f32]) -> Result<Vec<(usize, f32)>> {
        let top = top_k_indices(logits, self.top_k);
        let top_logits = top.iter().map(|idx| logits[*idx]).collect::<Vec<_>>();
        let weights = softmax(&top_logits, 1.0);
        let mut out = Vec::with_capacity(top.len());
        for (rank, expert_idx) in top.into_iter().enumerate() {
            let scale = match &self.per_expert_scale {
                Some(tensor) => tensor.data().get(expert_idx).copied().ok_or_else(|| {
                    InferError::Dimension(format!(
                        "per_expert_scale absent pour expert {expert_idx}"
                    ))
                })?,
                None => 1.0,
            };
            out.push((expert_idx, weights[rank] * scale));
        }
        Ok(out)
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

    use proptest::prelude::*;

    /// Construit un MoE minimal dont seul le routage (`top_weights`) est exercé.
    fn routing_moe(top_k: usize) -> MoeMlp {
        let router = Linear::new(
            Tensor::from_vec(vec![1, 2], vec![1.0, 1.0]).expect("invariant: routeur"),
            None,
        )
        .expect("invariant: routeur linear");
        MoeMlp::new(router, vec![constant_expert(1.0)], None, None, top_k)
            .expect("invariant: MoE routage valide")
    }

    /// Ordre du routage CPU : logits décroissants (`total_cmp`), égalités par
    /// indice croissant (tri stable de `top_k_indices`).
    fn beats(logits: &[f32], winner: usize, loser: usize) -> bool {
        match logits[winner].total_cmp(&logits[loser]) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => winner < loser,
            std::cmp::Ordering::Less => false,
        }
    }

    /// Vérifie que tous les logits sont distincts sous `total_cmp`.
    fn all_distinct(logits: &[f32]) -> bool {
        let mut sorted = logits.to_vec();
        sorted.sort_by(f32::total_cmp);
        sorted
            .windows(2)
            .all(|pair| pair[0].total_cmp(&pair[1]).is_ne())
    }

    /// Tailles réalistes de routeur MoE (Qwen3-MoE : jusqu'à 128 experts / top-8).
    fn routing_cases() -> impl Strategy<Value = (Vec<f32>, usize)> {
        (
            prop_oneof![Just(8_usize), Just(64), Just(128)],
            prop_oneof![Just(1_usize), Just(4), Just(8)],
        )
            .prop_flat_map(|(expert_count, top_k)| {
                (
                    proptest::collection::vec(-1.0e4_f32..1.0e4, expert_count),
                    Just(top_k),
                )
            })
    }

    /// Mélange de logits saturés (±1e4) et intermédiaires pour la stabilité.
    fn extreme_routing_cases() -> impl Strategy<Value = (Vec<f32>, usize)> {
        (
            prop_oneof![Just(8_usize), Just(64), Just(128)],
            prop_oneof![Just(1_usize), Just(4), Just(8)],
        )
            .prop_flat_map(|(expert_count, top_k)| {
                (
                    proptest::collection::vec(
                        prop_oneof![
                            2 => Just(-1.0e4_f32),
                            2 => Just(1.0e4_f32),
                            3 => -1.0e4_f32..1.0e4,
                        ],
                        expert_count,
                    ),
                    Just(top_k),
                )
            })
    }

    /// Cas de routage augmentés d'une permutation aléatoire des experts.
    fn routing_cases_with_permutation() -> impl Strategy<Value = (Vec<f32>, usize, Vec<usize>)> {
        (
            prop_oneof![Just(8_usize), Just(64), Just(128)],
            prop_oneof![Just(1_usize), Just(4), Just(8)],
        )
            .prop_flat_map(|(expert_count, top_k)| {
                (
                    proptest::collection::vec(-1.0e4_f32..1.0e4, expert_count),
                    Just(top_k),
                    Just((0..expert_count).collect::<Vec<_>>()).prop_shuffle(),
                )
            })
    }

    proptest! {
        // (a) « norm_topk_prob » : les poids renormalisés sur le top-k somment à 1.
        #[test]
        fn topk_weights_sum_to_one((logits, top_k) in routing_cases()) {
            let weights = routing_moe(top_k)
                .top_weights(&logits)
                .expect("invariant: routage valide");
            prop_assert_eq!(weights.len(), top_k);
            let sum: f32 = weights.iter().map(|(_, weight)| weight).sum();
            prop_assert!((sum - 1.0).abs() <= 1.0e-4, "somme = {}", sum);
        }

        // (b) Les k sélectionnés dominent tout non-sélectionné, et sortent triés,
        // dans l'ordre exact du code : total_cmp décroissant, égalités par indice
        // croissant (tri stable).
        #[test]
        fn topk_selects_the_k_max_logits((logits, top_k) in routing_cases()) {
            let selected = routing_moe(top_k)
                .top_weights(&logits)
                .expect("invariant: routage valide")
                .into_iter()
                .map(|(expert_idx, _)| expert_idx)
                .collect::<Vec<_>>();
            let selected_set = selected
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            prop_assert_eq!(selected_set.len(), top_k, "indices dupliqués");
            for pair in selected.windows(2) {
                prop_assert!(
                    beats(&logits, pair[0], pair[1]),
                    "ordre interne violé: {:?}",
                    pair
                );
            }
            for outsider in (0..logits.len()).filter(|idx| !selected_set.contains(idx)) {
                for &winner in &selected {
                    prop_assert!(
                        beats(&logits, winner, outsider),
                        "l'expert {} devrait battre {}",
                        outsider,
                        winner
                    );
                }
            }
        }

        // (c) Permuter les logits permute la sélection, poids inchangés rang par
        // rang. Restreint aux logits DISTINCTS : les égalités se départagent par
        // indice, qui n'est pas permutation-invariant.
        #[test]
        fn permuting_logits_permutes_selection(
            (logits, top_k, perm) in routing_cases_with_permutation(),
        ) {
            prop_assume!(all_distinct(&logits));
            let moe = routing_moe(top_k);
            let base = moe.top_weights(&logits).expect("invariant: routage valide");
            let permuted_logits: Vec<f32> = perm.iter().map(|&src| logits[src]).collect();
            let permuted = moe
                .top_weights(&permuted_logits)
                .expect("invariant: routage valide");
            prop_assert_eq!(base.len(), permuted.len());
            for ((base_idx, base_weight), (perm_idx, perm_weight)) in
                base.iter().zip(permuted.iter())
            {
                prop_assert_eq!(perm[*perm_idx], *base_idx, "sélection non permutée");
                // Mêmes logits top-k dans le même ordre → softmax bit-identique.
                prop_assert_eq!(perm_weight, base_weight, "poids modifié par la permutation");
            }
        }

        // (d) Logits extrêmes (±1e4) : poids finis dans [0, 1], somme 1, zéro NaN.
        #[test]
        fn extreme_logits_yield_finite_weights(
            (logits, top_k) in extreme_routing_cases(),
        ) {
            let weights = routing_moe(top_k)
                .top_weights(&logits)
                .expect("invariant: routage valide");
            let mut sum = 0.0_f32;
            for (_, weight) in &weights {
                prop_assert!(weight.is_finite(), "poids non fini: {}", weight);
                prop_assert!((0.0..=1.0).contains(weight), "poids hors [0,1]: {}", weight);
                sum += weight;
            }
            prop_assert!((sum - 1.0).abs() <= 1.0e-4, "somme = {}", sum);
        }

        // per_expert_scale multiplie le poids softmax de l'expert sélectionné,
        // sans changer la sélection.
        #[test]
        fn per_expert_scale_multiplies_weights((logits, top_k) in routing_cases()) {
            let base = routing_moe(top_k)
                .top_weights(&logits)
                .expect("invariant: routage valide");
            let scales: Vec<f32> = (0..logits.len())
                .map(|expert_idx| 0.5 + (expert_idx % 4) as f32)
                .collect();
            let scaled_moe = routing_moe(top_k).with_per_expert_scale(
                Tensor::from_vec(vec![1, logits.len()], scales.clone())
                    .expect("invariant: scales valides"),
            );
            let scaled = scaled_moe
                .top_weights(&logits)
                .expect("invariant: routage scalé valide");
            prop_assert_eq!(base.len(), scaled.len());
            for ((base_idx, base_weight), (scaled_idx, scaled_weight)) in
                base.iter().zip(scaled.iter())
            {
                prop_assert_eq!(base_idx, scaled_idx, "sélection modifiée par l'échelle");
                prop_assert_eq!(
                    *scaled_weight,
                    base_weight * scales[*base_idx],
                    "échelle non appliquée"
                );
            }
        }
    }
}
