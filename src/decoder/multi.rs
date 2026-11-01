//! Pas de decode multi-flux dense (light-batch à N flux) : N tokens de N
//! séquences différentes dans UN command buffer.
//!
//! Les projections (q/k/v/o, in/out-proj linear-attn, MLP dense, lm_head) sont
//! batchées sur N lignes — les poids sont lus une fois par pas au lieu de N —
//! tandis que RoPE, append K/V, attention et conv/SSM restent des dispatches
//! PAR FLUX sur l'état de chaque flux, avec les kernels du solo. Les matmuls en
//! lignes ne sont pas bit-identiques au qmv mono-ligne du solo : la dérive
//! (~1e-6) est mesurée par l'oracle multi-flux, d'où l'activation explicite.

#[cfg(all(target_os = "macos", feature = "metal"))]
use super::*;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::decode_resident::MultiFullAttnStream;

#[cfg(all(target_os = "macos", feature = "metal"))]
impl CausalDecoder {
    /// Renvoie `true` si le pas multi-flux dense s'applique au cache résident :
    /// couches full-attn denses (porte de sortie, sans fenêtre glissante, sinks,
    /// normes Gemma ni MoE flint) et linear-attn denses, buffers résidents
    /// `FullDense`/`LinearDense` — exactement ceux que le pas encode.
    pub(crate) fn supports_resident_multi_dense(&self, cache: &CausalDecoderCache) -> bool {
        if self.forward_runtime().metal_executor().is_none() {
            return false;
        }
        let Some(arena) = cache.resident.as_ref() else {
            return false;
        };
        if self.config.metal_unsupported_reason().is_some()
            || !self.config.attn_output_gate
            || self.config.head_dim.is_none()
            || self.config.attention_sinks
        {
            return false;
        }
        self.layers.iter().enumerate().all(|(index, layer)| {
            if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
                return false;
            }
            if self.config.is_resident_full_attention_layer(index) {
                // `v_proj` absent (k_eq_v) : le solo prend la disposition `[q|k|v]`
                // sans porte, alors que le pas multi suppose `[q+gate|k|v]`.
                matches!(&layer.attention, AttentionBlock::Full(attention) if attention.v_proj.is_some())
                    && self.config.layer_sliding_window(index).is_none()
                    && matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::FullDense(resolved))
                            if resolved.parallel_moe_weights().is_none()
                                && resolved.sigmoid_moe_weights().is_none()
                                && resolved.pre_feedforward_norm.is_none()
                                && resolved.post_feedforward_norm.is_none()
                    )
            } else {
                matches!(&layer.attention, AttentionBlock::Linear(_))
                    && matches!(
                        arena.layers.get(index),
                        Some(ResidentLayerBuffers::LinearDense(_))
                    )
            }
        })
    }

    /// Décode UN token pour CHACUN des `caches.len()` flux (au moins 2) dans un
    /// command buffer unique, argmax on-device par ligne. Renvoie les tokens
    /// dans l'ordre des caches (ligne compacte = rang dans `caches`).
    ///
    /// Les poids et le scratch batché viennent de l'arène du premier flux ; le
    /// scratch par flux est namespacé par le slot de l'arène de chaque flux.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une précondition manque (arène, état par couche,
    /// dimensions, couche hors périmètre) ou si un encodage Metal échoue.
    pub(super) fn decode_tokens_resident_multi(
        &self,
        caches: &mut [&mut CausalDecoderCache],
        tokens: &[usize],
    ) -> Result<Vec<usize>> {
        let rows = caches.len();
        if rows < 2 || tokens.len() != rows {
            return Err(InferError::Dimension(format!(
                "decode multi: {rows} flux pour {} tokens (au moins 2)",
                tokens.len()
            )));
        }
        let Some(metal) = self.forward_runtime().metal_executor() else {
            return Err(InferError::Metal(
                "decode multi sans executor Metal".to_string(),
            ));
        };
        let theta = self
            .config
            .rope_theta
            .ok_or_else(|| InferError::Config("rope_theta manquant (decode multi)".to_string()))?;
        let eps = self.config.rms_eps;
        let hidden = self.final_norm.data().len();
        let linear_dims = if self.has_resident_linear_attention_layer() {
            let la_config = self.config.linear_attention_config()?;
            let la_spec = LinearAttentionStepSpec {
                num_key_heads: la_config.num_key_heads,
                num_value_heads: la_config.num_value_heads,
                key_head_dim: la_config.key_head_dim,
                value_head_dim: la_config.value_head_dim,
                conv_kernel_dim: la_config.conv_kernel_dim,
                rms_eps: la_config.rms_eps,
            };
            let key_dim = la_config.key_dim()?;
            let value_dim = la_config.value_dim()?;
            let conv_dim = key_dim
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(value_dim))
                .ok_or_else(|| InferError::Shape("conv_dim déborde (decode multi)".to_string()))?;
            Some((
                la_spec,
                LinearAttnResidentDims {
                    in_dim: hidden,
                    conv_dim,
                    value_dim,
                    key_dim,
                },
            ))
        } else {
            None
        };

        // Borrows disjoints par flux : états par couche (mutables) vs arène.
        let mut parts = Vec::with_capacity(rows);
        for cache in caches.iter_mut() {
            let CausalDecoderCache {
                layers,
                position,
                resident,
            } = &mut **cache;
            let arena = resident
                .as_ref()
                .ok_or_else(|| InferError::Metal("arène résidente absente (multi)".to_string()))?;
            parts.push((layers, position, arena));
        }
        let slots: Vec<u64> = parts
            .iter()
            .map(|(_, _, arena)| arena.state.scratch_namespace())
            .collect();
        let arena = parts
            .first()
            .map(|(_, _, arena)| *arena)
            .ok_or_else(|| InferError::Metal("decode multi sans flux".to_string()))?;

        let embed = self.embed_scaled(tokens)?;
        let (batch, embed_hidden) = embed.as_matrix()?;
        if batch != rows || embed_hidden != hidden {
            return Err(InferError::Dimension(format!(
                "embedding multi [{batch},{embed_hidden}], attendu [{rows},{hidden}]"
            )));
        }
        let elements = rows
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("decode multi hidden déborde".to_string()))?;
        let rows_a = arena.state.scratch().lease(elements, GpuElement::F32)?;
        let rows_b = arena.state.scratch().lease(elements, GpuElement::F32)?;
        arena.state.upload(rows_a.tensor(), embed.data())?;

        let command_buffer = arena.state.queue().new_command_buffer();
        let _barrier_guard = crate::metal_backend::resident_concurrent_enabled()
            .then(crate::metal_backend::install_dispatch_barrier_scope);
        let _namespace_guard =
            crate::metal_backend::install_scratch_namespace(arena.state.scratch_namespace());
        let encoder = new_resident_compute_encoder(command_buffer);
        let encoder_guard = crate::metal_backend::EncoderEndGuard::new(encoder);
        let mut owned: Vec<metal::Buffer> = Vec::new();

        let mut current = &rows_a;
        let mut other = &rows_b;
        for (index, layer) in self.layers.iter().enumerate() {
            if !matches!(layer.mlp.as_ref(), Some(FeedForward::Dense(_))) {
                return Err(InferError::Config(format!(
                    "decode multi: MLP dense attendu couche {index}"
                )));
            }
            match arena.layers.get(index) {
                Some(ResidentLayerBuffers::FullDense(resolved))
                    if self.config.is_resident_full_attention_layer(index) =>
                {
                    let AttentionBlock::Full(attention) = &layer.attention else {
                        return Err(InferError::Config(format!(
                            "decode multi: couche full-attn {index} attendue"
                        )));
                    };
                    let weights = FullAttnDenseLayerWeights {
                        input_norm: &resolved.input_norm,
                        qkv_proj: resolved.qkv_proj.as_ref(),
                        qkv_proj_without_gate: attention.v_proj.is_none()
                            || self.config.attention_sinks,
                        value_norm: resolved.value_norm,
                        q_proj: &resolved.q_proj,
                        k_proj: &resolved.k_proj,
                        v_proj: &resolved.v_proj,
                        o_proj: &resolved.o_proj,
                        q_norm: &resolved.q_norm,
                        k_norm: &resolved.k_norm,
                        post_norm: &resolved.post_norm,
                        pre_feedforward_norm: resolved.pre_feedforward_norm.as_ref(),
                        post_feedforward_norm: resolved.post_feedforward_norm.as_ref(),
                        layer_scalar: resolved.layer_scalar,
                        gate_proj: &resolved.gate_proj,
                        up_proj: &resolved.up_proj,
                        down_proj: &resolved.down_proj,
                        tail_score: &arena.dense_tail_score,
                        parallel_moe: resolved.parallel_moe_weights(),
                        sigmoid_moe: resolved.sigmoid_moe_weights(),
                    };
                    let mut streams = Vec::with_capacity(rows);
                    for ((layers, position, _), slot) in parts.iter_mut().zip(&slots) {
                        let kv = layers[index].full.as_mut().ok_or_else(|| {
                            InferError::Metal("KV full-attn absent (multi)".to_string())
                        })?;
                        let dims = self.config.resident_windowed_full_attn_layer_dims(
                            index, hidden, **position, eps, theta,
                        )?;
                        streams.push(MultiFullAttnStream {
                            kv,
                            dims,
                            slot: *slot,
                        });
                    }
                    arena.state.encode_full_attn_dense_layer_multi(
                        metal,
                        encoder_guard.encoder(),
                        &mut owned,
                        &mut streams,
                        weights,
                        current.tensor().buffer(),
                        other.tensor().buffer(),
                    )?;
                }
                Some(ResidentLayerBuffers::LinearDense(resolved))
                    if !self.config.is_resident_full_attention_layer(index) =>
                {
                    let Some((la_spec, res_dims)) = linear_dims else {
                        return Err(InferError::Config(
                            "dims linear-attn multi absentes".to_string(),
                        ));
                    };
                    let states = parts
                        .iter()
                        .map(|(layers, _, _)| {
                            layers[index].linear.metal_state().ok_or_else(|| {
                                InferError::Metal("état linear-attn absent (multi)".to_string())
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let weights = LinearAttnDenseLayerWeights {
                        input_norm: &resolved.input_norm,
                        linear: &resolved.linear,
                        post_norm: &resolved.post_norm,
                        gate_proj: &resolved.gate_proj,
                        up_proj: &resolved.up_proj,
                        down_proj: &resolved.down_proj,
                        tail_score: &arena.dense_tail_score,
                    };
                    arena.state.encode_linear_attn_dense_layer_multi(
                        metal,
                        encoder_guard.encoder(),
                        &mut owned,
                        &states,
                        &slots,
                        weights,
                        la_spec,
                        res_dims,
                        hidden,
                        eps,
                        current.tensor().buffer(),
                        other.tensor().buffer(),
                    )?;
                }
                _ => {
                    return Err(InferError::Config(format!(
                        "decode multi: buffers résidents denses absents couche {index}"
                    )));
                }
            }
            std::mem::swap(&mut current, &mut other);
        }

        let final_rows = arena.state.scratch().lease(elements, GpuElement::F32)?;
        metal.encode_rms_norm_rows(
            encoder_guard.encoder(),
            current.tensor().buffer(),
            &arena.final_norm,
            final_rows.tensor().buffer(),
            rows,
            hidden,
            eps,
        )?;
        let indices = arena.state.scratch().lease(rows, GpuElement::U32)?;
        metal.encode_lm_head_argmax_rows_buffers(
            encoder_guard.encoder(),
            final_rows.tensor().buffer(),
            &arena.lm_head,
            indices.tensor().buffer(),
            0,
            0,
            rows,
            hidden,
        )?;
        encoder_guard.end();
        crate::metal_backend::commit_and_wait(command_buffer)?;
        drop(owned);

        let raw = crate::metal_backend::read_u32_buffer(indices.tensor().buffer(), rows)?;
        if raw.len() != rows {
            return Err(InferError::Metal(format!(
                "decode multi: {} tokens relus pour {rows} flux",
                raw.len()
            )));
        }
        for (_, position, _) in &mut parts {
            **position += 1;
        }
        raw.into_iter()
            .map(|token| {
                usize::try_from(token)
                    .map_err(|_| InferError::Metal(format!("token multi hors usize: {token}")))
            })
            .collect()
    }
}

/// Oracles multi-flux sur le modèle dense hybride réel (Qwen3.8-27B) :
/// `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` pointe vers ses poids. Mode strict
/// conseillé : `SARAGOSSA_RUST_QMM_SKINNY=0`.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;

    fn real_model() -> Option<CausalDecoder> {
        let model_dir = crate::runtime_flags::env_var_os("SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .filter(|path| path.is_dir());
        let model_dir = crate::test_support::require_real_model(
            model_dir,
            "SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL pointant vers le modèle dense hybride",
        )?;
        let assets =
            crate::ModelAssets::load_local(&model_dir).expect("invariant: assets du modèle réel");
        Some(
            crate::load_causal_decoder(&assets)
                .and_then(CausalDecoder::with_metal_runtime)
                .expect("invariant: modèle réel chargé avec Metal"),
        )
    }

    /// Prompt synthétique déterministe de `len` tokens (ids du vocabulaire courant).
    fn prompt(seed: usize, len: usize) -> Vec<usize> {
        (0..len)
            .map(|index| 1_000 + (seed * 7_919 + index * 104_729) % 30_000)
            .collect()
    }

    fn first_divergence(batch: &[usize], solo: &[usize]) -> Option<usize> {
        (batch != solo).then(|| {
            batch
                .iter()
                .zip(solo)
                .position(|(left, right)| left != right)
                .unwrap_or(batch.len().min(solo.len()))
        })
    }

    /// Gate 1 : N flux au prompt identique sortent les mêmes octets (aucune
    /// fuite d'état entre flux), par le pas multi-flux réellement exercé.
    #[test]
    #[ignore = "modèle dense hybride réel requis"]
    fn multi_dense_identical_prompts_give_identical_streams() {
        let Some(model) = real_model() else {
            return;
        };
        let options = GenerationOptions::default();
        for streams in [2_usize, 4, 8] {
            let prompts = vec![prompt(0, 48); streams];
            let (outputs, profile) = model
                .generate_greedy_lightbatch_profiled(&prompts, &vec![32; streams], &options)
                .expect("invariant: light-batch multi");
            assert!(profile.multi_steps > 0, "N={streams}: pas multi non exercé");
            for (stream, output) in outputs.iter().enumerate() {
                assert_eq!(
                    output.tokens, outputs[0].tokens,
                    "N={streams} flux {stream}"
                );
            }
        }
    }

    /// Gate 2 : prompts de longueurs différentes, chaque flux comparé à son
    /// solo. La première divergence par flux est imprimée ; une divergence avant
    /// le 16e token est un bug (une dérive near-tie tardive est tolérée).
    #[test]
    #[ignore = "modèle dense hybride réel requis"]
    fn multi_dense_streams_track_their_solo() {
        let Some(model) = real_model() else {
            return;
        };
        let options = GenerationOptions::default();
        let bases = [prompt(1, 64), prompt(2, 40)];
        let max_new = 48;
        for streams in [2_usize, 4, 8] {
            let prompts: Vec<Vec<usize>> = (0..streams)
                .map(|stream| bases[stream % 2][stream..].to_vec())
                .collect();
            let (outputs, profile) = model
                .generate_greedy_lightbatch_profiled(&prompts, &vec![max_new; streams], &options)
                .expect("invariant: light-batch multi");
            assert!(profile.multi_steps > 0, "N={streams}: pas multi non exercé");
            for (stream, (output, prompt)) in outputs.iter().zip(&prompts).enumerate() {
                let solo = model
                    .generate_greedy_timed_with_options(prompt, max_new, &options)
                    .expect("invariant: solo")
                    .tokens;
                let divergence = first_divergence(&output.tokens, &solo);
                eprintln!("multi N={streams} flux={stream} première_divergence={divergence:?}");
                // Aucune divergence observée (défaut et QMM_SKINNY=0) : l'oracle exige
                // l'identité ; une dérive de kernel future devra être justifiée par
                // un quasi ex aequo mesuré, pas absorbée par une tolérance d'indice.
                assert!(
                    divergence.is_none(),
                    "N={streams} flux {stream}: divergence {divergence:?}"
                );
            }
        }
    }

    /// Transitions 5 → 4 → 3 → 2 → 1 : budgets distincts (le flux 0 finit le
    /// premier, fins non contiguës) et un stop-token ; chaque flux suit son solo
    /// avec le même budget et N décroît pas à pas jusqu'au solo.
    #[test]
    #[ignore = "modèle dense hybride réel requis"]
    fn multi_dense_shrinks_as_streams_finish() {
        let Some(model) = real_model() else {
            return;
        };
        let prompts: Vec<Vec<usize>> = (0..5).map(|stream| prompt(10 + stream, 32)).collect();
        let budgets = [3_usize, 14, 6, 20, 9];
        // Stop-token tiré du solo du flux 3 : il l'arrête avant son budget.
        let free = model
            .generate_greedy_timed_with_options(&prompts[3], budgets[3], &Default::default())
            .expect("invariant: solo libre")
            .tokens;
        let options = GenerationOptions {
            stop_token_ids: vec![free[11]],
            ..GenerationOptions::default()
        };
        let (outputs, profile) = model
            .generate_greedy_lightbatch_profiled(&prompts, &budgets, &options)
            .expect("invariant: light-batch multi");
        assert!(profile.multi_steps > 0, "pas multi non exercé");
        let widths: Vec<usize> = profile.steps.iter().map(|(width, _)| *width).collect();
        eprintln!("multi transitions largeurs={widths:?}");
        assert!(
            widths.windows(2).all(|pair| pair[1] <= pair[0]),
            "la largeur ne doit que décroître: {widths:?}"
        );
        assert_eq!(widths.first(), Some(&5), "{widths:?}");
        assert_eq!(widths.last(), Some(&1), "{widths:?}");
        for (stream, (output, prompt)) in outputs.iter().zip(&prompts).enumerate() {
            let solo = model
                .generate_greedy_timed_with_options(prompt, budgets[stream], &options)
                .expect("invariant: solo")
                .tokens;
            assert_eq!(output.tokens.len(), solo.len(), "flux {stream}: longueur");
            assert!(
                first_divergence(&output.tokens, &solo).is_none_or(|index| index >= 16),
                "flux {stream}: {:?} != {solo:?}",
                output.tokens
            );
        }
    }
}
