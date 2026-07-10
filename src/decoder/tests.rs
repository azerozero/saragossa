use super::attention_ops::{
    apply_rope_heads_at, attention_layout, cached_attention_one, causal_attention,
    full_attention_from_tensors, rms_norm_heads, rms_norm_rope_heads_at, AttentionLayout,
    RopeParams,
};
use super::*;
use safetensors::{serialize, Dtype, View};
use std::borrow::Cow;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
use std::path::{Path, PathBuf};

#[test]
fn causal_decoder_generates_expected_token() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward décodeur valide");
    assert_eq!(logits.shape(), &[1, 3]);
    let generated = model
        .generate_greedy(&[0], 2)
        .expect("invariant: greedy valide");
    assert_eq!(generated, vec![0, 0]);
}

#[test]
fn causal_decoder_loads_from_safetensors() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let buffer =
        serialize(tensor_views(test_weights()), None).expect("invariant: safetensors sérialisable");
    std::fs::write(tmp.path(), buffer).expect("invariant: écriture temporaire");
    let model = CausalDecoder::from_safetensors(tmp.path(), CausalDecoderConfig::default())
        .expect("invariant: safetensors chargeable");
    let generated = model
        .generate_greedy(&[0], 1)
        .expect("invariant: greedy valide");
    assert_eq!(generated, vec![0]);
}

#[test]
fn causal_decoder_applies_optional_mlp_block() {
    let base = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let mut weights = test_weights();
    insert_identity_mlp(&mut weights);
    let with_mlp = CausalDecoder::from_tensors(weights, CausalDecoderConfig::default())
        .expect("invariant: poids MLP valides");

    let base_logits = base
        .next_logits(&[0])
        .expect("invariant: forward attention valide");
    let mlp_logits = with_mlp
        .next_logits(&[0])
        .expect("invariant: forward MLP valide");
    assert_ne!(base_logits.data(), mlp_logits.data());
}

#[test]
#[cfg(feature = "devtools")]
fn dflash_acceptance_matches_autoregressive_greedy() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let draft = tiny_dflash_draft();
    let options = GenerationOptions::default();
    let prompt = [0_usize, 1];

    let ar = model
        .generate_greedy_cached_with_options(&prompt, 5, &options)
        .expect("invariant: AR greedy valide");
    let spec = model
        .generate_greedy_dflash_batched_with_options(&prompt, 5, &options, &draft, 2)
        .expect("invariant: DFlash greedy valide");

    assert_eq!(spec.tokens, ar);
    assert!(spec.stats.proposed > 0);
    assert!(spec.stats.verifications > 0);
}

#[test]
fn causal_decoder_supports_grouped_query_attention() {
    let config = CausalDecoderConfig {
        rope_theta: None,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let model =
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: poids GQA cohérents");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward GQA valide");
    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn causal_decoder_applies_qk_norm_when_present() {
    let base = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let mut weights = test_weights();
    insert_qk_norm(&mut weights, 0, vec![2.0, 0.25], vec![0.25, 2.0]);
    let with_qk_norm = CausalDecoder::from_tensors(weights, CausalDecoderConfig::default())
        .expect("invariant: q_norm/k_norm valides");

    let base_logits = base
        .next_logits(&[0, 1])
        .expect("invariant: forward sans qk norm valide");
    let qk_logits = with_qk_norm
        .next_logits(&[0, 1])
        .expect("invariant: forward avec qk norm valide");

    assert_ne!(base_logits.data(), qk_logits.data());
    assert!(qk_logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn causal_decoder_applies_attention_output_gate_when_enabled() {
    let config = CausalDecoderConfig {
        attn_output_gate: true,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let gated = CausalDecoder::from_tensors(gated_attention_weights(), config)
        .expect("invariant: attention gated valide");
    let base = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: attention non gated valide");

    let gated_logits = gated
        .next_logits(&[0, 1])
        .expect("invariant: forward gated valide");
    let base_logits = base
        .next_logits(&[0, 1])
        .expect("invariant: forward base valide");

    assert_ne!(gated_logits.data(), base_logits.data());
    assert!(gated_logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn causal_decoder_stacks_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let mut weights = test_weights();
        for layer in 1..layer_count {
            insert_attention_layer(&mut weights, layer);
        }
        let config = CausalDecoderConfig {
            num_hidden_layers: layer_count,
            ..CausalDecoderConfig::default()
        };
        let model = CausalDecoder::from_tensors(weights, config)
            .expect("invariant: pile multi-couches valide");
        let logits = model
            .next_logits(&[0, 1])
            .expect("invariant: forward multi-couches valide");

        assert_eq!(logits.shape(), &[1, 3]);
        assert!(logits.data().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn cached_greedy_matches_full_sequence_for_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let mut weights = test_weights();
        for layer in 1..layer_count {
            insert_attention_layer(&mut weights, layer);
        }
        let config = CausalDecoderConfig {
            num_hidden_layers: layer_count,
            ..CausalDecoderConfig::default()
        };
        let model = CausalDecoder::from_tensors(weights, config)
            .expect("invariant: pile multi-couches valide");

        let full = model
            .generate_greedy_full_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy complet valide");
        let cached = model
            .generate_greedy_cached_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy cache valide");

        assert_eq!(cached, full, "layer_count={layer_count}");
    }
}

#[test]
fn hybrid_cached_greedy_matches_full_sequence_for_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let config = hybrid_config(layer_count);
        let model = CausalDecoder::from_tensors(hybrid_weights(layer_count, 2), config)
            .expect("invariant: pile hybride valide");

        let full = model
            .generate_greedy_full_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy complet hybride valide");
        let cached = model
            .generate_greedy_cached_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy cache hybride valide");

        assert_eq!(cached, full, "layer_count={layer_count}");
    }
}

#[test]
fn snapshot_prompt_state_greedy_matches_cold_prefill() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let prompt = [0_usize, 1, 2, 1, 0];
    let options = GenerationOptions::default();
    let cold = model
        .generate_greedy_timed_with_options(&prompt, 4, &options)
        .expect("invariant: génération froide valide")
        .tokens;

    let mut state = model
        .prefill_prompt_state_uncached(&prompt[..2])
        .expect("invariant: snapshot préfixe valide");
    model
        .extend_prompt_state(&mut state, &prompt[2..])
        .expect("invariant: extension suffixe valide");
    let warm = model
        .generate_greedy_timed_from_prompt_state_with_options(state, Duration::ZERO, 4, &options)
        .expect("invariant: génération depuis snapshot valide")
        .tokens;

    assert_eq!(warm, cold);
}

#[test]
fn hybrid_snapshot_prompt_state_greedy_matches_cold_prefill() {
    let config = hybrid_config(4);
    let model =
        CausalDecoder::from_tensors(hybrid_weights(4, 2), config).expect("invariant: hybride");
    let prompt = [0_usize, 1, 2, 1, 0, 2];
    let options = GenerationOptions::default();
    let cold = model
        .generate_greedy_timed_with_options(&prompt, 4, &options)
        .expect("invariant: génération froide hybride valide")
        .tokens;

    let mut state = model
        .prefill_prompt_state_uncached(&prompt[..3])
        .expect("invariant: snapshot préfixe hybride valide");
    model
        .extend_prompt_state(&mut state, &prompt[3..])
        .expect("invariant: extension suffixe hybride valide");
    let warm = model
        .generate_greedy_timed_from_prompt_state_with_options(state, Duration::ZERO, 4, &options)
        .expect("invariant: génération hybride depuis snapshot valide")
        .tokens;

    assert_eq!(warm, cold);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn hybrid_cloned_prompt_state_metal_matches_cold_prefill() {
    let config = CausalDecoderConfig {
        head_dim: Some(2),
        ..hybrid_config(4)
    };
    let model =
        CausalDecoder::from_tensors(hybrid_weights(4, 2), config).expect("invariant: hybride");
    let model = match model.with_metal_runtime() {
        Ok(model) => model,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return,
        Err(error) => panic!("runtime Metal indisponible: {error:?}"),
    };
    let prompt = [0_usize, 1, 2, 1, 0, 2];
    let cold_state = model
        .prefill_prompt_state_uncached(&prompt)
        .expect("invariant: prefill froid hybride Metal valide");

    let prefix_state = model
        .prefill_prompt_state_uncached(&prompt[..3])
        .expect("invariant: snapshot préfixe hybride Metal valide");
    assert!(
        prefix_state
            .cache
            .layers
            .iter()
            .any(|layer| layer.linear.metal_state().is_some()),
        "le préfill hybride Metal doit porter un état GDN résident"
    );
    let snapshot = model
        .snapshot_prompt_state_metal(&prefix_state)
        .expect("invariant: snapshot Metal hybride valide");
    assert!(
        snapshot.estimated_bytes() > 0,
        "le snapshot hybride Metal doit capturer l'état récurrent GDN"
    );
    let mut warm_state = prefix_state.clone();
    let snapshot = model
        .copy_prompt_state_metal_snapshot(&snapshot)
        .expect("invariant: copie snapshot Metal hybride valide");
    model
        .restore_prompt_state_metal(&mut warm_state, snapshot)
        .expect("invariant: restore snapshot Metal hybride valide");
    model
        .extend_prompt_state(&mut warm_state, &prompt[3..])
        .expect("invariant: extension suffixe hybride Metal valide");

    assert_eq!(warm_state.final_state.data(), cold_state.final_state.data());
}

#[test]
fn hybrid_batched_prefill_matches_tokenwise_prefill_state() {
    let config = hybrid_config(4);
    let model =
        CausalDecoder::from_tensors(hybrid_weights(4, 2), config).expect("invariant: hybride");
    let prompt = [0_usize, 1, 2, 1, 0, 2];

    let (mut tokenwise_cache, tokenwise_state) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill tokenwise valide");
    let (mut batched_cache, batched_state) = model
        .prefill_cache_state_batched_for_test(&prompt)
        .expect("invariant: prefill batched hybride valide");

    assert_eq!(tokenwise_cache.position(), prompt.len());
    assert_eq!(batched_cache.position(), prompt.len());
    assert_close(tokenwise_state.data(), batched_state.data(), 1.0e-5);

    let tokenwise_next = model
        .next_final_state_cached(&mut tokenwise_cache, 1)
        .expect("invariant: next token tokenwise valide");
    let batched_next = model
        .next_final_state_cached(&mut batched_cache, 1)
        .expect("invariant: next token batched valide");

    assert_close(tokenwise_next.data(), batched_next.data(), 1.0e-5);
}

#[test]
fn hybrid_chunked_batched_prefill_matches_tokenwise_prefill_state() {
    let config = hybrid_config(5);
    let model =
        CausalDecoder::from_tensors(hybrid_weights(5, 2), config).expect("invariant: hybride");
    let prompt = [0_usize, 1, 2, 1, 0, 2, 2, 1];
    let (tokenwise_cache, tokenwise_state) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill tokenwise valide");

    for chunk_size in [1_usize, 2, 3, 5] {
        let (mut chunked_cache, chunked_state) = model
            .prefill_cache_state_batched_chunked_for_test(&prompt, chunk_size)
            .expect("invariant: prefill chunked hybride valide");

        assert_eq!(chunked_cache.position(), prompt.len());
        assert_close(tokenwise_state.data(), chunked_state.data(), 1.0e-5);

        let mut tokenwise_next_cache = tokenwise_cache.clone();
        let tokenwise_next = model
            .next_final_state_cached(&mut tokenwise_next_cache, 1)
            .expect("invariant: next token tokenwise valide");
        let chunked_next = model
            .next_final_state_cached(&mut chunked_cache, 1)
            .expect("invariant: next token chunked valide");

        assert_close(tokenwise_next.data(), chunked_next.data(), 1.0e-5);
    }
}

#[test]
fn cached_next_logits_match_full_prefix_logits() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let mut cache = model.empty_cache();
    let tokens = [0_usize, 1, 2];

    for prefix_len in 1..=3 {
        let prompt = &tokens[..prefix_len];
        let cached = model
            .next_logits_cached(&mut cache, prompt[prefix_len - 1])
            .expect("invariant: forward cache valide");
        let full = model
            .next_logits(prompt)
            .expect("invariant: forward complet valide");

        assert_close(cached.data(), full.data(), 1.0e-5);
        assert_eq!(cache.position(), prefix_len);
        assert_eq!(cache.layer_count(), 1);
    }
}

#[test]
fn prefix_cache_selects_longest_append_prefix() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let final_state = Tensor::row(vec![0.0]).expect("invariant: tenseur test valide");
    let entries = vec![
        PrefixCacheEntry {
            tokens: vec![1, 2],
            cache: model.empty_cache(),
            final_state: final_state.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            linear_metal: Vec::new(),
        },
        PrefixCacheEntry {
            tokens: vec![1, 2, 3],
            cache: model.empty_cache(),
            final_state: final_state.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            linear_metal: Vec::new(),
        },
        PrefixCacheEntry {
            tokens: vec![1],
            cache: model.empty_cache(),
            final_state,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            linear_metal: Vec::new(),
        },
    ];

    assert_eq!(
        generation::longest_prefix_entry_index(&entries, &[1, 2, 3, 4]),
        Some(1)
    );
    assert_eq!(
        generation::longest_prefix_entry_index(&entries, &[1, 9]),
        Some(2)
    );
    assert_eq!(generation::longest_prefix_entry_index(&entries, &[9]), None);
}

#[test]
fn prefix_cache_extends_append_suffix_with_batched_prefill() {
    let config = hybrid_config(4);
    let weights = hybrid_weights(4, 2);
    let prefix = [0_usize, 1, 2];
    let prompt = [0_usize, 1, 2, 1, 0, 2, 2];

    let state_model = CausalDecoder::from_tensors(weights.clone(), config.clone())
        .expect("invariant: pile hybride valide");
    let _ = state_model
        .prefill_cache_state(&prefix)
        .expect("invariant: prefix cache préchauffé");
    let (cached_cache, cached_state) = state_model
        .prefill_cache_state(&prompt)
        .expect("invariant: suffixe prefix-cache valide");
    let (batched_cache, batched_state) = state_model
        .prefill_cache_state_batched_for_test(&prompt)
        .expect("invariant: prefill batché référence valide");

    assert_eq!(cached_cache.position(), prompt.len());
    assert_eq!(batched_cache.position(), prompt.len());
    assert_close(cached_state.data(), batched_state.data(), 1.0e-5);

    let token_model =
        CausalDecoder::from_tensors(weights, config).expect("invariant: pile hybride valide");
    let _ = token_model
        .prefill_cache_state(&prefix)
        .expect("invariant: prefix cache préchauffé");
    let cached = token_model
        .generate_greedy_cached_with_options(&prompt, 5, &GenerationOptions::default())
        .expect("invariant: greedy cache suffixe valide");
    let full = token_model
        .generate_greedy_full_with_options(&prompt, 5, &GenerationOptions::default())
        .expect("invariant: greedy complet valide");

    assert_eq!(cached, full);
}

#[test]
fn cached_greedy_matches_full_sequence_with_grouped_query_attention() {
    let config = CausalDecoderConfig {
        rope_theta: Some(10_000.0),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let model =
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: poids GQA cohérents");

    let full = model
        .generate_greedy_full_with_options(&[0, 1], 3, &GenerationOptions::default())
        .expect("invariant: greedy complet GQA valide");
    let cached = model
        .generate_greedy_cached_with_options(&[0, 1], 3, &GenerationOptions::default())
        .expect("invariant: greedy cache GQA valide");

    assert_eq!(cached, full);
}

#[test]
fn cached_sampling_is_seed_deterministic() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let options = GenerationOptions {
        temperature: 0.8,
        top_p: 1.0,
        seed: 123,
        ..GenerationOptions::default()
    };

    let left = model
        .generate_greedy_cached_with_options(&[0, 1], 4, &options)
        .expect("invariant: sampling cache valide");
    let right = model
        .generate_greedy_cached_with_options(&[0, 1], 4, &options)
        .expect("invariant: sampling cache valide");

    assert_eq!(left, right);
    assert_eq!(left.len(), 4);
}

#[test]
fn cached_decoder_stops_before_emitting_stop_token() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let generated = model
        .generate_greedy_cached_with_options(
            &[0],
            4,
            &GenerationOptions {
                stop_token_ids: vec![0],
                ..GenerationOptions::default()
            },
        )
        .expect("invariant: greedy cache avec stop valide");
    assert!(generated.is_empty());
}

#[test]
fn cached_decoder_stops_after_emitting_stop_sequence() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let generated = model
        .generate_greedy_cached_with_options(
            &[0],
            4,
            &GenerationOptions {
                stop_sequences: vec![vec![0, 0]],
                ..GenerationOptions::default()
            },
        )
        .expect("invariant: greedy cache avec séquence stop valide");
    assert_eq!(generated, vec![0, 0]);
}

#[test]
fn causal_decoder_rejects_empty_prompt() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let err = model
        .generate_greedy(&[], 1)
        .expect_err("invariant: prompt vide rejeté");
    assert!(matches!(err, InferError::Dimension(_)));
}

#[test]
fn causal_decoder_stops_before_emitting_stop_token() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let generated = model
        .generate_greedy_with_options(
            &[0],
            4,
            &GenerationOptions {
                stop_token_ids: vec![0],
                ..GenerationOptions::default()
            },
        )
        .expect("invariant: greedy avec stop valide");
    assert!(generated.is_empty());
}

#[test]
fn layer_overrides_follow_gemma3_sliding_pattern() {
    let config = CausalDecoderConfig {
        rope_theta: Some(1_000_000.0),
        rope_local_base_freq: Some(10_000.0),
        sliding_window: Some(512),
        sliding_window_pattern: Some(6),
        ..CausalDecoderConfig::default()
    };
    // Couches locales : toutes sauf la 6e du motif (indices 5, 11, 17, 23…).
    for layer in [0_usize, 1, 4, 6, 10, 24] {
        assert_eq!(config.layer_rope_theta_override(layer), Some(10_000.0));
        assert_eq!(config.layer_sliding_window(layer), Some(512));
    }
    for layer in [5_usize, 11, 17, 23] {
        assert_eq!(config.layer_rope_theta_override(layer), None);
        assert_eq!(config.layer_sliding_window(layer), None);
    }
}

#[test]
fn layer_overrides_absent_without_sliding_pattern() {
    let config = CausalDecoderConfig::default();
    for layer in 0..8 {
        assert_eq!(config.layer_rope_theta_override(layer), None);
        assert_eq!(config.layer_sliding_window(layer), None);
        assert_eq!(config.layer_rope_position_scale(layer), None);
    }
}

#[test]
fn layer_rope_position_scale_targets_global_layers_only() {
    // Gemma 3 4B : rope_scaling linear ×8 appliqué aux SEULES couches globales
    // (la 6e du motif), les locales gardent leurs positions brutes sur
    // rope_local_base_freq — même câblage que gemma3_text.py de mlx_lm.
    let config = CausalDecoderConfig {
        rope_theta: Some(1_000_000.0),
        rope_local_base_freq: Some(10_000.0),
        rope_position_scale: Some(0.125),
        sliding_window: Some(1024),
        sliding_window_pattern: Some(6),
        ..CausalDecoderConfig::default()
    };
    for layer in [0_usize, 1, 4, 6, 10, 24] {
        assert_eq!(config.layer_rope_position_scale(layer), None);
        assert_eq!(config.layer_rope_theta_override(layer), Some(10_000.0));
    }
    for layer in [5_usize, 11, 17, 23] {
        assert_eq!(config.layer_rope_position_scale(layer), Some(0.125));
        assert_eq!(config.layer_rope_theta_override(layer), None);
    }
    // Linear historique (Llama 2) : pas de motif sliding → toutes les couches.
    let uniform = CausalDecoderConfig {
        rope_position_scale: Some(0.25),
        ..CausalDecoderConfig::default()
    };
    for layer in 0..8 {
        assert_eq!(uniform.layer_rope_position_scale(layer), Some(0.25));
    }
}

#[test]
fn gemma_layer_wiring_matches_manual_reference() {
    // Câblage Gemma : norme post-attention AVANT le résiduel, double norme
    // pre/post feed-forward, MLP GeGLU, embeddings mis à l'échelle.
    let post_attn = Tensor::from_vec(vec![2], vec![0.5, 1.5]).expect("invariant: norm valide");
    let pre_ffn = Tensor::from_vec(vec![2], vec![1.25, 0.75]).expect("invariant: norm valide");
    let post_ffn = Tensor::from_vec(vec![2], vec![2.0, 0.5]).expect("invariant: norm valide");
    let mut weights = test_weights();
    weights.insert(
        "layers.0.post_attention_layernorm.weight".to_string(),
        post_attn.clone(),
    );
    weights.insert(
        "layers.0.pre_feedforward_layernorm.weight".to_string(),
        pre_ffn.clone(),
    );
    weights.insert(
        "layers.0.post_feedforward_layernorm.weight".to_string(),
        post_ffn.clone(),
    );
    for prefix in [
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
    ] {
        weights.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
    let config = CausalDecoderConfig {
        embed_scale: Some(2.0),
        activation: crate::Activation::GeluTanh,
        ..CausalDecoderConfig::default()
    };
    let eps = config.rms_eps;
    let model =
        CausalDecoder::from_tensors(weights, config).expect("invariant: poids Gemma valides");
    let logits = model
        .next_logits(&[2])
        .expect("invariant: forward Gemma valide");

    // Référence manuelle : projections identité + prompt d'un seul token à la
    // position 0 → la sortie d'attention vaut exactement l'entrée normalisée.
    let ones = Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide");
    let x = Tensor::from_vec(vec![1, 2], vec![2.0, 2.0]).expect("invariant: embed ×2 valide");
    let attn_out = rms_norm(&x, &ones, eps).expect("invariant: rms valide");
    let hidden = x
        .add(&rms_norm(&attn_out, &post_attn, eps).expect("invariant: rms valide"))
        .expect("invariant: add valide");
    let ffn_in = rms_norm(&hidden, &pre_ffn, eps).expect("invariant: rms valide");
    let ffn_out = crate::gelu_tanh(&ffn_in)
        .mul_elementwise(&ffn_in)
        .expect("invariant: GeGLU valide");
    let y = hidden
        .add(&rms_norm(&ffn_out, &post_ffn, eps).expect("invariant: rms valide"))
        .expect("invariant: add valide");
    let final_state = rms_norm(&y, &ones, eps).expect("invariant: rms valide");
    let final_row = final_state.as_row().expect("invariant: ligne valide");
    let lm_head = [[1.0_f32, 0.0], [-1.0, 0.0], [0.0, 1.0]];
    let expected = lm_head
        .iter()
        .map(|row| row[0] * final_row[0] + row[1] * final_row[1])
        .collect::<Vec<_>>();
    assert_close(
        logits.as_row().expect("invariant: logits ligne"),
        &expected,
        1.0e-6,
    );
}

#[test]
fn causal_attention_sliding_window_matches_truncated_context() {
    let full_layout = AttentionLayout {
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        rope_dims: 2,
        attn_scalar: 2.0,
        sliding_window: None,
    };
    let windowed_layout = AttentionLayout {
        sliding_window: Some(2),
        ..full_layout
    };
    let q = Tensor::from_vec(vec![4, 2], vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0])
        .expect("invariant: q valide");
    let k = Tensor::from_vec(vec![4, 2], vec![0.5, 0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.75])
        .expect("invariant: k valide");
    let v = Tensor::from_vec(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("invariant: v valide");
    let windowed =
        causal_attention(&q, &k, &v, &windowed_layout).expect("invariant: attention valide");

    let truncate = |tensor: &Tensor, start: usize, end: usize| -> Tensor {
        let mut data = Vec::new();
        for row in start..end {
            data.extend_from_slice(tensor.row_slice(row).expect("invariant: ligne valide"));
        }
        Tensor::from_vec(vec![end - start, 2], data).expect("invariant: tronqué valide")
    };
    for pos in 0..4_usize {
        let start = (pos + 1).saturating_sub(2);
        let reference = causal_attention(
            &truncate(&q, start, pos + 1),
            &truncate(&k, start, pos + 1),
            &truncate(&v, start, pos + 1),
            &full_layout,
        )
        .expect("invariant: référence valide");
        assert_close(
            windowed.row_slice(pos).expect("invariant: ligne valide"),
            reference
                .row_slice(pos - start)
                .expect("invariant: ligne valide"),
            1.0e-6,
        );
    }
}

#[test]
fn cached_attention_sliding_window_matches_truncated_cache() {
    let full_layout = AttentionLayout {
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        rope_dims: 2,
        attn_scalar: 2.0,
        sliding_window: None,
    };
    let windowed_layout = AttentionLayout {
        sliding_window: Some(3),
        ..full_layout
    };
    let keys = [
        [0.5_f32, 0.0],
        [1.0, 0.5],
        [0.25, 1.0],
        [0.0, 0.75],
        [0.75, 0.25],
    ];
    let values = [
        [1.0_f32, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ];
    let row = |data: [f32; 2]| {
        Tensor::from_vec(vec![1, 2], data.to_vec()).expect("invariant: ligne valide")
    };
    let mut windowed_cache = LayerKvCache::default();
    for (k, v) in keys.iter().zip(values.iter()) {
        windowed_cache
            .append(&row(*k), &row(*v), &windowed_layout)
            .expect("invariant: append valide");
    }
    let mut truncated_cache = LayerKvCache::default();
    for (k, v) in keys.iter().zip(values.iter()).skip(2) {
        truncated_cache
            .append(&row(*k), &row(*v), &full_layout)
            .expect("invariant: append valide");
    }
    let q = row([1.0, 0.25]);
    let windowed = cached_attention_one(&q, &mut windowed_cache, &windowed_layout)
        .expect("invariant: attention fenêtrée valide");
    let reference = cached_attention_one(&q, &mut truncated_cache, &full_layout)
        .expect("invariant: attention référence valide");
    assert_close(
        windowed.as_row().expect("invariant: ligne valide"),
        reference.as_row().expect("invariant: ligne valide"),
        1.0e-6,
    );
}

#[test]
fn attention_scale_uses_query_pre_attn_scalar() {
    // Échelle Gemma : scores · 1/√query_pre_attn_scalar au lieu de 1/√head_dim.
    let layout = AttentionLayout {
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        rope_dims: 2,
        attn_scalar: 8.0,
        sliding_window: None,
    };
    let row = |data: [f32; 2]| {
        Tensor::from_vec(vec![1, 2], data.to_vec()).expect("invariant: ligne valide")
    };
    let mut cache = LayerKvCache::default();
    cache
        .append(&row([1.0, 0.0]), &row([1.0, 0.0]), &layout)
        .expect("invariant: append valide");
    cache
        .append(&row([0.0, 0.0]), &row([0.0, 1.0]), &layout)
        .expect("invariant: append valide");
    let out = cached_attention_one(&row([1.0, 0.0]), &mut cache, &layout)
        .expect("invariant: attention valide");
    // scores = [1/√8, 0] → p0 = 1/(1 + e^{-1/√8}), sortie = p0·v0 + (1−p0)·v1.
    let p0 = 1.0 / (1.0 + (-1.0_f32 / 8.0_f32.sqrt()).exp());
    assert_close(
        out.as_row().expect("invariant: ligne valide"),
        &[p0, 1.0 - p0],
        1.0e-6,
    );
}

#[test]
fn apply_rope_styles_match_mlx_reference() {
    // Références : mx.fast.rope(x, 4, base=10000, scale=1.0, offset=1) —
    // traditional=True pour Interleaved, traditional=False (rotate-half) pour
    // Halves.
    let x = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("invariant: x valide");
    let rope = RopeParams {
        theta: 10_000.0,
        frequency_dim: 4,
        position_scale: 1.0,
    };
    let interleaved = apply_rope_heads_at(&x, 1, 4, 4, rope, 1, RopeStyle::Interleaved)
        .expect("invariant: rope valide");
    assert_close(
        interleaved.as_row().expect("invariant: ligne valide"),
        &[-1.142_64, 1.922_08, 2.959_85, 4.029_8],
        1.0e-4,
    );
    let halves = apply_rope_heads_at(&x, 1, 4, 4, rope, 1, RopeStyle::Halves)
        .expect("invariant: rope valide");
    assert_close(
        halves.as_row().expect("invariant: ligne valide"),
        &[-1.984_11, 1.959_9, 2.462_38, 4.019_8],
        1.0e-4,
    );
}

#[test]
fn apply_rope_halves_partial_rotates_inside_rope_dims() {
    // Qwen3.5/Qwen3.6 full-attn utilise un RoPE partiel: seule la tranche
    // rotative tourne, puis le reste de la tête est concaténé inchangé.
    let x = Tensor::from_vec(vec![1, 8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("invariant: x valide");
    let rope = RopeParams {
        theta: 10_000.0,
        frequency_dim: 4,
        position_scale: 1.0,
    };

    let halves = apply_rope_heads_at(&x, 1, 8, 4, rope, 1, RopeStyle::Halves)
        .expect("invariant: rope valide");

    assert_close(
        halves.as_row().expect("invariant: ligne valide"),
        &[-1.984_11, 1.959_9, 2.462_38, 4.019_8, 5.0, 6.0, 7.0, 8.0],
        1.0e-4,
    );
}

#[test]
fn apply_rope_position_scale_matches_mlx_reference() {
    // Références : mx.fast.rope(x, 4, base=…, scale=0.125, offset=…) — le
    // scaling linear ×8 des couches globales Gemma 3 multiplie les positions
    // par 1/8 AVANT la fréquence (traditional=False pour Halves, True pour
    // Interleaved).
    let x = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("invariant: x valide");
    let gemma_global = RopeParams {
        theta: 1_000_000.0,
        frequency_dim: 4,
        position_scale: 0.125,
    };
    let halves = apply_rope_heads_at(&x, 1, 4, 4, gemma_global, 5, RopeStyle::Halves)
        .expect("invariant: rope valide");
    assert_close(
        halves.as_row().expect("invariant: ligne valide"),
        &[-0.944_329, 1.997_5, 3.017_987, 4.001_249],
        1.0e-4,
    );
    let interleaved = apply_rope_heads_at(&x, 1, 4, 4, gemma_global, 5, RopeStyle::Interleaved)
        .expect("invariant: rope valide");
    assert_close(
        interleaved.as_row().expect("invariant: ligne valide"),
        &[-0.359_231, 2.207_024, 2.997_5, 4.001_874],
        1.0e-4,
    );
    let scaled_small_base = RopeParams {
        theta: 10_000.0,
        frequency_dim: 4,
        position_scale: 0.125,
    };
    let halves = apply_rope_heads_at(&x, 1, 4, 4, scaled_small_base, 3, RopeStyle::Halves)
        .expect("invariant: rope valide");
    assert_close(
        halves.as_row().expect("invariant: ligne valide"),
        &[-0.168_310, 1.984_986, 3.157_795, 4.007_472],
        1.0e-4,
    );
}

#[test]
fn apply_rope_halves_partial_uses_rotary_slice_half_offset() {
    let x = Tensor::from_vec(vec![1, 8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("invariant: x valide");
    let rope = RopeParams {
        theta: 10_000.0,
        frequency_dim: 4,
        position_scale: 1.0,
    };

    let out = apply_rope_heads_at(&x, 1, 8, 4, rope, 1, RopeStyle::Halves)
        .expect("invariant: rope valide");
    let angle0 = 1.0_f32;
    let angle1 = 1.0_f32 / 10_000.0_f32.powf(0.5);
    let expected = vec![
        1.0 * angle0.cos() - 3.0 * angle0.sin(),
        2.0 * angle1.cos() - 4.0 * angle1.sin(),
        1.0 * angle0.sin() + 3.0 * angle0.cos(),
        2.0 * angle1.sin() + 4.0 * angle1.cos(),
        5.0,
        6.0,
        7.0,
        8.0,
    ];

    assert_close(out.data(), &expected, 1.0e-6);
}

#[test]
fn rope_position_scale_one_is_bit_identical_to_raw_positions() {
    // Byte-identité des modèles non scalés : `pos × 1.0` doit produire des
    // rotations bit-identiques aux positions brutes (chemin Qwen prod).
    let x = Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, -0.5, 0.25, 1.5, -2.0])
        .expect("invariant: x valide");
    for style in [RopeStyle::Interleaved, RopeStyle::Halves] {
        let raw = apply_rope_heads_at(
            &x,
            1,
            4,
            4,
            RopeParams {
                theta: 1_000_000.0,
                frequency_dim: 4,
                position_scale: 1.0,
            },
            7,
            style,
        )
        .expect("invariant: rope valide");
        // Référence indépendante : rotations recalculées à la main sans échelle.
        let mut expected = x.data().to_vec();
        for pos in 0..2 {
            let position = (7 + pos) as f32;
            for pair in 0..2 {
                let exponent = (2 * pair) as f32 / 4.0;
                let angle = position / 1_000_000.0_f32.powf(exponent);
                let (cos, sin) = (angle.cos(), angle.sin());
                let (first_index, second_index) = match style {
                    RopeStyle::Interleaved => (pos * 4 + 2 * pair, pos * 4 + 2 * pair + 1),
                    RopeStyle::Halves => (pos * 4 + pair, pos * 4 + pair + 2),
                };
                let first = x.data()[first_index];
                let second = x.data()[second_index];
                expected[first_index] = first * cos - second * sin;
                expected[second_index] = first * sin + second * cos;
            }
        }
        assert_eq!(raw.data(), expected.as_slice(), "style={style:?}");
    }
}

#[test]
fn fused_rms_rope_matches_two_step_for_both_styles() {
    let x = Tensor::from_vec(
        vec![2, 4],
        vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -2.0, 0.125],
    )
    .expect("invariant: x valide");
    let weight =
        Tensor::from_vec(vec![4], vec![1.5, 0.5, 1.0, 2.0]).expect("invariant: poids valide");
    for position_scale in [1.0_f32, 0.125] {
        let rope = RopeParams {
            theta: 10_000.0,
            frequency_dim: 4,
            position_scale,
        };
        for style in [RopeStyle::Interleaved, RopeStyle::Halves] {
            let fused = rms_norm_rope_heads_at(&x, 1, 4, 4, &weight, 1.0e-6, rope, 3, style)
                .expect("invariant: fused valide");
            let normed = rms_norm_heads(&x, 1, 4, &weight, 1.0e-6).expect("invariant: rms valide");
            let two_step = apply_rope_heads_at(&normed, 1, 4, 4, rope, 3, style)
                .expect("invariant: rope valide");
            assert_close(fused.data(), two_step.data(), 1.0e-6);
        }
    }
}

#[test]
fn attention_layout_wires_scale_and_window() {
    let mut tensors: HashMap<String, DecoderTensor> = HashMap::new();
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        tensors.insert(
            format!("layers.0.self_attn.{name}.weight"),
            DecoderTensor::Dense(identity2().expect("invariant: identité valide")),
        );
    }
    let config = CausalDecoderConfig {
        head_dim: Some(2),
        rope_dims: Some(2),
        ..CausalDecoderConfig::default()
    };
    let attention = full_attention_from_tensors(&config, &mut tensors, "layers.0", 0)
        .expect("invariant: attention valide");
    let AttentionBlock::Full(mut attention) = attention else {
        panic!("invariant: attention full attendue");
    };
    let q = Tensor::from_vec(vec![1, 2], vec![1.0, 0.0]).expect("invariant: q valide");

    // Gemma : query_pre_attn_scalar + fenêtre de couche locale.
    attention.sliding_window = Some(512);
    let gemma_config = CausalDecoderConfig {
        head_dim: Some(2),
        query_pre_attn_scalar: Some(168.0),
        ..CausalDecoderConfig::default()
    };
    let layout =
        attention_layout(&gemma_config, &attention, &q, &q, &q).expect("invariant: layout valide");
    assert_eq!(layout.attn_scalar, 168.0);
    assert_eq!(layout.sliding_window, Some(512));

    // Défauts Qwen/Llama : échelle head_dim, attention pleine.
    attention.sliding_window = None;
    let default_config = CausalDecoderConfig {
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let layout = attention_layout(&default_config, &attention, &q, &q, &q)
        .expect("invariant: layout valide");
    assert_eq!(layout.attn_scalar, 2.0);
    assert_eq!(layout.sliding_window, None);
}

fn test_weights() -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("invariant: embedding valide"),
    );
    tensors.insert(
        "layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    for prefix in [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
    ] {
        tensors.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
            .expect("invariant: lm_head valide"),
    );
    tensors
}

#[cfg(feature = "devtools")]
fn tiny_dflash_draft() -> crate::DFlashDraft {
    crate::DFlashDraft {
        info: crate::DFlashDraftInfo {
            draft_dir: std::path::PathBuf::new(),
            weight_path: std::path::PathBuf::new(),
            tensor_count: 14,
            block_size: 2,
            mask_token_id: 2,
            target_layer_ids: vec![0],
            num_hidden_layers: 1,
            hidden_size: 2,
            intermediate_size: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1.0e-6,
            rope_theta: 10_000.0,
            layer_types: Vec::new(),
            sliding_window: None,
        },
        hidden_norm: Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
        fc: Linear::new(identity2().expect("invariant: fc valide"), None)
            .expect("invariant: linear valide"),
        layers: vec![crate::DFlashDraftLayer {
            input_norm: Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
            attention: crate::DFlashAttentionWeights {
                q_proj: Linear::new(identity2().expect("invariant: q valide"), None)
                    .expect("invariant: linear valide"),
                k_proj: Linear::new(identity2().expect("invariant: k valide"), None)
                    .expect("invariant: linear valide"),
                v_proj: Linear::new(identity2().expect("invariant: v valide"), None)
                    .expect("invariant: linear valide"),
                o_proj: Linear::new(identity2().expect("invariant: o valide"), None)
                    .expect("invariant: linear valide"),
                q_norm: Tensor::from_vec(vec![2], vec![1.0, 1.0])
                    .expect("invariant: q norm valide"),
                k_norm: Tensor::from_vec(vec![2], vec![1.0, 1.0])
                    .expect("invariant: k norm valide"),
            },
            post_attention_norm: Tensor::from_vec(vec![2], vec![1.0, 1.0])
                .expect("invariant: post norm valide"),
            mlp: GatedMlp::new(
                Linear::new(identity2().expect("invariant: gate valide"), None)
                    .expect("invariant: linear valide"),
                Linear::new(identity2().expect("invariant: up valide"), None)
                    .expect("invariant: linear valide"),
                Linear::new(identity2().expect("invariant: down valide"), None)
                    .expect("invariant: linear valide"),
            ),
        }],
        norm: Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    }
}

fn hybrid_weights(layer_count: usize, full_attention_interval: usize) -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("invariant: embedding valide"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
            .expect("invariant: lm_head valide"),
    );
    for layer in 0..layer_count {
        if (layer + 1) % full_attention_interval == 0 {
            insert_attention_layer(&mut tensors, layer);
        } else {
            insert_linear_attention_layer(&mut tensors, layer);
        }
    }
    tensors
}

fn hybrid_config(layer_count: usize) -> CausalDecoderConfig {
    CausalDecoderConfig {
        num_hidden_layers: layer_count,
        full_attention_interval: Some(2),
        linear_num_key_heads: Some(1),
        linear_num_value_heads: Some(1),
        linear_key_head_dim: Some(2),
        linear_value_head_dim: Some(2),
        linear_conv_kernel_dim: Some(2),
        ..CausalDecoderConfig::default()
    }
}

fn gated_attention_weights() -> HashMap<String, Tensor> {
    let mut tensors = test_weights();
    tensors.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        Tensor::from_vec(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0])
            .expect("invariant: q_proj gated valide"),
    );
    tensors
}

fn insert_attention_layer(tensors: &mut HashMap<String, Tensor>, layer: usize) {
    tensors.insert(
        format!("layers.{layer}.input_layernorm.weight"),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    for prefix in [
        format!("layers.{layer}.self_attn.q_proj"),
        format!("layers.{layer}.self_attn.k_proj"),
        format!("layers.{layer}.self_attn.v_proj"),
        format!("layers.{layer}.self_attn.o_proj"),
    ] {
        tensors.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
}

fn insert_linear_attention_layer(tensors: &mut HashMap<String, Tensor>, layer: usize) {
    tensors.insert(
        format!("layers.{layer}.input_layernorm.weight"),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_qkv.weight"),
        Tensor::from_vec(
            vec![6, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0],
        )
        .expect("invariant: qkv valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_z.weight"),
        identity2().expect("invariant: z valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_b.weight"),
        Tensor::from_vec(vec![1, 2], vec![1.0, 0.0]).expect("invariant: b valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_a.weight"),
        Tensor::from_vec(vec![1, 2], vec![0.0, 1.0]).expect("invariant: a valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.out_proj.weight"),
        identity2().expect("invariant: out valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.conv1d.weight"),
        Tensor::from_vec(
            vec![6, 2, 1],
            vec![
                0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0,
            ],
        )
        .expect("invariant: conv valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.A_log"),
        Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: A_log valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.dt_bias"),
        Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: dt_bias valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.norm.weight"),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm lin valide"),
    );
}

fn insert_qk_norm(tensors: &mut HashMap<String, Tensor>, layer: usize, q: Vec<f32>, k: Vec<f32>) {
    tensors.insert(
        format!("layers.{layer}.self_attn.q_norm.weight"),
        Tensor::from_vec(vec![q.len()], q).expect("invariant: q_norm valide"),
    );
    tensors.insert(
        format!("layers.{layer}.self_attn.k_norm.weight"),
        Tensor::from_vec(vec![k.len()], k).expect("invariant: k_norm valide"),
    );
}

fn insert_identity_mlp(tensors: &mut HashMap<String, Tensor>) {
    tensors.insert(
        "layers.0.post_attention_layernorm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    for prefix in ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
    tensors.insert(
        "layers.0.mlp.down_proj.weight".to_string(),
        identity2().expect("invariant: identité valide"),
    );
}

pub(super) fn gqa_weights() -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .expect("invariant: embedding GQA valide"),
    );
    tensors.insert(
        "layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("invariant: norm GQA valide"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0])
            .expect("invariant: norm finale GQA valide"),
    );
    tensors.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        identity4().expect("invariant: q_proj GQA valide"),
    );
    for prefix in ["layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .expect("invariant: projection KV GQA valide"),
        );
    }
    tensors.insert(
        "layers.0.self_attn.o_proj.weight".to_string(),
        identity4().expect("invariant: o_proj GQA valide"),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .expect("invariant: lm_head GQA valide"),
    );
    tensors
}

fn identity2() -> Result<Tensor> {
    Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
}

fn identity4() -> Result<Tensor> {
    Tensor::from_vec(
        vec![4, 4],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "index={idx} left={a} right={b} tolerance={tolerance}"
        );
    }
}

struct F32View {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for F32View {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn tensor_views(tensors: HashMap<String, Tensor>) -> Vec<(String, F32View)> {
    tensors
        .into_iter()
        .map(|(name, tensor)| {
            let data = tensor
                .data()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            (
                name,
                F32View {
                    shape: tensor.shape().to_vec(),
                    data,
                },
            )
        })
        .collect()
}

// --- 1b.3 : oracle du decode full-attn résident (GPU) vs CPU ---

/// Modèle GQA synthétique (q=2, kv=1, hd=2) **métal-activé**, ou `None` si
/// aucun device Metal (skip propre, comme les tests `metal_backend`).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_test_model() -> Option<CausalDecoder> {
    let config = CausalDecoderConfig {
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        rope_theta: Some(10_000.0),
        ..CausalDecoderConfig::default()
    };
    let model =
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: modèle GQA valide");
    match model.with_metal_runtime() {
        Ok(model) => Some(model),
        Err(InferError::Metal(message)) if message.contains("aucun device") => None,
        Err(error) => panic!("runtime Metal indisponible: {error:?}"),
    }
}

/// Décode `n` tokens en greedy T=0, chemin résident forcé (`resident`) ou CPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn greedy_resident_ab(
    model: &CausalDecoder,
    prompt: &[usize],
    n: usize,
    resident: bool,
) -> Vec<usize> {
    let options = GenerationOptions::default();
    let mut sampler = DeterministicSampler::new(options.seed);
    let (mut cache, final_state) = model
        .prefill_cache_state_tokenwise(prompt)
        .expect("invariant: prefill valide");
    if resident {
        model
            .setup_resident_decode(&mut cache, n + 1, false)
            .expect("invariant: setup résident valide");
    }
    let mut tokens = Vec::with_capacity(n);
    let mut token = model
        .sample_token_from_state(&final_state, &options, &mut sampler)
        .expect("invariant: sampling valide");
    for _ in 0..n {
        tokens.push(token);
        let state = model
            .next_final_state_cached(&mut cache, token)
            .expect("invariant: decode valide");
        token = model
            .sample_token_from_state(&state, &options, &mut sampler)
            .expect("invariant: sampling valide");
    }
    tokens
}

/// 1b.3 / R2 — le clone du cache (prefix-cache) DROP l'état Metal résident
/// (`full` → None) sans toucher l'original (pas de double-ownership MTLBuffer).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_cache_clone_drops_metal() {
    let Some(model) = resident_test_model() else {
        return;
    };
    let prompt = [0_usize, 1];
    let (mut cache, _) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill valide");
    model
        .setup_resident_decode(&mut cache, 16, false)
        .expect("invariant: setup résident valide");
    assert!(
        cache.layers.iter().any(|layer| layer.full.is_some()),
        "le setup résident doit poser au moins un KV full-attn"
    );
    let clone = cache.clone();
    assert!(
        clone.layers.iter().all(|layer| layer.full.is_none()),
        "le clone doit dropper l'état Metal résident (full = None)"
    );
    assert!(
        cache.layers.iter().any(|layer| layer.full.is_some()),
        "l'original conserve son KV résident après le clone"
    );
}

/// 1b.3 — oracle teacher-forced : sur > 256 tokens (KV croissant, au-delà du
/// plafond 256 du kernel de prefill), l'état final par token du chemin
/// résident GPU est numériquement identique au chemin CPU (tolérance, réserve
/// E : le kernel change l'ordre de réduction f32).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_decode_state_matches_cpu_over_long_kv() {
    let Some(model) = resident_test_model() else {
        return;
    };
    let prompt = [0_usize, 1];
    let n = 320; // > 256
    let feed: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let (mut cache_off, _) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill OFF");
    let (mut cache_on, _) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill ON");
    model
        .setup_resident_decode(&mut cache_on, n + 1, false)
        .expect("invariant: setup résident");
    let mut max_abs = 0.0_f32;
    for &token in &feed {
        let off = model
            .next_final_state_cached(&mut cache_off, token)
            .expect("invariant: decode OFF");
        let on = model
            .next_final_state_cached(&mut cache_on, token)
            .expect("invariant: decode ON");
        assert_eq!(off.shape(), on.shape());
        for (a, b) in off.data().iter().zip(on.data()) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    // Résidu mesuré : max_abs ≈ 6.6e-7 sur 320 tokens (l'erreur f32 de l'ordre
    // de réduction ne diverge pas). Borne = garde-fou de régression robuste.
    assert!(
        max_abs <= 1.0e-4,
        "état résident vs CPU: max_abs={max_abs:e} sur {n} tokens"
    );
}

/// 1b.3 — oracle greedy : séquences T=0 token-identiques OFF vs ON, sur ≥256
/// tokens, pour deux prompts (multi-tour à cache frais → reset propre).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_decode_greedy_tokens_match_cpu() {
    let Some(model) = resident_test_model() else {
        return;
    };
    for prompt in [vec![0_usize, 1], vec![1_usize, 0, 1]] {
        let n = 256;
        let off = greedy_resident_ab(&model, &prompt, n, false);
        let on = greedy_resident_ab(&model, &prompt, n, true);
        assert_eq!(off, on, "greedy OFF vs ON divergent (prompt {prompt:?})");
    }
}

// --- 1c.0 : ossature decode résident complet (préconditions + délégation) ---

/// 1c.0 / MAJEUR 6 — la validation des préconditions REFUSE un modèle non
/// supporté en résident (ici le modèle GQA de test n'a pas de MoE/shared-expert)
/// → tout-ou-rien : on ne démarre jamais le command buffer unique sans garantie.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_full_precondition_rejects_model_without_moe() {
    let Some(model) = resident_test_model() else {
        return;
    };
    assert!(
        !model.supports_resident_full_decode(),
        "un modèle sans MoE/shared-expert ne doit PAS être déclaré supporté en résident"
    );
}

// --- MTP : oracle résident two-row committed-history (GPU + modèle réel) ---

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_env_ready() -> bool {
    let required = [
        ("RETI_RUST_MTP_HISTORY", "committed"),
        ("RETI_RUST_PREFIX_CACHE", "0"),
        ("RETI_RUST_DECODE_RESIDENT_FULL", "1"),
        ("RETI_RUST_DECODE_RESIDENT_FULL_LINEAR", "1"),
        ("RETI_RUST_GPU_ARGMAX", "1"),
    ];
    for (key, expected) in required {
        match std::env::var(key) {
            Ok(actual) if actual == expected => {}
            Ok(actual) => {
                eprintln!("skip MTP verifier integration: {key}={actual}, expected {expected}");
                return false;
            }
            Err(_) => {
                eprintln!("skip MTP verifier integration: {key} is not set to {expected}");
                return false;
            }
        }
    }
    true
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_model_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("RETI_RUST_MTP_VERIFIER_TEST_MODEL") {
        let path = PathBuf::from(path);
        if path.is_dir() {
            return Some(path);
        }
        eprintln!(
            "skip MTP verifier integration: RETI_RUST_MTP_VERIFIER_TEST_MODEL is not a directory: {}",
            path.display()
        );
        return None;
    }
    let default = PathBuf::from("/Users/ludwig/workspace/reti/models/Qwen3.6-27B-OptiQ-4bit");
    if default.is_dir() {
        return Some(default);
    }
    eprintln!(
        "skip MTP verifier integration: model missing, set RETI_RUST_MTP_VERIFIER_TEST_MODEL"
    );
    None
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_prompt_file(path: &Path, fallback: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!(
                "MTP verifier integration: using fallback prompt, cannot read {}: {error}",
                path.display()
            );
            fallback.to_string()
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_prompt_ids(assets: &crate::ModelAssets, prompt: &str) -> Result<Vec<usize>> {
    let templated = crate::render_qwen_chatml(
        &[crate::ChatTemplateMessage::new("user", prompt)],
        true,
        false,
    );
    assets
        .encode_prompt(&templated)?
        .into_iter()
        .map(|id| {
            usize::try_from(id)
                .map_err(|_| InferError::Dimension(format!("token id hors plage: {id}")))
        })
        .collect()
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_options(stop_token_ids: Vec<usize>) -> GenerationOptions {
    GenerationOptions {
        stop_token_ids,
        stop_sequences: Vec::new(),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    }
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_case(
    model: &CausalDecoder,
    name: &str,
    prompt: &[usize],
    max_new_tokens: usize,
    stop_token_ids: Vec<usize>,
) -> Result<SpeculativeOutput> {
    let options = mtp_verifier_options(stop_token_ids);
    eprintln!("MTP verifier integration: case {name}, max_new_tokens={max_new_tokens}");
    let ar = model.generate_greedy_cached_with_options(prompt, max_new_tokens, &options)?;
    let spec =
        model.generate_greedy_mtp_batched_with_options(prompt, max_new_tokens, &options, 1)?;
    assert_eq!(
        spec.tokens, ar,
        "MTP two-row committed verifier diverged from AR in case {name}"
    );
    eprintln!(
        "MTP verifier integration: case {name} ok generated={} proposed={} accepted={} rejected={} verifications={}",
        spec.tokens.len(),
        spec.stats.proposed,
        spec.stats.accepted,
        spec.stats.rejected,
        spec.stats.verifications
    );
    Ok(spec)
}

/// Valide le chemin GPU-only `next_mtp_spec_one_resident` en historique MTP
/// committed depth-1. Le test est ignoré par défaut car il charge le 27B réel ;
/// il se lance explicitement avec les flags résident/MTP ci-dessous.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
#[ignore = "requires Metal + 27B MTP model; set RETI_RUST_MTP_HISTORY=committed and resident decode flags"]
fn mtp_committed_two_row_verifier_edges_match_ar() -> Result<()> {
    if !mtp_verifier_env_ready() {
        return Ok(());
    }
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    assert!(
        assets.mtp.is_available(),
        "MTP verifier integration requires an MTP sidecar in {}",
        model_dir.display()
    );
    let Some(mtp_path) = assets.mtp.path.as_ref() else {
        return Err(InferError::MissingArtifact {
            path: model_dir,
            what: "MTP sidecar",
        });
    };
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(mtp_path)?;
    assert!(
        model.mtp.is_some(),
        "MTP verifier integration must load the MTP head"
    );
    assert!(
        model.supports_resident_full_decode(),
        "MTP verifier integration requires resident full decode support"
    );

    let coding_prompt = mtp_verifier_prompt_file(
        Path::new("/tmp/mtp_sustained_long_code_prompt.txt"),
        "Write a complete Rust module implementing a deterministic LRU cache with tests. Include the public API, error handling, and examples.",
    );
    let realistic_prompt = mtp_verifier_prompt_file(
        Path::new("/tmp/mtp_realistic_prompt.txt"),
        "Explain the tradeoffs between prefix caching and batched prefill for a short multi-turn voice assistant.",
    );
    let coding_ids = mtp_verifier_prompt_ids(&assets, &coding_prompt)?;
    let realistic_ids = mtp_verifier_prompt_ids(&assets, &realistic_prompt)?;

    let accepted = mtp_verifier_case(&model, "accepted-draft-bonus", &coding_ids, 64, Vec::new())?;
    assert!(
        accepted.stats.accepted > 0,
        "coding prompt should consume at least one accepted draft from the two-row verifier"
    );
    assert!(
        accepted.stats.verifications >= accepted.tokens.len().saturating_sub(1),
        "accepted case should carry the next distribution across the two-row verify"
    );

    let rejected = mtp_verifier_case(
        &model,
        "rejected-draft-rollback",
        &realistic_ids,
        64,
        Vec::new(),
    )?;
    assert!(
        rejected.stats.rejected > 0,
        "realistic prompt should exercise reject rollback and subsequent byte-correct continuation"
    );

    let probe_options = mtp_verifier_options(Vec::new());
    let probe = model.generate_greedy_cached_with_options(&coding_ids, 16, &probe_options)?;
    let Some(stop_token) = probe.get(1).copied() else {
        return Err(InferError::Dimension(
            "MTP stop case needs at least two AR probe tokens".to_string(),
        ));
    };
    let stopped = mtp_verifier_case(
        &model,
        "stop-token-mid-cycle",
        &coding_ids,
        32,
        vec![stop_token],
    )?;
    assert!(
        stopped.tokens.len() < 32,
        "stop-token case should terminate before the token budget"
    );

    let mut budget_match = None;
    for max_new_tokens in 2..=16 {
        let budget = mtp_verifier_case(
            &model,
            "end-of-budget-mid-cycle",
            &coding_ids,
            max_new_tokens,
            Vec::new(),
        )?;
        if budget.stats.accepted > 0 {
            budget_match = Some((max_new_tokens, budget));
            break;
        }
    }
    let Some((budget_tokens, budget)) = budget_match else {
        return Err(InferError::Dimension(
            "MTP end-of-budget case did not find an accepted draft within 16 tokens".to_string(),
        ));
    };
    assert_eq!(
        budget.tokens.len(),
        budget_tokens,
        "end-of-budget case should stop exactly at max_new_tokens"
    );
    assert!(
        budget.stats.accepted > 0,
        "end-of-budget case should reach the budget after accepting a draft"
    );

    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_sampling_predicates_route_default_and_readback() {
    let default_sample = GenerationOptions {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        ..GenerationOptions::default()
    };
    assert!(super::resident::resident_sampling_supported(
        &default_sample
    ));
    assert!(super::resident::resident_sampling_on_device(
        &default_sample
    ));

    let top_k64 = GenerationOptions {
        top_k: 64,
        ..default_sample.clone()
    };
    assert!(super::resident::resident_sampling_supported(&top_k64));
    assert!(!super::resident::resident_sampling_on_device(&top_k64));

    let nucleus = GenerationOptions {
        top_p: 0.95,
        top_k: 0,
        ..default_sample
    };
    assert!(super::resident::resident_sampling_supported(&nucleus));
    assert!(!super::resident::resident_sampling_on_device(&nucleus));
}

// NOTE: l'ancien test d'ossature 1c.0 (`next_final_state_resident` déléguant au
// per-op) est retiré : le corps résident est désormais l'assemblage complet 1c
// (`decode_token_resident`), non instanciable par le modèle synthétique de test
// (sans MoE → `supports_resident_full_decode()=false`). La validation 1c est
// l'oracle 35B réel (texte greedy ==CPU + A/B).
