use super::attention_ops::{
    apply_rope_heads_at, attention_layout, cached_attention_one, causal_attention,
    full_attention_append_kv_rows, full_attention_context_prefill, full_attention_from_tensors,
    rms_norm_heads, rms_norm_rope_heads_at, AttentionLayout, RopeParams,
};
use super::*;
use safetensors::{serialize, Dtype, View};
use std::borrow::Cow;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
use std::path::{Path, PathBuf};

#[test]
fn qwen_embedding_recast_is_opt_in_and_rounds_to_bf16() {
    let values = vec![1.003_906_2_f32, -0.333_251_95, 42.125];
    let hidden = Tensor::from_vec(vec![1, values.len()], values.clone())
        .expect("invariant: tenseur embedding valide");

    let disabled = recast_qwen_embedding(hidden.clone(), true, false);
    let unrelated = recast_qwen_embedding(hidden.clone(), false, true);
    let enabled = recast_qwen_embedding(hidden, true, true);

    assert_eq!(disabled.data(), values.as_slice());
    assert_eq!(unrelated.data(), values.as_slice());
    assert_eq!(enabled.data(), &[1.0, -0.333_984_38, 42.0]);
}

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
fn callback_greedy_observes_tokens_and_can_stop() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let state = model
        .prefill_prompt_state_uncached(&[0])
        .expect("invariant: prefill valide");
    let mut seen = Vec::new();

    let output = model
        .generate_greedy_timed_from_prompt_state_with_options_and_callback(
            state,
            Duration::ZERO,
            5,
            &GenerationOptions::default(),
            |token| {
                seen.push(token);
                seen.len() < 2
            },
        )
        .expect("invariant: génération callback valide");

    assert_eq!(output.tokens, seen);
    assert_eq!(output.tokens.len(), 2);
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
            mtp_history: None,
            captures: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            linear_metal: Vec::new(),
        },
        PrefixCacheEntry {
            tokens: vec![1, 2, 3],
            cache: model.empty_cache(),
            final_state: final_state.clone(),
            mtp_history: None,
            captures: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            linear_metal: Vec::new(),
        },
        PrefixCacheEntry {
            tokens: vec![1],
            cache: model.empty_cache(),
            final_state,
            mtp_history: None,
            captures: None,
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

/// Le plafond en octets compte une fois les segments partagés et garde
/// toujours l'entrée la plus récente.
#[test]
fn prefix_cache_byte_cap_counts_shared_segments_once() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let final_state = Tensor::row(vec![0.0]).expect("invariant: tenseur test valide");
    let history = || {
        let mut history = LayerKvCache::default();
        history.keys.push_owned(vec![0.0; 1000]);
        history.values.push_owned(vec![0.0; 1000]);
        history
    };
    let shared = history();
    let entry = |tokens: Vec<usize>, mtp_history: LayerKvCache| PrefixCacheEntry {
        tokens,
        cache: model.empty_cache(),
        final_state: final_state.clone(),
        mtp_history: Some(mtp_history),
        captures: None,
        #[cfg(all(target_os = "macos", feature = "metal"))]
        linear_metal: Vec::new(),
    };
    let mut cache = PrefixCache {
        entries: vec![
            entry(vec![1, 2, 3], shared.clone()),
            entry(vec![1, 2], shared),
            entry(vec![7], history()),
        ],
    };
    let newest = cache.entries[0].estimated_bytes_unseen(&mut Default::default());
    cache.enforce_byte_cap(newest + 1000);
    assert_eq!(cache.entries.len(), 2, "segments partagés comptés une fois");
    cache.enforce_byte_cap(0);
    assert_eq!(cache.entries.len(), 1, "entrée la plus récente gardée");
}

/// `prefix_cache_put` retire l'entrée dont le prolongement a recopié le K/V,
/// garde celles qui partagent une allocation avec le nouvel état, les branches
/// divergentes et les entrées plus courtes à buffer propre.
#[test]
fn prefix_cache_put_removes_superseded_copies() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let final_state = Tensor::row(vec![0.0])?;
    let own = |base: Option<&CausalDecoderCache>| {
        let mut cache = base.cloned().unwrap_or_else(|| model.empty_cache());
        cache.layers[0].keys.push_owned(vec![0.0; 8]);
        cache
    };
    let put = |tokens: &[usize], cache: CausalDecoderCache| {
        model.prefix_cache_put(tokens, &(cache, final_state.clone()), None, None)
    };
    let cached = || -> Result<Vec<Vec<usize>>> {
        let cache = model
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
        Ok(cache.entries.iter().map(|e| e.tokens.clone()).collect())
    };

    // Retour en arrière : [1] a son propre buffer, inséré après [1, 2].
    let turn2 = own(None);
    put(&[1, 2], turn2.clone())?;
    // Branche divergente, buffer propre.
    put(&[1, 3], own(None))?;
    put(&[1], own(None))?;
    // Prolongement en place : la vue partage l'allocation de [1, 2].
    put(&[1, 2, 4], own(Some(&turn2)))?;
    assert_eq!(
        cached()?,
        vec![vec![1, 2, 4], vec![1], vec![1, 3], vec![1, 2]]
    );
    // Prolongement par copie de [1, 2, 4] : elle et [1, 2], qui partage son
    // allocation, sont retirées ; [1] et [1, 3] restent.
    put(&[1, 2, 4, 5], own(None))?;
    assert_eq!(cached()?, vec![vec![1, 2, 4, 5], vec![1], vec![1, 3]]);
    Ok(())
}

/// Le remplacement exact d'une entrée garde son historique MTP.
#[test]
fn prefix_cache_put_exact_replacement_keeps_metadata() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    let mut history = LayerKvCache::default();
    history.keys.push_owned(vec![0.0; 4]);
    model.prefix_cache_put(&[1, 2], &state, Some(&history), None)?;
    model.prefix_cache_put(&[1, 2], &state, None, None)?;
    let cache = model
        .prefix_cache
        .lock()
        .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
    assert_eq!(cache.entries.len(), 1);
    assert!(cache.entries[0].mtp_history.is_some());
    Ok(())
}

/// Un hit DFlash exige les mêmes couches et des captures couvrant la fenêtre
/// demandée ; une entrée à fenêtre plus courte est ignorée (prefill complet).
#[test]
#[cfg(feature = "devtools")]
fn prefix_cache_dflash_hit_requires_capture_coverage() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let prompt = [1usize, 2, 3, 4];
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    let captures = PrefixCaptures::tail(
        &[0],
        0,
        Tensor::from_vec(vec![4, 1], vec![1.0, 2.0, 3.0, 4.0])?,
        2,
    )?;
    assert_eq!(captures.from, 2);
    model.prefix_cache_put(&prompt, &state, None, Some(&captures))?;
    let hit = model
        .prefix_cache_get_with_captures(&prompt, &[0], 2)?
        .ok_or_else(|| InferError::Config("hit attendu".to_string()))?;
    assert_eq!(hit.3, 2);
    assert_eq!(hit.2.data(), &[3.0, 4.0]);
    assert!(model
        .prefix_cache_get_with_captures(&prompt, &[0], 4)?
        .is_none());
    assert!(model
        .prefix_cache_get_with_captures(&prompt, &[1], 2)?
        .is_none());
    Ok(())
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
fn resident_full_attention_uses_gemma4_layer_types() {
    let config = CausalDecoderConfig {
        is_gemma4: true,
        head_dim: Some(256),
        global_head_dim: Some(512),
        rope_dims: Some(256),
        rope_full_dims: Some(128),
        rope_sliding_dims: Some(256),
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_global_key_value_heads: Some(2),
        layer_types: vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ],
        ..CausalDecoderConfig::default()
    };

    assert!(config.is_full_attention_layer(0));
    assert!(config.is_resident_full_attention_layer(0));
    assert!(config.is_resident_full_attention_layer(2));
    let global_head_dim = config
        .layer_head_dim(2)
        .expect("invariant: global_head_dim Gemma4 présent");
    assert_eq!(global_head_dim, 512);
    assert_eq!(config.layer_num_key_value_heads(2), 2);
    assert_eq!(config.layer_rope_dims(2, global_head_dim), 128);
}

#[test]
fn resident_full_attention_keeps_qwen_interval_and_uniform_dims() {
    let config = CausalDecoderConfig {
        is_gemma4: false,
        head_dim: Some(128),
        global_head_dim: Some(512),
        rope_dims: Some(32),
        rope_full_dims: Some(128),
        full_attention_interval: Some(5),
        layer_types: vec![
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ],
        ..CausalDecoderConfig::default()
    };

    assert!(!config.is_resident_full_attention_layer(0));
    assert!(config.is_resident_full_attention_layer(4));
    let head_dim = config
        .layer_head_dim(4)
        .expect("invariant: head_dim Qwen présent");
    assert_eq!(head_dim, 128);
    assert_eq!(
        config.layer_num_key_value_heads(4),
        config.num_key_value_heads
    );
    assert_eq!(config.layer_rope_dims(4, head_dim), 32);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_linear_detection_uses_attention_block_variants() {
    let mut gemma_weights = HashMap::new();
    gemma_weights.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("invariant: embedding valide"),
    );
    gemma_weights.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    gemma_weights.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
            .expect("invariant: lm_head valide"),
    );
    insert_attention_layer(&mut gemma_weights, 0);
    insert_attention_layer(&mut gemma_weights, 1);
    let gemma = CausalDecoder::from_tensors(
        gemma_weights,
        CausalDecoderConfig {
            is_gemma4: true,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_global_key_value_heads: Some(1),
            head_dim: Some(2),
            global_head_dim: Some(2),
            rope_dims: Some(2),
            rope_full_dims: Some(2),
            rope_sliding_dims: Some(2),
            rope_theta: Some(10_000.0),
            sliding_window: Some(4),
            layer_types: vec![
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ],
            ..CausalDecoderConfig::default()
        },
    )
    .expect("invariant: Gemma synthétique chargeable");

    assert!(gemma.config.is_resident_full_attention_layer(0));
    assert!(matches!(
        &gemma.layers[0].attention,
        AttentionBlock::Full(attention) if attention.sliding_window == Some(4)
    ));
    assert!(!gemma.has_resident_linear_attention_layer());

    let qwen = CausalDecoder::from_tensors(hybrid_weights(2, 2), hybrid_config(2))
        .expect("invariant: hybride Qwen synthétique chargeable");
    assert!(qwen.has_resident_linear_attention_layer());
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
        causal_attention(&q, &k, &v, &windowed_layout, None).expect("invariant: attention valide");

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
            None,
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
    let windowed = cached_attention_one(&q, &mut windowed_cache, &windowed_layout, None)
        .expect("invariant: attention fenêtrée valide");
    let reference = cached_attention_one(&q, &mut truncated_cache, &full_layout, None)
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
    let out = cached_attention_one(&row([1.0, 0.0]), &mut cache, &layout, None)
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

#[test]
fn layer_kv_cache_clone_shares_rows_until_written() -> Result<()> {
    let layout = AttentionLayout {
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        rope_dims: 2,
        attn_scalar: 1.0,
        sliding_window: None,
    };
    let row = |a: f32, b: f32| Tensor::from_vec(vec![1, 2], vec![a, b]);
    let mut cache = LayerKvCache::default();
    cache.append(&row(1.0, 2.0)?, &row(3.0, 4.0)?, &layout)?;
    let snapshot = cache.clone();
    assert!(cache.keys.shares_segments_with(&snapshot.keys));
    assert!(cache.values.shares_segments_with(&snapshot.values));

    cache.append(&row(5.0, 6.0)?, &row(7.0, 8.0)?, &layout)?;
    assert_eq!(cache.len(), 2);
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot.keys.to_vec(), vec![1.0, 2.0]);
    assert_eq!(cache.keys.to_vec(), vec![1.0, 2.0, 5.0, 6.0]);
    // Le préfixe reste partagé après l'ajout : aucune recopie.
    assert!(cache.keys.shares_segments_with(&snapshot.keys));

    let shared = cache.clone();
    cache.truncate(2)?;
    assert!(cache.keys.shares_segments_with(&shared.keys));
    cache.truncate(1)?;
    assert_eq!(shared.len(), 2);
    assert_eq!(cache.len(), 1);
    Ok(())
}

#[test]
fn full_attention_append_kv_rows_matches_prefill_cache_bytes() -> Result<()> {
    // Hidden 4, deux têtes q de dim 2, une tête kv, qk norm et RoPE : le semis
    // MTP sans attention doit laisser exactement les mêmes K/V dans le cache.
    let mut tensors: HashMap<String, DecoderTensor> = HashMap::new();
    let dense = |rows: usize, cols: usize, seed: f32| -> Result<DecoderTensor> {
        let data = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.8)
            .collect();
        Ok(DecoderTensor::Dense(Tensor::from_vec(
            vec![rows, cols],
            data,
        )?))
    };
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), dense(4, 4, 0.1)?);
    tensors.insert("layers.0.self_attn.k_proj.weight".into(), dense(2, 4, 0.2)?);
    tensors.insert("layers.0.self_attn.v_proj.weight".into(), dense(2, 4, 0.3)?);
    tensors.insert("layers.0.self_attn.o_proj.weight".into(), dense(4, 4, 0.4)?);
    tensors.insert(
        "layers.0.self_attn.q_norm.weight".into(),
        DecoderTensor::Dense(Tensor::from_vec(vec![2], vec![1.5, 0.5])?),
    );
    tensors.insert(
        "layers.0.self_attn.k_norm.weight".into(),
        DecoderTensor::Dense(Tensor::from_vec(vec![2], vec![0.75, 1.25])?),
    );
    let config = CausalDecoderConfig {
        rope_theta: Some(10_000.0),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let AttentionBlock::Full(attention) =
        full_attention_from_tensors(&config, &mut tensors, "layers.0", 0)?
    else {
        return Err(InferError::Config("attention full attendue".to_string()));
    };
    let rows = |n: usize, seed: f32| -> Result<Tensor> {
        Tensor::from_vec(
            vec![n, 4],
            (0..n * 4).map(|i| (i as f32 * 0.61 + seed).cos()).collect(),
        )
    };
    let (first, second) = (rows(3, 0.0)?, rows(2, 1.0)?);
    let runtime = ForwardRuntime::cpu();

    let mut with_attention = LayerKvCache::default();
    let mut kv_only = LayerKvCache::default();
    for (normed, offset) in [(&first, 0), (&second, 3)] {
        let _ = full_attention_context_prefill(
            &config,
            normed,
            &mut with_attention,
            offset,
            &attention,
            runtime,
        )?;
        full_attention_append_kv_rows(&config, normed, &mut kv_only, offset, &attention, runtime)?;
    }
    assert_eq!(kv_only.len(), 5);
    assert_eq!(kv_only.len(), with_attention.len());
    assert_eq!(kv_only.keys.to_vec(), with_attention.keys.to_vec());
    assert_eq!(kv_only.values.to_vec(), with_attention.values.to_vec());
    assert_eq!(kv_only.kv_dim, with_attention.kv_dim);
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma4_global_attention_aliases_k_as_resident_v_projection() -> Result<()> {
    let mut tensors: HashMap<String, DecoderTensor> = HashMap::new();
    for name in ["q_proj", "k_proj", "o_proj"] {
        tensors.insert(
            format!("layers.0.self_attn.{name}.weight"),
            DecoderTensor::Dense(identity2().expect("invariant: identité valide")),
        );
    }
    for name in ["q_norm", "k_norm"] {
        tensors.insert(
            format!("layers.0.self_attn.{name}.weight"),
            DecoderTensor::Dense(
                Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norme valide"),
            ),
        );
    }
    let config = CausalDecoderConfig {
        is_gemma4: true,
        attention_k_eq_v: true,
        head_dim: Some(2),
        layer_types: vec!["full_attention".to_string()],
        ..CausalDecoderConfig::default()
    };
    let attention = full_attention_from_tensors(&config, &mut tensors, "layers.0", 0)
        .expect("invariant: attention Gemma4 valide");
    let AttentionBlock::Full(attention) = attention else {
        return Err(InferError::Config(
            "attention full Gemma4 attendue".to_string(),
        ));
    };

    assert!(attention.v_proj.is_none());
    assert!(std::ptr::eq(attention.resident_v_proj(), &attention.k_proj));
    Ok(())
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
            vocab_size: 0,
            conv_kernel_size: 0,
            conv_group_size: 0,
            selector_rank: 0,
            selector_top_k: 0,
        },
        quant: None,
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
            attention_conv: None,
            mlp_conv: None,
        }],
        norm: Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
        selector: None,
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

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn mtp_final_state_row_range_is_checked() -> Result<()> {
    assert_eq!(MtpFinalState::row_range(2, 8)?, 16..24);
    assert!(MtpFinalState::row_range(usize::MAX, 8).is_err());
    assert!(MtpFinalState::row_range(1, usize::MAX).is_err());
    Ok(())
}

#[test]
fn mtp_final_state_cpu_roundtrip() -> Result<()> {
    let state = MtpFinalState::Cpu(Tensor::row(vec![1.0, -2.0, 3.0])?);
    assert_eq!(state.into_cpu()?.data(), &[1.0, -2.0, 3.0]);
    Ok(())
}

// --- MTP : oracle résident two-row committed-history (GPU + modèle réel) ---

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_env_ready() -> bool {
    let required = [
        ("SARAGOSSA_RUST_MTP_HISTORY", "committed"),
        ("SARAGOSSA_RUST_PREFIX_CACHE", "0"),
        ("SARAGOSSA_RUST_DECODE_RESIDENT_FULL", "1"),
        ("SARAGOSSA_RUST_DECODE_RESIDENT_FULL_LINEAR", "1"),
        ("SARAGOSSA_RUST_GPU_ARGMAX", "1"),
    ];
    for (key, expected) in required {
        match std::env::var(key) {
            Ok(actual) if actual == expected => {}
            Ok(actual) => {
                eprintln!("skip MTP verifier integration: {key}={actual}, expected {expected}");
                let _ = crate::test_support::require_real_model(
                    None::<()>,
                    &format!("{key}={expected} pour le test MTP"),
                );
                return false;
            }
            Err(_) => {
                eprintln!("skip MTP verifier integration: {key} is not set to {expected}");
                let _ = crate::test_support::require_real_model(
                    None::<()>,
                    &format!("{key}={expected} pour le test MTP"),
                );
                return false;
            }
        }
    }
    true
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_verifier_model_dir() -> Option<PathBuf> {
    let model_dir = if let Some(path) =
        crate::runtime_flags::env_var_os("SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL")
    {
        let path = PathBuf::from(path);
        if path.is_dir() {
            Some(path)
        } else {
            eprintln!(
                "skip MTP verifier integration: SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL is not a directory: {}",
                path.display()
            );
            None
        }
    } else {
        eprintln!(
            "skip MTP verifier integration: model missing, set SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL"
        );
        None
    };
    crate::test_support::require_real_model(
        model_dir,
        "SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL pointant vers le modèle MTP",
    )
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
        token_constraint: None,
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
#[ignore = "requires Metal + 27B MTP model; set SARAGOSSA_RUST_MTP_HISTORY=committed and resident decode flags"]
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

// --- MTP D1 : états hidden résidents (harnais pairwise `Case::hidden`) ---

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn mtp_hidden_and_embedding_off_outside_harness() {
    #[cfg(feature = "devtools")]
    if crate::runtime_flags::pairwise::current().is_some() {
        return;
    }
    assert!(!crate::runtime_flags::mtp_resident_hidden_enabled());
    assert!(!crate::runtime_flags::mtp_gpu_embedding_enabled());
}

/// Garde pairwise D1/Fresh selon l'environnement courant, hidden forcé.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn d1_hidden_guard(resident: bool) -> Result<crate::runtime_flags::pairwise::Guard> {
    use crate::runtime_flags::pairwise::{Case, Guard, Mode};
    let mode = if super::flags::mtp_fresh_cache_enabled() {
        Mode::Fresh
    } else {
        Mode::D1
    };
    Guard::set(
        Case {
            mode,
            hidden: resident,
            ..Case::default()
        },
        true,
    )
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
fn mtp_hidden_pairwise_guard_is_scoped() -> Result<()> {
    {
        let _on = d1_hidden_guard(true)?;
        assert!(crate::runtime_flags::mtp_resident_hidden_enabled());
        {
            let _off = d1_hidden_guard(false)?;
            assert!(!crate::runtime_flags::mtp_resident_hidden_enabled());
        }
        assert!(crate::runtime_flags::mtp_resident_hidden_enabled());
    }
    assert!(!crate::runtime_flags::mtp_resident_hidden_enabled());
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn d1_transfers_reset() {
    MTP_D1_HIDDEN_TRANSFERS.with(|counts| counts.set((0, 0)));
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn d1_transfers() -> (usize, usize) {
    MTP_D1_HIDDEN_TRANSFERS.with(std::cell::Cell::get)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn d1_strict_arena() -> Result<DecodeResidentState> {
    let device = metal::Device::system_default()
        .ok_or_else(|| InferError::Metal("test D1 strict : aucun device Metal".to_string()))?;
    DecodeResidentState::new(device)
}

/// Bail résident : survit à la réallocation du pool et au drop de l'arène, sans
/// alias ; `into_cpu` relit exactement les valeurs (1 lecture, 0 upload).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requires Metal device"]
fn mtp_d1_resident_state_lease_survives_pool_reuse() -> Result<()> {
    let arena = d1_strict_arena()?;
    let values: Vec<f32> = (0..64u16).map(|i| f32::from(i) * 0.5 - 3.0).collect();
    let lease = arena.scratch().lease(values.len(), GpuElement::F32)?;
    arena.upload(lease.tensor(), &values)?;
    let kept = lease.tensor().buffer().gpu_address();
    let state = MtpFinalState::Resident(lease);

    let other = arena.scratch().lease(values.len(), GpuElement::F32)?;
    assert_ne!(
        other.tensor().buffer().gpu_address(),
        kept,
        "bail vivant aliasé"
    );
    arena.upload(other.tensor(), &[9.0; 64])?;
    let freed = other.tensor().buffer().gpu_address();
    drop(other);
    let reused = arena.scratch().lease(values.len(), GpuElement::F32)?;
    assert_ne!(
        reused.tensor().buffer().gpu_address(),
        kept,
        "slot réattribué au bail vivant"
    );
    arena.upload(reused.tensor(), &[7.0; 64])?;
    assert_eq!(
        reused.tensor().buffer().gpu_address(),
        freed,
        "le slot libéré doit être réemployé"
    );

    // Repli : `from_lease(.., false)` matérialise immédiatement.
    let cpu_lease = arena.scratch().lease(values.len(), GpuElement::F32)?;
    arena.upload(cpu_lease.tensor(), &values)?;
    d1_transfers_reset();
    let materialized = MtpFinalState::from_lease(cpu_lease, false)?;
    assert!(matches!(materialized, MtpFinalState::Cpu(_)));
    assert_eq!(d1_transfers(), (1, 0));
    assert_eq!(materialized.into_cpu()?.data(), values.as_slice());

    drop(reused);
    drop(arena);
    d1_transfers_reset();
    let cpu = state.into_cpu()?;
    assert_eq!(
        cpu.data(),
        values.as_slice(),
        "bail relu après drop de l'arène"
    );
    assert_eq!(d1_transfers(), (1, 0));
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn d1_strict_model() -> Result<(crate::ModelAssets, CausalDecoder)> {
    if !mtp_verifier_env_ready() {
        return Err(InferError::Config(
            "test D1 strict : flags MTP/résident absents".to_string(),
        ));
    }
    if super::flags::mtp_adaptive_depth_enabled() {
        return Err(InferError::Config(
            "test D1 strict : ADAPTIVE_DEPTH off requis".to_string(),
        ));
    }
    let model_dir = mtp_verifier_model_dir().ok_or_else(|| {
        InferError::Config("test D1 strict : SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL".to_string())
    })?;
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let mtp_path = assets
        .mtp
        .path
        .clone()
        .ok_or_else(|| InferError::MissingArtifact {
            path: model_dir.clone(),
            what: "MTP sidecar",
        })?;
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(&mtp_path)?;
    if model.mtp.is_none() || !model.supports_resident_full_decode() {
        return Err(InferError::Config(
            "test D1 strict : MTP résident non supporté".to_string(),
        ));
    }
    Ok((assets, model))
}

/// (sortie, (lectures, uploads) hidden D1, delta (CB, dispatchs) profilé).
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
type D1Run = (SpeculativeOutput, (usize, usize), Option<(u64, u64)>);

/// Une génération MTP sous flag D1 forcé : sortie, compteurs D1 hors prefill, et
/// delta (CB, dispatchs) si `SARAGOSSA_RUST_DECODE_PROFILE=1`.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn d1_run(
    model: &CausalDecoder,
    prompt: &[usize],
    budget: usize,
    options: &GenerationOptions,
    depth: usize,
    resident: bool,
) -> Result<D1Run> {
    let _flag = d1_hidden_guard(resident)?;
    let profile = crate::runtime_flags::decode_profile_enabled();
    let before = crate::metal_backend::decode_profile_snapshot();
    d1_transfers_reset();
    let output = model.generate_greedy_mtp_batched_with_options(prompt, budget, options, depth)?;
    let transfers = d1_transfers();
    let after = crate::metal_backend::decode_profile_snapshot();
    let delta = profile.then(|| (after.0 - before.0, after.3 - before.3));
    Ok((output, transfers, delta))
}

/// Vérifie OFF/ON == AR, stats identiques et compteurs D1 attendus.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn d1_check_off_on(
    model: &CausalDecoder,
    name: &str,
    prompt: &[usize],
    budget: usize,
    options: &GenerationOptions,
) -> Result<SpeculativeOutput> {
    let ar = model.generate_greedy_cached_with_options(prompt, budget, options)?;
    let (off, off_io, off_profile) = d1_run(model, prompt, budget, options, 1, false)?;
    let (on, on_io, on_profile) = d1_run(model, prompt, budget, options, 1, true)?;
    assert_eq!(off.tokens, ar, "{name}: D1 OFF diverge de l'AR");
    assert_eq!(on.tokens, ar, "{name}: D1 ON diverge de l'AR");
    assert_eq!(on.stats, off.stats, "{name}: stats OFF/ON différentes");
    let cycles = off.stats.proposed;
    assert_eq!(off_io, (2 * cycles, cycles), "{name}: transferts D1 OFF");
    assert_eq!(
        on_io,
        (0, usize::from(cycles > 0)),
        "{name}: transferts D1 ON"
    );
    assert_eq!(
        on_profile, off_profile,
        "{name}: (CB, dispatchs) OFF/ON différents"
    );
    eprintln!(
        "D1 hidden {name}: n={} proposed={cycles} accepted={} rejected={} off={off_io:?} on={on_io:?} profile={on_profile:?}",
        ar.len(),
        off.stats.accepted,
        off.stats.rejected
    );
    Ok(off)
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
const D1_MERGE_SORT_PROMPT: &str = "Écris une fonction Python de tri fusion avec une explication.";

/// Oracle 64 tokens : OFF == ON == AR, acceptations et rejets présents,
/// OFF=(2n,n) et ON=(0,1) transferts hidden D1, CB/dispatchs égaux si profilés.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
#[ignore = "requires Metal + 27B MTP model; committed history, resident flags; FRESH_CACHE=0 (D1) ou 1 (Fresh)"]
fn mtp_d1_resident_hidden_oracle_matches_ar() -> Result<()> {
    let (assets, model) = d1_strict_model()?;
    let prompt = mtp_verifier_prompt_ids(&assets, D1_MERGE_SORT_PROMPT)?;
    let plain = mtp_verifier_options(Vec::new());
    // Exclure l'initialisation paresseuse du contrôle de cardinalité CB/dispatchs.
    for resident in [false, true] {
        let _ = d1_run(&model, &prompt, 4, &plain, 1, resident)?;
    }
    let out = d1_check_off_on(&model, "tri-fusion-64", &prompt, 64, &plain)?;
    assert_eq!(out.tokens.len(), 64);
    assert!(out.stats.accepted > 0, "aucune acceptation D1");
    assert!(out.stats.rejected > 0, "aucun rejet D1");
    Ok(())
}

/// Budgets : séparés des arrêts pour borner chaque exécution Metal.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
#[ignore = "requires Metal + 27B MTP model; committed history, resident flags; FRESH_CACHE=0 (D1) ou 1 (Fresh)"]
fn mtp_d1_resident_hidden_edges_match_ar() -> Result<()> {
    let (assets, model) = d1_strict_model()?;
    let prompt = mtp_verifier_prompt_ids(&assets, D1_MERGE_SORT_PROMPT)?;
    let plain = mtp_verifier_options(Vec::new());
    for resident in [false, true] {
        let _ = d1_run(&model, &prompt, 4, &plain, 1, resident)?;
    }

    for budget in [0, 1, 2, 3, 4, 9] {
        let out = d1_check_off_on(&model, &format!("budget-{budget}"), &prompt, budget, &plain)?;
        assert_eq!(out.tokens.len(), budget, "budget {budget} non respecté");
    }
    Ok(())
}

/// EOS, arrêt streaming, D2 hors compteurs D1 et repli direct, sans refaire les budgets.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
#[ignore = "requires Metal + 27B MTP model; committed history, resident flags; FRESH_CACHE=0 (D1) ou 1 (Fresh)"]
fn mtp_d1_resident_hidden_stop_and_fallback_match_ar() -> Result<()> {
    let (assets, model) = d1_strict_model()?;
    let prompt = mtp_verifier_prompt_ids(&assets, D1_MERGE_SORT_PROMPT)?;
    let plain = mtp_verifier_options(Vec::new());
    for resident in [false, true] {
        let _ = d1_run(&model, &prompt, 4, &plain, 1, resident)?;
    }
    let probe = model.generate_greedy_cached_with_options(&prompt, 4, &plain)?;
    for position in 0..4 {
        let stop = *probe
            .get(position)
            .ok_or_else(|| InferError::Dimension("sonde AR D1 trop courte".to_string()))?;
        let options = mtp_verifier_options(vec![stop]);
        let out = d1_check_off_on(&model, &format!("eos-{position}"), &prompt, 16, &options)?;
        assert!(
            out.tokens.len() <= position,
            "EOS en position {position} ignoré"
        );
    }

    for stop_after in 1..=4 {
        let mut outputs = Vec::new();
        for resident in [false, true] {
            let _flag = d1_hidden_guard(resident)?;
            let mut seen = Vec::new();
            let out = model.generate_greedy_mtp_streaming_with_options(
                &prompt,
                16,
                &plain,
                1,
                |token| {
                    seen.push(token);
                    seen.len() < stop_after
                },
            )?;
            assert_eq!(
                seen.len(),
                stop_after,
                "callback supplémentaire après arrêt"
            );
            assert_eq!(seen.as_slice(), &probe[..stop_after], "tokens émis != AR");
            outputs.push((out.tokens, out.stats));
        }
        assert_eq!(
            outputs[0], outputs[1],
            "arrêt streaming {stop_after}: OFF/ON différents"
        );
    }

    let (d2_off, d2_off_io, _) = d1_run(&model, &prompt, 9, &plain, 2, false)?;
    let (d2_on, d2_on_io, _) = d1_run(&model, &prompt, 9, &plain, 2, true)?;
    assert_eq!(d2_on.tokens, d2_off.tokens, "D2 flag ON diverge");
    assert_eq!(d2_on.stats, d2_off.stats);
    assert_eq!(d2_off_io, (0, 0), "D2 ne doit pas passer par le D1 fused");
    assert_eq!(
        d2_on_io,
        (0, 0),
        "D2 ne doit pas utiliser les transferts du chemin D1"
    );

    // Repli : sans arène résidente, le D1 rend None sans lire ni uploader l'état,
    // et `into_cpu` matérialise le bail une seule fois.
    let arena = d1_strict_arena()?;
    let hidden = model.final_norm.data().len();
    let values: Vec<f32> = (0..hidden).map(|i| (i % 17) as f32 - 8.0).collect();
    let lease = arena.scratch().lease(hidden, GpuElement::F32)?;
    arena.upload(lease.tensor(), &values)?;
    let state = MtpFinalState::Resident(lease);
    let mut cache = model.empty_cache();
    d1_transfers_reset();
    assert!(model
        .next_mtp_spec_one_resident(&mut cache, &state, probe[0], 0, &[], true, None)?
        .is_none());
    assert_eq!(d1_transfers(), (0, 0));
    assert_eq!(state.into_cpu()?.data(), values.as_slice());
    assert_eq!(d1_transfers(), (1, 0));
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
#[ignore = "requires Metal + 27B MTP model; committed history and resident flags"]
fn mtp_gpu_block_d2_and_capacity_fallback_without_mutation() -> Result<()> {
    use crate::runtime_flags::pairwise::{self, Case, Guard, Hit, Mode};
    let (assets, model) = d1_strict_model()?;
    let prompt = mtp_verifier_prompt_ids(&assets, D1_MERGE_SORT_PROMPT)?;
    let options = mtp_verifier_options(Vec::new());
    let oracle = model.generate_greedy_cached_with_options(&prompt, 9, &options)?;
    assert_eq!(oracle.len(), 9);
    let _guard = Guard::set(
        Case {
            mode: Mode::Block2,
            embedding: true,
            ..Case::default()
        },
        true,
    )?;
    let fallback = model.generate_greedy_mtp_batched_with_options(&prompt, 9, &options, 2)?;
    assert_eq!(fallback.tokens, oracle);
    assert_eq!(
        pairwise::hits()[Hit::Block as usize],
        0,
        "D2 doit contourner bloc2"
    );
    let (mut cache, state) = model.prefill_cache_state(&prompt)?;
    assert!(model.setup_resident_full_decode(&mut cache, 2, 1, false)?);
    let arena = cache
        .resident
        .as_ref()
        .expect("invariant: setup résident réussi");
    let kernels = crate::metal_backend::mtp_control::MtpD1Kernels::new(arena.state.device())?;
    let position = cache.position;
    let lengths: Vec<_> = cache
        .layers
        .iter()
        .filter_map(|l| l.full.as_ref())
        .map(|kv| {
            assert_eq!(
                kv.capacity(),
                position + 3,
                "capacité volontairement insuffisante pour quatre écritures"
            );
            kv.len()
        })
        .collect();
    assert!(!lengths.is_empty());
    let before = crate::metal_backend::decode_profile_snapshot();
    let before_io = pairwise::io();
    assert!(model
        .next_mtp_gpu_block(&mut cache, &state, oracle[0], 2, &[], &kernels)?
        .is_none());
    assert_eq!(cache.position, position);
    assert_eq!(
        cache
            .layers
            .iter()
            .filter_map(|l| l.full.as_ref())
            .map(|kv| kv.len())
            .collect::<Vec<_>>(),
        lengths
    );
    assert_eq!(
        pairwise::io(),
        before_io,
        "aucun transfert partagé avant repli"
    );
    let after = crate::metal_backend::decode_profile_snapshot();
    assert_eq!(
        (after.0, after.3),
        (before.0, before.3),
        "aucune soumission ni dispatch avant repli"
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

// --- MTP échantillonné : accept/reject lossless sur le modèle réel ---

/// À température quasi nulle la trajectoire est celle de l'AR greedy, une même
/// graine rejoue la même sortie ; à 0,7 le decode accepte des drafts et va au
/// budget. Requiert `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` avec sidecar MTP.
#[test]
#[ignore = "modèle réel avec sidecar MTP requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_sampled_matches_ar_at_low_temperature_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let Some(mtp_path) = assets.mtp.path.as_ref() else {
        eprintln!("skip MTP échantillonné : sidecar MTP absent");
        return Ok(());
    };
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(mtp_path)?;
    let prompt = mtp_verifier_prompt_ids(
        &assets,
        "Write a Rust function that parses a semantic version string and returns its three components.",
    )?;
    let max_new_tokens = 32;
    let greedy = model.generate_greedy_cached_with_options(
        &prompt,
        max_new_tokens,
        &mtp_verifier_options(Vec::new()),
    )?;
    let cold = GenerationOptions {
        temperature: 0.05,
        seed: 7,
        ..mtp_verifier_options(Vec::new())
    };
    let first = model.generate_greedy_mtp_core(&prompt, max_new_tokens, &cold, 2, |_| true)?;
    let second = model.generate_greedy_mtp_core(&prompt, max_new_tokens, &cold, 2, |_| true)?;
    eprintln!(
        "MTP échantillonné T=0,05 : {} ms, {} acceptés / {} proposés",
        first.loop_duration.as_millis(),
        first.stats.accepted,
        first.stats.proposed
    );
    assert_eq!(
        first.tokens, greedy,
        "MTP échantillonné à T=0,05 divergent de l'AR greedy"
    );
    assert_eq!(
        first.tokens, second.tokens,
        "MTP échantillonné non déterministe à seed fixe"
    );
    let warm = GenerationOptions {
        temperature: 0.7,
        ..cold
    };
    let sampled = model.generate_greedy_mtp_core(&prompt, max_new_tokens, &warm, 2, |_| true)?;
    eprintln!(
        "MTP échantillonné T=0,7 : {} ms, {} acceptés / {} proposés",
        sampled.loop_duration.as_millis(),
        sampled.stats.accepted,
        sampled.stats.proposed
    );
    assert_eq!(sampled.tokens.len(), max_new_tokens);
    assert!(sampled.stats.accepted > 0, "aucun draft accepté à T=0,7");
    Ok(())
}

/// Premier token MTP émis avant le semis de l'historique : le flux émet chaque
/// token une seule fois, et un refus dès le premier rend ce seul token, à froid
/// comme en extension du cache de préfixe. Requiert
/// `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` avec sidecar MTP.
#[test]
#[ignore = "modèle réel avec sidecar MTP requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_early_first_token_streams_once_and_stops_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let Some(mtp_path) = assets.mtp.path.as_ref() else {
        eprintln!("skip premier token MTP : sidecar MTP absent");
        return Ok(());
    };
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(mtp_path)?;
    let prompt = mtp_verifier_prompt_ids(
        &assets,
        "Explain in two sentences why the borrow checker rejects aliasing mutable references.",
    )?;
    let options = mtp_verifier_options(Vec::new());
    let max_new_tokens = 24;
    let mut streamed = Vec::new();
    let full = model.generate_greedy_mtp_core(&prompt, max_new_tokens, &options, 2, |token| {
        streamed.push(token);
        true
    })?;
    assert_eq!(streamed, full.tokens, "flux MTP différent de la sortie");

    for prompt in [prompt.clone(), [prompt.as_slice(), &full.tokens].concat()] {
        let mut seen = Vec::new();
        let cut =
            model.generate_greedy_mtp_core(&prompt, max_new_tokens, &options, 2, |token| {
                seen.push(token);
                false
            })?;
        assert_eq!(seen.len(), 1, "un seul token émis avant l'arrêt");
        assert_eq!(cut.tokens, seen, "sortie après refus au premier token");
    }
    assert_eq!(
        full.tokens,
        model
            .generate_greedy_mtp_core(&prompt, max_new_tokens, &options, 2, |_| true)?
            .tokens,
        "un refus précoce ne doit pas polluer le cache de préfixe"
    );
    Ok(())
}

// --- DFlash : boucle résidente contre boucle CPU (GPU + modèle réel + draft) ---

/// Oracle de la boucle DFlash résidente : mêmes tokens que la boucle CPU et que
/// l'AR, même acceptance ; la variante streamée émet exactement ces tokens et
/// s'arrête sur le callback ; le greedy MTP route vers DFlash quand le drafter
/// est chargé. Requiert `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` et
/// `SARAGOSSA_RUST_DFLASH_DRAFT_DIR` (quantification via
/// `SARAGOSSA_RUST_DFLASH_QUANT_BITS`, u4 si absente, appliquée aux deux boucles).
#[test]
#[ignore = "modèle réel + draft DFlash requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_resident_matches_cpu_loop_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let Some(draft_dir) = crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_DRAFT_DIR")
    else {
        eprintln!("skip DFlash résident: SARAGOSSA_RUST_DFLASH_DRAFT_DIR absent");
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let mut model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_dflash_draft_from_env(&assets.config)?;
    // Sidecar MTP pour le repli après bascule (section ignorée sans sidecar).
    if let Some(mtp_path) = assets.mtp.path.as_ref() {
        model = model.with_mtp_sidecar(mtp_path)?;
    }
    let draft = model
        .dflash_draft()
        .ok_or_else(|| InferError::Config(format!("drafter non chargé depuis {draft_dir:?}")))?
        .clone();
    let prompt = mtp_verifier_prompt_ids(
        &assets,
        "Write a Rust function that parses a semantic version string and returns its three components.",
    )?;
    let options = mtp_verifier_options(Vec::new());
    let max_new_tokens = 32;
    let ar = model.generate_greedy_cached_with_options(&prompt, max_new_tokens, &options)?;
    let cpu = model.generate_greedy_dflash_batched_with_options(
        &prompt,
        max_new_tokens,
        &options,
        &draft,
        draft.info.block_size,
    )?;
    let resident = model.generate_greedy_dflash_resident_with_options(
        &prompt,
        max_new_tokens,
        &options,
        &draft,
        draft.info.block_size,
    )?;
    eprintln!(
        "DFlash résident: cpu {} ms ({} acceptés / {} proposés), résident {} ms ({} / {})",
        cpu.loop_duration.as_millis(),
        cpu.stats.accepted,
        cpu.stats.proposed,
        resident.loop_duration.as_millis(),
        resident.stats.accepted,
        resident.stats.proposed
    );
    assert_eq!(resident.tokens, ar, "boucle résidente divergente de l'AR");
    assert_eq!(cpu.tokens, ar, "boucle CPU divergente de l'AR");
    // Le drafter résident tourne sur le kernel skinny bf16 : ses propositions
    // peuvent différer de la boucle CPU f32, pas les tokens. Tolérance de deux
    // acceptations sur 32 tokens.
    assert!(
        resident.stats.accepted + 2 >= cpu.stats.accepted,
        "drafter résident moins accepté : {} contre {}",
        resident.stats.accepted,
        cpu.stats.accepted
    );
    let mut streamed = Vec::new();
    let streaming = model.generate_greedy_dflash_resident_streaming_with_options(
        &prompt,
        max_new_tokens,
        &options,
        &draft,
        draft.info.block_size,
        |token| {
            streamed.push(token);
            true
        },
    )?;
    assert_eq!(streaming.tokens, ar, "boucle streamée divergente de l'AR");
    assert_eq!(streamed, ar, "callback DFlash divergent de la sortie");
    let mut seen = 0usize;
    let early = model.generate_greedy_dflash_resident_streaming_with_options(
        &prompt,
        max_new_tokens,
        &options,
        &draft,
        draft.info.block_size,
        |_| {
            seen += 1;
            seen < 5
        },
    )?;
    assert_eq!(early.tokens, ar[..5], "arrêt par callback DFlash");
    // Stop token au milieu de la sortie AR : exclu de la sortie, que DFlash le
    // rencontre en primaire ou en draft accepté (CPU, résident, streamé).
    let stop = ar[10];
    let options_stop = mtp_verifier_options(vec![stop]);
    let ar_stop =
        model.generate_greedy_cached_with_options(&prompt, max_new_tokens, &options_stop)?;
    let cut = ar
        .iter()
        .position(|token| *token == stop)
        .unwrap_or(ar.len());
    assert_eq!(ar_stop, ar[..cut], "AR avec stop token");
    let cpu_stop = model.generate_greedy_dflash_batched_with_options(
        &prompt,
        max_new_tokens,
        &options_stop,
        &draft,
        draft.info.block_size,
    )?;
    assert_eq!(cpu_stop.tokens, ar_stop, "boucle CPU avec stop token");
    let mut streamed_stop = Vec::new();
    let resident_stop = model.generate_greedy_dflash_resident_streaming_with_options(
        &prompt,
        max_new_tokens,
        &options_stop,
        &draft,
        draft.info.block_size,
        |token| {
            streamed_stop.push(token);
            true
        },
    )?;
    assert_eq!(
        resident_stop.tokens, ar_stop,
        "boucle résidente avec stop token"
    );
    assert_eq!(streamed_stop, ar_stop, "callback avec stop token");
    // Extension du cache de préfixe : prompt + 3 tokens déjà générés, suffixe
    // et captures calculés per-op sur l'entrée précédente, même sortie que l'AR.
    let mut extended = prompt.clone();
    extended.extend_from_slice(&ar[..3]);
    let resident_ext = model.generate_greedy_dflash_resident_with_options(
        &extended,
        max_new_tokens - 3,
        &options,
        &draft,
        draft.info.block_size,
    )?;
    // Référence AR sans cache de préfixe : l'oracle ne partage pas l'état étendu.
    let ar_ext = model
        .generate_greedy_timed_from_prompt_state_with_options(
            model.prefill_prompt_state_uncached(&extended)?,
            Duration::ZERO,
            max_new_tokens - 3,
            &options,
        )?
        .tokens;
    assert_eq!(
        resident_ext.tokens, ar_ext,
        "extension du cache de préfixe avec captures"
    );
    // Repli MTP après bascule : le cœur MTP sur prompt + généré retrouve
    // l'entrée DFlash (avec historique si `SARAGOSSA_RUST_DFLASH_SWITCH_MIN` est
    // posé, sinon prefill complet) et produit la même suite que l'AR.
    if model.mtp.is_some() {
        let mtp_ext =
            model.generate_greedy_mtp_core(&extended, max_new_tokens - 3, &options, 2, |_| true)?;
        assert_eq!(mtp_ext.tokens, ar_ext, "repli MTP après extension DFlash");
    } else {
        eprintln!("skip repli MTP : sidecar MTP absent");
    }
    // Extension longue (32 lignes : matmuls GEMM, attention en lignes par
    // groupes) : prompt + toute la génération, suite comparée à l'AR non caché.
    let mut extended_long = prompt.clone();
    extended_long.extend_from_slice(&ar);
    let resident_long = model.generate_greedy_dflash_resident_with_options(
        &extended_long,
        16,
        &options,
        &draft,
        draft.info.block_size,
    )?;
    let ar_long = model
        .generate_greedy_timed_from_prompt_state_with_options(
            model.prefill_prompt_state_uncached(&extended_long)?,
            Duration::ZERO,
            16,
            &options,
        )?
        .tokens;
    assert_eq!(
        resident_long.tokens, ar_long,
        "extension longue du cache de préfixe"
    );
    // Routage : le greedy MTP passe par DFlash (8 positions de draft, pas 2).
    let routed = model.generate_greedy_mtp_streaming_with_options(
        &prompt,
        max_new_tokens,
        &options,
        2,
        |_| true,
    )?;
    assert_eq!(
        routed.tokens, ar,
        "greedy MTP routé vers DFlash divergent de l'AR"
    );
    assert_eq!(
        routed.stats.proposed_by_position.len(),
        draft.info.block_size
    );
    Ok(())
}

/// Semis MTP Metal contre l'oracle CPU sur la vraie tête : deux blocs pour
/// couvrir le décalage de position RoPE, dérive bornée. Le GEMM tuilé du FC et
/// de K|V accumule en bf16 (2e-3 relatif mesuré sur K) ; statistiques de draft
/// et texte restent identiques sur quatre prompts de 8k à 32k. Requiert
/// `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` avec sidecar MTP.
#[test]
#[ignore = "modèle réel avec sidecar MTP requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_seed_metal_matches_cpu_oracle_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let Some(mtp_path) = assets.mtp.path.as_ref() else {
        eprintln!("skip semis MTP : sidecar MTP absent");
        return Ok(());
    };
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(mtp_path)?;
    let head = model
        .mtp
        .as_ref()
        .ok_or_else(|| InferError::Config("tête MTP absente".to_string()))?;
    let metal = model
        .forward_runtime()
        .metal_executor()
        .ok_or_else(|| InferError::Config("runtime Metal absent".to_string()))?;
    let hidden = model.final_norm.data().len();
    let tokens: Vec<usize> = (0..1500).map(|i| (i * 7919 + 13) % 50_000).collect();
    // Embeddings en guise de hidden final : même échelle, sans prefill.
    let raw = model.embed_scaled(&tokens)?;
    let states = rms_norm(&raw, &model.final_norm, model.config.rms_eps)?;
    let mut cpu = LayerKvCache::default();
    let mut gpu = LayerKvCache::default();
    for (states, tokens) in states.data().chunks(1024 * hidden).zip(tokens.chunks(1024)) {
        model.seed_mtp_history_chunk_cpu(head, states, tokens, &mut cpu)?;
        let states = crate::metal_backend::MtpSeedStates::Normed(states);
        assert!(model.seed_mtp_history_chunk_metal(metal, head, states, tokens, &mut gpu)?);
    }
    // Hidden non normé lu en place, norme finale sur le GPU.
    let mut gpu_raw = LayerKvCache::default();
    let raw = metal.shared_f32_rows_for_test(raw.data())?;
    assert!(model.seed_mtp_history_raw(&raw, &tokens, &mut gpu_raw)?);
    assert_eq!(cpu.len(), gpu.len());
    assert_eq!(cpu.len(), gpu_raw.len());
    for (label, cpu, gpu) in [
        ("K", cpu.keys.to_vec(), gpu.keys.to_vec()),
        ("V", cpu.values.to_vec(), gpu.values.to_vec()),
        ("K brut", cpu.keys.to_vec(), gpu_raw.keys.to_vec()),
        ("V brut", cpu.values.to_vec(), gpu_raw.values.to_vec()),
    ] {
        assert_eq!(cpu.len(), gpu.len(), "longueur {label}");
        assert!(
            cpu.iter().chain(&gpu).all(|value| value.is_finite()),
            "{label} non fini"
        );
        let scale = cpu.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let max_err = cpu
            .iter()
            .zip(&gpu)
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        eprintln!("semis MTP {label} : max_abs={scale:e} max_err={max_err:e}");
        assert!(
            max_err <= 5e-3 * scale,
            "dérive {label} {max_err:e} / {scale:e}"
        );
    }
    Ok(())
}

/// Logits des tokens `forced` après le prompt, sur un KV résident forcé au
/// format `format` : vérification résidente alternant une ligne (boucle
/// mono-ligne) et trois lignes (chemin en lignes fusionné), comme le decode
/// AR et la vérification MTP.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn kv_quant_forced_logits(
    model: &CausalDecoder,
    prompt: &[usize],
    forced: &[usize],
    format: crate::decode_resident::KvFormat,
) -> Result<Vec<Vec<f32>>> {
    let (mut cache, _) = model.prefill_prompt_state_uncached(prompt)?.into_parts();
    let ready = crate::decode_resident::with_kv_format_override(format, || {
        model.setup_resident_full_decode(&mut cache, forced.len() + 1, 0, true)
    })?;
    if !ready {
        return Err(InferError::Config(format!(
            "setup résident refusé pour {format:?}"
        )));
    }
    // Un repli silencieux (kill-switch, forme refusée) comparerait bf16 à
    // lui-même : au moins une couche doit porter le format demandé.
    let exercised = cache
        .layers
        .iter()
        .filter_map(|layer| layer.full.as_ref())
        .any(|full| full.format() == format);
    if !exercised {
        return Err(InferError::Config(format!(
            "aucune couche n'exerce le format KV {format:?}"
        )));
    }
    let hidden = model.final_norm.data().len();
    let mut logits = Vec::with_capacity(forced.len());
    let (mut start, mut call) = (0, 0);
    while start < forced.len() {
        let rows = if call % 2 == 0 { 1 } else { 3 };
        call += 1;
        let chunk = &forced[start..(start + rows).min(forced.len())];
        let output = model
            .next_final_states_resident_verify(
                &mut cache,
                chunk,
                ResidentVerifyHead::Skip,
                false,
                None,
            )?
            .ok_or_else(|| InferError::Config("vérification résidente non exercée".into()))?;
        for row in output.states.data().chunks(hidden) {
            let state = Tensor::from_vec(vec![1, hidden], row.to_vec())?;
            logits.push(model.logits_from_final_state(&state)?.as_row()?.to_vec());
        }
        start += chunk.len();
    }
    Ok(logits)
}

/// Rang 1 et 2 d'une ligne de logits : `(argmax, marge top1 − top2)`.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn top2(logits: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best.1 {
            second = best.1;
            best = (index, value);
        } else if value > second {
            second = value;
        }
    }
    (best.0, best.1 - second)
}

/// Dérive du KV résident quantifié contre bf16 sur le vrai modèle : prompt de
/// ~6k tokens (source de ce dépôt) puis 64 tokens imposés (la suite du texte),
/// identiques pour tous les formats ; f32 donne l'échelle de la dérive bf16.
/// Affiche par format la dérive max et médiane des logits, l'accord top-1 et
/// la marge bf16 aux désaccords. Seul garde-fou : logits finis et accord
/// top-1 Q8 ≥ 90 %. Requiert `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL`.
#[test]
#[ignore = "modèle réel requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn kv_quant_logit_drift_vs_bf16_real_model() -> Result<()> {
    use crate::decode_resident::KvFormat;
    const PROMPT_TOKENS: usize = 6000;
    const FORCED: usize = 64;
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let text = concat!(
        include_str!("../decode_resident/attention.rs"),
        include_str!("../decode_resident/rows.rs"),
    );
    let ids = mtp_verifier_prompt_ids(&assets, text)?;
    let prompt_len = PROMPT_TOKENS.min(ids.len().saturating_sub(FORCED));
    let (prompt, forced) = (&ids[..prompt_len], &ids[prompt_len..prompt_len + FORCED]);
    let reference = kv_quant_forced_logits(&model, prompt, forced, KvFormat::Bf16)?;
    let mut q8_agree = 0;
    for format in [KvFormat::F32, KvFormat::Q8, KvFormat::K8V4, KvFormat::Q4] {
        let logits = kv_quant_forced_logits(&model, prompt, forced, format)?;
        assert_eq!(logits.len(), reference.len());
        let mut step_max = Vec::with_capacity(logits.len());
        let mut agree = 0;
        let mut flips = Vec::new();
        for (step, (row, base)) in logits.iter().zip(&reference).enumerate() {
            assert!(
                row.iter().all(|value| value.is_finite()),
                "{format:?} pas {step}: logit non fini"
            );
            step_max.push(
                row.iter()
                    .zip(base)
                    .fold(0.0_f32, |acc, (a, b)| acc.max((a - b).abs())),
            );
            let ((top, _), (base_top, base_margin)) = (top2(row), top2(base));
            if top == base_top {
                agree += 1;
            } else {
                flips.push(format!("pas {step} marge bf16 {base_margin:.4}"));
            }
        }
        let max = step_max.iter().copied().fold(0.0_f32, f32::max);
        step_max.sort_by(f32::total_cmp);
        let median = step_max[step_max.len() / 2];
        eprintln!(
            "kv_quant {format:?} vs bf16 (prompt {prompt_len}, {FORCED} pas) : dérive logits \
             max={max:.4} médiane={median:.4} top-1={agree}/{FORCED} désaccords=[{}]",
            flips.join(", ")
        );
        if format == KvFormat::Q8 {
            q8_agree = agree;
        }
    }
    let margins: Vec<f32> = reference.iter().map(|row| top2(row).1).collect();
    let min_margin = margins.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!("kv_quant marge top1-top2 bf16 : min={min_margin:.4}");
    assert!(
        q8_agree * 10 >= FORCED * 9,
        "Q8 : accord top-1 {q8_agree}/{FORCED} < 90 %"
    );
    Ok(())
}
