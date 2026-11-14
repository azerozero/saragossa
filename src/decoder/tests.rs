use super::attention_ops::{
    apply_rope_heads_at, attention_layout, cached_attention_one, causal_attention,
    full_attention_append_kv_rows, full_attention_context_prefill, full_attention_from_tensors,
    rms_norm_heads, rms_norm_rope_heads_at, AttentionLayout, RopeParams,
};
use super::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[path = "gemma4_oracles.rs"]
mod gemma4_oracles;
use crate::oracle::assert_close_abs as assert_close;
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
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[ignore = "requiert un device Metal ; poids synthétiques uniquement"]
fn dflash_decoder_entry_point_preserves_shared_storage(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    const CHILD: &str = "SARAGOSSA_TEST_DFLASH_STORAGE_CHILD";
    if let Ok(single_copy) = std::env::var(CHILD) {
        let target: crate::ModelConfig = serde_json::from_str(&std::fs::read_to_string(
            PathBuf::from(std::env::var("SARAGOSSA_RUST_DFLASH_DRAFT_DIR")?).join("config.json"),
        )?)?;
        let weights = test_weights()
            .into_iter()
            .map(|(key, tensor)| {
                let shape: Vec<_> = tensor
                    .shape()
                    .iter()
                    .map(|&n| if n == 2 { 64 } else { n })
                    .collect();
                Ok((key, Tensor::zeros(shape)?))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        // Passer par l'entrée publique rend observable un retour au chargeur CPU,
        // même si les tests unitaires de DraftWeights continuent à réussir.
        let decoder = CausalDecoder::from_tensors(weights, CausalDecoderConfig::default())?
            .with_metal_runtime()?
            .with_dflash_draft_from_env(&target)?;
        let draft = decoder
            .dflash_draft()
            .ok_or("drafter demandé par l'environnement absent")?;
        assert_eq!(draft.layers.len(), 1);
        let layer = &draft.layers[0];
        let (gate, up, down) = layer.mlp.projections();
        for linear in [
            &draft.fc,
            &layer.attention.q_proj,
            &layer.attention.k_proj,
            &layer.attention.v_proj,
            &layer.attention.o_proj,
            gate,
            up,
            down,
        ] {
            let LinearWeight::AffineQuantized(weight) = linear.weight() else {
                return Err("la quantification par défaut doit rester u4 gs64".into());
            };
            assert_eq!(weight.bits(), 4);
            assert_eq!(weight.group_size(), 64);
            let clone = weight.clone();
            for (view, cloned) in [
                (weight.packed_metal_view(), clone.packed_metal_view()),
                (weight.scales_metal_view(), clone.scales_metal_view()),
                (weight.biases_metal_view(), clone.biases_metal_view()),
            ] {
                assert_eq!(view.is_some(), single_copy == "1");
                if let Some((buffer, offset)) = view {
                    assert_eq!(buffer.storage_mode(), metal::MTLStorageMode::Shared);
                    assert_eq!(
                        Some((buffer.contents(), offset)),
                        cloned.map(|(b, o)| (b.contents(), o))
                    );
                }
            }
        }
        return Ok(());
    }

    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("config.json"),
        r#"{
        "architectures":["DFlashDraftModel"],
        "hidden_size":64, "num_hidden_layers":1, "num_target_layers":1,
        "intermediate_size":64, "num_attention_heads":1, "num_key_value_heads":1,
        "head_dim":64, "rms_norm_eps":0.000001, "rope_theta":10000.0, "vocab_size":3,
        "dflash_config":{"block_size":2,"mask_token_id":2,"target_layer_ids":[0]}
    }"#,
    )?;
    let mut tensors = HashMap::new();
    for key in [
        "hidden_norm",
        "norm",
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "layers.0.self_attn.q_norm",
        "layers.0.self_attn.k_norm",
    ] {
        tensors.insert(
            format!("{key}.weight"),
            Tensor::from_vec(vec![64], vec![1.0; 64])?,
        );
    }
    for key in [
        "fc",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
    ] {
        tensors.insert(
            format!("{key}.weight"),
            Tensor::from_vec(
                vec![64, 64],
                (0..4096)
                    .map(|i| ((i * 73 % 257) as f32 - 128.0) / 64.0)
                    .collect(),
            )?,
        );
    }
    std::fs::write(
        dir.path().join("model.safetensors"),
        serialize(tensor_views(tensors), None)?,
    )?;
    // Le flag single-copy est figé par OnceLock : chaque variante doit démarrer
    // dans un processus neuf, sans modifier l'environnement des autres tests.
    for single_copy in ["1", "0"] {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "decoder::tests::dflash_decoder_entry_point_preserves_shared_storage",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, single_copy)
            .env("RETI_RUST_SINGLE_COPY_WEIGHTS", single_copy)
            .env("SARAGOSSA_RUST_DFLASH_DRAFT_DIR", dir.path())
            .env_remove("SARAGOSSA_RUST_DFLASH_QUANT_BITS")
            .env_remove("RETI_RUST_DFLASH_QUANT_BITS")
            .status()?;
        assert!(status.success(), "single-copy={single_copy}: {status}");
    }
    Ok(())
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

/// `prefix_cache_put` retire l'entrée dont le prolongement a recopié le K/V et
/// toute entrée, même divergente, qui partage son allocation sans en partager
/// avec le nouvel état ; il garde celles qui partagent une allocation avec le
/// nouvel état, et les branches divergentes ou entrées plus courtes à buffer
/// propre.
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
    let turn3 = own(Some(&turn2));
    put(&[1, 2, 4], turn3.clone())?;
    assert_eq!(
        cached()?,
        vec![vec![1, 2, 4], vec![1], vec![1, 3], vec![1, 2]]
    );
    // Branche [1, 2, 4, 9] en place sur [1, 2, 4] ; le plafond de 4 entrées
    // pousse [1, 2] dehors.
    put(&[1, 2, 4, 9], own(Some(&turn3)))?;
    assert_eq!(
        cached()?,
        vec![vec![1, 2, 4, 9], vec![1, 2, 4], vec![1], vec![1, 3]]
    );
    // Prolongement par copie de [1, 2, 4] : elle est retirée, et la branche
    // divergente [1, 2, 4, 9] aussi, puisqu'elle partage l'allocation remplacée
    // et aucune du nouvel état ; [1] et [1, 3] restent.
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

/// Insertion, hits et remplacement sans captures partagent les mêmes octets ;
/// une éviction laisse le consommateur vivant sans réécrire ses données.
#[test]
#[cfg(feature = "devtools")]
fn prefix_cache_dflash_shares_captures_until_eviction() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let prompt = [1usize, 2, 3, 4];
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    let bits = [0x8000_0000, 0x7fc0_0123, 0x0000_0001, 0x3f80_0000];
    let captures = PrefixCaptures::new(
        &[0],
        0,
        Tensor::from_vec(vec![4, 1], bits.map(f32::from_bits).to_vec())?,
    );
    model.prefix_cache_put(&prompt, &state, None, Some(&captures))?;
    // Un chemin AR réinsère le même préfixe : conserver aussi le partage.
    model.prefix_cache_put(&prompt, &state, None, None)?;
    let (_, _, hidden, from) = model
        .prefix_cache_get_with_captures(&prompt, &[0], 4)?
        .ok_or_else(|| InferError::Config("hit attendu".to_string()))?;
    assert_eq!(from, 0);
    assert!(Arc::ptr_eq(&hidden, &captures.hidden));
    {
        let mut cache = model
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
        let stored = cache.entries[0]
            .captures
            .as_ref()
            .ok_or_else(|| InferError::Config("captures attendues".to_string()))?;
        assert!(Arc::ptr_eq(&hidden, &stored.hidden));
        cache.evict_lru_entry();
    }
    drop(captures);
    assert_eq!(Arc::strong_count(&hidden), 1);
    assert_eq!(
        hidden
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
    Ok(())
}

/// Le plafond compte les captures partagées une fois, même avec un consommateur
/// externe ; l'éviction ne les libère qu'au départ du dernier propriétaire.
#[test]
fn prefix_cache_byte_cap_counts_shared_captures_once() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let entry = |captures: Option<PrefixCaptures>| -> Result<PrefixCacheEntry> {
        Ok(PrefixCacheEntry {
            tokens: vec![1, 2, 3, 4],
            cache: model.empty_cache(),
            final_state: Tensor::row(vec![0.0])?,
            mtp_history: None,
            captures,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            linear_metal: Vec::new(),
        })
    };
    let base = entry(None)?.estimated_bytes_unseen(&mut Default::default());
    for consumer_alive in [true, false] {
        let captures = PrefixCaptures::new(&[0], 0, Tensor::zeros(vec![4, 8])?);
        let allocation = Arc::downgrade(&captures.hidden);
        let capture_bytes = captures.hidden.capacity_bytes() as u64;
        let mut cache = PrefixCache {
            entries: vec![
                entry(Some(captures.clone()))?,
                entry(Some(captures.clone()))?,
            ],
        };
        let consumer = consumer_alive.then_some(captures);
        let total = 2 * base + capture_bytes;
        let mut seen = Default::default();
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|entry| entry.estimated_bytes_unseen(&mut seen))
                .sum::<u64>(),
            total
        );
        cache.enforce_byte_cap(total);
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.evict_lru_entry(), Some(base));
        assert!(allocation.upgrade().is_some());
        assert_eq!(
            cache.evict_lru_entry(),
            Some(base + if consumer_alive { 0 } else { capture_bytes })
        );
        assert_eq!(allocation.upgrade().is_some(), consumer_alive);
        assert_eq!(cache.evict_lru_entry(), None);
        drop(consumer);
        assert!(allocation.upgrade().is_none());
    }
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
fn prefix_captures_tail_preserves_absolute_range_cpu() -> Result<()> {
    // Une queue déjà bornée garde sa position absolue et son allocation.
    let hidden = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let ptr = hidden.data().as_ptr();
    let captures = PrefixCaptures::tail(&[1, 0], 17, hidden, 2)?;
    assert_eq!(captures.from, 17);
    assert_eq!(captures.layer_ids, [1, 0]);
    assert_eq!(captures.hidden.data().as_ptr(), ptr);
    let captures = PrefixCaptures::tail(&[1, 0], captures.from, captures.hidden, 1)?;
    assert_eq!(captures.from, 18);
    assert_eq!(captures.hidden.data(), &[3.0, 4.0]);
    Ok(())
}

#[test]
#[cfg(feature = "devtools")]
fn prefix_captures_stitch_tail_boundaries_cpu() -> Result<()> {
    // Deux colonnes distinguent les couches, chaque valeur encode sa position.
    let captures = |from: usize, to: usize| -> Result<PrefixCaptures> {
        let data = (from..to)
            .flat_map(|row| [row as f32, 100.0 + row as f32])
            .collect();
        Ok(PrefixCaptures::new(
            &[1, 0],
            from,
            Tensor::from_vec(vec![to - from, 2], data)?,
        ))
    };
    for (prefix_from, cached_len, prompt_len, keep, suffix_from) in [
        (0, 6, 8, 4, 6),  // suffix.len() < keep, suffix_from == cached_len.
        (0, 6, 10, 4, 6), // suffix.len() == keep, frontière sans troncature.
        (0, 6, 13, 4, 9), // suffix.len() > keep, suffixe résident tronqué.
        (0, 6, 13, 4, 6), // Même suffixe entier, repli per-op.
        (3, 6, 8, 4, 6),  // Anciennes captures déjà bornées, encore nécessaires.
        (3, 6, 13, 4, 9), // Anciennes captures bornées devenues inutiles.
    ] {
        let prefix = captures(prefix_from, cached_len)?;
        let suffix = captures(suffix_from, prompt_len)?;
        let prefix_snapshot = Arc::clone(&prefix.hidden);
        let suffix_snapshot = Arc::clone(&suffix.hidden);
        let result = generation::stitch_captures(prefix, suffix, cached_len, prompt_len, keep)?;
        assert_eq!(result.from, prompt_len - keep);
        assert_eq!(
            Arc::ptr_eq(&result.hidden, &suffix_snapshot),
            prompt_len - suffix_from == keep,
            "adopter le suffixe qui couvre exactement la fenêtre"
        );
        assert_eq!(prefix_snapshot, captures(prefix_from, cached_len)?.hidden);
        assert_eq!(suffix_snapshot, captures(suffix_from, prompt_len)?.hidden);
        assert_eq!(result.from + result.hidden.as_matrix()?.0, prompt_len);
        assert_eq!(result.layer_ids, [1, 0]);
        assert_eq!(
            result.hidden,
            captures(prompt_len - keep, prompt_len)?.hidden
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "devtools")]
fn prefix_captures_stitch_rejects_shifted_from_cpu() -> Result<()> {
    // L'invariant du préfixe est exigé même quand seul le suffixe est conservé.
    for (prompt_len, suffix_from) in [(8, 6), (13, 9)] {
        for shifted_from in [2, 4] {
            let prefix = PrefixCaptures::new(&[0], shifted_from, Tensor::zeros(vec![3, 1])?);
            let suffix = PrefixCaptures::new(
                &[0],
                suffix_from,
                Tensor::zeros(vec![prompt_len - suffix_from, 1])?,
            );
            assert!(matches!(
                generation::stitch_captures(prefix, suffix, 6, prompt_len, 4),
                Err(InferError::Dimension(_))
            ));
        }
    }
    // Un suffixe décalé, ou tronqué au-delà de la fenêtre, est également refusé.
    for (suffix_from, rows) in [(5, 2), (7, 2), (7, 1)] {
        let prefix = PrefixCaptures::new(&[0], 3, Tensor::zeros(vec![3, 1])?);
        let suffix = PrefixCaptures::new(&[0], suffix_from, Tensor::zeros(vec![rows, 1])?);
        assert!(matches!(
            generation::stitch_captures(prefix, suffix, 6, 8, 4),
            Err(InferError::Dimension(_))
        ));
    }
    Ok(())
}

#[test]
#[cfg(feature = "devtools")]
fn prefix_cache_dflash_exact_hit_bounds_unlimited_captures_cpu() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let prompt = [1usize, 2, 3, 4, 5, 6];
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    let captures = PrefixCaptures::tail(
        &[0],
        0,
        Tensor::from_vec(vec![6, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?,
        usize::MAX,
    )?;
    model.prefix_cache_put(&prompt, &state, None, Some(&captures))?;
    for keep in [2, 1, 6, usize::MAX] {
        let hit = model
            .prefix_cache_get_with_captures(&prompt, &[0], keep)?
            .ok_or_else(|| InferError::Config("hit exact attendu".to_string()))?;
        let from = prompt.len().saturating_sub(keep);
        assert_eq!(hit.3, from);
        assert_eq!(hit.2.shape(), &[prompt.len() - from, 1]);
        assert_eq!(hit.2.data(), &captures.hidden.data()[from..]);
        assert_eq!(Arc::ptr_eq(&hit.2, &captures.hidden), from == 0);
    }
    // La queue copiée et le hit partagé conservent la couverture de l'entrée.
    let cache = model
        .prefix_cache
        .lock()
        .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
    let stored = cache.entries[0]
        .captures
        .as_ref()
        .ok_or_else(|| InferError::Config("captures attendues".to_string()))?;
    assert_eq!(stored.from, 0);
    assert_eq!(stored.hidden, captures.hidden);
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
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_theta_override,
            Some(10_000.0)
        );
        assert_eq!(
            LayerPlan::build(&config, layer).resident.sliding_window,
            Some(512)
        );
    }
    for layer in [5_usize, 11, 17, 23] {
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_theta_override,
            None
        );
        assert_eq!(
            LayerPlan::build(&config, layer).resident.sliding_window,
            None
        );
    }
}

#[test]
fn layer_overrides_absent_without_sliding_pattern() {
    let config = CausalDecoderConfig::default();
    for layer in 0..8 {
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_theta_override,
            None
        );
        assert_eq!(
            LayerPlan::build(&config, layer).resident.sliding_window,
            None
        );
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_position_scale,
            None
        );
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
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_position_scale,
            None
        );
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_theta_override,
            Some(10_000.0)
        );
    }
    for layer in [5_usize, 11, 17, 23] {
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_position_scale,
            Some(0.125)
        );
        assert_eq!(
            LayerPlan::build(&config, layer)
                .resident
                .rope_theta_override,
            None
        );
    }
    // Linear historique (Llama 2) : pas de motif sliding → toutes les couches.
    let uniform = CausalDecoderConfig {
        rope_position_scale: Some(0.25),
        ..CausalDecoderConfig::default()
    };
    for layer in 0..8 {
        assert_eq!(
            LayerPlan::build(&uniform, layer)
                .resident
                .rope_position_scale,
            Some(0.25)
        );
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

    assert!(LayerPlan::build(&config, 0).is_full());
    assert!(LayerPlan::build(&config, 0).resident_full);
    assert!(LayerPlan::build(&config, 2).resident_full);
    let global_head_dim = LayerPlan::build(&config, 2)
        .resident
        .head_dim()
        .expect("invariant: global_head_dim Gemma4 présent");
    assert_eq!(global_head_dim, 512);
    assert_eq!(LayerPlan::build(&config, 2).resident.kv_heads, 2);
    assert_eq!(
        LayerPlan::build(&config, 2)
            .resident
            .rope_dims(global_head_dim),
        128
    );
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

    assert!(!LayerPlan::build(&config, 0).resident_full);
    assert!(LayerPlan::build(&config, 4).resident_full);
    let head_dim = LayerPlan::build(&config, 4)
        .resident
        .head_dim()
        .expect("invariant: head_dim Qwen présent");
    assert_eq!(head_dim, 128);
    assert_eq!(
        LayerPlan::build(&config, 4).resident.kv_heads,
        config.num_key_value_heads
    );
    let plan = LayerPlan::build(&config, 4).resident;
    assert_eq!(plan.rope_dims(head_dim), 32);
    assert_eq!(plan.rope_frequency_dim(head_dim), 32);
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

    assert!(gemma.plan.layers[0].resident_full);
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
        // Même fréquence commune, mais positions brutes sans multiplication par l’échelle.
        let mut expected = x.data().to_vec();
        let inv = crate::rope::inv_freq(4, 1_000_000.0);
        for pos in 0..2 {
            let position = (7 + pos) as f32;
            for (pair, frequency) in inv.iter().enumerate() {
                let angle = position * frequency;
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

/// Verrouille l’ordre des opérations aux longues positions et la clé du cache,
/// y compris la dimension fréquentielle globale distincte de la largeur rotée.
#[test]
fn rope_rotations_use_scaled_positions_times_common_frequencies() -> Result<()> {
    for frequency_dim in [128, 256] {
        for theta in [10_000.0, 1_000_000.0] {
            let inv = crate::rope::inv_freq(frequency_dim, theta);
            for rope_dims in [frequency_dim, frequency_dim / 2] {
                for position_scale in [1.0, 0.125] {
                    let rope = RopeParams {
                        theta,
                        frequency_dim,
                        position_scale,
                    };
                    for position in [5, 2051, 32_767, 131_071, 262_143] {
                        let rotations =
                            super::attention_ops::rope_rotations(position, rope_dims, rope)?;
                        let cached =
                            super::attention_ops::rope_rotations(position, rope_dims, rope)?;
                        assert!(std::sync::Arc::ptr_eq(&rotations, &cached));
                        assert_eq!(rotations.len(), rope_dims / 2);
                        for ((cos, sin), frequency) in rotations.iter().zip(&inv) {
                            let angle = (position as f32 * position_scale) * frequency;
                            // `cosf`/`sinf` peuvent différer d'un ulp entre debug
                            // et release (optimisations de libm) : 2 ulp admis.
                            // L'ancienne forme `/powf` décale l'angle d'environ
                            // un ulp d'angle, soit bien plus de 2 ulp de cos/sin
                            // aux positions longues.
                            let ulps =
                                |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
                            assert!(ulps(*cos, angle.cos()) <= 2, "cos à {position}");
                            assert!(ulps(*sin, angle.sin()) <= 2, "sin à {position}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
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
    let attention = full_attention_from_tensors(
        &config,
        &mut tensors,
        "layers.0",
        &LayerPlan::build(&config, 0).resident,
    )
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
    let AttentionBlock::Full(attention) = full_attention_from_tensors(
        &config,
        &mut tensors,
        "layers.0",
        &LayerPlan::build(&config, 0).resident,
    )?
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
    let attention = full_attention_from_tensors(
        &config,
        &mut tensors,
        "layers.0",
        &LayerPlan::build(&config, 0).resident,
    )
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

pub(super) fn test_weights() -> HashMap<String, Tensor> {
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
pub(super) fn tiny_dflash_draft() -> crate::DFlashDraft {
    crate::DFlashDraft {
        snapshot_sources: Default::default(),
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

pub(super) fn hybrid_weights(
    layer_count: usize,
    full_attention_interval: usize,
) -> HashMap<String, Tensor> {
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

pub(super) fn hybrid_config(layer_count: usize) -> CausalDecoderConfig {
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

pub(super) struct F32View {
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

pub(super) fn tensor_views(tensors: HashMap<String, Tensor>) -> Vec<(String, F32View)> {
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

/// Le cap du modèle doit atteindre le sampler non résident partagé avec MTP.
/// Le greedy conserve son argmax et n'avance pas le RNG.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn softcap_sampling_decoder_routes_and_greedy() -> Result<()> {
    let Some(mut model) = resident_test_model() else {
        return Ok(());
    };
    model.config.final_logit_softcapping = Some(30.0);
    model.lm_head = Linear::new(
        Tensor::from_vec(
            vec![3, 4],
            vec![
                45.0, 0.0, 0.0, 0.0, 38.0, 0.0, 0.0, 0.0, 31.0, 0.0, 0.0, 0.0,
            ],
        )?,
        None,
    )?;
    let state = Tensor::row(vec![1.0, 0.0, 0.0, 0.0])?;
    let logits = model.logits_from_final_state(&state)?;
    let options = GenerationOptions {
        temperature: 2.5,
        top_k: 2,
        top_p: 0.9,
        ..GenerationOptions::default()
    };
    let mut differences = 0;
    for seed in 0..128 {
        let mut cpu_rng = DeterministicSampler::new(seed);
        let expected = sample_token_top_k_top_p(
            logits.as_row()?,
            options.temperature,
            options.top_p,
            options.top_k,
            &mut cpu_rng,
        )?;
        let raw = sample_token_top_k_top_p(
            &[45.0, 38.0, 31.0],
            options.temperature,
            options.top_p,
            options.top_k,
            &mut DeterministicSampler::new(seed),
        )?;
        differences += usize::from(raw != expected);
        let mut rng = DeterministicSampler::new(seed);
        assert_eq!(
            model.sample_token_from_state(&state, &options, &mut rng)?,
            expected
        );
        assert_eq!(rng.state(), cpu_rng.state());
    }
    assert!(differences > 10, "l'oracle doit détecter l'absence de cap");
    for cap in [None, Some(30.0)] {
        model.config.final_logit_softcapping = cap;
        let mut rng = DeterministicSampler::new(123);
        assert_eq!(
            model.sample_token_from_state(&state, &GenerationOptions::default(), &mut rng)?,
            0
        );
        assert_eq!(rng.state(), 123);
    }
    Ok(())
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

/// Stop sequences couvrant chaque position de `generated` : pour `end`, le plus
/// court suffixe de `generated[..=end]` dont la première occurrence finit en
/// `end`. L'AR avec cette seule séquence rend donc exactement
/// `generated[..=end]` ; balayer toutes les positions couvre les arrêts au
/// milieu des drafts acceptés comme les séquences à cheval sur deux tours.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn stop_sequence_cases(generated: &[usize]) -> Vec<(Vec<usize>, usize)> {
    (0..generated.len())
        .map(|end| {
            let sequence = (1..=end + 1)
                .map(|len| &generated[end + 1 - len..=end])
                .find(|suffix| (0..end).all(|i| !generated[..=i].ends_with(suffix)))
                .unwrap_or(&generated[..=end]);
            (sequence.to_vec(), end)
        })
        .collect()
}

/// Stop sequence qui traverse une frontière de tour observée sur un flux
/// horodaté : chaque tour émet ses tokens en rafale, séparées par le travail
/// GPU. Rend la plus courte séquence (première occurrence) finissant en `end`
/// et commençant avant `end`, où `end` ouvre une rafale et est suivi d'un token
/// de la même rafale : le stop tombe au milieu des tokens d'un tour, la
/// séquence commence dans le tour précédent.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn crossing_stop_case(tokens: &[usize], times: &[Instant]) -> Option<(Vec<usize>, usize)> {
    let gap = Duration::from_millis(2);
    let burst = Duration::from_micros(500);
    // Au-delà des premiers tokens : le premier est émis à part (prefill ou
    // émission anticipée), la frontière cherchée est celle d'un tour plein.
    (3..tokens.len().saturating_sub(1)).find_map(|end| {
        let opens_burst = times[end].duration_since(times[end - 1]) >= gap
            && times[end + 1].duration_since(times[end]) <= burst;
        if !opens_burst {
            return None;
        }
        let sequence = (2..=end + 1)
            .map(|len| &tokens[end + 1 - len..=end])
            .find(|suffix| (0..end).all(|i| !tokens[..=i].ends_with(suffix)))?;
        Some((sequence.to_vec(), end))
    })
}

/// Annulation par le callback avant ou sur le stop : aucun callback après
/// `false`, tokens émis = préfixe de l'oracle = tokens rendus.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn assert_cancelled_stream(
    label: &str,
    oracle: &[usize],
    cancel_at: usize,
    run: impl FnOnce(&mut dyn FnMut(usize) -> bool) -> Result<SpeculativeOutput>,
) -> Result<()> {
    let mut seen = Vec::new();
    let output = run(&mut |token| {
        seen.push(token);
        seen.len() <= cancel_at
    })?;
    assert_eq!(
        seen,
        oracle[..=cancel_at],
        "{label}, annulation en {cancel_at}"
    );
    assert_eq!(
        output.tokens, seen,
        "{label}, sortie après annulation en {cancel_at}"
    );
    Ok(())
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

/// Stop sequences dans le cœur MTP sans drafter : sortie et flux identiques à
/// l'AR avec les mêmes séquences, à toutes les positions, en défaut de
/// production (fresh adaptatif) et forcés D1, D2, émission différée et lookup
/// (vérifieur général). En échantillonné, la sortie est le préfixe de la
/// trajectoire sans séquence, terminé par la séquence. Requiert
/// `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` avec sidecar MTP.
#[test]
#[ignore = "modèle réel avec sidecar MTP requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn mtp_stop_sequences_match_ar_real_model() -> Result<()> {
    use crate::runtime_flags::pairwise::{Case, Guard, Mode};
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let Some(mtp_path) = assets.mtp.path.as_ref() else {
        eprintln!("skip stop sequences MTP : sidecar MTP absent");
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
    let options = mtp_verifier_options(Vec::new());
    let ar = model.generate_greedy_cached_with_options(&prompt, max_new_tokens, &options)?;
    let with_sequence = |sequence: Vec<usize>, base: &GenerationOptions| GenerationOptions {
        stop_sequences: vec![sequence],
        ..base.clone()
    };
    let cases = stop_sequence_cases(&ar);
    for (sequence, end) in &cases {
        let ar_seq = model.generate_greedy_cached_with_options(
            &prompt,
            max_new_tokens,
            &with_sequence(sequence.clone(), &options),
        )?;
        assert_eq!(ar_seq, ar[..=*end], "AR, stop sequence en {end}");
    }
    // Frontière de tour observée sur le flux par défaut (horodaté).
    let mut times = Vec::new();
    let timed = model.generate_greedy_mtp_streaming_with_options(
        &prompt,
        max_new_tokens,
        &options,
        2,
        |_| {
            times.push(Instant::now());
            true
        },
    )?;
    assert_eq!(timed.tokens, ar, "MTP horodaté divergent de l'AR");
    let crossing = crossing_stop_case(&ar, &times)
        .ok_or_else(|| InferError::Config("aucune frontière de tour MTP observée".into()))?;
    eprintln!(
        "stop sequence MTP à cheval : {:?} en {}",
        crossing.0, crossing.1
    );
    let mut cases = cases;
    cases.push(crossing.clone());
    let configs = [
        ("défaut", None),
        ("D1", Some(Case::default())),
        (
            "D1 différé",
            Some(Case {
                overlap: true,
                ..Case::default()
            }),
        ),
        (
            "D2",
            Some(Case {
                mode: Mode::D2,
                ..Case::default()
            }),
        ),
        (
            "D2 différé",
            Some(Case {
                mode: Mode::D2,
                overlap: true,
                ..Case::default()
            }),
        ),
        (
            "lookup",
            Some(Case {
                lookup: 8,
                ..Case::default()
            }),
        ),
        (
            "fresh",
            Some(Case {
                mode: Mode::Fresh,
                ..Case::default()
            }),
        ),
    ];
    for (name, case) in configs {
        let _guard = case.map(|case| Guard::set(case, false)).transpose()?;
        let depth = case.map_or(2, Case::depth);
        // Le chemin attendu accepte bien des drafts (deux par tour en D2).
        let free = model.generate_greedy_mtp_batched_with_options(
            &prompt,
            max_new_tokens,
            &options,
            depth,
        )?;
        assert_eq!(free.tokens, ar, "MTP {name} sans séquence");
        assert!(free.stats.accepted > 0, "MTP {name} : aucun draft accepté");
        if depth == 2 && case.is_some() {
            assert!(
                free.stats.accepted_by_position[1] > 0,
                "MTP {name} : second draft jamais accepté"
            );
        }
        let crossing_options = with_sequence(crossing.0.clone(), &options);
        for cancel_at in [crossing.1 - 1, crossing.1] {
            assert_cancelled_stream(&format!("MTP {name}"), &ar, cancel_at, |on_token| {
                model.generate_greedy_mtp_streaming_with_options(
                    &prompt,
                    max_new_tokens,
                    &crossing_options,
                    depth,
                    on_token,
                )
            })?;
        }
        for (sequence, end) in &cases {
            let mut streamed = Vec::new();
            let spec = model.generate_greedy_mtp_streaming_with_options(
                &prompt,
                max_new_tokens,
                &with_sequence(sequence.clone(), &options),
                depth,
                |token| {
                    streamed.push(token);
                    true
                },
            )?;
            assert_eq!(
                spec.tokens,
                ar[..=*end],
                "MTP {name}, stop sequence en {end}"
            );
            assert_eq!(
                streamed, spec.tokens,
                "flux MTP {name}, stop sequence en {end}"
            );
        }
        eprintln!("stop sequences MTP {name} : {} positions ok", cases.len());
    }
    // Plusieurs séquences et stop token au même token, chemin par défaut.
    let multiple = GenerationOptions {
        stop_sequences: vec![cases[20].0.clone(), cases[8].0.clone()],
        ..options.clone()
    };
    let spec =
        model.generate_greedy_mtp_batched_with_options(&prompt, max_new_tokens, &multiple, 2)?;
    assert_eq!(spec.tokens, ar[..=8], "MTP, séquences multiples");
    let both = GenerationOptions {
        stop_sequences: vec![cases[10].0.clone()],
        ..mtp_verifier_options(vec![ar[10]])
    };
    let ar_both = model.generate_greedy_cached_with_options(&prompt, max_new_tokens, &both)?;
    let spec = model.generate_greedy_mtp_batched_with_options(&prompt, max_new_tokens, &both, 2)?;
    assert_eq!(
        spec.tokens, ar_both,
        "MTP, stop token et séquence au même token"
    );
    // Échantillonné : préfixe de la trajectoire libre, à seed fixe.
    let warm = GenerationOptions {
        temperature: 0.7,
        seed: 7,
        ..options.clone()
    };
    let free = model.generate_greedy_mtp_core(&prompt, max_new_tokens, &warm, 2, |_| true)?;
    for (sequence, end) in stop_sequence_cases(&free.tokens).into_iter().step_by(3) {
        let mut streamed = Vec::new();
        let sampled = model.generate_greedy_mtp_core(
            &prompt,
            max_new_tokens,
            &with_sequence(sequence, &warm),
            2,
            |token| {
                streamed.push(token);
                true
            },
        )?;
        assert_eq!(streamed, sampled.tokens, "flux MTP échantillonné en {end}");
        assert_eq!(
            sampled.tokens,
            free.tokens[..=end],
            "MTP échantillonné, stop sequence en {end}"
        );
    }
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

// --- DFlash : vérification en arbre (GPU + modèle réel + draft) ---

/// Vérification en arbre DFlash 2 sur le modèle réel :
/// - la chaîne gloutonne passée par la plomberie arbre (scan arbre, rollback
///   par chemin, compaction KV, rassemblement des hidden) redonne exactement la
///   boucle en chaîne : tokens, statistiques et contexte du drafter ;
/// - l'arbre best-first B=8 sort les tokens de l'AR et de la boucle CPU f32
///   (code et prose), y compris sous KV bf16, arrêt par callback, stop token
///   et petites limites de sortie ; B = 1, 4, 10, 12, 15 aussi, B = 16 refusé ;
/// - le contexte du drafter (hidden des lignes du chemin) reste celui de la
///   chaîne aux réductions près.
///
/// Requiert `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` et
/// `SARAGOSSA_RUST_DFLASH_DRAFT_DIR`.
#[test]
#[ignore = "modèle réel + draft DFlash requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_tree_resident_matches_chain_and_ar_real_model() -> Result<()> {
    use crate::decoder::resident::dflash::{DFlashTreeMode, DFLASH_APPENDED_CONTEXT};
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    if crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_DRAFT_DIR").is_none() {
        eprintln!("skip DFlash arbre: SARAGOSSA_RUST_DFLASH_DRAFT_DIR absent");
        return Ok(());
    }
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_dflash_draft_from_env(&assets.config)?;
    let draft = model
        .dflash_draft()
        .ok_or_else(|| InferError::Config("drafter DFlash non chargé".to_string()))?
        .clone();
    let block = draft.info.block_size;
    let options = mtp_verifier_options(Vec::new());
    let run = |prompt: &[usize], max_new: usize, options: &GenerationOptions, mode| {
        model.generate_greedy_dflash_resident_tree(
            prompt,
            max_new,
            options,
            &draft,
            block,
            (mode, true),
            |_| true,
        )
    };
    let recorded =
        |prompt: &[usize], max_new: usize, mode| -> Result<(SpeculativeOutput, Vec<f32>)> {
            DFLASH_APPENDED_CONTEXT.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
            let output = run(prompt, max_new, &options, mode);
            let context = DFLASH_APPENDED_CONTEXT.with(|slot| slot.borrow_mut().take());
            Ok((output?, context.unwrap_or_default()))
        };
    let cols = draft.info.target_layer_ids.len() * model.final_norm.len();
    for text in [
        "Write a Rust function that parses a semantic version string and returns its three components.",
        "Raconte en quelques phrases l'histoire d'un phare isolé sur une côte bretonne.",
    ] {
        let prompt = mtp_verifier_prompt_ids(&assets, text)?;
        let max_new = 64;
        let ar = model.generate_greedy_cached_with_options(&prompt, max_new, &options)?;
        let (chain, chain_context) = recorded(&prompt, max_new, DFlashTreeMode::Off)?;
        let (plumbing, plumbing_context) = recorded(&prompt, max_new, DFlashTreeMode::Chain)?;
        let (tree, tree_context) = recorded(&prompt, max_new, DFlashTreeMode::BestFirst(8))?;
        eprintln!(
            "DFlash arbre « {} » : chaîne {} acceptés / {} vérifiés, arbre B=8 {} / {} ({} tokens)",
            &text[..20],
            chain.stats.accepted,
            chain.stats.proposed,
            tree.stats.accepted,
            tree.stats.proposed,
            tree.tokens.len()
        );
        assert_eq!(chain.tokens, ar, "chaîne divergente de l'AR");
        // Plomberie arbre en chaîne dégénérée : bit à bit la chaîne.
        assert_eq!(plumbing.tokens, chain.tokens, "plomberie arbre : tokens");
        assert_eq!(plumbing.stats.accepted, chain.stats.accepted);
        assert_eq!(plumbing.stats.proposed, chain.stats.proposed);
        assert_eq!(plumbing.stats.rejected, chain.stats.rejected);
        assert_eq!(
            plumbing.stats.accepted_by_position,
            chain.stats.accepted_by_position
        );
        // Le dernier tour s'arrête à la limite de sortie : la chaîne appende
        // encore ses lignes, l'arbre non (génération finie). Préfixe commun.
        let common = plumbing_context.len().min(chain_context.len());
        assert!(chain_context.len() - common <= (block + 1) * cols);
        let first_diff = (0..common).find(|&i| plumbing_context[i].to_bits() != chain_context[i].to_bits());
        assert_eq!(
            first_diff.map(|i| (i / cols, i % cols)),
            None,
            "plomberie arbre : contexte du drafter (ligne, colonne) sur {} lignes",
            common / cols
        );
        assert_eq!(tree.tokens, ar, "arbre B=8 divergent de l'AR");
        // Oracle f32 : la boucle CPU de référence sort les mêmes tokens.
        let cpu = model.generate_greedy_dflash_batched_with_options(
            &prompt, max_new, &options, &draft, block,
        )?;
        assert_eq!(cpu.tokens, tree.tokens, "arbre B=8 contre la boucle CPU f32");
        // Contexte du drafter : mêmes positions dans le même ordre, calculées
        // dans des lots différents (écart de réduction seul). Une ligne mal
        // rassemblée s'écarterait d'un ordre de grandeur.
        let rows = chain_context.len().min(tree_context.len()) / cols;
        assert!(rows > prompt.len(), "contexte du drafter trop court ({rows} lignes)");
        let mut worst = 0.0_f32;
        for row in 0..rows {
            let range = row * cols..(row + 1) * cols;
            let scale = chain_context[range.clone()]
                .iter()
                .fold(0.0_f32, |m, v| m.max(v.abs()))
                .max(1.0);
            let delta = chain_context[range.clone()]
                .iter()
                .zip(&tree_context[range])
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()))
                / scale;
            worst = worst.max(delta);
        }
        eprintln!("DFlash arbre : écart relatif max du contexte drafter {worst:e} sur {rows} lignes");
        assert!(worst <= 1.0e-3, "contexte du drafter divergent : {worst:e}");
    }
    // Bords : KV bf16 (compaction bf16), arrêt par callback, stop token, petites
    // limites de sortie (profondeur bornée par les tokens restants).
    let prompt = mtp_verifier_prompt_ids(
        &assets,
        "Write a Rust function that parses a semantic version string and returns its three components.",
    )?;
    let ar = model.generate_greedy_cached_with_options(&prompt, 32, &options)?;
    let bf16 = crate::decode_resident::with_kv_format_override(
        crate::decode_resident::KvFormat::Bf16,
        || run(&prompt, 32, &options, DFlashTreeMode::BestFirst(8)),
    )?;
    assert_eq!(bf16.tokens, ar, "arbre B=8 sous KV bf16");
    let mut seen = 0usize;
    let early = model.generate_greedy_dflash_resident_tree(
        &prompt,
        32,
        &options,
        &draft,
        block,
        (DFlashTreeMode::BestFirst(8), true),
        |_| {
            seen += 1;
            seen < 5
        },
    )?;
    assert_eq!(early.tokens, ar[..5], "arbre : arrêt par callback");
    let stop = ar[10];
    let options_stop = mtp_verifier_options(vec![stop]);
    let ar_stop = model.generate_greedy_cached_with_options(&prompt, 32, &options_stop)?;
    let tree_stop = run(&prompt, 32, &options_stop, DFlashTreeMode::BestFirst(8))?;
    assert_eq!(tree_stop.tokens, ar_stop, "arbre avec stop token");
    for max_new in [1, 2, 3, 9] {
        let tree = run(&prompt, max_new, &options, DFlashTreeMode::BestFirst(8))?;
        assert_eq!(tree.tokens, ar[..max_new], "arbre, {max_new} tokens");
    }
    // Budgets sous la chaîne et au-delà (jusqu'aux 16 lignes du journal).
    for budget in [1, 4, 10, 12, 15] {
        let tree = run(&prompt, 32, &options, DFlashTreeMode::BestFirst(budget))?;
        assert_eq!(tree.tokens, ar, "arbre B={budget}");
    }
    // Stop sequences, toutes positions : oracle AR par construction de
    // `stop_sequence_cases` (vérifié dans le test de la chaîne), B = 8 partout,
    // B = 1 sur quelques positions.
    for (sequence, end) in stop_sequence_cases(&ar) {
        let mut options_seq = options.clone();
        options_seq.stop_sequences = vec![sequence];
        let tree = run(&prompt, 32, &options_seq, DFlashTreeMode::BestFirst(8))?;
        assert_eq!(tree.tokens, ar[..=end], "arbre B=8, stop sequence en {end}");
        if [3, 11, 22].contains(&end) {
            let tree = run(&prompt, 32, &options_seq, DFlashTreeMode::BestFirst(1))?;
            assert_eq!(tree.tokens, ar[..=end], "arbre B=1, stop sequence en {end}");
        }
    }
    // Au-delà du journal (17 lignes) : refus explicite avant toute génération.
    assert!(run(&prompt, 32, &options, DFlashTreeMode::BestFirst(16)).is_err());
    Ok(())
}

// --- DFlash : boucle résidente contre boucle CPU (GPU + modèle réel + draft) ---

/// Oracle de la boucle DFlash résidente : mêmes tokens que la boucle CPU et que
/// l'AR, même acceptance que la boucle CPU pour la chaîne résidente (tolérance
/// de deux) ; la variante streamée émet exactement ces tokens et
/// s'arrête sur le callback ; le greedy MTP route vers DFlash quand le drafter
/// est chargé. Requiert `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` et
/// `SARAGOSSA_RUST_DFLASH_DRAFT_DIR` (quantification via
/// `SARAGOSSA_RUST_DFLASH_QUANT_BITS`, u4 si absente, appliquée aux deux boucles).
#[test]
#[ignore = "modèle réel + draft DFlash requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_resident_matches_cpu_loop_real_model() -> Result<()> {
    use crate::decoder::resident::dflash::DFlashTreeMode;
    // NOTE: les points d'entrée publics lisent `SARAGOSSA_RUST_DFLASH_TREE` ; sans
    // la variable, l'identité des tokens porte sur l'arbre par défaut (B = 8, ou
    // B = 3 en greedy strict). L'acceptance comparée à la boucle CPU porte, quelle
    // que soit la variable, sur la chaîne résidente explicite (mode `Off`).
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
    // Plancher du chemin par défaut : un arbre qui n'accepte rien garderait les
    // tokens de l'AR par construction.
    assert!(
        resident.stats.accepted > 0,
        "arbre résident par défaut : aucun draft accepté"
    );
    // L'acceptance se compare à structure égale : la boucle CPU est une chaîne,
    // alors que le résident par défaut est un arbre dont le budget plafonne les
    // acceptés par tour (B = 3 en greedy strict, d'où 23 contre 26 sur ce
    // prompt). Le résident est donc forcé en chaîne pour cette comparaison.
    let chain = model.generate_greedy_dflash_resident_tree(
        &prompt,
        max_new_tokens,
        &options,
        &draft,
        draft.info.block_size,
        (DFlashTreeMode::Off, true),
        |_| true,
    )?;
    eprintln!(
        "DFlash résident en chaîne: {} acceptés / {} proposés",
        chain.stats.accepted, chain.stats.proposed
    );
    assert_eq!(chain.tokens, ar, "chaîne résidente divergente de l'AR");
    // Le drafter résident tourne sur le kernel skinny bf16 : ses propositions
    // peuvent différer de la boucle CPU f32, pas les tokens. Tolérance de deux
    // acceptations sur 32 tokens.
    assert!(
        chain.stats.accepted + 2 >= cpu.stats.accepted,
        "drafter résident moins accepté : {} contre {}",
        chain.stats.accepted,
        cpu.stats.accepted
    );
    // Au-delà du seuil greedy, l'arène du vérifieur passe en bf16
    // (`resident_kv_bf16`) : forcée ici sur un prompt court, déjà en cache de
    // préfixe, elle exerce adoption, vérification bf16 et rollback par troncature.
    let resident_bf16 = crate::decode_resident::with_kv_format_override(
        crate::decode_resident::KvFormat::Bf16,
        || {
            model.generate_greedy_dflash_resident_with_options(
                &prompt,
                max_new_tokens,
                &options,
                &draft,
                draft.info.block_size,
            )
        },
    )?;
    assert_eq!(
        resident_bf16.tokens, ar,
        "boucle résidente bf16 divergente de l'AR"
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
    // Stop sequences (incluses dans la sortie) contre l'oracle AR, toutes
    // positions. Un tour au moins accepte deux drafts : une séquence finissant
    // sur le premier tombe au milieu des tokens acceptés.
    assert!(
        resident
            .stats
            .accepted_by_position
            .get(1)
            .is_some_and(|&count| count > 0),
        "aucun tour à deux drafts acceptés : milieu de tour non couvert"
    );
    let with_stops = |sequences: Vec<Vec<usize>>, stop_token_ids: Vec<usize>| {
        let mut options = mtp_verifier_options(stop_token_ids);
        options.stop_sequences = sequences;
        options
    };
    // Point d'entrée public (selon l'environnement ; sinon arbre B = 8, ou B = 3 en
    // greedy strict) et chaîne explicite.
    let resident_modes = [
        ("défaut", None),
        ("chaîne", Some((DFlashTreeMode::Off, true))),
    ];
    let run_resident = |options: &GenerationOptions,
                        mode: Option<(DFlashTreeMode, bool)>,
                        on_token: &mut dyn FnMut(usize) -> bool| {
        match mode {
            None => model.generate_greedy_dflash_resident_streaming_with_options(
                &prompt,
                max_new_tokens,
                options,
                &draft,
                draft.info.block_size,
                on_token,
            ),
            Some(mode) => model.generate_greedy_dflash_resident_tree(
                &prompt,
                max_new_tokens,
                options,
                &draft,
                draft.info.block_size,
                mode,
                on_token,
            ),
        }
    };
    let check_chain = |options: &GenerationOptions, label: &str| -> Result<Vec<usize>> {
        let ar = model.generate_greedy_cached_with_options(&prompt, max_new_tokens, options)?;
        for (mode_name, mode) in resident_modes {
            let mut streamed = Vec::new();
            let output = run_resident(options, mode, &mut |token| {
                streamed.push(token);
                true
            })?;
            assert_eq!(
                output.tokens, ar,
                "DFlash {mode_name}, stop sequence {label}"
            );
            assert_eq!(streamed, ar, "callback {mode_name}, stop sequence {label}");
        }
        Ok(ar)
    };
    for (sequence, end) in stop_sequence_cases(&ar) {
        let label = format!("finissant en {end}");
        let ar_seq = check_chain(&with_stops(vec![sequence], Vec::new()), &label)?;
        assert_eq!(ar_seq, ar[..=end], "AR, stop sequence {label}");
    }
    let cases = stop_sequence_cases(&ar);
    // Plusieurs séquences : la première terminée gagne.
    let multiple = with_stops(vec![cases[20].0.clone(), cases[8].0.clone()], Vec::new());
    assert_eq!(check_chain(&multiple, "multiple")?, ar[..=8]);
    // Séquence présente seulement dans le prompt : sans effet.
    if let Some(window) = prompt
        .windows(2)
        .find(|window| !ar.windows(2).any(|other| other == *window))
    {
        let prompt_only = with_stops(vec![window.to_vec()], Vec::new());
        assert_eq!(check_chain(&prompt_only, "du prompt")?, ar);
    }
    // Stop token et séquence finissant au même token : le stop token gagne.
    let both = with_stops(vec![cases[10].0.clone()], vec![ar[10]]);
    check_chain(&both, "et stop token")?;
    // Frontière de tour observée (flux horodaté) : séquence commencée dans un
    // tour, stop au milieu des drafts acceptés du suivant ; annulation par le
    // callback juste avant et sur le stop.
    for (mode_name, mode) in resident_modes {
        let mut times = Vec::new();
        let timed = run_resident(&options, mode, &mut |_| {
            times.push(Instant::now());
            true
        })?;
        assert_eq!(timed.tokens, ar, "DFlash {mode_name} horodaté");
        let (sequence, end) = crossing_stop_case(&ar, &times).ok_or_else(|| {
            InferError::Config(format!("aucune frontière de tour DFlash {mode_name}"))
        })?;
        eprintln!("stop sequence DFlash {mode_name} à cheval : {sequence:?} en {end}");
        let crossing = with_stops(vec![sequence], Vec::new());
        assert_eq!(check_chain(&crossing, "à cheval")?, ar[..=end]);
        for cancel_at in [end - 1, end] {
            assert_cancelled_stream(&format!("DFlash {mode_name}"), &ar, cancel_at, |on_token| {
                run_resident(&crossing, mode, on_token)
            })?;
        }
    }
    // Routage MTP → DFlash avec stop sequence.
    let routed_seq = model.generate_greedy_mtp_streaming_with_options(
        &prompt,
        max_new_tokens,
        &with_stops(vec![cases[13].0.clone()], Vec::new()),
        2,
        |_| true,
    )?;
    assert_eq!(
        routed_seq.tokens,
        ar[..=13],
        "routage DFlash, stop sequence"
    );
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

/// Poids d'un Qwen dense synthétique à une couche full-attention d256
/// (16 têtes Q, 2 K/V, QK-norm, MLP SwiGLU) : la forme qu'emprunte le
/// prefill résident par morceaux avec K/V stocké en bf16 (NAX d256).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_bf16_qwen_weights(
    vocab: usize,
    hidden: usize,
    attn_output_gate: bool,
) -> Result<HashMap<String, Tensor>> {
    let (q_heads, kv_heads, head_dim, inter) = (16, 2, 256, 32);
    let q_proj_dim = q_heads * head_dim * if attn_output_gate { 2 } else { 1 };
    let tensor = |rows: usize, cols: usize, seed: usize, scale: f32| {
        let data = (0..rows * cols)
            .map(|i| ((i * 7919 + seed * 104_729) as f32 * 0.000_37).sin() * scale)
            .collect();
        if cols == 1 {
            Tensor::from_vec(vec![rows], data)
        } else {
            Tensor::from_vec(vec![rows, cols], data)
        }
    };
    let norm = |dim: usize, seed: usize| {
        Tensor::from_vec(
            vec![dim],
            (0..dim)
                .map(|i| 1.0 + ((i + seed) as f32 * 0.37).sin() * 0.1)
                .collect(),
        )
    };
    let mut weights = HashMap::new();
    for (name, value) in [
        ("embed_tokens.weight", tensor(vocab, hidden, 1, 0.5)?),
        ("layers.0.input_layernorm.weight", norm(hidden, 2)?),
        (
            "layers.0.self_attn.q_proj.weight",
            tensor(q_proj_dim, hidden, 3, 0.06)?,
        ),
        (
            "layers.0.self_attn.k_proj.weight",
            tensor(kv_heads * head_dim, hidden, 4, 0.06)?,
        ),
        (
            "layers.0.self_attn.v_proj.weight",
            tensor(kv_heads * head_dim, hidden, 5, 0.06)?,
        ),
        (
            "layers.0.self_attn.o_proj.weight",
            tensor(hidden, q_heads * head_dim, 6, 0.02)?,
        ),
        ("layers.0.self_attn.q_norm.weight", norm(head_dim, 7)?),
        ("layers.0.self_attn.k_norm.weight", norm(head_dim, 8)?),
        ("layers.0.post_attention_layernorm.weight", norm(hidden, 9)?),
        (
            "layers.0.mlp.gate_proj.weight",
            tensor(inter, hidden, 10, 0.1)?,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            tensor(inter, hidden, 11, 0.1)?,
        ),
        (
            "layers.0.mlp.down_proj.weight",
            tensor(hidden, inter, 12, 0.1)?,
        ),
        ("norm.weight", norm(hidden, 13)?),
        ("lm_head.weight", tensor(vocab, hidden, 14, 0.1)?),
    ] {
        weights.insert(name.to_string(), value);
    }
    Ok(weights)
}

/// Parcours multi-tours par le vrai cache de préfixe du décodeur
/// (`prefill_cache_state` → `prefix_cache_get`/`prefix_cache_put`,
/// `PrefillKvPlacement` raccordé à `replace_prefill`) :
/// - tour 2 prolongé en place dans l'allocation du tour 1, K/V identique au
///   prefill à froid, snapshot du tour 1 conservé ;
/// - entrée ancienne prolongée sur une autre branche sous une vue plus
///   longue : copie, lignes publiées intactes, entrées remplacées libérées ;
/// - rollback (troncature du cache) puis prolongement : de nouveau en place.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn prefix_cache_extends_bf16_kv_in_place_across_turns() -> Result<()> {
    if !prefix_cache_enabled()
        || prefix_cache_capacity() < 2
        || !crate::runtime_flags::prefill_kv_shared_enabled()
    {
        return Ok(());
    }
    const VOCAB: usize = 97;
    const HIDDEN: usize = 64;
    const KV_DIM: usize = 2 * 256;
    let config = CausalDecoderConfig {
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: Some(256),
        rope_theta: Some(10_000_000.0),
        ..CausalDecoderConfig::default()
    };
    let model = match CausalDecoder::from_tensors(
        resident_bf16_qwen_weights(VOCAB, HIDDEN, false)?,
        config,
    )?
    .with_metal_runtime()
    {
        Ok(model) => model,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
        Err(error) => return Err(error),
    };
    let ids = |len: usize, step: usize, offset: usize| -> Vec<usize> {
        (0..len).map(|i| (i * step + offset) % VOCAB).collect()
    };
    let tokens = ids(1200, 31, 7);
    let (p1, p2) = (&tokens[..1100], &tokens[..1200]);
    let branch = [p1, &ids(100, 17, 3)].concat();
    let p3 = [&tokens[..1150], &ids(60, 13, 5)].concat();
    // Vue bf16 unique de K et de V : (allocations K/V, éléments). K et V
    // doivent suivre le même placement (même allocation propre, même longueur).
    let sole_view = |cache: &CausalDecoderCache| {
        let layer = &cache.layers[0];
        match (
            layer.keys.segments().as_slice(),
            layer.values.segments().as_slice(),
        ) {
            (
                [crate::decode_resident::KvSegment::SharedBf16(keys)],
                [crate::decode_resident::KvSegment::SharedBf16(values)],
            ) if keys.len() == values.len() => {
                Some(((keys.allocation_id(), values.allocation_id()), keys.len()))
            }
            _ => None,
        }
    };
    let kv = |cache: &CausalDecoderCache| cache.layer_kv_for_test(0);
    let cold = |prompt: &[usize]| -> Result<_> {
        Ok(kv(&model.prefill_prompt_state_uncached(prompt)?.cache))
    };
    let cached = || -> Result<Vec<Vec<usize>>> {
        let cache = model
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
        Ok(cache
            .entries
            .iter()
            .map(|entry| entry.tokens.clone())
            .collect())
    };

    // Morceaux de 512 et stockage bf16 dès la première ligne : le chemin de
    // production des longs prompts, à l'échelle d'un modèle synthétique.
    crate::metal_backend::with_prefill_kv_layout_override(512, 0, || -> Result<()> {
        let (turn1, _) = model.prefill_cache_state(p1)?;
        let Some((home, len)) = sole_view(&turn1) else {
            eprintln!("prolongement en place du décodeur sauté : K/V bf16 indisponible (NAX)");
            return Ok(());
        };
        assert_eq!(len, 1100 * KV_DIM);
        let turn1_kv = kv(&turn1);

        // Tour 2 : hit sur l'entrée du tour 1, prolongée en place.
        let (turn2, _) = model.prefill_cache_state(p2)?;
        assert_eq!(
            sole_view(&turn2),
            Some((home, 1200 * KV_DIM)),
            "tour 2 en place"
        );
        assert_eq!(kv(&turn2), cold(p2)?, "tour 2 : K/V vs froid");
        assert_eq!(
            cached()?,
            vec![p2.to_vec(), p1.to_vec()],
            "snapshot du tour 1 gardé"
        );
        assert_eq!(kv(&turn1), turn1_kv, "état du tour 1 intact");

        // Autre branche depuis l'entrée du tour 1 : une vue plus longue vit
        // (tour 2), le prolongement recopie sans toucher aux lignes publiées.
        let turn2_kv = kv(&turn2);
        let (branched, _) = model.prefill_cache_state(&branch)?;
        let (copy, _) = sole_view(&branched).expect("invariant: vue entière d'un buffer neuf");
        assert_ne!(copy, home, "branche recopiée");
        assert_eq!(kv(&branched), cold(&branch)?, "branche : K/V vs froid");
        assert_eq!(kv(&turn2), turn2_kv, "lignes publiées intactes");
        // Les entrées de l'allocation remplacée sont libérées.
        assert_eq!(cached()?, vec![branch.clone()]);
        drop((turn1, branched));

        // Rollback du tour 2 à 1150 positions, mémorisé, puis prolongé :
        // plus aucune vue ne couvre au-delà, le prolongement reste en place.
        let mut rolled = turn2;
        for layer in &mut rolled.layers {
            layer.truncate(1150)?;
        }
        rolled.position = 1150;
        let final_state = Tensor::row(vec![0.0; HIDDEN])?;
        model.prefix_cache_put(&p3[..1150], &(rolled, final_state), None, None)?;
        let (turn3, _) = model.prefill_cache_state(&p3)?;
        assert_eq!(
            sole_view(&turn3),
            Some((home, 1210 * KV_DIM)),
            "rollback puis prolongement en place"
        );
        assert_eq!(kv(&turn3), cold(&p3)?, "rollback : K/V vs froid");
        Ok(())
    })
}

/// Seul un Qwen dense à k/v 5120→1024 (Qwen3.8 27B) ouvre ses k/v au
/// bf16×3 : Qwen 30B et Qwen3-Next (MoE), flint (sinks), un autre Qwen dense
/// et un modèle hétérogène gardent le plancher.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn qwen_kv_nax_x3_shape_targets_dense_qwen_27b_only() {
    let qwen = CausalDecoderConfig {
        is_qwen: true,
        ..CausalDecoderConfig::default()
    };
    let kv_27b: &[usize] = &[1024, 5120];
    assert_eq!(
        qwen_kv_nax_x3_shape(&qwen, &[kv_27b; 16]),
        Some((5120, 1024))
    );
    let moe = |experts| CausalDecoderConfig {
        num_experts: Some(experts),
        ..qwen.clone()
    };
    // Qwen 30B-A3B (2048→512, 128 experts) et Qwen3-Next (2048→512, 512).
    let kv_moe: &[usize] = &[512, 2048];
    assert_eq!(qwen_kv_nax_x3_shape(&moe(128), &[kv_moe; 4]), None);
    assert_eq!(qwen_kv_nax_x3_shape(&moe(512), &[kv_moe; 4]), None);
    // Même forme k/v sous MoE : refusée par l'identité, pas par la forme.
    assert_eq!(qwen_kv_nax_x3_shape(&moe(128), &[kv_27b; 4]), None);
    let flint = CausalDecoderConfig {
        attention_sinks: true,
        ..qwen.clone()
    };
    assert_eq!(qwen_kv_nax_x3_shape(&flint, &[kv_27b; 4]), None);
    let not_qwen = CausalDecoderConfig::default();
    assert_eq!(qwen_kv_nax_x3_shape(&not_qwen, &[kv_27b; 4]), None);
    // Autre Qwen dense (4096→1024) et modèle hétérogène.
    let kv_8b: &[usize] = &[1024, 4096];
    assert_eq!(qwen_kv_nax_x3_shape(&qwen, &[kv_8b; 4]), None);
    assert_eq!(qwen_kv_nax_x3_shape(&qwen, &[kv_27b, kv_8b]), None);
    assert_eq!(qwen_kv_nax_x3_shape(&qwen, &[]), None);
}

/// Coût de la vérification résidente en fonction du nombre de lignes, au contexte
/// `SARAGOSSA_RUST_VERIFY_ROWS_BENCH_CTX` (défaut 4096). Avec
/// `SARAGOSSA_RUST_DFLASH_DRAFT_DIR` : substitut de coût pour un arbre DFlash de
/// B nœuds (B + 1 lignes), 9 et 11 lignes mesurées avec et sans captures. Sans
/// drafter (modèles MoE, lookup) : `SARAGOSSA_RUST_VERIFY_ROWS_BENCH_ROWS=1,2,4`
/// choisit les lignes, captures linéaires actives si le modèle en a. Médiane de
/// 8 répétitions après 2 d'échauffement, temps mural (soumission, attente et
/// relecture comprises). Entre deux répétitions : avec captures, rollback réel à
/// une position acceptée (le KV et les états linéaires avancent d'un token,
/// comme un pas où seul le premier token est gardé) ; sans (dont la référence
/// séquentielle, qui n'en rend pas), troncature du KV seule : les états
/// linéaires avancent à chaque répétition, donc entrées et routage MoE dérivent.
/// Ces temps-là sont synthétiques, pas une comparaison contrôlée. Échoue si plus d'une ligne est
/// demandée sans le chemin en lignes, sauf `SARAGOSSA_RUST_VERIFY_ROWS_BENCH_SEQUENTIAL=1`
/// (référence séquentielle voulue).
#[test]
#[ignore = "modèle réel requis, bench"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn verify_rows_cost_bench_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let mut model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    // Captures hidden du drafter DFlash si présent (coût DFlash), sinon aucune.
    let layer_ids = if crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_DRAFT_DIR").is_some()
    {
        model = model.with_dflash_draft_from_env(&assets.config)?;
        model
            .dflash_draft()
            .ok_or_else(|| InferError::Config("drafter non chargé".to_string()))?
            .info
            .target_layer_ids
            .clone()
    } else {
        Vec::new()
    };
    let context = crate::runtime_flags::env_var("SARAGOSSA_RUST_VERIFY_ROWS_BENCH_CTX")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4096);
    let sequential_ok =
        crate::runtime_flags::env_flag("SARAGOSSA_RUST_VERIFY_ROWS_BENCH_SEQUENTIAL", false);
    let has_linear = model.has_resident_linear_attention_layer();
    let rows_list: Vec<(usize, bool)> =
        match crate::runtime_flags::env_var("SARAGOSSA_RUST_VERIFY_ROWS_BENCH_ROWS") {
            Some(list) => list
                .split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .map(|rows| {
                    (
                        rows,
                        has_linear
                            && crate::runtime_flags::env_flag(
                                "SARAGOSSA_RUST_VERIFY_ROWS_BENCH_CAPTURES",
                                true,
                            ),
                    )
                })
                .collect(),
            None => vec![
                (9, true),
                (11, true),
                (9, false),
                (11, false),
                (13, false),
                (16, false),
                (17, false),
                (25, false),
                (33, false),
            ],
        };
    // Tokens synthétiques : le coût ne dépend pas du contenu, au routage MoE
    // près. `SARAGOSSA_RUST_VERIFY_ROWS_BENCH_TEXT=1` prend préfixe et lignes
    // candidates dans du code réel (recouvrement d'experts représentatif).
    let text_ids = if crate::runtime_flags::env_flag("SARAGOSSA_RUST_VERIFY_ROWS_BENCH_TEXT", false)
    {
        let text = concat!(
            include_str!("../decode_resident/attention.rs"),
            include_str!("../decode_resident/rows.rs"),
            include_str!("../decode_resident/layers.rs"),
        );
        let ids = mtp_verifier_prompt_ids(&assets, text)?;
        if ids.len() < context + 64 {
            return Err(InferError::Config(format!(
                "texte du bench trop court: {} tokens pour un contexte {context}",
                ids.len()
            )));
        }
        Some(ids)
    } else {
        None
    };
    let prompt: Vec<usize> = match &text_ids {
        Some(ids) => ids[..context].to_vec(),
        None => (0..context).map(|i| 1000 + (i * 7919) % 50_000).collect(),
    };
    let mut cache = if layer_ids.is_empty() {
        model.prefill_cache_state(&prompt)?.0
    } else {
        model
            .prefill_cache_state_with_layer_capture_cached(&prompt, &layer_ids, usize::MAX, None)?
            .0
    };
    let kv_bf16 = CausalDecoder::resident_kv_bf16(false, cache.position);
    // Une avance d'un token par répétition avec captures : réserve large.
    assert!(model.setup_resident_full_decode(&mut cache, 512, 0, kv_bf16)?);
    let batched = cache
        .resident
        .as_ref()
        .is_some_and(|arena| model.resident_layer_batch_supported(arena));
    // Seule portée où `SARAGOSSA_RUST_VERIFY_SKIP` agit (ablation par poste).
    let _skip_scope = crate::runtime_flags::VerifySkipScope::install();
    for (rows, capture) in rows_list {
        if rows > 1 && !batched && !sequential_ok {
            return Err(InferError::Config(format!(
                "bench {rows} lignes: chemin en lignes indisponible (repli séquentiel)"
            )));
        }
        let tokens: Vec<usize> = match &text_ids {
            Some(ids) => ids[context..context + rows].to_vec(),
            None => (0..rows).map(|i| 2000 + i * 31).collect(),
        };
        let mut times = Vec::new();
        for rep in 0..10 {
            let base = cache.position;
            let started = std::time::Instant::now();
            let output = model
                .next_final_states_resident_verify(
                    &mut cache,
                    &tokens,
                    crate::decoder::ResidentVerifyHead::Argmax,
                    capture,
                    (!layer_ids.is_empty()).then_some(layer_ids.as_slice()),
                    None,
                )?
                .ok_or_else(|| InferError::Config("vérification indisponible".to_string()))?;
            if rep >= 2 {
                times.push(started.elapsed().as_secs_f64() * 1e3);
            }
            match output.captures.as_ref() {
                Some(captures) => model.restore_mtp_resident_progress(&mut cache, captures, 1)?,
                None if rows == 1 => {}
                None => {
                    cache.position = base;
                    for layer in &mut cache.layers {
                        if let Some(full) = layer.full.as_mut() {
                            full.truncate(base)?;
                        }
                    }
                }
            }
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "verify_rows_bench ctx={context} text={} kv_bf16={kv_bf16} batched={batched} rows={rows} captures={capture} median_ms={:.2} min_ms={:.2}",
            text_ids.is_some(),
            times[times.len() / 2],
            times[0]
        );
    }
    Ok(())
}

/// Oracle de la vérification MoE en lignes (`SARAGOSSA_RUST_MOE_VERIFY_ROWS=1`,
/// modèle `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL`, Qwen3.5/3.6 MoE) : R lignes
/// d'un coup contre R pas mono-ligne sur le même texte imposé. Exige des
/// comptes exacts (tokens et états), l'argmax identique (aucune quasi-égalité
/// tolérée : la référence documentée est zéro) et des états finaux à 5e-3
/// relatif. Le passage avec captures est comparé lui aussi. Puis, pour
/// plusieurs longueurs acceptées `j`, rollback par captures à `j`, nouvelle
/// vérification en lignes, puis suite mono-ligne : KV et états linéaires
/// restaurés sont validés contre la référence.
#[test]
#[ignore = "modèle MoE réel requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn moe_verify_rows_match_single_rows_real_model() -> Result<()> {
    const PROMPT_TOKENS: usize = 512;
    const FORCED: usize = 16;
    const NEAR_TIE: f32 = 0.05;
    if crate::runtime_flags::moe_verify_rows_override() != Some(true) {
        eprintln!("skip oracle MoE en lignes: SARAGOSSA_RUST_MOE_VERIFY_ROWS absent");
        return Ok(());
    }
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
    let fresh = || -> Result<CausalDecoderCache> {
        let mut cache = model.prefill_cache_state(prompt)?.0;
        if !model.setup_resident_full_decode(&mut cache, FORCED + 8, 0, false)? {
            return Err(InferError::Config("setup résident refusé".to_string()));
        }
        Ok(cache)
    };
    let hidden = model.final_norm.data().len();
    let verify = |cache: &mut CausalDecoderCache,
                  tokens: &[usize],
                  capture: bool|
     -> Result<(Vec<usize>, Vec<f32>, Option<ResidentVerifyOutput>)> {
        let output = model
            .next_final_states_resident_verify(
                cache,
                tokens,
                ResidentVerifyHead::Argmax,
                capture,
                None,
                None,
            )?
            .ok_or_else(|| InferError::Config("vérification résidente non exercée".into()))?;
        let argmax = output
            .tokens
            .clone()
            .ok_or_else(|| InferError::Config("argmax de vérification absent".to_string()))?;
        let states = output.states.data().to_vec();
        if argmax.len() != tokens.len() || states.len() != tokens.len() * hidden {
            return Err(InferError::Dimension(format!(
                "vérification de {} lignes: {} argmax, {} valeurs d'état",
                tokens.len(),
                argmax.len(),
                states.len()
            )));
        }
        Ok((argmax, states, capture.then_some(output)))
    };

    // Référence : un pas mono-ligne par token imposé.
    let mut reference = fresh()?;
    let batched = reference
        .resident
        .as_ref()
        .is_some_and(|arena| model.resident_layer_batch_supported(arena));
    assert!(batched, "chemin MoE en lignes non éligible sur ce modèle");
    let mut ref_states = Vec::with_capacity(FORCED);
    let mut ref_tokens = Vec::with_capacity(FORCED);
    let mut ref_margins = Vec::with_capacity(FORCED);
    for &token in forced {
        let (tokens, state, _) = verify(&mut reference, &[token], false)?;
        let (_, margin) = top2(
            model
                .logits_from_final_state(&Tensor::from_vec(vec![1, hidden], state.clone())?)?
                .as_row()?,
        );
        ref_tokens.extend(tokens);
        ref_states.push(state);
        ref_margins.push(margin);
    }

    let compare = |label: &str, offset: usize, tokens: &[usize], states: &[f32]| {
        assert_eq!(
            states.len(),
            tokens.len() * hidden,
            "{label}: taille d'état"
        );
        for (row, (token, state)) in tokens.iter().zip(states.chunks(hidden)).enumerate() {
            let at = offset + row;
            let expected = &ref_states[at];
            let scale = expected.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
            let max_err = expected
                .iter()
                .zip(state)
                .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
            assert!(
                state.iter().all(|value| value.is_finite()),
                "{label} ligne {at}: état non fini"
            );
            assert!(
                max_err <= 5e-3 * scale,
                "{label} ligne {at}: dérive {max_err:e} / {scale:e}"
            );
            assert_eq!(
                *token, ref_tokens[at],
                "{label} ligne {at}: argmax divergent (marge de référence {}, < {NEAR_TIE} = quasi-égalité)",
                ref_margins[at]
            );
        }
    };
    let continue_single =
        |cache: &mut CausalDecoderCache, label: &str, from: usize| -> Result<()> {
            for (step, &token) in forced[from..from + 3].iter().enumerate() {
                let (tokens, states, _) = verify(cache, &[token], false)?;
                compare(label, from + step, &tokens, &states);
            }
            Ok(())
        };

    for rows in [2, 3, 5, 9] {
        // Tout accepté : R lignes puis la suite mono-ligne.
        let mut cache = fresh()?;
        let (tokens, states, _) = verify(&mut cache, &forced[..rows], false)?;
        compare(&format!("lignes={rows}"), 0, &tokens, &states);
        continue_single(&mut cache, &format!("lignes={rows} suite"), rows)?;
        // Rollback par captures à `accepted` lignes, nouvelle vérification en
        // lignes depuis l'état restauré, puis suite mono-ligne.
        for accepted in [1, rows / 2, rows - 1] {
            if accepted == 0 {
                continue;
            }
            let label = format!("lignes={rows} rollback={accepted}");
            let mut cache = fresh()?;
            let (tokens, states, output) = verify(&mut cache, &forced[..rows], true)?;
            compare(&format!("{label} captures"), 0, &tokens, &states);
            let output = output
                .ok_or_else(|| InferError::Config("sortie avec captures absente".to_string()))?;
            let captures = output.captures.as_ref().ok_or_else(|| {
                InferError::Config("captures de vérification absentes".to_string())
            })?;
            model.restore_mtp_resident_progress(&mut cache, captures, accepted)?;
            assert_eq!(cache.position, prompt_len + accepted);
            let again = accepted + 2;
            let (tokens, states, _) = verify(&mut cache, &forced[accepted..again], false)?;
            compare(&format!("{label} relance"), accepted, &tokens, &states);
            continue_single(&mut cache, &format!("{label} suite"), again)?;
        }
    }
    eprintln!("oracle MoE en lignes: vert, aucune quasi-égalité");
    Ok(())
}

/// Pairwise ABBA de la boucle lookup greedy (`SARAGOSSA_RUST_LOOKUP_GREEDY`)
/// contre le greedy AR, même processus, même modèle : `SARAGOSSA_RUST_MOESPEC_PAIRS`
/// paires (défaut 3) par cas, drafts `SARAGOSSA_RUST_MOESPEC_LOOKUP` (défaut 2),
/// prompts JSON `SARAGOSSA_RUST_MOESPEC_PROMPTS` (`{nom: texte}`), budget
/// `SARAGOSSA_RUST_MOESPEC_TOKENS` (défaut 256).
/// Affiche le débit de decode de chaque passage et le gain médian ; échoue si
/// une sortie diffère de l'AR. Exige `SARAGOSSA_RUST_MOE_VERIFY_ROWS=1` sur
/// un modèle MoE.
#[test]
#[ignore = "modèle réel requis, bench"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn lookup_greedy_abba_vs_ar_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let Some(prompts_path) = crate::runtime_flags::env_var_os("SARAGOSSA_RUST_MOESPEC_PROMPTS")
    else {
        eprintln!("skip lookup ABBA: SARAGOSSA_RUST_MOESPEC_PROMPTS absent");
        return Ok(());
    };
    let prompts: std::collections::BTreeMap<String, String> = serde_json::from_str(
        &std::fs::read_to_string(&prompts_path)
            .map_err(|error| InferError::Config(format!("prompts lookup: {error}")))?,
    )
    .map_err(|error| InferError::Config(format!("prompts lookup: {error}")))?;
    let env_usize = |name: &str, default: usize| {
        crate::runtime_flags::env_var(name)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(default)
    };
    let cases = crate::runtime_flags::env_var("SARAGOSSA_RUST_MOESPEC_CASES")
        .unwrap_or_else(|| "T0,T2,T6".to_string());
    let max_new = env_usize("SARAGOSSA_RUST_MOESPEC_TOKENS", 256);
    let pairs = env_usize("SARAGOSSA_RUST_MOESPEC_PAIRS", 3);
    let drafts = env_usize("SARAGOSSA_RUST_MOESPEC_LOOKUP", 2);
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let options = mtp_verifier_options(assets.stop_token_ids());
    let run = |prompt: &[usize], lookup: usize| -> Result<(Vec<usize>, f64)> {
        let output = crate::runtime_flags::with_lookup_greedy_override(lookup, || {
            model.generate_greedy_timed_with_options(prompt, max_new, &options)
        })?;
        let rate = output.timings.decode_tokens as f64
            / output.timings.decode.as_secs_f64().max(f64::MIN_POSITIVE);
        Ok((output.tokens, rate))
    };
    for name in cases.split(',').map(str::trim) {
        let text = prompts
            .get(name)
            .ok_or_else(|| InferError::Config(format!("prompt {name} absent")))?;
        let prompt = mtp_verifier_prompt_ids(&assets, text)?;
        // Échauffement des deux chemins (pipelines, scratch).
        let (reference, _) = run(&prompt, 0)?;
        run(&prompt, drafts)?;
        let mut gains = Vec::with_capacity(pairs * 2);
        for pair in 0..pairs {
            let mut rates = [0.0f64; 2];
            for (slot, lookup) in [(0, 0), (1, drafts), (1, drafts), (0, 0)] {
                let (tokens, rate) = run(&prompt, lookup)?;
                assert_eq!(tokens, reference, "{name} lookup={lookup}: sortie ≠ AR");
                rates[slot] += rate / 2.0;
                eprintln!(
                    "lookup_abba case={name} paire={pair} lookup={lookup} tokens={} tok_s={rate:.1}",
                    tokens.len()
                );
            }
            gains.push((rates[1] / rates[0] - 1.0) * 100.0);
        }
        gains.sort_by(f64::total_cmp);
        eprintln!(
            "lookup_abba case={name} drafts={drafts} sortie={} gain_median={:+.1}% gains={gains:.1?}",
            reference.len(),
            gains[gains.len() / 2],
        );
    }
    Ok(())
}

/// Oracle de la boucle lookup greedy contre le greedy AR, modèle réel
/// (`SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL` ; MoE : avec
/// `SARAGOSSA_RUST_MOE_VERIFY_ROWS=1`). Prompt qui fait recopier un fichier de
/// ce dépôt, pour que le lookup propose et que le pipeline soit vidangé. Sorties
/// identiques exigées pour 0, 1, 2 et 8 drafts et des budgets de 0, 1, 2, 9 et
/// 96 tokens ; arrêts sur des tokens de la sortie AR ; callback qui interrompt
/// le streaming. Les compteurs de la boucle prouvent qu'au moins une proposition
/// est acceptée, qu'au moins une vidange du pipeline est suivie d'une reprise et
/// qu'au moins un arrêt tombe parmi des drafts acceptés.
#[test]
#[ignore = "modèle réel requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn lookup_greedy_matches_ar_real_model() -> Result<()> {
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let text = format!(
        "Recopie ce fichier à l'identique.\n\n```rust\n{}\n```",
        include_str!("lookup.rs")
    );
    let prompt = mtp_verifier_prompt_ids(&assets, &text)?;
    let options = mtp_verifier_options(assets.stop_token_ids());
    // Éligibilité réelle : sans elle, l'égalité serait triviale.
    let mut cache = model.prefill_cache_state(&prompt)?.0;
    assert!(model.setup_resident_full_decode(&mut cache, 96, 0, false)?);
    assert!(
        crate::runtime_flags::with_lookup_greedy_override(2, || {
            model.lookup_greedy_eligible(&cache, &options)
        }),
        "boucle lookup inéligible sur ce modèle (MoE : SARAGOSSA_RUST_MOE_VERIFY_ROWS=1)"
    );
    drop(cache);
    let generate = |drafts: usize, max_new: usize, options: &GenerationOptions| {
        crate::runtime_flags::with_lookup_greedy_override(drafts, || {
            model.generate_greedy_timed_with_options(&prompt, max_new, options)
        })
        .map(|output| output.tokens)
    };
    let reference = generate(0, 96, &options)?;
    let mut traces = crate::decoder::resident::take_lookup_greedy_trace();
    let mut stops_in_drafts = 0;
    for drafts in [1, 2, 8] {
        for max_new in [0, 1, 2, 9, 96] {
            let expected = &reference[..max_new.min(reference.len())];
            crate::decoder::resident::take_lookup_greedy_trace();
            assert_eq!(
                generate(drafts, max_new, &options)?,
                expected,
                "drafts={drafts} budget={max_new}"
            );
            let trace = crate::decoder::resident::take_lookup_greedy_trace();
            traces.accepted_verifies += trace.accepted_verifies;
            traces.drains += trace.drains;
        }
        // Arrêt sur un token de la sortie : la sortie AR s'arrête juste avant
        // sa première occurrence, la boucle lookup doit faire de même.
        for &at in &[5, 20, 40, 70] {
            let Some(&stop) = reference.get(at) else {
                continue;
            };
            let mut stopping = options.clone();
            stopping.stop_token_ids.push(stop);
            let expected = generate(0, 96, &stopping)?;
            crate::decoder::resident::take_lookup_greedy_trace();
            assert_eq!(
                generate(drafts, 96, &stopping)?,
                expected,
                "drafts={drafts} arrêt sur {stop}"
            );
            stops_in_drafts += crate::decoder::resident::take_lookup_greedy_trace().stop_in_drafts;
        }
        // Callback qui interrompt : mêmes tokens émis qu'en AR.
        for limit in [1, 7, 33] {
            let stream = |lookup: usize| -> Result<Vec<usize>> {
                let mut seen = Vec::new();
                crate::runtime_flags::with_lookup_greedy_override(lookup, || {
                    model.generate_greedy_streaming_with_options(&prompt, 96, &options, |token| {
                        seen.push(token);
                        seen.len() < limit
                    })
                })?;
                Ok(seen)
            };
            assert_eq!(
                stream(drafts)?,
                stream(0)?,
                "drafts={drafts} callback={limit}"
            );
        }
    }
    // Le scénario doit être exercé : propositions vérifiées et acceptées, et
    // pipeline vidangé puis repris (chemin non streamé) ; arrêt sur un token
    // d'arrêt tombé parmi des drafts acceptés (tokens fréquents de la sortie
    // recopiée : retour à la ligne, indentation).
    eprintln!(
        "lookup greedy ≡ AR : {} tokens de référence, {traces:?}, arrêts dans les drafts={stops_in_drafts}",
        reference.len()
    );
    assert!(traces.accepted_verifies > 0, "aucune proposition acceptée");
    assert!(traces.drains > 0, "aucune vidange du pipeline");
    assert!(stops_in_drafts > 0, "aucun arrêt parmi des drafts acceptés");
    Ok(())
}

// --- DFlash échantillonné : tirage par ligne et descente, modèle réel ---

/// Diagnostic d'un écart AR/DFlash au token `index` : probabilités, sous la
/// cible filtrée de l'AR (logits du decode résident forcé sur le préfixe
/// commun), des deux tokens et uniforme du tirage `index` de la graine.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_sampled_divergence(
    model: &CausalDecoder,
    prompt: &[usize],
    ar: &[usize],
    other: &[usize],
    options: &GenerationOptions,
) -> Result<String> {
    let Some(index) = ar.iter().zip(other).position(|(a, b)| a != b) else {
        return Ok(format!("longueurs {} contre {}", ar.len(), other.len()));
    };
    if index == 0 {
        return Ok("premier token différent".to_string());
    }
    let kv_bf16 = CausalDecoder::resident_kv_bf16(true, prompt.len());
    let forced = model.resident_forced_logits_for_test(prompt, &ar[..index], kv_bf16)?;
    let distribution = crate::sampling::token_distribution_top_k_top_p(
        &forced[index - 1],
        options.temperature,
        options.top_p,
        options.top_k,
    )?;
    let probability = |token: usize| {
        distribution
            .iter()
            .find(|(candidate, _)| *candidate == token)
            .map_or(0.0, |(_, p)| *p)
    };
    let u =
        crate::DeterministicSampler::new(crate::sampling::rng_state_for_draw(options.seed, index))
            .next_unit_f32();
    Ok(format!(
        "écart au token {index} : AR {} (p = {}) contre {} (p = {}), u = {u}",
        ar[index],
        probability(ar[index]),
        other[index],
        probability(other[index])
    ))
}

/// DFlash échantillonné sur le modèle réel (défaut, coupé par `SARAGOSSA_RUST_DFLASH_SAMPLED=0`) :
/// - à T = 0,05 (top-k 20, top-p 0,9), chaîne et arbre B = 8 sortent les tokens
///   de l'AR échantillonné de même graine, et une graine rejoue sa sortie ;
/// - à T = 0,7, sur `SARAGOSSA_RUST_DFLASH_SAMPLED_SEEDS` graines (24 par
///   défaut), chaîne et arbre B = 8 sortent exactement les tokens de l'AR de
///   même graine (campagne déterministe : 96/96 le 3 novembre). Un écart échoue
///   avec son diagnostic (probabilités des deux tokens sous l'AR, uniforme).
///
/// Requiert `SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL`,
/// et `SARAGOSSA_RUST_DFLASH_DRAFT_DIR`.
#[test]
#[ignore = "modèle réel + draft DFlash requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_sampled_matches_ar_real_model() -> Result<()> {
    use crate::decoder::resident::dflash::DFlashTreeMode;
    let Some((model, assets, draft)) = dflash_sampled_model()? else {
        return Ok(());
    };
    let seeds = crate::runtime_flags::env_var("SARAGOSSA_RUST_DFLASH_SAMPLED_SEEDS")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(24);
    let block = draft.info.block_size;
    let run = |prompt: &[usize], max_new: usize, options: &GenerationOptions, mode| {
        model.generate_greedy_dflash_resident_tree(
            prompt,
            max_new,
            options,
            &draft,
            block,
            (mode, true),
            |_| true,
        )
    };
    for text in [
        "Write a Rust function that parses a semantic version string and returns its three components.",
        "Raconte en quelques phrases l'histoire d'un phare isolé sur une côte bretonne.",
    ] {
        let prompt = mtp_verifier_prompt_ids(&assets, text)?;
        let cold = GenerationOptions {
            temperature: 0.05,
            seed: 7,
            ..dflash_sampled_options()
        };
        let ar = model.generate_greedy_cached_with_options(&prompt, 32, &cold)?;
        for mode in [DFlashTreeMode::Off, DFlashTreeMode::BestFirst(8)] {
            let first = run(&prompt, 32, &cold, mode)?;
            let second = run(&prompt, 32, &cold, mode)?;
            assert_eq!(first.tokens, ar, "T=0,05 {mode:?} « {} »", &text[..20]);
            assert_eq!(first.tokens, second.tokens, "rejeu de graine {mode:?}");
        }
        let max_new = 32;
        for mode in [DFlashTreeMode::Off, DFlashTreeMode::BestFirst(8)] {
            let (mut accepted, mut proposed) = (0, 0);
            for seed in 0..seeds {
                let warm = GenerationOptions {
                    temperature: 0.7,
                    seed,
                    ..dflash_sampled_options()
                };
                let ar = model.generate_greedy_cached_with_options(&prompt, max_new, &warm)?;
                let spec = run(&prompt, max_new, &warm, mode)?;
                accepted += spec.stats.accepted;
                proposed += spec.stats.proposed;
                if spec.tokens != ar {
                    let why = dflash_sampled_divergence(&model, &prompt, &ar, &spec.tokens, &warm)?;
                    panic!("DFlash échantillonné {mode:?} graine {seed} différent de l'AR : {why}");
                }
            }
            eprintln!(
                "DFlash échantillonné T=0,7 {mode:?} « {} » : {seeds}/{seeds} sorties identiques à l'AR, acceptance {accepted}/{proposed}",
                &text[..20],
            );
        }
    }
    Ok(())
}

/// Modèle, assets et drafter des tests DFlash échantillonnés (`None` : skip).
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_sampled_model() -> Result<Option<(CausalDecoder, crate::ModelAssets, crate::DFlashDraft)>>
{
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(None);
    };
    if crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_DRAFT_DIR").is_none()
        || !crate::runtime_flags::dflash_sampled_enabled()
    {
        eprintln!("skip DFlash échantillonné: drafter absent ou SARAGOSSA_RUST_DFLASH_SAMPLED=0");
        return Ok(None);
    }
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_dflash_draft_from_env(&assets.config)?;
    let draft = model
        .dflash_draft()
        .ok_or_else(|| InferError::Config("drafter DFlash non chargé".to_string()))?
        .clone();
    Ok(Some((model, assets, draft)))
}

#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_sampled_options() -> GenerationOptions {
    GenerationOptions {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 20,
        ..mtp_verifier_options(Vec::new())
    }
}

/// Arrêts du DFlash échantillonné sur les entrées de production, chaîne
/// (`Off`) et arbre B = 8, à T = 0,7 : stop token à chaque position de la
/// trajectoire libre (stop initial, stop accepté parmi les drafts, rejet ou
/// bonus devenu stop), stop sequence de deux tokens à chaque position (donc à
/// cheval sur deux tours quelque part), callback qui arrête et limite de sortie
/// à chaque longueur. Les tokens rendus ET émis valent ceux de l'AR
/// échantillonné de même graine avec les mêmes arrêts.
///
/// Requiert les mêmes variables que `dflash_sampled_matches_ar_real_model`.
#[test]
#[ignore = "modèle réel + draft DFlash requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn dflash_sampled_stops_match_ar_real_model() -> Result<()> {
    use crate::decoder::resident::dflash::DFlashTreeMode;
    let Some((model, assets, draft)) = dflash_sampled_model()? else {
        return Ok(());
    };
    let block = draft.info.block_size;
    let prompt = mtp_verifier_prompt_ids(
        &assets,
        "Write a Rust function that parses a semantic version string and returns its three components.",
    )?;
    let max_new = 24;
    for seed in [3_u64, 11] {
        let base = GenerationOptions {
            seed,
            ..dflash_sampled_options()
        };
        let free = model.generate_greedy_cached_with_options(&prompt, max_new, &base)?;
        assert_eq!(
            free.len(),
            max_new,
            "trajectoire libre courte (graine {seed})"
        );
        let mut cases: Vec<(String, GenerationOptions, usize, usize)> = Vec::new();
        for position in 0..max_new {
            cases.push((
                format!("stop token {} (position {position})", free[position]),
                GenerationOptions {
                    stop_token_ids: vec![free[position]],
                    ..base.clone()
                },
                max_new,
                usize::MAX,
            ));
            if position + 1 < max_new {
                cases.push((
                    format!("stop sequence en {position}"),
                    GenerationOptions {
                        stop_sequences: vec![free[position..position + 2].to_vec()],
                        ..base.clone()
                    },
                    max_new,
                    usize::MAX,
                ));
            }
            cases.push((
                format!("callback {}", position + 1),
                base.clone(),
                max_new,
                position + 1,
            ));
            cases.push((
                format!("limite {}", position + 1),
                base.clone(),
                position + 1,
                usize::MAX,
            ));
        }
        for (label, options, limit, callback) in &cases {
            fn recorder(
                emitted: &mut Vec<usize>,
                callback: usize,
            ) -> impl FnMut(usize) -> bool + '_ {
                move |token| {
                    emitted.push(token);
                    emitted.len() < callback
                }
            }
            let mut ar_emitted = Vec::new();
            let ar = model
                .generate_greedy_streaming_with_options(
                    &prompt,
                    *limit,
                    options,
                    recorder(&mut ar_emitted, *callback),
                )?
                .tokens;
            assert_eq!(ar, ar_emitted, "AR {label} : sortie et flux");
            for mode in [DFlashTreeMode::Off, DFlashTreeMode::BestFirst(8)] {
                let mut emitted = Vec::new();
                let spec = model.generate_greedy_dflash_resident_tree(
                    &prompt,
                    *limit,
                    options,
                    &draft,
                    block,
                    (mode, true),
                    recorder(&mut emitted, *callback),
                )?;
                assert_eq!(spec.tokens, ar, "graine {seed} {mode:?} {label}");
                assert_eq!(emitted, ar, "flux graine {seed} {mode:?} {label}");
            }
        }
        eprintln!(
            "DFlash échantillonné arrêts graine {seed} : {} cas identiques à l'AR",
            cases.len()
        );
    }
    Ok(())
}

/// Prompts du corpus figé (`tests/fixtures/rowsexact`) : liste de noms sans
/// extension (`SARAGOSSA_RUST_IDENTITY_PROMPTS`, virgules), sinon les prompts
/// courts (chat et code 4k ; le cas prose32k-1 se demande explicitement).
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn identity_corpus() -> Result<Vec<(String, String)>> {
    let names =
        crate::runtime_flags::env_var("SARAGOSSA_RUST_IDENTITY_PROMPTS").unwrap_or_else(|| {
            "chat-1,chat-2,chat-3,chat-4,chat-5,chat-6,code4k-1,code4k-2,code4k-3".to_string()
        });
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rowsexact");
    let manifest = std::fs::read_to_string(fixtures.join("MANIFEST.tsv"))
        .map_err(|error| InferError::Config(format!("manifeste du corpus: {error}")))?;
    let external =
        crate::runtime_flags::env_var_os("SARAGOSSA_RUST_IDENTITY_EXTERNAL_DIR").map(PathBuf::from);
    let mut corpus = Vec::new();
    for name in names.split(',').map(str::trim) {
        let file = format!("{name}.txt");
        let expected = manifest
            .lines()
            .find_map(|line| {
                let mut fields = line.split('\t');
                (fields.next() == Some(file.as_str())).then(|| fields.nth(1))?
            })
            .ok_or_else(|| InferError::Config(format!("{file} absent du manifeste")))?;
        // Versionné dans le dépôt, sinon (texte sous licence tierce) dossier externe.
        let candidates = [
            Some(fixtures.join(&file)),
            external.as_ref().map(|dir| dir.join(&file)),
        ];
        let Some(path) = crate::test_support::require_real_model(
            candidates.into_iter().flatten().find(|path| path.is_file()),
            &format!("{file} (SARAGOSSA_RUST_IDENTITY_EXTERNAL_DIR)"),
        ) else {
            eprintln!(
                "skip corpus {file}: hors dépôt, SARAGOSSA_RUST_IDENTITY_EXTERNAL_DIR absent"
            );
            continue;
        };
        let text = std::fs::read_to_string(&path)
            .map_err(|error| InferError::Config(format!("corpus {}: {error}", path.display())))?;
        let digest: String = <sha2::Sha256 as sha2::Digest>::digest(text.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if digest != expected {
            return Err(InferError::Config(format!(
                "{file}: sha256 {digest} ≠ manifeste {expected}"
            )));
        }
        corpus.push((name.to_string(), text));
    }
    Ok(corpus)
}

/// Bilan d'un mode spéculatif contre l'AR sur un prompt.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[derive(Default, Debug)]
struct IdentityTally {
    /// Lignes vérifiées dont le chemin d'entrée est celui de l'AR.
    compared_rows: usize,
    /// Parmi elles, lignes d'un lot d'au moins deux lignes passé par les
    /// kernels en lignes (le repli séquentiel, enregistré comme tel, est exclu).
    multi_row_compared: usize,
    /// Vérifications en arbre réellement branché (au moins un frère).
    branched_records: usize,
    /// Parmi elles, lignes dont l'état final diffère de l'AR en bits.
    state_mismatches: usize,
    /// Premier écart (position absolue, valeurs différentes, lignes du lot, ligne).
    first_mismatch: Option<(usize, usize, usize, usize)>,
    /// Premier token généré différent de l'AR.
    first_token_divergence: Option<usize>,
}

/// Compare les vérifications enregistrées (`VERIFY_RECORDER`) aux états AR :
/// une ligne n'est comparée que si toute sa chaîne d'ancêtres (chaîne ou
/// arbre) porte les tokens de l'AR aux mêmes positions.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn identity_compare_records(
    records: &[super::resident::VerifyRecord],
    sequence: &[usize],
    prompt_len: usize,
    ar_states: &[Vec<f32>],
    hidden: usize,
    tally: &mut IdentityTally,
) {
    for record in records {
        let rows = record.tokens.len();
        let branched = record.parents.as_ref().is_some_and(|parents| {
            parents
                .iter()
                .enumerate()
                .skip(1)
                .any(|(row, &p)| p as usize + 1 != row)
        });
        tally.branched_records += usize::from(branched);
        let parent = |row: usize| -> Option<usize> {
            if row == 0 {
                return None;
            }
            Some(record.parents.as_ref().map_or(row - 1, |p| p[row] as usize))
        };
        let mut depths = vec![0usize; rows];
        for row in 1..rows {
            depths[row] = parent(row).map_or(0, |p| depths[p] + 1);
        }
        for row in 0..rows {
            let mut node = Some(row);
            let mut on_path = true;
            while let Some(current) = node {
                let position = record.base_position + depths[current];
                if sequence.get(position) != Some(&record.tokens[current]) {
                    on_path = false;
                    break;
                }
                node = parent(current);
            }
            let position = record.base_position + depths[row];
            let Some(reference) = position
                .checked_sub(prompt_len)
                .and_then(|i| ar_states.get(i))
            else {
                continue;
            };
            if !on_path {
                continue;
            }
            let state = &record.states[row * hidden..(row + 1) * hidden];
            let differing = state
                .iter()
                .zip(reference)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            tally.compared_rows += 1;
            tally.multi_row_compared += usize::from(record.batched && rows >= 2);
            if differing > 0 {
                tally.state_mismatches += 1;
                if tally.first_mismatch.is_none() {
                    tally.first_mismatch = Some((position, differing, rows, row));
                }
            }
        }
    }
}

/// Oracles O2 et O3 de l'identité greedy sur modèle réel
/// (`SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL`), corpus figé du dépôt :
/// - O3 : tokens de chaque mode spéculatif disponible (MTP si sidecar,
///   lookup, DFlash chaîne et arbre B = 8 si `SARAGOSSA_RUST_DFLASH_DRAFT_DIR`)
///   identiques à l'AR, avec le compteur des quasi-égalités traversées (marge
///   top-1/top-2 de la tête AR de production < 1e-2 et < 1e-3) ;
/// - O2 : l'état final de chaque ligne vérifiée dont le chemin d'entrée est
///   celui de l'AR égale en bits l'état AR à la même position, relevé depuis
///   les entrées de production (captures linéaires comprises).
///
/// Budget : `SARAGOSSA_RUST_IDENTITY_TOKENS` (128 par défaut, au moins 200
/// pour `prose32k-1`, dont l'écart historique tombe au token 182). Échoue si
/// un mode n'est pas identique, si un mode n'a comparé aucune ligne issue d'un
/// lot de plusieurs lignes sur tout le corpus, si l'arbre DFlash n'a jamais
/// été branché, ou si tout le corpus est sauté. La promesse ne vaut que pour la
/// configuration (flags, GPU) du lancement ; commandes et résultats attendus
/// dans docs/perf/greedy-identity.md.
#[test]
#[ignore = "modèle réel requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn greedy_identity_states_real_model() -> Result<()> {
    use crate::decoder::resident::dflash::DFlashTreeMode;
    use crate::decoder::resident::VERIFY_RECORDER;
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    let max_new = crate::runtime_flags::env_var("SARAGOSSA_RUST_IDENTITY_TOKENS")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(128);
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let mut model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let has_mtp = match assets.mtp.path.as_ref() {
        Some(path) => {
            model = model.with_mtp_sidecar(path)?;
            true
        }
        None => false,
    };
    let draft = if crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_DRAFT_DIR").is_some() {
        model = model.with_dflash_draft_from_env(&assets.config)?;
        model.dflash_draft().cloned()
    } else {
        None
    };
    let options = mtp_verifier_options(Vec::new());
    let hidden = model.final_norm.data().len();
    let (mut ties_2, mut ties_3, mut tokens_seen) = (0usize, 0usize, 0usize);
    let mut failures = Vec::new();
    let mut coverage: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    let mut tree_widths: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut skip_coverage: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut replay_coverage: std::collections::BTreeMap<&str, usize> = Default::default();
    // Modes sélectionnés, enregistrés avant tout saut : un mode dont tous les
    // prompts sont sautés échoue au lieu de passer sans couverture.
    let mut selected: std::collections::BTreeSet<&str> = Default::default();
    let corpus = identity_corpus()?;
    if corpus.is_empty() {
        return Err(InferError::Config(
            "corpus d'identité entièrement sauté".into(),
        ));
    }
    for (name, text) in corpus {
        let prompt = mtp_verifier_prompt_ids(&assets, &text)?;
        let max_new = if name == "prose32k-1" {
            max_new.max(200)
        } else {
            max_new
        };
        let ar = model.generate_greedy_cached_with_options(&prompt, max_new, &options)?;
        let kv_bf16 = CausalDecoder::resident_kv_bf16(false, prompt.len());
        // Marges de la tête de production : logits du decode résident forcé.
        let logits = model.resident_forced_logits_for_test(&prompt, &ar, kv_bf16)?;
        let mut min_margin = (f32::INFINITY, 0usize);
        for (i, row) in logits.iter().enumerate().take(ar.len().saturating_sub(1)) {
            let (top, margin) = top2(row);
            assert_eq!(
                top,
                ar[i + 1],
                "{name}: tête forcée ≠ AR au token {}",
                i + 1
            );
            tokens_seen += 1;
            ties_2 += usize::from(margin < 1e-2);
            ties_3 += usize::from(margin < 1e-3);
            if margin < min_margin.0 {
                min_margin = (margin, i + 1);
            }
        }
        // États AR : pas mono-ligne de production après le même prefill.
        let mut cache = model.prefill_cache_state(&prompt)?.0;
        if !model.setup_resident_full_decode(&mut cache, max_new + 16, 16, kv_bf16)? {
            return Err(InferError::Config("setup résident refusé".to_string()));
        }
        let mut ar_states = Vec::with_capacity(ar.len());
        for &token in &ar[..ar.len().saturating_sub(1)] {
            let out = model
                .next_final_states_resident_verify(
                    &mut cache,
                    &[token],
                    ResidentVerifyHead::Argmax,
                    false,
                    None,
                    None,
                )?
                .ok_or_else(|| InferError::Config("pas mono indisponible".into()))?;
            ar_states.push(out.states.data().to_vec());
        }
        drop(cache);
        let mut sequence = prompt.clone();
        sequence.extend_from_slice(&ar);
        let mut modes: Vec<&str> = vec!["lookup"];
        if has_mtp {
            // `mtp-skip` : repli sans argmax GPU (tête `Skip`) avec une
            // proposition lookup multiligne, posés par le test lui-même.
            modes.extend(["mtp", "mtp-skip"]);
        }
        if draft.is_some() {
            // `dflash-défaut` : entrée publique, arbre choisi par l'environnement
            // (B = 3 en mode strict greedy, B = 8 sinon, sauf `SARAGOSSA_RUST_DFLASH_TREE`).
            // `dflash-cpu` : boucle batchée de référence, rejeu en lignes après rejet.
            modes.extend([
                "dflash-chaîne",
                "dflash-arbre",
                "dflash-défaut",
                "dflash-cpu",
            ]);
        }
        // Sélection explicite (`SARAGOSSA_RUST_IDENTITY_MODES`, virgules) : un mode
        // demandé mais indisponible (sidecar ou drafter absent) fait échouer.
        if let Some(wanted) = crate::runtime_flags::env_var("SARAGOSSA_RUST_IDENTITY_MODES") {
            let wanted: Vec<String> = wanted.split(',').map(|m| m.trim().to_string()).collect();
            if let Some(missing) = wanted.iter().find(|m| !modes.contains(&m.as_str())) {
                return Err(InferError::Config(format!(
                    "mode d'identité {missing} indisponible"
                )));
            }
            modes.retain(|mode| wanted.iter().any(|m| m == mode));
        }
        for mode in modes {
            selected.insert(mode);
            // NOTE: la boucle DFlash CPU de référence prend son propre prefill avec
            // captures, distinct de celui de l'AR au-delà de 16k : hors périmètre.
            if mode == "dflash-cpu" && prompt.len() > 16 * 1024 {
                eprintln!(
                    "identité {name} dflash-cpu : prompt de {} tokens sauté",
                    prompt.len()
                );
                continue;
            }
            VERIFY_RECORDER.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
            let tokens = match mode {
                "lookup" => crate::runtime_flags::with_lookup_greedy_override(8, || {
                    model.generate_greedy_timed_with_options(&prompt, max_new, &options)
                })
                .map(|output| output.tokens),
                "mtp" => model
                    .generate_greedy_mtp_streaming_with_options(
                        &prompt,
                        max_new,
                        &options,
                        2,
                        |_| true,
                    )
                    .map(|output| output.tokens),
                // Boucle MTP directe : avec un drafter chargé, l'entrée publique
                // MTP passe par DFlash (tête argmax) et ne vérifie jamais en `Skip`.
                "mtp-skip" => crate::runtime_flags::with_mtp_skip_override(8, || {
                    model.generate_greedy_mtp_core(&prompt, max_new, &options, 2, |_| true)
                })
                .map(|output| output.tokens),
                "dflash-cpu" => {
                    let draft = draft
                        .as_ref()
                        .ok_or_else(|| InferError::Config("drafter".into()))?;
                    model
                        .generate_greedy_dflash_batched_with_options(
                            &prompt,
                            max_new,
                            &options,
                            draft,
                            draft.info.block_size,
                        )
                        .map(|output| output.tokens)
                }
                "dflash-défaut" => {
                    // Le défaut se lit sans surcharge : un `DFLASH_TREE` hérité
                    // mesurerait autre chose que le budget choisi par le mode.
                    if crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_TREE").is_some() {
                        return Err(InferError::Config(
                            "dflash-défaut : SARAGOSSA_RUST_DFLASH_TREE hérité refusé".into(),
                        ));
                    }
                    let draft = draft
                        .as_ref()
                        .ok_or_else(|| InferError::Config("drafter".into()))?;
                    model
                        .generate_greedy_dflash_resident_streaming_with_options(
                            &prompt,
                            max_new,
                            &options,
                            draft,
                            draft.info.block_size,
                            |_| true,
                        )
                        .map(|output| output.tokens)
                }
                _ => {
                    let draft = draft
                        .as_ref()
                        .ok_or_else(|| InferError::Config("drafter".into()))?;
                    let tree = if mode == "dflash-arbre" {
                        DFlashTreeMode::BestFirst(8)
                    } else {
                        DFlashTreeMode::Off
                    };
                    model
                        .generate_greedy_dflash_resident_tree(
                            &prompt,
                            max_new,
                            &options,
                            draft,
                            draft.info.block_size,
                            (tree, true),
                            |_| true,
                        )
                        .map(|output| output.tokens)
                }
            };
            let records = VERIFY_RECORDER
                .with(|slot| slot.borrow_mut().take())
                .unwrap_or_default();
            let tokens = tokens?;
            let mut tally = IdentityTally {
                first_token_divergence: tokens
                    .iter()
                    .zip(&ar)
                    .position(|(a, b)| a != b)
                    .or((tokens.len() != ar.len()).then(|| tokens.len().min(ar.len()))),
                ..IdentityTally::default()
            };
            identity_compare_records(
                &records,
                &sequence,
                prompt.len(),
                &ar_states,
                hidden,
                &mut tally,
            );
            eprintln!(
                "identité {name} {mode}: {} vérifications, {tally:?}, marge AR min {:.6} au token {}",
                records.len(),
                min_margin.0,
                min_margin.1
            );
            let seen = coverage.entry(mode).or_default();
            seen.0 += tally.multi_row_compared;
            seen.1 += tally.branched_records;
            // Largeur d'arbre réellement vérifiée (B + 1 lignes au plus).
            let widest_tree = records
                .iter()
                .filter(|record| record.parents.is_some())
                .map(|record| record.tokens.len())
                .max()
                .unwrap_or(0);
            let tree_width = tree_widths.entry(mode).or_insert(0);
            *tree_width = (*tree_width).max(widest_tree);
            // Repli sans argmax GPU : vérifications `Skip` de 5 lignes et plus
            // (lookup multiligne), lignes comparées à l'AR comprises.
            let skip_rows = records
                .iter()
                .filter(|record| record.skip_head && record.batched && record.tokens.len() >= 5)
                .count();
            *skip_coverage.entry(mode).or_insert(0) += skip_rows;
            // Rejeux en lignes après rejet : lot `Skip` de 5 lignes et plus qui
            // reprend, à la même position, le début de la vérification précédente.
            let replays = records
                .windows(2)
                .filter(|pair| {
                    let (verify, replay) = (&pair[0], &pair[1]);
                    replay.skip_head
                        && replay.batched
                        && replay.tokens.len() >= 5
                        && replay.base_position == verify.base_position
                        && replay.tokens.len() < verify.tokens.len()
                        && verify.tokens.starts_with(&replay.tokens)
                })
                .count();
            *replay_coverage.entry(mode).or_insert(0) += replays;
            if tally.first_token_divergence.is_some() || tally.state_mismatches > 0 {
                failures.push(format!("{name} {mode}: {tally:?}"));
            }
        }
    }
    eprintln!(
        "identité : {tokens_seen} tokens AR, quasi-égalités traversées < 1e-2 : {ties_2}, < 1e-3 : {ties_3}, couverture {coverage:?}"
    );
    for (mode, (multi, branched)) in &coverage {
        if *multi == 0 {
            failures.push(format!(
                "{mode}: aucune ligne d'un lot de plusieurs lignes comparée"
            ));
        }
        if matches!(*mode, "dflash-arbre" | "dflash-défaut") && *branched == 0 {
            failures.push(format!("{mode}: aucun arbre branché vérifié"));
        }
    }
    eprintln!("identité : largeur d'arbre max {tree_widths:?}, vérifications Skip ≥ 5 lignes {skip_coverage:?}");
    // `dflash-défaut` : la largeur observée est celle du budget par défaut
    // (B = 3 en greedy strict, B = 8 sinon), pas une autre.
    for mode in &selected {
        if !coverage.contains_key(mode) {
            failures.push(format!(
                "{mode}: tous ses prompts sautés, aucune vérification"
            ));
        }
    }
    if selected.contains("dflash-défaut") {
        let width = tree_widths.get("dflash-défaut").copied().unwrap_or(0);
        let expected = crate::runtime_flags::dflash_tree_default_budget(true) + 1;
        if width != expected {
            failures.push(format!(
                "dflash-défaut : arbre de {width} lignes au plus, {expected} attendu"
            ));
        }
    }
    eprintln!("identité : rejeux en lignes ≥ 5 lignes {replay_coverage:?}");
    // Couvertures obligatoires des modes de repli : `mtp-skip` a vérifié par la
    // tête `Skip` des lots de 5 lignes et plus (proposition lookup), `dflash-cpu`
    // a rejoué en lignes au moins 5 tokens acceptés après un rejet.
    if selected.contains("mtp-skip") && skip_coverage.get("mtp-skip").copied().unwrap_or(0) == 0 {
        failures.push("mtp-skip : aucune vérification Skip de 5 lignes et plus".to_string());
    }
    if selected.contains("dflash-cpu")
        && replay_coverage.get("dflash-cpu").copied().unwrap_or(0) == 0
    {
        failures.push("dflash-cpu : aucun rejeu de 5 lignes et plus".to_string());
    }
    assert!(
        failures.is_empty(),
        "identité greedy rompue : {failures:#?}"
    );
    Ok(())
}

/// Oracle O2 de la réutilisation partielle du cache de préfixe : le prompt moins
/// un suffixe de 5, 17 ou 31 tokens est mis en cache par le mode lui-même, puis
/// le prompt complet le reprend et prolonge le suffixe. Les tokens et les états
/// des lignes vérifiées doivent égaler ceux de l'AR soumis à la même reprise :
/// la portée greedy (mode strict compris) ne couvre que les vérifications
/// spéculatives, pas l'extension du préfixe. Exige le cache de préfixe actif ;
/// à lancer en mode strict et en défaut (le défaut peut diverger par le skinny
/// et le NAX des vérifications, voir docs/perf/greedy-identity.md).
#[test]
#[ignore = "modèle réel requis"]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn greedy_identity_prefix_reuse_real_model() -> Result<()> {
    use crate::decoder::resident::VERIFY_RECORDER;
    let Some(model_dir) = mtp_verifier_model_dir() else {
        return Ok(());
    };
    for name in [
        "SARAGOSSA_RUST_PREFIX_CACHE",
        "SARAGOSSA_RUST_PREFIX_CACHE_CAP",
    ] {
        if crate::runtime_flags::env_var(name).is_some_and(|v| v.trim() == "0") {
            return Err(InferError::Config(format!(
                "{name}=0 : oracle de réutilisation sans objet"
            )));
        }
    }
    // Chaque génération du prompt complet doit reprendre exactement le préfixe
    // déposé et prolonger le suffixe une seule fois, par le chemin résident.
    let traced = |label: &str, suffix: usize, run: &mut dyn FnMut() -> Result<()>| -> Result<()> {
        super::generation::PREFIX_EXTEND_TRACE.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
        let result = run();
        let trace = super::generation::PREFIX_EXTEND_TRACE
            .with(|slot| slot.borrow_mut().take())
            .unwrap_or_default();
        result?;
        if trace != [(suffix, true)] {
            return Err(InferError::Config(format!(
                "{label} : extension {trace:?}, attendu une reprise de suffixe {suffix} résidente"
            )));
        }
        Ok(())
    };
    let max_new = 48usize;
    let assets = crate::ModelAssets::load_local(&model_dir)?;
    let mut model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    let has_mtp = match assets.mtp.path.as_ref() {
        Some(path) => {
            model = model.with_mtp_sidecar(path)?;
            true
        }
        None => false,
    };
    let draft = if crate::runtime_flags::env_var_os("SARAGOSSA_RUST_DFLASH_DRAFT_DIR").is_some() {
        model = model.with_dflash_draft_from_env(&assets.config)?;
        model.dflash_draft().cloned()
    } else {
        None
    };
    let mut modes = Vec::new();
    if has_mtp {
        modes.push("mtp");
    }
    if draft.is_some() {
        modes.push("dflash-défaut");
    }
    if modes.is_empty() {
        return Err(InferError::Config("ni tête MTP ni drafter DFlash".into()));
    }
    let options = mtp_verifier_options(Vec::new());
    let hidden = model.final_norm.data().len();
    let run_mode = |mode: &str, prompt: &[usize], tokens: usize| -> Result<Vec<usize>> {
        match (mode, draft.as_ref()) {
            ("dflash-défaut", Some(draft)) => model
                .generate_greedy_dflash_resident_streaming_with_options(
                    prompt,
                    tokens,
                    &options,
                    draft,
                    draft.info.block_size,
                    |_| true,
                )
                .map(|output| output.tokens),
            _ => model
                .generate_greedy_mtp_streaming_with_options(prompt, tokens, &options, 2, |_| true)
                .map(|output| output.tokens),
        }
    };
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut exercised = std::collections::BTreeSet::new();
    for (name, text) in identity_corpus()? {
        let prompt = mtp_verifier_prompt_ids(&assets, &text)?;
        for suffix in [5usize, 17, 31] {
            // Le cache de préfixe refuse une reprise dont le suffixe dépasse la
            // moitié du prompt (repli à froid, plus rapide) : cas sauté ici, mais
            // chaque taille de suffixe doit être exercée au moins une fois.
            if suffix * 2 > prompt.len() {
                continue;
            }
            exercised.insert(suffix);
            let head = &prompt[..prompt.len() - suffix];
            // AR : préfixe en cache, puis le prompt complet le reprend.
            model.clear_prefix_cache();
            model.generate_greedy_cached_with_options(head, 1, &options)?;
            let mut ar = Vec::new();
            traced(&format!("{name} {suffix} AR"), suffix, &mut || {
                ar = model.generate_greedy_cached_with_options(&prompt, max_new, &options)?;
                Ok(())
            })?;
            // États AR après la même reprise partielle : pas mono-ligne.
            model.clear_prefix_cache();
            model.generate_greedy_cached_with_options(head, 1, &options)?;
            let kv_bf16 = CausalDecoder::resident_kv_bf16(false, prompt.len());
            let mut cache = None;
            traced(&format!("{name} {suffix} états AR"), suffix, &mut || {
                cache = Some(model.prefill_cache_state(&prompt)?.0);
                Ok(())
            })?;
            let mut cache =
                cache.ok_or_else(|| InferError::Config("cache AR absent".to_string()))?;
            if !model.setup_resident_full_decode(&mut cache, max_new + 16, 16, kv_bf16)? {
                return Err(InferError::Config("setup résident refusé".to_string()));
            }
            let mut ar_states = Vec::with_capacity(ar.len());
            for &token in &ar[..ar.len().saturating_sub(1)] {
                let out = model
                    .next_final_states_resident_verify(
                        &mut cache,
                        &[token],
                        ResidentVerifyHead::Argmax,
                        false,
                        None,
                        None,
                    )?
                    .ok_or_else(|| InferError::Config("pas mono indisponible".into()))?;
                ar_states.push(out.states.data().to_vec());
            }
            drop(cache);
            let mut sequence = prompt.clone();
            sequence.extend_from_slice(&ar);
            for &mode in &modes {
                // Préfixe déposé par le mode lui-même (historique MTP, captures DFlash).
                model.clear_prefix_cache();
                run_mode(mode, head, 1)?;
                VERIFY_RECORDER.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
                let mut tokens = Ok(Vec::new());
                let traced_run = traced(&format!("{name} {suffix} {mode}"), suffix, &mut || {
                    tokens = run_mode(mode, &prompt, max_new);
                    Ok(())
                });
                let records = VERIFY_RECORDER
                    .with(|slot| slot.borrow_mut().take())
                    .unwrap_or_default();
                let tokens = tokens?;
                traced_run?;
                let mut tally = IdentityTally {
                    first_token_divergence: tokens
                        .iter()
                        .zip(&ar)
                        .position(|(a, b)| a != b)
                        .or((tokens.len() != ar.len()).then(|| tokens.len().min(ar.len()))),
                    ..IdentityTally::default()
                };
                identity_compare_records(
                    &records,
                    &sequence,
                    prompt.len(),
                    &ar_states,
                    hidden,
                    &mut tally,
                );
                eprintln!("réutilisation {name} suffixe {suffix} {mode}: {tally:?}");
                compared += tally.compared_rows;
                if tally.first_token_divergence.is_some() || tally.state_mismatches > 0 {
                    failures.push(format!("{name} suffixe {suffix} {mode}: {tally:?}"));
                }
            }
        }
    }
    model.clear_prefix_cache();
    assert!(
        compared > 0,
        "réutilisation de préfixe : aucune ligne comparée"
    );
    assert_eq!(
        exercised.into_iter().collect::<Vec<_>>(),
        vec![5, 17, 31],
        "réutilisation de préfixe : chaque suffixe doit être exercé (prompts trop courts ?)"
    );
    assert!(
        failures.is_empty(),
        "réutilisation de préfixe ≠ AR : {failures:#?}"
    );
    Ok(())
}

/// Le branchement du prefill MTP doit être exercé par la génération : le repli
/// calcule aussi les bons K/V, mais les matérialise sur CPU avant leur publication.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "device Metal requis, données synthétiques"]
fn mtp_generation_publishes_shared_seed_history() -> Result<()> {
    use crate::decode_resident::KvSegment;

    assert!(prefix_cache_enabled());
    assert_eq!(mtp_history_policy(), MtpHistoryPolicy::Committed);
    const KV_DIM: usize = 2 * 256;
    let (model, prompt) = mtp_seed_synthetic_model()?;
    let history = || -> Result<LayerKvCache> {
        let cache = model
            .prefix_cache
            .lock()
            .map_err(|_| InferError::Config("cache préfixe empoisonné".into()))?;
        cache
            .entries
            .iter()
            .find(|entry| entry.tokens == prompt)
            .and_then(|entry| entry.mtp_history.clone())
            .ok_or_else(|| InferError::Config("historique MTP absent du préfixe".into()))
    };
    // Respecter la cible minimale du runtime (512) : les trois morceaux
    // ouvrent Steel/NAX dès le premier bloc (342 > 256). Une cible
    // artificielle de 32 force un repli per-op avant même le semis MTP.
    crate::metal_backend::with_prefill_kv_layout_override(512, usize::MAX, || -> Result<()> {
        let mut emitted = Vec::new();
        let first = model.generate_greedy_mtp_streaming_with_options(
            &prompt,
            1,
            &GenerationOptions::default(),
            1,
            |token| {
                emitted.push(token);
                true
            },
        )?;
        assert_eq!(first.tokens.len(), 1);
        assert_eq!(emitted, first.tokens);
        assert_eq!(first.stats.reused_prefix_tokens, 0);
        let saved = history()?;
        assert_eq!(saved.len(), prompt.len() - 1);
        for rows in [&saved.keys, &saved.values] {
            let segments = rows.segments();
            let [KvSegment::Shared(shared)] = segments.as_slice() else {
                return Err(InferError::Config(
                    "semis intégré non emprunté : historique MTP sur CPU".into(),
                ));
            };
            assert_eq!(shared.len(), (prompt.len() - 1) * KV_DIM);
        }

        let second = model.generate_greedy_mtp_streaming_with_options(
            &prompt,
            1,
            &GenerationOptions::default(),
            1,
            |_| true,
        )?;
        assert_eq!(second.tokens, first.tokens);
        assert_eq!(second.stats.reused_prefix_tokens, prompt.len());
        let reused = history()?;
        for (before, after) in [(&saved.keys, &reused.keys), (&saved.values, &reused.values)] {
            let before = before.segments();
            let after = after.segments();
            let ([KvSegment::Shared(before)], [KvSegment::Shared(after)]) =
                (before.as_slice(), after.as_slice())
            else {
                return Err(InferError::Config("historique MTP recopié sur CPU".into()));
            };
            assert_eq!(before.allocation_id(), after.allocation_id());
            assert_eq!(before.as_slice().as_ptr(), after.as_slice().as_ptr());
        }
        Ok(())
    })
}

/// Le préfixe retient les buffers MTP figés ; prolonger ou tronquer la copie
/// de travail ne réécrit aucune ligne publiée et ne matérialise pas de Vec.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "device Metal requis, données synthétiques"]
fn mtp_seed_shared_prefix_keeps_allocations() -> Result<()> {
    use crate::metal_backend::{MetalExecutor, PrefillKvRows};
    let metal = MetalExecutor::new()?;
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let k = metal.shared_f32_rows_for_test(&[1.0, 2.0, 3.0, 4.0])?;
    let v = metal.shared_f32_rows_for_test(&[5.0, 6.0, 7.0, 8.0])?;
    let mut history = LayerKvCache::default();
    let layout = AttentionLayout {
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        rope_dims: 2,
        attn_scalar: 2.0,
        sliding_window: None,
    };
    history.append_prefill(
        PrefillKvRows::Shared(k.clone()),
        PrefillKvRows::Shared(v.clone()),
        2,
        0,
        &layout,
    )?;
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    model.prefix_cache_put(&[0, 1, 2], &state, Some(&history), None)?;
    // Un remplacement exact sans historique doit garder les mêmes segments.
    model.prefix_cache_put(&[0, 1, 2], &state, None, None)?;
    history.append(
        &Tensor::row(vec![9.0, 10.0])?,
        &Tensor::row(vec![11.0, 12.0])?,
        &layout,
    )?;
    history.truncate(1)?;
    let cache = model
        .prefix_cache
        .lock()
        .map_err(|_| InferError::Config("cache préfixe empoisonné".to_string()))?;
    let saved = cache
        .entries
        .first()
        .and_then(|entry| entry.mtp_history.as_ref())
        .ok_or_else(|| InferError::Config("historique MTP absent du préfixe".to_string()))?;
    assert_eq!(saved.len(), 2);
    for (rows, source) in [(&saved.keys, &k), (&saved.values, &v)] {
        let segments = rows.segments();
        let [crate::decode_resident::KvSegment::Shared(shared)] = segments.as_slice() else {
            return Err(InferError::Dimension(
                "segment MTP recopié sur CPU".to_string(),
            ));
        };
        assert_eq!(shared.allocation_id(), source.allocation_id());
        assert_eq!(shared.as_slice().as_ptr(), source.as_slice().as_ptr());
        assert_eq!(rows.to_vec(), source.as_slice());
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn mtp_seed_synthetic_model() -> Result<(CausalDecoder, Vec<usize>)> {
    const VOCAB: usize = 97;
    const HIDDEN: usize = 64;
    let config = CausalDecoderConfig {
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: Some(256),
        rope_theta: Some(10_000_000.0),
        // Le vérificateur dense batché exige une porte de sortie ; la
        // projection Q porte donc Q et gate, dans le tronc comme dans MTP.
        attn_output_gate: true,
        ..CausalDecoderConfig::default()
    };
    let weights = resident_bf16_qwen_weights(VOCAB, HIDDEN, config.attn_output_gate)?;
    let mut model = CausalDecoder::from_tensors(weights, config)?;
    let layer = &model.layers[0];
    let AttentionBlock::Full(attention) = &layer.attention else {
        return Err(InferError::Config("fixture sans attention complète".into()));
    };
    model.mtp = Some(MtpHead {
        pre_fc_norm_embedding: layer.input_norm.clone(),
        pre_fc_norm_hidden: model.final_norm.clone(),
        fc: Linear::new(
            Tensor::from_vec(
                vec![HIDDEN, 2 * HIDDEN],
                (0..HIDDEN * 2 * HIDDEN)
                    .map(|i| ((i * 37 % 257) as f32 - 128.0) / 1280.0)
                    .collect(),
            )?,
            None,
        )?,
        layer: MtpLayer {
            input_norm: layer.input_norm.clone(),
            attention: attention.as_ref().clone(),
            post_attention_norm: layer
                .post_attention_norm
                .clone()
                .ok_or_else(|| InferError::Config("fixture sans norme MLP".into()))?,
            mlp: layer
                .mlp
                .clone()
                .ok_or_else(|| InferError::Config("fixture sans MLP".into()))?,
        },
        norm: model.final_norm.clone(),
    });
    let model = model.with_metal_runtime()?;
    let prompt: Vec<usize> = (0..1025).map(|i| (i * 31 + 7) % VOCAB).collect();
    Ok((model, prompt))
}

/// Couvre le branchement réel, l'adoption, la copie sous contention, les
/// préfixes chauds/étendus et le retour au contrat f32 forcé.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "device Metal requis, données synthétiques"]
fn mtp_bf16_history_generation_adoption_and_f32_fallback() -> Result<()> {
    use crate::decode_resident::{with_kv_format_override, KvFormat, KvSegment};
    assert!(decode_resident_full_enabled());
    assert_eq!(mtp_history_policy(), MtpHistoryPolicy::Committed);
    assert!(prefix_cache_enabled());
    let (model, prompt) = mtp_seed_synthetic_model()?;
    let capacity = prompt.len() + 16;
    crate::metal_backend::with_prefill_kv_layout_override(512, usize::MAX, || -> Result<()> {
        with_kv_format_override(KvFormat::Bf16, || -> Result<()> {
            let (mut cache, _, history, _) =
                super::mtp_history::with_storage(Some(capacity), || {
                    model.prefill_cache_state_with_mtp_history(&prompt, &mut |_| Ok(true))
                })?;
            let mut concurrent = cache.clone();
            assert!(model.setup_resident_full_decode_with_mtp_history(
                &mut cache,
                14,
                2,
                true,
                Some(&history),
            )?);
            assert!(model.setup_resident_full_decode_with_mtp_history(
                &mut concurrent,
                14,
                2,
                true,
                Some(&history),
            )?);
            let mtp = &cache
                .resident
                .as_ref()
                .expect("arène")
                .mtp
                .as_ref()
                .expect("MTP")
                .kv;
            let copy = &concurrent
                .resident
                .as_ref()
                .expect("arène")
                .mtp
                .as_ref()
                .expect("MTP")
                .kv;
            let mut seen = std::collections::HashSet::new();
            for (rows, adopted, copied) in [
                (&history.keys, mtp.keys(), copy.keys()),
                (&history.values, mtp.values(), copy.values()),
            ] {
                let segments = rows.segments();
                let [KvSegment::SharedBf16(view)] = segments.as_slice() else {
                    return Err(InferError::Config("semis bf16 intégré non exercé".into()));
                };
                assert_eq!(view.buffer().contents(), adopted.buffer().contents());
                assert_ne!(view.buffer().contents(), copied.buffer().contents());
                assert_eq!(
                    view.as_slice(),
                    crate::metal_backend::read_u16_buffer(copied.buffer(), view.len())?
                );
                assert_eq!(rows.capacity_bytes_unseen(&mut seen), capacity * 512 * 2);
            }
            // Le cache de préfixe doit conserver ces mêmes allocations.
            let prefixes = model.prefix_cache.lock().expect("préfixe");
            let saved = prefixes.entries[0]
                .mtp_history
                .as_ref()
                .expect("historique");
            assert_eq!(saved.keys.capacity_bytes_unseen(&mut seen), 0);
            assert_eq!(saved.values.capacity_bytes_unseen(&mut seen), 0);
            drop(prefixes);
            drop(cache);
            drop(concurrent);
            drop(history);

            let options = GenerationOptions::default();
            let mut extended = prompt.clone();
            extended.extend_from_slice(&[3, 7, 11]);
            for depth in [1, 2] {
                model.clear_prefix_cache();
                let ar = model.generate_greedy_cached_with_options(&prompt, 8, &options)?;
                model.clear_prefix_cache();
                for warm in [false, true] {
                    let output = model
                        .generate_greedy_mtp_batched_with_options(&prompt, 8, &options, depth)?;
                    assert_eq!(output.tokens, ar, "D{depth}, chaud={warm}");
                    assert_eq!(
                        output.stats.reused_prefix_tokens,
                        if warm { prompt.len() } else { 0 }
                    );
                }
                let output = model
                    .generate_greedy_mtp_batched_with_options(&extended, 8, &options, depth)?;
                assert_eq!(output.stats.reused_prefix_tokens, prompt.len());
                model.clear_prefix_cache();
                let ar = model.generate_greedy_cached_with_options(&extended, 8, &options)?;
                assert_eq!(output.tokens, ar, "D{depth}, extension");
            }
            // Remettre une entrée bf16, puis exiger une reconstruction f32.
            model.clear_prefix_cache();
            model.generate_greedy_mtp_batched_with_options(&prompt, 8, &options, 1)?;
            with_kv_format_override(KvFormat::F32, || -> Result<()> {
                let output =
                    model.generate_greedy_mtp_batched_with_options(&prompt, 8, &options, 1)?;
                assert_eq!(output.stats.reused_prefix_tokens, 0);
                let prefixes = model.prefix_cache.lock().expect("préfixe");
                let saved = prefixes.entries[0]
                    .mtp_history
                    .as_ref()
                    .expect("historique");
                assert!(!saved.keys.has_bf16() && !saved.values.has_bf16());
                drop(prefixes);
                model.clear_prefix_cache();
                let cold =
                    model.generate_greedy_mtp_batched_with_options(&prompt, 8, &options, 1)?;
                assert_eq!(output.tokens, cold.tokens);
                Ok(())
            })
        })
    })
}

/// La copie MTP refuse après l'installation du tronc : tous ses claims
/// doivent être rendus, et le refus doit rester un repli, pas une erreur.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "device Metal requis, données synthétiques"]
fn mtp_bf16_busy_claim_memory_guard_cleans_setup() -> Result<()> {
    use crate::decode_resident::{with_kv_format_override, KvFormat};
    use crate::metal_backend::InPlaceClaim;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (mut model, prompt) = mtp_seed_synthetic_model()?;
    crate::metal_backend::with_prefill_kv_layout_override(512, usize::MAX, || {
        with_kv_format_override(KvFormat::Bf16, || -> Result<()> {
            let (mut cache, _, history, _) = with_prefill_kv_reserve(16, || {
                super::mtp_history::with_storage(Some(prompt.len() + 16), || {
                    model.prefill_cache_state_with_mtp_history(&prompt, &mut |_| Ok(true))
                })
            })?;
            let keys = history.keys.adoptable_bf16_prefix().expect("semis bf16");
            let values = history.values.adoptable_bf16_prefix().expect("semis bf16");
            let claim = InPlaceClaim::try_after([keys, values]).expect("claim MTP libre");
            // Le tronc f32 impose une copie : estimation et copie passent,
            // puis la copie MTP sous claim occupé épuise le budget.
            assert_eq!(cache.layers.len(), 1);
            assert!(!cache.layers[0].keys.has_bf16());
            let probes = Arc::new(AtomicUsize::new(0));
            let observed = probes.clone();
            const LIMIT: u64 = 1 << 40;
            model.memory_guard = Some(MemoryGuard::from_limits(
                true,
                crate::memory_guard::MemoryLimits {
                    static_cap: Some(LIMIT),
                    host_memory: None,
                    headroom_bytes: 0,
                    metal_cap: None,
                },
                Arc::new(move || {
                    Some(if observed.fetch_add(1, Ordering::SeqCst) == 2 {
                        LIMIT
                    } else {
                        0
                    })
                }),
            ));
            assert!(!model.setup_resident_full_decode_with_mtp_history(
                &mut cache,
                14,
                2,
                true,
                Some(&history),
            )?);
            assert_eq!(probes.load(Ordering::SeqCst), 3);
            assert!(cache.resident.is_none());
            for layer in &cache.layers {
                assert!(layer.full.is_none());
            }
            drop(claim);
            model.memory_guard = None;
            assert!(model.setup_resident_full_decode_with_mtp_history(
                &mut cache,
                14,
                2,
                true,
                Some(&history),
            )?);
            Ok(())
        })
    })
}

/// Refus réel de la garde APRÈS semis bf16, à froid et après extension.
/// Compare le triplet reconstruit et la génération, émission précoce comprise.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "device Metal requis, données synthétiques"]
fn mtp_bf16_post_seed_refusal_rebuilds_f32_and_streams_once() -> Result<()> {
    use crate::decode_resident::{with_kv_format_override, KvFormat};
    use std::sync::atomic::{AtomicUsize, Ordering};
    assert!(decode_resident_full_enabled());
    assert!(prefix_cache_enabled());
    assert_eq!(mtp_history_policy(), MtpHistoryPolicy::Committed);
    assert!(crate::runtime_flags::mtp_early_first_token_enabled());
    assert!(!crate::runtime_flags::mtp_gpu_block_enabled());
    let (mut model, prompt) = mtp_seed_synthetic_model()?;
    let mut extended = prompt.clone();
    extended.extend_from_slice(&[3, 7, 11]);
    let options = GenerationOptions::default();
    crate::metal_backend::with_prefill_kv_layout_override(512, usize::MAX, || {
        with_kv_format_override(KvFormat::Bf16, || -> Result<()> {
            for depth in [1, 2] {
                for warm in [false, true] {
                    let target = if warm { &extended } else { &prompt };
                    let reserve = 8 + depth;
                    model.clear_prefix_cache();
                    let (reference_cache, reference_state, reference_history, _) =
                        with_prefill_kv_reserve(reserve, || {
                            super::mtp_history::with_storage(None, || {
                                model
                                    .prefill_cache_state_with_mtp_history(target, &mut |_| Ok(true))
                            })
                        })?;
                    let seed = || -> Result<_> {
                        if warm {
                            with_prefill_kv_reserve(reserve + 3, || {
                                super::mtp_history::with_storage(
                                    Some(target.len() + reserve),
                                    || {
                                        model.prefill_cache_state_with_mtp_history(
                                            &prompt,
                                            &mut |_| Ok(true),
                                        )
                                    },
                                )
                            })?;
                        }
                        with_prefill_kv_reserve(reserve, || {
                            super::mtp_history::with_storage(Some(target.len() + reserve), || {
                                model
                                    .prefill_cache_state_with_mtp_history(target, &mut |_| Ok(true))
                            })
                        })
                    };
                    model.clear_prefix_cache();
                    let (cache, state, history, reused) = seed()?;
                    assert_eq!(reused, if warm { prompt.len() } else { 0 });
                    assert!(history.keys.has_bf16() && history.values.has_bf16());
                    // Le probe atteste que le refus survient avec l'historique
                    // compact déjà publié ; le repli f32 ne doit pas le rappeler.
                    let prefixes = model.prefix_cache.clone();
                    let refusals = Arc::new(AtomicUsize::new(0));
                    let observed = refusals.clone();
                    model.memory_guard = Some(MemoryGuard::from_limits(
                        true,
                        crate::memory_guard::MemoryLimits {
                            static_cap: Some(1),
                            host_memory: None,
                            headroom_bytes: 0,
                            metal_cap: None,
                        },
                        Arc::new(move || {
                            let prefixes = prefixes.lock().expect("préfixe");
                            assert!(prefixes.entries.iter().any(|entry| entry
                                .mtp_history
                                .as_ref()
                                .is_some_and(|h| h.keys.has_bf16() && h.values.has_bf16())));
                            observed.fetch_add(1, Ordering::SeqCst);
                            Some(0)
                        }),
                    ));
                    // Un état sentinelle détecte la conservation accidentelle du
                    // premier final_state, même si les deux prefills coïncident.
                    let stale_state = Tensor::row(vec![f32::NAN; state.data().len()])?;
                    let (rebuilt, final_state, history) = model.prepare_mtp_decode(
                        target,
                        8,
                        depth,
                        &options,
                        (cache, stale_state, Some(history)),
                    )?;
                    assert_eq!(refusals.load(Ordering::SeqCst), 1);
                    assert_eq!(final_state.data(), reference_state.data());
                    assert_eq!(rebuilt.position, reference_cache.position);
                    for (actual, expected) in rebuilt.layers.iter().zip(&reference_cache.layers) {
                        assert_eq!(
                            actual.keys.snapshot_segments(),
                            expected.keys.snapshot_segments()
                        );
                        assert_eq!(
                            actual.values.snapshot_segments(),
                            expected.values.snapshot_segments()
                        );
                    }
                    let history = history.expect("historique reconstruit");
                    assert!(!history.keys.has_bf16() && !history.values.has_bf16());
                    assert_eq!(
                        history.keys.snapshot_segments(),
                        reference_history.keys.snapshot_segments()
                    );
                    assert_eq!(
                        history.values.snapshot_segments(),
                        reference_history.values.snapshot_segments()
                    );
                    drop((rebuilt, history, reference_cache, reference_history));
                    model.memory_guard = None;
                    model.clear_prefix_cache();
                    let ar = model.generate_greedy_cached_with_options(target, 8, &options)?;
                    model.clear_prefix_cache();
                    if warm {
                        with_prefill_kv_reserve(reserve + 3, || {
                            super::mtp_history::with_storage(Some(target.len() + reserve), || {
                                model.prefill_cache_state_with_mtp_history(&prompt, &mut |_| {
                                    Ok(true)
                                })
                            })
                        })?;
                    }
                    let prefixes = model.prefix_cache.clone();
                    let observed = refusals.clone();
                    let emissions = Arc::new(AtomicUsize::new(0));
                    let before_setup = emissions.clone();
                    // NOTE: l'extension du préfixe peut aussi tenter le résident
                    // pendant le prefill, avant toute émission ; seul le dernier
                    // refus est celui du setup MTP après le premier token.
                    let contexts = Arc::new(Mutex::new(Vec::new()));
                    let seen = contexts.clone();
                    model.memory_guard = Some(MemoryGuard::from_limits(
                        true,
                        crate::memory_guard::MemoryLimits {
                            static_cap: Some(1),
                            host_memory: None,
                            headroom_bytes: 0,
                            metal_cap: None,
                        },
                        Arc::new(move || {
                            let bf16 =
                                prefixes
                                    .lock()
                                    .expect("préfixe")
                                    .entries
                                    .iter()
                                    .any(|entry| {
                                        entry
                                            .mtp_history
                                            .as_ref()
                                            .is_some_and(|h| h.keys.has_bf16())
                                    });
                            seen.lock()
                                .expect("contextes")
                                .push((before_setup.load(Ordering::SeqCst), bf16));
                            observed.fetch_add(1, Ordering::SeqCst);
                            Some(0)
                        }),
                    ));
                    let mut emitted = Vec::new();
                    let output = model.generate_greedy_mtp_streaming_with_options(
                        target,
                        8,
                        &options,
                        depth,
                        |token| {
                            emissions.fetch_add(1, Ordering::SeqCst);
                            emitted.push(token);
                            true
                        },
                    )?;
                    let contexts = contexts.lock().expect("contextes").clone();
                    assert_eq!(refusals.load(Ordering::SeqCst), 1 + contexts.len());
                    assert!(
                        contexts.iter().all(|&(emitted, _)| emitted <= 1),
                        "{contexts:?}"
                    );
                    assert_eq!(
                        contexts.iter().filter(|c| c.0 == 1).count(),
                        1,
                        "{contexts:?}"
                    );
                    assert_eq!(
                        contexts.last().copied(),
                        Some((1, true)),
                        "setup MTP refusé après le premier token, historique bf16 présent"
                    );
                    assert_eq!(
                        output.stats.reused_prefix_tokens,
                        if warm { prompt.len() } else { 0 }
                    );
                    assert_eq!(output.tokens, ar, "D{depth}, extension={warm}");
                    assert_eq!(
                        emitted, output.tokens,
                        "chaque position émise une seule fois"
                    );
                    let prefixes = model.prefix_cache.lock().expect("préfixe");
                    let saved = prefixes
                        .entries
                        .iter()
                        .find(|entry| entry.tokens == *target)
                        .expect("préfixe reconstruit");
                    assert_eq!(saved.final_state.data(), reference_state.data());
                    assert!(prefixes.entries.iter().all(|entry| entry
                        .mtp_history
                        .as_ref()
                        .is_none_or(|h| !h.keys.has_bf16() && !h.values.has_bf16())));
                    drop(prefixes);
                    model.memory_guard = None;
                }
            }
            Ok(())
        })
    })
}

#[test]
fn mtp_f32_fallback_discards_all_bf16_prefix_references_cpu() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    let mut bf16 = LayerKvCache::default();
    bf16.keys.push_owned_bf16(vec![0x3f80]);
    bf16.values.push_owned_bf16(vec![0x4000]);
    // Entrées de branches différentes partageant le même historique compact.
    model.prefix_cache_put(&[1, 2], &state, Some(&bf16), None)?;
    model.prefix_cache_put(&[1, 3], &state, Some(&bf16), None)?;
    let f32 = LayerKvCache::default();
    model.prefix_cache_put(&[4, 5], &state, Some(&f32), None)?;
    model.discard_bf16_mtp_prefixes()?;
    let prefixes = model.prefix_cache.lock().expect("préfixe");
    assert_eq!(prefixes.entries.len(), 1);
    assert_eq!(prefixes.entries[0].tokens, [4, 5]);
    Ok(())
}

/// Une entrée compacte est un état résident, jamais un oracle f32 de draft.
#[test]
fn mtp_bf16_prefix_compatibility_and_shared_accounting_cpu() -> Result<()> {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())?;
    let state = (model.empty_cache(), Tensor::row(vec![0.0])?);
    let mut history = LayerKvCache::default();
    history
        .keys
        .push_owned_bf16(vec![0x3f80, 0x8000, 0x7fc1, 0xc000]);
    history
        .values
        .push_owned_bf16(vec![0x3f81, 0x0001, 0x7f80, 0x0000]);
    model.prefix_cache_put(&[1, 2], &state, Some(&history), None)?;
    assert!(model.prefix_cache_hit(&[1, 2], true, usize::MAX)?.is_none());
    super::mtp_history::with_storage(Some(16), || -> Result<()> {
        let (_, _, hit, _) = model
            .prefix_cache_hit(&[1, 2], true, usize::MAX)?
            .expect("préfixe bf16");
        let hit = hit.expect("historique");
        let mut seen = std::collections::HashSet::new();
        assert_eq!(
            history.keys.capacity_bytes_unseen(&mut seen)
                + history.values.capacity_bytes_unseen(&mut seen),
            16
        );
        assert_eq!(
            hit.keys.capacity_bytes_unseen(&mut seen) + hit.values.capacity_bytes_unseen(&mut seen),
            0
        );
        assert_eq!(
            hit.keys.snapshot_segments(),
            history.keys.snapshot_segments()
        );
        assert_eq!(
            hit.values.snapshot_segments(),
            history.values.snapshot_segments()
        );
        Ok(())
    })?;
    // Un historique f32 reste réutilisable dans les deux contrats.
    history.keys = kv_rows::KvRows::default();
    history.values = kv_rows::KvRows::default();
    history.keys.push_owned(vec![1.000_001, -0.0]);
    history.values.push_owned(vec![2.000_001, 0.0]);
    model.prefix_cache_put(&[1, 2], &state, Some(&history), None)?;
    let (_, _, hit, _) = model
        .prefix_cache_hit(&[1, 2], true, usize::MAX)?
        .expect("préfixe f32");
    let hit = hit.expect("historique");
    assert_eq!(
        hit.keys.snapshot_segments(),
        history.keys.snapshot_segments()
    );
    assert_eq!(
        hit.values.snapshot_segments(),
        history.values.snapshot_segments()
    );
    Ok(())
}

/// Validation explicite sur poids locaux : taille configurable pour 4k/32k/256k.
/// Aucune recherche ni acquisition de modèle ; absence de paramètres = échec.
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[test]
#[ignore = "requiert Metal et un modèle MTP local explicitement fourni"]
fn mtp_bf16_real_greedy_ids_and_memory() -> Result<()> {
    assert!(decode_resident_full_enabled());
    assert!(prefix_cache_enabled());
    assert_eq!(mtp_history_policy(), MtpHistoryPolicy::Committed);
    assert_eq!(crate::decode_resident::kv_element_bytes(true), 2);
    let dir = mtp_verifier_model_dir()
        .ok_or_else(|| InferError::Config("modèle local MTP requis".into()))?;
    let assets = crate::ModelAssets::load_local(&dir)?;
    let sidecar = assets
        .mtp
        .path
        .as_ref()
        .ok_or_else(|| InferError::Config("sidecar MTP local requis".into()))?;
    let model = crate::load_causal_decoder(&assets)?
        .with_metal_runtime()?
        .with_mtp_sidecar(sidecar)?;
    let number = |name: &str, default| -> Result<usize> {
        std::env::var(name).ok().map_or(Ok(default), |s| {
            s.parse()
                .map_err(|_| InferError::Config(format!("{name} invalide")))
        })
    };
    let count = number("SARAGOSSA_MTP_VALIDATION_TOKENS", 4096)?;
    let generated = number("SARAGOSSA_MTP_VALIDATION_NEW_TOKENS", 64)?;
    let depth = number("SARAGOSSA_MTP_VALIDATION_DEPTH", 1)?;
    assert!(count >= 2 && generated > 1 && (1..=2).contains(&depth));
    let text = match std::env::var_os("SARAGOSSA_MTP_VALIDATION_PROMPT") {
        Some(path) => {
            std::fs::read_to_string(path).map_err(|e| InferError::Config(e.to_string()))?
        }
        None => {
            "Analyse ce module Rust de cache et explique ses invariants de concurrence. ".repeat(64)
        }
    };
    let ids = mtp_verifier_prompt_ids(&assets, &text)?;
    assert!(!ids.is_empty());
    let prompt: Vec<_> = ids.into_iter().cycle().take(count).collect();
    let options = mtp_verifier_options(Vec::new());
    let ar = model.generate_greedy_cached_with_options(&prompt, generated, &options)?;
    model.clear_prefix_cache();
    for warm in [false, true] {
        let output =
            model.generate_greedy_mtp_batched_with_options(&prompt, generated, &options, depth)?;
        assert_eq!(output.tokens, ar, "greedy AR ≠ MTP, D{depth}, chaud={warm}");
        assert_eq!(
            output.stats.reused_prefix_tokens,
            if warm { count } else { 0 }
        );
        let prefixes = model.prefix_cache.lock().expect("préfixe");
        let history = prefixes
            .entries
            .iter()
            .find(|e| e.tokens == prompt)
            .and_then(|e| e.mtp_history.as_ref())
            .expect("historique committed publié");
        assert!(history.keys.has_bf16() && history.values.has_bf16());
        let mut seen = std::collections::HashSet::new();
        let bytes = history.keys.capacity_bytes_unseen(&mut seen)
            + history.values.capacity_bytes_unseen(&mut seen);
        let kv_dim = history.kv_dim.expect("dimension KV");
        let f32_bytes = (count - 1) * kv_dim * 2 * size_of::<f32>();
        eprintln!("MTP prompt={count} D{depth} chaud={warm}: historique={bytes} octets, ancien f32={f32_bytes}, ids={:?}", output.tokens);
        assert!(bytes < f32_bytes, "historique compact attendu");
    }
    Ok(())
}

#[test]
fn gemma4_proportional_cpu_paths_match_independent_oracle() -> Result<()> {
    let head_dim = 512;
    let heads = 2;
    let rows = 3;
    let mut input: Vec<f32> = (0..heads * rows * head_dim)
        .map(|i| ((i * 17 + 3) % 97) as f32 / 32.0 - 1.5)
        .collect();
    input[..head_dim].fill(0.0);
    input[0] = 1.0;
    let x = Tensor::from_vec(vec![rows, heads * head_dim], input.clone())?;
    let weight = Tensor::from_vec(vec![head_dim], vec![1.0; head_dim])?;
    // Gemma global 0.25 et 0.5, sliding complet, Qwen partiel.
    for (rotated_dims, rotation_dim, theta) in [
        (128, 512, 1_000_000.0),
        (256, 512, 1_000_000.0),
        (512, 512, 10_000.0),
        (128, 128, 10_000.0),
    ] {
        let rope = RopeParams {
            theta,
            frequency_dim: rotation_dim,
            position_scale: 1.0,
        };
        let plain = apply_rope_heads_at(
            &x,
            heads,
            head_dim,
            rotated_dims,
            rope,
            1,
            RopeStyle::Halves,
        )?;
        let fused = rms_norm_rope_heads_at(
            &x,
            heads,
            head_dim,
            rotated_dims,
            &weight,
            1.0e-6,
            rope,
            1,
            RopeStyle::Halves,
        )?;
        for (index, head) in input.chunks_exact(head_dim).enumerate() {
            let position = 1 + index / heads;
            let expected = crate::rope::reference_rotate_half(
                head,
                rotated_dims,
                rotation_dim,
                theta,
                position,
            );
            let start = index * head_dim;
            assert_close(&plain.data()[start..start + head_dim], &expected, 1.0e-6);
            let inv_rms =
                1.0 / (head.iter().map(|x| x * x).sum::<f32>() / head_dim as f32 + 1.0e-6).sqrt();
            let normed: Vec<_> = head.iter().map(|x| x * inv_rms).collect();
            let expected = crate::rope::reference_rotate_half(
                &normed,
                rotated_dims,
                rotation_dim,
                theta,
                position,
            );
            assert_close(&fused.data()[start..start + head_dim], &expected, 1.0e-6);
        }
        assert_eq!(plain.data()[0], 1.0_f32.cos());
        assert_eq!(plain.data()[rotation_dim / 2], 1.0_f32.sin());
        for col in 0..head_dim {
            let active = col < rotated_dims / 2
                || (col >= rotation_dim / 2 && col < rotation_dim / 2 + rotated_dims / 2);
            if !active {
                for (index, head) in input.chunks_exact(head_dim).enumerate() {
                    assert_eq!(plain.data()[index * head_dim + col], head[col]);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn gemma4_explicit_rope_parameters_reach_decoder() -> Result<()> {
    let raw: crate::RawModelConfig = serde_json::from_value(serde_json::json!({
        "model_type": "gemma4_text",
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 500000.0},
            "sliding_attention": {"rope_theta": 20000.0}
        }
    }))
    .expect("invariant: JSON de test valide");
    let resolved = raw.resolve()?;
    let config = CausalDecoderConfig::from(&resolved);
    let full = LayerPlan::build(&config, 1).resident;
    assert_eq!(full.rope_dims(512), 256);
    assert_eq!(full.rope_frequency_dim(512), 512);
    assert_eq!(full.rope_theta_override, Some(500_000.0));
    let sliding = LayerPlan::build(&config, 0).resident;
    assert_eq!(sliding.rope_dims(256), 256);
    assert_eq!(sliding.rope_frequency_dim(256), 256);
    assert_eq!(sliding.rope_theta_override, Some(20_000.0));
    Ok(())
}
