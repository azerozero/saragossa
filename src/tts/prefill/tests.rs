use super::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::InferError;
use crate::{CausalDecoder, CausalDecoderConfig};

// Locaux au thread pour observer le chemin réel sans coupler les tests parallèles.
thread_local! {
    pub(super) static FULL_PREFILLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static MEMO_REUSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn decoder(layers: usize) -> Result<CausalDecoder> {
    let mut weights = std::collections::HashMap::new();
    for name in ["embed_tokens.weight", "lm_head.weight"] {
        weights.insert(
            name.into(),
            Tensor::from_vec(vec![2, 2], vec![0.2, -0.3, 0.4, 0.7])?,
        );
    }
    weights.insert("norm.weight".into(), Tensor::row(vec![1.0; 2])?);
    for layer in 0..layers {
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            weights.insert(
                format!("layers.{layer}.{norm}.weight"),
                Tensor::row(vec![1.0; 2])?,
            );
        }
        for projection in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            weights.insert(
                format!("layers.{layer}.{projection}.weight"),
                Tensor::from_vec(vec![2, 2], vec![0.17, -0.31, 0.43, 0.79])?,
            );
        }
        for norm in ["q_norm", "k_norm"] {
            weights.insert(
                format!("layers.{layer}.self_attn.{norm}.weight"),
                Tensor::row(vec![1.0; 2])?,
            );
        }
    }
    // Aucun runtime Metal : ces poids synthétiques restent exclusivement CPU.
    CausalDecoder::from_tensors(
        weights,
        CausalDecoderConfig {
            num_hidden_layers: layers,
            head_dim: Some(2),
            rope_theta: Some(10_000.0),
            ..CausalDecoderConfig::default()
        },
    )
}

fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor.data().iter().map(|value| value.to_bits()).collect()
}

#[test]
fn tts_prefill_extension_matches_full_and_preserves_snapshot() -> Result<()> {
    let decoder = decoder(3)?;
    for voice in [0.1, -0.7, 1.3] {
        let prefix = Tensor::from_vec(vec![3, 2], vec![voice, 0.2, 0.3, -0.4, 0.9, 0.7])?;
        let (snapshot, prefix_state) = decoder.prefill_cache_from_embeddings(&prefix)?;
        let mut memo = TtsPrefillMemo::default();
        memo.insert(prefix.clone(), snapshot.clone(), prefix_state.clone());
        for first_text in [[0.7, -0.4], [-0.8, 0.9], [0.1, 0.6], [0.7, -0.4]] {
            let mut rows = prefix.data().to_vec();
            rows.extend(first_text);
            let input = Tensor::from_vec(vec![4, 2], rows)?;
            let (full, expected) = decoder.prefill_cache_from_embeddings(&input)?;
            let (mut cached, state) = memo.get(&prefix).expect("invariant: préfixe inséré");
            assert_eq!(bits(&state), bits(&prefix_state));
            let actual =
                decoder.extend_tts_prefill(&mut cached, &Tensor::row(first_text.to_vec())?)?;
            assert_eq!(bits(&actual), bits(&expected));
            assert_eq!(cached.position(), full.position());
            for layer in 0..3 {
                assert_eq!(
                    cached.layer_kv_for_test(layer),
                    full.layer_kv_for_test(layer)
                );
                assert_eq!(
                    memo.entries[0].cache.layer_kv_for_test(layer),
                    snapshot.layer_kv_for_test(layer)
                );
            }
            assert_eq!(memo.entries[0].cache.position(), 3);
        }
    }
    Ok(())
}

#[test]
fn tts_prefill_memo_keys_lru_and_model_lifetime() -> Result<()> {
    let decoder = decoder(3)?;
    let prefix = Tensor::row(vec![0.0, 0.2])?;
    let (cache, state) = decoder.prefill_cache_from_embeddings(&prefix)?;
    let mut memo = TtsPrefillMemo::default();
    memo.insert(prefix.clone(), cache.clone(), state.clone());
    let bytes = memo.bytes;
    memo.insert(prefix.clone(), cache.clone(), state.clone());
    assert_eq!(memo.bytes, bytes);
    assert!(memo.get(&Tensor::row(vec![-0.0, 0.2])?).is_none());
    assert!(memo
        .get(&Tensor::from_vec(vec![2, 1], prefix.data().to_vec())?)
        .is_none());
    for voice in 1..MAX_ENTRIES {
        memo.insert(
            Tensor::row(vec![voice as f32, 0.2])?,
            cache.clone(),
            state.clone(),
        );
    }
    assert!(memo.get(&prefix).is_some());
    memo.insert(Tensor::row(vec![9.0, 0.2])?, cache, state);
    assert_eq!(memo.entries.len(), MAX_ENTRIES);
    assert!(memo.get(&Tensor::row(vec![1.0, 0.2])?).is_none());
    assert!(memo.get(&prefix).is_some());
    assert_eq!(
        memo.bytes,
        memo.entries.iter().map(|entry| entry.bytes).sum::<usize>()
    );
    // Chaque chargement construit ce même Default ; aucune table globale ne
    // permet de réutiliser les états d'un modèle remplacé, même à chemin égal.
    assert!(TtsPrefillMemo::default().get(&prefix).is_none());
    Ok(())
}

#[test]
fn tts_prefill_memo_bounds_bytes_and_rejects_oversized_entry() -> Result<()> {
    let decoder = decoder(3)?;
    let prefix = Tensor::row(vec![0.0, 0.2])?;
    let (cache, _) = decoder.prefill_cache_from_embeddings(&prefix)?;
    let mut memo = TtsPrefillMemo::default();
    // Deux états dépassent ensemble le budget, avant la limite de quatre voix.
    let state = Tensor::row(vec![0.0; MAX_BYTES / size_of::<f32>() / 2])?;
    for voice in 0..2 {
        memo.insert(
            Tensor::row(vec![voice as f32, 0.2])?,
            cache.clone(),
            state.clone(),
        );
        assert!(memo.bytes <= MAX_BYTES);
    }
    assert_eq!(memo.entries.len(), 1);
    assert!(memo.get(&prefix).is_none());
    let before = memo.bytes;
    memo.insert(
        prefix,
        cache,
        Tensor::row(vec![0.0; MAX_BYTES / size_of::<f32>()])?,
    );
    assert_eq!(memo.bytes, before);
    assert_eq!(memo.entries.len(), 1);
    // Une petite longueur ne doit pas masquer la réserve réellement retenue.
    for reserve_prefix in [false, true] {
        let mut reserved = Vec::with_capacity(MAX_BYTES / size_of::<f32>());
        reserved.extend_from_slice(&[42.0, 0.2]);
        let small = Tensor::row(vec![42.0, 0.2])?;
        let large = Tensor::row(reserved)?;
        let cache = decoder.empty_cache();
        if reserve_prefix {
            memo.insert(large, cache, small);
        } else {
            memo.insert(small, cache, large);
        }
        assert_eq!(memo.bytes, before);
        assert_eq!(memo.entries.len(), 1);
    }
    Ok(())
}

#[test]
fn tts_prefill_flag_override_is_scoped() {
    use crate::runtime_flags::{tts_prefill_cache_enabled, with_tts_prefill_cache_for_test};
    with_tts_prefill_cache_for_test(false, || {
        assert!(!tts_prefill_cache_enabled());
        with_tts_prefill_cache_for_test(true, || assert!(tts_prefill_cache_enabled()));
        assert!(!tts_prefill_cache_enabled());
    });
}

#[test]
fn tts_prefill_extension_rejects_invalid_shape_or_empty_cache() -> Result<()> {
    let decoder = decoder(3)?;
    let row = Tensor::row(vec![0.1, 0.2])?;
    assert!(decoder
        .extend_tts_prefill(&mut decoder.empty_cache(), &row)
        .is_err());
    let (mut cache, _) = decoder.prefill_cache_from_embeddings(&row)?;
    assert!(decoder
        .extend_tts_prefill(&mut cache, &Tensor::row(vec![0.1])?)
        .is_err());
    assert_eq!(cache.position(), 1);
    Ok(())
}

#[test]
fn tts_prefill_memo_retains_compact_prefix_with_many_layers() -> Result<()> {
    for layers in [16, 24] {
        let decoder = decoder(layers)?;
        let prefix = Tensor::from_vec(vec![3, 2], vec![0.1, 0.2, 0.3, -0.4, 0.9, 0.7])?;
        let (original, state) = decoder.prefill_cache_from_embeddings(&prefix)?;
        // La réserve de decode dépasse à elle seule le budget du mémo.
        assert!(original.estimated_cpu_bytes() >= MAX_BYTES);
        let mut memo = TtsPrefillMemo::default();
        memo.insert(prefix.clone(), original.clone(), state.clone());
        assert_eq!(memo.entries.len(), 1, "préfixe refusé à {layers} couches");
        let retained = layers * 2 * prefix.len() * size_of::<f32>();
        assert_eq!(memo.entries[0].cache.estimated_cpu_bytes(), retained);
        assert_eq!(
            memo.bytes,
            retained + (prefix.len() + state.len()) * size_of::<f32>()
        );
        let (mut cached, _) = memo.get(&prefix).expect("invariant: préfixe conservé");
        let last = Tensor::row(vec![0.7, -0.4])?;
        let mut rows = prefix.data().to_vec();
        rows.extend_from_slice(last.data());
        let (full, expected) =
            decoder.prefill_cache_from_embeddings(&Tensor::from_vec(vec![4, 2], rows)?)?;
        let actual = decoder.extend_tts_prefill(&mut cached, &last)?;
        assert_eq!(bits(&actual), bits(&expected));
        for layer in 0..layers {
            assert_eq!(
                cached.layer_kv_for_test(layer),
                full.layer_kv_for_test(layer)
            );
            assert_eq!(
                memo.entries[0].cache.layer_kv_for_test(layer),
                original.layer_kv_for_test(layer)
            );
        }
        assert_eq!(memo.entries[0].cache.position(), 3);
        assert_eq!(memo.entries[0].cache.estimated_cpu_bytes(), retained);
    }
    Ok(())
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "Metal : modèle VoiceDesign local, égalité exacte multi-voix des états, K/V et codes"]
fn tts_prefill_memo_metal_codes_identical_multiple_voices() -> Result<()> {
    use crate::runtime_flags::with_tts_prefill_cache_for_test;
    let model_dir = std::env::var_os("SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR")
        .ok_or_else(|| InferError::Config("SARAGOSSA_QWEN3_TTS_VOICEDESIGN_DIR requis".into()))?;
    let mut model = TtsModel::load_local(&model_dir)?;
    assert!(model
        .prefill_memo
        .lock()
        .expect("invariant: mutex neuf")
        .entries
        .is_empty());
    for voice in [
        crate::tts::DEFAULT_INSTRUCT,
        "Une voix féminine française, calme et douce.",
        "Une voix masculine française, grave et énergique.",
        crate::tts::DEFAULT_INSTRUCT,
    ] {
        model = model.with_voice_design_instruction(voice);
        for text in [
            "Bonjour, voici un essai.",
            "Salut, comment allez-vous ?",
            "Bonsoir.",
        ] {
            let prepared = model.prepare_voicedesign_inputs(text)?;
            let (full_cache, full_state) = model
                .talker
                .prefill_cache_from_embeddings(&prepared.input)?;
            let (rows, hidden) = prepared.input.as_matrix()?;
            let prefix = Tensor::from_vec(
                vec![rows - 1, hidden],
                prepared.input.data()[..(rows - 1) * hidden].to_vec(),
            )?;
            let (mut direct_cache, _) = model.talker.prefill_cache_from_embeddings(&prefix)?;
            // Appel direct : le test doit échouer si l'extension échoue,
            // même si le repli du mémo pourrait masquer cette erreur.
            let direct_state = model.talker.extend_tts_prefill(
                &mut direct_cache,
                &Tensor::row(prepared.input.last_row()?.to_vec())?,
            )?;
            assert_eq!(bits(&direct_state), bits(&full_state));
            for _ in 0..2 {
                let (cached, state) = with_tts_prefill_cache_for_test(true, || {
                    model.prefill_talker(&prepared.input)
                })?;
                assert_eq!(
                    bits(&state),
                    bits(&full_state),
                    "voix={voice}, texte={text}"
                );
                assert_eq!(cached.position(), full_cache.position());
                for layer in 0..model.assets.model_config.talker_config.num_hidden_layers as usize {
                    assert_eq!(
                        cached.layer_kv_for_test(layer),
                        full_cache.layer_kv_for_test(layer)
                    );
                }
            }
            let full =
                with_tts_prefill_cache_for_test(false, || model.generate_codes_greedy(text, 24))?;
            FULL_PREFILLS.with(|count| count.set(0));
            MEMO_REUSES.with(|count| count.set(0));
            let cached =
                with_tts_prefill_cache_for_test(true, || model.generate_codes_greedy(text, 24))?;
            // L'égalité des codes seule laisserait passer un repli systématique
            // ou un appel direct au prefill complet depuis la génération.
            assert_eq!(FULL_PREFILLS.with(|count| count.get()), 0);
            assert_eq!(MEMO_REUSES.with(|count| count.get()), 1);
            assert!(!full.is_empty(), "oracle vide : voix={voice}, texte={text}");
            assert_eq!(cached, full, "voix={voice}, texte={text}");
        }
    }
    let memo = model.prefill_memo.lock().expect("invariant: mutex intact");
    assert_eq!(memo.entries.len(), 3);
    assert!(memo.bytes <= MAX_BYTES);
    drop(memo);
    drop(model);
    let reloaded = TtsModel::load_local(&model_dir)?;
    assert!(reloaded
        .prefill_memo
        .lock()
        .expect("invariant: mutex neuf")
        .entries
        .is_empty());
    Ok(())
}
