//! A/B synthétique des vrais draft/verifiers, y compris acceptation et rollback.
//! Ce contrôle unitaire ne qualifie pas les sidecars des checkpoints réels.
use super::*;
use crate::vision::prompt::{insert_embeddings, SuppliedImage};

#[test]
fn vision_decode_ar_mtp_dflash_identical_with_accept_reject_and_rollback() -> Result<()> {
    let mut model = crate::decoder::vision::tests::model()?;
    let layer = &model.layers[0];
    let AttentionBlock::Full(attention) = &layer.attention else {
        unreachable!()
    };
    let FeedForward::Dense(mlp) = layer.mlp.as_ref().unwrap() else {
        unreachable!()
    };
    let mlp = mlp.as_ref().clone();
    let norm = layer.input_norm.clone();
    let fc = Tensor::from_vec(
        vec![256, 512],
        (0..256 * 512)
            .map(|i| {
                if i % 512 == i / 512 || i % 512 == i / 512 + 256 {
                    0.5
                } else {
                    0.0
                }
            })
            .collect(),
    )?;
    model.mtp = Some(MtpHead {
        pre_fc_norm_embedding: norm.clone(),
        pre_fc_norm_hidden: norm.clone(),
        fc: Linear::new(fc, None)?,
        layer: MtpLayer {
            input_norm: norm.clone(),
            attention: attention.as_ref().clone(),
            post_attention_norm: norm.clone(),
            mlp: FeedForward::Dense(Box::new(mlp.clone())),
        },
        norm: norm.clone(),
    });
    let mut draft = crate::decoder::tests::tiny_dflash_draft();
    draft.info.hidden_size = 256;
    draft.info.intermediate_size = 256;
    draft.info.head_dim = 256;
    draft.info.rope_theta = 10_000_000.0;
    draft.info.block_size = 3;
    draft.hidden_norm = norm.clone();
    draft.norm = norm.clone();
    draft.fc = Linear::new(
        Tensor::from_vec(
            vec![256, 256],
            (0..256 * 256)
                .map(|i| if i % 256 == i / 256 { 1.0 } else { 0.0 })
                .collect(),
        )?,
        None,
    )?;
    draft.layers = vec![crate::DFlashDraftLayer {
        input_norm: norm.clone(),
        post_attention_norm: norm,
        attention: crate::DFlashAttentionWeights {
            q_proj: attention.q_proj.clone(),
            k_proj: attention.k_proj.clone(),
            v_proj: attention.v_proj.clone().unwrap(),
            o_proj: attention.o_proj.clone(),
            q_norm: attention.q_norm.clone().unwrap(),
            k_norm: attention.k_norm.clone().unwrap(),
        },
        mlp,
        attention_conv: None,
        mlp_conv: None,
    }];
    let tokens = [0, 7, 7, 7, 7, 1];
    let embeddings = Tensor::from_vec(
        vec![4, 256],
        (0..1024).map(|i| (i as f32 * 0.03).cos()).collect(),
    )?;
    let images = [SuppliedImage {
        start: 1,
        grid: [1, 4, 4],
        embeddings: &embeddings,
        content_hash: [1; 32],
        parameters: b"test",
    }];
    let (initial, state) = model.prefill_supplied_vision_state(&tokens, 7, &images, 100, false)?;
    let (input, positions) = insert_embeddings(&tokens, 7, &model.embed_scaled(&tokens)?, &images)?;
    let mut capture_cache = model.empty_cache();
    let captured = model.layers[0].forward_prefill(
        &model.config,
        &input,
        &mut capture_cache.layers[0],
        0,
        model.forward_runtime().with_rope_positions(&positions.rows),
    )?;
    let context = draft.project_target_hidden(&captured, model.forward_runtime())?;
    assert_ab(&model, &draft, || {
        Ok((initial.clone(), state.clone(), context.clone()))
    })
}

fn assert_ab(
    model: &CausalDecoder,
    draft: &DFlashDraft,
    make: impl Fn() -> Result<(CausalDecoderCache, Tensor, Tensor)>,
) -> Result<()> {
    let (mut ar_cache, mut ar_state, _) = make()?;
    let options = GenerationOptions::default();
    let mut ar = Vec::new();
    for _ in 0..16 {
        let token = crate::argmax(model.logits_from_final_state(&ar_state)?.data())?;
        ar.push(token);
        ar_state = model.next_final_state_cached(&mut ar_cache, token)?;
    }
    for dflash in [false, true] {
        for forced_branches in [false, true] {
            let (mut cache, mut final_state, mut projected_context) = make()?;
            let delta = cache.rope_delta;
            let mut sampler = DeterministicSampler::new(0);
            let mut pending =
                model.sample_token_from_state(&final_state, &options, &mut sampler)?;
            let mut generated = Vec::new();
            let mut stats = SpeculativeStats {
                proposed_by_position: vec![0; 2],
                accepted_by_position: vec![0; 2],
                ..Default::default()
            };
            let mut cycle = 0;
            while generated.len() < 16 {
                let primary = pending;
                generated.push(primary);
                if generated.len() == 16 {
                    break;
                }
                let mut drafts = if dflash {
                    model.dflash_draft_greedy_tokens(
                        draft,
                        &projected_context,
                        primary,
                        2,
                        &options,
                        &mut sampler,
                    )?
                } else {
                    let mut local = LayerKvCache::default();
                    let (first, hidden) = model.mtp_forward_greedy_with_hidden(
                        &final_state,
                        primary,
                        cache.current_pos()?,
                        &mut local,
                        &options,
                        &mut sampler,
                    )?;
                    let (second, _) = model.mtp_forward_greedy_with_hidden(
                        &hidden,
                        first,
                        cache.current_pos()? + 1,
                        &mut local,
                        &options,
                        &mut sampler,
                    )?;
                    vec![first, second]
                };
                if forced_branches {
                    for (offset, token) in drafts.iter_mut().enumerate() {
                        *token = ar.get(generated.len() + offset).copied().unwrap_or(0);
                    }
                    let rejected_position = match cycle % 3 {
                        0 => Some(0),
                        1 => None,
                        _ => Some(1),
                    };
                    if let Some(position) = rejected_position {
                        drafts[position] =
                            (drafts[position] + 1) % model.lm_head.weight().shape()[0];
                    }
                }

                let verified = if dflash {
                    model.verify_dflash_drafts_batched(DFlashVerify {
                        draft,
                        cache: &mut cache,
                        projected_context: &mut projected_context,
                        final_state: &mut final_state,
                        pending: &mut pending,
                        generated: &mut generated,
                        drafts,
                        primary,
                        max_new_tokens: 16,
                        options: &options,
                        sampler: &mut sampler,
                        stats: &mut stats,
                    })?
                } else {
                    model.verify_mtp_drafts_batched(MtpVerify {
                        cache: &mut cache,
                        final_state: &mut final_state,
                        pending: &mut pending,
                        generated: &mut generated,
                        drafts,
                        draft_distributions: vec![None, None],
                        primary,
                        max_new_tokens: 16,
                        options: &options,
                        sampler: &mut sampler,
                        stats: &mut stats,
                    })?
                };
                if verified.accepted_all && generated.len() < 16 {
                    let (state, hidden) = model.next_final_state_with_layer_capture(
                        &mut cache,
                        verified.verify_token,
                        &draft.info.target_layer_ids,
                    )?;
                    final_state = state;
                    pending =
                        model.sample_token_from_state(&final_state, &options, &mut sampler)?;
                    model.append_dflash_projected_context(
                        draft,
                        &mut projected_context,
                        &hidden,
                    )?;
                }
                assert_eq!(cache.rope_delta, delta);
                assert_eq!(cache.current_pos()? as i64, cache.position as i64 + delta);
                cycle += 1;
            }
            assert_eq!(generated, ar, "dflash={dflash} forced={forced_branches}");
            assert!(stats.proposed > 0);
            if forced_branches {
                assert!(stats.accepted > 0 && stats.rejected > 0);
                assert!(stats.proposed_by_position[1] > stats.accepted_by_position[1]);
            }
        }
    }
    Ok(())
}

#[cfg(gpu_metal)]
#[test]
#[ignore = "A/B réel : poids Qwen3.6, sidecars MTP/DFlash, fixtures et verrou GPU"]
fn vision_decode_ab_real_sidecars() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    let Some(model_dir) = crate::test_support::local_dir("SARAGOSSA_QWEN36_MOE_DIR") else {
        return Ok(());
    };
    let assets = crate::ModelAssets::load_local(model_dir)?;
    let executor = crate::MetalExecutor::new()?;
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)?
        .with_metal_executor(executor)?
        .with_mtp_weights(&assets.mtp)?
        .with_dflash_draft_from_env(&assets.config)?;
    let draft = model.dflash_draft().ok_or("sidecar DFlash absent")?;
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision/Qwen3.6-35B-A3B-4bit");
    let inputs: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("mrope-inputs.json"))?)?;
    // Un prompt image simple et un multi-image, mêmes fixtures que l’oracle.
    for index in [0, 3] {
        let case = &inputs["cases"][index];
        let tokens: Vec<usize> = serde_json::from_value(case["input_ids"].clone())?;
        let grids: Vec<[usize; 3]> = serde_json::from_value(case["image_grid_thw"].clone())?;
        let spans = case["image_spans"].as_array().ok_or("spans absents")?;
        let embeddings = spans
            .iter()
            .map(
                |span| -> std::result::Result<Tensor, Box<dyn std::error::Error>> {
                    let bytes = std::fs::read(
                        root.join(span["embedding_file"].as_str().ok_or("fichier absent")?),
                    )?;
                    let values: Vec<_> = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                        .collect();
                    let rows = span["length"].as_u64().ok_or("lignes absentes")? as usize;
                    Ok(Tensor::from_vec(vec![rows, values.len() / rows], values)?)
                },
            )
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let images: Vec<_> = spans
            .iter()
            .zip(&grids)
            .zip(&embeddings)
            .map(|((span, &grid), embedding)| SuppliedImage {
                start: span["start"].as_u64().unwrap() as usize,
                grid,
                embeddings: embedding,
                content_hash: [0; 32],
                parameters: b"versioned-fixture",
            })
            .collect();
        let (input, positions) = insert_embeddings(
            &tokens,
            tokens[images[0].start],
            &model.embed_scaled(&tokens)?,
            &images,
        )?;
        let delta = positions.delta_at(tokens.len())?;
        eprintln!(
            "vision A/B réel case={index} tokens={} delta={delta}",
            tokens.len()
        );
        assert_ab(&model, draft, || {
            let (mut cache, hidden, captures) = model
                .prefill_resident_cache_hidden_captures_and_mtp(
                    &tokens,
                    &draft.info.target_layer_ids,
                    usize::MAX,
                    true,
                    37,
                    None,
                    Some((&input, &positions.rows)),
                )?
                .ok_or_else(|| {
                    InferError::Config("prefill résident requis pour A/B vision".into())
                })?;
            cache.rope_delta = delta;
            cache.vision_positions = Some(positions.clone());
            let state = rms_norm(
                &hidden.into_tensor(model.final_norm.len())?,
                &model.final_norm,
                model.config.rms_eps,
            )?;
            let captures =
                captures.ok_or_else(|| InferError::Config("captures DFlash absentes".into()))?;
            let context = draft.project_target_hidden(&captures.hidden, model.forward_runtime())?;
            Ok((cache, state, context))
        })?;
    }
    Ok(())
}
