use super::*;

pub(in crate::decoder) fn model() -> Result<CausalDecoder> {
    let mut weights = HashMap::new();
    let width = 256;
    for (name, shape) in [
        ("embed_tokens.weight", vec![8, width]),
        ("lm_head.weight", vec![8, width]),
        ("norm.weight", vec![width]),
        ("layers.0.input_layernorm.weight", vec![width]),
        ("layers.0.post_attention_layernorm.weight", vec![width]),
        ("layers.0.self_attn.q_norm.weight", vec![width]),
        ("layers.0.self_attn.k_norm.weight", vec![width]),
        ("layers.0.self_attn.q_proj.weight", vec![width, width]),
        ("layers.0.self_attn.k_proj.weight", vec![width, width]),
        ("layers.0.self_attn.v_proj.weight", vec![width, width]),
        ("layers.0.self_attn.o_proj.weight", vec![width, width]),
        ("layers.0.mlp.gate_proj.weight", vec![width, width]),
        ("layers.0.mlp.up_proj.weight", vec![width, width]),
        ("layers.0.mlp.down_proj.weight", vec![width, width]),
    ] {
        let len = shape.iter().product();
        let data = (0..len)
            .map(|i| {
                if shape.len() == 1 {
                    1.0
                } else {
                    ((i * 13 + 7) as f32).sin() * 0.04
                }
            })
            .collect();
        weights.insert(name.into(), Tensor::from_vec(shape, data)?);
    }
    CausalDecoder::from_tensors(
        weights,
        CausalDecoderConfig {
            rope_theta: Some(10_000_000.0),
            head_dim: Some(256),
            rope_dims: Some(64),
            ..Default::default()
        },
    )
}

#[test]
fn vision_cpu_chunking_decode_and_text_identity() -> Result<()> {
    let model = model()?;
    let tokens = [0, 7, 7, 7, 7, 1];
    let vision = Tensor::from_vec(
        vec![4, 256],
        (0..1024).map(|i| (i as f32 * 0.03).cos()).collect(),
    )?;
    let image = || SuppliedImage {
        start: 1,
        grid: [1, 4, 4],
        embeddings: &vision,
        content_hash: [1; 32],
        parameters: b"test",
    };
    let (mut full, logits) = model.prefill_supplied_vision(&tokens, 7, &[image()], 100, false)?;
    assert_eq!(full.position, 6);
    assert_eq!(full.rope_delta, -2);
    assert_eq!(full.current_pos()?, 4);
    let (mut chunked, other) = model.prefill_supplied_vision(&tokens, 7, &[image()], 2, false)?;
    crate::oracle::assert_close_abs(logits.data(), other.data(), 1e-5);
    for token in [2, 3, 4, 5, 6, 1, 2, 0, 4, 3, 1, 5, 6, 1, 0, 2] {
        let left = model.next_logits_cached(&mut full, token)?;
        let right = model.next_logits_cached(&mut chunked, token)?;
        crate::oracle::assert_close_abs(left.data(), right.data(), 1e-5);
        assert_eq!(
            full.current_pos()? as i64,
            full.position as i64 + full.rope_delta
        );
    }
    let mut reused = full.clone();
    model.reset_resident_decode_cache(&mut reused)?;
    assert_eq!(reused.current_pos()?, 0);
    assert!(reused.vision_positions.is_none());
    let mut text = model.empty_cache();
    for &token in &[0, 1, 2, 3] {
        model.next_logits_cached(&mut text, token)?;
        assert_eq!(text.current_pos()?, text.position());
    }
    // Texte seul : la sélection des axes ne change aucun bit de la rotation.
    let input = Tensor::from_vec(
        vec![3, 256],
        (0..768).map(|i| i as f32 / 200.0 - 1.0).collect(),
    )?;
    let positioned = crate::rope::apply_mrope(&input, 1, 256, &[[5; 3], [6; 3], [7; 3]])?;
    let scalar = super::super::attention_ops::apply_rope_heads_at(
        &input,
        1,
        256,
        64,
        super::super::attention_ops::RopeParams {
            theta: 10_000_000.0,
            frequency_dim: 64,
            position_scale: 1.0,
            llama3: None,
        },
        5,
        RopeStyle::Halves,
    )?;
    assert_eq!(positioned.data(), scalar.data());
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "oracle Metal sous verrou GPU"]
fn vision_resident_prefill_and_decode_match_cpu() -> Result<()> {
    let cpu = model()?;
    let gpu = model()?.with_metal_runtime()?;
    let tokens = [0, 7, 7, 7, 7, 1];
    let vision = Tensor::from_vec(
        vec![4, 256],
        (0..1024).map(|i| (i as f32 * 0.03).cos()).collect(),
    )?;
    let image = || SuppliedImage {
        start: 1,
        grid: [1, 4, 4],
        embeddings: &vision,
        content_hash: [1; 32],
        parameters: b"test",
    };
    let (mut cpu_cache, expected) =
        cpu.prefill_supplied_vision(&tokens, 7, &[image()], 100, false)?;
    let (_, per_op) = gpu.prefill_supplied_vision(&tokens, 7, &[image()], 100, false)?;
    let (_, text_cpu) = cpu.prefill_cache(&tokens)?;
    let (_, text_hidden, _) = gpu
        .prefill_resident_cache_hidden_captures_and_mtp(
            &tokens,
            &[],
            usize::MAX,
            true,
            100,
            None,
            None,
        )?
        .expect("resident text control");
    let text_gpu = gpu.logits_from_final_state(&rms_norm(
        &text_hidden.into_tensor(256)?,
        &gpu.final_norm,
        gpu.config.rms_eps,
    )?)?;
    for (label, left, right) in [
        ("text resident/CPU", text_gpu.data(), text_cpu.data()),
        ("vision per-op/CPU", per_op.data(), expected.data()),
    ] {
        eprintln!(
            "{label}: max_abs={}",
            left.iter()
                .zip(right)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }
    for chunk in [100, 2] {
        let (mut gpu_cache, actual) =
            gpu.prefill_supplied_vision(&tokens, 7, &[image()], chunk, true)?;
        eprintln!("vision resident prefill chunk={chunk}");
        crate::oracle::assert_close_abs(actual.data(), expected.data(), 2e-4);
        assert!(gpu.setup_resident_full_decode(&mut gpu_cache, 16, 0, false)?);
        let mut reference = cpu_cache.clone();
        for token in [2, 3, 4, 5, 6, 1, 2, 0, 4, 3, 1, 5, 6, 1, 0, 2] {
            let expected = cpu.next_logits_cached(&mut reference, token)?;
            let actual = gpu.resident_reference_step(&mut gpu_cache, token)?;
            eprintln!(
                "vision resident decode chunk={chunk} token={token} position={}",
                gpu_cache.position
            );
            crate::oracle::assert_close_abs(&actual, expected.data(), 2e-4);
            assert_eq!(gpu_cache.position, reference.position);
            assert_eq!(gpu_cache.current_pos()?, reference.current_pos()?);
        }
    }
    // La copie garde le delta indépendamment de la session GPU.
    cpu_cache.rope_delta = i64::MIN;
    assert!(cpu_cache.current_pos().is_err());
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn vision_mtp_resident_cycles_fall_back_before_mutating_cache() -> Result<()> {
    use crate::decoder::MtpFinalState;
    use crate::vision::prompt::{MediaSpan, VisionPositions};

    let Some(_) = crate::test_support::metal_executor()? else {
        return Ok(());
    };
    let mut model = model()?.with_metal_runtime()?;
    // Le chemin texte atteint cette validation ; le repli vision doit la
    // précéder, même à une frontière textuelle dont le delta est encore nul.
    model.config.head_dim = None;
    let state = MtpFinalState::Cpu(Tensor::row(vec![0.0; 256])?);
    for depth in [1, 2] {
        for vision in [false, true] {
            for position in [1, 5] {
                let mut cache = model.empty_cache();
                cache.position = position;
                cache.rope_delta = if position == 1 { 0 } else { -2 };
                if vision {
                    cache.vision_positions = Some(VisionPositions {
                        rows: vec![[0; 3], [1; 3], [1, 1, 2], [1, 2, 1], [1, 2, 2]],
                        media: vec![MediaSpan {
                            start: 1,
                            end: 5,
                            hash: [42; 32],
                        }],
                    });
                }
                let mut submitted = false;
                let mut after_submit = || {
                    submitted = true;
                    true
                };
                let result = if depth == 1 {
                    model
                        .next_mtp_spec_one_resident(
                            &mut cache,
                            &state,
                            0,
                            0,
                            &[],
                            true,
                            Some(&mut after_submit),
                        )
                        .map(|output| output.is_none())
                } else {
                    model
                        .next_mtp_spec_two_resident(
                            &mut cache,
                            &state,
                            0,
                            0,
                            &[],
                            true,
                            Some(&mut after_submit),
                        )
                        .map(|output| output.is_none())
                };
                if vision {
                    assert!(result?, "repli vision D{depth}");
                } else {
                    assert!(
                        matches!(result, Err(InferError::Dimension(message)) if message.contains("head_dim"))
                    );
                }
                assert!(!submitted);
                assert_eq!(cache.position, position);
                assert_eq!(cache.rope_delta, if position == 1 { 0 } else { -2 });
                assert_eq!(cache.vision_positions.is_some(), vision);
                assert!(cache.resident.is_none());
            }
        }
    }
    Ok(())
}
