use super::*;
use std::path::PathBuf;

#[test]
fn extra_padding_matches_complete_frames() {
    assert_eq!(extra_padding(10, 4, 2, 2), 0);
    assert_eq!(extra_padding(11, 4, 2, 2), 1);
}

#[test]
fn edge_pad_repeats_border() -> Result<()> {
    let x = Nlc::new(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
    let out = pad_time(&x, 1, 2, PadMode::Edge)?;
    assert_eq!(
        out.data,
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );
    Ok(())
}

/// Distance euclidienne au carré "naïve", écrite indépendamment de
/// [`nearest_code`]/[`codebook_half_norms`] (pas de terme précalculé, pas
/// d'astuce de produit scalaire) pour servir d'oracle de comparaison.
fn naive_nearest_by_squared_distance(
    row: &[f32],
    embed: &Tensor,
    bins: usize,
    dim: usize,
) -> usize {
    let mut best = 0_usize;
    let mut best_dist = f32::INFINITY;
    for bin in 0..bins {
        let code = &embed.data()[bin * dim..(bin + 1) * dim];
        let dist: f32 = row
            .iter()
            .zip(code.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if dist < best_dist {
            best = bin;
            best_dist = dist;
        }
    }
    best
}

#[test]
fn nearest_code_matches_naive_argmin_search() -> Result<()> {
    // Codebook synthétique déterministe (4 bins, dim 3).
    let embed = Tensor::from_vec(
        vec![4, 3],
        vec![
            1.0, 2.0, 3.0, // bin 0
            0.0, 0.0, 0.0, // bin 1
            5.0, 5.0, 5.0, // bin 2
            1.5, 2.5, 2.5, // bin 3 (le plus proche de la requête)
        ],
    )?;
    let (bins, dim) = matrix_shape(&embed)?;
    let c2 = codebook_half_norms(&embed, bins, dim);
    let row = [1.0_f32, 2.0, 2.0];

    let via_half_norm_trick = nearest_code(&row, &embed, bins, dim, &c2);
    let via_naive_search = naive_nearest_by_squared_distance(&row, &embed, bins, dim);

    assert_eq!(via_half_norm_trick, via_naive_search);
    assert_eq!(via_half_norm_trick, 3);
    Ok(())
}

#[test]
fn nearest_code_exact_match_gives_zero_residual() -> Result<()> {
    let embed = Tensor::from_vec(
        vec![3, 2],
        vec![
            0.0, 0.0, // bin 0
            3.0, 4.0, // bin 1
            1.0, 0.0, // bin 2 == la requête
        ],
    )?;
    let (bins, dim) = matrix_shape(&embed)?;
    let c2 = codebook_half_norms(&embed, bins, dim);
    let row = [1.0_f32, 0.0];

    let idx = nearest_code(&row, &embed, bins, dim, &c2);
    assert_eq!(idx, 2);

    let code = embed.row_slice(idx)?;
    let mut residual = row;
    subtract_residual_row(&mut residual, code);
    assert_eq!(residual, [0.0, 0.0]);
    Ok(())
}

#[test]
fn nearest_code_tie_breaks_to_lowest_index() {
    // Deux bins strictement équidistants de la requête (dist=0.5 chacun) :
    // le code documente la conservation du premier trouvé (comparaison `<`).
    let embed = Tensor::from_vec(vec![2, 1], vec![-1.0, 1.0]).expect("shape valide");
    let (bins, dim) = matrix_shape(&embed).expect("rang 2");
    let c2 = codebook_half_norms(&embed, bins, dim);
    let row = [0.0_f32];

    assert_eq!(nearest_code(&row, &embed, bins, dim, &c2), 0);
}

#[test]
fn subtract_residual_row_computes_entry_minus_code() {
    let mut residual = [3.0_f32, -1.0, 0.5];
    let code = [1.0_f32, 1.0, 0.5];
    subtract_residual_row(&mut residual, &code);
    assert_eq!(residual, [2.0, -2.0, 0.0]);
}

#[test]
fn rvq_two_stage_encode_reduces_residual_error_each_stage() -> Result<()> {
    // Reproduit la boucle de `rvq_encode_one` sur 2 étages avec des codebooks
    // synthétiques : chaque étage doit réduire la norme du résidu.
    let stage1 = Tensor::from_vec(
        vec![2, 3],
        vec![
            0.0, 0.0, 0.0, // bin 0
            1.5, 2.5, 2.5, // bin 1 (proche de la cible)
        ],
    )?;
    let stage2 = Tensor::from_vec(
        vec![2, 3],
        vec![
            0.0, 0.0, 0.0, // bin 0
            -0.5, -0.5, -0.4, // bin 1 (proche du résidu de l'étage 1)
        ],
    )?;

    let target = [1.0_f32, 2.0, 2.1];
    let residual0_norm = l2_norm(&target);

    let (bins1, dim1) = matrix_shape(&stage1)?;
    let c2_1 = codebook_half_norms(&stage1, bins1, dim1);
    let idx1 = nearest_code(&target, &stage1, bins1, dim1, &c2_1);
    let mut residual1 = target;
    subtract_residual_row(&mut residual1, stage1.row_slice(idx1)?);
    let residual1_norm = l2_norm(&residual1);

    let (bins2, dim2) = matrix_shape(&stage2)?;
    let c2_2 = codebook_half_norms(&stage2, bins2, dim2);
    let idx2 = nearest_code(&residual1, &stage2, bins2, dim2, &c2_2);
    let mut residual2 = residual1;
    subtract_residual_row(&mut residual2, stage2.row_slice(idx2)?);
    let residual2_norm = l2_norm(&residual2);

    assert_eq!(idx1, 1);
    assert_eq!(idx2, 1);
    assert!(residual1_norm < residual0_norm);
    assert!(residual2_norm < residual1_norm);
    Ok(())
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Codes Mimi metal-rs ≡ golden mlx-rs figé (sans mlx-rs ; charge Qwen3-TTS Base
/// pour l'encodeur metal-rs). Même tolérance `mismatch_ratio<=0.015` que `live_mimi_*`.
#[test]
#[ignore = "golden: charge Qwen3-TTS Base (cache HF) pour l'encodeur Mimi metal-rs"]
fn golden_mimi_ref_codes_matches_fixture() -> Result<()> {
    const MISMATCH_RATIO_TOLERANCE: f32 = 0.015;

    let Some(model_dir) = local_tts_snapshot(
        "RETI_QWEN3_TTS_BASE_DIR",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-6bit",
    ) else {
        eprintln!("skip: snapshot Qwen3-TTS Base absent du cache HF");
        return Ok(());
    };
    let assets = crate::tts::TtsAssets::load_local(&model_dir)?;
    let encoder = TtsMimiEncoder::load(&model_dir, &assets.codec_config)?;

    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../voices/reti-fr.wav");
    let bytes = std::fs::read(&wav).map_err(|source| InferError::Io {
        path: wav.clone(),
        source,
    })?;
    let pcm = crate::tts_clone::load_wav_24k(&bytes)?;
    let rust_codes = encoder.encode_pcm_24k(&pcm)?;

    let (shape, golden) = crate::golden::read_i32("mimi_ref_codes")?;
    let quantizers = shape[0];
    let frames = shape[1];
    if rust_codes.len() != frames {
        return Err(InferError::Dimension(format!(
            "frames Mimi rust={} golden={frames}",
            rust_codes.len()
        )));
    }
    let mut mismatches = 0_usize;
    for time in 0..frames {
        for q in 0..quantizers {
            let rust = *rust_codes
                .get(time)
                .and_then(|frame| frame.get(q))
                .ok_or_else(|| {
                    InferError::Dimension(format!("code Mimi rust manquant t={time} q={q}"))
                })?;
            let golden_code = golden[q * frames + time];
            if rust != golden_code {
                mismatches += 1;
            }
        }
    }
    let total = frames * quantizers;
    let mismatch_ratio = mismatches as f32 / total as f32;
    assert!(
        mismatch_ratio <= MISMATCH_RATIO_TOLERANCE,
        "drift Mimi ref_codes mismatch_ratio={mismatch_ratio} > {MISMATCH_RATIO_TOLERANCE}"
    );
    Ok(())
}

fn local_tts_snapshot(env_var: &str, cache_name: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(env_var) {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            return Some(path);
        }
    }
    let snapshot = crate::hf_resolve::hf_cache_dir_from_env().and_then(|hub| {
        let snapshots = hub.join(cache_name).join("snapshots");
        let mut entries = std::fs::read_dir(snapshots)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        entries.sort();
        entries.pop()
    });
    crate::test_support::require_real_model(snapshot, "snapshot Qwen3-TTS Base")
}
