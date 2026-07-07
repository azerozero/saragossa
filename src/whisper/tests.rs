use super::*;
use std::path::PathBuf;

#[test]
fn conv1d_mel_matches_manual_padding_stride() {
    let mel =
        Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("invariant: mel valide");
    let weight =
        Tensor::from_vec(vec![1, 1, 3], vec![1.0, 10.0, 100.0]).expect("invariant: poids valide");
    let bias = Tensor::from_vec(vec![1], vec![0.5]).expect("invariant: biais valide");
    let out = conv1d_mel_hf(&mel, &weight, &bias, 2, 1).expect("invariant: conv valide");
    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(out.data(), &[210.5, 432.5]);
}

#[test]
fn non_causal_attention_identity_values() {
    let q = Tensor::from_vec(vec![2, 2], vec![10.0, 0.0, 0.0, 10.0]).expect("invariant: q valide");
    let k = q.clone();
    let v = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 4.0, 8.0]).expect("invariant: v valide");
    let out =
        multi_head_attention(&q, &k, &v, 1, false, ForwardRuntime::cpu()).expect("invariant: attn");
    assert!((out.data()[0] - 1.0).abs() < 1.0e-5);
    assert!((out.data()[1] - 2.0).abs() < 1.0e-5);
    assert!((out.data()[2] - 4.0).abs() < 1.0e-5);
    assert!((out.data()[3] - 8.0).abs() < 1.0e-5);
}

#[test]
fn causal_attention_masks_future_values() {
    let q = Tensor::from_vec(vec![2, 1], vec![1.0, 1.0]).expect("invariant: q valide");
    let k = Tensor::from_vec(vec![2, 1], vec![1.0, 1.0]).expect("invariant: k valide");
    let v = Tensor::from_vec(vec![2, 1], vec![3.0, 30.0]).expect("invariant: v valide");
    let out = multi_head_attention(&q, &k, &v, 1, true, ForwardRuntime::cpu())
        .expect("invariant: attn causale");
    assert!((out.data()[0] - 3.0).abs() < 1.0e-5);
    assert!(out.data()[1] > 3.0);
}

#[test]
fn cross_attention_accepts_distinct_lengths() {
    let q = Tensor::from_vec(vec![1, 2], vec![1.0, 0.0]).expect("invariant: q valide");
    let k = Tensor::from_vec(vec![2, 2], vec![4.0, 0.0, 0.0, 4.0]).expect("invariant: k valide");
    let v = Tensor::from_vec(vec![2, 2], vec![10.0, 0.0, 0.0, 20.0]).expect("invariant: v valide");
    let out = multi_head_attention(&q, &k, &v, 1, false, ForwardRuntime::cpu())
        .expect("invariant: cross-attn");
    assert_eq!(out.shape(), &[1, 2]);
    assert!(out.data()[0] > out.data()[1]);
}

#[test]
fn rejects_invalid_config_heads() {
    let cfg = WhisperConfig {
        d_model: 5,
        encoder_attention_heads: 2,
        encoder_layers: 1,
        decoder_attention_heads: 1,
        decoder_layers: 1,
        num_mel_bins: 80,
        max_target_positions: 16,
        vocab_size: 8,
    };
    assert!(cfg.validate().is_err());
}

/// Frontend mel metal-rs ≡ golden mlx-rs figé (sans mlx-rs, sans modèle : DSP pur
/// sur l'entrée PCM16k golden). Mêmes tolérances que `live_log_mel_*`.
#[test]
fn golden_log_mel_matches_fixture() -> Result<()> {
    let (_, samples) = crate::golden::read_f32("whisper_samples16k")?;
    let (mel_shape, ref_mel) = crate::golden::read_f32("whisper_mel")?;
    let n_mels = mel_shape[0];
    let frames = mel_shape[1];
    let mel = log_mel_spectrogram(&samples, n_mels)?;
    assert_eq!(mel.shape(), &[n_mels, frames]);
    let (max_abs, mean_abs) = drift(mel.data(), &ref_mel);
    assert!(
        mean_abs < 1.0e-6,
        "drift mel moyen trop élevé: {mean_abs:.6e}"
    );
    assert!(max_abs < 1.0e-5, "drift mel max trop élevé: {max_abs:.6e}");
    Ok(())
}

/// Encodeur Whisper metal-rs ≡ golden mlx-rs figé (sans mlx-rs ; charge le snapshot
/// whisper-tiny local pour l'inférence metal-rs). Mêmes tolérances que `live_encoder_*`.
#[test]
#[ignore = "golden: charge whisper-tiny (cache HF) pour l'inférence Metal metal-rs"]
fn golden_encoder_matches_fixture() -> Result<()> {
    let Some(model_dir) = local_whisper_tiny_dir() else {
        eprintln!("skip: snapshot openai/whisper-tiny absent du cache HF");
        return Ok(());
    };
    let (in_shape, mel_prefix) = crate::golden::read_f32("whisper_encoder_input_mel")?;
    let n_mels = in_shape[0];
    let keep_frames = in_shape[1];
    let (out_shape, ref_data) = crate::golden::read_f32("whisper_encoder_out")?;
    let ref_seq = out_shape[0];
    let ref_dim = out_shape[1];

    let encoder = WhisperEncoder::from_model_dir(model_dir)?;
    let mel_tensor = Tensor::from_vec(vec![n_mels, keep_frames], mel_prefix)?;
    let got = encoder.encode_mel(&mel_tensor, ForwardRuntime::cpu())?;
    assert_eq!(got.shape(), &[ref_seq, ref_dim]);
    let (max_abs, mean_abs) = drift(got.data(), &ref_data);

    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Ok(metal) = crate::MetalExecutor::new() {
        let got_metal = encoder.encode_mel(&mel_tensor, ForwardRuntime::metal(&metal))?;
        assert_eq!(got_metal.shape(), &[ref_seq, ref_dim]);
        let (metal_max_abs, metal_mean_abs) = drift(got_metal.data(), &ref_data);
        assert!(
            metal_mean_abs < 2.0e-2,
            "drift moyen Metal trop élevé: {metal_mean_abs:.6e}"
        );
        assert!(
            metal_max_abs < 3.0e-1,
            "drift max Metal trop élevé: {metal_max_abs:.6e}"
        );
    }
    assert!(mean_abs < 1.0e-2, "drift moyen trop élevé: {mean_abs:.6e}");
    assert!(max_abs < 2.0e-1, "drift max trop élevé: {max_abs:.6e}");
    Ok(())
}

fn local_whisper_tiny_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("RETI_WHISPER_TINY_DIR") {
        let path = PathBuf::from(path);
        if path.join("config.json").is_file() && path.join("model.safetensors").is_file() {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--openai--whisper-tiny/snapshots");
    let entries = std::fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.join("config.json").is_file() && path.join("model.safetensors").is_file() {
            return Some(path);
        }
    }
    None
}

fn drift(left: &[f32], right: &[f32]) -> (f32, f32) {
    assert_eq!(left.len(), right.len());
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (left, right) in left.iter().zip(right.iter()) {
        let diff = (left - right).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    (max_abs, sum_abs / left.len().max(1) as f32)
}
