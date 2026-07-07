//! Harnais et chargeurs de diagnostic hors moteur nominal.

pub mod dflash;

use std::env;
use std::time::{Duration, Instant};

pub use dflash::{
    load_dflash_draft_for_target, load_dflash_draft_weights_for_target, DFlashAttentionWeights,
    DFlashDraft, DFlashDraftInfo, DFlashDraftLayer,
};

use crate::runtime_flags::env_flag;
use crate::{CausalDecoder, GenerationOptions, InferError, ModelAssets, Result};

pub fn mtp_acceptance_enabled() -> bool {
    env_flag("RETI_RUST_MTP_ACCEPTANCE", false)
}

pub fn dflash_acceptance_enabled() -> bool {
    env_flag("RETI_RUST_DFLASH_ACCEPTANCE", false)
}

pub fn lightbatch_acceptance_enabled() -> bool {
    env_flag("RETI_RUST_LIGHTBATCH_ACCEPTANCE", false)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn resident_linear_xray_enabled() -> bool {
    env_flag("RETI_RUST_RESIDENT_LINEAR_XRAY", false)
}

/// Harnais d'acceptance light-batch (E2.1) : oracle de byte-identite par flux.
///
/// Decode chaque prompt SEUL (reference, API prod) puis les deux en light-batch
/// M=2 ; chaque flux du batch doit produire EXACTEMENT les tokens de son solo.
/// Les debits sont indicatifs (GPU potentiellement partage) - aucune conclusion
/// de perf sans GPU idle.
pub fn run_lightbatch_acceptance(
    assets: &ModelAssets,
    decoder: &CausalDecoder,
    prompt_a: &[usize],
    prompt_b: &[usize],
    max_tokens: usize,
    options: &GenerationOptions,
    load_elapsed: Duration,
    warmup_elapsed: Duration,
) -> Result<()> {
    let solo_a = decoder.generate_greedy_timed_with_options(prompt_a, max_tokens, options)?;
    let solo_b = decoder.generate_greedy_timed_with_options(prompt_b, max_tokens, options)?;
    let batch_started = Instant::now();
    let batch = decoder.generate_greedy_lightbatch_with_options(
        &[prompt_a.to_vec(), prompt_b.to_vec()],
        max_tokens,
        options,
    )?;
    let batch_elapsed = batch_started.elapsed();
    let [batch_a, batch_b] = batch.as_slice() else {
        return Err(InferError::Dimension(format!(
            "light-batch attendu 2 flux, reçu {}",
            batch.len()
        )));
    };
    let tokens_equal_a = batch_a.tokens == solo_a.tokens;
    let tokens_equal_b = batch_b.tokens == solo_b.tokens;
    for (label, output) in [("A", batch_a), ("B", batch_b)] {
        let generated = output
            .tokens
            .iter()
            .copied()
            .map(checked_generated_token)
            .collect::<Result<Vec<_>>>()?;
        let text = assets.decode_tokens(&generated, true)?;
        println!("--- flux {label} ---");
        println!("{}", text.trim());
    }

    let tok_s = |tokens: usize, seconds: f64| {
        if seconds > 0.0 {
            tokens as f64 / seconds
        } else {
            0.0
        }
    };
    // Mur decode du batch = temps total moins les prefills (valable pour le
    // time-slicing comme pour le pas duo, ou chaque flux compte le pas partage).
    let batch_decode = batch_elapsed
        .saturating_sub(batch_a.timings.prefill)
        .saturating_sub(batch_b.timings.prefill);
    let batch_tokens = batch_a.timings.decode_tokens + batch_b.timings.decode_tokens;
    eprintln!(
        "lightbatch_acceptance tokens_equal_a={} tokens_equal_b={} generated_a={} generated_b={} \
         solo_a_decode_tok_s={:.3} solo_b_decode_tok_s={:.3} \
         batch_eff_a_tok_s={:.3} batch_eff_b_tok_s={:.3} batch_agg_tok_s={:.3} \
         load_ms={} warmup_ms={} (débits indicatifs : à mesurer GPU idle)",
        tokens_equal_a,
        tokens_equal_b,
        batch_a.tokens.len(),
        batch_b.tokens.len(),
        tok_s(
            solo_a.timings.decode_tokens,
            solo_a.timings.decode.as_secs_f64()
        ),
        tok_s(
            solo_b.timings.decode_tokens,
            solo_b.timings.decode.as_secs_f64()
        ),
        tok_s(batch_a.timings.decode_tokens, batch_decode.as_secs_f64()),
        tok_s(batch_b.timings.decode_tokens, batch_decode.as_secs_f64()),
        tok_s(batch_tokens, batch_decode.as_secs_f64()),
        load_elapsed.as_millis(),
        warmup_elapsed.as_millis(),
    );
    if !tokens_equal_a || !tokens_equal_b {
        return Err(InferError::Config(
            "oracle light-batch échoué: un flux du batch diverge de son solo".to_string(),
        ));
    }
    Ok(())
}

pub fn run_mtp_acceptance(
    assets: &ModelAssets,
    decoder: &CausalDecoder,
    prompt_ids: &[usize],
    max_tokens: usize,
    options: &GenerationOptions,
    load_elapsed: Duration,
    warmup_elapsed: Duration,
) -> Result<()> {
    let max_draft = env::var("RETI_RUST_MTP_MAX_DRAFT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);
    let ar_started = Instant::now();
    let ar = decoder.generate_greedy_cached_with_options(prompt_ids, max_tokens, options)?;
    let ar_elapsed = ar_started.elapsed();
    let spec_started = Instant::now();
    let spec = decoder
        .generate_greedy_mtp_batched_with_options(prompt_ids, max_tokens, options, max_draft)?;
    let spec_elapsed = spec_started.elapsed();
    let tokens_equal = ar == spec.tokens;
    if !tokens_equal && env::var_os("RETI_RUST_MTP_ORACLE_DUMP").is_some() {
        let first_diff = ar
            .iter()
            .zip(spec.tokens.iter())
            .position(|(left, right)| left != right);
        let diff_index = first_diff.unwrap_or_else(|| ar.len().min(spec.tokens.len()));
        let ar_token = ar.get(diff_index).copied();
        let spec_token = spec.tokens.get(diff_index).copied();
        eprintln!(
            "mtp_oracle_diff index={} ar={:?} spec={:?} ar_len={} spec_len={}",
            diff_index,
            ar_token,
            spec_token,
            ar.len(),
            spec.tokens.len()
        );
        let start = diff_index.saturating_sub(6);
        let end = diff_index
            .checked_add(7)
            .map(|value| value.min(ar.len().max(spec.tokens.len())))
            .unwrap_or_else(|| ar.len().max(spec.tokens.len()));
        eprintln!(
            "mtp_oracle_window ar={:?} spec={:?}",
            ar.get(start..end.min(ar.len())).unwrap_or(&[]),
            spec.tokens
                .get(start..end.min(spec.tokens.len()))
                .unwrap_or(&[])
        );
    }
    let generated = spec
        .tokens
        .iter()
        .copied()
        .map(checked_generated_token)
        .collect::<Result<Vec<_>>>()?;
    let text = assets.decode_tokens(&generated, true)?;
    println!("{}", text.trim());

    // Taux d'acceptation STANDARD du decodage speculatif = drafts acceptes /
    // drafts proposes (Leviathan/Chen). C'est LE chiffre comparable a MTPLX
    // (`accepted_by_depth / drafted_by_depth`).
    let acceptance_rate = if spec.stats.proposed > 0 {
        spec.stats.accepted as f64 / spec.stats.proposed as f64
    } else {
        0.0
    };
    // NOTE: `avg_accepted_per_verify` = acceptes / forwards de verification. Ce
    // n'est PAS le taux d'acceptation : chaque pas depth-1 accepte declenche ~2
    // forwards (verify + bonus), donc cette valeur vaut ~1/2 du taux reel meme a
    // acceptation parfaite. Conserve pour diagnostic, jamais comme acceptance.
    let avg_accepted_per_verify = if spec.stats.verifications > 0 {
        spec.stats.accepted as f64 / spec.stats.verifications as f64
    } else {
        0.0
    };
    let spec_loop_ms = spec.loop_duration.as_millis();
    let spec_decode_tok_s = if spec.loop_duration.as_secs_f64() > 0.0 {
        spec.tokens.len() as f64 / spec.loop_duration.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "mtp_acceptance tokens_equal={} generated={} max_draft={} proposed={} accepted={} rejected={} acceptance_rate={:.3} verifications={} avg_accepted_per_verify={:.3} load_ms={} warmup_ms={} ar_ms={} spec_ms={} spec_loop_ms={} spec_decode_tok_s={:.3}",
        tokens_equal,
        spec.tokens.len(),
        max_draft,
        spec.stats.proposed,
        spec.stats.accepted,
        spec.stats.rejected,
        acceptance_rate,
        spec.stats.verifications,
        avg_accepted_per_verify,
        load_elapsed.as_millis(),
        warmup_elapsed.as_millis(),
        ar_elapsed.as_millis(),
        spec_elapsed.as_millis(),
        spec_loop_ms,
        spec_decode_tok_s,
    );
    for position in 0..max_draft {
        let proposed = spec
            .stats
            .proposed_by_position
            .get(position)
            .copied()
            .unwrap_or(0);
        let accepted = spec
            .stats
            .accepted_by_position
            .get(position)
            .copied()
            .unwrap_or(0);
        let rate = if proposed > 0 {
            accepted as f64 / proposed as f64
        } else {
            0.0
        };
        eprintln!(
            "mtp_acceptance_pos{} proposed={} accepted={} rate={:.3}",
            position + 1,
            proposed,
            accepted,
            rate
        );
    }
    if options.temperature <= f32::EPSILON && !tokens_equal {
        return Err(InferError::Config(
            "oracle AR==MTP échoué: tokens divergents en greedy".to_string(),
        ));
    }
    Ok(())
}

pub fn run_dflash_acceptance(
    assets: &ModelAssets,
    decoder: &CausalDecoder,
    draft: &DFlashDraft,
    prompt_ids: &[usize],
    max_tokens: usize,
    options: &GenerationOptions,
    load_elapsed: Duration,
    warmup_elapsed: Duration,
) -> Result<()> {
    let max_draft = env::var("RETI_RUST_DFLASH_MAX_DRAFT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(draft.info.block_size);
    let reference_started = Instant::now();
    let reference = decoder
        .generate_greedy_dflash_reference_with_options(prompt_ids, max_tokens, options, draft)?;
    let reference_elapsed = reference_started.elapsed();
    let spec_started = Instant::now();
    let spec = decoder.generate_greedy_dflash_batched_with_options(
        prompt_ids, max_tokens, options, draft, max_draft,
    )?;
    let spec_elapsed = spec_started.elapsed();
    let tokens_equal = reference == spec.tokens;
    let generated = spec
        .tokens
        .iter()
        .copied()
        .map(checked_generated_token)
        .collect::<Result<Vec<_>>>()?;
    let text = assets.decode_tokens(&generated, true)?;
    println!("{}", text.trim());

    let avg_accepted = if spec.stats.verifications > 0 {
        spec.stats.accepted as f64 / spec.stats.verifications as f64
    } else {
        0.0
    };
    eprintln!(
        "dflash_acceptance tokens_equal={} generated={} max_draft={} block_size={} proposed={} accepted={} rejected={} verifications={} avg_accepted_per_verify={:.3} load_ms={} warmup_ms={} ref_ms={} spec_ms={}",
        tokens_equal,
        spec.tokens.len(),
        max_draft,
        draft.info.block_size,
        spec.stats.proposed,
        spec.stats.accepted,
        spec.stats.rejected,
        spec.stats.verifications,
        avg_accepted,
        load_elapsed.as_millis(),
        warmup_elapsed.as_millis(),
        reference_elapsed.as_millis(),
        spec_elapsed.as_millis(),
    );
    for position in 0..max_draft.min(draft.info.block_size) {
        let proposed = spec
            .stats
            .proposed_by_position
            .get(position)
            .copied()
            .unwrap_or(0);
        let accepted = spec
            .stats
            .accepted_by_position
            .get(position)
            .copied()
            .unwrap_or(0);
        let rate = if proposed > 0 {
            accepted as f64 / proposed as f64
        } else {
            0.0
        };
        eprintln!(
            "dflash_acceptance_pos{} proposed={} accepted={} rate={:.3}",
            position + 1,
            proposed,
            accepted,
            rate
        );
    }
    if !tokens_equal {
        eprintln!(
            "dflash_acceptance_mismatch ref_prefix={} spec_prefix={}",
            token_prefix(&reference, 16),
            token_prefix(&spec.tokens, 16)
        );
        return Err(InferError::Config(
            "oracle AR==DFlash échoué: tokens divergents en greedy".to_string(),
        ));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn run_resident_linear_xray(
    assets: &ModelAssets,
    decoder: &CausalDecoder,
    prompt_ids: &[usize],
    options: &GenerationOptions,
) -> Result<()> {
    let report = decoder.resident_linear_xray(prompt_ids, options)?;
    let input_text = decode_one_token(assets, report.input_token)?;
    let reference_text = decode_one_token(assets, report.reference_token)?;
    let resident_text = decode_one_token(assets, report.resident_token)?;
    println!(
        "input={} reference={} resident={}",
        input_text.trim(),
        reference_text.trim(),
        resident_text.trim()
    );

    let first_divergent = report.layer_diffs.iter().find(|diff| diff.max_abs > 1.0e-4);
    eprintln!(
        "resident_linear_xray input_token={} reference_token={} resident_token={} final_max_abs={:.6e} final_mean_abs={:.6e}",
        report.input_token,
        report.reference_token,
        report.resident_token,
        report.final_max_abs,
        report.final_mean_abs
    );
    if let Some(layer_index) = report.probe_layer_index {
        eprintln!(
            "resident_linear_xray_probe layer={} normed_max_abs={:.6e} normed_mean_abs={:.6e} attn_max_abs={:.6e} attn_mean_abs={:.6e} attn_cpu_normed_max_abs={:.6e} attn_cpu_normed_mean_abs={:.6e}",
            layer_index,
            report.probe_normed_max_abs.unwrap_or_default(),
            report.probe_normed_mean_abs.unwrap_or_default(),
            report.probe_attn_max_abs.unwrap_or_default(),
            report.probe_attn_mean_abs.unwrap_or_default(),
            report.probe_attn_cpu_normed_max_abs.unwrap_or_default(),
            report.probe_attn_cpu_normed_mean_abs.unwrap_or_default()
        );
        eprintln!(
            "resident_linear_xray_init_state layer={} conv_max_abs={:.6e} conv_mean_abs={:.6e} ssm_max_abs={:.6e} ssm_mean_abs={:.6e}",
            layer_index,
            report
                .probe_init_state_conv_max_abs
                .unwrap_or_default(),
            report
                .probe_init_state_conv_mean_abs
                .unwrap_or_default(),
            report
                .probe_init_state_ssm_max_abs
                .unwrap_or_default(),
            report
                .probe_init_state_ssm_mean_abs
                .unwrap_or_default()
        );
        eprintln!(
            "resident_linear_xray_state layer={} conv_max_abs={:.6e} conv_mean_abs={:.6e} ssm_max_abs={:.6e} ssm_mean_abs={:.6e} cpu_normed_conv_max_abs={:.6e} cpu_normed_conv_mean_abs={:.6e} cpu_normed_ssm_max_abs={:.6e} cpu_normed_ssm_mean_abs={:.6e}",
            layer_index,
            report.probe_state_conv_max_abs.unwrap_or_default(),
            report.probe_state_conv_mean_abs.unwrap_or_default(),
            report.probe_state_ssm_max_abs.unwrap_or_default(),
            report.probe_state_ssm_mean_abs.unwrap_or_default(),
            report
                .probe_state_cpu_normed_conv_max_abs
                .unwrap_or_default(),
            report
                .probe_state_cpu_normed_conv_mean_abs
                .unwrap_or_default(),
            report
                .probe_state_cpu_normed_ssm_max_abs
                .unwrap_or_default(),
            report
                .probe_state_cpu_normed_ssm_mean_abs
                .unwrap_or_default()
        );
    }
    if let Some(diff) = first_divergent {
        eprintln!(
            "resident_linear_xray_first layer={} kind={} max_abs={:.6e} mean_abs={:.6e}",
            diff.layer_index, diff.attention_kind, diff.max_abs, diff.mean_abs
        );
    } else {
        eprintln!("resident_linear_xray_first none");
    }

    let mut top = report.layer_diffs.clone();
    top.sort_by(|left, right| {
        right
            .max_abs
            .partial_cmp(&left.max_abs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for diff in top.into_iter().take(12) {
        eprintln!(
            "resident_linear_xray_layer layer={} kind={} max_abs={:.6e} mean_abs={:.6e}",
            diff.layer_index, diff.attention_kind, diff.max_abs, diff.mean_abs
        );
    }
    Ok(())
}

pub fn print_dflash_draft(draft: &DFlashDraft) {
    let info = &draft.info;
    println!(
        "dflash_draft loaded tensors={} layers={} loaded_layers={} hidden={} block_size={} mask_token={} target_layers={}",
        info.tensor_count,
        info.num_hidden_layers,
        draft.layers.len(),
        info.hidden_size,
        info.block_size,
        info.mask_token_id,
        info.target_layer_ids
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn decode_one_token(assets: &ModelAssets, token: usize) -> Result<String> {
    let token = u32::try_from(token)
        .map_err(|_| InferError::Dimension(format!("token diagnostic hors plage: {token}")))?;
    assets.decode_tokens(&[token], true)
}

fn checked_generated_token(token: usize) -> Result<u32> {
    u32::try_from(token)
        .map_err(|_| InferError::Dimension(format!("token généré hors plage: {token}")))
}

fn token_prefix(tokens: &[usize], limit: usize) -> String {
    let mut out = tokens
        .iter()
        .take(limit)
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if tokens.len() > limit {
        out.push_str(",...");
    }
    out
}
