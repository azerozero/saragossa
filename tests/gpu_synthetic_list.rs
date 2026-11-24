//! Garde de la liste fermée de `.github/workflows/gpu-synthetic.yml` : tout
//! test `#[ignore]` dont la raison se dit « synthétique » doit y figurer, ou
//! être exempté ici avec sa raison.

use std::collections::BTreeSet;
use std::path::Path;

/// Tests synthétiques volontairement hors CI GPU.
/// NOTE: `na` exige GEMM/NAX, absent du runner macos-15 ; `bench` mesure la
/// vitesse ; `a-trier` reste à qualifier (candidats à la liste).
const EXEMPT: &[(&str, &str)] = &[
    ("dense_bf16_prefill_gemm_matches_f32", "na"),
    ("decision_dense_bf16_gemm_rounds_activation_once", "na"),
    ("dense_bf16_prefill_kill_switch_selects_qmv", "na"),
    ("dense_bf16_pair_gemm_matches_f32", "na"),
    ("dense_bf16_fused_mlp_matches_x1", "na"),
    ("dense_bf16_prefill_x1_scope_rounds_activation_once", "na"),
    ("dense_bf16_prefill_x1_kill_switch_selects_hi_lo", "na"),
    ("dense_bf16_prefill_x1_forced_outside_scope", "na"),
    ("tts_qmm_na_dequant_scope_controls_selection", "na"),
    ("qmm_u56_prefill_matches_bf16_cpu_and_routes", "na"),
    ("qmm_u3_tiled_prefill_matches_bf16_cpu_and_routes", "na"),
    ("affine_offsets_fast_kernels_bit_exact", "na"),
    ("affine_offsets_natural_selection_and_standalone", "na"),
    (
        "dflash_decoder_entry_point_preserves_shared_storage",
        "a-trier",
    ),
    ("golden_clone_e2e_audio_matches_fixture", "a-trier"),
    ("golden_voicedesign_e2e_audio_matches_fixture", "a-trier"),
    ("kernel_profiles_per_model_metal_parity", "a-trier"),
    ("moe_gather_overlap_cost_bench", "bench"),
    ("mtp_bf16_busy_claim_memory_guard_cleans_setup", "a-trier"),
    (
        "mtp_bf16_history_generation_adoption_and_f32_fallback",
        "a-trier",
    ),
    (
        "mtp_bf16_post_seed_refusal_rebuilds_f32_and_streams_once",
        "a-trier",
    ),
    ("mtp_generation_publishes_shared_seed_history", "a-trier"),
    ("mtp_seed_bf16_streamed_32768_bit_exact", "a-trier"),
    ("mtp_seed_bf16_streamed_bit_exact_and_adopted", "a-trier"),
    ("mtp_seed_shared_prefix_keeps_allocations", "a-trier"),
    ("mtp_seed_streamed_32768_bit_exact", "a-trier"),
    ("mtp_seed_streamed_4097_bit_exact", "a-trier"),
    ("mtp_seed_streamed_8297_bit_exact", "a-trier"),
    ("mtp_seed_streamed_private_scratch_bit_exact", "a-trier"),
    ("streamed_metal_weights_match_legacy_upload", "a-trier"),
    ("synthetic_metal_bf16_greedy_survives_restart", "a-trier"),
];

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("invariant: src lisible") {
        let path = entry.expect("invariant: entrée lisible").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Noms de fonctions `#[ignore]` dont la raison contient « synth ».
fn synthetic_ignored() -> BTreeSet<String> {
    let mut files = Vec::new();
    rust_files(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let mut names = BTreeSet::new();
    for file in files {
        let text = std::fs::read_to_string(&file).expect("invariant: source UTF-8");
        let mut pending = false;
        let mut attr = String::new();
        let mut in_attr = false;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("#[ignore") {
                in_attr = true;
                attr.clear();
            }
            if in_attr {
                attr.push_str(t);
                if t.ends_with(']') {
                    in_attr = false;
                    pending = attr.to_lowercase().contains("synth");
                }
                continue;
            }
            if pending {
                if let Some(rest) = t.strip_prefix("fn ").or_else(|| t.strip_prefix("pub fn ")) {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    names.insert(name);
                    pending = false;
                } else if !t.starts_with('#') && !t.starts_with("//") && !t.is_empty() {
                    pending = false;
                }
            }
        }
    }
    names
}

/// Noms de la liste `tests=( ... )` du workflow, dernier segment du chemin.
fn workflow_names() -> BTreeSet<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/gpu-synthetic.yml");
    let text = std::fs::read_to_string(path).expect("invariant: workflow présent");
    let mut names = BTreeSet::new();
    let mut inside = false;
    for line in text.lines() {
        let t = line.trim();
        if t == "tests=(" {
            inside = true;
        } else if t == ")" {
            inside = false;
        } else if inside && t.contains("::") {
            names.insert(t.rsplit("::").next().unwrap_or(t).to_string());
        }
    }
    names
}

#[test]
fn every_synthetic_ignored_test_is_listed_or_exempt() {
    let listed = workflow_names();
    assert!(listed.len() > 40, "liste du workflow illisible");
    let exempt: BTreeSet<&str> = EXEMPT.iter().map(|(name, _)| *name).collect();
    let missing: Vec<String> = synthetic_ignored()
        .into_iter()
        .filter(|name| !listed.contains(name) && !exempt.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "tests synthétiques ignorés absents de gpu-synthetic.yml (les ajouter, ou les exempter ici) :\n{}",
        missing.join("\n")
    );
}
