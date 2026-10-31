//! Oracle réel des k/v du Qwen 27B dense servis par le GEMM NA u4 bf16×3.

use super::flint_tests::{
    argmax, centered_max_gap, synthetic_long_prompt, MAX_GAP_MLX_BF16, MEAN_GAP_MLX_BF16,
};
use super::*;

/// k/v 5120→1024 du prefill Qwen 27B au bf16×3 contre le qmv f32 du plancher
/// 2048, sur le modèle réel (`SARAGOSSA_QWEN27B_MODEL_DIR`). Pour chaque
/// longueur de `SARAGOSSA_QWEN_KV_X3_LENS` (4096,16384,32000 par défaut),
/// texte de `SARAGOSSA_QWEN_KV_X3_PROMPT` ou synthétique, puis
/// `SARAGOSSA_QWEN_KV_X3_STEPS` tokens greedy (128 par défaut). La dérive des
/// logits est suivie à chaque pas le long de la continuation qmv. Une
/// divergence des greedy est rapportée avec sa marge et le bruit du qmv
/// contre lui-même (morceaux de prefill deux fois plus petits) ; seul le
/// garde-fou de dérive fait échouer le test, avec la preuve par compteur que
/// le bf16×3 a servi (et que le plancher forcé l'a coupé).
#[test]
#[ignore = "poids réels Qwen 27B 4-bit (≈15 Go) requis ; mesure de dérive, pas un gate"]
fn qwen27b_kv_nax_x3_greedy_matches_qmv() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_QWEN27B_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("test Qwen k/v x3 sauté : définir SARAGOSSA_QWEN27B_MODEL_DIR");
        return;
    };
    // Le cache de préfixe rendrait au second run l'état du premier : les deux
    // greedy partageraient le K/V qmv et seraient identiques par construction.
    assert!(
        !crate::runtime_flags::prefix_cache_enabled(),
        "lancer avec SARAGOSSA_RUST_PREFIX_CACHE=0"
    );
    assert!(
        crate::runtime_flags::qwen_kv_nax_x3_enabled(),
        "SARAGOSSA_RUST_QWEN_KV_NAX_X3=0 : rien à comparer"
    );
    let steps = std::env::var("SARAGOSSA_QWEN_KV_X3_STEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);
    let lens = std::env::var("SARAGOSSA_QWEN_KV_X3_LENS")
        .unwrap_or_else(|_| "4096,16384,32000".into())
        .split(',')
        .map(|value| value.trim().parse::<usize>().expect("invariant: longueur"))
        .collect::<Vec<_>>();
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets Qwen 27B");
    let text = std::env::var_os("SARAGOSSA_QWEN_KV_X3_PROMPT")
        .map(|path| std::fs::read_to_string(path).expect("invariant: prompt lisible"));
    let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
    let model = crate::load_causal_decoder_for_metal(&assets, &executor)
        .expect("invariant: Qwen 27B chargeable en Metal")
        .with_metal_executor(executor)
        .expect("invariant: runtime Metal attaché");
    let target = crate::runtime_flags::prefill_resident_chunk_tokens();
    // `None` : chemin de l'executor (k/v au bf16×3) ; `Some(2048)` : plancher
    // forcé, k/v au qmv f32.
    let with_floor = |floor: Option<usize>, run: &mut dyn FnMut()| {
        let before = crate::metal_backend::qmm_na_u4_x3_dispatches_for_test();
        match floor {
            Some(min_dim) => crate::metal_backend::with_qmm_na_u4_min_dim_override(min_dim, run),
            None => run(),
        }
        let dispatched = crate::metal_backend::qmm_na_u4_x3_dispatches_for_test() - before;
        match floor {
            Some(_) => assert_eq!(dispatched, 0, "bf16×3 dispatché sous plancher forcé"),
            None => assert!(dispatched > 0, "bf16×3 jamais dispatché"),
        }
        dispatched
    };
    for len in lens {
        let prompt = match &text {
            Some(text) => {
                let ids = assets
                    .tokenizer
                    .encode(text)
                    .expect("invariant: tokenisation");
                assert!(ids.len() >= len, "prompt de {} tokens < {len}", ids.len());
                ids[..len].iter().map(|&id| id as usize).collect::<Vec<_>>()
            }
            None => synthetic_long_prompt(&assets, len),
        };
        let greedy = |floor: Option<usize>| {
            let mut tokens = Vec::new();
            let dispatched = with_floor(floor, &mut || {
                let output = model
                    .generate_greedy_timed_with_options(
                        &prompt,
                        steps,
                        &crate::GenerationOptions::default(),
                    )
                    .expect("invariant: génération résidente");
                println!(
                    "{len} tokens, plancher {floor:?} : prefill {:.1?}, début {:?}",
                    output.timings.prefill,
                    &output.tokens[..output.tokens.len().min(8)]
                );
                tokens = output.tokens;
            });
            (tokens, dispatched)
        };
        // Logits du dernier rang du prompt puis de chaque pas forcé : la ligne
        // `i` prédit le token `i` de la continuation.
        let forced_rows = |floor: Option<usize>, forced: &[usize]| {
            let mut rows = Vec::new();
            with_floor(floor, &mut || {
                let hidden = model
                    .prefill_hidden_for_test(&prompt, Some(target))
                    .expect("invariant: prefill résident")
                    .expect("prefill résident Metal attendu pour Qwen 27B");
                rows = model
                    .logits_of_hidden_rows_for_test(&hidden, &[len - 1])
                    .expect("invariant: logits du prefill");
                drop(hidden);
                rows.extend(
                    model
                        .resident_forced_logits_for_test(&prompt, forced, false)
                        .expect("invariant: logits résidents"),
                );
            });
            rows
        };
        let (qmv_tokens, _) = greedy(Some(2048));
        let (x3_tokens, dispatched) = greedy(None);
        assert_eq!(qmv_tokens.len(), steps, "greedy qmv écourté");
        assert_eq!(x3_tokens.len(), steps, "greedy x3 écourté");
        let forced = &qmv_tokens[..steps - 1];
        let qmv_rows = forced_rows(Some(2048), forced);
        let x3_rows = forced_rows(None, forced);
        let (mut max, mut mean, mut argmax_ok) = (0.0_f32, 0.0_f32, 0_usize);
        let mut step_gaps = Vec::with_capacity(qmv_rows.len());
        for (step, (x3, qmv)) in x3_rows.iter().zip(&qmv_rows).enumerate() {
            // `f32::max` ignore les NaN : une ligne invalide passerait pour un
            // écart nul dans les réductions ci-dessous.
            assert!(
                x3.iter().chain(qmv).all(|value| value.is_finite()),
                "logits non finis au pas {step}"
            );
            let width = qmv.len() as f32;
            let shift = x3.iter().zip(qmv).map(|(a, b)| a - b).sum::<f32>() / width;
            let centered = x3.iter().zip(qmv).map(|(a, b)| (a - b - shift).abs());
            let row_max = centered.clone().fold(0.0_f32, f32::max);
            max = max.max(row_max);
            mean = mean.max(centered.sum::<f32>() / width);
            argmax_ok += usize::from(argmax(x3) == argmax(qmv));
            step_gaps.push(row_max);
        }
        let half = (target / 2).max(512);
        let mut qmv_half = Vec::new();
        with_floor(Some(2048), &mut || {
            let hidden = model
                .prefill_hidden_for_test(&prompt, Some(half))
                .expect("invariant: prefill résident")
                .expect("prefill résident Metal attendu pour Qwen 27B");
            qmv_half = model
                .logits_of_hidden_rows_for_test(&hidden, &[len - 1])
                .expect("invariant: logits du prefill")
                .remove(0);
        });
        let noise = centered_max_gap(&qmv_half, &qmv_rows[0]);
        println!(
            "{len} tokens : {dispatched} dispatches x3 ; dérive x3 vs qmv sur {} pas forcés : \
             max {max:.4}, pire moyenne {mean:.5}, argmax {argmax_ok}/{} ; bruit qmv au pas 0 \
             (morceaux {target} contre {half}) : max {noise:.5}, argmax {:?} / {:?}",
            qmv_rows.len(),
            qmv_rows.len(),
            argmax(&qmv_rows[0]),
            argmax(&qmv_half)
        );
        match qmv_tokens.iter().zip(&x3_tokens).position(|(a, b)| a != b) {
            None => println!("{len} tokens : greedy {steps} tokens identiques x3/qmv"),
            Some(step) => {
                let row = &qmv_rows[step];
                println!(
                    "{len} tokens : première divergence au pas {step} : qmv {} / x3 {}, \
                     marge qmv {:.5}, dérive du pas {:.5}",
                    qmv_tokens[step],
                    x3_tokens[step],
                    row[qmv_tokens[step]] - row[x3_tokens[step]],
                    step_gaps[step]
                );
            }
        }
        // Garde-fou large de mesure, pas un budget de qualité (cf.
        // MAX_GAP_MLX_BF16) : le bf16×3 colle au f32 à ~2e-7 près.
        assert!(
            max < MAX_GAP_MLX_BF16,
            "écart max {max} ≥ {MAX_GAP_MLX_BF16}"
        );
        assert!(
            mean < MEAN_GAP_MLX_BF16,
            "écart moyen {mean} ≥ {MEAN_GAP_MLX_BF16}"
        );
    }
}
