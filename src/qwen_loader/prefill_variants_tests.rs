use super::*;

/// Même modèle chargé, cache de préfixe coupé, chemins prouvés par compteurs.
/// Les quatre combinaisons isolent C1, B1 puis leur interaction.
#[test]
#[ignore = "modèle Qwen MoE hv32 local et verrou GPU requis"]
fn qwen_moe_prefill_variants_greedy_identity() -> Result<()> {
    use crate::metal_backend::{
        linear_nax_hv32_calls, moe_coop_tiled_trace_for_test, with_flint_moe_coop_tiled_override,
        with_linear_nax_disabled,
    };
    assert!(
        !crate::runtime_flags::prefix_cache_enabled(),
        "SARAGOSSA_RUST_PREFIX_CACHE=0 requis"
    );
    assert!(
        crate::runtime_flags::gated_delta_nax_hv32_enabled(),
        "SARAGOSSA_RUST_GATED_DELTA_NAX_HV32=1 et NAX linéaire requis"
    );
    let path = std::env::var_os("SARAGOSSA_TEST_PREFILL_MODEL")
        .expect("invariant: SARAGOSSA_TEST_PREFILL_MODEL doit désigner le modèle local");
    let assets = ModelAssets::load_local(std::path::Path::new(&path))?;
    let executor = crate::MetalExecutor::new()?;
    let model =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    let lengths =
        std::env::var("SARAGOSSA_TEST_PREFILL_LENGTHS").unwrap_or_else(|_| "512,4096".into());
    let steps = std::env::var("SARAGOSSA_TEST_PREFILL_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    assert!(steps > 0);
    let text = match std::env::var_os("SARAGOSSA_TEST_PREFILL_PROMPT") {
        Some(path) => std::fs::read_to_string(path).expect("invariant: corpus demandé lisible"),
        None => {
            "Explique les invariants de sécurité mémoire en Rust et donne un exemple de fonction. "
                .into()
        }
    };
    let seed = assets.encode_prompt(&text)?;
    assert!(!seed.is_empty());
    for length in lengths.split(',') {
        let len: usize = length.parse().expect("invariant: longueur entière");
        assert!(len >= 32);
        let prompt = seed
            .iter()
            .cycle()
            .take(len)
            .map(|&t| t as usize)
            .collect::<Vec<_>>();
        let mut reference = None;
        let mut hidden_reference = None;
        for (tiled, nax) in [(false, false), (true, false), (false, true), (true, true)] {
            let (hidden, tokens, tiled_calls, nax_calls) =
                with_flint_moe_coop_tiled_override(tiled, || {
                    with_linear_nax_disabled(!nax, || -> Result<_> {
                        let tiled_before = moe_coop_tiled_trace_for_test();
                        let nax_before = linear_nax_hv32_calls();
                        let hidden = model
                            .prefill_hidden_for_test(
                                &prompt,
                                Some(crate::runtime_flags::prefill_resident_chunk_tokens()),
                            )?
                            .expect("invariant: prefill résident requis");
                        assert_eq!(
                            moe_coop_tiled_trace_for_test() > tiled_before,
                            tiled,
                            "couverture BM32 du hidden"
                        );
                        assert_eq!(
                            linear_nax_hv32_calls() > nax_before,
                            nax,
                            "couverture NAX du hidden"
                        );
                        let tiled_before = moe_coop_tiled_trace_for_test();
                        let nax_before = linear_nax_hv32_calls();
                        let tokens = model
                            .generate_greedy_timed_with_options(
                                &prompt,
                                steps,
                                &crate::GenerationOptions::default(),
                            )?
                            .tokens;
                        Ok((
                            hidden,
                            tokens,
                            moe_coop_tiled_trace_for_test() - tiled_before,
                            linear_nax_hv32_calls() - nax_before,
                        ))
                    })
                })?;
            assert_eq!(tiled_calls > 0, tiled, "couverture BM32");
            assert_eq!(nax_calls > 0, nax, "couverture NAX hv32");
            eprintln!("prefill len={len} tiled={tiled} nax={nax} appels={tiled_calls}/{nax_calls} tokens={tokens:?}");
            if let Some(ref baseline) = reference {
                assert_eq!(
                    &tokens, baseline,
                    "greedy len={len} tiled={tiled} nax={nax}"
                );
            } else {
                reference = Some(tokens);
            }
            if !nax {
                let bits = hidden
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>();
                if let Some(ref baseline) = hidden_reference {
                    assert!(&bits == baseline, "hidden BM32 != BM16");
                } else {
                    hidden_reference = Some(bits);
                }
            }
        }
    }
    Ok(())
}
