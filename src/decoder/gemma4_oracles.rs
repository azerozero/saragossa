//! Qualification réelle Gemma 4, sans cache de préfixe ni repli silencieux.
use super::*;
use crate::argmax;

#[test]
fn gemma4_prefill_override_is_thread_local_and_restored() {
    let original = crate::runtime_flags::prefill_resident_gemma4_enabled();
    {
        let _outer = crate::runtime_flags::override_prefill_resident_gemma4_for_test(!original);
        let other = std::thread::spawn(crate::runtime_flags::prefill_resident_gemma4_enabled)
            .join()
            .expect("invariant: lecture du flag sans panique");
        assert_eq!(other, original);
        {
            let _inner = crate::runtime_flags::override_prefill_resident_gemma4_for_test(original);
            assert_eq!(
                crate::runtime_flags::prefill_resident_gemma4_enabled(),
                original
            );
        }
        assert_eq!(
            crate::runtime_flags::prefill_resident_gemma4_enabled(),
            !original
        );
    }
    assert_eq!(
        crate::runtime_flags::prefill_resident_gemma4_enabled(),
        original
    );
}

fn small_gemma4_model() -> Result<CausalDecoder> {
    let mut weights = test_weights();
    for norm in [
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
    ] {
        weights.insert(
            format!("layers.0.{norm}.weight"),
            Tensor::from_vec(vec![2], vec![1.0; 2])?,
        );
    }
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        weights.insert(format!("layers.0.mlp.{projection}.weight"), identity2()?);
    }
    CausalDecoder::from_tensors(
        weights,
        CausalDecoderConfig {
            is_gemma4: true,
            head_dim: Some(2),
            rope_dims: Some(2),
            rope_theta: Some(10_000.0),
            rope_style: RopeStyle::Halves,
            activation: crate::Activation::GeluTanh,
            ..CausalDecoderConfig::default()
        },
    )?
    .with_metal_runtime()
}

#[test]
fn gemma4_resident_prefill_token_limit_before_embedding() -> Result<()> {
    let model = small_gemma4_model()?;
    let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(true);
    // À la borne, le résident atteint l'embedding (ids volontairement invalides).
    // Au-delà, il refuse avant toute allocation dépendant du prompt.
    let max_tokens = crate::runtime_flags::prefill_resident_gemma4_max_tokens();
    let mut invalid_prompt = vec![usize::MAX; max_tokens];
    assert!(model
        .prefill_cache_state_metal_resident_for_test(&invalid_prompt)
        .is_err());
    invalid_prompt.push(usize::MAX);
    assert!(model
        .prefill_cache_state_metal_resident_for_test(&invalid_prompt)?
        .is_none());
    // Le dispatch ordinaire essaie bien le per-op après l'inéligibilité.
    assert!(model
        .prefill_cache_state_batched_for_test(&invalid_prompt)
        .is_err());
    Ok(())
}

#[test]
fn gemma4_resident_prefill_propagates_execution_error() -> Result<()> {
    let mut model = small_gemma4_model()?;
    let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(true);
    assert!(model
        .prefill_cache_state_metal_resident_for_test(&[0, 1])?
        .is_some());
    // La spec devient incohérente avec l'embedding après le chargement :
    // le backend doit renvoyer Dimension, jamais Ok(None) suivi d'un repli.
    model.final_norm = Tensor::from_vec(vec![3], vec![1.0; 3])?;
    let result = model.prefill_cache_state_metal_resident_for_test(&[0, 1]);
    assert!(matches!(result, Err(InferError::Dimension(ref message))
        if message.contains("prefill résident input=")));
    // Vérifier aussi le dispatch : il doit propager la même erreur au lieu
    // d'essayer le per-op après l'échec résident (y compris un kernel d512).
    let result = model.prefill_cache_state_batched_for_test(&[0, 1]);
    assert!(matches!(result, Err(InferError::Dimension(ref message))
        if message.contains("prefill résident input=")));
    Ok(())
}

fn model_and_prompt(tokens: usize) -> Result<(CausalDecoder, Vec<usize>)> {
    let path = crate::runtime_flags::env_var("SARAGOSSA_GEMMA4_DIR")
        .ok_or_else(|| InferError::Config("SARAGOSSA_GEMMA4_DIR requis pour cet oracle".into()))?;
    let assets = crate::ModelAssets::load_local(path)?;
    assert!(assets.config.is_gemma4());
    // Corpus fixe, tronqué avant l'enveloppe chat : les deux tailles traversent
    // les frontières 1024/2048 et la fenêtre glissante pendant le decode.
    let text = include_str!("../../tests/fixtures/goldens/kld/corpus.txt");
    let mut body = assets.encode_prompt(text)?;
    assert!(!body.is_empty());
    while body.len() < tokens {
        body.extend_from_within(..);
    }
    body.truncate(tokens);
    let text = assets.decode_tokens(&body, false)?;
    let chat = crate::render_gemma4_chat(
        &[crate::ChatTemplateMessage::new("user", text)],
        true,
        false,
    );
    let prompt = assets
        .encode_prompt(&chat)?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let executor = crate::MetalExecutor::new()?;
    let model =
        crate::load_causal_decoder_for_metal(&assets, &executor)?.with_metal_executor(executor)?;
    Ok((model, prompt))
}

fn resident_tokens(
    model: &CausalDecoder,
    mut cache: CausalDecoderCache,
    state: &Tensor,
) -> Result<Vec<usize>> {
    let mut token = argmax(model.logits_from_final_state(state)?.data())?;
    let kv_bf16 = CausalDecoder::resident_kv_bf16(false, cache.position);
    assert!(model.setup_resident_full_decode(&mut cache, 200, 1, kv_bf16)?);
    let mut tokens = vec![token];
    for _ in 1..200 {
        token = model.decode_token_resident(&mut cache, token)?;
        tokens.push(token);
    }
    Ok(tokens)
}

fn prefill_comparison(tokens: usize) -> Result<(Vec<usize>, Vec<usize>)> {
    let (model, prompt) = model_and_prompt(tokens)?;
    if tokens < 64 {
        assert!(prompt.len() < 64, "enveloppe chat comprise");
    }
    let (cache, state) = {
        let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(false);
        model.prefill_cache_state_batched_for_test(&prompt)?
    };
    let reference = resident_tokens(&model, cache, &state)?;
    let (cache, state) = {
        let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(true);
        model
            .prefill_cache_state_metal_resident_for_test(&prompt)?
            .ok_or_else(|| InferError::Config("oracle: prefill résident refusé".into()))?
    };
    let actual = resident_tokens(&model, cache, &state)?;
    let first = reference.iter().zip(&actual).position(|(a, b)| a != b);
    eprintln!(
        "Gemma4 prefill prompt={} greedy=200 first_div={first:?}",
        prompt.len()
    );
    assert_eq!(reference.len(), 200);
    assert_eq!(actual.len(), 200);
    Ok((reference, actual))
}

// Les deux oracles externes ci-dessous ne préremplissent que 88–96 tokens
// (chunks de 216–224 moins 128 positions notées). Ils couvrent le kernel court,
// ni mid (>256), ni long d512 (>2048). Les A/B 1k/4k restent des diagnostics
// greedy sans seuil de qualité. L'oracle long séparé enregistre le per-op f32
// à ~4100 tokens et compare le résident, avec trace du dispatch long d512 et
// les mêmes seuils KL/p99/top-1 ; ce n'est pas une référence MLX longue.
// L'identité stricte reste exigée pour le decode seul et le prompt français
// court historique (32 tokens générés).
#[test]
#[ignore = "oracle 4bit de référence local, lancer sous verrou GPU"]
fn gemma4_prefill_reference_kld_per_op() -> crate::golden::external::TestResult {
    crate::golden::kld::compare_gemma4_prefill(false)
}

#[test]
#[ignore = "oracle 4bit de référence local, lancer sous verrou GPU"]
fn gemma4_prefill_reference_kld_resident() -> crate::golden::external::TestResult {
    crate::golden::kld::compare_gemma4_prefill(true)
}

#[test]
#[ignore = "oracle long sur poids réels, borne explicite >4096, lancer sous verrou GPU"]
fn gemma4_prefill_long_per_op_kld_resident() -> crate::golden::external::TestResult {
    let (model, prompt) = model_and_prompt(4096)?;
    crate::golden::kld::compare_gemma4_long_prefill(&model, &prompt)
}

#[test]
#[ignore = "diagnostic court q4a8-ls divergent ; qualité exigée par les deux tests KL"]
fn gemma4_prefill_ab_short() -> Result<()> {
    prefill_comparison(32).map(|_| ())
}

#[test]
#[ignore = "diagnostic A/B modèle local ; qualité exigée par les deux tests KL"]
fn gemma4_prefill_ab_1k() -> Result<()> {
    prefill_comparison(1024).map(|_| ())
}

#[test]
#[ignore = "diagnostic A/B modèle local ; qualité exigée par les deux tests KL"]
fn gemma4_prefill_ab_4k() -> Result<()> {
    prefill_comparison(4096).map(|_| ())
}

#[test]
#[ignore = "oracle modèle local, lancer sous verrou GPU"]
fn gemma4_decode_identity_200() -> Result<()> {
    let (model, prompt) = model_and_prompt(1024)?;
    let (mut cache, state) = {
        let _flag = crate::runtime_flags::override_prefill_resident_gemma4_for_test(false);
        model.prefill_cache_state_batched_for_test(&prompt)?
    };
    let actual = resident_tokens(&model, cache.clone(), &state)?;
    let mut token = argmax(model.logits_from_final_state(&state)?.data())?;
    let mut reference = vec![token];
    for _ in 1..200 {
        token = argmax(model.next_logits_cached(&mut cache, token)?.data())?;
        reference.push(token);
    }
    let first = reference.iter().zip(&actual).position(|(a, b)| a != b);
    eprintln!("Gemma4 decode greedy=200 first_div={first:?}");
    assert_eq!(
        actual, reference,
        "decode résident/per-op: première divergence {first:?}"
    );
    Ok(())
}
