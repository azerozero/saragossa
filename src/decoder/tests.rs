use super::*;
use safetensors::{serialize, Dtype, View};
use std::borrow::Cow;

#[test]
fn causal_decoder_generates_expected_token() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let logits = model
        .next_logits(&[0])
        .expect("invariant: forward décodeur valide");
    assert_eq!(logits.shape(), &[1, 3]);
    let generated = model
        .generate_greedy(&[0], 2)
        .expect("invariant: greedy valide");
    assert_eq!(generated, vec![0, 0]);
}

#[test]
fn causal_decoder_loads_from_safetensors() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    let buffer =
        serialize(tensor_views(test_weights()), None).expect("invariant: safetensors sérialisable");
    std::fs::write(tmp.path(), buffer).expect("invariant: écriture temporaire");
    let model = CausalDecoder::from_safetensors(tmp.path(), CausalDecoderConfig::default())
        .expect("invariant: safetensors chargeable");
    let generated = model
        .generate_greedy(&[0], 1)
        .expect("invariant: greedy valide");
    assert_eq!(generated, vec![0]);
}

#[test]
fn causal_decoder_applies_optional_mlp_block() {
    let base = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let mut weights = test_weights();
    insert_identity_mlp(&mut weights);
    let with_mlp = CausalDecoder::from_tensors(weights, CausalDecoderConfig::default())
        .expect("invariant: poids MLP valides");

    let base_logits = base
        .next_logits(&[0])
        .expect("invariant: forward attention valide");
    let mlp_logits = with_mlp
        .next_logits(&[0])
        .expect("invariant: forward MLP valide");
    assert_ne!(base_logits.data(), mlp_logits.data());
}

#[test]
fn causal_decoder_supports_grouped_query_attention() {
    let config = CausalDecoderConfig {
        rope_theta: None,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let model =
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: poids GQA cohérents");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward GQA valide");
    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn causal_decoder_applies_qk_norm_when_present() {
    let base = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids attention valides");
    let mut weights = test_weights();
    insert_qk_norm(&mut weights, 0, vec![2.0, 0.25], vec![0.25, 2.0]);
    let with_qk_norm = CausalDecoder::from_tensors(weights, CausalDecoderConfig::default())
        .expect("invariant: q_norm/k_norm valides");

    let base_logits = base
        .next_logits(&[0, 1])
        .expect("invariant: forward sans qk norm valide");
    let qk_logits = with_qk_norm
        .next_logits(&[0, 1])
        .expect("invariant: forward avec qk norm valide");

    assert_ne!(base_logits.data(), qk_logits.data());
    assert!(qk_logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn causal_decoder_applies_attention_output_gate_when_enabled() {
    let config = CausalDecoderConfig {
        attn_output_gate: true,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let gated = CausalDecoder::from_tensors(gated_attention_weights(), config)
        .expect("invariant: attention gated valide");
    let base = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: attention non gated valide");

    let gated_logits = gated
        .next_logits(&[0, 1])
        .expect("invariant: forward gated valide");
    let base_logits = base
        .next_logits(&[0, 1])
        .expect("invariant: forward base valide");

    assert_ne!(gated_logits.data(), base_logits.data());
    assert!(gated_logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn causal_decoder_stacks_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let mut weights = test_weights();
        for layer in 1..layer_count {
            insert_attention_layer(&mut weights, layer);
        }
        let config = CausalDecoderConfig {
            num_hidden_layers: layer_count,
            ..CausalDecoderConfig::default()
        };
        let model = CausalDecoder::from_tensors(weights, config)
            .expect("invariant: pile multi-couches valide");
        let logits = model
            .next_logits(&[0, 1])
            .expect("invariant: forward multi-couches valide");

        assert_eq!(logits.shape(), &[1, 3]);
        assert!(logits.data().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn cached_greedy_matches_full_sequence_for_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let mut weights = test_weights();
        for layer in 1..layer_count {
            insert_attention_layer(&mut weights, layer);
        }
        let config = CausalDecoderConfig {
            num_hidden_layers: layer_count,
            ..CausalDecoderConfig::default()
        };
        let model = CausalDecoder::from_tensors(weights, config)
            .expect("invariant: pile multi-couches valide");

        let full = model
            .generate_greedy_full_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy complet valide");
        let cached = model
            .generate_greedy_cached_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy cache valide");

        assert_eq!(cached, full, "layer_count={layer_count}");
    }
}

#[test]
fn hybrid_cached_greedy_matches_full_sequence_for_multiple_layer_counts() {
    for layer_count in [1_usize, 2, 3, 4, 5, 6] {
        let config = hybrid_config(layer_count);
        let model = CausalDecoder::from_tensors(hybrid_weights(layer_count, 2), config)
            .expect("invariant: pile hybride valide");

        let full = model
            .generate_greedy_full_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy complet hybride valide");
        let cached = model
            .generate_greedy_cached_with_options(&[0, 1], 4, &GenerationOptions::default())
            .expect("invariant: greedy cache hybride valide");

        assert_eq!(cached, full, "layer_count={layer_count}");
    }
}

#[test]
fn speculative_greedy_with_injected_good_draft_matches_ar() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let options = GenerationOptions::default();
    let ar = model
        .generate_greedy_cached_with_options(&[0, 1], 6, &options)
        .expect("invariant: AR valide");
    let mut oracle_context = vec![0, 1];
    oracle_context.extend(ar.iter().copied());

    let spec = model
        .generate_greedy_speculative_with_draft(&[0, 1], 6, &options, 2, |context, limit| {
            let offset = context.len();
            oracle_context
                .get(offset..offset.saturating_add(limit))
                .unwrap_or(&[])
                .to_vec()
        })
        .expect("invariant: spéculatif valide");

    assert_eq!(spec.tokens, ar);
    assert!(spec.stats.accepted > 0);
    assert_eq!(spec.stats.rollbacks, 0);
}

#[test]
fn speculative_greedy_rolls_back_bad_draft_and_matches_ar_on_hybrid() {
    let model = CausalDecoder::from_tensors(hybrid_weights(3, 2), hybrid_config(3))
        .expect("invariant: modèle hybride valide");
    let options = GenerationOptions::default();
    let ar = model
        .generate_greedy_cached_with_options(&[0, 1], 6, &options)
        .expect("invariant: AR hybride valide");

    let spec = model
        .generate_greedy_speculative_with_draft(&[0, 1], 6, &options, 2, |_context, limit| {
            vec![1; limit]
        })
        .expect("invariant: spéculatif hybride valide");

    assert_eq!(spec.tokens, ar);
    assert!(spec.stats.rejected > 0);
    assert!(spec.stats.rollbacks > 0);
}

#[test]
fn cached_next_logits_match_full_prefix_logits() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let mut cache = model.empty_cache();
    let tokens = [0_usize, 1, 2];

    for prefix_len in 1..=3 {
        let prompt = &tokens[..prefix_len];
        let cached = model
            .next_logits_cached(&mut cache, prompt[prefix_len - 1])
            .expect("invariant: forward cache valide");
        let full = model
            .next_logits(prompt)
            .expect("invariant: forward complet valide");

        assert_close(cached.data(), full.data(), 1.0e-5);
        assert_eq!(cache.position(), prefix_len);
        assert_eq!(cache.layer_count(), 1);
    }
}

#[test]
fn cached_greedy_matches_full_sequence_with_grouped_query_attention() {
    let config = CausalDecoderConfig {
        rope_theta: Some(10_000.0),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        ..CausalDecoderConfig::default()
    };
    let model =
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: poids GQA cohérents");

    let full = model
        .generate_greedy_full_with_options(&[0, 1], 3, &GenerationOptions::default())
        .expect("invariant: greedy complet GQA valide");
    let cached = model
        .generate_greedy_cached_with_options(&[0, 1], 3, &GenerationOptions::default())
        .expect("invariant: greedy cache GQA valide");

    assert_eq!(cached, full);
}

#[test]
fn cached_sampling_is_seed_deterministic() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let options = GenerationOptions {
        temperature: 0.8,
        top_p: 1.0,
        seed: 123,
        ..GenerationOptions::default()
    };

    let left = model
        .generate_greedy_cached_with_options(&[0, 1], 4, &options)
        .expect("invariant: sampling cache valide");
    let right = model
        .generate_greedy_cached_with_options(&[0, 1], 4, &options)
        .expect("invariant: sampling cache valide");

    assert_eq!(left, right);
    assert_eq!(left.len(), 4);
}

#[test]
fn cached_decoder_stops_before_emitting_stop_token() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let generated = model
        .generate_greedy_cached_with_options(
            &[0],
            4,
            &GenerationOptions {
                stop_token_ids: vec![0],
                ..GenerationOptions::default()
            },
        )
        .expect("invariant: greedy cache avec stop valide");
    assert!(generated.is_empty());
}

#[test]
fn causal_decoder_rejects_empty_prompt() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let err = model
        .generate_greedy(&[], 1)
        .expect_err("invariant: prompt vide rejeté");
    assert!(matches!(err, InferError::Dimension(_)));
}

#[test]
fn causal_decoder_stops_before_emitting_stop_token() {
    let model = CausalDecoder::from_tensors(test_weights(), CausalDecoderConfig::default())
        .expect("invariant: poids cohérents");
    let generated = model
        .generate_greedy_with_options(
            &[0],
            4,
            &GenerationOptions {
                stop_token_ids: vec![0],
                ..GenerationOptions::default()
            },
        )
        .expect("invariant: greedy avec stop valide");
    assert!(generated.is_empty());
}

fn test_weights() -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("invariant: embedding valide"),
    );
    tensors.insert(
        "layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    for prefix in [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
    ] {
        tensors.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
            .expect("invariant: lm_head valide"),
    );
    tensors
}

fn hybrid_weights(layer_count: usize, full_attention_interval: usize) -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("invariant: embedding valide"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
            .expect("invariant: lm_head valide"),
    );
    for layer in 0..layer_count {
        if (layer + 1) % full_attention_interval == 0 {
            insert_attention_layer(&mut tensors, layer);
        } else {
            insert_linear_attention_layer(&mut tensors, layer);
        }
    }
    tensors
}

fn hybrid_config(layer_count: usize) -> CausalDecoderConfig {
    CausalDecoderConfig {
        num_hidden_layers: layer_count,
        full_attention_interval: Some(2),
        linear_num_key_heads: Some(1),
        linear_num_value_heads: Some(1),
        linear_key_head_dim: Some(2),
        linear_value_head_dim: Some(2),
        linear_conv_kernel_dim: Some(2),
        ..CausalDecoderConfig::default()
    }
}

fn gated_attention_weights() -> HashMap<String, Tensor> {
    let mut tensors = test_weights();
    tensors.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        Tensor::from_vec(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0])
            .expect("invariant: q_proj gated valide"),
    );
    tensors
}

fn insert_attention_layer(tensors: &mut HashMap<String, Tensor>, layer: usize) {
    tensors.insert(
        format!("layers.{layer}.input_layernorm.weight"),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    for prefix in [
        format!("layers.{layer}.self_attn.q_proj"),
        format!("layers.{layer}.self_attn.k_proj"),
        format!("layers.{layer}.self_attn.v_proj"),
        format!("layers.{layer}.self_attn.o_proj"),
    ] {
        tensors.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
}

fn insert_linear_attention_layer(tensors: &mut HashMap<String, Tensor>, layer: usize) {
    tensors.insert(
        format!("layers.{layer}.input_layernorm.weight"),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_qkv.weight"),
        Tensor::from_vec(
            vec![6, 2],
            vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0],
        )
        .expect("invariant: qkv valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_z.weight"),
        identity2().expect("invariant: z valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_b.weight"),
        Tensor::from_vec(vec![1, 2], vec![1.0, 0.0]).expect("invariant: b valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.in_proj_a.weight"),
        Tensor::from_vec(vec![1, 2], vec![0.0, 1.0]).expect("invariant: a valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.out_proj.weight"),
        identity2().expect("invariant: out valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.conv1d.weight"),
        Tensor::from_vec(
            vec![6, 2, 1],
            vec![
                0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0,
            ],
        )
        .expect("invariant: conv valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.A_log"),
        Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: A_log valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.dt_bias"),
        Tensor::from_vec(vec![1], vec![0.0]).expect("invariant: dt_bias valide"),
    );
    tensors.insert(
        format!("layers.{layer}.linear_attn.norm.weight"),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm lin valide"),
    );
}

fn insert_qk_norm(tensors: &mut HashMap<String, Tensor>, layer: usize, q: Vec<f32>, k: Vec<f32>) {
    tensors.insert(
        format!("layers.{layer}.self_attn.q_norm.weight"),
        Tensor::from_vec(vec![q.len()], q).expect("invariant: q_norm valide"),
    );
    tensors.insert(
        format!("layers.{layer}.self_attn.k_norm.weight"),
        Tensor::from_vec(vec![k.len()], k).expect("invariant: k_norm valide"),
    );
}

fn insert_identity_mlp(tensors: &mut HashMap<String, Tensor>) {
    tensors.insert(
        "layers.0.post_attention_layernorm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm valide"),
    );
    for prefix in ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            identity2().expect("invariant: identité valide"),
        );
    }
    tensors.insert(
        "layers.0.mlp.down_proj.weight".to_string(),
        identity2().expect("invariant: identité valide"),
    );
}

fn gqa_weights() -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .expect("invariant: embedding GQA valide"),
    );
    tensors.insert(
        "layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).expect("invariant: norm GQA valide"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0])
            .expect("invariant: norm finale GQA valide"),
    );
    tensors.insert(
        "layers.0.self_attn.q_proj.weight".to_string(),
        identity4().expect("invariant: q_proj GQA valide"),
    );
    for prefix in ["layers.0.self_attn.k_proj", "layers.0.self_attn.v_proj"] {
        tensors.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .expect("invariant: projection KV GQA valide"),
        );
    }
    tensors.insert(
        "layers.0.self_attn.o_proj.weight".to_string(),
        identity4().expect("invariant: o_proj GQA valide"),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .expect("invariant: lm_head GQA valide"),
    );
    tensors
}

fn identity2() -> Result<Tensor> {
    Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
}

fn identity4() -> Result<Tensor> {
    Tensor::from_vec(
        vec![4, 4],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tolerance,
            "index={idx} left={a} right={b} tolerance={tolerance}"
        );
    }
}

struct F32View {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for F32View {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn tensor_views(tensors: HashMap<String, Tensor>) -> Vec<(String, F32View)> {
    tensors
        .into_iter()
        .map(|(name, tensor)| {
            let data = tensor
                .data()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            (
                name,
                F32View {
                    shape: tensor.shape().to_vec(),
                    data,
                },
            )
        })
        .collect()
}

// --- 1b.3 : oracle du decode full-attn résident (GPU) vs CPU ---

/// Modèle GQA synthétique (q=2, kv=1, hd=2) **métal-activé**, ou `None` si
/// aucun device Metal (skip propre, comme les tests `metal_backend`).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn resident_test_model() -> Option<CausalDecoder> {
    let config = CausalDecoderConfig {
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(2),
        rope_theta: Some(10_000.0),
        ..CausalDecoderConfig::default()
    };
    let model =
        CausalDecoder::from_tensors(gqa_weights(), config).expect("invariant: modèle GQA valide");
    match model.with_metal_runtime() {
        Ok(model) => Some(model),
        Err(InferError::Metal(message)) if message.contains("aucun device") => None,
        Err(error) => panic!("runtime Metal indisponible: {error:?}"),
    }
}

/// Décode `n` tokens en greedy T=0, chemin résident forcé (`resident`) ou CPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn greedy_resident_ab(
    model: &CausalDecoder,
    prompt: &[usize],
    n: usize,
    resident: bool,
) -> Vec<usize> {
    let options = GenerationOptions::default();
    let mut sampler = DeterministicSampler::new(options.seed);
    let (mut cache, final_state) = model
        .prefill_cache_state_tokenwise(prompt)
        .expect("invariant: prefill valide");
    if resident {
        model
            .setup_resident_decode(&mut cache, n + 1)
            .expect("invariant: setup résident valide");
    }
    let mut tokens = Vec::with_capacity(n);
    let mut token = model
        .sample_token_from_state(&final_state, &options, &mut sampler)
        .expect("invariant: sampling valide");
    for _ in 0..n {
        tokens.push(token);
        let state = model
            .next_final_state_cached(&mut cache, token)
            .expect("invariant: decode valide");
        token = model
            .sample_token_from_state(&state, &options, &mut sampler)
            .expect("invariant: sampling valide");
    }
    tokens
}

/// 1b.3 / R2 — le clone du cache (prefix-cache) DROP l'état Metal résident
/// (`full` → None) sans toucher l'original (pas de double-ownership MTLBuffer).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_cache_clone_drops_metal() {
    let Some(model) = resident_test_model() else {
        return;
    };
    let prompt = [0_usize, 1];
    let (mut cache, _) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill valide");
    model
        .setup_resident_decode(&mut cache, 16)
        .expect("invariant: setup résident valide");
    assert!(
        cache.layers.iter().any(|layer| layer.full.is_some()),
        "le setup résident doit poser au moins un KV full-attn"
    );
    let clone = cache.clone();
    assert!(
        clone.layers.iter().all(|layer| layer.full.is_none()),
        "le clone doit dropper l'état Metal résident (full = None)"
    );
    assert!(
        cache.layers.iter().any(|layer| layer.full.is_some()),
        "l'original conserve son KV résident après le clone"
    );
}

/// 1b.3 — oracle teacher-forced : sur > 256 tokens (KV croissant, au-delà du
/// plafond 256 du kernel de prefill), l'état final par token du chemin
/// résident GPU est numériquement identique au chemin CPU (tolérance, réserve
/// E : le kernel change l'ordre de réduction f32).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_decode_state_matches_cpu_over_long_kv() {
    let Some(model) = resident_test_model() else {
        return;
    };
    let prompt = [0_usize, 1];
    let n = 320; // > 256
    let feed: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let (mut cache_off, _) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill OFF");
    let (mut cache_on, _) = model
        .prefill_cache_state_tokenwise(&prompt)
        .expect("invariant: prefill ON");
    model
        .setup_resident_decode(&mut cache_on, n + 1)
        .expect("invariant: setup résident");
    let mut max_abs = 0.0_f32;
    for &token in &feed {
        let off = model
            .next_final_state_cached(&mut cache_off, token)
            .expect("invariant: decode OFF");
        let on = model
            .next_final_state_cached(&mut cache_on, token)
            .expect("invariant: decode ON");
        assert_eq!(off.shape(), on.shape());
        for (a, b) in off.data().iter().zip(on.data()) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    // Résidu mesuré : max_abs ≈ 6.6e-7 sur 320 tokens (l'erreur f32 de l'ordre
    // de réduction ne diverge pas). Borne = garde-fou de régression robuste.
    assert!(
        max_abs <= 1.0e-4,
        "état résident vs CPU: max_abs={max_abs:e} sur {n} tokens"
    );
}

/// 1b.3 — oracle greedy : séquences T=0 token-identiques OFF vs ON, sur ≥256
/// tokens, pour deux prompts (multi-tour à cache frais → reset propre).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_decode_greedy_tokens_match_cpu() {
    let Some(model) = resident_test_model() else {
        return;
    };
    for prompt in [vec![0_usize, 1], vec![1_usize, 0, 1]] {
        let n = 256;
        let off = greedy_resident_ab(&model, &prompt, n, false);
        let on = greedy_resident_ab(&model, &prompt, n, true);
        assert_eq!(off, on, "greedy OFF vs ON divergent (prompt {prompt:?})");
    }
}

// --- 1c.0 : ossature decode résident complet (préconditions + délégation) ---

/// 1c.0 / MAJEUR 6 — la validation des préconditions REFUSE un modèle non
/// supporté en résident (ici le modèle GQA de test n'a pas de MoE/shared-expert)
/// → tout-ou-rien : on ne démarre jamais le command buffer unique sans garantie.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn resident_full_precondition_rejects_model_without_moe() {
    let Some(model) = resident_test_model() else {
        return;
    };
    assert!(
        !model.supports_resident_full_decode(),
        "un modèle sans MoE/shared-expert ne doit PAS être déclaré supporté en résident"
    );
}

// NOTE: l'ancien test d'ossature 1c.0 (`next_final_state_resident` déléguant au
// per-op) est retiré : le corps résident est désormais l'assemblage complet 1c
// (`decode_token_resident`), non instanciable par le modèle synthétique de test
// (sans MoE → `supports_resident_full_decode()=false`). La validation 1c est
// l'oracle 35B réel (texte greedy ==CPU + A/B), cf. /tmp/rust_infer_1c4_handoff.md.
