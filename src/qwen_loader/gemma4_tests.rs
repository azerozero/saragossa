use super::gemma4_test_fixtures::*;
use super::*;

use crate::{render_gemma4_chat, ChatTemplateMessage, GenerationOptions, RawModelConfig};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[test]
fn resolves_gemma4_unified_dense_text_config_with_gemma4_flags() {
    let raw = r#"{
        "architectures":["Gemma4UnifiedForConditionalGeneration"],
        "model_type":"gemma4_unified",
        "text_config":{
            "model_type":"gemma4_unified_text",
            "hidden_size":3840,
            "num_hidden_layers":48,
            "num_attention_heads":16,
            "num_key_value_heads":8,
            "num_global_key_value_heads":1,
            "head_dim":256,
            "global_head_dim":512,
            "intermediate_size":15360,
            "rms_norm_eps":1e-06,
            "attention_k_eq_v":true,
            "enable_moe_block":false,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{
                "full_attention":{
                    "partial_rotary_factor":0.25,
                    "rope_theta":1000000.0
                },
                "sliding_attention":{
                    "rope_theta":10000.0
                }
            }
        }
    }"#;
    let cfg: RawModelConfig = serde_json::from_str(raw).expect("invariant: JSON valide");
    let cfg = cfg
        .resolve()
        .expect("invariant: config Gemma 4 unified valide");

    assert_eq!(cfg.model_type, "gemma4_unified");
    assert!(cfg.is_gemma4());
    assert!(!cfg.enable_moe_block);
    assert_eq!(cfg.num_experts, None);
    assert_eq!(cfg.num_experts_per_tok, None);
    assert!(cfg.tie_word_embeddings);
    assert_eq!(cfg.final_logit_softcapping, Some(30.0));
    assert_eq!(cfg.layer_head_dim(0), 256);
    assert_eq!(cfg.layer_head_dim(1), 512);
    assert_eq!(cfg.layer_num_key_value_heads(1), 1);
    assert_eq!(cfg.layer_rope_dims(1), 128);

    let decoder = crate::CausalDecoderConfig::from(&cfg);
    assert!(decoder.is_gemma4);
    assert!(!decoder.parallel_moe);
    assert!(decoder.attention_value_norm);
    assert_eq!(decoder.query_pre_attn_scalar, Some(1.0));
    assert_eq!(decoder.activation, crate::Activation::GeluTanh);
}

#[test]
fn loads_gemma4_moe_switch_glu_with_all_decoder_keys_mapped() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_gemma4_moe_safetensors(tmp.path());
    let config = gemma4_moe_config();
    let catalog = catalog_for(tmp.path());

    assert_all_decoder_keys_mapped(&config, &catalog);
    verify_decoder_contract_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: contrat Gemma 4 MoE tiny valide");

    let model = load_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: Gemma 4 MoE switch_glu chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward Gemma 4 MoE valide");

    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));

    let full = model
        .generate_greedy_full_with_options(&[0, 1], 3, &GenerationOptions::default())
        .expect("invariant: greedy full Gemma 4 MoE valide");
    let cached = model
        .generate_greedy_cached_with_options(&[0, 1], 3, &GenerationOptions::default())
        .expect("invariant: greedy cache Gemma 4 MoE valide");
    assert_eq!(cached, full);
}

#[test]
fn loads_gemma4_unified_dense_with_all_decoder_keys_mapped() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_gemma4_unified_dense_safetensors(tmp.path());
    let config = gemma4_unified_dense_config();
    let catalog = catalog_for(tmp.path());

    assert_all_decoder_keys_mapped(&config, &catalog);
    verify_decoder_contract_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: contrat Gemma 4 unified tiny valide");

    let decoder_config = crate::CausalDecoderConfig::from(&config);
    assert!(decoder_config.is_gemma4);
    assert!(!decoder_config.parallel_moe);
    assert!(decoder_config.attention_value_norm);
    assert_eq!(decoder_config.rope_full_dims, Some(2));
    assert_eq!(decoder_config.rope_sliding_dims, Some(2));
    assert_eq!(decoder_config.query_pre_attn_scalar, Some(1.0));
    assert_eq!(decoder_config.activation, crate::Activation::GeluTanh);

    let model = load_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: Gemma 4 unified dense chargeable");
    let logits = model
        .next_logits(&[0, 1])
        .expect("invariant: forward Gemma 4 unified valide");
    assert_eq!(logits.shape(), &[1, 3]);
    assert!(logits.data().iter().all(|value| value.is_finite()));
}

#[test]
fn gemma4_unified_tied_embeddings_match_explicit_lm_head() {
    let tied = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_gemma4_unified_dense_safetensors(tied.path());
    let tied_catalog = catalog_for(tied.path());
    let tied_config = gemma4_unified_dense_config();
    assert!(!tied_catalog.contains("language_model.lm_head.weight"));
    let tied_model =
        load_causal_decoder_from_shards(&tied_config, &[tied.path().to_path_buf()], &tied_catalog)
            .expect("invariant: modèle Gemma 4 unified lié chargeable");

    let explicit = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_gemma4_unified_dense_explicit_lm_head(explicit.path());
    let explicit_catalog = catalog_for(explicit.path());
    let mut explicit_config = gemma4_unified_dense_config();
    explicit_config.tie_word_embeddings = false;
    assert!(explicit_catalog.contains("language_model.lm_head.weight"));
    let explicit_tensors = load_decoder_tensors(
        &explicit_config,
        &[explicit.path().to_path_buf()],
        &explicit_catalog,
        &QwenPrefixes::detect(&explicit_catalog),
    )
    .expect("invariant: tenseurs explicites chargeables");
    let DecoderTensor::Dense(explicit_head) = explicit_tensors
        .get("lm_head.weight")
        .expect("invariant: lm_head explicite présent")
    else {
        panic!("invariant: lm_head explicite dense");
    };
    assert_eq!(explicit_head.shape(), &[3, 4]);
    assert_eq!(explicit_head.data(), EMBED_VALUES.as_slice());
    let explicit_model = load_causal_decoder_from_shards(
        &explicit_config,
        &[explicit.path().to_path_buf()],
        &explicit_catalog,
    )
    .expect("invariant: modèle Gemma 4 unified explicite chargeable");

    let tied_logits = tied_model
        .next_logits(&[0, 1])
        .expect("invariant: forward lié valide");
    let explicit_logits = explicit_model
        .next_logits(&[0, 1])
        .expect("invariant: forward explicite valide");
    assert_eq!(tied_logits.shape(), &[1, 3]);
    assert_eq!(tied_logits.data(), explicit_logits.data());
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn gemma4_moe_greedy_tokens_match_cpu_and_gpu() {
    let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
    write_gemma4_moe_safetensors(tmp.path());
    let config = gemma4_moe_config();
    let catalog = catalog_for(tmp.path());
    let cpu = load_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: modèle CPU chargeable");
    let gpu = match load_causal_decoder_from_shards(&config, &[tmp.path().to_path_buf()], &catalog)
        .expect("invariant: modèle GPU chargeable")
        .with_metal_runtime()
    {
        Ok(model) => model,
        Err(InferError::Metal(message)) if message.contains("aucun device") => return,
        Err(error) => panic!("runtime Metal indisponible: {error:?}"),
    };

    let options = GenerationOptions::default();
    let prompt = [0_usize, 1];
    let cpu_tokens = cpu
        .generate_greedy_cached_with_options(&prompt, 4, &options)
        .expect("invariant: greedy CPU valide");
    let gpu_tokens = gpu
        .generate_greedy_cached_with_options(&prompt, 4, &options)
        .expect("invariant: greedy GPU valide");
    assert_eq!(gpu_tokens, cpu_tokens);
}

#[test]
#[ignore = "requiert le snapshot HF local gemma-4-26b-a4b-it-4bit"]
fn generates_coherent_french_gemma4_moe() {
    let Some(model_dir) = real_gemma4_snapshot() else {
        eprintln!("snapshot Gemma 4 réel absent, test ignoré proprement");
        return;
    };
    let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets Gemma 4 lisibles");
    let contract =
        verify_decoder_contract(&assets).expect("invariant: contrat Gemma 4 réel valide");
    assert_eq!(contract.shard_count, 3);
    assert!(contract.required_specs > 0);
    assert!(contract.present_specs >= contract.required_specs);

    let decoder = load_causal_decoder(&assets).expect("invariant: Gemma 4 réel chargeable");
    let prompt = render_gemma4_chat(
        &[ChatTemplateMessage::new(
            "user",
            "Explique en français, en deux phrases, pourquoi le ciel est bleu.",
        )],
        true,
        false,
    );
    let prompt_ids = assets
        .encode_prompt(&prompt)
        .expect("invariant: prompt Gemma 4 encodable")
        .into_iter()
        .map(|id| usize::try_from(id).expect("invariant: id tokenizer représentable"))
        .collect::<Vec<_>>();
    let options = GenerationOptions {
        stop_token_ids: assets.stop_token_ids(),
        ..GenerationOptions::default()
    };
    let output = decoder
        .generate_greedy_timed_with_options(&prompt_ids, 40, &options)
        .expect("invariant: génération Gemma 4 réelle valide");
    let tokens = output
        .tokens
        .into_iter()
        .map(|id| u32::try_from(id).expect("invariant: id généré représentable"))
        .collect::<Vec<_>>();
    let text = assets
        .decode_tokens(&tokens, true)
        .expect("invariant: sortie Gemma 4 décodable");
    println!("gemma4_smoke_output={}", text.trim());
    assert!(text.chars().any(char::is_alphabetic));
    assert!(!text.contains('\u{FFFD}'));
}

#[test]
#[ignore = "requiert un snapshot HF local gemma4_unified coder"]
fn generates_if_present_gemma4_unified_dense() {
    let Some(model_dir) = real_gemma4_unified_snapshot() else {
        eprintln!("snapshot Gemma 4 unified coder absent, test ignoré proprement");
        return;
    };
    let assets =
        ModelAssets::load_local(&model_dir).expect("invariant: assets Gemma 4 unified lisibles");
    assert!(assets.config.is_gemma4());
    assert!(!assets.config.enable_moe_block);
    let contract =
        verify_decoder_contract(&assets).expect("invariant: contrat Gemma 4 unified valide");
    assert!(contract.required_specs > 0);
    assert!(contract.present_specs >= contract.required_specs);

    let decoder = load_causal_decoder(&assets).expect("invariant: Gemma 4 unified chargeable");
    let prompt = render_gemma4_chat(
        &[ChatTemplateMessage::new(
            "user",
            "Réponds en français en une phrase courte.",
        )],
        true,
        false,
    );
    let prompt_ids = assets
        .encode_prompt(&prompt)
        .expect("invariant: prompt Gemma 4 unified encodable")
        .into_iter()
        .map(|id| usize::try_from(id).expect("invariant: id tokenizer représentable"))
        .collect::<Vec<_>>();
    let options = GenerationOptions {
        stop_token_ids: assets.stop_token_ids(),
        ..GenerationOptions::default()
    };
    let output = decoder
        .generate_greedy_timed_with_options(&prompt_ids, 16, &options)
        .expect("invariant: génération Gemma 4 unified valide");
    assert!(!output.tokens.is_empty());
}

fn catalog_for(path: &Path) -> WeightCatalog {
    WeightCatalog::from_shards(&[path.to_path_buf()]).expect("invariant: catalog chargeable")
}

fn assert_all_decoder_keys_mapped(config: &ModelConfig, catalog: &WeightCatalog) {
    let prefixes = QwenPrefixes::detect(catalog);
    let specs = decoder_specs(config, &prefixes, catalog);
    let sources = specs
        .iter()
        .map(|spec| spec.source.as_str())
        .collect::<HashSet<_>>();
    for key in catalog.keys() {
        assert!(sources.contains(key.as_str()), "clé non mappée: {key}");
    }
}

fn real_gemma4_snapshot() -> Option<PathBuf> {
    let path = PathBuf::from(
        "/Users/ludwig/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/efbeee6e582ebfd06abc9d65e90839c4b5d2116b",
    );
    path.is_dir().then_some(path)
}

fn real_gemma4_unified_snapshot() -> Option<PathBuf> {
    [
        "/Users/ludwig/.cache/huggingface/hub/models--mlx-community--gemma-4-12b-coder-fable5-composer2.5-8bit/snapshots",
        "/Users/ludwig/.cache/huggingface/hub/models--mlx-community--gemma-4-12b-coder-fable5-composer2.5-4bit/snapshots",
    ]
    .into_iter()
    .find_map(first_snapshot_dir)
}

fn first_snapshot_dir(path: &str) -> Option<PathBuf> {
    let mut dirs = std::fs::read_dir(path)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    dirs.sort();
    dirs.into_iter().next()
}
