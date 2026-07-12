use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use saragossa::{
    CausalDecoder, CausalDecoderConfig, ModelAssets, ModelConfig, MtpWeightsInfo, RustTokenizer,
    Tensor, WeightCatalog,
};

use super::*;

#[test]
fn json_object_completion_is_parseable_with_tiny_model() {
    let mut loaded = tiny_loaded_model();
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "JSON"}],
        "max_tokens": 1,
        "response_format": {"type": "json_object"}
    }))
    .expect("invariant: requête test valide");

    let completion = loaded
        .complete(request, 1, &MemoryGuard::serve())
        .expect("invariant: complétion guidée valide");

    serde_json::from_str::<serde_json::Value>(&completion.content)
        .expect("invariant: sortie JSON parseable");
    assert_eq!(completion.content, "{}");
}

fn tiny_loaded_model() -> LoadedModel {
    let tmp = tempfile::tempdir().expect("invariant: tempdir");
    let model_dir = tmp.path().to_path_buf();
    let config_path = tmp.path().join("config.json");
    fs::write(
        &config_path,
        r#"{
          "model_type":"qwen3",
          "hidden_size":2,
          "num_hidden_layers":1,
          "num_attention_heads":1,
          "num_key_value_heads":1,
          "rms_norm_eps":1e-6,
          "rope_theta":10000.0,
          "vocab_size":3,
          "eos_token_id":2
        }"#,
    )
    .expect("invariant: config test écrite");
    let tokenizer_path = tmp.path().join("tokenizer.json");
    save_json_tokenizer(&tokenizer_path);
    let catalog_path = tmp.path().join("empty.safetensors");
    write_empty_safetensors_header(&catalog_path);
    let tokenizer = RustTokenizer::from_file(&tokenizer_path).expect("invariant: tokenizer test");
    let assets = ModelAssets {
        model_dir,
        config: ModelConfig::from_file(&config_path).expect("invariant: config test"),
        tokenizer,
        shards: vec![catalog_path.clone()],
        catalog: WeightCatalog::from_shards(&[catalog_path]).expect("invariant: catalogue test"),
        mtp: MtpWeightsInfo::default(),
    };
    LoadedModel {
        id: "tiny".to_string(),
        assets,
        decoder: CausalDecoder::from_tensors(tiny_weights(), CausalDecoderConfig::default())
            .expect("invariant: modèle tiny valide"),
        preset: None,
        json_token_catalog: OnceLock::new(),
        prefix_cache: BlockAwarePrefixCache::new(1, 0),
    }
}

fn save_json_tokenizer(path: &Path) {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer;

    let vocab_path = path.with_file_name("json-vocab.json");
    fs::write(&vocab_path, r#"{"{}":0,"<pad>":1,"<eos>":2}"#).expect("invariant: vocab test écrit");
    let vocab = vocab_path
        .to_str()
        .expect("invariant: chemin vocab UTF-8")
        .to_string();
    let model =
        WordLevel::from_file(&vocab, "{}".to_string()).expect("invariant: modèle WordLevel test");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .save(path, true)
        .expect("invariant: tokenizer test écrit");
}

fn write_empty_safetensors_header(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(b"{}");
    fs::write(path, bytes).expect("invariant: safetensors test écrit");
}

fn tiny_weights() -> HashMap<String, Tensor> {
    let mut tensors = HashMap::new();
    tensors.insert(
        "embed_tokens.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("invariant: embedding tiny"),
    );
    tensors.insert(
        "layers.0.input_layernorm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm tiny"),
    );
    tensors.insert(
        "norm.weight".to_string(),
        Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: norm finale tiny"),
    );
    for prefix in [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
    ] {
        tensors.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .expect("invariant: identité tiny"),
        );
    }
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
            .expect("invariant: lm head tiny"),
    );
    tensors
}
