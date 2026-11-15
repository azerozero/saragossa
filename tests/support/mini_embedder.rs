//! BERT synthétique à une couche : deux directions orthogonales, aucun poids réel.
use std::collections::BTreeMap;
use std::path::Path;

pub fn write_model(path: &Path) {
    const H: usize = 384;
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut add = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
        tensors.insert(
            name.into(),
            (shape, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        );
    };
    let direction = |kind: usize| -> Vec<f32> {
        (0..H)
            .map(|d| if (d >> kind) % 2 == 0 { 1.0 } else { -1.0 })
            .collect()
    };
    let a = direction(0);
    let b = direction(1);
    let words = [vec![0.0; H], a.clone(), b.clone(), a, b].concat();
    add("embeddings.word_embeddings.weight", vec![5, H], words);
    add(
        "embeddings.position_embeddings.weight",
        vec![16, H],
        vec![0.0; 16 * H],
    );
    add(
        "embeddings.token_type_embeddings.weight",
        vec![1, H],
        vec![0.0; H],
    );
    for prefix in [
        "embeddings.LayerNorm",
        "encoder.layer.0.attention.output.LayerNorm",
        "encoder.layer.0.output.LayerNorm",
    ] {
        add(&format!("{prefix}.weight"), vec![H], vec![1.0; H]);
        add(&format!("{prefix}.bias"), vec![H], vec![0.0; H]);
    }
    for (name, identity) in [
        ("attention.self.query", false),
        ("attention.self.key", false),
        ("attention.self.value", true),
        ("attention.output.dense", true),
    ] {
        let mut w = vec![0.0; H * H];
        if identity {
            for d in 0..H {
                w[d * H + d] = 1.0;
            }
        }
        let prefix = format!("encoder.layer.0.{name}");
        add(&format!("{prefix}.weight"), vec![H, H], w);
        add(&format!("{prefix}.bias"), vec![H], vec![0.0; H]);
    }
    add(
        "encoder.layer.0.intermediate.dense.weight",
        vec![1, H],
        vec![0.0; H],
    );
    add(
        "encoder.layer.0.intermediate.dense.bias",
        vec![1],
        vec![0.0],
    );
    add(
        "encoder.layer.0.output.dense.weight",
        vec![H, 1],
        vec![0.0; H],
    );
    add("encoder.layer.0.output.dense.bias", vec![H], vec![0.0; H]);
    let views: BTreeMap<_, _> = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name,
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("invariant: tenseur synthétique"),
            )
        })
        .collect();
    let bytes =
        safetensors::tensor::serialize(views, None).expect("invariant: safetensors synthétique");
    std::fs::write(path.join("model.safetensors"), bytes).expect("invariant: dossier temporaire");
    std::fs::write(path.join("config.json"), r#"{"hidden_size":384,"num_hidden_layers":1,"num_attention_heads":1,"layer_norm_eps":1e-12}"#)
        .expect("invariant: dossier temporaire");
    // WhitespaceSplit conserve le préfixe avec son deux-points comme un token.
    let tokenizer = serde_json::json!({
        "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
        "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
        "post_processor":null, "decoder":null,
        "model":{"type":"WordLevel", "vocab":{"[UNK]":0,"query:":1,"passage:":2,"rouge":3,"bleu":4},"unk_token":"[UNK]"}
    });
    std::fs::write(path.join("tokenizer.json"), tokenizer.to_string())
        .expect("invariant: dossier temporaire");
}

pub fn assert_golden(vector: &[f32], mixed: bool, passage: bool) {
    assert_eq!(vector.len(), 384);
    // Oracle analytique : LN conserve les directions, attention uniforme ;
    // moyenne de deux directions orthogonales = (a+b)/sqrt(2*384).
    for (d, &actual) in vector.iter().enumerate() {
        let a = if d % 2 == 0 { 1.0 } else { -1.0 };
        let b = if (d / 2) % 2 == 0 { 1.0 } else { -1.0 };
        let expected = if mixed {
            (a + b) / 768.0_f32.sqrt()
        } else {
            (if passage { b } else { a }) / 384.0_f32.sqrt()
        };
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-6,
            "dimension {d}: {actual} != {expected}"
        );
    }
    let norm: f32 = vector.iter().map(|x| x * x).sum();
    assert!((norm - 1.0).abs() <= 1e-5, "norme carrée {norm}");
}

/// Trois tokens : une direction A puis deux B. La normalisation de chaque
/// ligne après attention rend cet oracle sensible à une couche sautée.
pub fn assert_uneven_golden(vector: &[f32]) {
    assert_eq!(vector.len(), 384);
    // Après attention uniforme + résidu + LN :
    // ligne A = (4A+2B)/sqrt(20), lignes B = (A+5B)/sqrt(26).
    let a = 4.0_f64 / 20.0_f64.sqrt() + 2.0 / 26.0_f64.sqrt();
    let b = 2.0_f64 / 20.0_f64.sqrt() + 10.0 / 26.0_f64.sqrt();
    let norm = (384.0 * (a * a + b * b)).sqrt();
    for (d, &actual) in vector.iter().enumerate() {
        let sign_a = if d % 2 == 0 { 1.0 } else { -1.0 };
        let sign_b = if (d / 2) % 2 == 0 { 1.0 } else { -1.0 };
        let expected = ((sign_a * a + sign_b * b) / norm) as f32;
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-6,
            "dimension asymétrique {d}: {actual} != {expected}"
        );
    }
}
