/// Complète une config Gemma 3 avec les défauts de mlx_lm.
///
/// Les configs mlx-community des Gemma 3 multimodaux (4B/12B/27B) sont
/// minimales : leur `text_config` n'énumère que les clés hors-défaut
/// (hidden_size, intermediate_size, num_hidden_layers, rope_scaling,
/// sliding_window). mlx_lm reconstruit le reste à deux niveaux — le wrapper
/// `gemma3.py` (vocab 262208 écrasant celui du text_config, 8 têtes Q / 4 KV)
/// puis `gemma3_text.ModelArgs` (head_dim 256, rope_theta 1e6, base locale
/// 1e4, query_pre_attn_scalar 256, motif sliding 6, GeLU tanh câblé en dur).
/// Sans ces défauts, le parsing échoue (champs requis absents) ou le forward
/// diverge silencieusement (base locale, motif sliding, activation).
pub(super) fn apply_gemma3_defaults(
    map: &mut serde_json::Map<String, serde_json::Value>,
    top_vocab_size: Option<serde_json::Value>,
) {
    let model_type = map.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let multimodal = model_type == "gemma3";
    if !multimodal && model_type != "gemma3_text" {
        return;
    }
    if multimodal {
        // gemma3.py __post_init__ : le vocab du wrapper (top-level, défaut
        // 262208) ÉCRASE celui du text_config ; têtes Q/KV par défaut.
        let vocab = top_vocab_size.unwrap_or_else(|| serde_json::json!(262_208));
        map.insert("vocab_size".to_string(), vocab);
        map.entry("num_attention_heads".to_string())
            .or_insert(serde_json::json!(8));
        map.entry("num_key_value_heads".to_string())
            .or_insert(serde_json::json!(4));
    }
    // gemma3_text.ModelArgs : défauts du tronc texte (valeurs du 1B) ;
    // l'activation est `gelu_approx` câblée en dur dans le MLP de mlx_lm.
    for (key, value) in [
        ("hidden_size", serde_json::json!(1152)),
        ("num_hidden_layers", serde_json::json!(26)),
        ("intermediate_size", serde_json::json!(6912)),
        ("num_attention_heads", serde_json::json!(4)),
        ("num_key_value_heads", serde_json::json!(1)),
        ("head_dim", serde_json::json!(256)),
        ("rms_norm_eps", serde_json::json!(1.0e-6)),
        ("vocab_size", serde_json::json!(262_144)),
        ("rope_theta", serde_json::json!(1_000_000.0)),
        ("rope_local_base_freq", serde_json::json!(10_000.0)),
        ("query_pre_attn_scalar", serde_json::json!(256)),
        ("sliding_window", serde_json::json!(512)),
        ("sliding_window_pattern", serde_json::json!(6)),
        ("hidden_activation", serde_json::json!("gelu_pytorch_tanh")),
    ] {
        map.entry(key.to_string()).or_insert(value);
    }
}

/// Complète une config Gemma 4 avec les défauts et alias de mlx_lm/HF.
///
/// Gemma 4 sépare les paramètres RoPE par type de couche et nomme le top-k MoE
/// `top_k_experts`. Le décodeur Saragossa garde une config normalisée plate :
/// cette fonction conserve les champs d'origine tout en ajoutant les alias
/// consommés par le loader et le forward.
pub(super) fn apply_gemma4_defaults(
    map: &mut serde_json::Map<String, serde_json::Value>,
    top_vocab_size: Option<serde_json::Value>,
) {
    let model_type = map.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    if !matches!(
        model_type,
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text"
    ) {
        return;
    }

    let vocab = top_vocab_size.unwrap_or_else(|| serde_json::json!(262_144));
    map.entry("vocab_size".to_string()).or_insert(vocab);
    for (key, value) in [
        ("hidden_size", serde_json::json!(1536)),
        ("num_hidden_layers", serde_json::json!(35)),
        ("intermediate_size", serde_json::json!(6144)),
        ("num_attention_heads", serde_json::json!(8)),
        ("num_key_value_heads", serde_json::json!(1)),
        ("head_dim", serde_json::json!(256)),
        ("global_head_dim", serde_json::json!(512)),
        ("rms_norm_eps", serde_json::json!(1.0e-6)),
        ("rope_theta", serde_json::json!(1_000_000.0)),
        ("rope_local_base_freq", serde_json::json!(10_000.0)),
        ("rope_full_partial_rotary_factor", serde_json::json!(0.25)),
        ("rope_sliding_partial_rotary_factor", serde_json::json!(1.0)),
        ("sliding_window", serde_json::json!(512)),
        ("sliding_window_pattern", serde_json::json!(5)),
        ("hidden_activation", serde_json::json!("gelu_pytorch_tanh")),
        ("final_logit_softcapping", serde_json::json!(30.0)),
        ("tie_word_embeddings", serde_json::json!(true)),
    ] {
        map.entry(key.to_string()).or_insert(value);
    }

    if let Some(top_k) = map.get("top_k_experts").cloned() {
        map.entry("num_experts_per_tok".to_string())
            .or_insert(top_k);
    }

    let Some(rope_parameters) = map
        .get("rope_parameters")
        .and_then(|value| value.as_object())
        .cloned()
    else {
        return;
    };
    if let Some(full) = rope_parameters
        .get("full_attention")
        .and_then(|value| value.as_object())
    {
        if let Some(theta) = full.get("rope_theta").cloned() {
            map.insert("rope_theta".to_string(), theta.clone());
            map.entry("rope_full_base_freq".to_string())
                .or_insert(theta);
        }
        if let Some(factor) = full.get("partial_rotary_factor").cloned() {
            map.entry("rope_full_partial_rotary_factor".to_string())
                .or_insert(factor);
        }
    }
    if let Some(sliding) = rope_parameters
        .get("sliding_attention")
        .and_then(|value| value.as_object())
    {
        if let Some(theta) = sliding.get("rope_theta").cloned() {
            map.entry("rope_local_base_freq".to_string())
                .or_insert(theta);
        }
        if let Some(factor) = sliding.get("partial_rotary_factor").cloned() {
            map.entry("rope_sliding_partial_rotary_factor".to_string())
                .or_insert(factor);
        }
    }
}
