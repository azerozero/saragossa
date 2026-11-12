//! Parcours serve CPU avec le vrai tokenizer Gemma 3 et des poids synthétiques.
//! Aucun poids de modèle, téléchargement ni executor Metal.

use super::json_tests::tiny_loaded_model;
use super::*;
use saragossa::{CausalDecoderConfig, ModelConfig, RustTokenizer, Tensor};
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn local_tokenizer() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SARAGOSSA_TOKENIZER_GEMMA3") {
        return Some(path.into());
    }
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|p| PathBuf::from(p).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache/huggingface/hub"))
        })?;
    let path = hub.join(concat!(
        "models--mlx-community--gemma-3-4b-it-4bit/snapshots/",
        "93724907d4ed1745d2fe50baadf3b0b01a65abf2/tokenizer.json"
    ));
    path.is_file().then_some(path)
}

/// L'embedding identifie le dernier token ; la tête propose son successeur.
/// BOS et EOT ont des logits supérieurs : seule la vraie contrainte JSON
/// permet d'atteindre l'objet complet. Aucun remplacement du sampler/masque.
fn scripted_gemma(path: &Path, pieces: &[&str]) -> (LoadedModel, usize) {
    let tokenizer =
        RustTokenizer::from_file(path).expect("invariant: tokenizer Gemma local lisible");
    let ids: Vec<usize> = pieces
        .iter()
        .map(|piece| {
            tokenizer
                .token_to_id(piece)
                .expect("invariant: token Gemma présent") as usize
        })
        .collect();
    assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
    let vocab = tokenizer.vocab_size();
    let hidden = (ids.len() + 1).next_multiple_of(2);
    let mut config = ModelConfig::from_file(path.with_file_name("config.json"))
        .expect("invariant: config Gemma locale lisible");
    assert!(matches!(
        config.model_type.as_str(),
        "gemma3" | "gemma3_text"
    ));
    config.hidden_size = hidden;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = Some(hidden);
    config.vocab_size = vocab;
    let mut embedding = vec![0.0; vocab * hidden];
    for row in embedding.chunks_exact_mut(hidden) {
        row[0] = 1.0;
    }
    let mut head = vec![0.0; vocab * hidden];
    for (step, &id) in ids.iter().enumerate() {
        embedding[id * hidden] = 0.0;
        embedding[id * hidden + step + 1] = 1.0;
        head[id * hidden + step] = 5.0;
    }
    for (piece, logit) in [("<bos>", 20.0), ("<end_of_turn>", 10.0), ("<eos>", 9.0)] {
        let id = tokenizer
            .token_to_id(piece)
            .expect("invariant: token spécial Gemma") as usize;
        head[id * hidden..(id + 1) * hidden].fill(logit);
    }
    let mut weights = HashMap::new();
    weights.insert(
        "embed_tokens.weight".into(),
        Tensor::from_vec(vec![vocab, hidden], embedding).expect("invariant: embedding synthétique"),
    );
    weights.insert(
        "lm_head.weight".into(),
        Tensor::from_vec(vec![vocab, hidden], head).expect("invariant: tête synthétique"),
    );
    for name in ["norm.weight", "layers.0.input_layernorm.weight"] {
        weights.insert(
            name.into(),
            Tensor::from_vec(vec![hidden], vec![1.0; hidden])
                .expect("invariant: norme synthétique"),
        );
    }
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        weights.insert(
            format!("layers.0.self_attn.{projection}.weight"),
            Tensor::from_vec(vec![hidden, hidden], vec![0.0; hidden * hidden])
                .expect("invariant: attention nulle"),
        );
    }
    let decoder = CausalDecoder::from_tensors(weights, CausalDecoderConfig::from(&config))
        .expect("invariant: décodeur Gemma CPU synthétique");
    let mut loaded = tiny_loaded_model();
    loaded.assets.config = config;
    loaded.assets.tokenizer = tokenizer;
    loaded.decoder = decoder;
    (loaded, ids.len())
}

#[test]
fn gemma3_local_json_object_generation_and_streaming() {
    let Some(path) = local_tokenizer() else {
        eprintln!("SKIP: tokenizer Gemma 3 absent ; définir SARAGOSSA_TOKENIZER_GEMMA3");
        return;
    };
    for (pieces, expected) in [
        (
            vec!["▁{", "\"", "greeting", "\":", "▁\"", "Bonjour", "\"}"],
            "Bonjour",
        ),
        (
            vec![
                "▁{", "\"", "greeting", "\":", "▁\"", "Bonjour", "<0xC3>", "<0xA9>", "<0x22>",
                "<0x7D>",
            ],
            "Bonjouré",
        ),
    ] {
        let (mut loaded, length) = scripted_gemma(&path, &pieces);
        // Réutilise le même modèle/catalogue, mais exige un automate neuf par requête.
        for streaming in [false, true] {
            for budget in [length - 1, length + 2] {
                let request = ChatCompletionRequest::parse(
                    serde_json::json!({
                        "model": "tiny", "temperature": 0, "stream": streaming,
                        "max_tokens": budget, "response_format": {"type":"json_object"},
                        "messages": [{"role":"user", "content":"Return a JSON object with a greeting: \"Bonjour\"."}]
                    }).to_string().as_bytes()
                ).expect("invariant: requête test valide");
                let mut deltas = String::new();
                let mut terminal = false;
                let result = if streaming {
                    loaded.complete_streaming(
                        request,
                        budget,
                        None,
                        &MemoryGuard::serve(),
                        &mut |event| {
                            match event {
                                CompletionStreamEvent::Delta(delta) => deltas.push_str(delta),
                                CompletionStreamEvent::TerminalError(error) => {
                                    assert_eq!(error.error_type, "incomplete_json");
                                    terminal = true;
                                }
                                CompletionStreamEvent::Start(_) => {}
                            }
                            Ok(())
                        },
                    )
                } else {
                    loaded.complete(request, budget, None, &MemoryGuard::serve())
                };
                if budget < length {
                    assert!(
                        matches!(result, Err(ServeError::IncompleteJson(_))),
                        "{result:?}"
                    );
                    assert_eq!(terminal, streaming);
                } else {
                    let completion = result.expect("invariant: objet fermé avant EOT");
                    let value: serde_json::Value = serde_json::from_str(&completion.content)
                        .expect("invariant: contenu JSON valide");
                    assert_eq!(value, serde_json::json!({"greeting": expected}));
                    assert_eq!(completion.finish_reason, "stop");
                    assert!(!terminal);
                    if streaming {
                        assert_eq!(deltas, completion.content);
                    }
                }
            }
        }
    }
}
