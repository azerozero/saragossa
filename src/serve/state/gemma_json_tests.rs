//! Parcours serve CPU avec le vrai tokenizer Gemma 3 et des poids synthétiques.
//! Aucun poids de modèle, téléchargement ni executor Metal.

use super::json_tests::tiny_loaded_model;
use super::*;
use saragossa::{CausalDecoderConfig, ModelConfig, RustTokenizer, Tensor, TokenConstraint};
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

#[test]
fn gemma3_local_first_json_mask_matches_tokenizer_and_json_parser() {
    let Some(path) = local_tokenizer() else {
        eprintln!("SKIP: tokenizer Gemma 3 absent ; définir SARAGOSSA_TOKENIZER_GEMMA3");
        return;
    };
    let tokenizer = RustTokenizer::from_file(&path).expect("invariant: tokenizer local");
    let reference = tokenizers::Tokenizer::from_file(&path).expect("invariant: tokenizer oracle");
    let catalog = Arc::new(JsonTokenCatalog::from_tokenizer(&tokenizer).expect("catalogue"));
    let stops: Vec<usize> = ["<eos>", "<end_of_turn>"]
        .iter()
        .map(|s| tokenizer.token_to_id(s).expect("token stop") as usize)
        .collect();
    let constraint = JsonTokenConstraint::new(Arc::clone(&catalog), &stops, false);
    let mut logits = vec![0.0; tokenizer.vocab_size()];
    constraint.mask_logits(&mut logits).expect("masque initial");
    let mut allowed = Vec::new();
    for (id, logit) in logits.iter().enumerate() {
        let id = id as u32;
        let surface = reference.id_to_token(id).expect("token du vocabulaire");
        // Oracle indépendant du catalogue : décodage HF et octets fallback
        // bruts (un octet UTF-8 isolé ne doit pas devenir U+FFFD).
        let bytes = if surface.starts_with("<0x") && surface.len() == 6 {
            vec![u8::from_str_radix(&surface[3..5], 16).expect("octet fallback")]
        } else {
            reference
                .decode(&[id], true)
                .expect("décodage HF")
                .into_bytes()
        };
        assert_eq!(
            tokenizer.decode_token_bytes(id).expect("octets"),
            bytes,
            "{surface:?}"
        );
        let first = bytes
            .iter()
            .find(|b| !matches!(b, b' ' | b'\n' | b'\r' | b'\t'));
        let expected = !bytes.is_empty()
            && (first.is_none()
                || (first == Some(&b'{')
                    && match serde_json::from_slice::<serde_json::Value>(&bytes) {
                        Ok(value) => value.is_object(),
                        Err(error) => error.is_eof(),
                    }));
        assert_eq!(
            logit.is_finite(),
            expected,
            "id={id} surface={surface:?} bytes={bytes:?}"
        );
        if expected {
            allowed.push((id, surface));
        }
    }
    eprintln!(
        "Gemma premier pas : {} tokens autorisés (oracle HF + serde_json)",
        allowed.len()
    );
    for (id, surface) in &allowed {
        if surface.contains('{') || surface == "<0x7B>" {
            eprintln!("  id={id} surface={surface:?}");
        }
    }
    for piece in ["{", "{\"", "▁{", "▁{\"", "{}", "▁{}", "<0x7B>", "<0x20>"] {
        assert!(
            logits[tokenizer.token_to_id(piece).expect("token JSON") as usize].is_finite(),
            "{piece:?}"
        );
    }
    // Certains tokenizers fusionnent l'accolade et le saut de ligne ; Gemma
    // peut les encoder séparément. Chaque morceau doit rester admissible.
    for text in ["{\"", "{\n", " {\"", "{\n  \"greeting\":"] {
        let constraint = JsonTokenConstraint::new(Arc::clone(&catalog), &stops, false);
        for id in tokenizer.encode(text).expect("encodage préfixe") {
            let mut logits = vec![0.0; tokenizer.vocab_size()];
            constraint.mask_logits(&mut logits).expect("masque préfixe");
            assert!(logits[id as usize].is_finite(), "{text:?}: token {id}");
            constraint
                .accept_token(id as usize)
                .expect("préfixe valide");
        }
    }
    constraint
        .accept_token(tokenizer.token_to_id("{}").expect("objet vide") as usize)
        .expect("fermeture racine");
    let mut logits = vec![100.0; tokenizer.vocab_size()];
    for &stop in &stops {
        logits[stop] = -10.0;
    }
    constraint
        .mask_logits(&mut logits)
        .expect("masque terminal Gemma");
    let remaining: Vec<_> = logits
        .iter()
        .enumerate()
        .filter_map(|(id, logit)| logit.is_finite().then_some(id))
        .collect();
    assert_eq!(remaining, stops, "seuls EOS et EOT restent admissibles");
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
    // Après le dernier morceau, les espaces dominent même EOT et bouclent
    // sur eux-mêmes. Le masque doit les interdire dès la fermeture racine.
    for piece in ["\n\n", "▁"] {
        let id = tokenizer.token_to_id(piece).expect("espace Gemma") as usize;
        assert!(!ids.contains(&id));
        embedding[id * hidden] = 0.0;
        embedding[id * hidden + ids.len()] = 1.0;
        head[id * hidden + ids.len()] = 30.0;
    }
    // L'objet vide est un concurrent valide au premier pas, mais ne doit
    // pas gagner face à un préfixe non vide dont le logit est supérieur.
    let empty = tokenizer.token_to_id("{}").expect("objet vide Gemma") as usize;
    if !ids.contains(&empty) {
        head[empty * hidden] = 4.0;
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
        (vec!["{}"], serde_json::json!({})),
        (
            vec!["{\"", "greeting", "\":", "▁\"", "Bonjour", "\"}"],
            serde_json::json!({"greeting": "Bonjour"}),
        ),
        (
            // Le prompt se termine déjà par le token "\n" : son embedding
            // doit garder l'état initial. L'octet fallback produit le même
            // préfixe "{\n" sans aliaser un état du script avec le prompt.
            vec![
                "{", "<0x0A>", "\"", "greeting", "\":", "▁\"", "Bonjour", "\"}",
            ],
            serde_json::json!({"greeting": "Bonjour"}),
        ),
        (
            vec!["▁{", "\"", "greeting", "\":", "▁\"", "Bonjour", "\"}"],
            serde_json::json!({"greeting": "Bonjour"}),
        ),
        (
            vec![
                "▁{", "\"", "greeting", "\":", "▁\"", "Bonjour", "<0xC3>", "<0xA9>", "<0x22>",
                "<0x7D>",
            ],
            serde_json::json!({"greeting": "Bonjouré"}),
        ),
    ] {
        let (mut loaded, length) = scripted_gemma(&path, &pieces);
        // Réutilise le même modèle/catalogue, mais exige un automate neuf par requête.
        for streaming in [false, true] {
            let mut budgets = vec![256, length];
            if length > 1 {
                budgets.push(length - 1);
            }
            for budget in budgets {
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
                    assert_eq!(value, expected);
                    assert_eq!(completion.finish_reason, "stop");
                    assert_eq!(completion.content, completion.content.trim_end());
                    let usage = serde_json::to_value(completion.usage).expect("usage JSON");
                    assert_eq!(usage["completion_tokens"], length);
                    assert!(!terminal);
                    if streaming {
                        assert_eq!(deltas, completion.content);
                    }
                }
            }
        }
    }
}
