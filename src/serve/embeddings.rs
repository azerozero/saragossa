//! Endpoint embeddings OpenAI-compatible de `saragossa serve`.
//!
//! `POST /v1/embeddings` : embedder sémantique e5-small (BERT 384-dim, Rust pur
//! f32 CPU). Requête JSON `{"model", "input"}` où `input` est une chaîne OU un
//! tableau de chaînes. Réponse :
//! `{"object":"list","data":[{"object":"embedding","index":i,"embedding":[…]}],"model":…}`.
//!
//! Le modèle est opt-in (`--embed-model <dir>`) et chargé paresseusement au
//! premier appel — même patron que les endpoints audio ; sans configuration, la
//! route renvoie une erreur 400 JSON claire.

use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use saragossa::{TextEmbedError, TextEmbedder, DEFAULT_TEXT_EMBED_REPO};

use super::args::ServeArgs;
use super::error::{ServeError, ServeResult};
use super::http::send_json;

const EMBED_NOT_CONFIGURED: &str =
    "endpoint embeddings non configuré: relancez saragossa serve avec --embed-model <dir>";

/// Etat embeddings du serveur : modèle e5-small chargé à la demande.
pub(super) struct EmbeddingsState {
    path: Option<PathBuf>,
    model: Option<TextEmbedder>,
}

impl EmbeddingsState {
    /// Construit l'état embeddings depuis le chemin CLI, sans charger les poids.
    pub(super) fn new(args: &ServeArgs) -> Self {
        Self {
            path: args.embed_model.clone(),
            model: None,
        }
    }

    /// Embede un passage et renvoie le vecteur L2-normalisé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le modèle n'est pas configuré ou si l'encodage échoue.
    pub(super) fn embed(&mut self, text: &str) -> ServeResult<Vec<f32>> {
        self.ensure_loaded()?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ServeError::args("modèle embeddings absent après chargement"))?;
        let vector = model.embed_passage(text).map_err(embed_error)?;
        Ok(vector.to_vec())
    }

    fn ensure_loaded(&mut self) -> ServeResult<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let path = self
            .path
            .clone()
            .ok_or_else(|| ServeError::args(EMBED_NOT_CONFIGURED))?;
        if !path.is_dir() {
            return Err(ServeError::args(format!(
                "dossier modèle embeddings introuvable: {}",
                path.display()
            )));
        }
        eprintln!(
            "saragossa serve loading embeddings model path={}",
            path.display()
        );
        let model = TextEmbedder::load_local(&path).map_err(embed_error)?;
        self.model = Some(model);
        Ok(())
    }
}

/// Mappe une [`TextEmbedError`] en [`ServeError`] (misconfiguration serveur → 400).
fn embed_error(error: TextEmbedError) -> ServeError {
    ServeError::args(format!("embeddings: {error}"))
}

/// Sert `POST /v1/embeddings`.
pub(super) fn handle_embeddings<S: Write>(
    stream: &mut S,
    state: &mut EmbeddingsState,
    body: &[u8],
) -> ServeResult<()> {
    let request: EmbeddingsRequest = serde_json::from_slice(body)
        .map_err(|e| ServeError::json("désérialisation embeddings", e))?;
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_TEXT_EMBED_REPO.to_string());
    let texts = request.input.into_texts();
    if texts.is_empty() {
        return Err(ServeError::Http(
            "champ 'input' vide pour embeddings".to_string(),
        ));
    }
    let mut data = Vec::with_capacity(texts.len());
    for (index, text) in texts.iter().enumerate() {
        let embedding = state.embed(text)?;
        data.push(EmbeddingData {
            object: "embedding",
            index,
            embedding,
        });
    }
    let response = EmbeddingsResponse {
        object: "list",
        data,
        model,
    };
    send_json(stream, 200, &response, Vec::new())
}

/// Requête `/v1/embeddings`.
#[derive(Debug, Deserialize)]
struct EmbeddingsRequest {
    /// Identifiant de modèle renvoyé tel quel (le serveur en expose un seul).
    #[serde(default)]
    model: Option<String>,
    /// Texte(s) à embeder : une chaîne ou un tableau de chaînes.
    input: EmbeddingInput,
}

/// Champ `input` OpenAI : chaîne unique ou tableau de chaînes.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    /// Une seule chaîne.
    One(String),
    /// Plusieurs chaînes.
    Many(Vec<String>),
}

impl EmbeddingInput {
    fn into_texts(self) -> Vec<String> {
        match self {
            EmbeddingInput::One(text) => vec![text],
            EmbeddingInput::Many(texts) => texts,
        }
    }
}

/// Réponse `/v1/embeddings`.
#[derive(Debug, Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
}

/// Un vecteur d'embedding dans la réponse.
#[derive(Debug, Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::super::args::ServeArgs;
    use super::*;

    #[test]
    fn input_accepts_string_or_array() {
        let one: EmbeddingsRequest =
            serde_json::from_str(r#"{"model":"e5","input":"bonjour"}"#).expect("invariant: string");
        assert_eq!(one.input.into_texts(), vec!["bonjour".to_string()]);

        let many: EmbeddingsRequest =
            serde_json::from_str(r#"{"input":["a","b"]}"#).expect("invariant: array");
        assert_eq!(
            many.input.into_texts(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn response_serializes_openai_shape() {
        let response = EmbeddingsResponse {
            object: "list",
            data: vec![EmbeddingData {
                object: "embedding",
                index: 0,
                embedding: vec![0.1, 0.2],
            }],
            model: "e5".to_string(),
        };
        let value: serde_json::Value =
            serde_json::to_value(&response).expect("invariant: sérialisable");
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"][0]["object"], "embedding");
        assert_eq!(value["data"][0]["index"], 0);
        let second = value["data"][0]["embedding"][1]
            .as_f64()
            .expect("invariant: composante numérique");
        assert!((second - 0.2).abs() < 1e-6, "composante {second}");
        assert_eq!(value["model"], "e5");
    }

    #[test]
    fn embeddings_without_model_returns_clear_error() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = EmbeddingsState::new(&args);
        let mut stream = Cursor::new(Vec::new());
        let body = br#"{"model":"e5","input":"bonjour"}"#;

        let error = handle_embeddings(&mut stream, &mut state, body)
            .expect_err("invariant: embeddings non configuré refusé");

        assert!(error.to_string().contains("embeddings non configuré"));
    }

    #[test]
    fn embeddings_rejects_empty_array() {
        let args = ServeArgs::parse(Vec::<String>::new()).expect("invariant: args valides");
        let mut state = EmbeddingsState::new(&args);
        let mut stream = Cursor::new(Vec::new());
        let body = br#"{"input":[]}"#;

        let error = handle_embeddings(&mut stream, &mut state, body)
            .expect_err("invariant: input vide refusé");

        assert!(error.to_string().contains("'input' vide"));
    }
}
