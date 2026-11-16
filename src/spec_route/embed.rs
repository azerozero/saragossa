//! Classifieur code/prose sur embeddings `multilingual-e5-small` (opt-in
//! `SARAGOSSA_RUST_SPEC_ROUTE_EMBED_DIR=<snapshot local>`), pour router aussi
//! la prose non latine vers MTP. Aucun téléchargement : sans dossier valide,
//! la règle « écriture latine » reste en place.
//!
//! Exemples de référence par classe et par langue (`refs.json`), score =
//! moyenne des `K` meilleurs cosinus côté code moins côté prose. MTP seulement
//! si le score passe sous `-MARGIN` : l'ambigu reste à DFlash.

use std::sync::OnceLock;

use crate::text_embedder::{TextEmbedError, TextEmbedder, TEXT_EMBED_DIM};

const REFS: &str = include_str!("refs.json");
const K: usize = 5;
/// Écart de cosinus moyen en dessous duquel la consigne est jugée ambiguë.
const MARGIN: f32 = 0.01;
/// Seule la fin du message est encodée : la consigne y est, et le coût du
/// forward croît avec la longueur (512 tokens au plus pour E5).
const TAIL_CHARS: usize = 600;

type Embedding = [f32; TEXT_EMBED_DIM];

/// Embedder et références encodées une fois au chargement.
pub struct EmbedRouter {
    embedder: TextEmbedder,
    code: Vec<Embedding>,
    prose: Vec<Embedding>,
}

impl EmbedRouter {
    /// Charge l'embedder local et encode les références.
    ///
    /// # Errors
    ///
    /// Erreur de chargement de l'embedder, d'encodage ou de références mal formées.
    pub fn load(model_dir: &std::path::Path) -> Result<Self, TextEmbedError> {
        let embedder = TextEmbedder::load_local(model_dir)?;
        let refs: serde_json::Value = serde_json::from_str(REFS)
            .map_err(|error| TextEmbedError::Config(format!("refs.json : {error}")))?;
        let encode = |class: &str| -> Result<Vec<Embedding>, TextEmbedError> {
            let languages = refs
                .get(class)
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| {
                    TextEmbedError::Config(format!("refs.json : classe {class} absente"))
                })?;
            languages
                .values()
                .filter_map(serde_json::Value::as_array)
                .flatten()
                .filter_map(serde_json::Value::as_str)
                .map(|text| embedder.embed_query(text))
                .collect()
        };
        let code = encode("code")?;
        let prose = encode("prose")?;
        Ok(Self {
            embedder,
            code,
            prose,
        })
    }

    /// Score code moins prose, positif quand la consigne ressemble au code.
    ///
    /// # Errors
    ///
    /// Erreur d'encodage du texte.
    pub fn score(&self, text: &str) -> Result<f32, TextEmbedError> {
        let start = text
            .char_indices()
            .rev()
            .nth(TAIL_CHARS - 1)
            .map_or(0, |(index, _)| index);
        let query = self.embedder.embed_query(&text[start..])?;
        Ok(top_k_mean(&query, &self.code) - top_k_mean(&query, &self.prose))
    }

    /// Vrai seulement si la consigne est nettement de la prose ; une erreur
    /// d'encodage compte comme ambiguë.
    pub fn is_prose(&self, text: &str) -> bool {
        self.score(text).is_ok_and(|score| score < -MARGIN)
    }
}

fn top_k_mean(query: &Embedding, refs: &[Embedding]) -> f32 {
    let mut scores: Vec<f32> = refs
        .iter()
        .map(|reference| query.iter().zip(reference).map(|(a, b)| a * b).sum())
        .collect();
    scores.sort_unstable_by(|a, b| b.total_cmp(a));
    let top = &scores[..K.min(scores.len())];
    top.iter().sum::<f32>() / top.len().max(1) as f32
}

/// Routeur chargé au premier usage depuis `SARAGOSSA_RUST_SPEC_ROUTE_EMBED_DIR`,
/// `None` sans la variable ou si le chargement échoue (message sur stderr).
pub fn router() -> Option<&'static EmbedRouter> {
    static ROUTER: OnceLock<Option<EmbedRouter>> = OnceLock::new();
    ROUTER
        .get_or_init(|| {
            let dir = crate::runtime_flags::spec_route_embed_dir()?;
            EmbedRouter::load(&dir)
                .map_err(|error| {
                    eprintln!(
                        "routage par embeddings désactivé ({}) : {error}",
                        dir.display()
                    );
                })
                .ok()
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::EmbedRouter;

    /// Exactitude et latence sur le jeu multilingue tenu à part
    /// (`tools/fixtures/spec_route_embed_test.json`, distinct de `refs.json`).
    /// Requiert `SARAGOSSA_RUST_SPEC_ROUTE_EMBED_DIR` (snapshot local d'e5-small).
    #[test]
    #[ignore = "poids multilingual-e5-small locaux requis"]
    fn embed_router_accuracy_and_latency() -> Result<(), Box<dyn std::error::Error>> {
        let Some(dir) = crate::runtime_flags::spec_route_embed_dir() else {
            eprintln!("skip : SARAGOSSA_RUST_SPEC_ROUTE_EMBED_DIR absent");
            return Ok(());
        };
        let router = EmbedRouter::load(&dir)?;
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../tools/fixtures/spec_route_embed_test.json"
        ))?;
        assert!(cases.len() >= 60);
        let (mut right, mut prose_to_dflash, mut code_to_mtp) = (0, 0, 0);
        let mut latencies = Vec::new();
        for case in &cases {
            let text = case["text"].as_str().ok_or("text")?;
            let is_code = case["class"] == "code";
            let started = std::time::Instant::now();
            let prose = router.is_prose(text);
            latencies.push(started.elapsed().as_secs_f64() * 1000.0);
            match (is_code, prose) {
                (true, false) | (false, true) => right += 1,
                (false, false) => prose_to_dflash += 1,
                (true, true) => {
                    code_to_mtp += 1;
                    eprintln!("code classé prose : {text}");
                }
            }
        }
        latencies.sort_by(f64::total_cmp);
        eprintln!(
            "embed_router n={} justes={right} prose→DFlash={prose_to_dflash} code→MTP={code_to_mtp} \
             latence médiane={:.2} ms p95={:.2} ms",
            cases.len(),
            latencies[latencies.len() / 2],
            latencies[latencies.len() * 95 / 100]
        );
        assert_eq!(
            code_to_mtp, 0,
            "un code envoyé vers MTP perd le gain DFlash"
        );
        Ok(())
    }
}
