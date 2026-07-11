//! Embedder sémantique : **BERT multilingue** (`intfloat/multilingual-e5-small`)
//! en **Rust pur f32 CPU**, mean-pooling masqué + L2-normalize.
//!
//! ## Pipeline forward
//!
//!   ids [L] + mask [L]
//!     └─ embeddings : word[ids] + position[0..L] + token_type[0] → LayerNorm
//!     └─ N=12 BertLayer : (self-attn non causale + add&LN) ; (FFN gelu + add&LN)
//!     └─ mean-pooling MASQUÉ : Σ(h · mask) / Σmask  (E5 = mean tokens, PAS CLS)
//!     └─ L2-normalize → vecteur unitaire [384]
//!
//! ## Parité
//!
//! Mêmes poids f32, mêmes ops, même ordre de pipeline que le port mlx-rs
//! d'origine ; seuls l'ordre des réductions (CPU vs GPU) et l'approximation erf
//! diffèrent (~1e-6 relatif). La parité est MESURÉE côté reti (test golden
//! `embed_parity_rust_vs_golden` de `src/memory/embed_rust.rs`) : cosinus par
//! phrase et rang top-5, jamais supposée.

use std::path::Path;

use super::config::BertConfig;
use super::error::TextEmbedError;
use super::math::{dot, gelu_inplace, layer_norm_inplace, linear, softmax_inplace};
use super::weights::{BertLayerW, BertWeights};
use rayon::prelude::*;
use tokenizers::Tokenizer;

/// Repo HF du modèle d'embedding par défaut. `multilingual-e5-small` (MIT,
/// 384-dim, multilingue). Le 1ᵉʳ usage télécharge ~470 Mo dans le cache HF.
pub const DEFAULT_TEXT_EMBED_REPO: &str = "intfloat/multilingual-e5-small";

/// Dimensionnalité de sortie du modèle (`hidden_size`). Doit rester cohérent
/// avec `EMBED_DIM_SEMANTIC` côté `src/memory`. E5-small = 384.
pub const TEXT_EMBED_DIM: usize = 384;

/// Borne de longueur en tokens. E5 plafonne à 512 (`max_position_embeddings`) ;
/// un tour de conversation court tient très en dessous. On tronque par sécurité.
const MAX_TOKENS: usize = 512;

/// Embedder sémantique texte chargé en mémoire (poids f32 CPU + tokenizer).
///
/// **Concurrence** : tout est CPU et `&self` (le forward ne mute rien) →
/// `Send + Sync`, aucun verrou requis. Le parallélisme interne passe par le
/// pool rayon global (partagé avec le reste de `saragossa`).
pub struct TextEmbedder {
    w: BertWeights,
    cfg: BertConfig,
    tokenizer: Tokenizer,
}

impl TextEmbedder {
    /// Charge l'embedder depuis un répertoire local contenant `config.json`,
    /// `tokenizer.json` et `model.safetensors` (snapshot HF déjà résolu par
    /// l'appelant — même idiome que `WhisperModel::from_model_dir` /
    /// `TtsModel::load_local`).
    ///
    /// # Errors
    ///
    /// - [`TextEmbedError::Config`] / [`TextEmbedError::Tokenizer`] /
    ///   [`TextEmbedError::Weights`] si un artefact est illisible, une clé de
    ///   poids manque ou une shape est incohérente.
    pub fn load_local(model_dir: impl AsRef<Path>) -> Result<Self, TextEmbedError> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");

        let cfg: BertConfig =
            serde_json::from_reader(std::fs::File::open(&config_path).map_err(|e| {
                TextEmbedError::Config(format!("ouverture {}: {e}", config_path.display()))
            })?)
            .map_err(|e| TextEmbedError::Config(format!("parsing config.json: {e}")))?;

        // Garde-fou : on a porté le corps BERT 384-dim ; un autre hidden_size
        // remonterait des formes incohérentes au forward → on échoue tôt et clair.
        if cfg.hidden_size as usize != TEXT_EMBED_DIM {
            return Err(TextEmbedError::Config(format!(
                "hidden_size={} attendu {TEXT_EMBED_DIM} (port figé sur e5-small)",
                cfg.hidden_size
            )));
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TextEmbedError::Tokenizer(format!("{e}")))?;

        let buf = std::fs::read(&weights_path)
            .map_err(|e| TextEmbedError::Weights(format!("lecture model.safetensors: {e}")))?;
        // NOTE: pic RAM transitoire ~2× la taille des poids (buffer brut + copies
        // f32 typées) le temps du chargement ; ~470 Mo résidents ensuite.
        let w = BertWeights::from_safetensors_bytes(
            &buf,
            cfg.hidden_size as usize,
            cfg.num_hidden_layers.max(0) as usize,
            cfg.num_attention_heads.max(0) as usize,
        )?;

        Ok(Self { w, cfg, tokenizer })
    }

    /// Embede une **requête** E5 (préfixe `"query: "`). À utiliser côté recall.
    ///
    /// # Errors
    ///
    /// [`TextEmbedError::Tokenizer`] si l'encodage échoue (le forward CPU est
    /// infaillible, shapes validées au chargement).
    pub fn embed_query(&self, text: &str) -> Result<[f32; TEXT_EMBED_DIM], TextEmbedError> {
        self.embed_prefixed("query: ", text)
    }

    /// Embede un **passage** E5 (préfixe `"passage: "`). À utiliser côté
    /// stockage (tours de conversation, chunks de code indexés).
    ///
    /// # Errors
    ///
    /// Cf. [`TextEmbedder::embed_query`].
    pub fn embed_passage(&self, text: &str) -> Result<[f32; TEXT_EMBED_DIM], TextEmbedError> {
        self.embed_prefixed("passage: ", text)
    }

    /// Tokenise `prefix + text`, fait le forward BERT, mean-pool masqué, L2-norm.
    fn embed_prefixed(
        &self,
        prefix: &str,
        text: &str,
    ) -> Result<[f32; TEXT_EMBED_DIM], TextEmbedError> {
        let full = format!("{prefix}{text}");
        // `encode(_, true)` ajoute les special tokens (<s> … </s>) attendus par E5.
        let enc = self
            .tokenizer
            .encode(full, true)
            .map_err(|e| TextEmbedError::Tokenizer(format!("encode: {e}")))?;

        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let mut mask: Vec<f32> = enc.get_attention_mask().iter().map(|&m| m as f32).collect();
        // Texte vide ou tokenizer renvoyant 0 token : forward d'une séquence d'au
        // moins 1 token (les special tokens devraient l'éviter, mais on est sûr).
        if ids.is_empty() {
            ids.push(self.cfg.pad_token_id);
            mask.push(0.0);
        }
        let cap = MAX_TOKENS.min(self.w.max_pos);
        if ids.len() > cap {
            ids.truncate(cap);
            mask.truncate(cap);
        }
        // Un id hors vocab ne peut pas sortir du tokenizer du même repo ; on le
        // traite quand même en erreur claire plutôt qu'en panique d'indexation.
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= self.w.vocab) {
            return Err(TextEmbedError::Tokenizer(format!(
                "id {bad} hors vocab ({}) — tokenizer/poids incohérents",
                self.w.vocab
            )));
        }

        let pooled = self.forward(&ids, &mask);
        Ok(l2_normalize(&pooled))
    }

    /// Forward complet : ids + mask → vecteur mean-poolé `[hidden]`.
    /// Infaillible : ids bornés au vocab et shapes validées au chargement.
    fn forward(&self, ids: &[u32], mask: &[f32]) -> Vec<f32> {
        let l = ids.len();
        let h = self.w.h;
        let eps = self.cfg.layer_norm_eps;

        // ----- embeddings : word + position + token_type, puis LayerNorm. -----
        let mut hs = vec![0.0f32; l * h];
        for (i, &id) in ids.iter().enumerate() {
            let word = &self.w.word[id as usize * h..(id as usize + 1) * h];
            let pos = &self.w.pos[i * h..(i + 1) * h];
            let dst = &mut hs[i * h..(i + 1) * h];
            for d in 0..h {
                dst[d] = word[d] + pos[d] + self.w.tok_type0[d];
            }
        }
        layer_norm_inplace(&mut hs, h, &self.w.ln_emb, eps);

        // ----- N couches BertLayer (post-LayerNorm). -----
        for layer in &self.w.layers {
            self.bert_layer(&mut hs, l, layer, eps);
        }

        // ----- mean-pooling masqué : Σ(h · mask) / Σmask. -----
        let mut pooled = vec![0.0f32; h];
        for (row, &m) in hs.chunks_exact(h).zip(mask.iter()) {
            for (p, &v) in pooled.iter_mut().zip(row.iter()) {
                *p += v * m;
            }
        }
        // clamp ≥ 1e-9 : un mask tout-à-zéro (entrée dégénérée) → pas de div par 0.
        let denom = mask.iter().sum::<f32>().max(1e-9);
        for p in pooled.iter_mut() {
            *p /= denom;
        }
        pooled
    }

    /// Une couche BertLayer (post-LN) : self-attn → add&LN → FFN(gelu) → add&LN.
    fn bert_layer(&self, hs: &mut [f32], l: usize, lw: &BertLayerW, eps: f32) {
        // self-attention non causale puis projection de sortie.
        let q = linear(hs, l, &lw.q);
        let k = linear(hs, l, &lw.k);
        let v = linear(hs, l, &lw.v);
        let ctx = self.attention(&q, &k, &v, l);
        let attn = linear(&ctx, l, &lw.o);
        add_inplace(hs, &attn);
        layer_norm_inplace(hs, self.w.h, &lw.ln_attn, eps);

        // FFN : intermediate.dense → gelu → output.dense.
        let mut ff = linear(hs, l, &lw.ff_in);
        gelu_inplace(&mut ff);
        let ff2 = linear(&ff, l, &lw.ff_out);
        add_inplace(hs, &ff2);
        layer_norm_inplace(hs, self.w.h, &lw.ln_ffn, eps);
    }

    /// Multi-head self-attention BERT (non causale) : pour chaque ligne i et
    /// chaque tête, softmax(q·kᵀ·scale)·v sur les colonnes de la tête (layout
    /// row-major `[L, h]`, tête t = colonnes `t·hd..(t+1)·hd`). Parallélisé par
    /// ligne de sortie (déterministe). Renvoie le contexte `[L, h]` AVANT
    /// `attention.output.dense` (appliqué par l'appelant).
    ///
    /// Mask de padding négligé, comme le port mlx-rs : 1 seul texte par appel,
    /// séquence à longueur exacte (pas de padding) → tous les tokens sont
    /// valides, le mean-pool masque déjà l'agrégation.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], l: usize) -> Vec<f32> {
        let h = self.w.h;
        let nh = self.w.heads;
        let hd = h / nh;
        let scale = (hd as f32).powf(-0.5);

        let mut out = vec![0.0f32; l * h];
        out.par_chunks_mut(h).enumerate().for_each(|(i, orow)| {
            let mut scores = vec![0.0f32; l];
            for t in 0..nh {
                let off = t * hd;
                let qi = &q[i * h + off..i * h + off + hd];
                for (j, s) in scores.iter_mut().enumerate() {
                    *s = dot(qi, &k[j * h + off..j * h + off + hd]) * scale;
                }
                softmax_inplace(&mut scores);
                let ctx = &mut orow[off..off + hd];
                for (j, &p) in scores.iter().enumerate() {
                    let vj = &v[j * h + off..j * h + off + hd];
                    for (c, &x) in ctx.iter_mut().zip(vj.iter()) {
                        *c += p * x;
                    }
                }
            }
        });
        out
    }
}

/// Addition élément à élément in-place (résidu).
fn add_inplace(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len(), "résidu: shapes différentes");
    for (a, b) in x.iter_mut().zip(y.iter()) {
        *a += b;
    }
}

/// L2-normalise le vecteur poolé en `[f32; TEXT_EMBED_DIM]` (cosinus = produit
/// scalaire ensuite). Plancher 1e-9 : vecteur nul → reste nul, pas de NaN.
fn l2_normalize(v: &[f32]) -> [f32; TEXT_EMBED_DIM] {
    debug_assert_eq!(
        v.len(),
        TEXT_EMBED_DIM,
        "le forward doit produire TEXT_EMBED_DIM"
    );
    let mut out = [0.0f32; TEXT_EMBED_DIM];
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        *o = x / norm;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::weights::BertWeights;
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    /// Construit un mini-BERT synthétique (h=8, 1 couche, 2 têtes, vocab=16,
    /// max_pos=8, ffn=16) sérialisé safetensors en mémoire : exerce TOUT le
    /// chemin chargement+forward sans réseau ni vrai checkpoint.
    fn tiny_bert() -> BertWeights {
        const H: usize = 8;
        const FFN: usize = 16;
        const VOCAB: usize = 16;
        const POS: usize = 8;

        // Valeurs pseudo-aléatoires déterministes, petites (réseau stable).
        let gen = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                    ((x % 1000) as f32 / 1000.0 - 0.5) * 0.2
                })
                .collect()
        };

        let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = vec![
            (
                "embeddings.word_embeddings.weight".into(),
                gen(VOCAB * H, 1),
                vec![VOCAB, H],
            ),
            (
                "embeddings.position_embeddings.weight".into(),
                gen(POS * H, 2),
                vec![POS, H],
            ),
            (
                "embeddings.token_type_embeddings.weight".into(),
                gen(2 * H, 3),
                vec![2, H],
            ),
            ("embeddings.LayerNorm.weight".into(), vec![1.0; H], vec![H]),
            ("embeddings.LayerNorm.bias".into(), vec![0.0; H], vec![H]),
        ];
        let layer0 = [
            ("attention.self.query", H, H),
            ("attention.self.key", H, H),
            ("attention.self.value", H, H),
            ("attention.output.dense", H, H),
            ("intermediate.dense", FFN, H),
            ("output.dense", H, FFN),
        ];
        for (idx, (name, out, inp)) in layer0.into_iter().enumerate() {
            let seed = 10 + idx as u32;
            tensors.push((
                format!("encoder.layer.0.{name}.weight"),
                gen(out * inp, seed),
                vec![out, inp],
            ));
            tensors.push((
                format!("encoder.layer.0.{name}.bias"),
                gen(out, 100 + seed),
                vec![out],
            ));
        }
        for ln in ["attention.output.LayerNorm", "output.LayerNorm"] {
            tensors.push((
                format!("encoder.layer.0.{ln}.weight"),
                vec![1.0; H],
                vec![H],
            ));
            tensors.push((format!("encoder.layer.0.{ln}.bias"), vec![0.0; H], vec![H]));
        }

        let bytes: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .into_iter()
            .map(|(n, d, s)| (n, d.iter().flat_map(|x| x.to_le_bytes()).collect(), s))
            .collect();
        let views: Vec<(&str, TensorView<'_>)> = bytes
            .iter()
            .map(|(n, d, s)| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, s.clone(), d).expect("view synthétique valide"),
                )
            })
            .collect();
        let buf = safetensors::serialize(views, None).expect("serialize mini-BERT");
        BertWeights::from_safetensors_bytes(&buf, H, 1, 2).expect("chargement mini-BERT")
    }

    fn tiny_embedder() -> TextEmbedder {
        // Tokenizer minimal jamais utilisé par `forward` (appel direct) ; un
        // WordLevel vide suffit à construire la struct.
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .build()
            .expect("wordlevel vide");
        TextEmbedder {
            w: tiny_bert(),
            cfg: BertConfig {
                hidden_size: 8,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                layer_norm_eps: 1e-12,
                pad_token_id: 0,
            },
            tokenizer: Tokenizer::new(model),
        }
    }

    #[test]
    fn forward_is_deterministic_and_finite() {
        let emb = tiny_embedder();
        let ids = [1u32, 5, 9, 3];
        let mask = [1.0f32; 4];
        let a = emb.forward(&ids, &mask);
        let b = emb.forward(&ids, &mask);
        assert_eq!(a, b, "même entrée → même sortie (bit-à-bit, CPU)");
        assert_eq!(a.len(), 8);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn masked_tokens_do_not_change_pooling_weighting() {
        // Le mean-pool masqué ignore les positions à mask=0 : le pooled d'une
        // séquence [a, b] avec mask [1, 0] = la ligne de a seule (l'attention
        // voit b — comme le port mlx-rs, qui néglige le mask d'attention —
        // mais l'agrégation ne pondère que a).
        let emb = tiny_embedder();
        let full = emb.forward(&[2, 7], &[1.0, 1.0]);
        let half = emb.forward(&[2, 7], &[1.0, 0.0]);
        assert_ne!(full, half, "le mask doit changer l'agrégation");
        assert!(half.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_mask_yields_zero_vector_not_nan() {
        let emb = tiny_embedder();
        let pooled = emb.forward(&[1], &[0.0]);
        assert!(pooled.iter().all(|&v| v == 0.0), "Σmask=0 → vecteur nul");
        let out = l2_normalize(&vec![0.0; TEXT_EMBED_DIM]);
        assert!(out.iter().all(|&v| v == 0.0), "norme nulle → pas de NaN");
    }

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![0.0f32; TEXT_EMBED_DIM];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i as f32 * 0.13).sin();
        }
        let out = l2_normalize(&v);
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norme {norm}");
    }
}
