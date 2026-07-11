//! Chargement du `model.safetensors` HF (layout `BertModel`) en structs
//! **typées** : toutes les clés et shapes sont validées ICI, une fois, au
//! chargement → le forward (`model.rs`) s'exécute ensuite sans aucun chemin
//! d'erreur (zéro `unwrap`, zéro indexation hasardeuse).
//!
//! La conversion octets → f32 se fait par `f32::from_le_bytes` sur des
//! `chunks_exact(4)` : alignement et boutisme maîtrisés quel que soit l'offset
//! du tenseur dans le buffer (safetensors ne garantit pas l'alignement 4).
//! Les dtypes F16/BF16 sont upcastés f32 par défense (le checkpoint E5 est
//! f32), comme le fait le port mlx-rs.

use super::error::TextEmbedError;
use super::math::{LayerNormW, Linear};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

/// Une couche `BertLayer` (poids HF `encoder.layer.N.*`), post-LayerNorm.
pub(super) struct BertLayerW {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    /// `attention.output.dense` (projection de sortie de l'attention).
    pub o: Linear,
    /// `attention.output.LayerNorm` (après le résidu d'attention).
    pub ln_attn: LayerNormW,
    /// `intermediate.dense` (montée FFN, gelu appliqué par l'appelant).
    pub ff_in: Linear,
    /// `output.dense` (descente FFN).
    pub ff_out: Linear,
    /// `output.LayerNorm` (après le résidu FFN).
    pub ln_ffn: LayerNormW,
}

/// L'ensemble des poids du corps `BertModel`, shapes validées.
pub(super) struct BertWeights {
    /// `embeddings.word_embeddings.weight` `[vocab, h]`.
    pub word: Vec<f32>,
    pub vocab: usize,
    /// `embeddings.position_embeddings.weight` `[max_pos, h]`.
    pub pos: Vec<f32>,
    pub max_pos: usize,
    /// Ligne 0 de `embeddings.token_type_embeddings.weight` (séquence unique
    /// → token_type=0 partout, comme le port mlx-rs).
    pub tok_type0: Vec<f32>,
    /// `embeddings.LayerNorm`.
    pub ln_emb: LayerNormW,
    pub layers: Vec<BertLayerW>,
    /// `hidden_size` (=384), recopié ici pour que le forward soit autonome.
    pub h: usize,
    /// Nombre de têtes d'attention (=12).
    pub heads: usize,
}

impl BertWeights {
    /// Charge et valide les poids depuis le buffer brut d'un `model.safetensors`.
    ///
    /// `h`/`layers`/`heads` viennent du `config.json` (déjà gardés-fous côté
    /// [`super::model::TextEmbedder::load_local`]).
    ///
    /// # Errors
    ///
    /// [`TextEmbedError::Weights`] si le fichier n'est pas un safetensors valide,
    /// si une clé attendue du layout `BertModel` manque, ou si une shape / un
    /// dtype est incohérent.
    pub fn from_safetensors_bytes(
        buf: &[u8],
        h: usize,
        layers: usize,
        heads: usize,
    ) -> Result<Self, TextEmbedError> {
        let st = SafeTensors::deserialize(buf)
            .map_err(|e| TextEmbedError::Weights(format!("safetensors invalide: {e}")))?;

        let word = matrix(&st, "embeddings.word_embeddings.weight", None, h)?;
        let vocab = word.len() / h;
        let pos = matrix(&st, "embeddings.position_embeddings.weight", None, h)?;
        let max_pos = pos.len() / h;
        // token_type : on ne garde que la ligne 0 (type_vocab_size ≥ 1).
        let tok_type = matrix(&st, "embeddings.token_type_embeddings.weight", None, h)?;
        if tok_type.len() < h {
            return Err(TextEmbedError::Weights(
                "token_type_embeddings: moins d'une ligne".to_string(),
            ));
        }
        let tok_type0 = tok_type[..h].to_vec();
        let ln_emb = layer_norm(&st, "embeddings.LayerNorm", h)?;

        if heads == 0 || h % heads != 0 {
            return Err(TextEmbedError::Weights(format!(
                "num_attention_heads={heads} ne divise pas hidden_size={h}"
            )));
        }

        let mut layer_ws = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("encoder.layer.{i}");
            // La montée FFN fixe la dim intermédiaire ; la descente doit lui
            // être symétrique (validé par `linear` via expected_out/in).
            let ff_in = linear_w(&st, &format!("{p}.intermediate.dense"), None, h)?;
            let ffn = ff_in.out_dim;
            layer_ws.push(BertLayerW {
                q: linear_w(&st, &format!("{p}.attention.self.query"), Some(h), h)?,
                k: linear_w(&st, &format!("{p}.attention.self.key"), Some(h), h)?,
                v: linear_w(&st, &format!("{p}.attention.self.value"), Some(h), h)?,
                o: linear_w(&st, &format!("{p}.attention.output.dense"), Some(h), h)?,
                ln_attn: layer_norm(&st, &format!("{p}.attention.output.LayerNorm"), h)?,
                ff_in,
                ff_out: linear_w(&st, &format!("{p}.output.dense"), Some(h), ffn)?,
                ln_ffn: layer_norm(&st, &format!("{p}.output.LayerNorm"), h)?,
            });
        }

        Ok(Self {
            word,
            vocab,
            pos,
            max_pos,
            tok_type0,
            ln_emb,
            layers: layer_ws,
            h,
            heads,
        })
    }
}

/// Récupère un tenseur par clé et le convertit en `Vec<f32>` (LE, upcast
/// F16/BF16 défensif).
fn tensor_f32(st: &SafeTensors<'_>, key: &str) -> Result<(Vec<f32>, Vec<usize>), TextEmbedError> {
    let view = st
        .tensor(key)
        .map_err(|_| TextEmbedError::Weights(format!("poids manquant: {key}")))?;
    let data = to_f32(&view, key)?;
    Ok((data, view.shape().to_vec()))
}

fn to_f32(view: &TensorView<'_>, key: &str) -> Result<Vec<f32>, TextEmbedError> {
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        Dtype::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(TextEmbedError::Weights(format!(
            "{key}: dtype {other:?} non supporté (attendu F32/F16/BF16)"
        ))),
    }
}

/// Convertit un half IEEE 754 (binary16) en f32, sous-normaux/inf/NaN compris.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x3ff);
    match (exp, man) {
        (0, 0) => f32::from_bits(sign),
        // Sous-normal : valeur = man · 2⁻²⁴ (exposant min binary16 −14, 10 bits).
        (0, m) => {
            let v = m as f32 * 5.960_464_5e-8;
            f32::from_bits(sign | v.to_bits())
        }
        (0x1f, 0) => f32::from_bits(sign | 0x7f80_0000),
        (0x1f, m) => f32::from_bits(sign | 0x7f80_0000 | (m << 13)),
        (e, m) => f32::from_bits(sign | ((e + 127 - 15) << 23) | (m << 13)),
    }
}

/// Charge une matrice 2D `[rows, cols]` en validant `cols` (et `rows` si fourni).
fn matrix(
    st: &SafeTensors<'_>,
    key: &str,
    expected_rows: Option<usize>,
    expected_cols: usize,
) -> Result<Vec<f32>, TextEmbedError> {
    let (data, shape) = tensor_f32(st, key)?;
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        other => {
            return Err(TextEmbedError::Weights(format!(
                "{key}: shape {other:?}, attendu 2D"
            )))
        }
    };
    if cols != expected_cols || expected_rows.is_some_and(|r| r != rows) {
        return Err(TextEmbedError::Weights(format!(
            "{key}: shape [{rows}, {cols}], attendu [{}, {expected_cols}]",
            expected_rows.map_or_else(|| "*".to_string(), |r| r.to_string())
        )));
    }
    Ok(data)
}

/// Charge un vecteur 1D `[len]` en validant sa longueur.
fn vector(
    st: &SafeTensors<'_>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, TextEmbedError> {
    let (data, shape) = tensor_f32(st, key)?;
    if shape.as_slice() != [expected_len] {
        return Err(TextEmbedError::Weights(format!(
            "{key}: shape {shape:?}, attendu [{expected_len}]"
        )));
    }
    Ok(data)
}

/// Charge `{prefix}.weight` `[out, in]` + `{prefix}.bias` `[out]` (BertModel a
/// un biais sur toutes ses linéaires).
fn linear_w(
    st: &SafeTensors<'_>,
    prefix: &str,
    expected_out: Option<usize>,
    expected_in: usize,
) -> Result<Linear, TextEmbedError> {
    let w = matrix(st, &format!("{prefix}.weight"), expected_out, expected_in)?;
    let out_dim = w.len() / expected_in;
    let b = vector(st, &format!("{prefix}.bias"), out_dim)?;
    Ok(Linear {
        w,
        b,
        out_dim,
        in_dim: expected_in,
    })
}

/// Charge la paire γ/β d'un LayerNorm (`{prefix}.weight` / `{prefix}.bias`).
fn layer_norm(st: &SafeTensors<'_>, prefix: &str, h: usize) -> Result<LayerNormW, TextEmbedError> {
    Ok(LayerNormW {
        gamma: vector(st, &format!("{prefix}.weight"), h)?,
        beta: vector(st, &format!("{prefix}.bias"), h)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_roundtrips_known_values() {
        // 1.0 = 0x3c00 ; -2.0 = 0xc000 ; 0.5 = 0x3800 ; +inf = 0x7c00.
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        // Sous-normal minimal : 2⁻²⁴.
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
        // Sous-normal négatif.
        assert!(f16_to_f32(0x8001) < 0.0);
    }

    #[test]
    fn missing_key_is_clear_error() {
        // safetensors minimal : un seul tenseur, donc tout le layout BertModel manque.
        let bytes = le_bytes(&[0f32; 4]);
        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 2], &bytes)
            .expect("view valide");
        let buf = safetensors::serialize([("autre.poids", view)], None).expect("serialize");
        let err = match BertWeights::from_safetensors_bytes(&buf, 2, 1, 1) {
            Ok(_) => panic!("layout BertModel absent: le chargement aurait dû échouer"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("word_embeddings"),
            "erreur attendue sur la 1ʳᵉ clé structurante: {err}"
        );
        assert!(!err.is_retryable());
    }

    /// Encode un `&[f32]` en octets LE (l'écrivain canonique safetensors).
    fn le_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}
