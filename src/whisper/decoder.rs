//! Décodeur Whisper CPU, avec cross-attention sur les features audio.

use super::{
    take, take_layer_norm, take_linear, validate_vector, whisper_weights_path, WhisperConfig,
    WhisperEncoder, WhisperLayerNorm, WhisperSelfAttention,
};
use crate::{
    argmax, load_float_tensors, ForwardRuntime, InferError, Linear, Result, RustTokenizer, Tensor,
};
use std::collections::HashMap;
use std::path::Path;

const SOT_TOKEN: &str = "<|startoftranscript|>";
const EOT_TOKEN: &str = "<|endoftext|>";
const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";

#[derive(Clone, Debug)]
struct WhisperDecoderLayer {
    self_attn_layer_norm: WhisperLayerNorm,
    self_attn: WhisperSelfAttention,
    encoder_attn_layer_norm: WhisperLayerNorm,
    encoder_attn: WhisperSelfAttention,
    final_layer_norm: WhisperLayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// Décodeur texte Whisper avec cross-attention audio.
#[derive(Clone, Debug)]
pub struct WhisperDecoder {
    config: WhisperConfig,
    token_embeddings: Tensor,
    positions: Tensor,
    layers: Vec<WhisperDecoderLayer>,
    layer_norm: WhisperLayerNorm,
}

/// Pipeline STT Whisper complet, du PCM vers le texte.
#[derive(Clone, Debug)]
pub struct WhisperModel {
    encoder: WhisperEncoder,
    decoder: WhisperDecoder,
    tokenizer: RustTokenizer,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
}

/// État du decode RÉSIDENT (poids GPU + KV résident) lié à une génération.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct ResidentDecode<'a> {
    metal: &'a crate::MetalExecutor,
    dec: crate::metal_backend::WhisperResidentDecoder,
    kv: crate::metal_backend::WhisperDecodeKv,
}

enum DecodeStep {
    Advanced,
    Logits(Tensor),
    Token(u32),
}

impl WhisperDecoder {
    /// Charge le décodeur depuis un dossier HF local.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la config ou les poids nécessaires sont absents.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = WhisperConfig::from_file(model_dir.join("config.json"))?;
        let weights = whisper_weights_path(model_dir)?;
        let mut tensors = load_float_tensors(weights)?;
        Self::from_tensor_map(config, &mut tensors)
    }

    pub(super) fn from_tensor_map(
        config: WhisperConfig,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        config.validate()?;
        let token_embeddings = take(tensors, "model.decoder.embed_tokens.weight")?;
        let positions = take(tensors, "model.decoder.embed_positions.weight")?;
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for layer in 0..config.decoder_layers {
            let prefix = format!("model.decoder.layers.{layer}");
            layers.push(WhisperDecoderLayer {
                self_attn_layer_norm: take_layer_norm(
                    tensors,
                    &format!("{prefix}.self_attn_layer_norm"),
                )?,
                self_attn: WhisperSelfAttention {
                    q_proj: take_linear(tensors, &format!("{prefix}.self_attn.q_proj"))?,
                    k_proj: take_linear(tensors, &format!("{prefix}.self_attn.k_proj"))?,
                    v_proj: take_linear(tensors, &format!("{prefix}.self_attn.v_proj"))?,
                    out_proj: take_linear(tensors, &format!("{prefix}.self_attn.out_proj"))?,
                },
                encoder_attn_layer_norm: take_layer_norm(
                    tensors,
                    &format!("{prefix}.encoder_attn_layer_norm"),
                )?,
                encoder_attn: WhisperSelfAttention {
                    q_proj: take_linear(tensors, &format!("{prefix}.encoder_attn.q_proj"))?,
                    k_proj: take_linear(tensors, &format!("{prefix}.encoder_attn.k_proj"))?,
                    v_proj: take_linear(tensors, &format!("{prefix}.encoder_attn.v_proj"))?,
                    out_proj: take_linear(tensors, &format!("{prefix}.encoder_attn.out_proj"))?,
                },
                final_layer_norm: take_layer_norm(tensors, &format!("{prefix}.final_layer_norm"))?,
                fc1: take_linear(tensors, &format!("{prefix}.fc1"))?,
                fc2: take_linear(tensors, &format!("{prefix}.fc2"))?,
            });
        }
        let layer_norm = take_layer_norm(tensors, "model.decoder.layer_norm")?;
        let decoder = Self {
            config,
            token_embeddings,
            positions,
            layers,
            layer_norm,
        };
        decoder.validate_shapes()?;
        Ok(decoder)
    }

    /// Décode greedily depuis un préfixe de tokens et des features audio.
    ///
    /// Decode **incrémental** : le KV self-attn est mis en cache et grandit d'une
    /// ligne par token ; le KV cross-attn (encodeur figé) est **précalculé une
    /// fois**. Chaque pas ne traite que le nouveau token (single-query), ce qui
    /// supprime le coût O(n²) et la re-projection des 1500 frames par token de
    /// l'ancien `forward_tokens(&context)`. Byte-identique : mêmes matmuls,
    /// LayerNorm, GELU, attention (mêmes valeurs, simplement hoistées), même head.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les dimensions ou les projections échouent.
    pub fn generate_greedy(
        &self,
        audio_features: &Tensor,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        eot_token: u32,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Err(InferError::Config("préfixe Whisper vide".to_string()));
        }
        // Backend decode : RÉSIDENT (GPU, un command buffer/token, zéro readback)
        // par défaut sur Metal ; `RETI_STT_DECODER_PEROP=1` rebascule per-op.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let mut resident: Option<ResidentDecode<'_>> = match runtime.metal_executor() {
            Some(metal) if std::env::var_os("RETI_STT_DECODER_PEROP").is_none() => {
                let dec = self.build_resident_decoder(metal)?;
                let kv = metal.build_whisper_decode_kv_resident(
                    audio_features,
                    &dec,
                    self.config.max_target_positions,
                )?;
                Some(ResidentDecode { metal, dec, kv })
            }
            _ => None,
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let mut resident: Option<()> = None;

        let cross = if resident.is_none() {
            self.precompute_cross_kv(audio_features, runtime)?
        } else {
            Vec::new()
        };
        let mut self_kv: Vec<LayerSelfKv> = (0..self.layers.len())
            .map(|_| LayerSelfKv::new(self.config.d_model))
            .collect();

        let cross_ref = &cross;
        let mut step = |token: u32,
                        position: usize,
                        emit_next: bool,
                        self_kv: &mut [LayerSelfKv]|
         -> Result<DecodeStep> {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if let Some(rd) = resident.as_mut() {
                let h = self.embed_tokens_at(&[token], position)?;
                let token = rd
                    .metal
                    .encode_whisper_decode_step(&h, &rd.dec, &mut rd.kv, emit_next)?;
                return Ok(match token {
                    Some(token) => DecodeStep::Token(token),
                    None => DecodeStep::Advanced,
                });
            }
            let logits = self.forward_step(token, position, cross_ref, self_kv, runtime)?;
            if emit_next {
                Ok(DecodeStep::Logits(logits))
            } else {
                Ok(DecodeStep::Advanced)
            }
        };

        // Prefill : on alimente le préfixe token par token (le KV self s'accumule) ;
        // les logits après le dernier token du préfixe prédisent le 1er token généré.
        let mut next_from_resident = None;
        let mut logits = None;
        for (position, &token) in prompt_tokens.iter().enumerate() {
            let emit_next = position + 1 == prompt_tokens.len();
            match step(token, position, emit_next, &mut self_kv)? {
                DecodeStep::Advanced => {}
                DecodeStep::Token(token) => next_from_resident = Some(token),
                DecodeStep::Logits(value) => logits = Some(value),
            }
        }
        if next_from_resident.is_none() && logits.is_none() {
            return Err(InferError::Config("préfixe Whisper vide".to_string()));
        }

        let mut generated = Vec::new();
        let budget = max_new_tokens.min(self.config.max_target_positions);
        let mut ctx_len = prompt_tokens.len();
        for _step in 0..budget {
            let next = match next_from_resident.take() {
                Some(token) => token,
                None => {
                    let logits = logits.take().ok_or_else(|| {
                        InferError::Config("decode Whisper sans logits".to_string())
                    })?;
                    u32::try_from(argmax(logits.data())?).map_err(|_| {
                        InferError::Dimension("token argmax hors plage u32".to_string())
                    })?
                }
            };
            if next == eot_token {
                break;
            }
            generated.push(next);
            let position = ctx_len;
            ctx_len += 1;
            if ctx_len >= self.config.max_target_positions {
                break;
            }
            match step(next, position, true, &mut self_kv)? {
                DecodeStep::Advanced => {}
                DecodeStep::Token(token) => next_from_resident = Some(token),
                DecodeStep::Logits(value) => logits = Some(value),
            }
        }
        Ok(generated)
    }

    /// Précalcule le KV cross-attn de chaque couche depuis les features encodeur
    /// (figées pour l'énoncé) — une seule fois, au lieu d'à chaque token.
    fn precompute_cross_kv(
        &self,
        audio_features: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Vec<LayerCrossKv>> {
        let [frames, _dim] = audio_features.shape() else {
            return Err(InferError::Dimension(format!(
                "features audio attendues rang 2, reçu {:?}",
                audio_features.shape()
            )));
        };
        self.layers
            .iter()
            .map(|layer| {
                let k = layer.encoder_attn.project_k(audio_features, runtime)?;
                let v = layer.encoder_attn.project_v(audio_features, runtime)?;
                Ok(LayerCrossKv {
                    k: k.into_data(),
                    v: v.into_data(),
                    frames: *frames,
                })
            })
            .collect()
    }

    /// Prépare les poids GPU (mémoïsés) du décodeur résident.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn build_resident_decoder(
        &self,
        metal: &crate::MetalExecutor,
    ) -> Result<crate::metal_backend::WhisperResidentDecoder> {
        use crate::metal_backend::{
            MetalLinearWeightBuffers, WhisperDecodeLayer, WhisperResidentDecoder,
            WhisperResidentNorm, WhisperResidentProj,
        };
        let proj = |linear: &Linear| -> Result<WhisperResidentProj> {
            let tensor = match linear.weight() {
                crate::LinearWeight::Dense(tensor) => tensor,
                crate::LinearWeight::AffineQuantized(_) => {
                    return Err(InferError::Config(
                        "décodeur résident: projection non dense".to_string(),
                    ));
                }
            };
            let weight = metal.cached_buffer_from_f32(tensor.data(), "whisper_dec_w")?;
            let weight_na = if crate::metal_backend::whisper_bf16_gemm_enabled() {
                Some(metal.cached_rhs_t_bf16(tensor)?)
            } else {
                None
            };
            let weight_bf16 = if crate::metal_backend::whisper_decode_bf16_qmv_enabled() {
                Some(metal.cached_buffer_from_f32_as_bf16(tensor.data(), "whisper_dec_w_bf16")?)
            } else {
                None
            };
            let bias = match linear.bias() {
                Some(bias) => Some(metal.cached_buffer_from_f32(bias.data(), "whisper_dec_b")?),
                None => None,
            };
            Ok(WhisperResidentProj {
                weight,
                bias,
                weight_na,
                weight_bf16,
            })
        };
        let norm = |ln: &super::WhisperLayerNorm| -> Result<WhisperResidentNorm> {
            Ok(WhisperResidentNorm {
                weight: metal.cached_buffer_from_f32(ln.weight.data(), "whisper_dec_ln_w")?,
                bias: metal.cached_buffer_from_f32(ln.bias.data(), "whisper_dec_ln_b")?,
            })
        };

        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (q, k, v, o) = layer.self_attn.linears();
            let (cross_q, cross_k, cross_v, cross_o) = layer.encoder_attn.linears();
            layers.push(WhisperDecodeLayer {
                self_ln: norm(&layer.self_attn_layer_norm)?,
                q: proj(q)?,
                k: proj(k)?,
                v: proj(v)?,
                o: proj(o)?,
                cross_ln: norm(&layer.encoder_attn_layer_norm)?,
                cross_q: proj(cross_q)?,
                cross_k: proj(cross_k)?,
                cross_v: proj(cross_v)?,
                cross_o: proj(cross_o)?,
                final_ln: norm(&layer.final_layer_norm)?,
                fc1: proj(&layer.fc1)?,
                fc2: proj(&layer.fc2)?,
            });
        }
        let ffn_dim = self
            .layers
            .first()
            .and_then(|l| l.fc1.weight().shape().first().copied())
            .unwrap_or(0);
        let [vocab, in_dim] = self.token_embeddings.shape() else {
            return Err(InferError::Dimension(format!(
                "lm_head Whisper attendu rang 2, reçu {:?}",
                self.token_embeddings.shape()
            )));
        };
        Ok(WhisperResidentDecoder {
            layers,
            final_ln: norm(&self.layer_norm)?,
            lm_head: MetalLinearWeightBuffers::Dense {
                rhs: metal
                    .cached_buffer_from_f32(self.token_embeddings.data(), "whisper_lm_head")?,
                rhs_bf16: if crate::metal_backend::whisper_decode_bf16_qmv_enabled() {
                    Some(metal.cached_buffer_from_f32_as_bf16(
                        self.token_embeddings.data(),
                        "whisper_lm_head_bf16",
                    )?)
                } else {
                    None
                },
                out_dim: *vocab,
                in_dim: *in_dim,
            },
            d_model: self.config.d_model,
            heads: self.config.decoder_attention_heads,
            ffn_dim,
            eps: super::LAYER_NORM_EPS,
        })
    }

    /// Traite UN token à `position` : embed + 4 couches (self-attn KV-caché,
    /// cross-attn sur KV statique, FFN) + head. Append la K/V self du token dans
    /// `self_kv`. Renvoie les logits `[1, vocab]`.
    fn forward_step(
        &self,
        token: u32,
        position: usize,
        cross: &[LayerCrossKv],
        self_kv: &mut [LayerSelfKv],
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let heads = self.config.decoder_attention_heads;
        let mut h = self.embed_tokens_at(&[token], position)?;

        for ((layer, xkv), skv) in self.layers.iter().zip(cross).zip(self_kv.iter_mut()) {
            let residual = h.clone();
            let normed = layer.self_attn_layer_norm.forward(&h)?;
            let q = layer.self_attn.project_q(&normed, runtime)?;
            let k = layer.self_attn.project_k(&normed, runtime)?;
            let v = layer.self_attn.project_v(&normed, runtime)?;
            skv.append(k.data(), v.data())?;
            let ctx = single_query_attention(q.data(), &skv.k, &skv.v, skv.len, heads)?;
            let attn = layer.self_attn.project_out(&Tensor::row(ctx)?, runtime)?;
            h = residual.add(&attn)?;

            let residual = h.clone();
            let normed = layer.encoder_attn_layer_norm.forward(&h)?;
            let q = layer.encoder_attn.project_q(&normed, runtime)?;
            let ctx = single_query_attention(q.data(), &xkv.k, &xkv.v, xkv.frames, heads)?;
            let attn = layer
                .encoder_attn
                .project_out(&Tensor::row(ctx)?, runtime)?;
            h = residual.add(&attn)?;

            let residual = h.clone();
            let normed = layer.final_layer_norm.forward(&h)?;
            let ff = feed_forward(&normed, &layer.fc1, &layer.fc2, runtime)?;
            h = residual.add(&ff)?;
        }

        self.decoder_head(&h, runtime)
    }

    fn embed_tokens_at(&self, tokens: &[u32], offset: usize) -> Result<Tensor> {
        let [vocab, dim] = self.token_embeddings.shape() else {
            return Err(InferError::Dimension(format!(
                "model.decoder.embed_tokens.weight attendu rang 2, reçu {:?}",
                self.token_embeddings.shape()
            )));
        };
        let [pos_count, pos_dim] = self.positions.shape() else {
            return Err(InferError::Dimension(format!(
                "model.decoder.embed_positions.weight attendu rang 2, reçu {:?}",
                self.positions.shape()
            )));
        };
        if *pos_dim != *dim || offset + tokens.len() > *pos_count {
            return Err(InferError::Dimension(format!(
                "positions decodeur shape={:?}, offset={offset}, tokens={}",
                self.positions.shape(),
                tokens.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len() * *dim);
        for (idx, token) in tokens.iter().copied().enumerate() {
            let token = usize::try_from(token)
                .map_err(|_| InferError::Dimension("token Whisper hors plage usize".to_string()))?;
            if token >= *vocab {
                return Err(InferError::Dimension(format!(
                    "token Whisper {token} hors vocab {vocab}"
                )));
            }
            let token_base = token * *dim;
            let pos_base = (offset + idx) * *dim;
            for col in 0..*dim {
                out.push(
                    self.token_embeddings.data()[token_base + col]
                        + self.positions.data()[pos_base + col],
                );
            }
        }
        Tensor::from_vec(vec![tokens.len(), *dim], out)
    }

    fn decoder_head(&self, h: &Tensor, runtime: ForwardRuntime<'_>) -> Result<Tensor> {
        let h = self.layer_norm.forward(h)?;
        let last = Tensor::row(h.last_row()?.to_vec())?;
        // Head = embeddings liés (biasless). Sur Metal, le matmul vocab (51866×1280)
        // part sur GPU (buffer poids mémoïsé par ptr) au lieu du matmul CPU 66 M
        // MAC/token ; l'argmax reste CPU sur les logits relus (~207 ko).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = runtime.metal_executor() {
            return metal.matmul_rhs_t_dense(&last, &self.token_embeddings);
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = runtime;
        last.matmul_rhs_t(&self.token_embeddings)
    }

    fn validate_shapes(&self) -> Result<()> {
        match self.token_embeddings.shape() {
            [vocab, dim] if *vocab == self.config.vocab_size && *dim == self.config.d_model => {}
            shape => {
                return Err(InferError::Dimension(format!(
                    "model.decoder.embed_tokens.weight attendu [{},{}], reçu {shape:?}",
                    self.config.vocab_size, self.config.d_model
                )));
            }
        }
        match self.positions.shape() {
            [positions, dim]
                if *positions >= self.config.max_target_positions
                    && *dim == self.config.d_model => {}
            shape => {
                return Err(InferError::Dimension(format!(
                    "model.decoder.embed_positions.weight attendu [>={},{}], reçu {shape:?}",
                    self.config.max_target_positions, self.config.d_model
                )));
            }
        }
        validate_vector(
            &self.layer_norm.weight,
            self.config.d_model,
            "model.decoder.layer_norm.weight",
        )?;
        validate_vector(
            &self.layer_norm.bias,
            self.config.d_model,
            "model.decoder.layer_norm.bias",
        )
    }
}

impl WhisperModel {
    /// Charge le pipeline Whisper complet depuis un dossier HF local.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un artefact, un poids ou un token spécial manque.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = WhisperConfig::from_file(model_dir.join("config.json"))?;
        let mut tensors = load_float_tensors(whisper_weights_path(model_dir)?)?;
        let encoder = WhisperEncoder::from_tensor_map(config.clone(), &mut tensors)?;
        let decoder = WhisperDecoder::from_tensor_map(config, &mut tensors)?;
        let tokenizer = RustTokenizer::from_file(model_dir.join("tokenizer.json"))?;
        let sot_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            sot_token,
            transcribe_token,
            eot_token,
            no_timestamps_token,
        })
    }

    /// Transcrit une utterance PCM f32 mono 16 kHz.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'encodage, le décodage ou le tokenizer échoue.
    pub fn transcribe(
        &self,
        samples: &[f32],
        lang: &str,
        runtime: ForwardRuntime<'_>,
    ) -> Result<(String, String)> {
        let timing = std::env::var_os("RETI_STT_TIMING").is_some();
        let t_enc = std::time::Instant::now();
        let audio_features = self.encoder.encode_samples(samples, runtime)?;
        if timing {
            eprintln!("[stt] encode: {} ms", t_enc.elapsed().as_millis());
        }
        self.transcribe_with_features(&audio_features, lang, runtime)
    }

    /// Transcrit depuis des features audio Whisper déjà encodées.
    ///
    /// Le chemin ANE de `reti` injecte ici les hidden states de l'encodeur
    /// CoreML, tandis que `transcribe` conserve l'encodage saragossa existant.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le décodage ou le tokenizer échoue.
    pub fn transcribe_with_features(
        &self,
        audio_features: &Tensor,
        lang: &str,
        runtime: ForwardRuntime<'_>,
    ) -> Result<(String, String)> {
        let timing = std::env::var_os("RETI_STT_TIMING").is_some();
        let mut prompt = Vec::with_capacity(4);
        prompt.push(self.sot_token);
        if let Some(language_token) = self.language_token(lang) {
            prompt.push(language_token);
        }
        prompt.push(self.transcribe_token);
        prompt.push(self.no_timestamps_token);

        let t_dec = std::time::Instant::now();

        let generated = self.decoder.generate_greedy(
            audio_features,
            &prompt,
            self.decoder.config.max_target_positions.min(448),
            self.eot_token,
            runtime,
        )?;
        if timing {
            eprintln!(
                "[stt] decode: {} ms ({} tokens)",
                t_dec.elapsed().as_millis(),
                generated.len()
            );
        }
        let text = self.tokenizer.decode(&generated, true)?;
        let effective_lang = if lang == "auto" { "auto" } else { lang }.to_string();
        Ok((text.trim().to_string(), effective_lang))
    }

    fn language_token(&self, lang: &str) -> Option<u32> {
        if lang == "auto" {
            return None;
        }
        self.tokenizer.token_to_id(&format!("<|{lang}|>"))
    }
}

/// KV self-attn d'UNE couche, en cache CPU, append-only (grandit d'une ligne par
/// token décodé). Remplace la re-projection de tout le contexte à chaque pas.
struct LayerSelfKv {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    dim: usize,
}

impl LayerSelfKv {
    fn new(dim: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            dim,
        }
    }

    fn append(&mut self, k_row: &[f32], v_row: &[f32]) -> Result<()> {
        if k_row.len() != self.dim || v_row.len() != self.dim {
            return Err(InferError::Dimension(format!(
                "append KV self: attendu {}, reçu k={} v={}",
                self.dim,
                k_row.len(),
                v_row.len()
            )));
        }
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
        self.len += 1;
        Ok(())
    }
}

/// KV cross-attn d'UNE couche, précalculé une fois sur les features encodeur.
struct LayerCrossKv {
    k: Vec<f32>,
    v: Vec<f32>,
    frames: usize,
}

/// Attention multi-têtes single-query (le token courant attend tout le KV caché).
/// Miroir exact de `multi_head_attention` (whisper.rs) pour `q_seq = 1`, sans
/// masque (le cache ne contient que des positions ≤ courante en self, toutes les
/// frames en cross) : softmax stable, `scale = 1/sqrt(head_dim)`, mêmes ordres
/// d'accumulation → byte-identique à la dernière ligne de l'ancien chemin.
fn single_query_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_seq: usize,
    heads: usize,
) -> Result<Vec<f32>> {
    let dim = q.len();
    if heads == 0 || dim % heads != 0 {
        return Err(InferError::Dimension(format!(
            "single-query attention dim={dim} heads={heads}"
        )));
    }
    let expected = kv_seq
        .checked_mul(dim)
        .ok_or_else(|| InferError::Dimension("single-query attention kv déborde".to_string()))?;
    if k.len() != expected || v.len() != expected {
        return Err(InferError::Dimension(format!(
            "single-query attention k={} v={} attendu {expected}",
            k.len(),
            v.len()
        )));
    }
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; dim];
    let mut scores = vec![0.0_f32; kv_seq];
    for head in 0..heads {
        let head_base = head * head_dim;
        let mut max_score = f32::NEG_INFINITY;
        for row_k in 0..kv_seq {
            let mut dot = 0.0_f32;
            for col in 0..head_dim {
                dot += q[head_base + col] * k[row_k * dim + head_base + col];
            }
            let score = dot * scale;
            scores[row_k] = score;
            max_score = max_score.max(score);
        }
        let mut denom = 0.0_f32;
        for score in scores.iter_mut() {
            *score = (*score - max_score).exp();
            denom += *score;
        }
        for col in 0..head_dim {
            let mut acc = 0.0_f32;
            for (row_v, prob) in scores.iter().enumerate() {
                acc += (*prob / denom) * v[row_v * dim + head_base + col];
            }
            out[head_base + col] = acc;
        }
    }
    Ok(out)
}

fn feed_forward(
    x: &Tensor,
    fc1: &Linear,
    fc2: &Linear,
    runtime: ForwardRuntime<'_>,
) -> Result<Tensor> {
    let h = fc1.forward_with_runtime(x, runtime)?;
    fc2.forward_with_runtime(&crate::gelu(&h), runtime)
}

fn token_id(tokenizer: &RustTokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| InferError::Config(format!("token Whisper introuvable: {token}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn embed_tokens_adds_positional_offset() {
        let decoder = tiny_decoder_without_layers();
        let out = decoder
            .embed_tokens_at(&[0, 2], 1)
            .expect("invariant: embeddings valides");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[11.0, 22.0, 35.0, 46.0]);
    }

    #[test]
    fn generate_greedy_rejects_empty_prompt() {
        let decoder = tiny_decoder_without_layers();
        let audio = Tensor::from_vec(vec![1, 2], vec![0.0, 0.0]).expect("invariant: audio");
        let err = decoder
            .generate_greedy(&audio, &[], 1, 0, ForwardRuntime::cpu())
            .expect_err("invariant: préfixe vide rejeté");
        assert!(matches!(err, InferError::Config(_)));
    }

    /// Transcription metal-rs ≡ golden mlx-rs figé (sans mlx-rs ; charge whisper-tiny
    /// pour l'inférence metal-rs sur l'entrée PCM16k golden). Égalité exacte texte+lang.
    #[test]
    #[ignore = "golden: charge whisper-tiny (cache HF) pour la transcription metal-rs"]
    fn golden_transcribe_matches_fixture() -> Result<()> {
        let Some(model_dir) = local_whisper_tiny_dir() else {
            eprintln!("skip: snapshot openai/whisper-tiny absent du cache HF");
            return Ok(());
        };
        let (_, samples) = crate::golden::read_f32("whisper_samples16k")?;
        let expected = crate::golden::read_text("whisper_transcribe_text")?;
        let expected_lang = crate::golden::read_text("whisper_transcribe_lang")?;

        let model = WhisperModel::from_model_dir(model_dir)?;
        let runtime = live_runtime();
        let (got, got_lang) = model.transcribe(&samples, "fr", runtime)?;
        assert_eq!(got_lang, expected_lang);
        assert_eq!(got, expected);
        Ok(())
    }

    fn local_whisper_tiny_dir() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("RETI_WHISPER_TINY_DIR") {
            let path = PathBuf::from(path);
            if path.join("config.json").is_file() && path.join("model.safetensors").is_file() {
                return Some(path);
            }
        }
        let snapshot = crate::hf_resolve::hf_cache_dir_from_env().and_then(|hub| {
            let snapshots = hub.join("models--openai--whisper-tiny/snapshots");
            std::fs::read_dir(snapshots)
                .ok()?
                .flatten()
                .map(|entry| entry.path())
                .find(|path| {
                    path.join("config.json").is_file() && path.join("model.safetensors").is_file()
                })
        });
        crate::test_support::require_real_model(snapshot, "snapshot openai/whisper-tiny")
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn live_runtime() -> ForwardRuntime<'static> {
        let metal = Box::leak(Box::new(
            crate::MetalExecutor::new().expect("invariant: Metal disponible pour le test live"),
        ));
        ForwardRuntime::metal(metal)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn live_runtime() -> ForwardRuntime<'static> {
        ForwardRuntime::cpu()
    }

    fn tiny_decoder_without_layers() -> WhisperDecoder {
        WhisperDecoder {
            config: WhisperConfig {
                d_model: 2,
                encoder_attention_heads: 1,
                encoder_layers: 1,
                decoder_attention_heads: 1,
                decoder_layers: 1,
                num_mel_bins: 80,
                max_target_positions: 4,
                vocab_size: 3,
            },
            token_embeddings: Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("invariant: embeddings"),
            positions: Tensor::from_vec(
                vec![4, 2],
                vec![0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            )
            .expect("invariant: positions"),
            layers: Vec::new(),
            layer_norm: WhisperLayerNorm {
                weight: Tensor::from_vec(vec![2], vec![1.0, 1.0]).expect("invariant: ln weight"),
                bias: Tensor::from_vec(vec![2], vec![0.0, 0.0]).expect("invariant: ln bias"),
            },
        }
    }
}
