//! Décodeur codec Qwen3-TTS f32 pour le pipeline TTS saragossa.

use crate::tts::{SafetensorPayload, TtsCodecConfig, TtsDecoderConfig};
use crate::{rms_norm, InferError, Result, Tensor};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct TtsCodec {
    cfg: TtsDecoderConfig,
    sample_rate: u32,
    weights: HashMap<String, Tensor>,
    /// Forward GPU résident de la section chaude (`decoder.decoder.*`), construit
    /// au chargement quand Metal est dispo et `RETI_TTS_CODEC_GPU` actif (défaut).
    /// Repli CPU (octet-identique) si absent.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    gpu: Option<crate::tts_codec_gpu::CodecGpu>,
}

#[derive(Debug, Default)]
pub(crate) struct TtsCodecStreamState {
    generated_frames: usize,
    hidden_time: usize,
    emitted_samples: usize,
    prefix: TtsCodecPrefixStreamState,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    gpu: Option<crate::tts_codec_gpu::CodecGpuStream>,
}

#[derive(Debug, Default)]
struct TtsCodecPrefixStreamState {
    pre_conv: CpuConvStreamState,
    transformer: PreTransformerStreamState,
    upsample: Vec<UpsampleStreamState>,
}

#[derive(Debug, Default)]
struct CpuConvStreamState {
    context: Option<Nlc>,
}

#[derive(Debug, Default)]
struct PreTransformerStreamState {
    layers: Vec<CodecAttentionStreamState>,
}

#[derive(Debug, Default)]
struct CodecAttentionStreamState {
    k: Option<Nlc>,
    v: Option<Nlc>,
}

#[derive(Debug, Default)]
struct UpsampleStreamState {
    transpose: CpuConvStreamState,
    convnext: CpuConvStreamState,
}

#[derive(Clone, Debug)]
struct Nlc {
    time: usize,
    channels: usize,
    data: Vec<f32>,
}

impl TtsCodec {
    pub(crate) fn load(payload: &SafetensorPayload, config: &TtsCodecConfig) -> Result<Self> {
        let cfg = config.decoder_config.clone();
        let sample_rate = u32::try_from(config.output_sample_rate).map_err(|_| {
            InferError::Config(format!(
                "output_sample_rate TTS négatif: {}",
                config.output_sample_rate
            ))
        })?;
        let mut weights = HashMap::new();
        let mut cluster = HashMap::new();
        let mut embsum = HashMap::new();

        for name in payload.names() {
            if let Some(base) = name.strip_suffix("._codebook.cluster_usage") {
                cluster.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if let Some(base) = name.strip_suffix("._codebook.embedding_sum") {
                embsum.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if let Some(base) = name.strip_suffix(".codebook.cluster_usage") {
                cluster.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if let Some(base) = name.strip_suffix(".codebook.embed_sum") {
                embsum.insert(
                    format!("{base}.codebook"),
                    payload.read_dense_tensor(&name)?,
                );
            } else if name.ends_with(".codebook.initialized") {
                continue;
            } else {
                weights.insert(name.clone(), payload.read_dense_tensor(&name)?);
            }
        }

        for (base, usage) in cluster {
            let sum = embsum
                .get(&base)
                .ok_or_else(|| InferError::MissingWeight(format!("{base}.embedding_sum codec")))?;
            weights.insert(format!("{base}.embed.weight"), codebook_embed(&usage, sum)?);
        }

        // Forward GPU résident de la section chaude : best-effort au chargement.
        // Échec (pas de Metal, kernel KO, poids absent) ⇒ repli CPU silencieux.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let gpu = if crate::runtime_flags::env_flag("RETI_TTS_CODEC_GPU", true) {
            match crate::tts_codec_gpu::CodecGpu::new(&weights, &cfg) {
                Ok(gpu) => Some(gpu),
                Err(error) => {
                    eprintln!("codec GPU indisponible, repli CPU: {error}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            cfg,
            sample_rate,
            weights,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            gpu,
        })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Indique si le forward GPU résident du codec est actif (tests de parité).
    #[cfg(test)]
    pub(crate) fn gpu_active(&self) -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.gpu.is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    pub(crate) fn decode_codes(&self, generated: &[Vec<i32>]) -> Result<Vec<f32>> {
        if generated.is_empty() {
            return Ok(Vec::new());
        }
        let hidden = self.decode_prefix(generated)?;
        // Section chaude (`decoder.decoder.*`, ~94 % du coût) : forward GPU résident
        // si disponible, sinon repli CPU octet-identique (l'oracle de parité).
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(gpu) = self.gpu.as_ref() {
            let pcm = gpu.decode_tail(hidden.time, hidden.channels, &hidden.data)?;
            return Ok(pcm.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect());
        }
        self.decode_tail_cpu(&hidden)
    }

    pub(crate) fn new_stream_state(&self) -> TtsCodecStreamState {
        TtsCodecStreamState {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            gpu: self.gpu.as_ref().map(|gpu| gpu.new_stream()),
            ..TtsCodecStreamState::default()
        }
    }

    /// Décode un préfixe croissant et renvoie uniquement le nouveau PCM.
    ///
    /// Le chemin Metal incrémente le préfixe CPU causal puis la section chaude
    /// `decoder.decoder.*` avec les contextes conv/transpose et les K/V attention.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le préfixe régresse ou si le codec échoue.
    pub(crate) fn decode_codes_streaming(
        &self,
        state: &mut TtsCodecStreamState,
        generated: &[Vec<i32>],
    ) -> Result<Vec<f32>> {
        if generated.len() < state.generated_frames {
            return Err(InferError::Dimension(format!(
                "codec streaming préfixe régressif: {} < {}",
                generated.len(),
                state.generated_frames
            )));
        }
        if generated.is_empty() || generated.len() == state.generated_frames {
            return Ok(Vec::new());
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(gpu_state) = state.gpu.as_mut() {
            let hidden = self
                .decode_prefix_streaming(&mut state.prefix, &generated[state.generated_frames..])?;
            let pcm = self
                .gpu
                .as_ref()
                .ok_or_else(|| {
                    InferError::Metal("codec streaming GPU sans forward GPU actif".to_string())
                })?
                .decode_tail_streaming(gpu_state, hidden.time, hidden.channels, &hidden.data)?;
            state.hidden_time = state.hidden_time.saturating_add(hidden.time);
            state.generated_frames = generated.len();
            state.emitted_samples = state.emitted_samples.saturating_add(pcm.len());
            return Ok(pcm.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect());
        }

        let pcm = self.decode_codes(generated)?;
        if pcm.len() < state.emitted_samples {
            return Err(InferError::Dimension(format!(
                "codec streaming PCM régressif: {} < {}",
                pcm.len(),
                state.emitted_samples
            )));
        }
        let delta = pcm[state.emitted_samples..].to_vec();
        state.emitted_samples = pcm.len();
        state.generated_frames = generated.len();
        Ok(delta)
    }

    /// Préfixe CPU (RVQ → `pre_conv` → `pre_transformer` → boucle d'upsampling),
    /// produisant l'entrée `hidden` de la section `decoder.decoder` (commune au
    /// chemin GPU et CPU).
    fn decode_prefix(&self, generated: &[Vec<i32>]) -> Result<Nlc> {
        let mut hidden = self.rvq_decode(generated)?;
        hidden = self.causal_conv1d(
            &hidden,
            "decoder.pre_conv.conv.weight",
            "decoder.pre_conv.conv.bias",
            1,
            1,
        )?;
        hidden = self.pre_transformer(&hidden)?;

        for (idx, factor) in self.cfg.upsampling_ratios.iter().copied().enumerate() {
            hidden = self.causal_transpose1d(
                &hidden,
                &format!("decoder.upsample.{idx}.0.conv.weight"),
                &format!("decoder.upsample.{idx}.0.conv.bias"),
                usize_from_i32(factor, "upsampling ratio")?,
            )?;
            hidden = self.convnext_block(
                &hidden,
                &format!("decoder.upsample.{idx}.1"),
                usize_from_i32(self.cfg.latent_dim, "latent_dim")?,
            )?;
        }
        Ok(hidden)
    }

    fn decode_prefix_streaming(
        &self,
        state: &mut TtsCodecPrefixStreamState,
        new_codes: &[Vec<i32>],
    ) -> Result<Nlc> {
        if new_codes.is_empty() {
            return Nlc::new(
                0,
                usize_from_i32(self.cfg.latent_dim, "latent_dim")?,
                Vec::new(),
            );
        }
        let mut hidden = self.rvq_decode(new_codes)?;
        hidden = self.causal_conv1d_stream(
            &mut state.pre_conv,
            &hidden,
            "decoder.pre_conv.conv.weight",
            "decoder.pre_conv.conv.bias",
            1,
            1,
        )?;
        hidden = self.pre_transformer_stream(&mut state.transformer, &hidden)?;

        let blocks = self.cfg.upsampling_ratios.len();
        if state.upsample.len() < blocks {
            state
                .upsample
                .resize_with(blocks, UpsampleStreamState::default);
        }
        for (idx, factor) in self.cfg.upsampling_ratios.iter().copied().enumerate() {
            hidden = self.causal_transpose1d_stream(
                &mut state.upsample[idx].transpose,
                &hidden,
                &format!("decoder.upsample.{idx}.0.conv.weight"),
                &format!("decoder.upsample.{idx}.0.conv.bias"),
                usize_from_i32(factor, "upsampling ratio")?,
            )?;
            hidden = self.convnext_block_stream(
                &mut state.upsample[idx].convnext,
                &hidden,
                &format!("decoder.upsample.{idx}.1"),
                usize_from_i32(self.cfg.latent_dim, "latent_dim")?,
            )?;
        }
        Ok(hidden)
    }

    /// Décode entièrement sur CPU (oracle de parité, contourne le forward GPU).
    #[cfg(test)]
    pub(crate) fn decode_codes_cpu(&self, generated: &[Vec<i32>]) -> Result<Vec<f32>> {
        if generated.is_empty() {
            return Ok(Vec::new());
        }
        let hidden = self.decode_prefix(generated)?;
        self.decode_tail_cpu(&hidden)
    }

    /// Section `decoder.decoder.0` → tail sur CPU (scalaire rayon). Oracle de
    /// parité : la sortie de [`Self::decode_codes`] sur GPU est comparée à ce
    /// chemin (tolérance audio). `hidden` = sortie de la boucle d'upsampling.
    fn decode_tail_cpu(&self, hidden: &Nlc) -> Result<Vec<f32>> {
        let mut wav = self.causal_conv1d(
            hidden,
            "decoder.decoder.0.conv.weight",
            "decoder.decoder.0.conv.bias",
            1,
            1,
        )?;
        for (idx, rate) in self.cfg.upsample_rates.iter().copied().enumerate() {
            let block = idx + 1;
            wav = self.snake_beta(
                &wav,
                &format!("decoder.decoder.{block}.block.0.alpha"),
                &format!("decoder.decoder.{block}.block.0.beta"),
            )?;
            wav = self.causal_transpose1d(
                &wav,
                &format!("decoder.decoder.{block}.block.1.conv.weight"),
                &format!("decoder.decoder.{block}.block.1.conv.bias"),
                usize_from_i32(rate, "upsample rate")?,
            )?;
            for (slot, dilation) in [(2_usize, 1_usize), (3, 3), (4, 9)] {
                wav = self.decoder_residual_unit(
                    &wav,
                    &format!("decoder.decoder.{block}.block.{slot}"),
                    dilation,
                )?;
            }
        }
        wav = self.snake_beta(&wav, "decoder.decoder.5.alpha", "decoder.decoder.5.beta")?;
        wav = self.causal_conv1d(
            &wav,
            "decoder.decoder.6.conv.weight",
            "decoder.decoder.6.conv.bias",
            1,
            1,
        )?;
        if wav.channels != 1 {
            return Err(InferError::Dimension(format!(
                "codec wav attendu mono, reçu {} canaux",
                wav.channels
            )));
        }
        Ok(wav.data.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect())
    }

    fn rvq_decode(&self, generated: &[Vec<i32>]) -> Result<Nlc> {
        let semantic = usize_from_i32(self.cfg.num_semantic_quantizers, "num_semantic_quantizers")?;
        let total = usize_from_i32(self.cfg.num_quantizers, "num_quantizers")?;
        let first = self.rvq_decode_one(generated, 0, semantic, "decoder.quantizer.rvq_first")?;
        if total <= semantic {
            return Ok(first);
        }
        let rest = self.rvq_decode_one(
            generated,
            semantic,
            total - semantic,
            "decoder.quantizer.rvq_rest",
        )?;
        first.add(&rest)
    }

    fn rvq_decode_one(
        &self,
        generated: &[Vec<i32>],
        group_offset: usize,
        n_q: usize,
        prefix: &str,
    ) -> Result<Nlc> {
        let emb0 = self.weight(&format!("{prefix}.vq.layers.0.codebook.embed.weight"))?;
        let inner = matrix_shape(emb0)?.1;
        let mut acc = Nlc::zeros(generated.len(), inner);
        for q in 0..n_q {
            let emb = self.weight(&format!("{prefix}.vq.layers.{q}.codebook.embed.weight"))?;
            for (time, frame) in generated.iter().enumerate() {
                let code = frame.get(group_offset + q).ok_or_else(|| {
                    InferError::Dimension(format!("frame TTS sans codebook {}", group_offset + q))
                })?;
                let row = row_i32(emb, *code)?;
                acc.add_row_in_place(time, row)?;
            }
        }
        self.conv1x1(&acc, &format!("{prefix}.output_proj.weight"))
    }

    fn pre_transformer(&self, x: &Nlc) -> Result<Nlc> {
        let mut h = self.linear(x, "decoder.pre_transformer.input_proj")?;
        for layer in 0..usize_from_i32(self.cfg.num_hidden_layers, "codec layers")? {
            let prefix = format!("decoder.pre_transformer.layers.{layer}");
            let residual = h.clone();
            let hn = self.rms_norm_nlc(
                &h,
                &format!("{prefix}.input_layernorm.weight"),
                self.cfg.rms_norm_eps,
            )?;
            let attn = self.codec_attention(&hn, &format!("{prefix}.self_attn"))?;
            let attn = self.mul_channel(&attn, &format!("{prefix}.self_attn_layer_scale.scale"))?;
            h = residual.add(&attn)?;

            let residual = h.clone();
            let hn = self.rms_norm_nlc(
                &h,
                &format!("{prefix}.post_attention_layernorm.weight"),
                self.cfg.rms_norm_eps,
            )?;
            let gate = self.linear(&hn, &format!("{prefix}.mlp.gate_proj"))?;
            let up = self.linear(&hn, &format!("{prefix}.mlp.up_proj"))?;
            let act = gate.map(silu_scalar);
            let ff = self.linear(&act.mul(&up)?, &format!("{prefix}.mlp.down_proj"))?;
            let ff = self.mul_channel(&ff, &format!("{prefix}.mlp_layer_scale.scale"))?;
            h = residual.add(&ff)?;
        }
        let h = self.rms_norm_nlc(
            &h,
            "decoder.pre_transformer.norm.weight",
            self.cfg.rms_norm_eps,
        )?;
        self.linear(&h, "decoder.pre_transformer.output_proj")
    }

    fn pre_transformer_stream(
        &self,
        state: &mut PreTransformerStreamState,
        x: &Nlc,
    ) -> Result<Nlc> {
        let mut h = self.linear(x, "decoder.pre_transformer.input_proj")?;
        let layers = usize_from_i32(self.cfg.num_hidden_layers, "codec layers")?;
        if state.layers.len() < layers {
            state
                .layers
                .resize_with(layers, CodecAttentionStreamState::default);
        }
        for layer in 0..layers {
            let prefix = format!("decoder.pre_transformer.layers.{layer}");
            let residual = h.clone();
            let hn = self.rms_norm_nlc(
                &h,
                &format!("{prefix}.input_layernorm.weight"),
                self.cfg.rms_norm_eps,
            )?;
            let attn = self.codec_attention_stream(
                &mut state.layers[layer],
                &hn,
                &format!("{prefix}.self_attn"),
            )?;
            let attn = self.mul_channel(&attn, &format!("{prefix}.self_attn_layer_scale.scale"))?;
            h = residual.add(&attn)?;

            let residual = h.clone();
            let hn = self.rms_norm_nlc(
                &h,
                &format!("{prefix}.post_attention_layernorm.weight"),
                self.cfg.rms_norm_eps,
            )?;
            let gate = self.linear(&hn, &format!("{prefix}.mlp.gate_proj"))?;
            let up = self.linear(&hn, &format!("{prefix}.mlp.up_proj"))?;
            let act = gate.map(silu_scalar);
            let ff = self.linear(&act.mul(&up)?, &format!("{prefix}.mlp.down_proj"))?;
            let ff = self.mul_channel(&ff, &format!("{prefix}.mlp_layer_scale.scale"))?;
            h = residual.add(&ff)?;
        }
        let h = self.rms_norm_nlc(
            &h,
            "decoder.pre_transformer.norm.weight",
            self.cfg.rms_norm_eps,
        )?;
        self.linear(&h, "decoder.pre_transformer.output_proj")
    }

    fn codec_attention(&self, x: &Nlc, prefix: &str) -> Result<Nlc> {
        let head_dim = usize_from_i32(self.cfg.head_dim, "codec head_dim")?;
        let n_heads = usize_from_i32(self.cfg.num_attention_heads, "codec heads")?;
        let n_kv = usize_from_i32(self.cfg.num_key_value_heads, "codec kv heads")?;
        if n_heads % n_kv != 0 {
            return Err(InferError::Dimension(format!(
                "codec heads {n_heads} non divisibles par kv_heads {n_kv}"
            )));
        }
        let q = self.linear(x, &format!("{prefix}.q_proj"))?;
        let k = self.linear(x, &format!("{prefix}.k_proj"))?;
        let v = self.linear(x, &format!("{prefix}.v_proj"))?;
        let q = rope_heads(&q, n_heads, head_dim, self.cfg.rope_theta)?;
        let k = rope_heads(&k, n_kv, head_dim, self.cfg.rope_theta)?;
        let scale = (head_dim as f32).powf(-0.5);
        let mut out = Nlc::zeros(x.time, n_heads * head_dim);
        let kv_repeat = n_heads / n_kv;

        for time in 0..x.time {
            for head in 0..n_heads {
                let kv_head = head / kv_repeat;
                let mut scores = vec![0.0_f32; time + 1];
                for (key_time, score) in scores.iter_mut().enumerate().take(time + 1) {
                    let mut dot = 0.0_f32;
                    for dim in 0..head_dim {
                        dot += q.get(time, head * head_dim + dim)
                            * k.get(key_time, kv_head * head_dim + dim);
                    }
                    *score = dot * scale;
                }
                softmax_in_place(&mut scores);
                for dim in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (key_time, score) in scores.iter().copied().enumerate() {
                        acc += score * v.get(key_time, kv_head * head_dim + dim);
                    }
                    out.set(time, head * head_dim + dim, acc);
                }
            }
        }
        self.linear(&out, &format!("{prefix}.o_proj"))
    }

    fn codec_attention_stream(
        &self,
        state: &mut CodecAttentionStreamState,
        x: &Nlc,
        prefix: &str,
    ) -> Result<Nlc> {
        let head_dim = usize_from_i32(self.cfg.head_dim, "codec head_dim")?;
        let n_heads = usize_from_i32(self.cfg.num_attention_heads, "codec heads")?;
        let n_kv = usize_from_i32(self.cfg.num_key_value_heads, "codec kv heads")?;
        if n_heads % n_kv != 0 {
            return Err(InferError::Dimension(format!(
                "codec heads {n_heads} non divisibles par kv_heads {n_kv}"
            )));
        }
        let past = state.k.as_ref().map_or(0, |cache| cache.time);
        let q = self.linear(x, &format!("{prefix}.q_proj"))?;
        let k = self.linear(x, &format!("{prefix}.k_proj"))?;
        let v = self.linear(x, &format!("{prefix}.v_proj"))?;
        let q = rope_heads_at(&q, n_heads, head_dim, self.cfg.rope_theta, past)?;
        let k = rope_heads_at(&k, n_kv, head_dim, self.cfg.rope_theta, past)?;
        state.k = Some(append_optional_nlc(state.k.as_ref(), &k)?);
        state.v = Some(append_optional_nlc(state.v.as_ref(), &v)?);
        let k_cache = state
            .k
            .as_ref()
            .ok_or_else(|| InferError::Dimension("cache K codec vide".to_string()))?;
        let v_cache = state
            .v
            .as_ref()
            .ok_or_else(|| InferError::Dimension("cache V codec vide".to_string()))?;
        let scale = (head_dim as f32).powf(-0.5);
        let mut out = Nlc::zeros(x.time, n_heads * head_dim);
        let kv_repeat = n_heads / n_kv;

        for time in 0..x.time {
            let global_time = past + time;
            for head in 0..n_heads {
                let kv_head = head / kv_repeat;
                let mut scores = vec![0.0_f32; global_time + 1];
                for (key_time, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0_f32;
                    for dim in 0..head_dim {
                        dot += q.get(time, head * head_dim + dim)
                            * k_cache.get(key_time, kv_head * head_dim + dim);
                    }
                    *score = dot * scale;
                }
                softmax_in_place(&mut scores);
                for dim in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (key_time, score) in scores.iter().copied().enumerate() {
                        acc += score * v_cache.get(key_time, kv_head * head_dim + dim);
                    }
                    out.set(time, head * head_dim + dim, acc);
                }
            }
        }
        self.linear(&out, &format!("{prefix}.o_proj"))
    }

    fn convnext_block(&self, x: &Nlc, prefix: &str, dim: usize) -> Result<Nlc> {
        let residual = x.clone();
        let h = self.causal_conv1d(
            x,
            &format!("{prefix}.dwconv.conv.weight"),
            &format!("{prefix}.dwconv.conv.bias"),
            1,
            dim,
        )?;
        let h = self.layer_norm_nlc(
            &h,
            &format!("{prefix}.norm.weight"),
            &format!("{prefix}.norm.bias"),
            1.0e-6,
        )?;
        let h = self
            .linear(&h, &format!("{prefix}.pwconv1"))?
            .map(gelu_scalar);
        let h = self.linear(&h, &format!("{prefix}.pwconv2"))?;
        let h = self.mul_channel(&h, &format!("{prefix}.gamma"))?;
        residual.add(&h)
    }

    fn convnext_block_stream(
        &self,
        state: &mut CpuConvStreamState,
        x: &Nlc,
        prefix: &str,
        dim: usize,
    ) -> Result<Nlc> {
        let residual = x.clone();
        let h = self.causal_conv1d_stream(
            state,
            x,
            &format!("{prefix}.dwconv.conv.weight"),
            &format!("{prefix}.dwconv.conv.bias"),
            1,
            dim,
        )?;
        let h = self.layer_norm_nlc(
            &h,
            &format!("{prefix}.norm.weight"),
            &format!("{prefix}.norm.bias"),
            1.0e-6,
        )?;
        let h = self
            .linear(&h, &format!("{prefix}.pwconv1"))?
            .map(gelu_scalar);
        let h = self.linear(&h, &format!("{prefix}.pwconv2"))?;
        let h = self.mul_channel(&h, &format!("{prefix}.gamma"))?;
        residual.add(&h)
    }

    fn decoder_residual_unit(&self, x: &Nlc, prefix: &str, dilation: usize) -> Result<Nlc> {
        let residual = x.clone();
        let h = self.snake_beta(
            x,
            &format!("{prefix}.act1.alpha"),
            &format!("{prefix}.act1.beta"),
        )?;
        let h = self.causal_conv1d(
            &h,
            &format!("{prefix}.conv1.conv.weight"),
            &format!("{prefix}.conv1.conv.bias"),
            dilation,
            1,
        )?;
        let h = self.snake_beta(
            &h,
            &format!("{prefix}.act2.alpha"),
            &format!("{prefix}.act2.beta"),
        )?;
        let h = self.conv1d_nopad(
            &h,
            &format!("{prefix}.conv2.conv.weight"),
            &format!("{prefix}.conv2.conv.bias"),
            1,
        )?;
        residual.add(&h)
    }

    fn linear(&self, x: &Nlc, prefix: &str) -> Result<Nlc> {
        let weight = self.weight(&format!("{prefix}.weight"))?;
        let (out_dim, in_dim) = matrix_shape(weight)?;
        if x.channels != in_dim {
            return Err(InferError::Dimension(format!(
                "linear {prefix} attendu in={in_dim}, reçu {}",
                x.channels
            )));
        }
        let bias = match self.opt(&format!("{prefix}.bias")) {
            Some(b) => Some(channel_slice(b)?),
            None => None,
        };
        let wdata = weight.data();
        let xdata = &x.data;
        let mut out = vec![0.0_f32; x.time * out_dim];
        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(time, out_row)| {
                let row = &xdata[time * in_dim..time * in_dim + in_dim];
                for out_ch in 0..out_dim {
                    let w = &wdata[out_ch * in_dim..out_ch * in_dim + in_dim];
                    let mut acc = dot(row, w);
                    if let Some(b) = bias {
                        acc += b[out_ch];
                    }
                    out_row[out_ch] = acc;
                }
            });
        Nlc::new(x.time, out_dim, out)
    }

    fn conv1x1(&self, x: &Nlc, weight_key: &str) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        match weight.shape() {
            [out_dim, in_dim, 1] => {
                if *in_dim != x.channels {
                    return Err(InferError::Dimension(format!(
                        "conv1x1 {weight_key} attendu in={in_dim}, reçu {}",
                        x.channels
                    )));
                }
                let out_dim = *out_dim;
                let in_dim = *in_dim;
                let wdata = weight.data();
                let xdata = &x.data;
                let mut out = vec![0.0_f32; x.time * out_dim];
                out.par_chunks_mut(out_dim)
                    .enumerate()
                    .for_each(|(time, out_row)| {
                        let row = &xdata[time * in_dim..time * in_dim + in_dim];
                        for (out_ch, slot) in out_row.iter_mut().enumerate() {
                            let start = out_ch * in_dim;
                            *slot = dot(row, &wdata[start..start + in_dim]);
                        }
                    });
                Nlc::new(x.time, out_dim, out)
            }
            shape => Err(InferError::Dimension(format!(
                "conv1x1 {weight_key} attendu [out,in,1], reçu {shape:?}"
            ))),
        }
    }

    fn causal_conv1d(
        &self,
        x: &Nlc,
        weight_key: &str,
        bias_key: &str,
        dilation: usize,
        groups: usize,
    ) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let [out_dim, in_per_group, kernel] = conv_weight_shape(weight, weight_key)?;
        if groups == 0 || x.channels != in_per_group * groups || out_dim % groups != 0 {
            return Err(InferError::Dimension(format!(
                "conv {weight_key} incompatible x_channels={} groups={groups} weight={:?}",
                x.channels,
                weight.shape()
            )));
        }
        let bias = match self.opt(bias_key) {
            Some(b) => Some(channel_slice(b)?),
            None => None,
        };
        let pad = (kernel - 1) * dilation;
        let out_per_group = out_dim / groups;
        let wdata = weight.data();
        // Poids `[out, in, k]` réorganisé en `[out, k, in]` : la réduction interne
        // (sur `in_ch`) lit alors des tranches contiguës au lieu d'un accès au pas
        // `kernel`. Réorganisation pure → byte-identique.
        let mut wkc = vec![0.0_f32; out_dim * in_per_group * kernel];
        wkc.par_chunks_mut(kernel * in_per_group)
            .enumerate()
            .for_each(|(out_ch, seg)| {
                for kk in 0..kernel {
                    let dst = &mut seg[kk * in_per_group..kk * in_per_group + in_per_group];
                    for (in_ch, slot) in dst.iter_mut().enumerate() {
                        *slot = wdata[(out_ch * in_per_group + in_ch) * kernel + kk];
                    }
                }
            });
        let xdata = &x.data;
        let chans = x.channels;
        let x_time = x.time;
        let mut out = vec![0.0_f32; x.time * out_dim];
        // Chaque ligne de sortie (instant `time`) ne lit que des lignes
        // d'entrée en lecture seule → parallélisable sans changer l'ordre
        // d'accumulation (byte-identique à la version scalaire).
        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(time, out_row)| {
                for out_ch in 0..out_dim {
                    let group = out_ch / out_per_group;
                    let in_start = group * in_per_group;
                    let mut acc = bias.map_or(0.0, |b| b[out_ch]);
                    for k in 0..kernel {
                        let Some(src_time) = conv_source_time(time, k, pad, dilation) else {
                            continue;
                        };
                        if src_time >= x_time {
                            continue;
                        }
                        let xrow = &xdata[src_time * chans + in_start..];
                        let wseg = &wkc[(out_ch * kernel + k) * in_per_group..][..in_per_group];
                        for in_ch in 0..in_per_group {
                            acc += xrow[in_ch] * wseg[in_ch];
                        }
                    }
                    out_row[out_ch] = acc;
                }
            });
        Nlc::new(x.time, out_dim, out)
    }

    fn causal_conv1d_stream(
        &self,
        state: &mut CpuConvStreamState,
        x: &Nlc,
        weight_key: &str,
        bias_key: &str,
        dilation: usize,
        groups: usize,
    ) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let [_, _, kernel] = conv_weight_shape(weight, weight_key)?;
        let keep_rows = (kernel - 1).checked_mul(dilation).ok_or_else(|| {
            InferError::Dimension(format!("contexte conv streaming déborde {weight_key}"))
        })?;
        let context_time = state.context.as_ref().map_or(0, |context| context.time);
        let local = append_optional_nlc(state.context.as_ref(), x)?;
        let out = self.causal_conv1d(&local, weight_key, bias_key, dilation, groups)?;
        let suffix = out.slice_rows(context_time, x.time)?;
        state.context = tail_nlc(&local, keep_rows)?;
        Ok(suffix)
    }

    fn conv1d_nopad(
        &self,
        x: &Nlc,
        weight_key: &str,
        bias_key: &str,
        groups: usize,
    ) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let [out_dim, in_per_group, kernel] = conv_weight_shape(weight, weight_key)?;
        if kernel > x.time {
            return Err(InferError::Dimension(format!(
                "conv valid {weight_key} kernel={kernel} > time={}",
                x.time
            )));
        }
        let out_time = x.time - kernel + 1;
        let bias = match self.opt(bias_key) {
            Some(b) => Some(channel_slice(b)?),
            None => None,
        };
        let out_per_group = out_dim / groups;
        let wdata = weight.data();
        // Voir `causal_conv1d` : `[out, in, k]` → `[out, k, in]` pour des lectures
        // contiguës dans la réduction (byte-identique).
        let mut wkc = vec![0.0_f32; out_dim * in_per_group * kernel];
        wkc.par_chunks_mut(kernel * in_per_group)
            .enumerate()
            .for_each(|(out_ch, seg)| {
                for kk in 0..kernel {
                    let dst = &mut seg[kk * in_per_group..kk * in_per_group + in_per_group];
                    for (in_ch, slot) in dst.iter_mut().enumerate() {
                        *slot = wdata[(out_ch * in_per_group + in_ch) * kernel + kk];
                    }
                }
            });
        let xdata = &x.data;
        let chans = x.channels;
        let mut out = vec![0.0_f32; out_time * out_dim];
        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(time, out_row)| {
                for out_ch in 0..out_dim {
                    let group = out_ch / out_per_group;
                    let in_start = group * in_per_group;
                    let mut acc = bias.map_or(0.0, |b| b[out_ch]);
                    for k in 0..kernel {
                        let xrow = &xdata[(time + k) * chans + in_start..];
                        let wseg = &wkc[(out_ch * kernel + k) * in_per_group..][..in_per_group];
                        for in_ch in 0..in_per_group {
                            acc += xrow[in_ch] * wseg[in_ch];
                        }
                    }
                    out_row[out_ch] = acc;
                }
            });
        Nlc::new(out_time, out_dim, out)
    }

    fn causal_transpose1d(
        &self,
        x: &Nlc,
        weight_key: &str,
        bias_key: &str,
        stride: usize,
    ) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let [in_dim, out_dim, kernel] = conv_weight_shape(weight, weight_key)?;
        if x.channels != in_dim || stride == 0 {
            return Err(InferError::Dimension(format!(
                "conv transpose {weight_key} attendu in={in_dim}, stride={stride}, reçu {}",
                x.channels
            )));
        }
        let raw_time = (x.time - 1) * stride + kernel;
        let trim = kernel - stride;
        let out_time = raw_time.checked_sub(trim).ok_or_else(|| {
            InferError::Dimension(format!("trim conv transpose invalide {weight_key}"))
        })?;
        let bias = match self.opt(bias_key) {
            Some(b) => Some(channel_slice(b)?),
            None => None,
        };
        let wdata = weight.data();
        // Le poids transposé est stocké `[in, out, k]`. La boucle gather réduit
        // sur `in_ch` : on le réorganise une fois en `[out, k, in]` pour que cette
        // réduction lise des tranches contiguës (sinon chaque pas `in_ch` saute
        // `out_dim*kernel` éléments → une ligne de cache gaspillée par élément).
        // Réorganisation pure (mêmes valeurs) → byte-identique.
        let mut wkc = vec![0.0_f32; in_dim * out_dim * kernel];
        wkc.par_chunks_mut(kernel * in_dim)
            .enumerate()
            .for_each(|(out_ch, seg)| {
                for kk in 0..kernel {
                    let dst = &mut seg[kk * in_dim..kk * in_dim + in_dim];
                    for (in_ch, slot) in dst.iter_mut().enumerate() {
                        *slot = wdata[(in_ch * out_dim + out_ch) * kernel + kk];
                    }
                }
            });
        let xdata = &x.data;
        let x_time = x.time;
        // Reformulation gather : la version scalaire dispersait (scatter) chaque
        // entrée dans plusieurs sorties (écritures qui se chevauchent → non
        // parallélisable). Ici chaque ligne de sortie `p` rassemble ses
        // contributions des instants d'entrée `t` tels que `t*stride + k = p`,
        // visités dans le MÊME ordre (t croissant, in_ch croissant) que le
        // scatter, puis ajoute le biais en dernier → byte-identique.
        let mut out = vec![0.0_f32; out_time * out_dim];
        out.par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(p, out_row)| {
                let t_max = (p / stride).min(x_time - 1);
                let t_min = if p >= kernel {
                    (p - kernel) / stride + 1
                } else {
                    0
                };
                for out_ch in 0..out_dim {
                    let mut acc = 0.0_f32;
                    let mut t = t_min;
                    while t <= t_max {
                        let k = p - t * stride;
                        let xrow = &xdata[t * in_dim..t * in_dim + in_dim];
                        let wseg = &wkc[(out_ch * kernel + k) * in_dim..][..in_dim];
                        for in_ch in 0..in_dim {
                            acc += xrow[in_ch] * wseg[in_ch];
                        }
                        t += 1;
                    }
                    if let Some(b) = bias {
                        acc += b[out_ch];
                    }
                    out_row[out_ch] = acc;
                }
            });
        Nlc::new(out_time, out_dim, out)
    }

    fn causal_transpose1d_stream(
        &self,
        state: &mut CpuConvStreamState,
        x: &Nlc,
        weight_key: &str,
        bias_key: &str,
        stride: usize,
    ) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let [_, _, kernel] = conv_weight_shape(weight, weight_key)?;
        let keep_rows = kernel.saturating_sub(1);
        let context_time = state.context.as_ref().map_or(0, |context| context.time);
        let local = append_optional_nlc(state.context.as_ref(), x)?;
        let out = self.causal_transpose1d(&local, weight_key, bias_key, stride)?;
        let start = context_time.checked_mul(stride).ok_or_else(|| {
            InferError::Dimension(format!("offset transpose streaming déborde {weight_key}"))
        })?;
        let len = x.time.checked_mul(stride).ok_or_else(|| {
            InferError::Dimension(format!("taille transpose streaming déborde {weight_key}"))
        })?;
        let suffix = out.slice_rows(start, len)?;
        state.context = tail_nlc(&local, keep_rows)?;
        Ok(suffix)
    }

    fn snake_beta(&self, x: &Nlc, alpha_key: &str, beta_key: &str) -> Result<Nlc> {
        let alpha = self.weight(alpha_key)?;
        let beta = self.weight(beta_key)?;
        let alpha_s = channel_slice(alpha)?;
        let beta_s = channel_slice(beta)?;
        let chans = x.channels;
        if alpha_s.len() < chans || beta_s.len() < chans {
            return Err(InferError::Dimension(format!(
                "snake_beta canaux={chans} alpha={} beta={}",
                alpha_s.len(),
                beta_s.len()
            )));
        }
        // `exp(alpha[ch])` et `exp(beta[ch])` ne dépendent que du canal :
        // précalculés une fois (la version scalaire les recalculait à chaque
        // instant). Valeurs identiques → byte-identique.
        let a_exp: Vec<f32> = (0..chans).map(|ch| alpha_s[ch].exp()).collect();
        let b_exp: Vec<f32> = (0..chans).map(|ch| beta_s[ch].exp()).collect();
        let xdata = &x.data;
        let mut out = vec![0.0_f32; x.data.len()];
        out.par_chunks_mut(chans)
            .enumerate()
            .for_each(|(time, out_row)| {
                let xrow = &xdata[time * chans..time * chans + chans];
                for ch in 0..chans {
                    let value = xrow[ch];
                    out_row[ch] = value + (value * a_exp[ch]).sin().powi(2) / (b_exp[ch] + 1.0e-9);
                }
            });
        Nlc::new(x.time, chans, out)
    }

    fn rms_norm_nlc(&self, x: &Nlc, weight_key: &str, eps: f32) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let tensor = Tensor::from_vec(vec![x.time, x.channels], x.data.clone())?;
        let out = rms_norm(&tensor, weight, eps)?;
        Nlc::new(x.time, x.channels, out.into_data())
    }

    fn layer_norm_nlc(&self, x: &Nlc, weight_key: &str, bias_key: &str, eps: f32) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let bias = self.weight(bias_key)?;
        let mut out = vec![0.0_f32; x.data.len()];
        for time in 0..x.time {
            let row = x.row(time);
            let mean = row.iter().sum::<f32>() / x.channels as f32;
            let var = row
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / x.channels as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for ch in 0..x.channels {
                out[time * x.channels + ch] =
                    (row[ch] - mean) * inv * channel_value(weight, ch)? + channel_value(bias, ch)?;
            }
        }
        Nlc::new(x.time, x.channels, out)
    }

    fn mul_channel(&self, x: &Nlc, weight_key: &str) -> Result<Nlc> {
        let weight = self.weight(weight_key)?;
        let mut out = x.clone();
        for time in 0..x.time {
            for ch in 0..x.channels {
                out.set(time, ch, x.get(time, ch) * channel_value(weight, ch)?);
            }
        }
        Ok(out)
    }

    fn weight(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .ok_or_else(|| InferError::MissingWeight(key.to_string()))
    }

    fn opt(&self, key: &str) -> Option<&Tensor> {
        self.weights.get(key)
    }
}

/// PCM décodé + temps par sous-étape `(label, durée)` (sortie du profileur codec).
#[cfg(test)]
type ProfiledDecode = (Vec<f32>, Vec<(&'static str, std::time::Duration)>);

#[cfg(test)]
impl TtsCodec {
    /// Décode comme [`Self::decode_codes`] mais chronomètre chaque sous-étape.
    ///
    /// Outil de profil (ÉTAPE 1 du chantier perf codec) : le corps suit
    /// exactement `decode_codes` afin que les temps mesurés reflètent le chemin
    /// de production. Réservé aux tests (`#[cfg(test)]`).
    pub(crate) fn decode_codes_profiled(&self, generated: &[Vec<i32>]) -> Result<ProfiledDecode> {
        use std::time::Instant;
        let mut timings: Vec<(&'static str, std::time::Duration)> = Vec::new();
        if generated.is_empty() {
            return Ok((Vec::new(), timings));
        }
        macro_rules! timed {
            ($label:expr, $body:expr) => {{
                let start = Instant::now();
                let value = $body;
                timings.push(($label, start.elapsed()));
                value
            }};
        }

        let mut hidden = timed!("rvq_decode", self.rvq_decode(generated))?;
        hidden = timed!(
            "pre_conv",
            self.causal_conv1d(
                &hidden,
                "decoder.pre_conv.conv.weight",
                "decoder.pre_conv.conv.bias",
                1,
                1,
            )
        )?;
        hidden = timed!("pre_transformer", self.pre_transformer(&hidden))?;

        for (idx, factor) in self.cfg.upsampling_ratios.iter().copied().enumerate() {
            hidden = timed!(
                "upsample.transpose",
                self.causal_transpose1d(
                    &hidden,
                    &format!("decoder.upsample.{idx}.0.conv.weight"),
                    &format!("decoder.upsample.{idx}.0.conv.bias"),
                    usize_from_i32(factor, "upsampling ratio")?,
                )
            )?;
            hidden = timed!(
                "upsample.convnext",
                self.convnext_block(
                    &hidden,
                    &format!("decoder.upsample.{idx}.1"),
                    usize_from_i32(self.cfg.latent_dim, "latent_dim")?,
                )
            )?;
        }

        let mut wav = timed!(
            "decoder.0",
            self.causal_conv1d(
                &hidden,
                "decoder.decoder.0.conv.weight",
                "decoder.decoder.0.conv.bias",
                1,
                1,
            )
        )?;
        for (idx, rate) in self.cfg.upsample_rates.iter().copied().enumerate() {
            let block = idx + 1;
            wav = timed!(
                "blk_snake0",
                self.snake_beta(
                    &wav,
                    &format!("decoder.decoder.{block}.block.0.alpha"),
                    &format!("decoder.decoder.{block}.block.0.beta"),
                )
            )?;
            wav = timed!(
                "blk_transpose",
                self.causal_transpose1d(
                    &wav,
                    &format!("decoder.decoder.{block}.block.1.conv.weight"),
                    &format!("decoder.decoder.{block}.block.1.conv.bias"),
                    usize_from_i32(rate, "upsample rate")?,
                )
            )?;
            for (slot, dilation) in [(2_usize, 1_usize), (3, 3), (4, 9)] {
                // Décomposition du residual unit (mêmes appels que
                // decoder_residual_unit) pour attribuer le coût interne.
                let prefix = format!("decoder.decoder.{block}.block.{slot}");
                let residual = wav.clone();
                let h = timed!(
                    "ru_snake",
                    self.snake_beta(
                        &wav,
                        &format!("{prefix}.act1.alpha"),
                        &format!("{prefix}.act1.beta")
                    )
                )?;
                let h = timed!(
                    "ru_conv1",
                    self.causal_conv1d(
                        &h,
                        &format!("{prefix}.conv1.conv.weight"),
                        &format!("{prefix}.conv1.conv.bias"),
                        dilation,
                        1,
                    )
                )?;
                let h = timed!(
                    "ru_snake",
                    self.snake_beta(
                        &h,
                        &format!("{prefix}.act2.alpha"),
                        &format!("{prefix}.act2.beta")
                    )
                )?;
                let h = timed!(
                    "ru_conv2",
                    self.conv1d_nopad(
                        &h,
                        &format!("{prefix}.conv2.conv.weight"),
                        &format!("{prefix}.conv2.conv.bias"),
                        1
                    )
                )?;
                wav = residual.add(&h)?;
            }
        }
        wav = timed!(
            "tail_snake",
            self.snake_beta(&wav, "decoder.decoder.5.alpha", "decoder.decoder.5.beta")
        )?;
        wav = timed!(
            "tail_conv",
            self.causal_conv1d(
                &wav,
                "decoder.decoder.6.conv.weight",
                "decoder.decoder.6.conv.bias",
                1,
                1,
            )
        )?;
        if wav.channels != 1 {
            return Err(InferError::Dimension(format!(
                "codec wav attendu mono, reçu {} canaux",
                wav.channels
            )));
        }
        let pcm = wav.data.into_iter().map(|s| s.clamp(-1.0, 1.0)).collect();
        Ok((pcm, timings))
    }
}

impl Nlc {
    fn new(time: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        if time
            .checked_mul(channels)
            .ok_or_else(|| InferError::Shape("NLC trop grand".to_string()))?
            != data.len()
        {
            return Err(InferError::Shape(format!(
                "NLC [{time},{channels}] incompatible avec {} valeurs",
                data.len()
            )));
        }
        Ok(Self {
            time,
            channels,
            data,
        })
    }

    fn zeros(time: usize, channels: usize) -> Self {
        Self {
            time,
            channels,
            data: vec![0.0; time * channels],
        }
    }

    fn row(&self, time: usize) -> &[f32] {
        let start = time * self.channels;
        &self.data[start..start + self.channels]
    }

    fn slice_rows(&self, start: usize, len: usize) -> Result<Self> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| InferError::Shape("slice NLC déborde".to_string()))?;
        if end > self.time {
            return Err(InferError::Dimension(format!(
                "slice NLC [{start}..{end}] hors temps {}",
                self.time
            )));
        }
        let data_start = start
            .checked_mul(self.channels)
            .ok_or_else(|| InferError::Shape("slice NLC offset déborde".to_string()))?;
        let data_end = end
            .checked_mul(self.channels)
            .ok_or_else(|| InferError::Shape("slice NLC fin déborde".to_string()))?;
        Self::new(len, self.channels, self.data[data_start..data_end].to_vec())
    }

    fn get(&self, time: usize, channel: usize) -> f32 {
        self.data[time * self.channels + channel]
    }

    fn set(&mut self, time: usize, channel: usize, value: f32) {
        self.data[time * self.channels + channel] = value;
    }

    fn add(&self, rhs: &Self) -> Result<Self> {
        if self.time != rhs.time || self.channels != rhs.channels {
            return Err(InferError::Dimension(format!(
                "NLC add gauche=[{},{}] droite=[{},{}]",
                self.time, self.channels, rhs.time, rhs.channels
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(left, right)| left + right)
            .collect();
        Self::new(self.time, self.channels, data)
    }

    fn mul(&self, rhs: &Self) -> Result<Self> {
        if self.time != rhs.time || self.channels != rhs.channels {
            return Err(InferError::Dimension(format!(
                "NLC mul gauche=[{},{}] droite=[{},{}]",
                self.time, self.channels, rhs.time, rhs.channels
            )));
        }
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(left, right)| left * right)
            .collect();
        Self::new(self.time, self.channels, data)
    }

    fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        Self {
            time: self.time,
            channels: self.channels,
            data: self.data.iter().copied().map(f).collect(),
        }
    }

    /// Accumule un code RVQ dé-quantifié dans la trame `time` (`frame += code`).
    ///
    /// Reconstruction par somme des étages résiduels du décodeur RVQ : chaque
    /// étage `q` ajoute son code, la somme sur `0..n_q` approxime l'entrée.
    fn add_row_in_place(&mut self, time: usize, row: &[f32]) -> Result<()> {
        if row.len() != self.channels {
            return Err(InferError::Dimension(format!(
                "row len={} incompatible avec channels={}",
                row.len(),
                self.channels
            )));
        }
        let start = time * self.channels;
        for (idx, value) in row.iter().copied().enumerate() {
            self.data[start + idx] += value;
        }
        Ok(())
    }
}

fn append_optional_nlc(left: Option<&Nlc>, right: &Nlc) -> Result<Nlc> {
    let Some(left) = left else {
        return Ok(right.clone());
    };
    append_nlc(left, right)
}

fn append_nlc(left: &Nlc, right: &Nlc) -> Result<Nlc> {
    if left.channels != right.channels {
        return Err(InferError::Dimension(format!(
            "append NLC canaux gauche={} droite={}",
            left.channels, right.channels
        )));
    }
    let rows = left
        .time
        .checked_add(right.time)
        .ok_or_else(|| InferError::Shape("append NLC temps déborde".to_string()))?;
    let mut data = Vec::with_capacity(rows * left.channels);
    data.extend_from_slice(&left.data);
    data.extend_from_slice(&right.data);
    Nlc::new(rows, left.channels, data)
}

fn tail_nlc(tensor: &Nlc, keep_rows: usize) -> Result<Option<Nlc>> {
    if keep_rows == 0 || tensor.time == 0 {
        return Ok(None);
    }
    let keep = keep_rows.min(tensor.time);
    tensor.slice_rows(tensor.time - keep, keep).map(Some)
}

fn codebook_embed(usage: &Tensor, sum: &Tensor) -> Result<Tensor> {
    let rows = match usage.shape() {
        [n] => *n,
        shape => {
            return Err(InferError::Dimension(format!(
                "cluster_usage attendu [N], reçu {shape:?}"
            )));
        }
    };
    let (sum_rows, cols) = matrix_shape(sum)?;
    if rows != sum_rows {
        return Err(InferError::Dimension(format!(
            "cluster_usage rows={rows} incompatible embedding_sum rows={sum_rows}"
        )));
    }
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let denom = usage.data()[row].max(1.0e-5);
        for col in 0..cols {
            out[row * cols + col] = sum.data()[row * cols + col] / denom;
        }
    }
    Tensor::from_vec(vec![rows, cols], out)
}

/// Dé-quantifie un code RVQ : renvoie la ligne du codebook pointée par
/// l'index de code généré (le code est un indice, pas une valeur).
///
/// # Errors
///
/// Renvoie une erreur si `row` est négatif ou hors bornes du codebook.
fn row_i32(tensor: &Tensor, row: i32) -> Result<&[f32]> {
    let row = usize_from_i32(row, "codebook id")?;
    tensor.row_slice(row)
}

fn matrix_shape(tensor: &Tensor) -> Result<(usize, usize)> {
    tensor.as_matrix()
}

fn conv_weight_shape(weight: &Tensor, name: &str) -> Result<[usize; 3]> {
    match weight.shape() {
        [a, b, c] => Ok([*a, *b, *c]),
        shape => Err(InferError::Dimension(format!(
            "poids conv {name} attendu rang 3, reçu {shape:?}"
        ))),
    }
}

fn conv_source_time(time: usize, kernel_idx: usize, pad: usize, dilation: usize) -> Option<usize> {
    let offset = kernel_idx.checked_mul(dilation)?;
    (time + offset).checked_sub(pad)
}

fn channel_value(tensor: &Tensor, idx: usize) -> Result<f32> {
    match tensor.shape() {
        [n] if idx < *n => Ok(tensor.data()[idx]),
        [1, n] if idx < *n => Ok(tensor.data()[idx]),
        shape => Err(InferError::Dimension(format!(
            "paramètre canal idx={idx} incompatible shape={shape:?}"
        ))),
    }
}

/// Renvoie un paramètre par canal `[n]` ou `[1, n]` comme tranche contiguë.
///
/// Permet d'accéder aux biais/échelles dans les boucles parallèles sans refaire
/// le `match` de [`channel_value`] par élément.
///
/// # Errors
///
/// Renvoie une erreur si la forme n'est pas `[n]` ni `[1, n]`.
fn channel_slice(tensor: &Tensor) -> Result<&[f32]> {
    match tensor.shape() {
        [n] | [1, n] => Ok(&tensor.data()[..*n]),
        shape => Err(InferError::Dimension(format!(
            "paramètre canal incompatible shape={shape:?}"
        ))),
    }
}

fn rope_heads(x: &Nlc, heads: usize, head_dim: usize, base: f32) -> Result<Nlc> {
    rope_heads_at(x, heads, head_dim, base, 0)
}

fn rope_heads_at(
    x: &Nlc,
    heads: usize,
    head_dim: usize,
    base: f32,
    position_offset: usize,
) -> Result<Nlc> {
    if x.channels != heads * head_dim {
        return Err(InferError::Dimension(format!(
            "RoPE codec channels={} attendu {}",
            x.channels,
            heads * head_dim
        )));
    }
    let mut out = x.clone();
    let pairs = head_dim / 2;
    for time in 0..x.time {
        for pair in 0..pairs {
            let position = position_offset + time;
            let angle = position as f32 / base.powf((2 * pair) as f32 / head_dim as f32);
            let cos = angle.cos();
            let sin = angle.sin();
            for head in 0..heads {
                let start = head * head_dim;
                let left_idx = start + pair;
                let right_idx = start + pairs + pair;
                let left = x.get(time, left_idx);
                let right = x.get(time, right_idx);
                out.set(time, left_idx, left * cos - right * sin);
                out.set(time, right_idx, left * sin + right * cos);
            }
        }
    }
    Ok(out)
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn usize_from_i32(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} négatif: {value}")))
}

fn silu_scalar(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn gelu_scalar(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_approx(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests;
