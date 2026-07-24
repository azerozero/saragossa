use super::{
    add_into, gather_rows_i32, push_rows, PreparedVoiceDesign, TtsCloneContext, TtsModel,
    DEFAULT_INSTRUCT,
};
use crate::{InferError, Result, Tensor};

impl TtsModel {
    pub(super) fn prepare_inputs(&self, text: &str) -> Result<PreparedVoiceDesign> {
        match self.clone_ctx.as_ref() {
            Some(ctx) => self.prepare_icl_inputs(text, ctx),
            None => self.prepare_voicedesign_inputs(text),
        }
    }

    pub(super) fn prepare_icl_inputs(
        &self,
        text: &str,
        ctx: &TtsCloneContext,
    ) -> Result<PreparedVoiceDesign> {
        let cfg = &self.assets.model_config.talker_config;
        let target_chat =
            format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let target_ids = self.encode_ids(&target_chat)?;
        if target_ids.len() < 9 {
            return Err(InferError::Config(
                "prompt cible clone trop court".to_string(),
            ));
        }
        let text_ids = &target_ids[3..target_ids.len() - 5];
        let mut combined_ids = Vec::with_capacity(ctx.ref_text_ids.len() + text_ids.len());
        combined_ids.extend_from_slice(&ctx.ref_text_ids);
        combined_ids.extend_from_slice(text_ids);

        let tts_ids = [
            self.assets.model_config.tts_bos_token_id,
            self.assets.model_config.tts_eos_token_id,
            self.assets.model_config.tts_pad_token_id,
        ];
        let tts_embeds = self.text_embed(&tts_ids)?;
        let tts_bos = tts_embeds.row_slice(0)?.to_vec();
        let tts_eos = tts_embeds.row_slice(1)?.to_vec();
        let tts_pad = tts_embeds.row_slice(2)?.to_vec();

        let mut text_embed = self.text_embed(&combined_ids)?;
        let hidden = self.hidden_dim()?;
        let mut text_rows = text_embed.into_data();
        text_rows.extend_from_slice(&tts_eos);
        text_embed = Tensor::from_vec(vec![text_rows.len() / hidden, hidden], text_rows)?;

        let codec_bos = self.codec_embed(&[cfg.codec_bos_id])?;
        let mut codec_icl_rows = codec_bos.into_data();
        if let Some(ref_codec) = ctx.ref_codec_embed.as_ref() {
            codec_icl_rows.extend_from_slice(ref_codec.data());
        }
        let codec_icl =
            Tensor::from_vec(vec![codec_icl_rows.len() / hidden, hidden], codec_icl_rows)?;

        let codec_pad = self.codec_embed(&[cfg.codec_pad_id])?;
        let codec_pad = codec_pad.as_row()?.to_vec();
        let mut icl_rows =
            Vec::with_capacity((text_embed.shape()[0] + codec_icl.shape()[0]) * hidden);
        for row in 0..text_embed.shape()[0] {
            let mut item = text_embed.row_slice(row)?.to_vec();
            add_into(&mut item, &codec_pad);
            icl_rows.extend_from_slice(&item);
        }
        for row in 0..codec_icl.shape()[0] {
            let mut item = codec_icl.row_slice(row)?.to_vec();
            add_into(&mut item, &tts_pad);
            icl_rows.extend_from_slice(&item);
        }

        let lang_id = cfg.codec_language_id.get("french").copied();
        let codec_prefill: Vec<i32> = match lang_id {
            Some(lid) => vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                lid,
                cfg.codec_think_eos_id,
            ],
            None => vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ],
        };
        let mut codec_prefix_rows = self.codec_embed(&codec_prefill)?.into_data();
        codec_prefix_rows.extend_from_slice(ctx.speaker_embed.as_row()?);
        codec_prefix_rows.extend_from_slice(
            self.codec_embed(&[cfg.codec_pad_id, cfg.codec_bos_id])?
                .data(),
        );
        let codec_prefix = Tensor::from_vec(
            vec![codec_prefix_rows.len() / hidden, hidden],
            codec_prefix_rows,
        )?;

        let role_embed = self.text_embed(&target_ids[..3])?;
        let prefix_len = codec_prefix.shape()[0];
        if prefix_len < 2 {
            return Err(InferError::Config(
                "préfixe codec clone trop court".to_string(),
            ));
        }
        let mut rows = Vec::new();
        push_rows(&mut rows, &role_embed, 0, role_embed.shape()[0])?;
        for row in 0..(prefix_len - 1) {
            let mut item = if row + 1 == prefix_len - 1 {
                tts_bos.clone()
            } else {
                tts_pad.clone()
            };
            add_into(&mut item, codec_prefix.row_slice(row)?);
            rows.extend_from_slice(&item);
        }
        rows.extend_from_slice(&icl_rows);

        Ok(PreparedVoiceDesign {
            input: Tensor::from_vec(vec![rows.len() / hidden, hidden], rows)?,
            trailing: Tensor::row(tts_pad.clone())?,
            tts_pad: Tensor::row(tts_pad)?,
        })
    }

    pub(super) fn ref_codec_embed_sum(&self, ref_codes: &[Vec<i32>]) -> Result<Tensor> {
        let n_groups = usize::try_from(self.assets.model_config.talker_config.num_code_groups)
            .map_err(|_| InferError::Config("num_code_groups TTS négatif".to_string()))?;
        if ref_codes.iter().any(|frame| frame.len() < n_groups) {
            return Err(InferError::Dimension(
                "ref_codes clone sans tous les codebooks".to_string(),
            ));
        }
        let cb0 = ref_codes.iter().map(|frame| frame[0]).collect::<Vec<_>>();
        let mut acc = self.codec_embed(&cb0)?;
        for codebook in 1..n_groups {
            let ids = ref_codes
                .iter()
                .map(|frame| frame[codebook])
                .collect::<Vec<_>>();
            acc = acc.add(&self.code_predictor_embed(codebook - 1, &ids)?)?;
        }
        Ok(acc)
    }

    pub(super) fn prepare_voicedesign_inputs(&self, text: &str) -> Result<PreparedVoiceDesign> {
        let cfg = &self.assets.model_config.talker_config;
        let chat = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let ids = self.encode_ids(&chat)?;
        if ids.len() < 9 {
            return Err(InferError::Config(format!(
                "prompt TTS trop court après tokenisation: {} tokens",
                ids.len()
            )));
        }
        let text_embed = self.text_embed(&ids)?;

        let tts_ids = [
            self.assets.model_config.tts_bos_token_id,
            self.assets.model_config.tts_eos_token_id,
            self.assets.model_config.tts_pad_token_id,
        ];
        let tts_embeds = self.text_embed(&tts_ids)?;
        let tts_bos = tts_embeds.row_slice(0)?.to_vec();
        let tts_pad = tts_embeds.row_slice(2)?.to_vec();

        let lang_id = cfg.codec_language_id.get("french").copied();
        let codec_prefill: Vec<i32> = match lang_id {
            Some(lid) => vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                lid,
                cfg.codec_think_eos_id,
            ],
            None => vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ],
        };
        let mut codec_ids = codec_prefill;
        codec_ids.push(cfg.codec_pad_id);
        codec_ids.push(cfg.codec_bos_id);
        let codec_embed = self.codec_embed(&codec_ids)?;
        let codec_len = codec_embed.shape()[0];
        let hidden = self.hidden_dim()?;
        if codec_len < 2 {
            return Err(InferError::Config(
                "préfixe codec TTS trop court".to_string(),
            ));
        }

        let instruct = DEFAULT_INSTRUCT;
        let instruct_chat = format!("<|im_start|>user\n{instruct}<|im_end|>\n");
        let instruct_ids = self.encode_ids(&instruct_chat)?;
        let instruct_embed = self.text_embed(&instruct_ids)?;

        let mut rows = Vec::new();
        push_rows(&mut rows, &instruct_embed, 0, instruct_embed.shape()[0])?;
        push_rows(&mut rows, &text_embed, 0, 3)?;

        for i in 0..(codec_len - 1) {
            let mut row = if i + 1 == codec_len - 1 {
                tts_bos.clone()
            } else {
                tts_pad.clone()
            };
            add_into(&mut row, codec_embed.row_slice(i)?);
            rows.extend_from_slice(&row);
        }

        let mut first_text = text_embed.row_slice(3)?.to_vec();
        add_into(&mut first_text, codec_embed.row_slice(codec_len - 1)?);
        rows.extend_from_slice(&first_text);

        let trailing_start = 4;
        let trailing_end = ids
            .len()
            .checked_sub(5)
            .ok_or_else(|| InferError::Config("prompt TTS trop court pour trailing".to_string()))?;
        let mut trailing_rows = Vec::new();
        if trailing_start < trailing_end {
            push_rows(
                &mut trailing_rows,
                &text_embed,
                trailing_start,
                trailing_end,
            )?;
        }
        trailing_rows.extend_from_slice(tts_embeds.row_slice(1)?);

        Ok(PreparedVoiceDesign {
            input: Tensor::from_vec(vec![rows.len() / hidden, hidden], rows)?,
            trailing: Tensor::from_vec(vec![trailing_rows.len() / hidden, hidden], trailing_rows)?,
            tts_pad: Tensor::row(tts_pad)?,
        })
    }

    pub(super) fn text_embed(&self, ids: &[i32]) -> Result<Tensor> {
        let raw = gather_rows_i32(&self.text_embedding, ids)?;
        self.text_projection_fc2
            .forward(&crate::silu(&self.text_projection_fc1.forward(&raw)?))
    }

    pub(super) fn codec_embed(&self, ids: &[i32]) -> Result<Tensor> {
        gather_rows_i32(&self.codec_embedding, ids)
    }

    pub(super) fn code_predictor_embed(&self, codebook: usize, ids: &[i32]) -> Result<Tensor> {
        let table = self
            .code_predictor_embeddings
            .get(codebook)
            .ok_or_else(|| {
                InferError::MissingWeight(format!(
                    "code_predictor.model.codec_embedding.{codebook}"
                ))
            })?;
        gather_rows_i32(table, ids)
    }

    pub(super) fn encode_ids(&self, text: &str) -> Result<Vec<i32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| InferError::Tokenizer {
                path: self.assets.model_dir.clone(),
                message: err.to_string(),
            })?;
        enc.get_ids()
            .iter()
            .map(|id| {
                i32::try_from(*id)
                    .map_err(|_| InferError::Config(format!("token id TTS hors i32: {id}")))
            })
            .collect()
    }

    pub(super) fn hidden_dim(&self) -> Result<usize> {
        usize::try_from(self.assets.model_config.talker_config.hidden_size)
            .map_err(|_| InferError::Config("hidden_size TTS négatif".to_string()))
    }
}
