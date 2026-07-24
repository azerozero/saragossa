use super::{require_file, QWEN3_TTS_SPECIAL_TOKENS};
use crate::{InferError, Result};
use std::path::Path;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

pub(super) fn load_qwen3_tts_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let tokenizer_json = dir.join("tokenizer.json");
    if tokenizer_json.is_file() {
        return Tokenizer::from_file(&tokenizer_json).map_err(|err| InferError::Tokenizer {
            path: tokenizer_json,
            message: err.to_string(),
        });
    }
    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    require_file(&vocab, "vocab.json")?;
    require_file(&merges, "merges.txt")?;
    let vocab_str = vocab
        .to_str()
        .ok_or_else(|| InferError::Config(format!("chemin vocab non UTF-8: {vocab:?}")))?;
    let merges_str = merges
        .to_str()
        .ok_or_else(|| InferError::Config(format!("chemin merges non UTF-8: {merges:?}")))?;
    let bpe = BPE::from_file(vocab_str, merges_str)
        .build()
        .map_err(|err| InferError::Tokenizer {
            path: dir.to_path_buf(),
            message: format!("build BPE Qwen3-TTS: {err}"),
        })?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, true, true)));
    let added = QWEN3_TTS_SPECIAL_TOKENS
        .iter()
        .map(|(content, special)| AddedToken::from(*content, *special))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&added);
    Ok(tokenizer)
}
