//! Lecture « lettres » : chaque option reçoit un label (`A`, `B`…) et la
//! décision lit, au dernier token du prompt, les logits de la tête LM pour
//! ces seuls labels ; une température par type de question les étalonne.
//!
//! Seul le rendu H2O-Lightning-4B (h2oai, Apache-2.0) est pris en charge. La
//! lecture (dernier token, modes bidirectionnel et causal, cache de préfixe
//! causal) et l'étalonnage (lignes de labels, `softmax(z / T)`) ne dépendent
//! pas du prompt ; Sieve-31B ou openjev y ajouteront leur rendu et leurs
//! lignes (cf. `docs/decision-models.md`).

use super::{
    calibrated_readout, describe, invalid, load_headless, neutralize, read_json, special_tokens,
    truthy,
};
use crate::tokenizer::QWEN35_REGEX;
use crate::{CausalDecoder, InferError, Result, RustTokenizer, Tensor};
use serde_json::{json, Map, Value};
use std::path::Path;

const TOKENIZER_CONTRACT_SHA256: &str =
    "7eb96aaea6b7dbb495f0840b42f18014ca5a697697305f76f39d46cae890ccb2";
const LABEL_ROWS_FORMAT: &str = "lightning-label-rows-1";
/// Regex de `Qwen2Tokenizer`, avec laquelle H2O et d3 ont été entraînés ;
/// elle diverge de [`QWEN35_REGEX`] sur les marques combinantes.
use crate::tokenizer::QWEN2_REGEX;
/// Le chemin de décision non causal est validé jusqu'à 2048 tokens.
const MAX_BIDIRECTIONAL_TOKENS: usize = 2048;
/// `MAX_LEN` du `serve.sh` H2O (état de 32 000 tokens plus la question).
const MAX_CAUSAL_TOKENS: usize = 40_960;
const CHATML_GENERATION: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Masque des couches d'attention complète ; les gated-delta restent causales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMode {
    /// Chaque position voit tout le prompt (« two-way »), sans cache de préfixe.
    Bidirectional,
    /// Masque causal (« one-way ») : le KV d'un préfixe commun est réutilisable.
    Causal,
}

impl AttentionMode {
    fn parse(value: Option<&Value>, default: Self) -> Result<Self> {
        match value.map(|v| v.as_str().map(|s| s.trim().to_ascii_lowercase())) {
            None => Ok(default),
            Some(Some(name)) => match name.as_str() {
                "bidirectional" | "two-way" | "two_way" | "twoway" => Ok(Self::Bidirectional),
                "causal" | "one-way" | "one_way" | "oneway" => Ok(Self::Causal),
                _ => Err(invalid("attention_mode : bidirectional ou causal")),
            },
            Some(None) => Err(invalid("attention_mode : chaîne attendue")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bidirectional => "bidirectional",
            Self::Causal => "causal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Choice,
    Noul,
    Score,
}

/// Températures par type et plancher oui/non (`commit_noul`) d'un mode.
#[derive(Clone, Copy, Debug)]
struct Calibration {
    choice: f32,
    noul: f32,
    score: f32,
    noul_floor: f64,
}

impl Calibration {
    fn parse(temperature: &Value, floor: &Value) -> Result<Self> {
        let get = |kind: &str| -> Result<f32> {
            let t = match temperature {
                Value::Object(map) if map.len() == 3 => map.get(kind).and_then(Value::as_f64),
                Value::Object(_) => None,
                value => value.as_f64(),
            }
            .ok_or_else(|| invalid("température par type : choice, noul et score"))?;
            if !(0.05..=20.0).contains(&t) {
                return Err(invalid("température hors de [0,05, 20]"));
            }
            Ok(t as f32)
        };
        let noul_floor = floor.as_f64().unwrap_or(f64::NAN);
        if !(noul_floor == 0.0 || (noul_floor > 0.5 && noul_floor < 1.0)) {
            return Err(invalid("noul_floor : 0 ou dans ]0,5, 1["));
        }
        Ok(Self {
            choice: get("choice")?,
            noul: get("noul")?,
            score: get("score")?,
            noul_floor,
        })
    }

    fn temperature(&self, kind: Kind) -> f32 {
        match kind {
            Kind::Choice => self.choice,
            Kind::Noul => self.noul,
            Kind::Score => self.score,
        }
    }
}

/// Options d'une question : noms rendus et descriptions, dans l'ordre.
#[derive(Debug, PartialEq)]
struct Options {
    kind: Kind,
    names: Vec<String>,
    descriptions: Vec<String>,
}

/// Séquence d'une question, lue au dernier token.
#[derive(Debug)]
pub struct LetterRead {
    pub text: String,
    pub tokens: Vec<usize>,
    kind: Kind,
    count: usize,
}

/// Modèle de décision à lecture de lettres (H2O-Lightning-4B).
pub struct LetterModel {
    decoder: CausalDecoder,
    tokenizer: RustTokenizer,
    /// Lignes `[labels, hidden]` f32 de la tête LM, dans l'ordre des labels.
    rows: Tensor,
    labels: Vec<String>,
    escapes: Vec<String>,
    system: Option<String>,
    prefill: String,
    default_instructions: String,
    default_mode: AttentionMode,
    bidirectional: Calibration,
    causal: Calibration,
}

impl LetterModel {
    /// Charge un dossier H2O-Lightning (`serve_config.json`,
    /// `label_rows.lightning`, tokenizer et safetensors texte). Aucun code du
    /// checkpoint n'est exécuté.
    ///
    /// # Errors
    /// Refuse toute variante de prompt, de tokenizer ou de lignes non vérifiée.
    pub fn load(path: impl AsRef<Path>, metal: bool) -> Result<Self> {
        let path = path.as_ref();
        let serve = read_json(&path.join("serve_config.json"))?;
        let prompt = &serve["prompt"];
        if prompt["chat_template"] != true
            || prompt["enable_thinking"] != false
            || prompt["layout"] != "plain"
            || prompt["state_compact"] != true
        {
            return Err(invalid(
                "prompt H2O non pris en charge (chat, plain, compact)",
            ));
        }
        let text = |value: &Value| value.as_str().filter(|s| !s.is_empty()).map(str::to_owned);
        let prefill = text(&prompt["prefill"]).ok_or_else(|| invalid("prefill absent"))?;
        let default_instructions = text(&prompt["default_instructions"])
            .unwrap_or_else(|| "Answer the question below.".into());
        let max_options = serve["max_options"].as_u64().unwrap_or(u64::MAX);
        let labels: Vec<String> = serde_json::from_value::<Vec<String>>(serve["labels"].clone())
            .map_err(|e| invalid(format!("labels : {e}")))?
            .into_iter()
            .take(usize::try_from(max_options).unwrap_or(usize::MAX))
            .collect();
        let two_way = serve["two_way"]
            .as_object()
            .ok_or_else(|| invalid("bloc two_way absent"))?;
        let default_mode = AttentionMode::parse(
            Some(two_way.get("default").unwrap_or(&json!("bidirectional"))),
            AttentionMode::Bidirectional,
        )?;
        let causal = Calibration::parse(&serve["temperature"], &serve["noul_floor"])?;
        let bidirectional = Calibration::parse(
            two_way.get("temperature").unwrap_or(&serve["temperature"]),
            two_way.get("noul_floor").unwrap_or(&serve["noul_floor"]),
        )?;
        let rows_name = two_way
            .get("label_rows")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty() && !n.contains('/') && !n.contains(".."))
            .ok_or_else(|| invalid("two_way.label_rows : nom de fichier du dossier requis"))?;
        let (rows, ids) = label_rows(&path.join(rows_name), &labels)?;
        let tokenizer = h2o_tokenizer(path)?;
        for (label, &id) in labels.iter().zip(&ids) {
            if tokenizer.encode(&format!(" {label}"))? != [id] {
                return Err(invalid(format!("label {label:?} : token {id} attendu")));
            }
        }
        let escapes = special_tokens(path, &[])?;
        let (decoder, hidden) = load_headless(path, &tokenizer, metal)?;
        if rows.shape()[1] != hidden {
            return Err(invalid("lignes de labels de largeur incompatible"));
        }
        Ok(Self {
            decoder,
            tokenizer,
            rows,
            labels,
            escapes,
            system: text(&prompt["system"]).map(|s| s.trim().to_owned()),
            prefill,
            default_instructions,
            default_mode,
            bidirectional,
            causal,
        })
    }

    fn escape(&self, value: &Value) -> Value {
        match value {
            Value::String(s) => Value::String(self.escape_text(s)),
            Value::Array(a) => Value::Array(a.iter().map(|v| self.escape(v)).collect()),
            Value::Object(o) => Value::Object(
                o.iter()
                    .map(|(k, v)| (self.escape_text(k), self.escape(v)))
                    .collect(),
            ),
            _ => value.clone(),
        }
    }

    fn escape_text(&self, text: &str) -> String {
        neutralize(text, &self.escapes)
    }

    /// Mode demandé par une requête (`attention_mode`), sinon celui du modèle.
    ///
    /// # Errors
    /// Refuse un mode inconnu.
    pub fn mode(&self, request: &Value) -> Result<AttentionMode> {
        AttentionMode::parse(request.get("attention_mode"), self.default_mode)
    }

    /// Une séquence par question, dans l'ordre JSON (`Contract.question_prompt`).
    ///
    /// # Errors
    /// Refuse une requête hors contrat, une question invalide ou trop longue.
    pub fn encode(&self, request: &Value) -> Result<Vec<LetterRead>> {
        let (state, questions) = envelope(request)?;
        let limit = match self.mode(request)? {
            AttentionMode::Bidirectional => MAX_BIDIRECTIONAL_TOKENS,
            AttentionMode::Causal => MAX_CAUSAL_TOKENS,
        };
        let state = match self.escape(state) {
            Value::String(s) => s,
            other => other.to_string(),
        };
        questions
            .values()
            .map(|question| {
                let (user, options) = user_turn(
                    &state,
                    &self.escape(question),
                    &self.labels,
                    &self.default_instructions,
                )?;
                let system = self.system.as_ref().map_or_else(String::new, |s| {
                    format!("<|im_start|>system\n{s}<|im_end|>\n")
                });
                let text = format!(
                    "{system}<|im_start|>user\n{}<|im_end|>\n{CHATML_GENERATION}{}",
                    user.trim(),
                    self.prefill
                );
                let tokens: Vec<usize> = self
                    .tokenizer
                    .encode(&text)?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect();
                if tokens.len() > limit {
                    return Err(invalid(format!(
                        "{} tokens, maximum {limit} dans ce mode, sans troncature",
                        tokens.len()
                    )));
                }
                Ok(LetterRead {
                    text,
                    tokens,
                    kind: options.kind,
                    count: options.names.len(),
                })
            })
            .collect()
    }

    /// Hidden normés au dernier token de chaque séquence. En causal avec
    /// `share_prefix`, le préfixe commun ([`Self::exact_prefix`]) est préfillé
    /// une fois et chaque lecture prolonge une copie de son état (KV et
    /// récurrence gated-delta) : mêmes bits qu'un prefill complet.
    fn last_hidden(
        &self,
        reads: &[&[usize]],
        mode: AttentionMode,
        share_prefix: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if reads.iter().any(|t| t.is_empty()) {
            return Err(invalid("séquence vide"));
        }
        match mode {
            AttentionMode::Bidirectional => {
                let limit = crate::runtime_flags::decision_pack_tokens();
                let mut out = Vec::with_capacity(reads.len());
                let mut start = 0;
                while start < reads.len() {
                    // Lectures empaquetées par prefills d'au plus `limit` tokens.
                    let mut end = start + 1;
                    let mut total = reads[start].len();
                    while end < reads.len() && total + reads[end].len() <= limit {
                        total += reads[end].len();
                        end += 1;
                    }
                    let last: Vec<[usize; 1]> =
                        reads[start..end].iter().map(|t| [t.len() - 1]).collect();
                    let pack: Vec<(&[usize], &[usize])> = reads[start..end]
                        .iter()
                        .zip(&last)
                        .map(|(t, l)| (*t, &l[..]))
                        .collect();
                    for hidden in self.decoder.decision_hidden_packed(&pack)? {
                        out.push(hidden.row_slice(0)?.to_vec());
                    }
                    start = end;
                }
                Ok(out)
            }
            AttentionMode::Causal => {
                let shared = if share_prefix {
                    self.exact_prefix(reads)
                } else {
                    0
                };
                if shared == 0 {
                    return reads
                        .iter()
                        .map(|t| {
                            let state = self.decoder.prefill_prompt_state_uncached(t)?;
                            Ok(state.final_state().data().to_vec())
                        })
                        .collect();
                }
                let base = self
                    .decoder
                    .prefill_prompt_state_uncached(&reads[0][..shared])?;
                let snapshot = self.decoder.snapshot_prompt_state_metal(&base)?;
                reads
                    .iter()
                    .map(|t| {
                        let mut state = base.clone();
                        let copy = self.decoder.copy_prompt_state_metal_snapshot(&snapshot)?;
                        self.decoder.restore_prompt_state_metal(&mut state, copy)?;
                        self.decoder.extend_prompt_state(&mut state, &t[shared..])?;
                        Ok(state.final_state().data().to_vec())
                    })
                    .collect()
            }
        }
    }

    /// Préfixe partageable sans changer un bit : le prolongement résident
    /// refait alors les calculs d'un prefill complet, quel que soit
    /// l'alignement du préfixe ou le découpage en morceaux (mesuré sur H2O,
    /// préfixes de 3879 à 7910 tokens). Deux conditions : le prolongement doit
    /// passer par le prefill résident (sous `prefix_extend_resident_min_tokens`,
    /// 32, il retombe sur l'extension en lignes, qui diffère), et
    /// chaque passe (préfixe, suffixes) doit compter au moins
    /// `DENSE_BF16_GEMM_MIN_BATCH` lignes, sous lesquelles les projections
    /// passent du GEMM bf16 au qmv (63 lignes : écart 4,8e-3 sur le hidden).
    /// 0 : pas de partage (CPU, prolongement non résident, suffixes courts).
    fn exact_prefix(&self, reads: &[&[usize]]) -> usize {
        let longest = reads.iter().map(|t| t.len()).max().unwrap_or(0);
        if reads.len() < 2 || !self.decoder.prefix_extend_as_cheap_as_cold(longest) {
            return 0;
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let rows = crate::metal_backend::DENSE_BF16_GEMM_MIN_BATCH
            .max(crate::runtime_flags::prefix_extend_resident_min_tokens());
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let rows = usize::MAX;
        bounded_prefix(reads, rows)
    }

    /// Probabilités étalonnées de chaque lecture (oracles et mesures).
    ///
    /// # Errors
    /// Propage les erreurs du prefill et du readout.
    pub fn probabilities(
        &self,
        reads: &[LetterRead],
        mode: AttentionMode,
        share_prefix: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let tokens: Vec<&[usize]> = reads.iter().map(|r| &r.tokens[..]).collect();
        let calibration = match mode {
            AttentionMode::Bidirectional => &self.bidirectional,
            AttentionMode::Causal => &self.causal,
        };
        self.last_hidden(&tokens, mode, share_prefix)?
            .iter()
            .zip(reads)
            .map(|(hidden, read)| {
                calibrated_readout(
                    hidden,
                    &self.rows,
                    read.count,
                    calibration.temperature(read.kind),
                )
            })
            .collect()
    }

    /// Réponses au format du shim H2O (`format_answer`), avec le mode lu.
    /// En causal, les questions partagent le cache du préfixe commun.
    ///
    /// # Errors
    /// Refuse la requête entière si une question est invalide (422 du shim).
    pub fn decide(&self, request: &Value) -> Result<Value> {
        let mode = self.mode(request)?;
        let reads = self.encode(request)?;
        let probabilities = self.probabilities(&reads, mode, true)?;
        let (_, questions) = envelope(request)?;
        let floor = match mode {
            AttentionMode::Bidirectional => self.bidirectional.noul_floor,
            AttentionMode::Causal => self.causal.noul_floor,
        };
        let mut answers = Map::new();
        for ((key, question), p) in questions.iter().zip(&probabilities) {
            let options = options(question, self.labels.len())?;
            answers.insert(key.clone(), answer(&options, p, floor, threshold(question)));
        }
        let tokens: Vec<&[usize]> = reads.iter().map(|r| &r.tokens[..]).collect();
        let total: usize = tokens.iter().map(|t| t.len()).sum();
        let shared = if tokens.len() > 1 {
            common_prefix(&tokens)
        } else {
            0
        };
        let input = total - shared * (tokens.len() - 1);
        let cached = if mode == AttentionMode::Causal {
            self.exact_prefix(&tokens)
        } else {
            0
        };
        let processed = if mode == AttentionMode::Causal {
            total - cached * (tokens.len() - 1)
        } else {
            total
        };
        Ok(json!({"answers":answers,"attention_mode":mode.name(),
            "usage":{"input_tokens":input,"processed_input_tokens":processed,
                     "output_tokens":reads.len(),"sequences":reads.len()}}))
    }
}

fn envelope(request: &Value) -> Result<(&Value, &Map<String, Value>)> {
    let object = request
        .as_object()
        .ok_or_else(|| invalid("requête objet requise"))?;
    if object
        .keys()
        .any(|k| !["state", "questions", "attention_mode"].contains(&k.as_str()))
    {
        return Err(invalid(
            "seuls state, questions et attention_mode sont supportés (texte, sans images ni box)",
        ));
    }
    let state = object.get("state").ok_or_else(|| invalid("state requis"))?;
    let questions = object
        .get("questions")
        .and_then(Value::as_object)
        .filter(|q| !q.is_empty())
        .ok_or_else(|| invalid("questions doit être un objet non vide"))?;
    Ok((state, questions))
}

/// Lignes `W` f32 et `token_ids` d'un fichier `lightning-label-rows-1`
/// (safetensors renommé), restreintes aux labels du contrat.
fn label_rows(path: &Path, labels: &[String]) -> Result<(Tensor, Vec<u32>)> {
    let bytes = std::fs::read(path).map_err(|source| InferError::Io {
        path: path.into(),
        source,
    })?;
    let parse = |source| InferError::Safetensors {
        path: path.into(),
        source,
    };
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&bytes).map_err(parse)?;
    let meta = metadata
        .metadata()
        .as_ref()
        .ok_or_else(|| invalid("lignes de labels sans métadonnées"))?;
    if meta.get("format").map(String::as_str) != Some(LABEL_ROWS_FORMAT)
        || meta.get("softcap").map(String::as_str) != Some("none")
    {
        return Err(invalid(
            "lignes de labels : format lightning-label-rows-1 sans softcap requis",
        ));
    }
    let groups: Vec<(String, Vec<i64>)> = meta
        .get("groups")
        .and_then(|g| serde_json::from_str(g).ok())
        .ok_or_else(|| invalid("lignes de labels : groupes illisibles"))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(parse)?;
    let weight = tensors
        .tensor("W")
        .map_err(|_| InferError::MissingWeight("W".into()))?;
    let ids = tensors
        .tensor("token_ids")
        .map_err(|_| InferError::MissingWeight("token_ids".into()))?;
    let ids: Vec<i64> = ids
        .data()
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap_or_default()))
        .collect();
    let [rows, hidden] = weight.shape() else {
        return Err(invalid("W : matrice attendue"));
    };
    // Un token par label, dans l'ordre des labels du contrat.
    if weight.dtype() != safetensors::Dtype::F32
        || *rows != ids.len()
        || groups.len() != ids.len()
        || labels.len() > ids.len()
        || groups
            .iter()
            .zip(&ids)
            .any(|((_, group), id)| group.as_slice() != [*id])
        || groups
            .iter()
            .zip(labels)
            .any(|((label, _), want)| label != want)
    {
        return Err(invalid(
            "lignes de labels incompatibles avec les labels du contrat",
        ));
    }
    let data: Vec<f32> = weight
        .data()
        .chunks_exact(4)
        .take(labels.len() * hidden)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap_or_default()))
        .collect();
    if data.iter().any(|x| !x.is_finite()) {
        return Err(invalid("lignes de labels non finies"));
    }
    let ids = ids[..labels.len()]
        .iter()
        .map(|&id| u32::try_from(id).map_err(|_| invalid("token de label invalide")))
        .collect::<Result<_>>()?;
    Ok((Tensor::from_vec(vec![labels.len(), *hidden], data)?, ids))
}

/// Tokenizer d'entraînement : `Qwen2Tokenizer` découpe avec la regex Qwen2,
/// pas avec celle de `tokenizer.json`. Les sondes de `tokenizer_contract.json`
/// doivent toutes passer.
fn h2o_tokenizer(path: &Path) -> Result<RustTokenizer> {
    let contract_path = path.join("tokenizer_contract.json");
    let raw = std::fs::read(&contract_path).map_err(|source| InferError::Io {
        path: contract_path.clone(),
        source,
    })?;
    // Contrat publié avec H2O-Lightning (`TOKENIZER_CONTRACT_SHA256` du shim).
    let digest: String = <sha2::Sha256 as sha2::Digest>::digest(&raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if digest != TOKENIZER_CONTRACT_SHA256 {
        return Err(invalid("tokenizer_contract.json différent de celui publié"));
    }
    let contract: Value = serde_json::from_slice(&raw).map_err(|source| InferError::Json {
        path: contract_path,
        source,
    })?;
    if contract["actual_served_tokenizer_class"] != "Qwen2Tokenizer" {
        return Err(invalid("tokenizer servi Qwen2Tokenizer requis"));
    }
    let tokenizer = qwen2_tokenizer(path)?;
    let probes = contract["probes"]
        .as_array()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| invalid("sondes du tokenizer absentes"))?;
    for probe in probes {
        let expected: Vec<u32> = serde_json::from_value(probe["token_ids"].clone())
            .map_err(|e| invalid(e.to_string()))?;
        let text = probe["text"]
            .as_str()
            .ok_or_else(|| invalid("sonde sans texte"))?;
        if tokenizer.encode(text)? != expected {
            return Err(invalid(format!(
                "sonde {} du tokenizer différente",
                probe["id"]
            )));
        }
    }
    Ok(tokenizer)
}

/// `tokenizer.json` d'un dossier Qwen3.5 relu avec la regex de
/// `Qwen2Tokenizer`, que transformers 5.x (avant 5.18) impose à la place de
/// celle du fichier ; les modèles entraînés sous cette classe en dépendent.
pub(super) fn qwen2_tokenizer(path: &Path) -> Result<RustTokenizer> {
    let file = path.join("tokenizer.json");
    let mut spec = read_json(&file)?;
    let regex = spec
        .pointer_mut("/pre_tokenizer/pretokenizers/0/pattern/Regex")
        .filter(|regex| *regex == QWEN35_REGEX)
        .ok_or_else(|| invalid("pré-tokenizer Qwen3.5 attendu dans tokenizer.json"))?;
    *regex = json!(QWEN2_REGEX);
    RustTokenizer::from_json(&file, &spec.to_string())
}

/// Préfixe commun laissant au moins `rows` tokens à chaque lecture et en
/// comptant lui-même au moins `rows`, sinon 0.
fn bounded_prefix(reads: &[&[usize]], rows: usize) -> usize {
    let shortest = reads.iter().map(|t| t.len()).min().unwrap_or(0);
    let shared = common_prefix(reads).min(shortest.saturating_sub(rows));
    if shared >= rows {
        shared
    } else {
        0
    }
}

/// Préfixe de tokens commun, en laissant au moins un token à chaque lecture.
fn common_prefix(reads: &[&[usize]]) -> usize {
    let Some(first) = reads.first() else {
        return 0;
    };
    let limit = reads
        .iter()
        .map(|t| t.len().saturating_sub(1))
        .min()
        .unwrap_or(0);
    (0..limit)
        .find(|&i| reads.iter().any(|t| t[i] != first[i]))
        .unwrap_or(limit)
}

/// `json.dumps(value, indent=1)` de Python : ASCII échappé en `\uXXXX`.
fn python_indent1(value: &Value) -> Result<String> {
    let mut out = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    serde::Serialize::serialize(value, &mut serializer).map_err(|e| invalid(e.to_string()))?;
    let text = String::from_utf8_lossy(&out);
    let mut ascii = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            ascii.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0; 2]) {
                ascii.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(ascii)
}

/// `options_of` du shim.
fn options(question: &Value, max: usize) -> Result<Options> {
    let criteria = question.get("criteria");
    let kind = match question.get("type").map(Value::as_str) {
        None | Some(Some("choice")) => Kind::Choice,
        Some(Some("noul")) => Kind::Noul,
        Some(Some("score")) => Kind::Score,
        Some(Some("box")) => {
            return Err(invalid(
                "box : génération guidée, hors périmètre de la lecture de lettres",
            ))
        }
        _ => return Err(invalid("type attendu : choice, noul ou score")),
    };
    let (names, descriptions): (Vec<String>, Vec<String>) = match kind {
        Kind::Noul => {
            let empty = Map::new();
            let criteria = match criteria.filter(|v| truthy(v)) {
                None => &empty,
                Some(Value::Object(c)) => c,
                Some(_) => return Err(invalid("noul : critères objet true/false requis")),
            };
            let side = |key: &str, alias: &str, default: &str| match criteria
                .get(key)
                .or_else(|| criteria.get(alias))
            {
                None | Some(Value::Null) => default.to_owned(),
                Some(Value::String(s)) if s.is_empty() => default.to_owned(),
                Some(v) => describe(v),
            };
            (
                vec!["true".into(), "false".into()],
                vec![
                    side("true", "yes", "the statement holds"),
                    side("false", "no", "it does not"),
                ],
            )
        }
        Kind::Score => {
            let levels = criteria
                .and_then(Value::as_array)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| invalid("score : liste ordonnée de niveaux requise"))?;
            (
                (0..levels.len()).map(|i| i.to_string()).collect(),
                levels.iter().map(describe).collect(),
            )
        }
        Kind::Choice => {
            let criteria = criteria
                .and_then(Value::as_object)
                .filter(|c| !c.is_empty())
                .ok_or_else(|| invalid("choice : critères objet option -> description requis"))?;
            (
                criteria.keys().cloned().collect(),
                criteria
                    .iter()
                    .map(|(k, v)| if v.is_null() { k.clone() } else { describe(v) })
                    .collect(),
            )
        }
    };
    if names.len() > max {
        return Err(invalid(format!(
            "{} options, {max} labels au plus",
            names.len()
        )));
    }
    Ok(Options {
        kind,
        names,
        descriptions,
    })
}

/// Message utilisateur du layout `plain` pour un état déjà rendu.
fn user_turn(
    state: &str,
    question: &Value,
    labels: &[String],
    default_instructions: &str,
) -> Result<(String, Options)> {
    let options = options(question, labels.len())?;
    let instructions = match question.get("instructions") {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => python_indent1(v)?,
    };
    let instructions = match instructions.trim_start() {
        "" => default_instructions,
        text => text,
    };
    let lines: Vec<String> = labels
        .iter()
        .zip(options.names.iter().zip(&options.descriptions))
        .map(|(l, (n, d))| format!("{l}) {n}: {d}"))
        .collect();
    Ok((
        format!(
            "record: {state}\nquestion: {instructions}\noptions:\n{}",
            lines.join("\n")
        ),
        options,
    ))
}

/// Seuil oui/non optionnel (`threshold`, sinon `noul_threshold`) dans [0, 1].
fn threshold(question: &Value) -> Option<f64> {
    if question.get("type").and_then(Value::as_str) != Some("noul") {
        return None;
    }
    question
        .get("threshold")
        .or_else(|| question.get("noul_threshold"))
        .and_then(Value::as_f64)
        .filter(|t| (0.0..=1.0).contains(t))
}

/// `format_answer` du shim sur une distribution déjà étalonnée.
fn answer(options: &Options, p: &[f32], floor: f64, threshold: Option<f64>) -> Value {
    let p: Vec<f64> = p.iter().map(|&x| f64::from(x)).collect();
    let probabilities = |names: &[String]| -> Map<String, Value> {
        names
            .iter()
            .cloned()
            .zip(p.iter().map(|&x| json!(x)))
            .collect()
    };
    let top = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    match options.kind {
        Kind::Noul => {
            let yes = p.first().copied().unwrap_or(0.0);
            // `commit_noul` : sous le plancher, la probabilité y est ramenée.
            let yes = if floor == 0.0 || yes.max(1.0 - yes) >= floor {
                yes
            } else if yes >= 0.5 {
                floor
            } else {
                1.0 - floor
            };
            let mut out = json!({"type":"noul","noul":yes});
            if let Some(t) = threshold {
                out["threshold"] = json!(t);
                out["decision"] = json!(yes >= t);
            }
            out
        }
        Kind::Score => json!({
            "type":"score",
            "score":p.iter().enumerate().map(|(i, x)| i as f64 * x).sum::<f64>(),
            "legend":options.names.iter().cloned().zip(options.descriptions.iter().map(|d| json!(d))).collect::<Map<_, _>>(),
            "probabilities":probabilities(&options.names),
            "confidence":top,
        }),
        Kind::Choice => {
            let best = (1..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
            let n = p.len() as f64;
            let confidence = if p.len() < 2 {
                1.0
            } else {
                (top - 1.0 / n) / (1.0 - 1.0 / n)
            };
            json!({"type":"choice","choice":options.names.get(best),
                "probabilities":probabilities(&options.names),"confidence":confidence})
        }
    }
}

#[cfg(test)]
mod tests;
