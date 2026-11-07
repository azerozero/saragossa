//! Session de decode résident Flash-Next : un token = un command buffer.
//!
//! Par token : embedding (CPU, une ligne u8 déquantifiée, recopiée sur les
//! `hc` flux), hachage n-gram et lecture parallèle des lignes PLE avant le
//! commit, puis sur GPU les 48 couches (hc d'attention, DeltaNet ou QSA dense,
//! hc du MLP, MoE), le mixer final et le lm_head. Le decode attend chaque
//! token : les buffers partagés (embedding, octets PLE) ne sont réécrits
//! qu'après la fin du command buffer précédent.

use metal::{Buffer, CommandQueue, MTLResourceOptions};

use super::attention::{
    dense_max_visible, encode_qsa, IndexerDims, IndexerScratch, IndexerState, QsaDecodeDims,
    QsaScratch, QsaSparse,
};
use super::encoders::{Brick, Encoders};
use super::hc::{encode_hc_combine, encode_hc_mix, HcDims, HcScratch};
use super::kernels::FlashNextKernels;
use super::ple::{encode_ple, encode_rms_norm_1p, PleScratch, PleState};
use super::weights::{FlashNextWeights, LayerWeights, LoadStage, MixerWeights};
use crate::decode_resident::{DecodeResidentState, FullAttentionMetalState};
use crate::metal_backend::{
    commit_and_wait, read_f32_buffer, read_u32_buffer, write_f32_buffer, LinearAttentionMetalState,
    LinearAttentionStepSpec, LinearAttnResidentDims, MetalExecutor,
};
use crate::qwen4_exp::ngram::NgramContext;
use crate::qwen_loader::qwen4_exp::{Qwen4ExpCheckpoint, Qwen4ExpConfig};
use crate::{InferError, MemoryGuard, Result};

/// État récurrent ou cache KV d'une couche.
enum LayerState {
    Linear(LinearAttentionMetalState),
    /// KV de la couche QSA et état de son indexeur.
    Full(Box<(FullAttentionMetalState, IndexerState)>),
}

/// Tampons propres à la tête MTP : résiduels ping-pong (jamais ceux du tronc,
/// dont le résiduel avant mixer reste intact), entrée fc et sortie hc.
struct MtpBuffers {
    h: [Buffer; 2],
    embed: Buffer,
    e_normed: Buffer,
    fe: Buffer,
    h_normed: Buffer,
    fh: Buffer,
    /// Résiduel hc de sortie : `h` du draft suivant.
    out: Buffer,
    /// Instantané de l'anneau des clés brutes de l'indexeur MTP
    /// (`[compress, head_dim]`), pris par [`FlashNextSession::mtp_snapshot_raw`].
    raw_snapshot: Buffer,
}

/// Résiduel d'entrée d'un pas MTP : copié depuis l'hôte, ou depuis un buffer
/// résident (sortie d'un pas MTP précédent, sans aller-retour hôte).
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
)]
pub(crate) enum MtpInput<'a> {
    Host(&'a [f32]),
    Resident(&'a metal::BufferRef),
}

/// Sortie d'un pas MTP.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
)]
pub(crate) struct MtpOutput {
    /// Draft : argmax du lm_head partagé sur le mixer de tête.
    pub(crate) argmax: u32,
    pub(crate) logits: Option<Vec<f32>>,
    /// Résiduel hc de sortie (`[hc·hidden]`), relu sur demande.
    pub(crate) hidden: Option<Vec<f32>>,
    /// Oracles : entrée fc de la couche (`[hc·hidden]`), sélection QSA (régime
    /// creux) et experts routés.
    #[cfg(test)]
    pub(crate) input: Vec<f32>,
    #[cfg(test)]
    pub(crate) qsa: Option<QsaCapture>,
    #[cfg(test)]
    pub(crate) routing: Option<Vec<u32>>,
}

/// Lignes maximales d'une vérification : jeton en attente et 3 drafts.
pub(crate) const VERIFY_ROWS: usize = 4;

/// Tampons préalloués d'une vérification : entrées et sorties propres à chaque
/// ligne (aucune réécriture hôte d'un tampon encodé), et captures de l'état du
/// tronc après chaque ligne `r < VERIFY_ROWS − 1`, pour le retour arrière.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
)]
struct VerifyBuffers {
    /// `[ligne]` : embedding répété sur les flux (`[hc·hidden]`).
    embed: Vec<Buffer>,
    /// `[ligne]` : octets E4M3 PLE (`[ple_dim]`).
    ple_bytes: Vec<Buffer>,
    /// `[ligne]` : argmax du lm_head.
    index: Vec<Buffer>,
    /// `[ligne]` : résiduel hc avant le mixer final.
    residual: Vec<Buffer>,
    /// `[ligne][couche DeltaNet]` : états SSM et fenêtres conv.
    linear: Vec<Vec<LinearAttentionMetalState>>,
    /// `[ligne][couche QSA du tronc]` : clés brutes de l'indexeur.
    raw: Vec<Vec<Buffer>>,
    /// `[ligne]` : anneau PLE.
    ring: Vec<Buffer>,
}

/// Oracles : mutation de la vérification ou de l'acceptation (puissance des
/// tests de retour arrière).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum VerifyMutation {
    /// Restaure la capture de la ligne suivante.
    CaptureNext,
    /// Oublie la troncature des KV.
    NoTruncate,
    /// Garde le contexte n-gram de la ligne précédente.
    StaleContext,
    /// Pas PLE engagé décalé d'un cran.
    PleStepOff,
    /// Toutes les lignes lisent l'embedding et les octets PLE de la ligne 0.
    AliasFirstRow,
    /// Résiduel relu dans le mauvais tampon ping-pong.
    WrongResidual,
}

/// Oracles : état observable du tronc (préfixes valides seulement).
#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(crate) struct StateDigest {
    pub(crate) bits: Vec<u32>,
    pub(crate) context: NgramContext,
}

/// Profil en lignes : couches DeltaNet (poids, état) et MoE du tronc.
#[cfg(all(test, feature = "devtools"))]
pub(crate) struct DeltaNetAndMoe<'a> {
    pub(crate) linear: Vec<(
        &'a crate::metal_backend::MetalLinearAttnResidentDenseWeights,
        &'a LinearAttentionMetalState,
    )>,
    pub(crate) moe: Vec<&'a crate::metal_backend::MetalMoeSharedWeights>,
    pub(crate) spec: LinearAttentionStepSpec,
    pub(crate) dims: LinearAttnResidentDims,
    pub(crate) top_k: usize,
}

/// Vérification soumise, en attente de [`FlashNextSession::accept`].
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
)]
struct PendingVerify {
    position: usize,
    ple_step: usize,
    /// Contexte n-gram après chaque ligne.
    contexts: Vec<NgramContext>,
    rows: usize,
}

/// Sortie d'une ligne de vérification.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
)]
pub(crate) struct VerifyRow {
    pub(crate) argmax: u32,
    /// Résiduel hc avant le mixer final (`h` de la MTP pour le cycle suivant).
    pub(crate) residual: Vec<f32>,
}

/// Capacité maximale d'une session : 256k tokens (contexte cible, plan §8.1).
pub(crate) const MAX_CAPACITY: usize = 262_144;

/// Blocs QSA retenus et scores de l'indexeur d'une couche (oracles).
#[cfg(test)]
pub(crate) type QsaCapture = (Vec<u32>, Vec<f32>);

/// Point de reprise d'une session ([`FlashNextSession::checkpoint`]).
pub(crate) struct SessionCheckpoint {
    id: u64,
    position: usize,
    epoch: u64,
    linear: Vec<Option<LinearAttentionMetalState>>,
    kv_lens: Vec<Option<usize>>,
    /// Clés brutes du bloc en cours de chaque couche QSA (position non alignée).
    indexer_raw: Vec<Option<Buffer>>,
    ple_ring: Buffer,
    ple_context: NgramContext,
    ple_step: usize,
}

/// Tenseurs de l'indexeur relus après un pas (oracles).
#[cfg(test)]
pub(crate) struct IndexerProbe {
    /// Projection brute `[(têtes + 1)·d]` (requêtes puis clé).
    pub(crate) qk: Vec<f32>,
    /// Requêtes normées et tournées `[têtes·d]`.
    pub(crate) q: Vec<f32>,
    pub(crate) pooled: Vec<f32>,
    pub(crate) raw: Vec<f32>,
}

/// Sorties d'un pas.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "pas encore d'entrée de production (serve, étape 9) : exercé par les oracles"
    )
)]
pub(crate) struct StepOutput {
    pub(crate) argmax: u32,
    /// Logits `[vocab]` si demandés.
    pub(crate) logits: Option<Vec<f32>>,
    /// Résiduel élargi `[hc·hidden]` après chaque couche, si demandé.
    pub(crate) layers: Option<Vec<Vec<f32>>>,
    /// Experts routés par le GPU `[couche][slot]`, si les résiduels sont demandés
    /// (oracles seulement).
    #[cfg(test)]
    pub(crate) routing: Option<Vec<Vec<u32>>>,
    /// Blocs QSA retenus et scores de l'indexeur par couche (QSA en régime
    /// creux seulement), si les résiduels sont demandés (oracles).
    #[cfg(test)]
    pub(crate) qsa: Option<Vec<Option<QsaCapture>>>,
    /// Préparation CPU avant le commit (embedding, hachage, lectures PLE).
    pub(crate) host: std::time::Duration,
    /// Encodage CPU du command buffer (GPU inactif pendant ce temps).
    pub(crate) encode: std::time::Duration,
}

/// Decode résident Flash-Next d'une séquence (≤ 2051 tokens).
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "pas encore d'entrée de production (serve, étape 9) : exercé par les oracles"
    )
)]
pub(crate) struct FlashNextSession {
    ckpt: Qwen4ExpCheckpoint,
    executor: MetalExecutor,
    arena: DecodeResidentState,
    kernels: FlashNextKernels,
    queue: CommandQueue,
    weights: FlashNextWeights,
    /// États des couches du tronc, puis, en dernier, celui de la couche MTP si
    /// la tête est chargée : `reset`, points de reprise et budgets le couvrent
    /// ainsi d'office, l'encodage du tronc s'arrêtant aux couches.
    states: Vec<LayerState>,
    /// Tampons de la tête MTP (absents sans tête).
    mtp: Option<MtpBuffers>,
    /// Tampons et captures des vérifications (sortis pendant une vérification).
    verify: Option<VerifyBuffers>,
    /// Vérification soumise non encore acceptée : tout autre pas est refusé.
    pending: Option<PendingVerify>,
    /// Lignes de l'historique MTP au moment de l'instantané des clés brutes
    /// (`None` : pas d'instantané valide).
    mtp_raw_rows: Option<usize>,
    ple_state: PleState,
    h: [Buffer; 2],
    x: Buffer,
    branch: Buffer,
    logits: Buffer,
    index: Buffer,
    capture: Buffer,
    hc_scratch: HcScratch,
    ple_scratch: PleScratch,
    qsa_scratch: QsaScratch,
    /// KV compact du gather (régime creux), partagé entre couches QSA.
    gathered: FullAttentionMetalState,
    indexer_scratch: IndexerScratch,
    indexer_dims: IndexerDims,
    hc_dims: HcDims,
    qsa_dims: QsaDecodeDims,
    linear_spec: LinearAttentionStepSpec,
    linear_dims: LinearAttnResidentDims,
    position: usize,
    capacity: usize,
    /// Branche courante de l'historique : incrémentée à chaque `reset` et à
    /// chaque reprise, elle invalide les points pris sur une autre branche.
    epoch: u64,
    /// Point de reprise d'où part la branche courante (réutilisable).
    base_checkpoint: Option<u64>,
    /// Compteur d'identifiants des points de reprise.
    next_checkpoint: std::cell::Cell<u64>,
    /// États DeltaNet initiaux (nuls), pour [`Self::reset`].
    initial_linear: Vec<Option<LinearAttentionMetalState>>,
    /// Profil GPU par brique (`SARAGOSSA_FLASH_NEXT_PROFILE=1`).
    #[cfg(feature = "devtools")]
    profiler: Option<super::profile::StepProfiler>,
    /// Vrai après une erreur d'exécution GPU : états incertains, pas refusés.
    poisoned: bool,
    /// Oracles : experts routés par couche du dernier pas `[couches × top_k]`.
    #[cfg(test)]
    routing_capture: Buffer,
    /// Oracles : prochaine lecture n-gram en échec.
    #[cfg(test)]
    pub(crate) fail_next_ngram_read: bool,
    /// Oracles : échec de `reset` injecté après la restauration de la couche donnée.
    #[cfg(test)]
    pub(crate) fail_reset_after_layer: Option<usize>,
    /// Oracles : échec d'encodage injecté après la couche donnée.
    #[cfg(test)]
    pub(crate) fail_encode_after_layer: Option<usize>,
    /// Oracles : experts imposés au prochain pas, par couche (consommés).
    #[cfg(test)]
    pub(crate) forced_routing: Vec<Option<Vec<u32>>>,
    /// Oracles : blocs QSA imposés au prochain pas, par couche (consommés).
    #[cfg(test)]
    pub(crate) forced_selection: Vec<Option<Vec<u32>>>,
    /// Oracles : blocs et experts imposés au prochain pas MTP (consommés).
    #[cfg(test)]
    pub(crate) mtp_forced_selection: Option<Vec<u32>>,
    #[cfg(test)]
    pub(crate) mtp_forced_routing: Option<Vec<u32>>,
    /// Oracles : entrée fc du dernier pas MTP.
    #[cfg(test)]
    mtp_input_capture: Buffer,
    /// Oracles : mutation de vérification active.
    #[cfg(test)]
    pub(crate) verify_mutation: Option<VerifyMutation>,
    /// Oracles : échec d'encodage injecté à la ligne donnée d'une vérification.
    #[cfg(test)]
    pub(crate) fail_verify_row: Option<usize>,
    /// Oracles : troncature MTP sans restauration des clés brutes.
    #[cfg(test)]
    pub(crate) skip_mtp_raw_restore: bool,
    /// Oracles : sélection et scores de l'indexeur par couche du dernier pas
    /// (`[couches, keep]` u32, `[couches, blocs max]` f32).
    #[cfg(test)]
    selection_capture: Buffer,
    #[cfg(test)]
    score_capture: Buffer,
}

fn zeros(executor: &MetalExecutor, len: usize) -> Buffer {
    executor.device().new_buffer(
        (len.max(1) * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Octets des états et intermédiaires d'une session de `capacity` tokens :
/// - KV f32 des couches QSA ;
/// - états conv/SSM du DeltaNet, **deux fois** (états vivants et instantanés
///   initiaux du `reset`) ;
/// - anneau et octets PLE, buffers de la session (résiduels, logits, capture,
///   scratch hc/PLE/QSA) ;
/// - scratch interne de l'exécuteur, alloué au premier pas (DeltaNet, MoE,
///   argmax), majoré.
pub(crate) fn session_state_bytes(c: &Qwen4ExpConfig, capacity: usize) -> u64 {
    let f32_bytes = std::mem::size_of::<f32>();
    let width = c.hc_width();
    let full = c
        .layer_kinds
        .iter()
        .filter(|k| **k == crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Full)
        .count();
    let linear = c.layer_kinds.len() - full;
    let kv = full * capacity * 2 * c.kv_heads * c.head_dim;
    let dense = dense_max_visible(c.indexer_budget, c.indexer_compress);
    let conv_dim =
        2 * c.linear_key_heads * c.linear_key_dim + c.linear_value_heads * c.linear_value_dim;
    let value_dim = c.linear_value_heads * c.linear_value_dim;
    let deltanet = 2
        * linear
        * ((c.conv_kernel - 1) * conv_dim
            + c.linear_value_heads * c.linear_value_dim * c.linear_key_dim);
    let ple = (c.ple_conv_kernel - 1) * c.ngram_size * width + c.ple_dim;
    let q_dim = c.heads * c.head_dim;
    let kv_dim = c.kv_heads * c.head_dim;
    let session = 2 * width // résiduels ping-pong
        + 2 * c.hidden // entrée de branche et sortie de mélangeur
        + c.vocab + 1 // logits et argmax
        + c.layer_kinds.len() * width // capture des résiduels
        + 2 * width + c.hc_lowrank * (1 + super::hc::DOWN_CHUNKS) + c.hc_count * super::hc::INJECT_CHUNKS // scratch hc
        + c.ple_dim + 5 * width + c.hidden // scratch PLE
        + 6 * q_dim + 3 * kv_dim + c.heads * dense; // scratch QSA
                                                    // Indexeur : clés poolées et bloc en cours par couche QSA, KV compact du
                                                    // gather (`dense` lignes), scratch (qk, q, scores, sélection).
    let blocks = capacity / c.indexer_compress + 1;
    let indexer = full * (blocks + c.indexer_compress) * c.indexer_head_dim
        + 2 * dense * kv_dim
        + (2 * c.indexer_heads + 1) * c.indexer_head_dim
        + blocks
        + c.indexer_budget / c.indexer_compress;
    let executor = 4 * conv_dim + 6 * value_dim + 4 * c.linear_value_heads // DeltaNet
        + 2 * c.experts + 2 * c.top_k * (3 * c.moe_inter + c.hidden) // MoE routé
        + 3 * c.shared_inter + 2 * c.hidden // expert partagé
        + c.vocab; // argmax du lm_head
                   // Tête MTP : KV et indexeur d'une couche QSA, résiduels et entrée fc.
    let mtp = if c.mtp_layers == 1 {
        capacity * 2 * kv_dim
            + (blocks + c.indexer_compress) * c.indexer_head_dim
            + 5 * width // résiduels ping-pong, h normé, fc_hidden, sortie
            + 3 * c.hidden // embedding, e normé, fc_embedding
            + c.indexer_compress * c.indexer_head_dim // instantané des clés brutes
    } else {
        0
    };
    // Vérification : entrées et sorties par ligne, captures DeltaNet, clés
    // brutes et anneau PLE après chaque ligne sauf la dernière.
    let verify = if c.mtp_layers == 1 {
        VERIFY_ROWS * (2 * width + c.ple_dim.div_ceil(f32_bytes) + 1)
            + (VERIFY_ROWS - 1)
                * (deltanet / 2 + full * c.indexer_compress * c.indexer_head_dim + ple - c.ple_dim)
    } else {
        0
    };
    ((kv + deltanet + ple + session + executor + indexer + mtp + verify) * f32_bytes) as u64
}

/// Refuse, avant toute allocation, les variantes que le chemin résident ne
/// porte pas, et renvoie la visibilité maximale du régime dense.
///
/// # Errors
///
/// Renvoie [`InferError::Config`] pour une variante non prise en charge.
pub(crate) fn validate_resident_config(c: &Qwen4ExpConfig) -> Result<usize> {
    let refuse = |what: String| -> Result<usize> {
        Err(InferError::Config(format!(
            "Flash-Next résident : {what} non pris en charge"
        )))
    };
    if !c.norm_topk_prob {
        return refuse("norm_topk_prob = false".to_string());
    }
    if !c.sigmoid_output_gate {
        return refuse("gate DeltaNet SiLU".to_string());
    }
    if c.top_k == 0 || c.top_k > 16 || c.top_k > c.experts {
        return refuse(format!("top-k {} sur {} experts", c.top_k, c.experts));
    }
    if c.linear_key_dim != 128 || c.linear_value_dim != 128 || c.conv_kernel != 4 {
        return refuse(format!(
            "DeltaNet d_k {} / d_v {} / noyau {}",
            c.linear_key_dim, c.linear_value_dim, c.conv_kernel
        ));
    }
    if c.linear_key_heads == 0 || c.linear_value_heads % c.linear_key_heads != 0 {
        return refuse(format!(
            "DeltaNet {} têtes V pour {} têtes K",
            c.linear_value_heads, c.linear_key_heads
        ));
    }
    // Norme + RoPE par tête : un threadgroup de d threads (≤ 256, multiple de 32).
    if c.kv_heads == 0
        || c.heads % c.kv_heads != 0
        || c.rotary_dim > c.head_dim
        || c.head_dim > 256
        || c.head_dim % 32 != 0
    {
        return refuse(format!(
            "attention {} têtes / {} KV, RoPE {} sur {}",
            c.heads, c.kv_heads, c.rotary_dim, c.head_dim
        ));
    }
    // Pooling (un threadgroup de d threads, `values[256]`, `partial[8]`) et
    // RoPE de l'indexeur (dimensions tournées ≤ d).
    if c.indexer_head_dim == 0
        || c.indexer_head_dim > 256
        || c.indexer_head_dim % 32 != 0
        || c.rotary_dim > c.indexer_head_dim
        || c.indexer_heads == 0
    {
        return refuse(format!(
            "indexeur {} têtes de {} (RoPE {})",
            c.indexer_heads, c.indexer_head_dim, c.rotary_dim
        ));
    }
    if c.indexer_compress == 0 || c.indexer_budget % c.indexer_compress != 0 {
        return refuse(format!(
            "budget QSA {} par blocs de {}",
            c.indexer_budget, c.indexer_compress
        ));
    }
    if c.ple_conv_kernel == 0 || c.ple_layer >= c.layer_kinds.len() {
        return refuse(format!(
            "PLE couche {} noyau {}",
            c.ple_layer, c.ple_conv_kernel
        ));
    }
    Ok(dense_max_visible(c.indexer_budget, c.indexer_compress))
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "pas encore d'entrée de production (serve, étape 9) : exercé par les oracles"
    )
)]
impl FlashNextSession {
    /// Ouvre le checkpoint, charge le tronc sur GPU et alloue les états pour
    /// `capacity` tokens (au plus 2051, régime QSA dense).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de chargement, de garde mémoire, ou si
    /// `capacity` dépasse le régime dense.
    pub(crate) fn open(
        ckpt: Qwen4ExpCheckpoint,
        capacity: usize,
        guard: Option<&MemoryGuard>,
    ) -> Result<Self> {
        let dense_max = validate_resident_config(&ckpt.config)?;
        if capacity == 0 || capacity > MAX_CAPACITY {
            return Err(InferError::Config(format!(
                "Flash-Next résident : capacité {capacity} hors ]0, {MAX_CAPACITY}]"
            )));
        }
        let executor = MetalExecutor::new()?;
        let arena = DecodeResidentState::new(executor.device().clone())?;
        let kernels = FlashNextKernels::new(executor.device())?;
        let weights = FlashNextWeights::load(&executor, &ckpt, guard)?;
        let c = &ckpt.config;
        LoadStage {
            resident: session_state_bytes(c, capacity),
            transient: 0,
        }
        .check(guard)?;
        let hc_dims = HcDims {
            hidden: c.hidden,
            hc: c.hc_count,
            lowrank: c.hc_lowrank,
            eps: c.rms_eps,
        };
        let qsa_dims = QsaDecodeDims {
            hidden: c.hidden,
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            rotary_dim: c.rotary_dim,
            eps: c.rms_eps,
            dense_max,
            #[cfg(test)]
            query_offset: 0,
            #[cfg(test)]
            index_query_offset: 0,
        };
        let linear_spec = LinearAttentionStepSpec {
            num_key_heads: c.linear_key_heads,
            num_value_heads: c.linear_value_heads,
            key_head_dim: c.linear_key_dim,
            value_head_dim: c.linear_value_dim,
            conv_kernel_dim: c.conv_kernel,
            rms_eps: c.rms_eps,
            l2_qk_sigmoid_gate: true,
        };
        let key_dim = c.linear_key_heads * c.linear_key_dim;
        let value_dim = c.linear_value_heads * c.linear_value_dim;
        let linear_dims = LinearAttnResidentDims {
            in_dim: c.hidden,
            conv_dim: 2 * key_dim + value_dim,
            value_dim,
            key_dim,
        };
        let width = hc_dims.width();
        let (q_dim, kv_dim) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        // KV compact du régime creux : `keep·compress` lignes retenues + la queue.
        let gathered = arena.full_attention_with_format(
            dense_max,
            c.heads,
            c.kv_heads,
            c.head_dim,
            crate::decode_resident::KvFormat::F32,
            None,
        )?;
        let mut session = Self {
            arena,
            kernels,
            queue: executor.device().new_command_queue(),
            mtp: weights.mtp.as_ref().map(|_| MtpBuffers {
                h: [zeros(&executor, width), zeros(&executor, width)],
                embed: zeros(&executor, c.hidden),
                e_normed: zeros(&executor, c.hidden),
                fe: zeros(&executor, c.hidden),
                h_normed: zeros(&executor, width),
                fh: zeros(&executor, width),
                out: zeros(&executor, width),
                raw_snapshot: zeros(&executor, c.indexer_compress * c.indexer_head_dim),
            }),
            weights,
            verify: None,
            pending: None,
            mtp_raw_rows: None,
            states: Vec::with_capacity(c.layer_kinds.len() + 1),
            ple_state: PleState {
                bytes: executor
                    .device()
                    .new_buffer(c.ple_dim as u64, MTLResourceOptions::StorageModeShared),
                ring: zeros(&executor, (c.ple_conv_kernel - 1) * c.ngram_size * width),
                context: NgramContext::new(c.ngram_size, c.eos),
                step: 0,
            },
            h: [zeros(&executor, width), zeros(&executor, width)],
            x: zeros(&executor, c.hidden),
            branch: zeros(&executor, c.hidden),
            logits: zeros(&executor, c.vocab),
            index: executor
                .device()
                .new_buffer(4, MTLResourceOptions::StorageModeShared),
            capture: zeros(&executor, c.layer_kinds.len() * width),
            hc_scratch: HcScratch {
                normed: zeros(&executor, width),
                low: zeros(&executor, c.hc_lowrank),
                up: zeros(&executor, width),
                inject: zeros(&executor, c.hc_count * super::hc::INJECT_CHUNKS),
                down_partial: zeros(&executor, c.hc_lowrank * super::hc::DOWN_CHUNKS),
            },
            ple_scratch: PleScratch {
                emb: zeros(&executor, c.ple_dim),
                key: zeros(&executor, width),
                key_normed: zeros(&executor, width),
                value: zeros(&executor, c.hidden),
                query: zeros(&executor, width),
                gated: zeros(&executor, width),
                normed: zeros(&executor, width),
            },
            qsa_scratch: QsaScratch {
                q_gate: zeros(&executor, 2 * q_dim),
                q_raw: zeros(&executor, q_dim),
                gate: zeros(&executor, q_dim),
                q_roped: zeros(&executor, q_dim),
                k_raw: zeros(&executor, kv_dim),
                k_roped: zeros(&executor, kv_dim),
                v: zeros(&executor, kv_dim),
                scores: zeros(&executor, c.heads * dense_max),
                ctx: zeros(&executor, q_dim),
                gated: zeros(&executor, q_dim),
            },
            gathered,
            indexer_scratch: IndexerScratch {
                qk: zeros(&executor, (c.indexer_heads + 1) * c.indexer_head_dim),
                q: zeros(&executor, c.indexer_heads * c.indexer_head_dim),
                scores: zeros(&executor, capacity / c.indexer_compress + 1),
                selected: executor.device().new_buffer(
                    ((c.indexer_budget / c.indexer_compress) * std::mem::size_of::<u32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            },
            indexer_dims: IndexerDims {
                heads: c.indexer_heads,
                head_dim: c.indexer_head_dim,
                compress: c.indexer_compress,
                keep: c.indexer_budget / c.indexer_compress,
            },
            hc_dims,
            qsa_dims,
            linear_spec,
            linear_dims,
            position: 0,
            capacity,
            epoch: 0,
            base_checkpoint: None,
            next_checkpoint: std::cell::Cell::new(0),
            initial_linear: Vec::new(),
            #[cfg(feature = "devtools")]
            profiler: if super::profile::profile_requested() {
                Some(super::profile::StepProfiler::new(executor.device())?)
            } else {
                None
            },
            poisoned: false,
            #[cfg(test)]
            routing_capture: executor.device().new_buffer(
                ((c.layer_kinds.len() + 1) * c.top_k * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            #[cfg(test)]
            fail_next_ngram_read: false,
            #[cfg(test)]
            fail_reset_after_layer: None,
            #[cfg(test)]
            fail_encode_after_layer: None,
            #[cfg(test)]
            forced_routing: Vec::new(),
            #[cfg(test)]
            forced_selection: Vec::new(),
            #[cfg(test)]
            mtp_forced_selection: None,
            #[cfg(test)]
            mtp_forced_routing: None,
            #[cfg(test)]
            mtp_input_capture: zeros(&executor, width),
            #[cfg(test)]
            verify_mutation: None,
            #[cfg(test)]
            fail_verify_row: None,
            #[cfg(test)]
            skip_mtp_raw_restore: false,
            #[cfg(test)]
            selection_capture: executor.device().new_buffer(
                ((c.layer_kinds.len() + 1) * (c.indexer_budget / c.indexer_compress) * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            #[cfg(test)]
            score_capture: zeros(
                &executor,
                (c.layer_kinds.len() + 1) * (capacity / c.indexer_compress + 1),
            ),
            executor,
            ckpt,
        };
        session.states = session.fresh_states(capacity)?;
        session.initial_linear = session
            .states
            .iter()
            .map(|state| match state {
                LayerState::Linear(state) => {
                    session.executor.snapshot_linear_attn_state(state).map(Some)
                }
                LayerState::Full(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
        // Les vérifications ne servent qu'à la spéculation MTP.
        if session.weights.mtp.is_some() {
            session.verify = Some(session.verify_buffers()?);
        }
        Ok(session)
    }

    /// Alloue les tampons et captures des vérifications (`VERIFY_ROWS` lignes).
    fn verify_buffers(&self) -> Result<VerifyBuffers> {
        let c = &self.ckpt.config;
        let width = self.hc_dims.width();
        let device = self.executor.device();
        let trunk = &self.states[..c.layer_kinds.len()];
        let captures = VERIFY_ROWS - 1;
        let mut linear = Vec::with_capacity(captures);
        let mut raw = Vec::with_capacity(captures);
        let mut ring = Vec::with_capacity(captures);
        for _ in 0..captures {
            // Une soumission par ligne de capture (copie groupée des couches).
            let live: Vec<Option<&LinearAttentionMetalState>> = trunk
                .iter()
                .filter_map(|state| match state {
                    LayerState::Linear(state) => Some(Some(state)),
                    LayerState::Full(_) => None,
                })
                .collect();
            linear.push(
                self.executor
                    .snapshot_linear_attn_states(&live)?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            );
            raw.push(
                trunk
                    .iter()
                    .filter_map(|state| match state {
                        LayerState::Full(full) => {
                            Some(zeros(&self.executor, (full.1.raw.length() / 4) as usize))
                        }
                        LayerState::Linear(_) => None,
                    })
                    .collect(),
            );
            ring.push(zeros(
                &self.executor,
                (self.ple_state.ring.length() / 4) as usize,
            ));
        }
        let rows = |len: usize| {
            (0..VERIFY_ROWS)
                .map(|_| zeros(&self.executor, len))
                .collect()
        };
        Ok(VerifyBuffers {
            embed: rows(width),
            ple_bytes: (0..VERIFY_ROWS)
                .map(|_| device.new_buffer(c.ple_dim as u64, MTLResourceOptions::StorageModeShared))
                .collect(),
            index: (0..VERIFY_ROWS)
                .map(|_| device.new_buffer(4, MTLResourceOptions::StorageModeShared))
                .collect(),
            residual: rows(width),
            linear,
            raw,
            ring,
        })
    }

    /// Erreur sur une session invalidée (erreur GPU, reset incomplet ou
    /// restauration MTP manquée) : seul un `reset` réussi la réhabilite.
    fn refuse_poisoned(&self) -> Result<()> {
        if self.poisoned {
            return Err(InferError::Metal(
                "Flash-Next résident : session invalidée (erreur GPU ou reset incomplet)"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Erreur si une vérification attend son [`Self::accept`].
    fn refuse_pending(&self) -> Result<()> {
        if self.pending.is_some() {
            return Err(InferError::Config(
                "Flash-Next : vérification en attente d'acceptation".to_string(),
            ));
        }
        Ok(())
    }

    /// Vérifie `tokens` (jeton en attente puis drafts) aux positions suivantes :
    /// une suite de pas mono du tronc dans **un** command buffer, mêmes kernels
    /// et mêmes entrées que [`Self::step`] (mêmes bits que l'AR). Chaque ligne a
    /// ses entrées et sorties ; l'état après chaque ligne `r < R − 1` est capturé.
    /// La session attend ensuite [`Self::accept`].
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une vérification est déjà en attente, sur nombre de
    /// lignes, token ou capacité invalides, lecture n-gram, ou encodage et
    /// exécution Metal (rien n'est engagé si l'encodage échoue ; une erreur GPU
    /// invalide la session).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    pub(crate) fn verify(&mut self, tokens: &[usize]) -> Result<Vec<VerifyRow>> {
        if self.poisoned {
            return Err(InferError::Metal(
                "Flash-Next résident : session invalidée (erreur GPU ou reset incomplet)"
                    .to_string(),
            ));
        }
        self.refuse_pending()?;
        let c = &self.ckpt.config;
        let rows = tokens.len();
        if rows == 0 || rows > VERIFY_ROWS || tokens.iter().any(|&t| t >= c.vocab) {
            return Err(InferError::Shape(format!(
                "vérification : {rows} lignes (1 à {VERIFY_ROWS}), tokens {tokens:?}"
            )));
        }
        if self.position + rows > self.capacity {
            return Err(InferError::Config(format!(
                "Flash-Next : vérification de {rows} lignes au-delà de la capacité {}",
                self.capacity
            )));
        }
        let width = self.hc_dims.width();
        let buffers = self
            .verify
            .take()
            .ok_or_else(|| InferError::Config("tampons de vérification absents".to_string()))?;
        let result = self.verify_with(&buffers, tokens);
        self.verify = Some(buffers);
        let pending = result?;
        // Les lignes sont engagées sur GPU : une relecture en échec invalide la
        // session plutôt que de laisser une transaction sans sorties.
        let out = self
            .verify
            .as_ref()
            .ok_or_else(|| InferError::Config("tampons de vérification absents".to_string()))
            .and_then(|buffers| {
                (0..rows)
                    .map(|r| {
                        Ok(VerifyRow {
                            argmax: read_u32_buffer(&buffers.index[r], 1)?
                                .first()
                                .copied()
                                .ok_or_else(|| {
                                    InferError::Metal("argmax de vérification absent".to_string())
                                })?,
                            residual: read_f32_buffer(&buffers.residual[r], width)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            });
        match out {
            Ok(rows) => {
                self.pending = Some(pending);
                Ok(rows)
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    fn verify_with(&mut self, buffers: &VerifyBuffers, tokens: &[usize]) -> Result<PendingVerify> {
        let c = &self.ckpt.config;
        let rows = tokens.len();
        let width = self.hc_dims.width();
        // Entrées hôte de toutes les lignes, préparées avant l'encodage.
        let p = &self.ckpt.ngram_params;
        let mut context = self.ple_state.context.clone();
        let mut contexts = Vec::with_capacity(rows);
        for (r, &token) in tokens.iter().enumerate() {
            let row = self.ckpt.affine_rows(
                &Qwen4ExpCheckpoint::trunk("embed_tokens"),
                None,
                token..token + 1,
            )?;
            let mut streams = Vec::with_capacity(width);
            for _ in 0..c.hc_count {
                streams.extend_from_slice(row.data());
            }
            write_f32_buffer(&buffers.embed[r], &streams)?;
            let ids = context.step(
                token as u64,
                c.heads_per_ngram,
                &p.multipliers,
                &p.vocab_sizes,
                &p.offsets,
            );
            let mut bytes = vec![0_u8; c.ple_dim];
            self.ckpt.ngram_rows_raw(&ids, &mut bytes)?;
            crate::metal_backend::write_u8_buffer(&buffers.ple_bytes[r], &bytes)?;
            contexts.push(context.clone());
        }
        let base = (self.position, self.ple_state.step);
        let kv_lens: Vec<Option<usize>> = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Full(full) => Some(full.0.len()),
                LayerState::Linear(_) => None,
            })
            .collect();
        let queue = self.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let mut encoders = Encoders::single(command_buffer);
        let mut encoded = Ok(());
        for r in 0..rows {
            self.position = base.0 + r;
            encoded = self.encode_verify_row(&mut encoders, buffers, r, r + 1 < rows);
            if encoded.is_err() {
                break;
            }
        }
        if let Err(error) = encoded {
            drop(encoders);
            self.position = base.0;
            self.ple_state.step = base.1;
            for (state, len) in self.states.iter_mut().zip(kv_lens) {
                if let (LayerState::Full(kv), Some(len)) = (state, len) {
                    if kv.0.truncate(len).is_err() {
                        self.poisoned = true;
                    }
                }
            }
            return Err(error);
        }
        encoders.end();
        if let Err(error) = commit_and_wait(command_buffer) {
            self.poisoned = true;
            return Err(error);
        }
        self.position = base.0 + rows;
        Ok(PendingVerify {
            position: base.0,
            ple_step: base.1,
            contexts,
            rows,
        })
    }

    /// Ligne `r` d'une vérification : embedding de la ligne vers `h[0]`, pas du
    /// tronc (PLE, couches, mixer, argmax), copie du résiduel, puis captures de
    /// l'état si une ligne suit.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    fn encode_verify_row(
        &mut self,
        encoders: &mut Encoders<'_>,
        buffers: &VerifyBuffers,
        r: usize,
        capture: bool,
    ) -> Result<()> {
        let width = self.hc_dims.width();
        #[cfg(test)]
        if self.fail_verify_row == Some(r) {
            return Err(InferError::Metal(format!(
                "vérification : échec d'encodage injecté à la ligne {r}"
            )));
        }
        #[cfg(test)]
        let input = if self.verify_mutation == Some(VerifyMutation::AliasFirstRow) {
            0
        } else {
            r
        };
        #[cfg(not(test))]
        let input = r;
        let encoder = encoders.at(Brick::Hc)?;
        self.executor.encode_copy_with_offsets(
            encoder,
            &buffers.embed[input],
            0,
            &self.h[0],
            0,
            width,
        )?;
        crate::metal_backend::post_dispatch_barrier(encoder);
        let io = TokenIo {
            ple_bytes: buffers.ple_bytes[input].clone(),
            index: buffers.index[r].clone(),
            residual: Some(buffers.residual[r].clone()),
        };
        self.encode_token(encoders, false, false, true, &io)?;
        if !capture {
            return Ok(());
        }
        let encoder = encoders.at(Brick::Hc)?;
        crate::metal_backend::post_dispatch_barrier(encoder);
        let c = &self.ckpt.config;
        let (mut linear, mut full) = (0, 0);
        for state in &self.states[..c.layer_kinds.len()] {
            match state {
                LayerState::Linear(state) => {
                    self.executor.encode_capture_linear_attn_states(
                        encoder,
                        &[(state, &buffers.linear[r][linear])],
                    )?;
                    linear += 1;
                }
                LayerState::Full(kv) => {
                    let len = (kv.1.raw.length() / 4) as usize;
                    self.executor.encode_copy_with_offsets(
                        encoder,
                        &kv.1.raw,
                        0,
                        &buffers.raw[r][full],
                        0,
                        len,
                    )?;
                    full += 1;
                }
            }
        }
        let ring = (self.ple_state.ring.length() / 4) as usize;
        self.executor.encode_copy_with_offsets(
            encoder,
            &self.ple_state.ring,
            0,
            &buffers.ring[r],
            0,
            ring,
        )?;
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Engage les `k + 1` premières lignes de la vérification en attente
    /// (jeton en attente et `k` drafts acceptés) : restaure l'état capturé après
    /// la ligne `k` (en un command buffer), tronque les KV, et reprend le
    /// contexte et le pas PLE de cette ligne. Les blocs poolés et les clés
    /// d'indexeur des lignes rejetées sont réécrits avant d'être lus.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur sans vérification en attente, si `k` sort des lignes
    /// vérifiées, ou sur exécution Metal (session invalidée).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    pub(crate) fn accept(&mut self, k: usize) -> Result<()> {
        let rows = self
            .pending
            .as_ref()
            .map(|pending| pending.rows)
            .ok_or_else(|| InferError::Config("aucune vérification en attente".to_string()))?;
        if k >= rows {
            return Err(InferError::Config(format!(
                "acceptation jusqu'à la ligne {k} sur {rows} vérifiées"
            )));
        }
        if self.verify.is_none() {
            return Err(InferError::Config(
                "tampons de vérification absents".to_string(),
            ));
        }
        let pending = self
            .pending
            .take()
            .ok_or_else(|| InferError::Config("aucune vérification en attente".to_string()))?;
        let c = &self.ckpt.config;
        #[cfg(test)]
        let mutation = self.verify_mutation;
        #[cfg(test)]
        let capture = if mutation == Some(VerifyMutation::CaptureNext) && k + 2 < rows {
            k + 1
        } else {
            k
        };
        #[cfg(not(test))]
        let capture = k;
        if k + 1 < pending.rows {
            let buffers = self
                .verify
                .as_ref()
                .ok_or_else(|| InferError::Config("tampons de vérification absents".to_string()))?;
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let (mut linear, mut full) = (0, 0);
            let mut restored = Ok(());
            for state in &self.states[..c.layer_kinds.len()] {
                restored = match state {
                    LayerState::Linear(state) => {
                        linear += 1;
                        self.executor.encode_capture_linear_attn_states(
                            encoder,
                            &[(&buffers.linear[capture][linear - 1], state)],
                        )
                    }
                    LayerState::Full(kv) => {
                        full += 1;
                        let len = (kv.1.raw.length() / 4) as usize;
                        self.executor.encode_copy_with_offsets(
                            encoder,
                            &buffers.raw[capture][full - 1],
                            0,
                            &kv.1.raw,
                            0,
                            len,
                        )
                    }
                };
                if restored.is_err() {
                    break;
                }
            }
            if restored.is_ok() {
                let ring = (self.ple_state.ring.length() / 4) as usize;
                restored = self.executor.encode_copy_with_offsets(
                    encoder,
                    &buffers.ring[capture],
                    0,
                    &self.ple_state.ring,
                    0,
                    ring,
                );
            }
            encoder.end_encoding();
            // Une restauration partielle laisserait des états mêlés.
            if let Err(error) = restored.and_then(|()| commit_and_wait(command_buffer)) {
                self.poisoned = true;
                return Err(error);
            }
            let engaged = pending.position + k + 1;
            #[cfg(test)]
            let truncate = mutation != Some(VerifyMutation::NoTruncate);
            #[cfg(not(test))]
            let truncate = true;
            for state in self.states[..c.layer_kinds.len()]
                .iter_mut()
                .filter(|_| truncate)
            {
                if let LayerState::Full(full) = state {
                    if let Err(error) = full.0.truncate(engaged) {
                        self.poisoned = true;
                        return Err(error);
                    }
                }
            }
        }
        self.position = pending.position + k + 1;
        self.ple_state.step = pending.ple_step + k + 1;
        self.ple_state.context = pending.contexts[k].clone();
        #[cfg(test)]
        match mutation {
            Some(VerifyMutation::PleStepOff) => self.ple_state.step -= 1,
            Some(VerifyMutation::StaleContext) if k > 0 => {
                self.ple_state.context = pending.contexts[k - 1].clone();
            }
            _ => {}
        }
        Ok(())
    }

    /// Oracles : état observable du tronc, préfixes valides seulement (positions
    /// engagées) : états DeltaNet, lignes KV `[0, len)`, blocs poolés complets,
    /// clés brutes du bloc en cours, anneau, pas et contexte PLE.
    #[cfg(test)]
    pub(crate) fn state_digest(&self) -> Result<StateDigest> {
        let c = &self.ckpt.config;
        let mut bits = vec![self.position as u32, self.ple_state.step as u32];
        for state in &self.states[..c.layer_kinds.len()] {
            match state {
                LayerState::Linear(state) => bits.extend(state.state_bits()?),
                LayerState::Full(full) => self.full_bits(&full.0, &full.1, &mut bits)?,
            }
        }
        bits.extend(read_u32_buffer(
            &self.ple_state.ring,
            (self.ple_state.ring.length() / 4) as usize,
        )?);
        Ok(StateDigest {
            bits,
            context: self.ple_state.context.clone(),
        })
    }

    /// Oracles : empreinte de l'historique MTP (KV, clés poolées des blocs
    /// complets, clés brutes du bloc en cours).
    #[cfg(test)]
    pub(crate) fn mtp_state_digest(&self) -> Result<Vec<u32>> {
        match self.states.get(self.ckpt.config.layer_kinds.len()) {
            Some(LayerState::Full(full)) => {
                let mut bits = Vec::new();
                self.full_bits(&full.0, &full.1, &mut bits)?;
                Ok(bits)
            }
            _ => Err(InferError::Config(
                "Flash-Next : tête MTP absente".to_string(),
            )),
        }
    }

    /// Oracles : bits observables d'une couche QSA (KV, clés poolées des blocs
    /// complets, clés brutes du bloc en cours).
    #[cfg(test)]
    fn full_bits(
        &self,
        kv: &FullAttentionMetalState,
        indexer: &IndexerState,
        bits: &mut Vec<u32>,
    ) -> Result<()> {
        let c = &self.ckpt.config;
        let compress = self.indexer_dims.compress;
        let d = self.indexer_dims.head_dim;
        let kv_dim = c.kv_heads * c.head_dim;
        let len = kv.len();
        let (keys, values) = kv.dense_f32_buffers()?;
        bits.push(len as u32);
        bits.extend(read_u32_buffer(keys, len * kv_dim)?);
        bits.extend(read_u32_buffer(values, len * kv_dim)?);
        bits.extend(read_u32_buffer(&indexer.pooled, (len / compress) * d)?);
        bits.extend(read_u32_buffer(&indexer.raw, (len % compress) * d)?);
        Ok(())
    }

    fn fresh_states(&self, capacity: usize) -> Result<Vec<LayerState>> {
        let c = &self.ckpt.config;
        let spec = self.linear_spec;
        let dims = self.linear_dims;
        let conv_len = (spec.conv_kernel_dim - 1) * dims.conv_dim;
        let ssm_len = spec.num_value_heads * spec.value_head_dim * spec.key_head_dim;
        let mtp = self
            .weights
            .mtp
            .as_ref()
            .map(|_| crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Full);
        c.layer_kinds
            .iter()
            .chain(mtp.iter())
            .map(|kind| match kind {
                crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Linear => {
                    let mut state = None;
                    self.executor.ensure_linear_attention_metal_state(
                        &mut state,
                        &vec![0.0; conv_len],
                        &vec![0.0; ssm_len],
                        dims.conv_dim,
                        conv_len,
                        ssm_len,
                        spec,
                    )?;
                    state
                        .map(LayerState::Linear)
                        .ok_or_else(|| InferError::Metal("état DeltaNet non alloué".to_string()))
                }
                crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Full => {
                    // KV épinglé en f32 : ni `SARAGOSSA_RUST_KV_BF16` ni le KV
                    // quantifié ne s'appliquent (l'oracle N2 compare à 1e-5).
                    let kv = self.arena.full_attention_with_format(
                        capacity,
                        c.heads,
                        c.kv_heads,
                        c.head_dim,
                        crate::decode_resident::KvFormat::F32,
                        None,
                    )?;
                    let indexer = IndexerState {
                        pooled: zeros(
                            &self.executor,
                            (capacity / c.indexer_compress + 1) * c.indexer_head_dim,
                        ),
                        raw: zeros(&self.executor, c.indexer_compress * c.indexer_head_dim),
                    };
                    Ok(LayerState::Full(Box::new((kv, indexer))))
                }
            })
            .collect()
    }

    /// Octets alloués par la session hors poids (oracles) : buffers propres,
    /// états DeltaNet et instantanés, KV f32. Les buffers de capture des
    /// oracles (`#[cfg(test)]`) sont exclus de la mesure comme du budget de
    /// production.
    #[cfg(test)]
    pub(crate) fn owned_state_bytes(&self) -> u64 {
        let c = &self.ckpt.config;
        let own = [
            &self.h[0],
            &self.h[1],
            &self.x,
            &self.branch,
            &self.logits,
            &self.index,
            &self.capture,
            &self.hc_scratch.normed,
            &self.hc_scratch.low,
            &self.hc_scratch.up,
            &self.hc_scratch.inject,
            &self.hc_scratch.down_partial,
            &self.ple_scratch.emb,
            &self.ple_scratch.key,
            &self.ple_scratch.key_normed,
            &self.ple_scratch.value,
            &self.ple_scratch.query,
            &self.ple_scratch.gated,
            &self.ple_scratch.normed,
            &self.qsa_scratch.q_gate,
            &self.qsa_scratch.q_raw,
            &self.qsa_scratch.gate,
            &self.qsa_scratch.q_roped,
            &self.qsa_scratch.k_raw,
            &self.qsa_scratch.k_roped,
            &self.qsa_scratch.v,
            &self.qsa_scratch.scores,
            &self.qsa_scratch.ctx,
            &self.qsa_scratch.gated,
            &self.ple_state.bytes,
            &self.ple_state.ring,
            &self.indexer_scratch.qk,
            &self.indexer_scratch.q,
            &self.indexer_scratch.scores,
            &self.indexer_scratch.selected,
        ]
        .iter()
        .map(|b| b.length())
        .sum::<u64>()
            + self.verify.as_ref().map_or(0, |v| {
                v.embed
                    .iter()
                    .chain(&v.ple_bytes)
                    .chain(&v.index)
                    .chain(&v.residual)
                    .chain(v.raw.iter().flatten())
                    .chain(&v.ring)
                    .map(|b| b.length())
                    .sum::<u64>()
                    + v.linear
                        .iter()
                        .flatten()
                        .map(MetalExecutor::linear_attn_state_bytes)
                        .sum::<u64>()
            })
            + self.mtp.as_ref().map_or(0, |m| {
                [
                    &m.h[0],
                    &m.h[1],
                    &m.embed,
                    &m.e_normed,
                    &m.fe,
                    &m.h_normed,
                    &m.fh,
                    &m.out,
                    &m.raw_snapshot,
                ]
                .iter()
                .map(|b| b.length())
                .sum::<u64>()
            });
        let states: u64 = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Linear(state) => MetalExecutor::linear_attn_state_bytes(state),
                LayerState::Full(full) => {
                    (full.0.capacity() * 2 * c.kv_heads * c.head_dim * std::mem::size_of::<f32>())
                        as u64
                        + full.1.pooled.length()
                        + full.1.raw.length()
                }
            })
            .sum();
        let snapshots: u64 = self
            .initial_linear
            .iter()
            .flatten()
            .map(MetalExecutor::linear_attn_state_bytes)
            .sum();
        let gathered =
            (self.gathered.capacity() * 2 * c.kv_heads * c.head_dim * std::mem::size_of::<f32>())
                as u64;
        own + states + snapshots + gathered
    }

    /// Octets alloués sur le device Metal de la session (oracles).
    #[cfg(test)]
    pub(crate) fn device_allocated_bytes(&self) -> u64 {
        self.executor.device().current_allocated_size()
    }

    /// Profil GPU accumulé, s'il est actif.
    #[cfg(feature = "devtools")]
    pub(crate) fn profiler_mut(&mut self) -> Option<&mut super::profile::StepProfiler> {
        self.profiler.as_mut()
    }

    /// Relit la sélection et les scores QSA du dernier pas (oracles). La
    /// position a déjà avancé : le pas relu est `position − 1`.
    #[cfg(test)]
    fn read_qsa_capture(&self) -> Result<Vec<Option<QsaCapture>>> {
        let c = &self.ckpt.config;
        let keep = self.indexer_dims.keep;
        let complete = self.position / self.indexer_dims.compress;
        let stride = self.capacity / self.indexer_dims.compress + 1;
        let selections = read_u32_buffer(&self.selection_capture, c.layer_kinds.len() * keep)?;
        let scores = read_f32_buffer(&self.score_capture, c.layer_kinds.len() * stride)?;
        Ok(c.layer_kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                (*kind == crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind::Full && complete > keep)
                    .then(|| {
                        (
                            selections[index * keep..(index + 1) * keep].to_vec(),
                            scores[index * stride..index * stride + complete].to_vec(),
                        )
                    })
            })
            .collect())
    }

    /// Sonde de l'indexeur de la couche `layer` après un pas (oracles) :
    /// requêtes de l'indexeur (valables pour la dernière couche QSA du pas),
    /// clés poolées des `blocks` premiers blocs et clés brutes du bloc en cours.
    #[cfg(test)]
    pub(crate) fn indexer_probe(&self, layer: usize, blocks: usize) -> Result<IndexerProbe> {
        let d = self.indexer_dims.head_dim;
        let LayerState::Full(full) = &self.states[layer] else {
            return Err(InferError::Config(format!("couche {layer} sans indexeur")));
        };
        Ok(IndexerProbe {
            qk: read_f32_buffer(&self.indexer_scratch.qk, (self.indexer_dims.heads + 1) * d)?,
            q: read_f32_buffer(&self.indexer_scratch.q, self.indexer_dims.heads * d)?,
            pooled: read_f32_buffer(&full.1.pooled, blocks * d)?,
            raw: read_f32_buffer(&full.1.raw, self.indexer_dims.compress * d)?,
        })
    }

    /// Position du prochain token.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// Exécuteur, kernels, lm_head et largeur cachée (profil en lignes).
    #[cfg(all(test, feature = "devtools"))]
    pub(crate) fn head_parts(
        &self,
    ) -> (
        &MetalExecutor,
        &FlashNextKernels,
        &crate::metal_backend::MetalLinearWeightBuffers,
        usize,
    ) {
        (
            &self.executor,
            &self.kernels,
            &self.weights.lm_head,
            self.ckpt.config.hidden,
        )
    }

    /// Couches DeltaNet (poids, état) et MoE de toutes les couches, avec leurs
    /// dimensions (profil en lignes de l'étape 6 D).
    #[cfg(all(test, feature = "devtools"))]
    pub(crate) fn deltanet_and_moe(&self) -> DeltaNetAndMoe<'_> {
        let mut linear = Vec::new();
        for (layer, state) in self.weights.layers.iter().zip(&self.states) {
            if let (MixerWeights::Linear(w), LayerState::Linear(state)) = (&layer.mixer, state) {
                linear.push((&**w, state));
            }
        }
        DeltaNetAndMoe {
            linear,
            moe: self.weights.layers.iter().map(|layer| &layer.moe).collect(),
            spec: self.linear_spec,
            dims: self.linear_dims,
            top_k: self.ckpt.config.top_k,
        }
    }

    /// Largeur du résiduel hc (oracles).
    #[cfg(test)]
    pub(crate) fn hc_width(&self) -> usize {
        self.hc_dims.width()
    }

    /// Capacité de la session (tokens).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remet la session au début d'une séquence : états DeltaNet initiaux, KV
    /// vide, PLE remis à zéro. La session est invalidée pendant la
    /// restauration et ne redevient utilisable qu'après un succès complet, ce
    /// qui lève aussi l'invalidation d'une erreur GPU antérieure.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de copie Metal ou de troncature du KV.
    pub(crate) fn reset(&mut self) -> Result<()> {
        // Invalide pendant la restauration : un échec partiel laisse des états
        // mêlés, que seul un `reset` complet réhabilite.
        self.poisoned = true;
        self.pending = None;
        for (index, (state, initial)) in
            self.states.iter_mut().zip(&self.initial_linear).enumerate()
        {
            #[cfg(not(test))]
            let _ = index;
            #[cfg(test)]
            if self
                .fail_reset_after_layer
                .take_if(|after| *after < index)
                .is_some()
            {
                return Err(InferError::Metal(format!(
                    "reset en échec à la couche {index} (injecté)"
                )));
            }
            match (state, initial) {
                (LayerState::Linear(state), Some(initial)) => {
                    self.executor.restore_linear_attn_state(state, initial)?;
                }
                (LayerState::Full(full), None) => full.0.truncate(0)?,
                _ => {
                    return Err(InferError::Config(
                        "reset Flash-Next : états et instantanés désalignés".to_string(),
                    ))
                }
            }
        }
        let c = &self.ckpt.config;
        self.ple_state.reset(c.ngram_size, c.eos);
        self.position = 0;
        self.mtp_raw_rows = None;
        self.epoch += 1;
        self.base_checkpoint = None;
        self.poisoned = false;
        Ok(())
    }

    /// Point de reprise à la position courante : états DeltaNet, longueurs KV,
    /// anneau et contexte PLE. Le KV et les clés poolées des positions
    /// antérieures ne sont pas copiés : la reprise est valide tant que la
    /// session n'est pas remise à zéro (seules des positions ≥ point sont
    /// réécrites ensuite).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de copie Metal.
    pub(crate) fn checkpoint(&self, guard: Option<&MemoryGuard>) -> Result<SessionCheckpoint> {
        self.refuse_pending()?;
        if let Some(guard) = guard {
            let bytes: u64 = self
                .states
                .iter()
                .map(|state| match state {
                    LayerState::Linear(state) => MetalExecutor::linear_attn_state_bytes(state),
                    LayerState::Full(full) => full.1.raw.length(),
                })
                .sum::<u64>()
                + self.ple_state.ring.length();
            guard
                .check_allocation(bytes)
                .map_err(InferError::MemoryGuard)?;
        }
        let linear = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Linear(state) => {
                    self.executor.snapshot_linear_attn_state(state).map(Some)
                }
                LayerState::Full(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
        let kv_lens = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Full(full) => Some(full.0.len()),
                LayerState::Linear(_) => None,
            })
            .collect();
        let ring_len = (self.ple_state.ring.length() / 4) as usize;
        let ple_ring = zeros(&self.executor, ring_len);
        self.copy_f32(&self.ple_state.ring, &ple_ring, ring_len)?;
        let indexer_raw = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Full(full) => {
                    let len = (full.1.raw.length() / 4) as usize;
                    let copy = zeros(&self.executor, len);
                    self.copy_f32(&full.1.raw, &copy, len).map(|()| Some(copy))
                }
                LayerState::Linear(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
        let id = self.next_checkpoint.get();
        self.next_checkpoint.set(id + 1);
        Ok(SessionCheckpoint {
            id,
            position: self.position,
            epoch: self.epoch,
            linear,
            kv_lens,
            indexer_raw,
            ple_ring,
            ple_context: self.ple_state.context.clone(),
            ple_step: self.ple_state.step,
        })
    }

    /// Reprend au point `checkpoint`. La session est invalidée pendant la
    /// restauration et refuse un point d'une époque antérieure à un `reset`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de copie Metal, de troncature, ou un point périmé.
    pub(crate) fn restore(&mut self, checkpoint: &SessionCheckpoint) -> Result<()> {
        self.refuse_pending()?;
        // Valide si pris sur la branche courante, ou s'il est la base de cette
        // branche (reprises répétées depuis un même point).
        let same_branch = checkpoint.epoch == self.epoch && checkpoint.position <= self.position;
        if !same_branch && self.base_checkpoint != Some(checkpoint.id) {
            return Err(InferError::Config(
                "point de reprise Flash-Next périmé (autre branche de l'historique)".to_string(),
            ));
        }
        self.poisoned = true;
        for ((state, linear), len) in self
            .states
            .iter_mut()
            .zip(&checkpoint.linear)
            .zip(&checkpoint.kv_lens)
        {
            match (state, linear, len) {
                (LayerState::Linear(state), Some(snapshot), None) => {
                    self.executor.restore_linear_attn_state(state, snapshot)?;
                }
                (LayerState::Full(full), None, Some(len)) => full.0.truncate(*len)?,
                _ => {
                    return Err(InferError::Config(
                        "point de reprise Flash-Next désaligné".to_string(),
                    ))
                }
            }
        }
        for (state, raw) in self.states.iter().zip(&checkpoint.indexer_raw) {
            if let (LayerState::Full(full), Some(raw)) = (state, raw) {
                let len = (raw.length() / 4) as usize;
                self.copy_f32(raw, &full.1.raw, len)?;
            }
        }
        let ring_len = (checkpoint.ple_ring.length() / 4) as usize;
        self.copy_f32(&checkpoint.ple_ring, &self.ple_state.ring, ring_len)?;
        self.ple_state.context = checkpoint.ple_context.clone();
        self.ple_state.step = checkpoint.ple_step;
        self.position = checkpoint.position;
        self.mtp_raw_rows = None;
        self.epoch += 1;
        self.base_checkpoint = Some(checkpoint.id);
        self.poisoned = false;
        Ok(())
    }

    /// Copie `len` f32 de `src` vers `dst` (command buffer attendu).
    fn copy_f32(&self, src: &Buffer, dst: &Buffer, len: usize) -> Result<()> {
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.executor
            .encode_copy_with_offsets(encoder, src, 0, dst, 0, len)?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)
    }

    /// Décode `token` à la position courante. `logits` et `layers` demandent la
    /// relecture des logits et des résiduels par couche (oracles).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la capacité est atteinte, sur lecture n-gram ou
    /// sur encodage et exécution Metal.
    pub(crate) fn step(&mut self, token: usize, logits: bool, layers: bool) -> Result<StepOutput> {
        self.step_with(token, logits, layers, true)
    }

    /// Rejoue `token` (contexte) sans mixer final ni lm_head : états, KV et
    /// indexeur avancent, aucun argmax n'est produit.
    ///
    /// # Errors
    ///
    /// Mêmes erreurs que [`Self::step`].
    pub(crate) fn replay(&mut self, token: usize) -> Result<()> {
        self.step_with(token, false, false, false).map(|_| ())
    }

    fn step_with(
        &mut self,
        token: usize,
        logits: bool,
        layers: bool,
        head: bool,
    ) -> Result<StepOutput> {
        if self.poisoned {
            return Err(InferError::Metal(
                "Flash-Next résident : session invalidée (erreur GPU ou reset incomplet)"
                    .to_string(),
            ));
        }
        self.refuse_pending()?;
        let c = &self.ckpt.config;
        if self.position >= self.capacity {
            return Err(InferError::Config(format!(
                "Flash-Next résident : capacité de {} tokens atteinte",
                self.capacity
            )));
        }
        let host_started = std::time::Instant::now();
        // Embedding : ligne u8 déquantifiée, répétée sur les `hc` flux.
        let row = self.ckpt.affine_rows(
            &Qwen4ExpCheckpoint::trunk("embed_tokens"),
            None,
            token..token + 1,
        )?;
        let mut streams = Vec::with_capacity(self.hc_dims.width());
        for _ in 0..c.hc_count {
            streams.extend_from_slice(row.data());
        }
        write_f32_buffer(&self.h[0], &streams)?;
        // PLE : hachage sur un clone du contexte (engagé seulement si le pas
        // aboutit) et lectures n-gram terminées avant le commit.
        let p = &self.ckpt.ngram_params;
        let mut context = self.ple_state.context.clone();
        let ids = context.step(
            token as u64,
            c.heads_per_ngram,
            &p.multipliers,
            &p.vocab_sizes,
            &p.offsets,
        );
        let mut bytes = vec![0_u8; c.ple_dim];
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_ngram_read) {
            return Err(InferError::Config(
                "lecture n-gram en échec (injectée)".to_string(),
            ));
        }
        self.ckpt.ngram_rows_raw(&ids, &mut bytes)?;
        self.ple_state.stage_bytes(&bytes)?;
        let host = host_started.elapsed();

        // Métadonnées avancées pendant l'encodage, restaurées si celui-ci échoue :
        // rien n'est soumis, le GPU n'a donc rien modifié.
        let kv_lens: Vec<Option<usize>> = self
            .states
            .iter()
            .map(|state| match state {
                LayerState::Full(full) => Some(full.0.len()),
                LayerState::Linear(_) => None,
            })
            .collect();
        let ple_step = self.ple_state.step;
        let encode_started = std::time::Instant::now();
        let queue = self.queue.clone();
        let command_buffer = queue.new_command_buffer();
        #[cfg(feature = "devtools")]
        let mut profiler = self.profiler.take();
        #[cfg(feature = "devtools")]
        let mut encoders = match profiler.as_mut() {
            Some(profiler) => {
                profiler.begin_step();
                Encoders::profiled(command_buffer, profiler)
            }
            None => Encoders::single(command_buffer),
        };
        #[cfg(not(feature = "devtools"))]
        let mut encoders = Encoders::single(command_buffer);
        let io = TokenIo {
            ple_bytes: self.ple_state.bytes.clone(),
            index: self.index.clone(),
            residual: None,
        };
        let encoded = self.encode_token(&mut encoders, logits, layers, head, &io);
        if let Err(error) = encoded {
            drop(encoders);
            #[cfg(feature = "devtools")]
            {
                if let Some(profiler) = profiler.as_mut() {
                    profiler.abort_step();
                }
                self.profiler = profiler;
            }
            self.ple_state.step = ple_step;
            for (state, len) in self.states.iter_mut().zip(kv_lens) {
                if let (LayerState::Full(kv), Some(len)) = (state, len) {
                    if kv.0.truncate(len).is_err() {
                        self.poisoned = true;
                    }
                }
            }
            return Err(error);
        }
        encoders.end();
        let encode = encode_started.elapsed();
        #[cfg(feature = "devtools")]
        if let Some(profiler) = profiler.as_ref() {
            profiler.encode_resolve(command_buffer);
        }
        if let Err(error) = commit_and_wait(command_buffer) {
            // Exécution GPU partielle possible : états récurrents et KV incertains.
            self.poisoned = true;
            return Err(error);
        }
        #[cfg(feature = "devtools")]
        {
            if let Some(profiler) = profiler.as_mut() {
                profiler.end_step()?;
            }
            self.profiler = profiler;
        }
        self.ple_state.context = context;
        self.position += 1;
        let width = self.hc_dims.width();
        let (vocab, layer_count) = (self.ckpt.config.vocab, self.ckpt.config.layer_kinds.len());
        let argmax = if head {
            read_u32_buffer(&self.index, 1)?
                .first()
                .copied()
                .ok_or_else(|| InferError::Metal("argmax résident absent".to_string()))?
        } else {
            u32::MAX
        };
        Ok(StepOutput {
            argmax,
            logits: logits
                .then(|| read_f32_buffer(&self.logits, vocab))
                .transpose()?,
            layers: layers
                .then(|| {
                    read_f32_buffer(&self.capture, layer_count * width)
                        .map(|all| all.chunks_exact(width).map(<[f32]>::to_vec).collect())
                })
                .transpose()?,
            #[cfg(test)]
            qsa: layers.then(|| self.read_qsa_capture()).transpose()?,
            #[cfg(test)]
            routing: layers
                .then(|| {
                    read_u32_buffer(&self.routing_capture, layer_count * self.ckpt.config.top_k)
                        .map(|all| {
                            all.chunks_exact(self.ckpt.config.top_k)
                                .map(<[u32]>::to_vec)
                                .collect()
                        })
                })
                .transpose()?,
            host,
            encode,
        })
    }

    /// Oracles : gains bruts des normes d'entrée MTP (embedding, flux), pour les
    /// mutations de gain.
    #[cfg(test)]
    pub(crate) fn mtp_norm_buffers(&self) -> Option<(&Buffer, &Buffer)> {
        self.weights
            .mtp
            .as_ref()
            .map(|w| (&w.norm_embedding, &w.norm_hidden))
    }

    /// Tronque l'historique MTP à `rows` lignes : retire les lignes des drafts
    /// enchaînés, construites sur des résiduels spéculatifs, et restaure
    /// l'anneau des clés brutes depuis l'instantané pris à `rows` lignes
    /// ([`Self::mtp_snapshot_raw`]). Les blocs poolés de ces lignes sont
    /// réécrits avant d'être lus.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur sur une session invalidée, sans tête MTP, si `rows`
    /// dépasse l'historique, ou si le bloc en cours n'est pas vide sans
    /// instantané pris à `rows` lignes. Un échec de la copie de restauration
    /// invalide la session.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "spéculation MTP : boucle à l'étape 6 C")
    )]
    pub(crate) fn mtp_truncate(&mut self, rows: usize) -> Result<()> {
        self.refuse_poisoned()?;
        self.refuse_pending()?;
        let (Some(mtp), Some(LayerState::Full(full))) = (&self.mtp, self.states.last_mut()) else {
            return Err(InferError::Config(
                "Flash-Next : tête MTP absente".to_string(),
            ));
        };
        let len = full.0.len();
        if rows > len {
            return Err(InferError::Config(format!(
                "troncature MTP à {rows} lignes au-delà de l'historique ({len})"
            )));
        }
        if len == rows {
            return Ok(());
        }
        // Les lignes retirées ont écrit `raw[ligne % compress]` : sans
        // instantané pris à `rows` lignes, le bloc en cours serait poolé avec
        // la clé d'une ligne retirée. Un bloc en cours vide n'en lit aucune.
        let restore = rows % self.indexer_dims.compress != 0;
        if restore && self.mtp_raw_rows != Some(rows) {
            return Err(InferError::Config(format!(
                "troncature MTP à {rows} lignes sans instantané des clés brutes"
            )));
        }
        full.0.truncate(rows)?;
        #[cfg(test)]
        let restore = restore && !self.skip_mtp_raw_restore;
        if !restore {
            return Ok(());
        }
        let words = (mtp.raw_snapshot.length() / 4) as usize;
        let (snapshot, raw) = (mtp.raw_snapshot.clone(), full.1.raw.clone());
        // Longueur déjà réduite : une restauration manquée laisserait des clés
        // brutes retirées sous un historique d'apparence valide.
        let restored = self.copy_f32(&snapshot, &raw, words);
        if restored.is_err() {
            self.poisoned = true;
        }
        restored
    }

    /// Instantané de l'anneau des clés brutes de l'indexeur MTP aux lignes
    /// courantes : les drafts enchaînés qui suivent écrivent dans cet anneau,
    /// et [`Self::mtp_truncate`] le restaure en revenant à ces lignes.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une vérification est en attente, sans tête MTP ou
    /// sur copie Metal.
    pub(crate) fn mtp_snapshot_raw(&mut self) -> Result<()> {
        self.refuse_poisoned()?;
        self.refuse_pending()?;
        self.mtp_raw_rows = None;
        let (Some(mtp), Some(LayerState::Full(full))) = (&self.mtp, self.states.last()) else {
            return Err(InferError::Config(
                "Flash-Next : tête MTP absente".to_string(),
            ));
        };
        let len = (mtp.raw_snapshot.length() / 4) as usize;
        self.copy_f32(&full.1.raw, &mtp.raw_snapshot, len)?;
        self.mtp_raw_rows = Some(full.0.len());
        Ok(())
    }

    /// Résiduel de sortie du dernier pas MTP, résident (entrée du draft suivant).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
    )]
    pub(crate) fn mtp_hidden_buffer(&self) -> Option<&Buffer> {
        self.mtp.as_ref().map(|m| &m.out)
    }

    /// Oracles : décalages des seules requêtes, attention et indexeur.
    #[cfg(test)]
    pub(crate) fn set_query_offsets(&mut self, attention: usize, indexer: usize) {
        self.qsa_dims.query_offset = attention;
        self.qsa_dims.index_query_offset = indexer;
    }

    /// Lignes de l'historique MTP (longueur du KV de la tête) ; `None` sans tête.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
    )]
    pub(crate) fn mtp_rows(&self) -> Option<usize> {
        self.mtp.as_ref()?;
        match self.states.last() {
            Some(LayerState::Full(full)) => Some(full.0.len()),
            _ => None,
        }
    }

    /// Pas MTP sur la paire `(h, token)` : `h` est un résiduel hc du tronc avant
    /// le mixer final (ou la sortie d'un pas MTP précédent), `token` le token
    /// suivant. Ajoute la ligne `rows` à l'historique de la tête (position RoPE
    /// et blocs d'indexeur en indices de ligne) et renvoie le draft.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur sans tête MTP, sur forme ou token invalides, capacité
    /// atteinte, ou sur encodage et exécution Metal (session invalidée dans ce
    /// dernier cas, comme [`Self::step`]).
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
    )]
    pub(crate) fn mtp_step(
        &mut self,
        h: MtpInput<'_>,
        token: usize,
        logits: bool,
        hidden: bool,
    ) -> Result<MtpOutput> {
        if self.poisoned {
            return Err(InferError::Metal(
                "Flash-Next résident : session invalidée (erreur GPU ou reset incomplet)"
                    .to_string(),
            ));
        }
        self.refuse_pending()?;
        let c = &self.ckpt.config;
        let width = self.hc_dims.width();
        let rows = self
            .mtp_rows()
            .ok_or_else(|| InferError::Config("Flash-Next : tête MTP absente".to_string()))?;
        let len = match h {
            MtpInput::Host(h) => h.len(),
            MtpInput::Resident(buffer) => (buffer.length() / 4) as usize,
        };
        if len != width || token >= c.vocab {
            return Err(InferError::Shape(format!(
                "MTP : h de {len} valeurs (largeur {width}), token {token} (vocabulaire {})",
                c.vocab
            )));
        }
        if rows >= self.capacity {
            return Err(InferError::Config(format!(
                "Flash-Next MTP : capacité de {} lignes atteinte",
                self.capacity
            )));
        }
        let row = self.ckpt.affine_rows(
            &Qwen4ExpCheckpoint::trunk("embed_tokens"),
            None,
            token..token + 1,
        )?;
        let buffers = self
            .mtp
            .as_ref()
            .ok_or_else(|| InferError::Config("Flash-Next : tête MTP absente".to_string()))?;
        write_f32_buffer(&buffers.embed, row.data())?;
        let queue = self.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let mut encoders = Encoders::single(command_buffer);
        let encoded = match h {
            MtpInput::Host(h) => write_f32_buffer(&buffers.h[0], h),
            MtpInput::Resident(source) => encoders.at(Brick::Hc).and_then(|encoder| {
                self.executor.encode_copy_with_offsets(
                    encoder,
                    source,
                    0,
                    &buffers.h[0],
                    0,
                    width,
                )?;
                crate::metal_backend::post_dispatch_barrier(encoder);
                Ok(())
            }),
        }
        .and_then(|()| self.encode_mtp(&mut encoders, rows, logits));
        if let Err(error) = encoded {
            drop(encoders);
            // Rien n'est soumis : seule la longueur du KV a avancé.
            if let Some(LayerState::Full(full)) = self.states.last_mut() {
                if full.0.truncate(rows).is_err() {
                    self.poisoned = true;
                }
            }
            return Err(error);
        }
        encoders.end();
        if let Err(error) = commit_and_wait(command_buffer) {
            self.poisoned = true;
            return Err(error);
        }
        // Le pas est engagé (ligne `rows` ajoutée) : une relecture en échec le
        // signale explicitement, pour qu'aucune reprise ne rejoue la paire.
        let engaged = |error: InferError| {
            InferError::Metal(format!(
                "pas MTP engagé (ligne {rows} ajoutée), relecture en échec : {error}"
            ))
        };
        self.mtp_post_commit(rows, logits, hidden).map_err(engaged)
    }

    /// Relectures d'un pas MTP soumis et terminé.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
    )]
    fn mtp_post_commit(&self, rows: usize, logits: bool, hidden: bool) -> Result<MtpOutput> {
        // `rows` ne sert qu'aux captures d'oracles.
        #[cfg(not(test))]
        let _ = rows;
        let width = self.hc_dims.width();
        let argmax = read_u32_buffer(&self.index, 1)?
            .first()
            .copied()
            .ok_or_else(|| InferError::Metal("argmax MTP absent".to_string()))?;
        let c = &self.ckpt.config;
        #[cfg(test)]
        let slot = c.layer_kinds.len();
        Ok(MtpOutput {
            argmax,
            logits: logits
                .then(|| read_f32_buffer(&self.logits, c.vocab))
                .transpose()?,
            hidden: match (hidden, &self.mtp) {
                (true, Some(m)) => Some(read_f32_buffer(&m.out, width)?),
                _ => None,
            },
            #[cfg(test)]
            input: read_f32_buffer(&self.mtp_input_capture, width)?,
            #[cfg(test)]
            qsa: {
                let complete = (rows + 1) / self.indexer_dims.compress;
                let keep = self.indexer_dims.keep;
                if complete > keep {
                    let stride = self.capacity / self.indexer_dims.compress + 1;
                    let selection = read_u32_buffer(&self.selection_capture, (slot + 1) * keep)?;
                    let scores = read_f32_buffer(&self.score_capture, (slot + 1) * stride)?;
                    Some((
                        selection[slot * keep..(slot + 1) * keep].to_vec(),
                        scores[slot * stride..slot * stride + complete].to_vec(),
                    ))
                } else {
                    None
                }
            },
            #[cfg(test)]
            routing: Some(
                read_u32_buffer(&self.routing_capture, (slot + 1) * c.top_k)?
                    [slot * c.top_k..(slot + 1) * c.top_k]
                    .to_vec(),
            ),
        })
    }

    /// Encode un pas MTP à la ligne `rows` : entrée fc par flux, couche QSA +
    /// MoE ([`LayerEncoder::encode`]), copie du résiduel de sortie, mixer de
    /// tête et lm_head (argmax, logits sur demande). Ne soumet rien.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "tête MTP : spéculation à l'étape 6 C")
    )]
    fn encode_mtp(&mut self, encoders: &mut Encoders<'_>, rows: usize, logits: bool) -> Result<()> {
        let c = &self.ckpt.config;
        let width = self.hc_dims.width();
        let missing = || InferError::Config("Flash-Next : tête MTP absente".to_string());
        let weights = self.weights.mtp.as_ref().ok_or_else(missing)?;
        let buffers = self.mtp.as_ref().ok_or_else(missing)?;
        let state = self.states.last_mut().ok_or_else(missing)?;
        let encoder = encoders.at(Brick::Hc)?;
        // Normes : `e` en un groupe de `hidden`, `h` flux par flux.
        encode_rms_norm_1p(
            &self.kernels,
            encoder,
            &buffers.embed,
            &weights.norm_embedding,
            &buffers.e_normed,
            HcDims {
                hc: 1,
                ..self.hc_dims
            },
        );
        encode_rms_norm_1p(
            &self.kernels,
            encoder,
            &buffers.h[0],
            &weights.norm_hidden,
            &buffers.h_normed,
            self.hc_dims,
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        self.executor.encode_matmul_weight_buffers(
            encoder,
            &buffers.e_normed,
            1,
            c.hidden,
            &weights.fc_embedding,
            &buffers.fe,
            true,
        )?;
        // `fc_hidden` partagé par les flux : une ligne de `hidden` par flux.
        self.executor.encode_matmul_weight_buffers(
            encoder,
            &buffers.h_normed,
            c.hc_count,
            c.hidden,
            &weights.fc_hidden,
            &buffers.fh,
            true,
        )?;
        crate::metal_backend::post_dispatch_barrier(encoder);
        encoder.set_compute_pipeline_state(&self.kernels.mtp_add_streams);
        encoder.set_buffer(0, Some(&buffers.fh), 0);
        encoder.set_buffer(1, Some(&buffers.fe), 0);
        encoder.set_buffer(2, Some(&buffers.h[1]), 0);
        super::set_value(encoder, 3, &[width as u32, c.hidden as u32]);
        super::dispatch_1d(encoder, width);
        crate::metal_backend::post_dispatch_barrier(encoder);
        #[cfg(test)]
        {
            self.executor.encode_copy_with_offsets(
                encoder,
                &buffers.h[1],
                0,
                &self.mtp_input_capture,
                0,
                width,
            )?;
            crate::metal_backend::post_dispatch_barrier(encoder);
        }
        let ctx = LayerEncoder {
            kernels: &self.kernels,
            executor: &self.executor,
            arena: &self.arena,
            hc_scratch: &self.hc_scratch,
            hc_dims: self.hc_dims,
            qsa_scratch: &self.qsa_scratch,
            qsa_dims: self.qsa_dims,
            indexer_scratch: &self.indexer_scratch,
            indexer_dims: self.indexer_dims,
            linear_spec: self.linear_spec,
            linear_dims: self.linear_dims,
            x: &self.x,
            branch: &self.branch,
            hidden: c.hidden,
            top_k: c.top_k,
        };
        let mut cur = 1;
        let mut owned = Vec::new();
        #[cfg(test)]
        let hooks = LayerHooks {
            forced_selection: Some(&mut self.mtp_forced_selection),
            forced_routing: Some(&mut self.mtp_forced_routing),
            capture: Some(CaptureSlot {
                slot: c.layer_kinds.len(),
                selection: &self.selection_capture,
                scores: &self.score_capture,
                routing: &self.routing_capture,
                stride: self.capacity / self.indexer_dims.compress + 1,
            }),
        };
        ctx.encode(
            encoders,
            &mut owned,
            &mut self.gathered,
            &weights.layer,
            state,
            &buffers.h,
            &mut cur,
            rows,
            c.layer_kinds.len(),
            #[cfg(test)]
            hooks,
        )?;
        let encoder = encoders.at(Brick::Mixer)?;
        self.executor.encode_copy_with_offsets(
            encoder,
            &buffers.h[cur],
            0,
            &buffers.out,
            0,
            width,
        )?;
        encode_hc_mix(
            &self.kernels,
            &self.executor,
            encoder,
            &weights.mixer,
            &self.hc_scratch,
            self.hc_dims,
            &buffers.h[cur],
            &self.x,
        )?;
        let encoder = encoders.at(Brick::LmHead)?;
        if logits {
            self.executor.encode_matmul_weight_buffers(
                encoder,
                &self.x,
                1,
                c.hidden,
                &self.weights.lm_head,
                &self.logits,
                true,
            )?;
        }
        self.executor.encode_lm_head_argmax_buffers(
            encoder,
            &mut owned,
            &self.x,
            &self.weights.lm_head,
            &self.index,
            c.hidden,
        )?;
        Ok(())
    }

    /// Encode les couches, le mixer final et le lm_head du token courant.
    fn encode_token(
        &mut self,
        encoders: &mut Encoders<'_>,
        logits: bool,
        layers: bool,
        head: bool,
        io: &TokenIo,
    ) -> Result<()> {
        let c = &self.ckpt.config;
        let mut owned = Vec::new();
        let width = self.hc_dims.width();
        let mut cur = 0_usize;
        let ctx = LayerEncoder {
            kernels: &self.kernels,
            executor: &self.executor,
            arena: &self.arena,
            hc_scratch: &self.hc_scratch,
            hc_dims: self.hc_dims,
            qsa_scratch: &self.qsa_scratch,
            qsa_dims: self.qsa_dims,
            indexer_scratch: &self.indexer_scratch,
            indexer_dims: self.indexer_dims,
            linear_spec: self.linear_spec,
            linear_dims: self.linear_dims,
            x: &self.x,
            branch: &self.branch,
            hidden: c.hidden,
            top_k: c.top_k,
        };
        for (index, (layer, state)) in self
            .weights
            .layers
            .iter()
            .zip(self.states.iter_mut())
            .enumerate()
        {
            if index == c.ple_layer {
                let encoder = encoders.at(Brick::Ple)?;
                encode_ple(
                    &self.kernels,
                    &self.executor,
                    encoder,
                    &self.weights.ple,
                    &mut self.ple_state,
                    &io.ple_bytes,
                    &self.ple_scratch,
                    self.hc_dims,
                    &self.h[cur],
                    &self.h[1 - cur],
                )?;
                cur = 1 - cur;
            }
            #[cfg(test)]
            let hooks = LayerHooks {
                forced_selection: self.forced_selection.get_mut(index),
                forced_routing: self.forced_routing.get_mut(index),
                capture: layers.then_some(CaptureSlot {
                    slot: index,
                    selection: &self.selection_capture,
                    scores: &self.score_capture,
                    routing: &self.routing_capture,
                    stride: self.capacity / self.indexer_dims.compress + 1,
                }),
            };
            ctx.encode(
                encoders,
                &mut owned,
                &mut self.gathered,
                layer,
                state,
                &self.h,
                &mut cur,
                self.position,
                index,
                #[cfg(test)]
                hooks,
            )?;
            let encoder = encoders.at(Brick::Hc)?;
            #[cfg(test)]
            if self
                .fail_encode_after_layer
                .take_if(|after| *after == index)
                .is_some()
            {
                return Err(InferError::Metal(format!(
                    "encodage en échec après la couche {index} (injecté)"
                )));
            }
            if layers {
                self.executor.encode_copy_with_offsets(
                    encoder,
                    &self.h[cur],
                    0,
                    &self.capture,
                    (index * width * std::mem::size_of::<f32>()) as u64,
                    width,
                )?;
            }
        }
        if let Some(residual) = &io.residual {
            #[cfg(test)]
            let source = if self.verify_mutation == Some(VerifyMutation::WrongResidual) {
                1 - cur
            } else {
                cur
            };
            #[cfg(not(test))]
            let source = cur;
            let encoder = encoders.at(Brick::Hc)?;
            self.executor.encode_copy_with_offsets(
                encoder,
                &self.h[source],
                0,
                residual,
                0,
                width,
            )?;
        }
        if !head {
            return Ok(());
        }
        let encoder = encoders.at(Brick::Mixer)?;
        encode_hc_mix(
            &self.kernels,
            &self.executor,
            encoder,
            &self.weights.mixer,
            &self.hc_scratch,
            self.hc_dims,
            &self.h[cur],
            &self.x,
        )?;
        let encoder = encoders.at(Brick::LmHead)?;
        if logits {
            self.executor.encode_matmul_weight_buffers(
                encoder,
                &self.x,
                1,
                c.hidden,
                &self.weights.lm_head,
                &self.logits,
                true,
            )?;
        }
        self.executor.encode_lm_head_argmax_buffers(
            encoder,
            &mut owned,
            &self.x,
            &self.weights.lm_head,
            &io.index,
            c.hidden,
        )?;
        Ok(())
    }
}

/// Entrées et sorties d'un pas de tronc : octets PLE du token, argmax et, pour
/// une ligne de vérification, copie du résiduel hc avant le mixer final. Les
/// lignes d'une même vérification ont chacune les leurs : aucun buffer hôte
/// n'est réécrit après son encodage.
struct TokenIo {
    ple_bytes: Buffer,
    index: Buffer,
    residual: Option<Buffer>,
}

/// Ressources d'encodage partagées par les couches, tronc et MTP : kernels,
/// scratch et dimensions. Les couches s'encodent l'une après l'autre dans un
/// même ordre de commandes ; le scratch n'est jamais partagé par deux couches
/// en vol.
struct LayerEncoder<'s> {
    kernels: &'s FlashNextKernels,
    executor: &'s MetalExecutor,
    arena: &'s DecodeResidentState,
    hc_scratch: &'s HcScratch,
    hc_dims: HcDims,
    qsa_scratch: &'s QsaScratch,
    qsa_dims: QsaDecodeDims,
    indexer_scratch: &'s IndexerScratch,
    indexer_dims: IndexerDims,
    linear_spec: LinearAttentionStepSpec,
    linear_dims: LinearAttnResidentDims,
    x: &'s Buffer,
    branch: &'s Buffer,
    hidden: usize,
    top_k: usize,
}

/// Case de capture d'une couche dans les tampons d'oracles.
#[cfg(test)]
struct CaptureSlot<'s> {
    slot: usize,
    selection: &'s Buffer,
    scores: &'s Buffer,
    routing: &'s Buffer,
    /// Pas des scores par case (`capacité / compress + 1`).
    stride: usize,
}

/// Oracles d'une couche : impositions (consommées aux mêmes points qu'avant
/// la factorisation : sélection dans la branche QSA, experts juste avant le
/// MoE) et captures.
#[cfg(test)]
struct LayerHooks<'s> {
    forced_selection: Option<&'s mut Option<Vec<u32>>>,
    forced_routing: Option<&'s mut Option<Vec<u32>>>,
    capture: Option<CaptureSlot<'s>>,
}

impl LayerEncoder<'_> {
    /// Encode une couche (hc d'attention, DeltaNet ou QSA, hc du MLP, MoE) à
    /// `position` : entrée `h[cur]`, intermédiaire `h[1 − cur]`, sortie de
    /// nouveau dans `h[cur]` après les deux combine (`cur` inchangé au total).
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal : état, KV compact, résiduels et oracles explicites"
    )]
    fn encode(
        &self,
        encoders: &mut Encoders<'_>,
        owned: &mut Vec<Buffer>,
        gathered: &mut FullAttentionMetalState,
        layer: &LayerWeights,
        state: &mut LayerState,
        h: &[Buffer; 2],
        cur: &mut usize,
        position: usize,
        index: usize,
        #[cfg(test)] hooks: LayerHooks<'_>,
    ) -> Result<()> {
        #[cfg(test)]
        let LayerHooks {
            forced_selection,
            forced_routing,
            capture,
        } = hooks;
        let encoder = encoders.at(Brick::Hc)?;
        encode_hc_mix(
            self.kernels,
            self.executor,
            encoder,
            &layer.attn_hc,
            self.hc_scratch,
            self.hc_dims,
            &h[*cur],
            self.x,
        )?;
        match (&layer.mixer, state) {
            (MixerWeights::Linear(w), LayerState::Linear(state)) => {
                let encoder = encoders.at(Brick::DeltaNet)?;
                self.executor.encode_linear_attn_resident_dense_buffers(
                    encoder,
                    self.x,
                    None,
                    self.branch,
                    w,
                    state,
                    self.linear_spec,
                    self.linear_dims,
                )?;
            }
            (MixerWeights::Full(w), LayerState::Full(full)) => {
                let encoder = encoders.at(Brick::Qsa)?;
                let (kv, indexer_state) = &mut **full;
                let indexer = w.indexer.as_ref().ok_or_else(|| {
                    InferError::Config(format!("couche {index} : indexeur QSA absent"))
                })?;
                encode_qsa(
                    self.arena,
                    self.kernels,
                    self.executor,
                    encoder,
                    kv,
                    w,
                    self.qsa_scratch,
                    self.qsa_dims,
                    Some(QsaSparse {
                        weights: indexer,
                        state: indexer_state,
                        scratch: self.indexer_scratch,
                        gathered,
                        dims: self.indexer_dims,
                        #[cfg(test)]
                        forced: forced_selection.and_then(Option::take),
                    }),
                    position,
                    self.x,
                    self.branch,
                )?;
                #[cfg(test)]
                if let Some(capture) = &capture {
                    let complete = (position + 1) / self.indexer_dims.compress;
                    if complete > self.indexer_dims.keep {
                        let keep = self.indexer_dims.keep;
                        encoder.set_compute_pipeline_state(&self.kernels.copy_u32);
                        encoder.set_buffer(0, Some(&self.indexer_scratch.selected), 0);
                        encoder.set_buffer(1, Some(capture.selection), 0);
                        super::set_value(encoder, 2, &[keep as u32, (capture.slot * keep) as u32]);
                        super::dispatch_1d(encoder, keep);
                        self.executor.encode_copy_with_offsets(
                            encoder,
                            &self.indexer_scratch.scores,
                            0,
                            capture.scores,
                            (capture.slot * capture.stride * std::mem::size_of::<f32>()) as u64,
                            complete,
                        )?;
                        crate::metal_backend::post_dispatch_barrier(encoder);
                    }
                }
            }
            _ => {
                return Err(InferError::Config(format!(
                    "couche {index} : poids et état de types différents"
                )))
            }
        }
        let encoder = encoders.at(Brick::Hc)?;
        crate::metal_backend::post_dispatch_barrier(encoder);
        encode_hc_combine(
            self.kernels,
            encoder,
            self.hc_scratch,
            self.hc_dims,
            &h[*cur],
            self.branch,
            &h[1 - *cur],
        );
        *cur = 1 - *cur;
        encode_hc_mix(
            self.kernels,
            self.executor,
            encoder,
            &layer.mlp_hc,
            self.hc_scratch,
            self.hc_dims,
            &h[*cur],
            self.x,
        )?;
        #[cfg(test)]
        if let Some(experts) = forced_routing.and_then(Option::take) {
            crate::metal_backend::force_next_topk(experts);
        }
        let encoder = encoders.at(Brick::Moe)?;
        self.executor.encode_moe_shared_buffers(
            encoder,
            owned,
            self.x,
            None,
            self.branch,
            self.hidden,
            &layer.moe,
            self.top_k,
        )?;
        #[cfg(test)]
        if crate::metal_backend::forced_topk_pending() {
            return Err(InferError::Metal(format!(
                "couche {index} : routage imposé non consommé par le MoE"
            )));
        }
        #[cfg(test)]
        if let Some(capture) = &capture {
            let indices = crate::metal_backend::take_last_topk_indices().ok_or_else(|| {
                InferError::Metal(format!("couche {index} : routage GPU non capturé"))
            })?;
            encoder.set_compute_pipeline_state(&self.kernels.copy_u32);
            encoder.set_buffer(0, Some(&indices), 0);
            encoder.set_buffer(1, Some(capture.routing), 0);
            super::set_value(
                encoder,
                2,
                &[self.top_k as u32, (capture.slot * self.top_k) as u32],
            );
            super::dispatch_1d(encoder, self.top_k);
            crate::metal_backend::post_dispatch_barrier(encoder);
        }
        let encoder = encoders.at(Brick::Hc)?;
        crate::metal_backend::post_dispatch_barrier(encoder);
        encode_hc_combine(
            self.kernels,
            encoder,
            self.hc_scratch,
            self.hc_dims,
            &h[*cur],
            self.branch,
            &h[1 - *cur],
        );
        *cur = 1 - *cur;
        Ok(())
    }
}
