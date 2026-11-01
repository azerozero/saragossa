//! Décision MTP profondeur 1 (D1) encodable dans un command buffer existant.
//!
//! Après le verify cible sur `[primary, draft]`, le GPU décide seul du token
//! émis, de l'acceptation du bonus et de l'état du cycle suivant, sans aucun
//! readback CPU. Ce module ne branche aucune boucle : il fournit le kernel,
//! son encodage validé et l'oracle CPU qui fixe le contrat.
//!
//! Contrat d'un cycle (`ctrl` est un tableau `u32` de [`MTP_CTRL_LEN`]) :
//!
//! 1. `stopped != 0` : rien ne change (`keep_bonus = 0`, sélection gelée).
//! 2. `out_count >= min(budget, capacité)` : `stopped = STOP_BUDGET`, rien
//!    n'est émis, l'état est gelé.
//! 3. `primary` est un stop id : `stopped = STOP_EOS`, rien n'est émis.
//! 4. Sinon `primary` est émis. `keep = draft == target0 && target0 n'est pas
//!    un stop id && il reste du budget`. Si `keep`, `target0` est émis aussi.
//!    `next = keep ? target1 : target0`, `next_hidden = keep ? row1 : row0`,
//!    `position += 1 + keep`. Les stats suivent `generate_greedy_mtp_inner` :
//!    le verify n'est compté que s'il reste du budget après `primary`
//!    (`verified = out_count < budget`), alors `accepted += draft == target0`,
//!    `rejected += draft != target0` (indépendant de l'émission EOS) et
//!    `verifications += 1`, plus 1 si le bonus est gardé avec du budget restant.
//!    Si `target0` est un stop id, `stopped = STOP_EOS` sans l'émettre ; si le
//!    budget est atteint, `stopped = STOP_BUDGET`. Un bonus `target1` stop id
//!    devient le `primary` du cycle suivant, qui s'arrête sans l'émettre.
//!
//! `budget` est le reste réel de `max_new_tokens` ; l'émission est en plus
//! bornée par la capacité de sortie (fin de bloc, pas fin de génération).
//!
//! Invariant : `position - position_initiale == out_count <= min(budget,
//! capacité)`, ce que [`mtp_d1_control_init`] vérifie en `u32` côté hôte.
//!
//! [`MtpD1Kernels::encode_select_copy`] recopie un buffer seulement si
//! `ctrl[MTP_CTRL_HIDDEN_SELECT]` vaut une valeur donnée : c'est le rollback
//! GPU de l'état linear-attn (`0` → capture après la ligne 0, `NONE` → capture
//! pré-cycle, `1` → état courant conservé), sans readback ni command buffer
//! supplémentaire.

use super::*;
use metal::CompileOptions;

/// Position de la prochaine écriture KV (incrémentée de `1 + keep_bonus`).
pub(crate) const MTP_CTRL_POSITION: usize = 0;
/// Token décidé mais pas encore émis, entrée du verify du cycle.
pub(crate) const MTP_CTRL_PRIMARY: usize = 1;
/// Nombre de tokens déjà écrits dans `output_tokens`.
pub(crate) const MTP_CTRL_OUT_COUNT: usize = 2;
/// Raison d'arrêt : [`MTP_STOP_NONE`], [`MTP_STOP_EOS`] ou [`MTP_STOP_BUDGET`].
pub(crate) const MTP_CTRL_STOPPED: usize = 3;
/// 1 si le draft du dernier cycle a été accepté et le bonus émis.
pub(crate) const MTP_CTRL_KEEP_BONUS: usize = 4;
/// Cumul des drafts acceptés.
pub(crate) const MTP_CTRL_ACCEPTED: usize = 5;
/// Cumul des drafts différents de `target0`.
pub(crate) const MTP_CTRL_REJECTED: usize = 6;
/// Nombre total maximal de tokens émis (borné par la capacité de sortie).
pub(crate) const MTP_CTRL_BUDGET: usize = 7;
/// Ligne hidden à recopier : 0, 1 ou [`MTP_HIDDEN_SELECT_NONE`] (gel).
pub(crate) const MTP_CTRL_HIDDEN_SELECT: usize = 8;
/// Cumul des verifies comptés comme dans `generate_greedy_mtp_inner`.
pub(crate) const MTP_CTRL_VERIFICATIONS: usize = 9;
/// Taille du buffer de contrôle en `u32`.
pub(crate) const MTP_CTRL_LEN: usize = 10;

pub(crate) const MTP_STOP_NONE: u32 = 0;
pub(crate) const MTP_STOP_EOS: u32 = 1;
pub(crate) const MTP_STOP_BUDGET: u32 = 2;
pub(crate) const MTP_HIDDEN_SELECT_NONE: u32 = 2;

const DECIDE_KERNEL: &str = "mtp_d1_decide";
const SELECT_KERNEL: &str = "mtp_d1_select_hidden";
const SELECT_COPY_U32_KERNEL: &str = "mtp_select_copy_u32";
const SELECT_COPY_U16_KERNEL: &str = "mtp_select_copy_u16";

const KERNEL_BODY: &str = r#"
static bool mtp_is_stop(uint token, device const uint* stops, uint n_stop) {
    for (uint i = 0; i < n_stop; ++i) {
        if (stops[i] == token) {
            return true;
        }
    }
    return false;
}

kernel void mtp_d1_decide(
    device uint* ctrl [[buffer(0)]],
    device const uint* draft [[buffer(1)]],
    device const uint* targets [[buffer(2)]],
    device const uint* stops [[buffer(3)]],
    device uint* next_index [[buffer(4)]],
    device uint* output [[buffer(5)]],
    constant uint& n_stop [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) {
        return;
    }
    ctrl[MTP_CTRL_KEEP_BONUS] = 0u;
    ctrl[MTP_CTRL_HIDDEN_SELECT] = MTP_HIDDEN_SELECT_NONE;
    if (ctrl[MTP_CTRL_STOPPED] != MTP_STOP_NONE) {
        return;
    }
    const uint limit = min(ctrl[MTP_CTRL_BUDGET], capacity);
    uint count = ctrl[MTP_CTRL_OUT_COUNT];
    if (count >= limit) {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_BUDGET;
        return;
    }
    const uint primary = ctrl[MTP_CTRL_PRIMARY];
    if (mtp_is_stop(primary, stops, n_stop)) {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_EOS;
        return;
    }
    const uint d = draft[0];
    const uint t0 = targets[0];
    const uint t1 = targets[1];
    output[count] = primary;
    count += 1u;
    const bool verified = count < ctrl[MTP_CTRL_BUDGET];
    const bool t0_stop = mtp_is_stop(t0, stops, n_stop);
    const bool keep = d == t0 && !t0_stop && count < limit;
    if (keep) {
        output[count] = t0;
        count += 1u;
    }
    const uint next = keep ? t1 : t0;
    ctrl[MTP_CTRL_POSITION] += keep ? 2u : 1u;
    ctrl[MTP_CTRL_OUT_COUNT] = count;
    ctrl[MTP_CTRL_PRIMARY] = next;
    ctrl[MTP_CTRL_KEEP_BONUS] = keep ? 1u : 0u;
    ctrl[MTP_CTRL_ACCEPTED] += (verified && d == t0) ? 1u : 0u;
    ctrl[MTP_CTRL_REJECTED] += (verified && d != t0) ? 1u : 0u;
    ctrl[MTP_CTRL_VERIFICATIONS] +=
        (verified ? 1u : 0u) + ((keep && count < ctrl[MTP_CTRL_BUDGET]) ? 1u : 0u);
    ctrl[MTP_CTRL_HIDDEN_SELECT] = keep ? 1u : 0u;
    next_index[0] = next;
    if (t0_stop) {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_EOS;
    } else if (count >= limit) {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_BUDGET;
    }
}

kernel void mtp_d1_select_hidden(
    device const uint* ctrl [[buffer(0)]],
    device const float* rows [[buffer(1)]],
    device float* next_hidden [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dim) {
        return;
    }
    const uint row = ctrl[MTP_CTRL_HIDDEN_SELECT];
    if (row > 1u) {
        return;
    }
    next_hidden[gid] = rows[row * dim + gid];
}

kernel void mtp_select_copy_u32(
    device const uint* ctrl [[buffer(0)]],
    device const uint* src [[buffer(1)]],
    device uint* dst [[buffer(2)]],
    constant uint2& cfg [[buffer(3)]], // x = mots, y = sélection attendue
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.x || ctrl[MTP_CTRL_HIDDEN_SELECT] != cfg.y) {
        return;
    }
    dst[gid] = src[gid];
}

kernel void mtp_select_copy_u16(
    device const uint* ctrl [[buffer(0)]],
    device const ushort* src [[buffer(1)]],
    device ushort* dst [[buffer(2)]],
    constant uint2& cfg [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cfg.x || ctrl[MTP_CTRL_HIDDEN_SELECT] != cfg.y) {
        return;
    }
    dst[gid] = src[gid];
}
"#;

/// Source MSL : les constantes de layout viennent de Rust (source unique).
fn kernel_source() -> String {
    format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         #define MTP_CTRL_POSITION {MTP_CTRL_POSITION}u\n\
         #define MTP_CTRL_PRIMARY {MTP_CTRL_PRIMARY}u\n\
         #define MTP_CTRL_OUT_COUNT {MTP_CTRL_OUT_COUNT}u\n\
         #define MTP_CTRL_STOPPED {MTP_CTRL_STOPPED}u\n\
         #define MTP_CTRL_KEEP_BONUS {MTP_CTRL_KEEP_BONUS}u\n\
         #define MTP_CTRL_ACCEPTED {MTP_CTRL_ACCEPTED}u\n\
         #define MTP_CTRL_REJECTED {MTP_CTRL_REJECTED}u\n\
         #define MTP_CTRL_BUDGET {MTP_CTRL_BUDGET}u\n\
         #define MTP_CTRL_HIDDEN_SELECT {MTP_CTRL_HIDDEN_SELECT}u\n\
         #define MTP_CTRL_VERIFICATIONS {MTP_CTRL_VERIFICATIONS}u\n\
         #define MTP_STOP_NONE {MTP_STOP_NONE}u\n\
         #define MTP_STOP_EOS {MTP_STOP_EOS}u\n\
         #define MTP_STOP_BUDGET {MTP_STOP_BUDGET}u\n\
         #define MTP_HIDDEN_SELECT_NONE {MTP_HIDDEN_SELECT_NONE}u\n\
         {KERNEL_BODY}"
    )
}

/// Construit l'état de contrôle initial et vérifie qu'aucun compteur `u32` ne
/// peut déborder pendant le bloc. `budget` (reste de `max_new_tokens`) est
/// saturé à `u32::MAX` : l'émission reste bornée par `output_capacity`.
pub(crate) fn mtp_d1_control_init(
    position: usize,
    primary: u32,
    budget: usize,
    output_capacity: usize,
) -> Result<[u32; MTP_CTRL_LEN]> {
    let end = position
        .checked_add(budget.min(output_capacity))
        .ok_or_else(|| InferError::Config("MTP D1 position+budget déborde".to_string()))?;
    checked_u32(end, "MTP D1 position finale")?;
    checked_u32(output_capacity, "MTP D1 capacité de sortie")?;
    let mut ctrl = [0_u32; MTP_CTRL_LEN];
    ctrl[MTP_CTRL_POSITION] = checked_u32(position, "MTP D1 position")?;
    ctrl[MTP_CTRL_PRIMARY] = primary;
    ctrl[MTP_CTRL_BUDGET] = u32::try_from(budget).unwrap_or(u32::MAX);
    ctrl[MTP_CTRL_HIDDEN_SELECT] = MTP_HIDDEN_SELECT_NONE;
    Ok(ctrl)
}

/// Oracle CPU du kernel `mtp_d1_decide` (même layout, même arithmétique).
#[cfg(test)]
pub(crate) fn mtp_d1_decide_cpu(
    ctrl: &mut [u32; MTP_CTRL_LEN],
    draft: u32,
    targets: [u32; 2],
    stop_ids: &[u32],
    next_index: &mut u32,
    output: &mut [u32],
) {
    ctrl[MTP_CTRL_KEEP_BONUS] = 0;
    ctrl[MTP_CTRL_HIDDEN_SELECT] = MTP_HIDDEN_SELECT_NONE;
    if ctrl[MTP_CTRL_STOPPED] != MTP_STOP_NONE {
        return;
    }
    let capacity = u32::try_from(output.len()).unwrap_or(u32::MAX);
    let limit = ctrl[MTP_CTRL_BUDGET].min(capacity);
    let mut count = ctrl[MTP_CTRL_OUT_COUNT];
    if count >= limit {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_BUDGET;
        return;
    }
    let primary = ctrl[MTP_CTRL_PRIMARY];
    if stop_ids.contains(&primary) {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_EOS;
        return;
    }
    let [t0, t1] = targets;
    let mut emit = |token: u32, count: &mut u32| {
        if let Some(slot) = output.get_mut(*count as usize) {
            *slot = token;
        }
        *count += 1;
    };
    emit(primary, &mut count);
    let verified = count < ctrl[MTP_CTRL_BUDGET];
    let t0_stop = stop_ids.contains(&t0);
    let keep = draft == t0 && !t0_stop && count < limit;
    if keep {
        emit(t0, &mut count);
    }
    let next = if keep { t1 } else { t0 };
    ctrl[MTP_CTRL_POSITION] += 1 + u32::from(keep);
    ctrl[MTP_CTRL_OUT_COUNT] = count;
    ctrl[MTP_CTRL_PRIMARY] = next;
    ctrl[MTP_CTRL_KEEP_BONUS] = u32::from(keep);
    ctrl[MTP_CTRL_ACCEPTED] += u32::from(verified && draft == t0);
    ctrl[MTP_CTRL_REJECTED] += u32::from(verified && draft != t0);
    ctrl[MTP_CTRL_VERIFICATIONS] +=
        u32::from(verified) + u32::from(keep && count < ctrl[MTP_CTRL_BUDGET]);
    ctrl[MTP_CTRL_HIDDEN_SELECT] = u32::from(keep);
    *next_index = next;
    if t0_stop {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_EOS;
    } else if count >= limit {
        ctrl[MTP_CTRL_STOPPED] = MTP_STOP_BUDGET;
    }
}

/// Oracle CPU du kernel `mtp_d1_select_hidden` (`rows` = deux lignes contiguës).
#[cfg(test)]
pub(crate) fn mtp_d1_select_hidden_cpu(
    ctrl: &[u32; MTP_CTRL_LEN],
    rows: &[f32],
    next_hidden: &mut [f32],
) {
    let dim = next_hidden.len();
    let select = ctrl[MTP_CTRL_HIDDEN_SELECT] as usize;
    if select > 1 {
        return;
    }
    if let Some(row) = rows.get(select * dim..(select + 1) * dim) {
        next_hidden.copy_from_slice(row);
    }
}

/// Oracle CPU de [`MtpD1Kernels::encode_select_copy`].
#[cfg(test)]
pub(crate) fn mtp_select_copy_cpu<T: Copy>(
    ctrl: &[u32; MTP_CTRL_LEN],
    select: u32,
    src: &[T],
    dst: &mut [T],
) {
    if ctrl[MTP_CTRL_HIDDEN_SELECT] == select && src.len() == dst.len() {
        dst.copy_from_slice(src);
    }
}

/// Buffers d'un cycle D1. Les offsets sont en octets.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpD1Buffers<'a> {
    /// `u32[MTP_CTRL_LEN]`, lu et écrit par le GPU.
    pub ctrl: &'a BufferRef,
    /// `u32[1]` : draft MTP proposé après `primary`.
    pub draft_index: &'a BufferRef,
    /// `u32[2]` : argmax cible après `primary` puis après `draft`.
    pub target_indices: &'a BufferRef,
    pub target_indices_offset: u64,
    /// `f32[2 * hidden_dim]` : hidden cible des lignes 0 et 1.
    pub target_hidden: &'a BufferRef,
    pub target_hidden_offset: u64,
    /// `f32[hidden_dim]` : hidden d'entrée du draft suivant.
    pub next_hidden: &'a BufferRef,
    /// `u32[1]` : token d'entrée du cycle suivant.
    pub next_index: &'a BufferRef,
    /// `u32[stop_count]`.
    pub stop_ids: &'a BufferRef,
    /// `u32[output_capacity]`.
    pub output_tokens: &'a BufferRef,
}

/// Dimensions validées avant encodage.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpD1Dims {
    pub hidden_dim: usize,
    pub stop_count: usize,
    pub output_capacity: usize,
}

/// Pipelines de la décision D1, compilés une fois par device.
#[derive(Debug)]
pub(crate) struct MtpD1Kernels {
    decide: ComputePipelineState,
    select_hidden: ComputePipelineState,
    select_copy_u32: ComputePipelineState,
    select_copy_u16: ComputePipelineState,
}

fn ensure_room(buffer: &BufferRef, offset: u64, bytes: usize, label: &'static str) -> Result<()> {
    if offset % 4 != 0 {
        return Err(InferError::Metal(format!(
            "MTP D1 {label}: offset {offset} non aligné sur 4 octets"
        )));
    }
    let bytes = u64::try_from(bytes)
        .map_err(|_| InferError::Metal(format!("MTP D1 {label}: taille hors plage")))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| InferError::Metal(format!("MTP D1 {label}: fin déborde")))?;
    if end > buffer.length() {
        return Err(InferError::Metal(format!(
            "MTP D1 {label}: {end} octets requis, buffer de {}",
            buffer.length()
        )));
    }
    Ok(())
}

impl MtpD1Kernels {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let library = device
            .new_library_with_source(&kernel_source(), &CompileOptions::new())
            .map_err(|message| InferError::Metal(format!("library MTP D1: {message}")))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|message| InferError::Metal(format!("kernel {name}: {message}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| InferError::Metal(format!("pipeline {name}: {message}")))
        };
        Ok(Self {
            decide: pipeline(DECIDE_KERNEL)?,
            select_hidden: pipeline(SELECT_KERNEL)?,
            select_copy_u32: pipeline(SELECT_COPY_U32_KERNEL)?,
            select_copy_u16: pipeline(SELECT_COPY_U16_KERNEL)?,
        })
    }

    /// Encode décision puis sélection hidden dans `encoder`, sans commit ni
    /// lecture CPU. Deux dispatchs : la décision (1 thread) écrit
    /// `MTP_CTRL_HIDDEN_SELECT`, la copie parallèle le lit après la barrière,
    /// ce qui évite toute course inter-threadgroups.
    pub(crate) fn encode_mtp_d1_decision(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &MtpD1Buffers<'_>,
        dims: MtpD1Dims,
    ) -> Result<()> {
        let MtpD1Dims {
            hidden_dim,
            stop_count,
            output_capacity,
        } = dims;
        if hidden_dim == 0 || output_capacity == 0 {
            return Err(InferError::Metal(format!(
                "MTP D1 hidden_dim={hidden_dim} output_capacity={output_capacity} invalides"
            )));
        }
        let u32_bytes = |len: usize| byte_len_usize::<u32>(len);
        let rows = checked_len(hidden_dim, 2, "MTP D1 lignes hidden")?;
        ensure_room(buffers.ctrl, 0, u32_bytes(MTP_CTRL_LEN)?, "ctrl")?;
        ensure_room(buffers.draft_index, 0, u32_bytes(1)?, "draft_index")?;
        ensure_room(
            buffers.target_indices,
            buffers.target_indices_offset,
            u32_bytes(2)?,
            "target_indices",
        )?;
        ensure_room(
            buffers.target_hidden,
            buffers.target_hidden_offset,
            byte_len_usize::<f32>(rows)?,
            "target_hidden",
        )?;
        ensure_room(
            buffers.next_hidden,
            0,
            byte_len_usize::<f32>(hidden_dim)?,
            "next_hidden",
        )?;
        ensure_room(buffers.next_index, 0, u32_bytes(1)?, "next_index")?;
        ensure_room(buffers.stop_ids, 0, u32_bytes(stop_count)?, "stop_ids")?;
        ensure_room(
            buffers.output_tokens,
            0,
            u32_bytes(output_capacity)?,
            "output_tokens",
        )?;
        let n_stop = checked_u32(stop_count, "MTP D1 stop_count")?;
        let capacity = checked_u32(output_capacity, "MTP D1 output_capacity")?;
        let dim = checked_u32(hidden_dim, "MTP D1 hidden_dim")?;

        encoder.set_compute_pipeline_state(&self.decide);
        encoder.set_buffer(0, Some(buffers.ctrl), 0);
        encoder.set_buffer(1, Some(buffers.draft_index), 0);
        encoder.set_buffer(
            2,
            Some(buffers.target_indices),
            buffers.target_indices_offset,
        );
        encoder.set_buffer(3, Some(buffers.stop_ids), 0);
        encoder.set_buffer(4, Some(buffers.next_index), 0);
        encoder.set_buffer(5, Some(buffers.output_tokens), 0);
        set_u32_bytes(encoder, 6, &[n_stop], "mtp_d1_n_stop")?;
        set_u32_bytes(encoder, 7, &[capacity], "mtp_d1_capacity")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        post_dispatch_barrier(encoder);

        encoder.set_compute_pipeline_state(&self.select_hidden);
        encoder.set_buffer(0, Some(buffers.ctrl), 0);
        encoder.set_buffer(1, Some(buffers.target_hidden), buffers.target_hidden_offset);
        encoder.set_buffer(2, Some(buffers.next_hidden), 0);
        set_u32_bytes(encoder, 3, &[dim], "mtp_d1_hidden_dim")?;
        let width = self.select_hidden.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(hidden_dim, "MTP D1 threads")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Copie `bytes` octets de `src` vers `dst` seulement si
    /// `ctrl[MTP_CTRL_HIDDEN_SELECT] == select` (lu sur GPU, à encoder après la
    /// décision). Mots `u32` si `bytes % 4 == 0`, sinon `u16`.
    pub(crate) fn encode_select_copy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ctrl: &BufferRef,
        select: u32,
        src: &BufferRef,
        dst: &BufferRef,
        bytes: usize,
    ) -> Result<()> {
        if select > MTP_HIDDEN_SELECT_NONE {
            return Err(InferError::Metal(format!(
                "MTP select copy: sélection {select} invalide"
            )));
        }
        if bytes == 0 || bytes % 2 != 0 {
            return Err(InferError::Metal(format!(
                "MTP select copy: {bytes} octets non multiples de 2"
            )));
        }
        ensure_room(ctrl, 0, byte_len_usize::<u32>(MTP_CTRL_LEN)?, "ctrl")?;
        ensure_room(src, 0, bytes, "select copy src")?;
        ensure_room(dst, 0, bytes, "select copy dst")?;
        let (pipeline, units) = if bytes % 4 == 0 {
            (&self.select_copy_u32, bytes / 4)
        } else {
            (&self.select_copy_u16, bytes / 2)
        };
        let cfg = [checked_u32(units, "MTP select copy unités")?, select];
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(ctrl), 0);
        encoder.set_buffer(1, Some(src), 0);
        encoder.set_buffer(2, Some(dst), 0);
        set_u32_bytes(encoder, 3, &cfg, "mtp_select_copy_cfg")?;
        let width = pipeline.thread_execution_width().max(1);
        profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(checked_nsuint(units, "MTP select copy threads")?, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Rollback GPU d'un état linear-attn après la décision D1 :
    /// sélection `0` → `row0` (capture après la ligne 0), `NONE` → `pre`
    /// (capture pré-cycle), `1` → état courant (ligne 1) conservé.
    pub(crate) fn encode_linear_state_rollback(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ctrl: &BufferRef,
        state: &LinearAttentionMetalState,
        row0: &LinearAttentionMetalState,
        pre: &LinearAttentionMetalState,
    ) -> Result<()> {
        for capture in [row0, pre] {
            // Une capture en journal ne contient pas l'état de sa ligne : seule
            // `restore_linear_attn_captures` sait la relire.
            if capture.ssm_journal {
                return Err(InferError::Config(
                    "rollback MTP linear-attn: capture en mode journal".to_string(),
                ));
            }
            if capture.conv_len != state.conv_len
                || capture.ssm_len != state.ssm_len
                || capture.ssm_bf16 != state.ssm_bf16
            {
                return Err(InferError::Shape(format!(
                    "rollback MTP linear-attn: capture conv={}/ssm={}/bf16={} ≠ état conv={}/ssm={}/bf16={}",
                    capture.conv_len,
                    capture.ssm_len,
                    capture.ssm_bf16,
                    state.conv_len,
                    state.ssm_len,
                    state.ssm_bf16
                )));
            }
        }
        let conv_bytes = byte_len_usize::<f32>(state.conv_len)?;
        let ssm_bytes = if state.ssm_bf16 {
            byte_len_usize::<u16>(state.ssm_len)?
        } else {
            byte_len_usize::<f32>(state.ssm_len)?
        };
        for (select, capture) in [(0, row0), (MTP_HIDDEN_SELECT_NONE, pre)] {
            self.encode_select_copy(
                encoder,
                ctrl,
                select,
                &capture.conv,
                &state.conv,
                conv_bytes,
            )?;
            self.encode_select_copy(encoder, ctrl, select, &capture.ssm, &state.ssm, ssm_bytes)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STOP: u32 = 99;

    /// Un cycle CPU complet ; renvoie (ctrl, sortie émise, next_index).
    fn run(
        primary: u32,
        budget: usize,
        cycles: &[(u32, [u32; 2])],
    ) -> Result<([u32; MTP_CTRL_LEN], Vec<u32>, u32)> {
        let mut ctrl = mtp_d1_control_init(10, primary, budget, 8)?;
        let mut out = vec![0; 8];
        let mut next = u32::MAX;
        for &(draft, targets) in cycles {
            mtp_d1_decide_cpu(&mut ctrl, draft, targets, &[STOP], &mut next, &mut out);
        }
        out.truncate(ctrl[MTP_CTRL_OUT_COUNT] as usize);
        Ok((ctrl, out, next))
    }

    #[test]
    fn accepted_emits_primary_and_bonus() -> Result<()> {
        let (ctrl, out, next) = run(1, 8, &[(2, [2, 3])])?;
        assert_eq!(out, [1, 2]);
        assert_eq!(next, 3);
        assert_eq!(ctrl[MTP_CTRL_PRIMARY], 3);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 12);
        assert_eq!(ctrl[MTP_CTRL_KEEP_BONUS], 1);
        assert_eq!(ctrl[MTP_CTRL_HIDDEN_SELECT], 1);
        assert_eq!((ctrl[MTP_CTRL_ACCEPTED], ctrl[MTP_CTRL_REJECTED]), (1, 0));
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_NONE);
        Ok(())
    }

    #[test]
    fn rejected_emits_primary_only() -> Result<()> {
        let (ctrl, out, next) = run(1, 8, &[(5, [2, 3])])?;
        assert_eq!(out, [1]);
        assert_eq!(next, 2);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 11);
        assert_eq!(ctrl[MTP_CTRL_HIDDEN_SELECT], 0);
        assert_eq!((ctrl[MTP_CTRL_ACCEPTED], ctrl[MTP_CTRL_REJECTED]), (0, 1));
        Ok(())
    }

    #[test]
    fn primary_eos_freezes_without_emit() -> Result<()> {
        let (ctrl, out, next) = run(STOP, 8, &[(2, [2, 3]), (4, [4, 4])])?;
        assert!(out.is_empty());
        assert_eq!(next, u32::MAX);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 10);
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_EOS);
        assert_eq!(ctrl[MTP_CTRL_HIDDEN_SELECT], MTP_HIDDEN_SELECT_NONE);
        Ok(())
    }

    #[test]
    fn target0_eos_stops_without_emitting_it() -> Result<()> {
        // Draft égal à l'EOS cible : accepté pour les stats, mais pas émis.
        let (ctrl, out, _) = run(1, 8, &[(STOP, [STOP, 3]), (3, [3, 3])])?;
        assert_eq!(out, [1]);
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_EOS);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 11);
        assert_eq!((ctrl[MTP_CTRL_ACCEPTED], ctrl[MTP_CTRL_REJECTED]), (1, 0));
        Ok(())
    }

    #[test]
    fn bonus_eos_stops_next_cycle_without_emitting() -> Result<()> {
        let (ctrl, out, next) = run(1, 8, &[(2, [2, STOP]), (7, [7, 7])])?;
        assert_eq!(out, [1, 2]);
        assert_eq!(next, STOP);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 12);
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_EOS);
        Ok(())
    }

    #[test]
    fn budget_one_blocks_bonus() -> Result<()> {
        let (ctrl, out, next) = run(1, 1, &[(2, [2, 3]), (3, [3, 4])])?;
        assert_eq!(out, [1]);
        assert_eq!(next, 2);
        assert_eq!(ctrl[MTP_CTRL_KEEP_BONUS], 0);
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_BUDGET);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 11);
        Ok(())
    }

    #[test]
    fn budget_two_takes_bonus_then_freezes() -> Result<()> {
        let (ctrl, out, next) = run(1, 2, &[(2, [2, 3]), (3, [3, 4])])?;
        assert_eq!(out, [1, 2]);
        assert_eq!(next, 3);
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_BUDGET);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 12);
        Ok(())
    }

    #[test]
    fn budget_zero_emits_nothing() -> Result<()> {
        let (ctrl, out, _) = run(1, 0, &[(2, [2, 3])])?;
        assert!(out.is_empty());
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_BUDGET);
        Ok(())
    }

    #[test]
    fn early_stop_then_frozen_cycles() -> Result<()> {
        let (ctrl, out, next) = run(1, 8, &[(5, [STOP, 3]), (2, [2, 3]), (2, [2, 3])])?;
        assert_eq!(out, [1]);
        assert_eq!(next, STOP);
        assert_eq!(ctrl[MTP_CTRL_POSITION], 11);
        assert_eq!(ctrl[MTP_CTRL_REJECTED], 1);
        Ok(())
    }

    #[test]
    fn init_rejects_overflow_and_saturates_budget() -> Result<()> {
        assert!(mtp_d1_control_init(u32::MAX as usize, 1, 1, 8).is_err());
        assert!(mtp_d1_control_init(u32::MAX as usize - 1, 1, 1, 8).is_ok());
        // Le budget au-delà de la capacité est permis : seule la capacité borne
        // la position atteignable dans le bloc.
        assert!(mtp_d1_control_init(u32::MAX as usize - 8, 1, 9, 8).is_ok());
        assert!(mtp_d1_control_init(u32::MAX as usize - 7, 1, 9, 8).is_err());
        let ctrl = mtp_d1_control_init(0, 1, usize::MAX, 4)?;
        assert_eq!(ctrl[MTP_CTRL_BUDGET], u32::MAX);
        Ok(())
    }

    #[test]
    fn stats_follow_generation_budget_rules() -> Result<()> {
        // Accepté avec reste : verify + bonus vérifié.
        let (ctrl, _, _) = run(1, 8, &[(2, [2, 3])])?;
        assert_eq!(ctrl[MTP_CTRL_VERIFICATIONS], 2);
        // Accepté, bonus émis mais budget atteint : pas de verify bonus.
        let (ctrl, out, _) = run(1, 2, &[(2, [2, 3])])?;
        assert_eq!(out, [1, 2]);
        assert_eq!(
            (ctrl[MTP_CTRL_ACCEPTED], ctrl[MTP_CTRL_VERIFICATIONS]),
            (1, 1)
        );
        // Budget 1 : `primary` seul, generation n'exécute aucun verify.
        let (ctrl, _, _) = run(1, 1, &[(2, [2, 3])])?;
        assert_eq!(ctrl[MTP_CTRL_ACCEPTED], 0);
        assert_eq!(ctrl[MTP_CTRL_REJECTED], 0);
        assert_eq!(ctrl[MTP_CTRL_VERIFICATIONS], 0);
        // Rejet : un verify.
        let (ctrl, _, _) = run(1, 8, &[(5, [2, 3])])?;
        assert_eq!(ctrl[MTP_CTRL_VERIFICATIONS], 1);
        // Budget au-delà de la capacité : la capacité borne l'émission, pas les stats.
        let mut ctrl = mtp_d1_control_init(0, 1, 100, 4)?;
        let mut out = vec![0; 4];
        let mut next = 0;
        for _ in 0..2 {
            let draft = ctrl[MTP_CTRL_PRIMARY] + 1;
            mtp_d1_decide_cpu(
                &mut ctrl,
                draft,
                [draft, draft + 1],
                &[STOP],
                &mut next,
                &mut out,
            );
        }
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(ctrl[MTP_CTRL_STOPPED], MTP_STOP_BUDGET);
        assert_eq!(ctrl[MTP_CTRL_VERIFICATIONS], 4);
        assert_eq!(ctrl[MTP_CTRL_PRIMARY], 5);
        Ok(())
    }

    #[test]
    fn select_copy_matches_selection() -> Result<()> {
        let mut ctrl = mtp_d1_control_init(0, 1, 8, 8)?;
        let mut dst = [0_u16; 3];
        mtp_select_copy_cpu(&ctrl, 0, &[1, 2, 3], &mut dst);
        assert_eq!(dst, [0, 0, 0]);
        mtp_select_copy_cpu(&ctrl, MTP_HIDDEN_SELECT_NONE, &[1, 2, 3], &mut dst);
        assert_eq!(dst, [1, 2, 3]);
        ctrl[MTP_CTRL_HIDDEN_SELECT] = 0;
        mtp_select_copy_cpu(&ctrl, 0, &[4, 5, 6], &mut dst);
        assert_eq!(dst, [4, 5, 6]);
        Ok(())
    }

    #[test]
    fn select_hidden_follows_decision() -> Result<()> {
        let rows = [1.0, 2.0, 3.0, 4.0];
        let mut hidden = [0.0; 2];
        let mut ctrl = mtp_d1_control_init(0, 1, 8, 8)?;
        mtp_d1_select_hidden_cpu(&ctrl, &rows, &mut hidden);
        assert_eq!(hidden, [0.0, 0.0]);
        ctrl[MTP_CTRL_HIDDEN_SELECT] = 1;
        mtp_d1_select_hidden_cpu(&ctrl, &rows, &mut hidden);
        assert_eq!(hidden, [3.0, 4.0]);
        ctrl[MTP_CTRL_HIDDEN_SELECT] = 0;
        mtp_d1_select_hidden_cpu(&ctrl, &rows, &mut hidden);
        assert_eq!(hidden, [1.0, 2.0]);
        Ok(())
    }

    /// Exhaustif sur un petit univers : invariants de sortie et de position.
    #[test]
    fn exhaustive_invariants() -> Result<()> {
        let tokens = [1_u32, 2, STOP];
        let mut steps = Vec::new();
        for d in tokens {
            for t0 in tokens {
                for t1 in tokens {
                    steps.push((d, [t0, t1]));
                }
            }
        }
        for primary in tokens {
            for budget in 0..=4 {
                for a in &steps {
                    for b in &steps {
                        let (ctrl, out, _) = run(primary, budget, &[*a, *b])?;
                        let count = ctrl[MTP_CTRL_OUT_COUNT];
                        assert!(!out.contains(&STOP), "EOS émis: {out:?}");
                        assert!(count as usize <= budget);
                        assert_eq!(ctrl[MTP_CTRL_POSITION] - 10, count);
                        assert!(ctrl[MTP_CTRL_ACCEPTED] + ctrl[MTP_CTRL_REJECTED] <= 2);
                        if ctrl[MTP_CTRL_STOPPED] == MTP_STOP_NONE {
                            assert!((count as usize) < budget);
                            assert!(!(count == 0 && primary == STOP));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    struct GpuCase {
        primary: u32,
        budget: usize,
        cycles: [(u32, [u32; 2]); 2],
    }

    const HIDDEN: usize = 5;
    const CAPACITY: usize = 4;

    fn rows_for(cycle: usize) -> Vec<f32> {
        (0..2 * HIDDEN)
            .map(|i| (cycle * 100 + i) as f32 + 0.5)
            .collect()
    }

    fn shared_u32(device: &Device, data: &[u32]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            (std::mem::size_of_val(data)) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn shared_f32(device: &Device, data: &[f32]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            (std::mem::size_of_val(data)) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    }

    const CONV_LEN: usize = 5;
    const SSM_LEN: usize = 3;

    fn shared_u16(device: &Device, data: &[u16]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            (std::mem::size_of_val(data)) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// État linear-attn factice : conv f32 (chemin u32), SSM bf16 de longueur
    /// impaire (chemin u16).
    fn fake_linear_state(device: &Device, seed: u16) -> LinearAttentionMetalState {
        let conv = (0..CONV_LEN as u32)
            .map(|i| u32::from(seed) * 1000 + i)
            .collect::<Vec<_>>();
        let ssm = (0..SSM_LEN as u16)
            .map(|i| seed * 100 + i)
            .collect::<Vec<_>>();
        LinearAttentionMetalState {
            conv: shared_u32(device, &conv),
            ssm: shared_u16(device, &ssm),
            conv_len: CONV_LEN,
            ssm_len: SSM_LEN,
            conv_dim: CONV_LEN,
            conv_kernel_dim: 1,
            num_value_heads: 1,
            value_head_dim: SSM_LEN,
            key_head_dim: 1,
            ssm_bf16: true,
            ssm_journal: false,
        }
    }

    fn read_linear_state(state: &LinearAttentionMetalState) -> Result<(Vec<u32>, Vec<u16>)> {
        Ok((
            read_u32_buffer(&state.conv, CONV_LEN)?,
            read_u16_buffer(&state.ssm, SSM_LEN)?,
        ))
    }

    fn cpu_copy_state(
        ctrl: &[u32; MTP_CTRL_LEN],
        select: u32,
        src: &(Vec<u32>, Vec<u16>),
        dst: &mut (Vec<u32>, Vec<u16>),
    ) {
        mtp_select_copy_cpu(ctrl, select, &src.0, &mut dst.0);
        mtp_select_copy_cpu(ctrl, select, &src.1, &mut dst.1);
    }

    fn gpu_copy_state(
        kernels: &MtpD1Kernels,
        encoder: &ComputeCommandEncoderRef,
        ctrl: &BufferRef,
        select: u32,
        src: &LinearAttentionMetalState,
        dst: &LinearAttentionMetalState,
    ) -> Result<()> {
        kernels.encode_select_copy(encoder, ctrl, select, &src.conv, &dst.conv, CONV_LEN * 4)?;
        kernels.encode_select_copy(encoder, ctrl, select, &src.ssm, &dst.ssm, SSM_LEN * 2)
    }

    /// Deux décisions dans un seul command buffer, un seul commit, lecture
    /// unique après complétion, comparées bit à bit à l'oracle CPU. Chaque
    /// cycle simule aussi le verify linear-attn (capture pré-cycle, capture
    /// ligne 0, état ligne 1) puis le rollback GPU conditionnel.
    fn gpu_case(
        device: &Device,
        queue: &CommandQueue,
        kernels: &MtpD1Kernels,
        case: &GpuCase,
    ) -> Result<()> {
        let stops = [7_u32, STOP];
        let init = mtp_d1_control_init(10, case.primary, case.budget, CAPACITY)?;
        let ctrl = shared_u32(device, &init);
        let output = shared_u32(device, &[u32::MAX; CAPACITY]);
        let next_index = shared_u32(device, &[u32::MAX]);
        let next_hidden = shared_f32(device, &[-1.0; HIDDEN]);
        let stop_ids = shared_u32(device, &stops);
        // `always` : contrôle figé à la sélection 0 → copie inconditionnelle.
        let mut always_ctrl = init;
        always_ctrl[MTP_CTRL_HIDDEN_SELECT] = 0;
        let always = shared_u32(device, &always_ctrl);
        let state = fake_linear_state(device, 1);
        let row0 = fake_linear_state(device, 2);
        let pre = fake_linear_state(device, 3);
        let verify_row0 = [fake_linear_state(device, 10), fake_linear_state(device, 20)];
        let verify_row1 = [fake_linear_state(device, 11), fake_linear_state(device, 21)];
        let mut cpu_state = read_linear_state(&state)?;
        let mut cpu_row0 = read_linear_state(&row0)?;
        let mut cpu_pre = read_linear_state(&pre)?;
        let mut per_cycle = Vec::new();
        for (i, &(draft, [t0, t1])) in case.cycles.iter().enumerate() {
            let mut hidden = vec![0.0_f32; 1];
            hidden.extend(rows_for(i));
            per_cycle.push((
                shared_u32(device, &[draft]),
                shared_u32(device, &[0, 0, t0, t1]),
                shared_f32(device, &hidden),
            ));
        }
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for (cycle, (draft, targets, hidden)) in per_cycle.iter().enumerate() {
            gpu_copy_state(kernels, encoder, &always, 0, &state, &pre)?;
            gpu_copy_state(kernels, encoder, &always, 0, &verify_row0[cycle], &row0)?;
            gpu_copy_state(kernels, encoder, &always, 0, &verify_row1[cycle], &state)?;
            let buffers = MtpD1Buffers {
                ctrl: &ctrl,
                draft_index: draft,
                target_indices: targets,
                target_indices_offset: 8,
                target_hidden: hidden,
                target_hidden_offset: 4,
                next_hidden: &next_hidden,
                next_index: &next_index,
                stop_ids: &stop_ids,
                output_tokens: &output,
            };
            let dims = MtpD1Dims {
                hidden_dim: HIDDEN,
                stop_count: stops.len(),
                output_capacity: CAPACITY,
            };
            kernels.encode_mtp_d1_decision(encoder, &buffers, dims)?;
            kernels.encode_linear_state_rollback(encoder, &ctrl, &state, &row0, &pre)?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        wait_for_completion(command_buffer)?;

        let mut cpu_ctrl = init;
        let mut cpu_out = vec![u32::MAX; CAPACITY];
        let mut cpu_next = u32::MAX;
        let mut cpu_hidden = vec![-1.0_f32; HIDDEN];
        let cpu_always = always_ctrl;
        for (i, &(draft, targets)) in case.cycles.iter().enumerate() {
            cpu_copy_state(&cpu_always, 0, &cpu_state.clone(), &mut cpu_pre);
            cpu_copy_state(
                &cpu_always,
                0,
                &read_linear_state(&verify_row0[i])?,
                &mut cpu_row0,
            );
            cpu_copy_state(
                &cpu_always,
                0,
                &read_linear_state(&verify_row1[i])?,
                &mut cpu_state,
            );
            mtp_d1_decide_cpu(
                &mut cpu_ctrl,
                draft,
                targets,
                &stops,
                &mut cpu_next,
                &mut cpu_out,
            );
            mtp_d1_select_hidden_cpu(&cpu_ctrl, &rows_for(i), &mut cpu_hidden);
            cpu_copy_state(&cpu_ctrl, 0, &cpu_row0.clone(), &mut cpu_state);
            cpu_copy_state(
                &cpu_ctrl,
                MTP_HIDDEN_SELECT_NONE,
                &cpu_pre.clone(),
                &mut cpu_state,
            );
        }
        assert_eq!(read_u32_buffer(&ctrl, MTP_CTRL_LEN)?, cpu_ctrl);
        assert_eq!(read_u32_buffer(&output, CAPACITY)?, cpu_out);
        assert_eq!(read_u32_buffer(&next_index, 1)?, [cpu_next]);
        assert_eq!(
            read_linear_state(&state)?,
            cpu_state,
            "rollback linear-attn"
        );
        let gpu_hidden = read_f32_buffer(&next_hidden, HIDDEN)?;
        assert_eq!(
            gpu_hidden.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            cpu_hidden.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    #[ignore = "oracle Metal: exige un device GPU, échoue explicitement sinon"]
    fn gpu_matches_cpu_oracle_two_decisions_one_command_buffer() -> Result<()> {
        let device = Device::system_default()
            .ok_or_else(|| InferError::Metal("aucun device Metal pour l'oracle MTP D1".into()))?;
        let queue = device.new_command_queue();
        let kernels = MtpD1Kernels::new(&device)?;
        let cases = [
            // accepté puis rejeté
            GpuCase {
                primary: 1,
                budget: 4,
                cycles: [(2, [2, 3]), (5, [4, 6])],
            },
            // rejeté puis target0 EOS
            GpuCase {
                primary: 1,
                budget: 4,
                cycles: [(5, [2, 3]), (2, [STOP, 3])],
            },
            // draft égal à target0 EOS : accepté dans les stats, sans émission
            GpuCase {
                primary: 1,
                budget: 4,
                cycles: [(STOP, [STOP, 3]), (3, [3, 4])],
            },
            // primary EOS : gel complet
            GpuCase {
                primary: STOP,
                budget: 4,
                cycles: [(2, [2, 3]), (3, [3, 4])],
            },
            // bonus EOS (second stop id) : arrêt au cycle suivant sans émission
            GpuCase {
                primary: 1,
                budget: 4,
                cycles: [(2, [2, 7]), (3, [3, 4])],
            },
            // budget 1 : pas de bonus puis gel
            GpuCase {
                primary: 1,
                budget: 1,
                cycles: [(2, [2, 3]), (3, [3, 4])],
            },
            // budget 2 : bonus puis gel
            GpuCase {
                primary: 1,
                budget: 2,
                cycles: [(2, [2, 3]), (3, [3, 4])],
            },
            // budget = capacité : deux bonus remplissent la sortie
            GpuCase {
                primary: 1,
                budget: 4,
                cycles: [(2, [2, 3]), (4, [4, 5])],
            },
        ];
        for case in &cases {
            gpu_case(&device, &queue, &kernels, case)?;
        }
        Ok(())
    }
}
