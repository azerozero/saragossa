//! Positions lues sur GPU pour RoPE, append KV et SDPA decode.
//!
//! Contrat : `ctrl[slot]` (u32) contient la position `P` de la ligne 0 au
//! moment où le GPU exécute la couche. La ligne `row` est rotée à `P + row`,
//! écrite dans le KV à `P + row` et attend sur `P + row + 1` lignes. `len` hôte
//! du KV n'est ni lu ni modifié : l'appelant le resynchronise une seule fois
//! après le bloc (`FullAttentionMetalState::truncate(position_finale)`).
//!
//! `max_position` est une borne hôte de `ctrl[slot]` sur tout le bloc. Elle
//! dimensionne la capacité (`max_position + rows <= capacity`, vérifié avant
//! tout dispatch) et la grille 2-passes ; le GPU borne aussi `ctrl[slot]` à
//! `max_position` (aucune écriture hors capacité si le contrat est violé).
//!
//! Les kernels SDPA/RoPE existants sont réutilisés tels quels : leurs `dims`
//! `constant uint4` sont liés à un buffer écrit sur GPU (`gpu_position_params`)
//! au lieu de `set_bytes`. Seul l'append exige un kernel dédié
//! (`copy_at_position_f32`), l'offset de binding étant fixé par le CPU.
//!
//! Restrictions explicites (repli parent sinon) : KV f32, pas de fenêtre
//! glissante (`window_start == 0`), `rows <= GPU_POSITION_MAX_ROWS`.

use super::kernels::compile_kernel;
use super::utils::sdpa_2pass_blocks;
use super::*;

/// Nombre maximal de lignes par couche (templates passés par `set_bytes` ≤ 4 Ko).
pub(crate) const GPU_POSITION_MAX_ROWS: usize = 16;
/// Entrées de paramètres par ligne : RoPE q, RoPE k, SDPA, SDPA 2-passes.
const ENTRIES_PER_ROW: usize = 4;
/// Pas d'une entrée en u32 (256 octets : alignement des offsets `constant`).
const PARAM_STRIDE_U32: usize = 64;
const PARAM_STRIDE_BYTES: u64 = (PARAM_STRIDE_U32 * 4) as u64;
/// Valeur de `lane` qui laisse le template inchangé.
const NO_LANE: u32 = 4;

const GPU_POSITION_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GpuPositionTemplate {
    uint4 base;
    uint4 meta; // x = composante remplacée, y = ajout, z = valeur maximale
};

kernel void gpu_position_params(
    device const uint* ctrl [[buffer(0)]],
    constant GpuPositionTemplate* templates [[buffer(1)]],
    device uint* params [[buffer(2)]],
    constant uint2& cfg [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= cfg.y) { return; }
    const GpuPositionTemplate t = templates[i];
    uint4 v = t.base;
    if (t.meta.x < 4u) {
        v[t.meta.x] = min(ctrl[cfg.x], t.meta.z - t.meta.y) + t.meta.y;
    }
    device uint4* dst = (device uint4*)(params + i * 64u);
    *dst = v;
}

kernel void copy_at_position_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& p [[buffer(2)]], // n, slot, row, max_position
    device const uint* ctrl [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.x) { return; }
    const uint position = min(ctrl[p.y], p.w) + p.z;
    output[position * p.x + i] = input[i];
}
"#;

/// Position GPU d'un bloc : `ctrl[slot]` est la position de la ligne 0.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuPosition<'a> {
    pub ctrl: &'a BufferRef,
    pub slot: usize,
    pub max_position: usize,
}

/// Paramètre `constant uint4` : octets hôte ou buffer GPU.
#[derive(Clone, Copy)]
pub(super) enum U32x4Param<'a> {
    Bytes([u32; 4]),
    Buffer(&'a BufferRef, u64),
}

impl U32x4Param<'_> {
    pub(super) fn bind(&self, encoder: &ComputeCommandEncoderRef, index: u64) {
        match self {
            Self::Bytes(values) => encoder.set_bytes(
                index,
                std::mem::size_of::<[u32; 4]>() as u64,
                values.as_ptr().cast::<c_void>(),
            ),
            Self::Buffer(buffer, offset) => encoder.set_buffer(index, Some(buffer), *offset),
        }
    }
}

pub(super) fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| InferError::Dimension(format!("{label} hors u32")))
}

/// `dims` RoPE hôte `[heads, head_dim, rope_dims, position]` (chemin historique).
///
/// # Errors
///
/// Renvoie une erreur si une valeur déborde `u32`.
pub(super) fn rope_dims_param(
    num_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    position: usize,
) -> Result<U32x4Param<'static>> {
    Ok(U32x4Param::Bytes([
        to_u32(num_heads, "rope num_heads")?,
        to_u32(head_dim, "rope head_dim")?,
        to_u32(rope_dims, "rope rope_dims")?,
        to_u32(position, "rope position")?,
    ]))
}

/// Préconditions hôtes vérifiées avant tout dispatch d'un chemin position GPU.
///
/// # Errors
///
/// Renvoie une erreur `Config` pour une configuration non couverte (repli
/// parent) et `Dimension` pour une capacité insuffisante ou un débordement.
pub(crate) fn check_gpu_position_rows(
    capacity: usize,
    kv_dim: usize,
    kv_format: KvFormat,
    window_start: usize,
    max_position: usize,
    rows: usize,
) -> Result<()> {
    if rows == 0 || rows > GPU_POSITION_MAX_ROWS {
        return Err(InferError::Config(format!(
            "position GPU: rows={rows} hors [1, {GPU_POSITION_MAX_ROWS}]"
        )));
    }
    // Les kernels position GPU copient et lisent du f32 brut : bf16 et formats
    // quantifiés reviennent au chemin parent (position hôte).
    if kv_format != KvFormat::F32 {
        return Err(InferError::Config(format!(
            "position GPU: KV {kv_format:?} non supporté (f32 seulement)"
        )));
    }
    if window_start != 0 {
        return Err(InferError::Config(
            "position GPU: fenêtre glissante non supportée".to_string(),
        ));
    }
    let end = max_position
        .checked_add(rows)
        .ok_or_else(|| InferError::Dimension("position GPU: borne déborde".to_string()))?;
    if end > capacity {
        return Err(InferError::Dimension(format!(
            "position GPU: max_position {max_position} + rows {rows} > capacité {capacity}"
        )));
    }
    let cells = capacity
        .checked_mul(kv_dim)
        .ok_or_else(|| InferError::Dimension("position GPU: cellules KV débordent".to_string()))?;
    to_u32(cells, "position GPU: cellules KV")?;
    Ok(())
}

/// Templates `[base(4), lane, add, max, 0]` pour `rows` lignes, dans l'ordre
/// RoPE q, RoPE k, SDPA `[q, kv, hd, len]`, SDPA 2-passes `[q, kv, len, blocks]`.
///
/// # Errors
///
/// Renvoie une erreur si une dimension déborde `u32`.
pub(crate) fn gpu_position_templates(
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    max_position: usize,
    rows: usize,
) -> Result<Vec<[u32; 8]>> {
    let q = to_u32(q_heads, "position GPU q_heads")?;
    let kv = to_u32(kv_heads, "position GPU kv_heads")?;
    let hd = to_u32(head_dim, "position GPU head_dim")?;
    let rd = to_u32(rope_dims, "position GPU rope_dims")?;
    let mut templates = Vec::with_capacity(rows * ENTRIES_PER_ROW);
    for row in 0..rows {
        let r = to_u32(row, "position GPU row")?;
        let pos_max = to_u32(max_position + row, "position GPU position max")?;
        let len_max = to_u32(max_position + row + 1, "position GPU len max")?;
        let blocks = to_u32(
            sdpa_2pass_blocks(max_position + row + 1),
            "position GPU blocks",
        )?;
        templates.push([q, hd, rd, 0, 3, r, pos_max, 0]);
        templates.push([kv, hd, rd, 0, 3, r, pos_max, 0]);
        templates.push([q, kv, hd, 0, 3, r + 1, len_max, 0]);
        templates.push([q, kv, 0, blocks, 2, r + 1, len_max, 0]);
    }
    debug_assert!(templates.iter().all(|t| t[4] < NO_LANE && t[6] >= t[5]));
    Ok(templates)
}

/// Oracle CPU du kernel `gpu_position_params`.
#[cfg(test)]
pub(crate) fn gpu_position_params_cpu(templates: &[[u32; 8]], ctrl_value: u32) -> Vec<[u32; 4]> {
    templates
        .iter()
        .map(|t| {
            let mut v = [t[0], t[1], t[2], t[3]];
            if let Some(slot) = v.get_mut(t[4] as usize) {
                *slot = ctrl_value.min(t[6] - t[5]) + t[5];
            }
            v
        })
        .collect()
}

/// Paramètres GPU d'une couche (`rows` lignes), vivants jusqu'à la fin de son encodage.
pub(crate) struct GpuPositionRows<'a> {
    position: GpuPosition<'a>,
    params: ScratchLease,
    copy_at_position: ComputePipelineState,
}

/// Pipelines position GPU, compilés paresseusement : hors bloc MTP, aucun coût.
#[derive(Debug)]
pub(super) struct GpuPositionKernels {
    params: ComputePipelineState,
    copy_at_position: ComputePipelineState,
}

impl<'a> GpuPositionRows<'a> {
    pub(crate) fn position(&self) -> GpuPosition<'a> {
        self.position
    }

    pub(super) fn copy_at_position(&self) -> &ComputePipelineState {
        &self.copy_at_position
    }

    fn entry(&self, row: usize, kind: usize) -> U32x4Param<'_> {
        let index = (row * ENTRIES_PER_ROW + kind) as u64;
        U32x4Param::Buffer(self.params.tensor().buffer(), index * PARAM_STRIDE_BYTES)
    }

    pub(super) fn rope_q(&self, row: usize) -> U32x4Param<'_> {
        self.entry(row, 0)
    }

    pub(super) fn rope_k(&self, row: usize) -> U32x4Param<'_> {
        self.entry(row, 1)
    }

    pub(super) fn sdpa(&self, row: usize) -> U32x4Param<'_> {
        self.entry(row, 2)
    }

    pub(super) fn sdpa_2pass(&self, row: usize) -> U32x4Param<'_> {
        self.entry(row, 3)
    }

    /// Buffer brut des paramètres (tests : comparaison à l'oracle CPU).
    #[cfg(test)]
    pub(super) fn params(&self) -> &GpuTensor {
        self.params.tensor()
    }
}

impl DecodeResidentState {
    fn gpu_position_kernels(&self) -> Result<&GpuPositionKernels> {
        if let Some(kernels) = self.gpu_position_kernels.get() {
            return Ok(kernels);
        }
        let kernels = GpuPositionKernels {
            params: compile_kernel(&self.device, GPU_POSITION_KERNELS, "gpu_position_params")?,
            copy_at_position: compile_kernel(
                &self.device,
                GPU_POSITION_KERNELS,
                "copy_at_position_f32",
            )?,
        };
        Ok(self.gpu_position_kernels.get_or_init(|| kernels))
    }

    /// Vérifie les préconditions puis encode la génération des paramètres
    /// position GPU des `rows` lignes d'une couche full-attn. Ni commit, ni wait.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une précondition échoue (cf. [`check_gpu_position_rows`]),
    /// si `slot` sort de `ctrl`, ou si les têtes diffèrent de celles du KV.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: encoder + KV + position + dims RoPE"
    )]
    pub(crate) fn encode_gpu_position_rows<'a>(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kv: &FullAttentionMetalState,
        position: GpuPosition<'a>,
        q_heads: usize,
        kv_heads: usize,
        rope_dims: usize,
        window_start: usize,
        rows: usize,
    ) -> Result<GpuPositionRows<'a>> {
        check_gpu_position_rows(
            kv.capacity(),
            kv.kv_dim(),
            kv.format(),
            window_start,
            position.max_position,
            rows,
        )?;
        if q_heads != kv.q_heads() || kv_heads != kv.kv_heads() {
            return Err(InferError::Config(
                "position GPU: têtes couche ≠ têtes KV".to_string(),
            ));
        }
        let ctrl_end = position
            .slot
            .checked_add(1)
            .and_then(|end| end.checked_mul(4))
            .ok_or_else(|| InferError::Dimension("position GPU slot déborde".to_string()))?;
        if ctrl_end as u64 > position.ctrl.length() {
            return Err(InferError::Dimension(format!(
                "position GPU: slot {} hors ctrl ({} octets)",
                position.slot,
                position.ctrl.length()
            )));
        }
        let templates = gpu_position_templates(
            q_heads,
            kv_heads,
            kv.head_dim(),
            rope_dims,
            position.max_position,
            rows,
        )?;
        let params = self
            .scratch()
            .lease(templates.len() * PARAM_STRIDE_U32, GpuElement::U32)?;
        let cfg = [
            to_u32(position.slot, "position GPU slot")?,
            to_u32(templates.len(), "position GPU templates")?,
        ];
        let kernels = self.gpu_position_kernels()?;
        encoder.set_compute_pipeline_state(&kernels.params);
        encoder.set_buffer(0, Some(position.ctrl), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(templates.as_slice()) as u64,
            templates.as_ptr().cast::<c_void>(),
        );
        encoder.set_buffer(2, Some(params.tensor().buffer()), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 2]>() as u64,
            cfg.as_ptr().cast::<c_void>(),
        );
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(templates.len() as u64, 1, 1),
            MTLSize::new(templates.len() as u64, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(GpuPositionRows {
            position,
            params,
            copy_at_position: kernels.copy_at_position.clone(),
        })
    }
}
