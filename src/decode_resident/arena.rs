//! Arène Metal résidente et pool scratch à bail.

use super::kernels::{
    compile_kernel, ATTENTION_DECODE_KERNEL, GATE_KERNELS, ROPE_DECODE_COPY_KERNELS,
};
use super::utils::{write_f32_at, write_u32_at};
use super::*;

/// Type d'élément d'un buffer résident.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuElement {
    Bf16,
    F32,
    U32,
}

impl GpuElement {
    /// Renvoie la taille d'un élément en octets.
    pub(crate) const fn byte_size(self) -> usize {
        match self {
            GpuElement::Bf16 => 2,
            GpuElement::F32 | GpuElement::U32 => 4,
        }
    }
}

/// Poignée vers un buffer GPU résident : porte la longueur logique et le type.
///
/// Le clone est bon marché (`metal::Buffer` est compté par référence côté
/// Objective-C) **et partage le même `MTLBuffer`**. C'est précisément ce partage
/// qui impose la discipline de liveness de ce module : un scratch n'est exposé
/// que via un emprunt au [`ScratchLease`] vivant (cf. réserve B).
#[derive(Clone, Debug)]
pub(crate) struct GpuTensor {
    buffer: Buffer,
    len: usize,
    element: GpuElement,
}

impl GpuTensor {
    /// Renvoie le `MTLBuffer` sous-jacent (pour l'encodage des kernels).
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Renvoie la longueur logique en éléments.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Renvoie le type d'élément.
    pub(crate) fn element(&self) -> GpuElement {
        self.element
    }

    /// Renvoie la taille logique en octets.
    pub(crate) fn byte_len(&self) -> usize {
        self.len * self.element.byte_size()
    }
}

/// Convertit une longueur en octets, en signalant tout débordement.
fn checked_bytes(len: usize, element: GpuElement, label: &'static str) -> Result<u64> {
    if len == 0 {
        return Err(InferError::Metal(format!(
            "buffer résident {label} de longueur nulle"
        )));
    }
    let bytes = len
        .checked_mul(element.byte_size())
        .ok_or_else(|| InferError::Metal(format!("débordement taille buffer résident {label}")))?;
    u64::try_from(bytes)
        .map_err(|_| InferError::Metal(format!("taille buffer résident {label} hors u64")))
}

/// Alloue un buffer Metal et l'enveloppe dans un [`GpuTensor`] (sans le suivre).
pub(super) fn alloc_tensor(
    device: &Device,
    options: MTLResourceOptions,
    len: usize,
    element: GpuElement,
) -> Result<GpuTensor> {
    let bytes = checked_bytes(len, element, "tenseur résident")?;
    Ok(GpuTensor {
        buffer: device.new_buffer(bytes, options),
        len,
        element,
    })
}

/// Emplacement réutilisable du pool de scratch.
#[derive(Debug)]
struct ScratchSlot {
    buffer: Buffer,
    bytes: usize,
    element: GpuElement,
    busy: bool,
}

/// État interne du pool, protégé par `Mutex` (idiome du repo : Send+Sync, le
/// decode étant mono-thread, le verrou n'est jamais contendu).
#[derive(Debug)]
struct ScratchPoolInner {
    device: Device,
    options: MTLResourceOptions,
    slots: Vec<ScratchSlot>,
}

/// Pool de buffers scratch **à bail** (réserve B).
///
/// Interdit l'aliasing de deux scratch simultanément vivants (chaque bail
/// occupe un slot distinct), tout en réutilisant les buffers entre usages dont
/// la liveness est disjointe. Clonable (poignée partagée) et Send+Sync.
#[derive(Clone, Debug)]
pub(crate) struct ScratchPool {
    inner: Arc<Mutex<ScratchPoolInner>>,
}

impl ScratchPool {
    fn new(device: Device, options: MTLResourceOptions) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ScratchPoolInner {
                device,
                options,
                slots: Vec::new(),
            })),
        }
    }

    /// Réserve un buffer scratch de `len` éléments du type donné.
    ///
    /// Tant que le [`ScratchLease`] retourné est vivant, ce buffer n'est attribué
    /// à aucun autre bail (anti-aliasing). À sa libération, le slot redevient
    /// réutilisable. La réutilisation exige une correspondance **exacte** de la
    /// taille en octets et du type, pour des sémantiques de réemploi prévisibles
    /// sur les tailles fixes du decode.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `len == 0` (un scratch vide n'a pas de pointeur CPU)
    /// ou si la taille en octets déborde.
    pub(crate) fn lease(&self, len: usize, element: GpuElement) -> Result<ScratchLease> {
        let bytes = usize::try_from(checked_bytes(len, element, "scratch")?)
            .map_err(|_| InferError::Metal("taille scratch hors usize".to_string()))?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| InferError::Metal("pool scratch résident empoisonné".to_string()))?;
        let free = inner
            .slots
            .iter()
            .position(|slot| !slot.busy && slot.element == element && slot.bytes == bytes);
        let index = match free {
            Some(index) => index,
            None => {
                let buffer = inner.device.new_buffer(bytes as u64, inner.options);
                inner.slots.push(ScratchSlot {
                    buffer,
                    bytes,
                    element,
                    busy: false,
                });
                inner.slots.len() - 1
            }
        };
        inner.slots[index].busy = true;
        let buffer = inner.slots[index].buffer.clone();
        drop(inner);
        Ok(ScratchLease {
            pool: Arc::clone(&self.inner),
            index,
            tensor: GpuTensor {
                buffer,
                len,
                element,
            },
        })
    }

    /// Renvoie le nombre de slots physiques alloués (tests : prouve le réemploi).
    #[cfg(test)]
    pub(super) fn slot_count(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.slots.len())
            .unwrap_or(0)
    }
}

/// Bail RAII sur un buffer scratch.
///
/// `tensor()` ne rend qu'une référence empruntée : impossible d'encoder une op
/// sur le scratch après le drop du bail (le buffer pourrait être réattribué à un
/// autre bail alors que le GPU le référence encore — réserve B). Le drop rend le
/// slot au pool.
#[derive(Debug)]
pub(crate) struct ScratchLease {
    pool: Arc<Mutex<ScratchPoolInner>>,
    index: usize,
    tensor: GpuTensor,
}

impl ScratchLease {
    /// Renvoie le tenseur scratch, emprunté pour la durée du bail.
    pub(crate) fn tensor(&self) -> &GpuTensor {
        &self.tensor
    }
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.pool.lock() {
            if let Some(slot) = inner.slots.get_mut(self.index) {
                slot.busy = false;
            }
        }
    }
}

/// Kernel d'attention **decode single-query** GPU (remplace `cached_attention_one`).
///
/// Un threadgroup par tête de requête (`q_heads` groupes, 256 threads chacun).
/// Trois phases (port direct de la version CPU, correct pour tout `len`,
impl DecodeResidentState {
    /// Crée un état résident sur le device donné (buffers en `StorageModeShared`)
    /// et compile les kernels du chemin résident.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la compilation d'un kernel Metal échoue.
    pub(crate) fn new(device: Device) -> Result<Self> {
        let options = MTLResourceOptions::StorageModeShared;
        let queue = device.new_command_queue();
        let scratch = ScratchPool::new(device.clone(), options);
        let attention_decode_naive = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_naive_f32",
        )?;
        let attention_decode_flash = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_flash_f32",
        )?;
        let attention_decode_flash_d256 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_flash_d256_f32",
        )?;
        let attention_decode_2pass_1 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_2pass_1_d256_f32",
        )?;
        let attention_decode_2pass_1_d128 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_2pass_1_d128_f32",
        )?;
        let attention_decode_naive_bf16 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_naive_bf16",
        )?;
        let attention_decode_flash_bf16 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_flash_bf16",
        )?;
        let attention_decode_flash_d256_bf16 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_flash_d256_bf16",
        )?;
        let attention_decode_2pass_1_bf16 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_2pass_1_d256_bf16",
        )?;
        let attention_decode_2pass_1_d128_bf16 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_2pass_1_d128_bf16",
        )?;
        let attention_decode_2pass_2 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_2pass_2_d256_f32",
        )?;
        let attention_decode_2pass_2_d128 = compile_kernel(
            &device,
            ATTENTION_DECODE_KERNEL,
            "attention_decode_2pass_2_d128_f32",
        )?;
        let split_q_gate_kernel = compile_kernel(&device, GATE_KERNELS, "split_q_gate_f32")?;
        let attn_gate_kernel = compile_kernel(&device, GATE_KERNELS, "attn_gate_f32")?;
        let rope_decode_kernel = compile_kernel(
            &device,
            ROPE_DECODE_COPY_KERNELS,
            "rms_norm_rope_heads_decode_f32",
        )?;
        let copy_at_kernel = compile_kernel(&device, ROPE_DECODE_COPY_KERNELS, "copy_at_f32")?;
        let copy_at_f32_to_bf16_kernel =
            compile_kernel(&device, ROPE_DECODE_COPY_KERNELS, "copy_at_f32_to_bf16")?;
        let timer = GpuSectionTimer::try_new();
        Ok(Self {
            device,
            queue,
            options,
            persistent: Vec::new(),
            scratch,
            attention_decode_naive,
            attention_decode_flash,
            attention_decode_flash_d256,
            attention_decode_2pass_1,
            attention_decode_2pass_1_d128,
            attention_decode_naive_bf16,
            attention_decode_flash_bf16,
            attention_decode_flash_d256_bf16,
            attention_decode_2pass_1_bf16,
            attention_decode_2pass_1_d128_bf16,
            attention_decode_2pass_2,
            attention_decode_2pass_2_d128,
            split_q_gate_kernel,
            attn_gate_kernel,
            rope_decode_kernel,
            copy_at_kernel,
            copy_at_f32_to_bf16_kernel,
            timer,
            scratch_namespace: 0,
        })
    }

    /// Renvoie le timer per-section GPU s'il est actif (`RETI_RUST_GPU_COUNTERS`).
    pub(crate) fn gpu_timer(&self) -> Option<&GpuSectionTimer> {
        self.timer.as_ref()
    }

    /// Affecte le slot de flux (light-batch) : namespace du scratch label-keyed
    /// de l'exécuteur partagé. `0` (défaut) = comportement mono-flux historique.
    pub(crate) fn set_scratch_namespace(&mut self, namespace: u64) {
        self.scratch_namespace = namespace;
    }

    /// Renvoie le namespace scratch de ce decode (slot de flux light-batch).
    pub(crate) fn scratch_namespace(&self) -> u64 {
        self.scratch_namespace
    }

    /// Alloue un buffer **persistant** distinct, gardé vivant jusqu'au drop de
    /// l'état (KV-cache, conv/ssm, ping-pong). Jamais réutilisé par label →
    /// aucun aliasing possible (contrairement au scratch de `metal_backend`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `len == 0` ou si la taille en octets déborde.
    pub(crate) fn persistent(&mut self, len: usize, element: GpuElement) -> Result<GpuTensor> {
        let tensor = alloc_tensor(&self.device, self.options, len, element)?;
        self.persistent.push(tensor.buffer().clone());
        Ok(tensor)
    }

    /// Renvoie le pool de scratch à bail des intermédiaires transitoires.
    pub(crate) fn scratch(&self) -> &ScratchPool {
        &self.scratch
    }

    /// Renvoie le device Metal (pour la création du command buffer / des kernels).
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Renvoie la command-queue de l'arène (création du command buffer unique 1c).
    pub(crate) fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Téléverse `data` dans le buffer résident `dst` (`StorageModeShared`), avant
    /// commit. Sert au gather d'embedding du token courant vers le ping-pong hidden
    /// (input upload, PAS un readback). Borné par `dst.len()`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `dst` n'est pas f32 ou si `data` dépasse sa longueur.
    pub(crate) fn upload(&self, dst: &GpuTensor, data: &[f32]) -> Result<()> {
        write_f32_at(dst, 0, data)
    }

    /// Téléverse `data` dans le buffer résident u32 `dst`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `dst` n'est pas u32 ou si `data` dépasse sa longueur.
    pub(crate) fn upload_u32(&self, dst: &GpuTensor, data: &[u32]) -> Result<()> {
        write_u32_at(dst, 0, data)
    }
}
