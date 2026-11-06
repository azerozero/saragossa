//! Profil GPU du decode Flash-Next (`devtools`, `SARAGOSSA_FLASH_NEXT_PROFILE=1`).
//!
//! Chaque brique a son propre encodeur, échantillonné en début et en fin
//! d'étape : temps GPU par brique, cumulé sur les pas. Les frontières
//! d'encodeur ajoutent un surcoût : le total profilé dépasse le temps d'un pas
//! normal, la répartition entre briques reste l'information utile.

use metal::{
    Buffer, CommandBufferRef, ComputePassDescriptor, CounterSampleBuffer,
    CounterSampleBufferDescriptor, Device, MTLResourceOptions, MTLStorageMode, NSRange,
};

use super::encoders::Brick;
use crate::metal_backend::read_u32_buffer;
use crate::{InferError, Result};

/// Échantillons par pas : 2 par encodeur, ~8 encodeurs par couche.
const SAMPLES: usize = 2048;

/// Vrai si le profil est demandé (`SARAGOSSA_FLASH_NEXT_PROFILE=1`).
pub(crate) fn profile_requested() -> bool {
    crate::runtime_flags::env_var("SARAGOSSA_FLASH_NEXT_PROFILE").as_deref() == Some("1")
}

/// Temps GPU cumulé par brique.
pub(crate) struct StepProfiler {
    samples: CounterSampleBuffer,
    resolved: Buffer,
    next: usize,
    pairs: Vec<(Brick, usize)>,
    total_ns: [u128; 7],
    encoders: [u64; 7],
    steps: u64,
    dispatches: u64,
    step_dispatch_start: u64,
}

/// Ligne du rapport : brique, ms par pas, encodeurs par pas.
#[cfg_attr(
    not(all(test, feature = "devtools")),
    allow(dead_code, reason = "relevé par le test de profil (devtools)")
)]
pub(crate) struct ProfileRow {
    pub(crate) brick: Brick,
    pub(crate) ms_per_step: f64,
    pub(crate) encoders_per_step: f64,
}

impl StepProfiler {
    /// Alloue le tampon de compteurs timestamp.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le device n'expose pas de compteurs timestamp.
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let set = device
            .counter_sets()
            .into_iter()
            .find(|set| set.name().to_ascii_lowercase().contains("timestamp"))
            .ok_or_else(|| InferError::Metal("aucun compteur timestamp GPU".to_string()))?;
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&set);
        descriptor.set_sample_count(SAMPLES as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let samples = device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|e| InferError::Metal(format!("MTLCounterSampleBuffer : {e}")))?;
        let resolved = device.new_buffer(
            (SAMPLES * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            samples,
            resolved,
            next: 0,
            pairs: Vec::new(),
            total_ns: [0; 7],
            encoders: [0; 7],
            steps: 0,
            dispatches: 0,
            step_dispatch_start: 0,
        })
    }

    /// Descripteur d'un encodeur de `brick`, échantillonné en début et fin.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tampon d'échantillons du pas est plein.
    pub(crate) fn pass_descriptor(&mut self, brick: Brick) -> Result<ComputePassDescriptor> {
        if self.next + 2 > SAMPLES {
            return Err(InferError::Metal(
                "profil Flash-Next : tampon plein".to_string(),
            ));
        }
        let attachment = metal::ComputePassSampleBufferAttachmentDescriptor::new();
        attachment.set_sample_buffer(&self.samples);
        attachment.set_start_of_encoder_sample_index(self.next as u64);
        attachment.set_end_of_encoder_sample_index(self.next as u64 + 1);
        let descriptor = ComputePassDescriptor::new().to_owned();
        descriptor
            .sample_buffer_attachments()
            .set_object_at(0, Some(&attachment));
        self.pairs.push((brick, self.next));
        self.next += 2;
        Ok(descriptor)
    }

    /// Début d'un pas : relève le compteur de dispatches.
    pub(crate) fn begin_step(&mut self) {
        self.step_dispatch_start = crate::metal_backend::decode_profile_snapshot().3;
    }

    /// Abandonne un pas non soumis.
    pub(crate) fn abort_step(&mut self) {
        self.pairs.clear();
        self.next = 0;
    }

    /// Encode la résolution des compteurs après les encodeurs de calcul.
    pub(crate) fn encode_resolve(&self, command_buffer: &CommandBufferRef) {
        let blit = command_buffer.new_blit_command_encoder();
        blit.resolve_counters(
            &self.samples,
            NSRange::new(0, self.next as u64),
            &self.resolved,
            0,
        );
        blit.end_encoding();
    }

    /// Cumule le pas terminé (après l'attente du command buffer).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de relecture du tampon résolu.
    pub(crate) fn end_step(&mut self) -> Result<()> {
        let words = read_u32_buffer(&self.resolved, 2 * self.next)?;
        let at = |i: usize| u64::from(words[2 * i]) | (u64::from(words[2 * i + 1]) << 32);
        for &(brick, start) in &self.pairs {
            let (begin, end) = (at(start), at(start + 1));
            if begin != u64::MAX && end != u64::MAX && end > begin {
                let slot = Brick::ALL.iter().position(|b| *b == brick).unwrap_or(0);
                self.total_ns[slot] += u128::from(end - begin);
                self.encoders[slot] += 1;
            }
        }
        self.dispatches += crate::metal_backend::decode_profile_snapshot()
            .3
            .saturating_sub(self.step_dispatch_start);
        self.pairs.clear();
        self.next = 0;
        self.steps += 1;
        Ok(())
    }

    /// Rapport par brique (ms par pas) et dispatches par pas (0 sans
    /// `SARAGOSSA_RUST_DECODE_PROFILE=1`).
    #[cfg_attr(
        not(all(test, feature = "devtools")),
        allow(dead_code, reason = "relevé par le test de profil (devtools)")
    )]
    pub(crate) fn report(&self) -> (Vec<ProfileRow>, f64) {
        let steps = self.steps.max(1) as f64;
        let rows = Brick::ALL
            .iter()
            .enumerate()
            .map(|(i, &brick)| ProfileRow {
                brick,
                ms_per_step: self.total_ns[i] as f64 / 1e6 / steps,
                encoders_per_step: self.encoders[i] as f64 / steps,
            })
            .collect();
        (rows, self.dispatches as f64 / steps)
    }

    /// Remet les cumuls à zéro (ex. après le contexte).
    #[cfg_attr(
        not(all(test, feature = "devtools")),
        allow(dead_code, reason = "relevé par le test de profil (devtools)")
    )]
    pub(crate) fn clear(&mut self) {
        self.total_ns = [0; 7];
        self.encoders = [0; 7];
        self.steps = 0;
        self.dispatches = 0;
    }
}
