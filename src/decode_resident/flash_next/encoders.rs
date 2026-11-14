//! Encodeurs d'un pas Flash-Next : un seul encodeur en production ; en profil
//! (`devtools`), un encodeur par brique échantillonné aux frontières d'étape
//! (seul mode de compteurs des GPU Apple), pour attribuer le temps GPU.

use metal::{CommandBufferRef, ComputeCommandEncoderRef};

use crate::metal_backend::EncoderEndGuard;
use crate::Result;

/// Brique d'un pas, unité d'attribution du profil.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Brick {
    Ple,
    Hc,
    DeltaNet,
    Qsa,
    Moe,
    Mixer,
    LmHead,
}

impl Brick {
    pub(crate) const ALL: [Brick; 7] = [
        Brick::Ple,
        Brick::Hc,
        Brick::DeltaNet,
        Brick::Qsa,
        Brick::Moe,
        Brick::Mixer,
        Brick::LmHead,
    ];

    #[cfg_attr(
        not(all(test, feature = "devtools")),
        expect(dead_code, reason = "relevé par le test de profil (devtools)")
    )]
    pub(crate) fn label(self) -> &'static str {
        match self {
            Brick::Ple => "PLE",
            Brick::Hc => "hyper-connections",
            Brick::DeltaNet => "DeltaNet",
            Brick::Qsa => "QSA",
            Brick::Moe => "MoE (routeur compris)",
            Brick::Mixer => "mixer final",
            Brick::LmHead => "lm_head",
        }
    }
}

/// Fournit l'encodeur de chaque brique d'un pas.
pub(crate) struct Encoders<'a> {
    command_buffer: &'a CommandBufferRef,
    guard: Option<EncoderEndGuard<'a>>,
    current: Option<Brick>,
    #[cfg(feature = "devtools")]
    profiler: Option<&'a mut super::profile::StepProfiler>,
}

impl<'a> Encoders<'a> {
    /// Un seul encodeur pour tout le pas.
    pub(crate) fn single(command_buffer: &'a CommandBufferRef) -> Self {
        Self {
            command_buffer,
            guard: Some(EncoderEndGuard::new(
                command_buffer.new_compute_command_encoder(),
            )),
            current: None,
            #[cfg(feature = "devtools")]
            profiler: None,
        }
    }

    /// Un encodeur par brique, échantillonné par `profiler`.
    #[cfg(feature = "devtools")]
    pub(crate) fn profiled(
        command_buffer: &'a CommandBufferRef,
        profiler: &'a mut super::profile::StepProfiler,
    ) -> Self {
        Self {
            command_buffer,
            guard: None,
            current: None,
            profiler: Some(profiler),
        }
    }

    /// Encodeur de `brick` : l'encodeur unique, ou en profil celui de la
    /// brique (un nouvel encodeur à chaque changement de brique).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tampon d'échantillons du profil est plein.
    pub(crate) fn at(&mut self, brick: Brick) -> Result<&'a ComputeCommandEncoderRef> {
        #[cfg(feature = "devtools")]
        if let Some(profiler) = self.profiler.as_deref_mut() {
            if self.current != Some(brick) || self.guard.is_none() {
                if let Some(guard) = self.guard.take() {
                    guard.end();
                }
                let descriptor = profiler.pass_descriptor(brick)?;
                self.guard = Some(EncoderEndGuard::new(
                    self.command_buffer
                        .compute_command_encoder_with_descriptor(&descriptor),
                ));
                self.current = Some(brick);
            }
        }
        let _ = brick;
        self.guard
            .as_ref()
            .map(EncoderEndGuard::encoder)
            .ok_or_else(|| crate::InferError::Metal("encodeur Flash-Next absent".to_string()))
    }

    /// Ferme l'encodeur courant.
    pub(crate) fn end(mut self) {
        if let Some(guard) = self.guard.take() {
            guard.end();
        }
    }
}
