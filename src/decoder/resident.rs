//! Decode résident Metal complet et préparation des arènes.

mod decode;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
mod dflash;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "branché sur la boucle DFlash à l'étape suivante")
)]
mod dflash_tree;
mod encode;
mod mtp;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod mtp_block;
mod setup;
mod types;
mod verify;

#[cfg(all(test, target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use decode::RESIDENT_DECODE_STEPS;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use decode::{
    resident_sample_spec, resident_sampling_on_device, resident_sampling_supported,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use types::{ResidentEmbeddingOut, ResidentSampleSpec};
