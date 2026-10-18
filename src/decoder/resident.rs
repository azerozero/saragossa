//! Decode résident Metal complet et préparation des arènes.

mod decode;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
mod dflash;
mod encode;
mod mtp;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod mtp_block;
mod setup;
mod types;
mod verify;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use decode::{
    resident_sample_spec, resident_sampling_on_device, resident_sampling_supported,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use types::{ResidentEmbeddingOut, ResidentSampleSpec};
