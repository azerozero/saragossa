//! Decode résident Metal complet et préparation des arènes.

mod decode;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
pub(in crate::decoder) mod dflash;
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
mod dflash_tree;
mod encode;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod lookup_greedy;
#[cfg(all(test, target_os = "macos", feature = "metal", feature = "devtools"))]
pub(in crate::decoder) use lookup_greedy::take_lookup_greedy_trace;
mod mtp;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod mtp_block;
mod setup;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[cfg(feature = "tts")]
pub(crate) mod tts_codebooks;
mod types;
mod verify;

#[cfg(all(test, target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use decode::RESIDENT_DECODE_STEPS;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use decode::{
    resident_sample_spec, resident_sampling_on_device, resident_sampling_supported,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(in crate::decoder) use types::{
    ResidentEmbeddingHead, ResidentEmbeddingOut, ResidentSampleSpec,
};
#[cfg(all(test, target_os = "macos", feature = "metal", feature = "devtools"))]
pub(in crate::decoder) use verify::{VerifyRecord, VERIFY_RECORDER};
