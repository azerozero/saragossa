#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    if data.len() > 1_048_576 {
        return;
    }
    if let Ok(samples) = saragossa::wire_audio::wav_to_mono_16k(data) {
        assert!(samples.len() <= saragossa::wire_audio::MAX_WAV_SAMPLES as usize);
    }
});
