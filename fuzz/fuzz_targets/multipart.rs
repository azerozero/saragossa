#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
    if data.len() > 65_536 {
        return;
    }
    let split = data.first().copied().unwrap_or(0) as usize % (data.len() + 1);
    let boundary = String::from_utf8_lossy(&data[..split]);
    for boundary in [boundary.as_ref(), "BND", ""] {
        if let Ok(fields) = saragossa::wire_audio::parse_multipart(&data[split..], boundary) {
            assert!(fields.iter().map(|f| f.data.len()).sum::<usize>() <= data.len());
        }
    }
    let _ = saragossa::wire_audio::extract_boundary(&String::from_utf8_lossy(data));
});
