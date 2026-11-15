#![no_main]
use libfuzzer_sys::fuzz_target;
use saragossa::sse::SseParser;
fuzz_target!(|data: &[u8]| {
    if data.len() > 65_536 {
        return;
    }
    let mut whole = SseParser::default();
    let expected = whole.push(data);
    for size in [1, 7, 127] {
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        let mut failed = false;
        for chunk in data.chunks(size) {
            match parser.push(chunk) {
                Ok(batch) => events.extend(batch.into_iter().map(|e| (e.done, e.content))),
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        assert_eq!(failed, expected.is_err());
        if let Ok(batch) = &expected {
            let expected: Vec<_> = batch.iter().map(|e| (e.done, e.content.clone())).collect();
            assert_eq!(events, expected);
        }
    }
});
