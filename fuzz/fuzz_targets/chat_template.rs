#![no_main]
use libfuzzer_sys::fuzz_target;
use saragossa::chat_template::{render_gemma_chat, render_qwen_chatml, ChatTemplateMessage};
fuzz_target!(|data: &[u8]| {
    if data.len() > 16_384 {
        return;
    }
    let text = String::from_utf8_lossy(data);
    let messages: Vec<_> = text
        .splitn(16, '\0')
        .enumerate()
        .map(|(i, text)| {
            ChatTemplateMessage::new(if i % 2 == 0 { "user" } else { "assistant" }, text)
        })
        .collect();
    for generation in [false, true] {
        let rendered = render_qwen_chatml(&messages, generation, false);
        assert_eq!(rendered, render_qwen_chatml(&messages, generation, false));
        assert!(render_gemma_chat(&messages, generation).is_ok());
    }
});
