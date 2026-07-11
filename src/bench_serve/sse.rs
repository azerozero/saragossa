//! Parseur SSE minimal pour les chunks OpenAI.

use std::str;

use serde_json::Value;

use super::{BenchResult, BenchServeError};

pub(super) struct SseEvent {
    pub(super) done: bool,
    pub(super) content: Option<String>,
}

#[derive(Default)]
pub(super) struct SseParser {
    buffer: Vec<u8>,
}

impl SseParser {
    pub(super) fn push(&mut self, bytes: &[u8]) -> BenchResult<Vec<SseEvent>> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((index, sep_len)) = find_sse_event_end(&self.buffer) {
            let frame = self.buffer[..index].to_vec();
            self.buffer.drain(..index + sep_len);
            if frame.iter().all(|byte| byte.is_ascii_whitespace()) {
                continue;
            }
            let text = str::from_utf8(&frame)
                .map_err(|e| BenchServeError::new(format!("chunk SSE non UTF-8: {e}")))?;
            events.push(parse_sse_frame(text)?);
        }
        Ok(events)
    }
}

fn parse_sse_frame(frame: &str) -> BenchResult<SseEvent> {
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.trim() == "[DONE]" {
        return Ok(SseEvent {
            done: true,
            content: None,
        });
    }
    if data.trim().is_empty() {
        return Ok(SseEvent {
            done: false,
            content: None,
        });
    }
    let value: Value = serde_json::from_str(&data)
        .map_err(|e| BenchServeError::new(format!("JSON chunk SSE invalide: {e}")))?;
    let content = value
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|choice| choice.get("delta"))
        .filter_map(|delta| delta.get("content"))
        .filter_map(Value::as_str)
        .collect::<String>();
    Ok(SseEvent {
        done: false,
        content: (!content.is_empty()).then_some(content),
    })
}

fn find_sse_event_end(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4));
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(found), None) | (None, Some(found)) => Some(found),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_extracts_split_content_and_done() {
        let mut parser = SseParser::default();
        let first = br#"data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"bon"#;
        let second = br#"jour"}}]}

data: [DONE]

"#;

        let events = parser.push(first).expect("invariant: premier fragment ok");
        assert_eq!(events.len(), 1);
        assert!(events[0].content.is_none());

        let events = parser.push(second).expect("invariant: second fragment ok");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].content.as_deref(), Some("bonjour"));
        assert!(events[1].done);
    }
}
