//! Blocs Anthropic construits pendant le streaming texte.

use std::io::Write;

use serde_json::json;

use super::super::error::ServeResult;
use super::super::state::ServedCompletion;
use super::{write_anthropic_content_block, write_anthropic_sse_event};

pub(super) struct AnthropicLiveBlocks {
    text_index: Option<usize>,
    text_closed: bool,
    next_block_index: usize,
}

impl AnthropicLiveBlocks {
    pub(super) fn new() -> Self {
        Self {
            text_index: None,
            text_closed: false,
            next_block_index: 0,
        }
    }

    pub(super) fn common_event<S: Write>(
        &mut self,
        stream: &mut S,
        event: super::FlintEvent,
    ) -> ServeResult<()> {
        match event {
            super::FlintEvent::Content(text) => {
                if text.is_empty() {
                    return Ok(());
                }
                let index = self.ensure_text_block(stream)?;
                write_anthropic_sse_event(
                    stream,
                    "content_block_delta",
                    &json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":text}}),
                )
            }
            super::FlintEvent::Reasoning(_) => Ok(()),
            super::FlintEvent::ToolCall(call) => {
                self.close_text_block(stream)?;
                let block = super::AnthropicContentBlock::ToolUse {
                    id: call.id,
                    name: call.function.name,
                    input: serde_json::from_str(&call.function.arguments)
                        .map_err(|e| super::ServeError::json("arguments outil", e))?,
                };
                write_anthropic_content_block(stream, self.next_block_index, &block)?;
                self.next_block_index += 1;
                Ok(())
            }
        }
    }

    pub(super) fn push_text_delta<S: Write>(
        &mut self,
        stream: &mut S,
        delta: &str,
    ) -> ServeResult<()> {
        self.common_event(stream, super::FlintEvent::Content(delta.to_owned()))
    }

    pub(super) fn finish<S: Write>(
        &mut self,
        stream: &mut S,
        _completion: &ServedCompletion,
    ) -> ServeResult<bool> {
        if self.next_block_index == 0 {
            self.ensure_text_block(stream)?;
        }
        self.close_text_block(stream)?;
        Ok(false)
    }

    fn ensure_text_block<S: Write>(&mut self, stream: &mut S) -> ServeResult<usize> {
        if let Some(index) = self.text_index {
            if !self.text_closed {
                return Ok(index);
            }
        }
        let index = self.next_block_index;
        write_anthropic_sse_event(
            stream,
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""}
            }),
        )?;
        self.text_index = Some(index);
        self.text_closed = false;
        self.next_block_index = self.next_block_index.saturating_add(1);
        Ok(index)
    }

    pub(super) fn close_text_block<S: Write>(&mut self, stream: &mut S) -> ServeResult<()> {
        let Some(index) = self.text_index else {
            return Ok(());
        };
        if self.text_closed {
            return Ok(());
        }
        write_anthropic_sse_event(
            stream,
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        )?;
        self.text_closed = true;
        Ok(())
    }
}
