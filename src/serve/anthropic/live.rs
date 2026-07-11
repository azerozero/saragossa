//! Blocs Anthropic construits pendant le streaming texte.

use std::io::Write;

use serde_json::json;

use super::super::error::ServeResult;
use super::super::state::ServedCompletion;
use super::{write_anthropic_content_block, write_anthropic_sse_event, AssistantOutput};

const TOOL_PREFIX: &str = "<tool_call";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

pub(super) struct AnthropicLiveBlocks {
    pending_prefix: String,
    text_index: Option<usize>,
    text_closed: bool,
    next_block_index: usize,
    tool_started: bool,
    tool_buffer: String,
    in_think: bool,
    drop_leading_text_ws: bool,
}

impl AnthropicLiveBlocks {
    pub(super) fn new() -> Self {
        Self {
            pending_prefix: String::new(),
            text_index: None,
            text_closed: false,
            next_block_index: 0,
            tool_started: false,
            tool_buffer: String::new(),
            in_think: false,
            drop_leading_text_ws: true,
        }
    }

    pub(super) fn push_text_delta<S: Write>(
        &mut self,
        stream: &mut S,
        delta: &str,
    ) -> ServeResult<()> {
        if self.tool_started {
            self.tool_buffer.push_str(delta);
            return Ok(());
        }
        self.pending_prefix.push_str(delta);
        self.drain_pending(stream)
    }

    pub(super) fn finish<S: Write>(
        &mut self,
        stream: &mut S,
        completion: &ServedCompletion,
    ) -> ServeResult<bool> {
        let output = AssistantOutput::parse(&completion.content);
        if output.has_tool_use {
            self.pending_prefix.clear();
            self.close_text_block(stream)?;
            for (index, block) in output.blocks.iter().enumerate().skip(self.next_block_index) {
                write_anthropic_content_block(stream, index, block)?;
            }
            return Ok(true);
        }
        if self.tool_started {
            let mut buffered = std::mem::take(&mut self.tool_buffer);
            buffered.push_str(&std::mem::take(&mut self.pending_prefix));
            self.tool_started = false;
            self.write_text_delta(stream, &buffered)?;
        }
        self.flush_pending_text(stream)?;
        if self.text_index.is_some() {
            self.close_text_block(stream)?;
            return Ok(false);
        }
        for (index, block) in output.blocks.iter().enumerate() {
            write_anthropic_content_block(stream, index, block)?;
        }
        Ok(false)
    }

    fn drain_pending<S: Write>(&mut self, stream: &mut S) -> ServeResult<()> {
        loop {
            if self.in_think {
                self.discard_think_prefix();
                if self.in_think {
                    return Ok(());
                }
                continue;
            }
            let Some(marker) = first_control_marker(&self.pending_prefix) else {
                return self.flush_safe_text_prefix(stream);
            };
            if marker.start > 0 {
                let text = self.pending_prefix[..marker.start].trim_end().to_string();
                self.write_text_delta(stream, &text)?;
            }
            match marker.kind {
                ControlKind::ToolCall => {
                    let buffered = self.pending_prefix[marker.start..].to_string();
                    self.pending_prefix.clear();
                    self.tool_buffer.push_str(&buffered);
                    self.tool_started = true;
                    return self.close_text_block(stream);
                }
                ControlKind::Think => {
                    let after_open = marker.start + THINK_OPEN.len();
                    self.pending_prefix = self.pending_prefix[after_open..].to_string();
                    self.in_think = true;
                }
            }
        }
    }

    fn discard_think_prefix(&mut self) {
        if let Some(end) = self.pending_prefix.find(THINK_CLOSE) {
            let after_close = end + THINK_CLOSE.len();
            self.pending_prefix = self.pending_prefix[after_close..].to_string();
            self.in_think = false;
            self.drop_leading_text_ws = true;
            return;
        }
        let hold = prefix_suffix_len(&self.pending_prefix, THINK_CLOSE);
        let discard = self.pending_prefix.len().saturating_sub(hold);
        if discard > 0 {
            self.pending_prefix.drain(..discard);
        }
    }

    fn flush_safe_text_prefix<S: Write>(&mut self, stream: &mut S) -> ServeResult<()> {
        let hold = pending_hold_len(&self.pending_prefix);
        let emit_len = self.pending_prefix.len().saturating_sub(hold);
        if emit_len == 0 {
            return Ok(());
        }
        let ready = self.pending_prefix[..emit_len].to_string();
        self.pending_prefix.drain(..emit_len);
        self.write_text_delta(stream, &ready)
    }

    fn flush_pending_text<S: Write>(&mut self, stream: &mut S) -> ServeResult<()> {
        let pending = std::mem::take(&mut self.pending_prefix);
        self.write_text_delta(stream, pending.trim_end())
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

    fn close_text_block<S: Write>(&mut self, stream: &mut S) -> ServeResult<()> {
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

    fn write_text_delta<S: Write>(&mut self, stream: &mut S, delta: &str) -> ServeResult<()> {
        let text = if self.drop_leading_text_ws || self.text_index.is_none() || self.text_closed {
            delta.trim_start()
        } else {
            delta
        };
        if text.is_empty() {
            return Ok(());
        }
        let index = self.ensure_text_block(stream)?;
        self.drop_leading_text_ws = false;
        write_anthropic_sse_event(
            stream,
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text}
            }),
        )
    }
}

#[derive(Clone, Copy)]
struct ControlMarker {
    start: usize,
    kind: ControlKind,
}

#[derive(Clone, Copy)]
enum ControlKind {
    ToolCall,
    Think,
}

fn first_control_marker(text: &str) -> Option<ControlMarker> {
    let tool = text.find(TOOL_PREFIX).map(|start| ControlMarker {
        start,
        kind: ControlKind::ToolCall,
    });
    let think = text.find(THINK_OPEN).map(|start| ControlMarker {
        start,
        kind: ControlKind::Think,
    });
    match (tool, think) {
        (Some(tool), Some(think)) if tool.start <= think.start => Some(tool),
        (Some(_), Some(think)) => Some(think),
        (Some(tool), None) => Some(tool),
        (None, Some(think)) => Some(think),
        (None, None) => None,
    }
}

fn pending_hold_len(text: &str) -> usize {
    let control = control_prefix_suffix_len(text);
    let mut hold = trailing_whitespace_len(text);
    if control > 0 {
        let prefix_len = text.len().saturating_sub(control);
        hold = hold.max(control + trailing_whitespace_len(&text[..prefix_len]));
    }
    hold
}

fn control_prefix_suffix_len(text: &str) -> usize {
    tool_call_prefix_suffix_len(text).max(prefix_suffix_len(text, THINK_OPEN))
}

fn tool_call_prefix_suffix_len(text: &str) -> usize {
    let mut hold = 0;
    for (index, _) in text.char_indices() {
        let suffix = &text[index..];
        if suffix.len() < TOOL_PREFIX.len() && could_be_tool_call_prefix(suffix) {
            hold = hold.max(suffix.len());
        }
    }
    hold
}

fn could_be_tool_call_prefix(text: &str) -> bool {
    TOOL_PREFIX.starts_with(text) || text.starts_with(TOOL_PREFIX)
}

fn prefix_suffix_len(text: &str, marker: &str) -> usize {
    let mut hold = 0;
    for (index, _) in text.char_indices() {
        let suffix = &text[index..];
        if suffix.len() < marker.len() && marker.starts_with(suffix) {
            hold = hold.max(suffix.len());
        }
    }
    hold
}

fn trailing_whitespace_len(text: &str) -> usize {
    let mut len = 0;
    for ch in text.chars().rev() {
        if !ch.is_whitespace() {
            break;
        }
        len += ch.len_utf8();
    }
    len
}
