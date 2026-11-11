//! Adaptation commune des sorties d'outils vers les transports HTTP.
//!
//! Qwen/Gemma : les exemples Markdown et citations restent littéraux. Un bloc
//! est limité à 64 Kio UTF-8, ses arguments JSON à 32 Kio, et une réponse à
//! 64 appels (un seul sans parallélisme). Un dépassement de bloc désactive
//! l'interprétation pour le reste de la réponse ; un appel incomplet, invalide
//! ou aux arguments trop longs n'est jamais émis. Les curseurs avancent une
//! seule fois sur chaque caractère, hors préfixes de marqueurs de taille fixe.
use super::flint_output::{FlintEvent, FlintOutput, FlintOutputParser};
use super::protocol::{ToolCall, ToolCallFunction};
use serde_json::{Map, Value};
use std::time::{SystemTime, UNIX_EPOCH};

mod framing;
use framing::{FrameScan, LiteralContext};

const MAX_FRAME_BYTES: usize = 64 * 1024;
const MAX_ARGUMENT_BYTES: usize = 32 * 1024;
const MAX_CALLS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolFamily {
    Flint,
    Qwen,
    Gemma4,
}

impl ToolFamily {
    pub(super) fn from_config(config: &saragossa::ModelConfig) -> Option<Self> {
        if config.is_flint() {
            Some(Self::Flint)
        } else if config.is_gemma4() {
            Some(Self::Gemma4)
        } else if ["qwen3_5", "qwen3_6", "qwen3_8"].iter().any(|prefix| {
            config.model_type == *prefix || config.model_type.starts_with(&format!("{prefix}_"))
        }) {
            Some(Self::Qwen)
        } else {
            None
        }
    }
}

pub(super) enum ToolOutputParser {
    Flint(Box<FlintOutputParser>),
    Framed(Box<FramedParser>),
}

impl From<FlintOutputParser> for ToolOutputParser {
    fn from(parser: FlintOutputParser) -> Self {
        Self::Flint(Box::new(parser))
    }
}

impl ToolOutputParser {
    pub(super) fn new(family: ToolFamily, tools: Vec<Value>, parallel: bool) -> Self {
        if family == ToolFamily::Flint {
            return FlintOutputParser::new(tools, parallel).into();
        }
        Self::Framed(Box::new(FramedParser {
            family,
            tools,
            max_calls: if parallel { MAX_CALLS } else { 1 },
            pending: String::new(),
            calls: 0,
            in_reasoning: false,
            frame: FrameScan::default(),
            literal: LiteralContext::default(),
            passthrough: false,
            id: format!(
                "call_{:x}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ),
        }))
    }
    pub(super) fn push(&mut self, delta: &str) -> Vec<FlintEvent> {
        match self {
            Self::Flint(p) => p.push(delta),
            Self::Framed(p) => {
                let mut events = Vec::new();
                // Borner aussi un delta unique, indépendamment du budget de génération.
                for ch in delta.chars() {
                    if p.pending.len() + ch.len_utf8() > MAX_FRAME_BYTES {
                        p.passthrough = true;
                        content_event(&mut events, std::mem::take(&mut p.pending));
                    }
                    p.pending.push(ch);
                    p.drain(false, &mut events);
                }
                events
            }
        }
    }
    pub(super) fn finish(&mut self) -> Vec<FlintEvent> {
        match self {
            Self::Flint(p) => p.finish(),
            Self::Framed(p) => {
                let mut events = Vec::new();
                p.drain(true, &mut events);
                events
            }
        }
    }
    pub(super) fn has_tool_calls(&self) -> bool {
        match self {
            Self::Flint(p) => p.has_tool_calls(),
            Self::Framed(p) => p.calls > 0,
        }
    }
    pub(super) fn parse_events(mut self, text: &str) -> Vec<FlintEvent> {
        let mut events = self.push(text);
        events.extend(self.finish());
        events
    }
    pub(super) fn parse(self, text: &str) -> FlintOutput {
        let events = self.parse_events(text);
        let mut output = FlintOutput::default();
        for event in events {
            match event {
                FlintEvent::Content(text) => output.content.push_str(&text),
                FlintEvent::Reasoning(text) => output
                    .reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&text),
                FlintEvent::ToolCall(mut call) => {
                    call.index = None;
                    output.tool_calls.push(call);
                }
            }
        }
        output
    }
}

pub(super) struct FramedParser {
    family: ToolFamily,
    tools: Vec<Value>,
    max_calls: usize,
    pending: String,
    calls: usize,
    id: String,
    in_reasoning: bool,
    frame: FrameScan,
    literal: LiteralContext,
    passthrough: bool,
}

fn content_event(events: &mut Vec<FlintEvent>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(FlintEvent::Content(previous)) = events.last_mut() {
        previous.push_str(&text);
    } else {
        events.push(FlintEvent::Content(text));
    }
}

impl FramedParser {
    fn drain(&mut self, done: bool, events: &mut Vec<FlintEvent>) {
        let (open, close, think, end_think) = if self.family == ToolFamily::Qwen {
            ("<tool_call>", "</tool_call>", "<think>", "</think>")
        } else {
            (
                "<|tool_call>",
                "<tool_call|>",
                "<|channel>thought\n",
                "<channel|>",
            )
        };
        while !self.pending.is_empty() {
            if self.passthrough {
                content_event(events, std::mem::take(&mut self.pending));
                break;
            }
            if self.in_reasoning {
                if self.pending.starts_with(end_think) {
                    self.pending.drain(..end_think.len());
                    self.in_reasoning = false;
                } else if !done && end_think.starts_with(&self.pending) {
                    break;
                } else {
                    let ch = self.pending.remove(0);
                    if let Some(FlintEvent::Reasoning(text)) = events.last_mut() {
                        text.push(ch);
                    } else {
                        events.push(FlintEvent::Reasoning(ch.to_string()));
                    }
                }
                continue;
            }
            let literal = self.literal.is_literal_before(&self.pending);
            if !literal && self.pending.starts_with(think) {
                self.pending.drain(..think.len());
                self.in_reasoning = true;
                continue;
            }
            if !literal && !self.tools.is_empty() && self.pending.starts_with(open) {
                let Some(end) = self.frame.end(&self.pending, self.family, close) else {
                    if done {
                        self.pending.clear();
                    }
                    break;
                };
                let body = &self.pending[open.len()..end];
                let parsed = if self.family == ToolFamily::Qwen {
                    parse_qwen(body, &self.tools)
                } else {
                    parse_gemma(body)
                };
                if let Some((name, arguments)) = parsed {
                    let arguments = arguments.to_string();
                    if self.calls < self.max_calls
                        && arguments.len() <= MAX_ARGUMENT_BYTES
                        && declared(&self.tools, &name).is_some()
                    {
                        events.push(FlintEvent::ToolCall(ToolCall {
                            index: Some(self.calls),
                            id: format!("{}_{}", self.id, self.calls),
                            kind: "function",
                            function: ToolCallFunction { name, arguments },
                        }));
                        self.calls += 1;
                    }
                }
                self.pending.drain(..end + close.len());
                self.frame = FrameScan::default();
                continue;
            }
            if !literal
                && !done
                && (think.starts_with(&self.pending)
                    || (!self.tools.is_empty() && open.starts_with(&self.pending)))
            {
                break;
            }
            let ch = self.pending.remove(0);
            self.literal.push(ch);
            content_event(events, ch.to_string());
        }
    }
}

fn declared<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    tools
        .iter()
        .map(|tool| tool.get("function").unwrap_or(tool))
        .find(|f| f["name"].as_str() == Some(name))
}

fn parse_qwen(body: &str, tools: &[Value]) -> Option<(String, Value)> {
    let body = body.trim().strip_prefix("<function=")?;
    let (name, mut rest) = body.split_once('>')?;
    if name.is_empty() {
        return None;
    }
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix("</function>") {
            return tail
                .trim()
                .is_empty()
                .then(|| (name.to_string(), Value::Object(arguments)));
        }
        let parameter = rest.strip_prefix("<parameter=")?;
        let (key, value) = parameter.split_once('>')?;
        if key.is_empty() {
            return None;
        }
        let (raw, tail) = value.split_once("</parameter>")?;
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let string = declared(tools, name).is_some_and(|f| {
            let kind = &f["parameters"]["properties"][key]["type"];
            kind == "string"
                || kind
                    .as_array()
                    .is_some_and(|types| types.iter().any(|t| t == "string"))
        });
        arguments.insert(
            key.to_string(),
            if string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).ok()?
            },
        );
        rest = tail;
    }
}

fn parse_gemma(body: &str) -> Option<(String, Value)> {
    let body = body.trim().strip_prefix("call:")?;
    let split = body.find('{')?;
    let name = body[..split].trim();
    if name.is_empty() {
        return None;
    }
    let mut rest = &body[split..];
    let arguments = gemma_value(&mut rest, 0)?;
    (rest.trim().is_empty() && arguments.is_object()).then(|| (name.to_string(), arguments))
}

/// Grammaire Gemma : clés nues, chaînes délimitées par un token dédié.
fn gemma_value(rest: &mut &str, depth: usize) -> Option<Value> {
    if depth > 64 {
        return None;
    }
    *rest = rest.trim_start();
    if let Some(text) = rest.strip_prefix("<|\"|>") {
        let (value, tail) = text.split_once("<|\"|>")?;
        *rest = tail;
        return Some(Value::String(value.to_string()));
    }
    if let Some(tail) = rest.strip_prefix('{') {
        *rest = tail;
        let mut object = Map::new();
        loop {
            *rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix('}') {
                *rest = tail;
                return Some(Value::Object(object));
            }
            let (key, tail) = rest.split_once(':')?;
            if key.trim().is_empty() || key.contains(['{', '}', ',', '[', ']']) {
                return None;
            }
            *rest = tail;
            let value = gemma_value(rest, depth + 1)?;
            object.insert(key.trim().to_string(), value);
            *rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix(',') {
                if tail.trim_start().starts_with(['}', ']']) {
                    return None;
                }
                *rest = tail;
            } else if !rest.starts_with('}') {
                return None;
            }
        }
    }
    if let Some(tail) = rest.strip_prefix('[') {
        *rest = tail;
        let mut values = Vec::new();
        loop {
            *rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix(']') {
                *rest = tail;
                return Some(Value::Array(values));
            }
            values.push(gemma_value(rest, depth + 1)?);
            *rest = rest.trim_start();
            if let Some(tail) = rest.strip_prefix(',') {
                if tail.trim_start().starts_with(['}', ']']) {
                    return None;
                }
                *rest = tail;
            } else if !rest.starts_with(']') {
                return None;
            }
        }
    }
    let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
    let value = if rest[..end].trim() == "None" {
        Value::Null
    } else {
        serde_json::from_str(rest[..end].trim()).ok()?
    };
    *rest = &rest[end..];
    Some(value)
}

#[cfg(test)]
mod tests;
