//! Adaptation commune des sorties d'outils vers les transports HTTP.
use super::flint_output::{FlintEvent, FlintOutput, FlintOutputParser};
use super::protocol::{ToolCall, ToolCallFunction};
use serde_json::{Map, Value};
use std::time::{SystemTime, UNIX_EPOCH};

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
    Framed(FramedParser),
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
        Self::Framed(FramedParser {
            family,
            tools,
            max_calls: if parallel { usize::MAX } else { 1 },
            pending: String::new(),
            calls: 0,
            in_reasoning: false,
            id: format!(
                "call_{:x}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ),
        })
    }
    pub(super) fn push(&mut self, delta: &str) -> Vec<FlintEvent> {
        match self {
            Self::Flint(p) => p.push(delta),
            Self::Framed(p) => {
                p.pending.push_str(delta);
                p.drain(false)
            }
        }
    }
    pub(super) fn finish(&mut self) -> Vec<FlintEvent> {
        match self {
            Self::Flint(p) => p.finish(),
            Self::Framed(p) => p.drain(true),
        }
    }
    pub(super) fn has_tool_calls(&self) -> bool {
        match self {
            Self::Flint(p) => p.has_tool_calls(),
            Self::Framed(p) => p.calls > 0,
        }
    }
    pub(super) fn parse(mut self, text: &str) -> FlintOutput {
        let mut events = self.push(text);
        events.extend(self.finish());
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
}

impl FramedParser {
    fn drain(&mut self, done: bool) -> Vec<FlintEvent> {
        let (open, close) = if self.family == ToolFamily::Qwen {
            ("<tool_call>", "</tool_call>")
        } else {
            ("<|tool_call>", "<tool_call|>")
        };
        let (think, end_think) = if self.family == ToolFamily::Qwen {
            ("<think>", "</think>")
        } else {
            ("<|channel>thought\n", "<channel|>")
        };
        let mut events = Vec::new();
        while !self.pending.is_empty() {
            if self.in_reasoning {
                if let Some(end) = self.pending.find(end_think) {
                    let text = self.pending[..end].to_string();
                    if !text.is_empty() {
                        events.push(FlintEvent::Reasoning(text));
                    }
                    self.pending.drain(..end + end_think.len());
                    self.in_reasoning = false;
                    continue;
                }
                if done {
                    self.pending.clear();
                }
                break;
            }
            if self.pending.starts_with(think) {
                self.pending.drain(..think.len());
                self.in_reasoning = true;
                continue;
            }
            if !self.tools.is_empty() && self.pending.starts_with(open) {
                let Some(end) = frame_end(&self.pending, self.family, close) else {
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
                    if self.calls < self.max_calls && declared(&self.tools, &name).is_some() {
                        events.push(FlintEvent::ToolCall(ToolCall {
                            index: Some(self.calls),
                            id: format!("{}_{}", self.id, self.calls),
                            kind: "function",
                            function: ToolCallFunction {
                                name,
                                arguments: arguments.to_string(),
                            },
                        }));
                        self.calls += 1;
                    }
                }
                self.pending.drain(..end + close.len());
                continue;
            }
            // Retenir un préfixe de marqueur coupé entre deux tokens, sur une frontière UTF-8.
            let mut markers = vec![think];
            if !self.tools.is_empty() {
                markers.push(open);
            }
            let end = markers
                .iter()
                .filter_map(|m| self.pending.find(m))
                .min()
                .unwrap_or(self.pending.len());
            let safe = if end < self.pending.len() || done {
                end
            } else {
                self.pending
                    .char_indices()
                    .find_map(|(i, _)| {
                        markers
                            .iter()
                            .any(|m| m.starts_with(&self.pending[i..]))
                            .then_some(i)
                    })
                    .unwrap_or(end)
            };
            if safe == 0 {
                break;
            }
            events.push(FlintEvent::Content(self.pending.drain(..safe).collect()));
        }
        events
    }
}

/// Les marqueurs écrits dans une valeur restent du texte de cette valeur.
fn frame_end(text: &str, family: ToolFamily, close: &str) -> Option<usize> {
    let mut cursor = 0;
    while cursor < text.len() {
        let rest = &text[cursor..];
        if rest.starts_with(close) {
            return Some(cursor);
        }
        if family == ToolFamily::Gemma4 && rest.starts_with("<|\"|>") {
            cursor += "<|\"|>".len();
            cursor += text[cursor..].find("<|\"|>")? + "<|\"|>".len();
        } else if family == ToolFamily::Qwen && rest.starts_with("<parameter=") {
            cursor += rest.find('>')? + 1;
            cursor += text[cursor..].find("</parameter>")? + "</parameter>".len();
        } else {
            cursor += rest.chars().next()?.len_utf8();
        }
    }
    None
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
        let string = declared(tools, name)
            .is_some_and(|f| f["parameters"]["properties"][key]["type"] == "string");
        arguments.insert(
            key.to_string(),
            if string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
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
