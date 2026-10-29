//! Sortie flint vers OpenAI : `<think>` en `reasoning_content`, appels XML
//! `<tool_call>` en `tool_calls`.
//!
//! Sémantique des parsers SGLang de flint (`FlintReasoningDetector`,
//! `FlintDetector` dérivé de `Glm47MoeDetector`) : un tour ne raisonne que
//! s'il ouvre `<think>`, un `<think>` jamais fermé est tout entier du
//! raisonnement, le texte est coupé au premier `<tool_call>` et seuls les
//! blocs fermés deviennent des appels. Une valeur d'argument reste une
//! chaîne quand le schéma de l'outil déclare `"type": "string"` ; sinon elle
//! est lue en JSON, et gardée en chaîne si elle n'en est pas.
//!
//! Le parseur est incrémental : le non-stream le nourrit du texte entier, le
//! stream de chaque delta. N'est émis que ce qu'aucune suite ne peut changer,
//! donc les deltas concaténés égalent le résultat non-stream.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use super::protocol::{ToolCall, ToolCallFunction};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const ARG_KEY_OPEN: &str = "<arg_key>";
const ARG_KEY_CLOSE: &str = "</arg_key>";
const ARG_VALUE_OPEN: &str = "<arg_value>";
const ARG_VALUE_CLOSE: &str = "</arg_value>";

/// Morceau de sortie flint prêt pour OpenAI.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum FlintEvent {
    /// Suite de `reasoning_content`.
    Reasoning(String),
    /// Suite de `content`.
    Content(String),
    /// Appel d'outil complet.
    ToolCall(ToolCall),
}

/// Sortie flint complète (non-stream).
#[derive(Debug, Default, PartialEq)]
pub(super) struct FlintOutput {
    /// Raisonnement, absent si le tour n'a pas ouvert `<think>`.
    pub(super) reasoning: Option<String>,
    /// Texte visible.
    pub(super) content: String,
    /// Appels d'outils.
    pub(super) tool_calls: Vec<ToolCall>,
}

/// Parseur incrémental de la sortie flint.
pub(super) struct FlintOutputParser {
    tools: Vec<Value>,
    call_id_prefix: String,
    buffer: String,
    reasoning_sent: usize,
    content_sent: usize,
    blocks_seen: usize,
    calls_sent: usize,
}

impl FlintOutputParser {
    /// `tools` : outils de la requête ; sans outils, `<tool_call>` reste du texte.
    pub(super) fn new(tools: Vec<Value>) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        Self {
            tools,
            call_id_prefix: format!("call_{nanos:x}"),
            buffer: String::new(),
            reasoning_sent: 0,
            content_sent: 0,
            blocks_seen: 0,
            calls_sent: 0,
        }
    }

    /// Parse une sortie complète.
    pub(super) fn parse(text: &str, tools: Vec<Value>) -> FlintOutput {
        let mut parser = Self::new(tools);
        let mut events = parser.push(text);
        events.extend(parser.finish());
        let mut output = FlintOutput::default();
        for event in events {
            match event {
                FlintEvent::Reasoning(text) => {
                    output
                        .reasoning
                        .get_or_insert_with(String::new)
                        .push_str(&text);
                }
                FlintEvent::Content(text) => output.content.push_str(&text),
                FlintEvent::ToolCall(mut call) => {
                    call.index = None;
                    output.tool_calls.push(call);
                }
            }
        }
        if output.reasoning.is_none() && parser.reasoning_opened() {
            output.reasoning = Some(String::new());
        }
        output
    }

    /// Ajoute un delta et renvoie ce qui est devenu définitif.
    pub(super) fn push(&mut self, delta: &str) -> Vec<FlintEvent> {
        self.buffer.push_str(delta);
        self.drain(false)
    }

    /// Termine la sortie et renvoie le reste.
    pub(super) fn finish(&mut self) -> Vec<FlintEvent> {
        self.drain(true)
    }

    /// Au moins un appel d'outil a-t-il été émis ?
    pub(super) fn has_tool_calls(&self) -> bool {
        self.calls_sent > 0
    }

    fn reasoning_opened(&self) -> bool {
        self.buffer.trim_start().starts_with(THINK_OPEN)
    }

    fn drain(&mut self, done: bool) -> Vec<FlintEvent> {
        let view = stable_view(&self.buffer, !self.tools.is_empty(), done);
        let mut events = Vec::new();
        if let Some(text) = unsent(view.reasoning, &mut self.reasoning_sent) {
            events.push(FlintEvent::Reasoning(text));
        }
        if let Some(text) = unsent(view.content, &mut self.content_sent) {
            events.push(FlintEvent::Content(text));
        }
        let blocks = closed_tool_blocks(view.calls);
        for block in blocks.iter().skip(self.blocks_seen) {
            let Some((name, arguments)) = parse_tool_block(block, &self.tools) else {
                continue;
            };
            events.push(FlintEvent::ToolCall(ToolCall {
                index: Some(self.calls_sent),
                id: format!("{}_{}", self.call_id_prefix, self.calls_sent),
                kind: "function",
                function: ToolCallFunction {
                    name,
                    arguments: arguments_json(&arguments),
                },
            }));
            self.calls_sent += 1;
        }
        self.blocks_seen = blocks.len();
        events
    }
}

/// Partie de `text` au-delà de `sent`, puis avance `sent`.
fn unsent(text: &str, sent: &mut usize) -> Option<String> {
    let rest = text.get(*sent..).filter(|rest| !rest.is_empty())?;
    *sent = text.len();
    Some(rest.to_string())
}

/// Ce qui de la sortie ne peut plus changer : raisonnement et texte émis
/// sont toujours des préfixes de leurs valeurs finales.
#[derive(Default)]
struct StableView<'a> {
    reasoning: &'a str,
    content: &'a str,
    /// Zone des appels, à partir du premier `<tool_call>`.
    calls: &'a str,
}

fn stable_view(buffer: &str, tools: bool, done: bool) -> StableView<'_> {
    let start = buffer.trim_start();
    let (reasoning, rest) = if let Some(after) = start.strip_prefix(THINK_OPEN) {
        let after = after.trim_start();
        match after.find(THINK_CLOSE) {
            Some(end) => (&after[..end], &after[end + THINK_CLOSE.len()..]),
            None if done => (after.trim_end(), ""),
            None => (without_partial_marker(after, THINK_CLOSE).trim_end(), ""),
        }
    } else if !done && THINK_OPEN.starts_with(start) {
        // Début ambigu : `<thi` peut encore ouvrir un raisonnement.
        return StableView::default();
    } else {
        ("", buffer)
    };
    let rest = rest.trim_start();
    if tools {
        if let Some(open) = rest.find(TOOL_OPEN) {
            return StableView {
                reasoning,
                content: rest[..open].trim_end(),
                calls: &rest[open..],
            };
        }
    }
    let content = if !done && tools {
        without_partial_marker(rest, TOOL_OPEN)
    } else {
        rest
    };
    StableView {
        reasoning,
        content: content.trim_end(),
        calls: "",
    }
}

/// Retire de la fin de `text` le plus long début strict de `marker`, qu'un
/// delta suivant pourrait compléter.
fn without_partial_marker<'a>(text: &'a str, marker: &str) -> &'a str {
    let longest = (1..marker.len())
        .rev()
        .find(|&len| text.ends_with(&marker[..len]))
        .unwrap_or(0);
    &text[..text.len() - longest]
}

/// Intérieurs des blocs `<tool_call>…</tool_call>` fermés, dans l'ordre.
fn closed_tool_blocks(region: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut rest = region;
    while let Some(open) = rest.find(TOOL_OPEN) {
        let after = &rest[open + TOOL_OPEN.len()..];
        let Some(close) = after.find(TOOL_CLOSE) else {
            break;
        };
        blocks.push(&after[..close]);
        rest = &after[close + TOOL_CLOSE.len()..];
    }
    blocks
}

/// Objet JSON des arguments, dans l'ordre où le modèle les a écrits.
fn arguments_json(arguments: &[(String, Value)]) -> String {
    let members: Vec<String> = arguments
        .iter()
        .map(|(key, value)| format!("{}:{value}", Value::String(key.clone())))
        .collect();
    format!("{{{}}}", members.join(","))
}

/// `name<arg_key>k</arg_key><arg_value>v</arg_value>…` vers nom et arguments
/// ordonnés ; une clé répétée garde sa place et sa dernière valeur.
fn parse_tool_block(block: &str, tools: &[Value]) -> Option<(String, Vec<(String, Value)>)> {
    let name_end = block.find(ARG_KEY_OPEN).unwrap_or(block.len());
    let name = block[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let mut arguments: Vec<(String, Value)> = Vec::new();
    let mut rest = &block[name_end..];
    while let Some(open) = rest.find(ARG_KEY_OPEN) {
        rest = &rest[open + ARG_KEY_OPEN.len()..];
        let Some(key_end) = rest.find(ARG_KEY_CLOSE) else {
            break;
        };
        let key = rest[..key_end].trim();
        let Some(value) = rest[key_end + ARG_KEY_CLOSE.len()..]
            .trim_start()
            .strip_prefix(ARG_VALUE_OPEN)
        else {
            continue;
        };
        let Some(value_end) = value.find(ARG_VALUE_CLOSE) else {
            break;
        };
        let raw = value[..value_end].trim();
        let typed = if argument_type(tools, name, key) == Some("string") {
            Value::String(raw.to_string())
        } else {
            // ponytail: pas de repli `ast.literal_eval` Python (`True`, `'x'`) ;
            // une valeur non JSON reste une chaîne.
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        match arguments.iter_mut().find(|(name, _)| name == key) {
            Some(slot) => slot.1 = typed,
            None => arguments.push((key.to_string(), typed)),
        }
        rest = &value[value_end + ARG_VALUE_CLOSE.len()..];
    }
    Some((name.to_string(), arguments))
}

/// Type JSON Schema déclaré pour l'argument `key` de l'outil `name`.
fn argument_type<'a>(tools: &'a [Value], name: &str, key: &str) -> Option<&'a str> {
    tools
        .iter()
        .map(|tool| tool.get("function").unwrap_or(tool))
        .find(|function| function.get("name").and_then(Value::as_str) == Some(name))?
        .get("parameters")?
        .get("properties")?
        .get(key)?
        .get("type")?
        .as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        serde_json::from_str(
            r#"[{"type":"function","function":{"name":"meteo","parameters":{"type":"object",
                "properties":{"ville":{"type":"string"},"jours":{"type":"integer"},
                "code":{"type":"string"}}}}},
               {"type":"function","function":{"name":"heure"}}]"#,
        )
        .expect("invariant: JSON valide")
    }

    fn call(output: &FlintOutput, index: usize) -> (String, Value) {
        let call = &output.tool_calls[index];
        let arguments =
            serde_json::from_str(&call.function.arguments).expect("invariant: arguments JSON");
        (call.function.name.clone(), arguments)
    }

    const MIXED: &str = "\n<think>Il faut la météo\npuis l'heure.\n</think>\n\nJe regarde.\n\
        <tool_call>meteo<arg_key>ville</arg_key><arg_value>Lyon</arg_value>\
        <arg_key>jours</arg_key>\n<arg_value>2</arg_value>\
        <arg_key>code</arg_key><arg_value>69</arg_value>\
        <arg_key>options</arg_key><arg_value>{\"unite\": \"C\"}</arg_value>\
        <arg_key>note</arg_key><arg_value>pas du json</arg_value></tool_call>\n\
        <tool_call>heure </tool_call>texte final ignoré<tool_call>meteo<arg_key>ville";

    #[test]
    fn parses_reasoning_text_and_typed_tool_calls() {
        let output = FlintOutputParser::parse(MIXED, tools());

        assert_eq!(
            output.reasoning.as_deref(),
            Some("Il faut la météo\npuis l'heure.\n")
        );
        assert_eq!(output.content, "Je regarde.");
        assert_eq!(output.tool_calls.len(), 2);
        let (name, arguments) = call(&output, 0);
        assert_eq!(name, "meteo");
        assert_eq!(
            arguments,
            serde_json::json!({
                "ville": "Lyon",
                "jours": 2,
                "code": "69",
                "options": {"unite": "C"},
                "note": "pas du json"
            })
        );
        assert_eq!(
            call(&output, 1),
            ("heure".to_string(), serde_json::json!({}))
        );
        assert_ne!(output.tool_calls[0].id, output.tool_calls[1].id);
        assert!(output.tool_calls[0]
            .function
            .arguments
            .starts_with(r#"{"ville":"Lyon","jours":2,"code":"69","#));
        assert_eq!(output.tool_calls[0].index, None);
    }

    #[test]
    fn plain_text_has_no_reasoning_and_unterminated_think_is_reasoning() {
        let plain = FlintOutputParser::parse(" Bonjour </think> ", Vec::new());
        assert_eq!(plain.reasoning, None);
        assert_eq!(plain.content, "Bonjour </think>");

        let cut = FlintOutputParser::parse("<think>je pense </thi", Vec::new());
        assert_eq!(cut.reasoning.as_deref(), Some("je pense </thi"));
        assert_eq!(cut.content, "");

        let short = FlintOutputParser::parse("<thi", Vec::new());
        assert_eq!(short.content, "<thi");
    }

    #[test]
    fn tool_markup_stays_text_without_tools() {
        let output = FlintOutputParser::parse("<tool_call>heure</tool_call>", Vec::new());

        assert!(output.tool_calls.is_empty());
        assert_eq!(output.content, "<tool_call>heure</tool_call>");
    }

    #[test]
    fn streaming_at_any_split_matches_non_stream() {
        let expected = FlintOutputParser::parse(MIXED, tools());
        let boundaries: Vec<usize> = MIXED
            .char_indices()
            .map(|(index, _)| index)
            .chain([MIXED.len()])
            .collect();
        // Chaque coupure en deux, puis un caractère par delta.
        let mut splits: Vec<Vec<&str>> = boundaries
            .iter()
            .map(|&at| vec![&MIXED[..at], &MIXED[at..]])
            .collect();
        splits.push(
            boundaries
                .windows(2)
                .map(|pair| &MIXED[pair[0]..pair[1]])
                .collect(),
        );
        for deltas in splits {
            let mut parser = FlintOutputParser::new(tools());
            let mut streamed = FlintOutput::default();
            let mut events = Vec::new();
            for delta in &deltas {
                events.extend(parser.push(delta));
            }
            events.extend(parser.finish());
            for event in events {
                match event {
                    FlintEvent::Reasoning(text) => {
                        streamed
                            .reasoning
                            .get_or_insert_with(String::new)
                            .push_str(&text);
                    }
                    FlintEvent::Content(text) => streamed.content.push_str(&text),
                    FlintEvent::ToolCall(mut call) => {
                        assert_eq!(call.index, Some(streamed.tool_calls.len()));
                        call.index = None;
                        streamed.tool_calls.push(call);
                    }
                }
            }
            assert_eq!(streamed.reasoning, expected.reasoning, "{deltas:?}");
            assert_eq!(streamed.content, expected.content, "{deltas:?}");
            let names = |output: &FlintOutput| {
                output
                    .tool_calls
                    .iter()
                    .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(names(&streamed), names(&expected), "{deltas:?}");
            assert!(parser.has_tool_calls());
        }
    }

    #[test]
    fn streaming_emits_text_before_the_turn_ends() {
        let mut parser = FlintOutputParser::new(tools());

        assert!(parser.push("<thi").is_empty());
        assert_eq!(
            parser.push("nk>abc</thi"),
            vec![FlintEvent::Reasoning("abc".to_string())]
        );
        assert_eq!(
            parser.push("nk>Salut <tool"),
            vec![FlintEvent::Content("Salut".to_string())]
        );
        let events = parser.push("_call>heure</tool_call>");
        assert!(
            matches!(&events[..], [FlintEvent::ToolCall(call)] if call.function.name == "heure")
        );
    }
}
