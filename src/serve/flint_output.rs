//! Sortie flint vers OpenAI : `<think>` en `reasoning_content`, appels XML
//! `<tool_call>` en `tool_calls`.
//!
//! Sémantique des parsers SGLang de flint (`FlintReasoningDetector`,
//! `FlintDetector` dérivé de `Glm47MoeDetector`) : un tour ne raisonne que
//! s'il ouvre `<think>` ; raisonnement et texte sont trimés, un raisonnement
//! vide est absent ; un `<think>` jamais fermé est tout entier du
//! raisonnement ; le texte s'arrête au premier `<tool_call>` et ce qui suit
//! ou sépare les appels est ignoré. Une valeur d'argument reste une chaîne
//! quand le schéma de l'outil déclare `"type": "string"`, sinon elle est lue
//! en JSON et gardée en chaîne si elle n'en est pas. Un appel à un outil non
//! déclaré est omis.
//!
//! Plus strict que les regex de SGLang : un bloc n'est un appel que si toute
//! sa structure est valide, `<tool_call>nom` puis des paires
//! `<arg_key>k</arg_key><arg_value>v</arg_value>` (blancs tolérés entre les
//! balises) et `</tool_call>`. Un bloc malformé ou coupé est omis, jamais
//! converti en appel aux arguments partiels. Une balise n'est reconnue qu'à
//! sa place : dans une valeur, tout jusqu'au premier `</arg_value>` est du
//! texte, si bien qu'un `</tool_call><tool_call>…` écrit dans une valeur ne
//! fabrique pas d'appel.
//!
//! Le parseur est un automate à curseur, linéaire en la taille de la sortie :
//! le non-stream le nourrit du texte entier, le stream de chaque delta. Il
//! n'émet que ce qu'aucune suite ne peut changer, donc les deltas concaténés
//! égalent le résultat non-stream.

use std::collections::HashMap;
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
    /// Raisonnement, absent s'il est vide ou si le tour n'a pas ouvert `<think>`.
    pub(super) reasoning: Option<String>,
    /// Texte visible.
    pub(super) content: String,
    /// Appels d'outils.
    pub(super) tool_calls: Vec<ToolCall>,
}

/// Étape de l'automate.
#[derive(Clone, Copy)]
enum Phase {
    /// Début du tour : `<think>` ou texte.
    Start,
    Reasoning,
    Content,
    /// Recherche du prochain `<tool_call>`.
    SeekCall,
    /// Nom de l'outil, depuis `start`.
    Name {
        start: usize,
    },
    /// Clé d'argument, depuis `start`.
    Key {
        start: usize,
    },
    /// Après `</arg_key>`, attend `<arg_value>`.
    BeforeValue,
    /// Valeur d'argument, depuis `start`.
    Value {
        start: usize,
    },
    /// Après `</arg_value>`, attend `<arg_key>` ou `</tool_call>`.
    AfterValue,
    /// Plus rien à lire (plafond d'appels atteint).
    Done,
}

/// Résultat d'une attente de balise après d'éventuels blancs.
enum Expect {
    /// Balise `markers[index]` trouvée ; `consumed` octets lus, blancs compris.
    Matched { index: usize, consumed: usize },
    /// Suite encore ambiguë ; `skipped` octets de blancs déjà lus.
    Wait { skipped: usize },
    /// Rien d'attendu ici.
    Mismatch,
}

/// Flux de texte trimé des deux côtés : les blancs de fin sont retenus tant
/// qu'un texte non blanc ne les suit pas.
#[derive(Default)]
struct TrimmedChannel {
    started: bool,
    held: String,
}

impl TrimmedChannel {
    fn push(&mut self, text: &str) -> Option<String> {
        let text = if self.started {
            text
        } else {
            text.trim_start()
        };
        let body = text.trim_end();
        if body.is_empty() {
            if self.started {
                self.held.push_str(text);
            }
            return None;
        }
        let mut out = std::mem::take(&mut self.held);
        out.push_str(body);
        self.held.push_str(&text[body.len()..]);
        self.started = true;
        Some(out)
    }
}

/// Appel en cours de lecture.
#[derive(Default)]
struct PendingCall {
    name: String,
    key: String,
    arguments: Vec<(String, Value)>,
    positions: HashMap<String, usize>,
}

impl PendingCall {
    /// Comme un `dict` Python : la dernière valeur gagne, à la première place.
    fn insert(&mut self, key: String, value: Value) {
        match self.positions.get(&key) {
            Some(&slot) => self.arguments[slot].1 = value,
            None => {
                self.positions.insert(key.clone(), self.arguments.len());
                self.arguments.push((key, value));
            }
        }
    }
}

/// Parseur incrémental de la sortie flint.
pub(super) struct FlintOutputParser {
    tools: Vec<Value>,
    max_calls: usize,
    call_id_prefix: String,
    buffer: String,
    cursor: usize,
    phase: Phase,
    reasoning: TrimmedChannel,
    content: TrimmedChannel,
    call: PendingCall,
    calls_sent: usize,
}

impl FlintOutputParser {
    /// `tools` : outils actifs de la requête ; sans outils, `<tool_call>`
    /// reste du texte. `parallel_tool_calls` faux ne garde que le premier appel.
    pub(super) fn new(tools: Vec<Value>, parallel_tool_calls: bool) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        Self {
            tools,
            max_calls: if parallel_tool_calls { usize::MAX } else { 1 },
            call_id_prefix: format!("call_{nanos:x}"),
            buffer: String::new(),
            cursor: 0,
            phase: Phase::Start,
            reasoning: TrimmedChannel::default(),
            content: TrimmedChannel::default(),
            call: PendingCall::default(),
            calls_sent: 0,
        }
    }

    /// Parse une sortie complète.
    pub(super) fn parse(mut self, text: &str) -> FlintOutput {
        let mut events = self.push(text);
        events.extend(self.finish());
        let mut output = FlintOutput::default();
        for event in events {
            match event {
                FlintEvent::Reasoning(text) => output
                    .reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&text),
                FlintEvent::Content(text) => output.content.push_str(&text),
                FlintEvent::ToolCall(mut call) => {
                    call.index = None;
                    output.tool_calls.push(call);
                }
            }
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

    fn drain(&mut self, done: bool) -> Vec<FlintEvent> {
        let mut events = Vec::new();
        while self.step(done, &mut events) {}
        events
    }

    /// Avance l'automate d'une étape ; faux quand il faut plus de texte.
    fn step(&mut self, done: bool, events: &mut Vec<FlintEvent>) -> bool {
        let rest = &self.buffer[self.cursor..];
        match self.phase {
            Phase::Start => {
                let trimmed = rest.trim_start();
                self.cursor += rest.len() - trimmed.len();
                if trimmed.starts_with(THINK_OPEN) {
                    self.cursor += THINK_OPEN.len();
                    self.phase = Phase::Reasoning;
                } else if !done && THINK_OPEN.starts_with(trimmed) {
                    // `<thi` peut encore ouvrir un raisonnement.
                    return false;
                } else {
                    self.phase = Phase::Content;
                }
                true
            }
            Phase::Reasoning => {
                let (end, after) = match rest.find(THINK_CLOSE) {
                    Some(end) => (end, Some(end + THINK_CLOSE.len())),
                    None => (safe_len(rest, THINK_CLOSE, done), None),
                };
                if let Some(text) = self.reasoning.push(&rest[..end]) {
                    events.push(FlintEvent::Reasoning(text));
                }
                self.cursor += after.unwrap_or(end);
                if after.is_some() {
                    self.phase = Phase::Content;
                }
                after.is_some()
            }
            Phase::Content => {
                let tools = !self.tools.is_empty();
                let (end, after) = match rest.find(TOOL_OPEN).filter(|_| tools) {
                    Some(open) => (open, Some(open + TOOL_OPEN.len())),
                    None if tools => (safe_len(rest, TOOL_OPEN, done), None),
                    None => (rest.len(), None),
                };
                if let Some(text) = self.content.push(&rest[..end]) {
                    events.push(FlintEvent::Content(text));
                }
                self.cursor += after.unwrap_or(end);
                if after.is_some() {
                    self.phase = Phase::Name { start: self.cursor };
                }
                after.is_some()
            }
            Phase::SeekCall => match rest.find(TOOL_OPEN) {
                Some(open) => {
                    self.cursor += open + TOOL_OPEN.len();
                    self.phase = Phase::Name { start: self.cursor };
                    true
                }
                None => {
                    self.cursor += safe_len(rest, TOOL_OPEN, done);
                    false
                }
            },
            Phase::Name { start } => {
                let Some(tag) = rest.find('<') else {
                    self.cursor = self.buffer.len();
                    return self.reject_if(done);
                };
                let tag = self.cursor + tag;
                let name = self.buffer[start..tag].trim().to_string();
                self.cursor = tag;
                match expect(&self.buffer[tag..], done, &[ARG_KEY_OPEN, TOOL_CLOSE]) {
                    Expect::Wait { .. } => false,
                    Expect::Matched { .. } if name.is_empty() => self.reject(),
                    Expect::Matched { index, consumed } => {
                        self.cursor += consumed;
                        self.call.name = name;
                        if index == 0 {
                            self.phase = Phase::Key { start: self.cursor };
                        } else {
                            self.complete_call(events);
                        }
                        true
                    }
                    Expect::Mismatch => self.reject(),
                }
            }
            Phase::Key { start } => match rest.find(ARG_KEY_CLOSE) {
                Some(end) => {
                    let end = self.cursor + end;
                    self.call.key = self.buffer[start..end].trim().to_string();
                    self.cursor = end + ARG_KEY_CLOSE.len();
                    self.phase = Phase::BeforeValue;
                    true
                }
                None => {
                    self.cursor += safe_len(rest, ARG_KEY_CLOSE, done);
                    self.reject_if(done)
                }
            },
            Phase::BeforeValue => match expect(rest, done, &[ARG_VALUE_OPEN]) {
                Expect::Matched { consumed, .. } => {
                    self.cursor += consumed;
                    self.phase = Phase::Value { start: self.cursor };
                    true
                }
                Expect::Wait { skipped } => {
                    self.cursor += skipped;
                    false
                }
                Expect::Mismatch => self.reject(),
            },
            Phase::Value { start } => match rest.find(ARG_VALUE_CLOSE) {
                Some(end) => {
                    let end = self.cursor + end;
                    let raw = self.buffer[start..end].trim();
                    let key = std::mem::take(&mut self.call.key);
                    let value = typed_argument(&self.tools, &self.call.name, &key, raw);
                    self.call.insert(key, value);
                    self.cursor = end + ARG_VALUE_CLOSE.len();
                    self.phase = Phase::AfterValue;
                    true
                }
                None => {
                    self.cursor += safe_len(rest, ARG_VALUE_CLOSE, done);
                    self.reject_if(done)
                }
            },
            Phase::AfterValue => match expect(rest, done, &[ARG_KEY_OPEN, TOOL_CLOSE]) {
                Expect::Matched { index, consumed } => {
                    self.cursor += consumed;
                    if index == 0 {
                        self.phase = Phase::Key { start: self.cursor };
                    } else {
                        self.complete_call(events);
                    }
                    true
                }
                Expect::Wait { skipped } => {
                    self.cursor += skipped;
                    false
                }
                Expect::Mismatch => self.reject(),
            },
            Phase::Done => {
                self.cursor = self.buffer.len();
                false
            }
        }
    }

    /// Abandonne le bloc en cours ; la recherche reprend au curseur.
    fn reject(&mut self) -> bool {
        self.call = PendingCall::default();
        self.phase = Phase::SeekCall;
        true
    }

    /// Bloc coupé par la fin de la sortie : omis.
    fn reject_if(&mut self, done: bool) -> bool {
        if done {
            self.call = PendingCall::default();
            self.phase = Phase::SeekCall;
        }
        false
    }

    fn complete_call(&mut self, events: &mut Vec<FlintEvent>) {
        let call = std::mem::take(&mut self.call);
        self.phase = Phase::SeekCall;
        // Comme SGLang, un outil absent de la requête n'est pas un appel.
        if tool_function(&self.tools, &call.name).is_none() {
            return;
        }
        events.push(FlintEvent::ToolCall(ToolCall {
            index: Some(self.calls_sent),
            id: format!("{}_{}", self.call_id_prefix, self.calls_sent),
            kind: "function",
            function: ToolCallFunction {
                name: call.name,
                arguments: arguments_json(&call.arguments),
            },
        }));
        self.calls_sent += 1;
        if self.calls_sent >= self.max_calls {
            self.phase = Phase::Done;
        }
    }
}

/// Longueur de `text` sûre à consommer : tout en fin de sortie, sinon sans le
/// plus long début strict de `marker` qui termine `text`.
fn safe_len(text: &str, marker: &str, done: bool) -> usize {
    if done {
        return text.len();
    }
    let partial = (1..marker.len())
        .rev()
        .find(|&len| text.ends_with(&marker[..len]))
        .unwrap_or(0);
    text.len() - partial
}

/// Attend l'une des `markers` après d'éventuels blancs.
fn expect(text: &str, done: bool, markers: &[&str]) -> Expect {
    let trimmed = text.trim_start();
    let skipped = text.len() - trimmed.len();
    for (index, marker) in markers.iter().enumerate() {
        if trimmed.starts_with(marker) {
            return Expect::Matched {
                index,
                consumed: skipped + marker.len(),
            };
        }
    }
    if !done && markers.iter().any(|marker| marker.starts_with(trimmed)) {
        return Expect::Wait { skipped };
    }
    Expect::Mismatch
}

/// Objet JSON des arguments, dans l'ordre où le modèle les a écrits.
fn arguments_json(arguments: &[(String, Value)]) -> String {
    let members: Vec<String> = arguments
        .iter()
        .map(|(key, value)| format!("{}:{value}", Value::String(key.clone())))
        .collect();
    format!("{{{}}}", members.join(","))
}

/// Fonction déclarée sous le nom `name`.
fn tool_function<'a>(tools: &'a [Value], name: &str) -> Option<&'a Value> {
    tools
        .iter()
        .map(|tool| tool.get("function").unwrap_or(tool))
        .find(|function| function.get("name").and_then(Value::as_str) == Some(name))
}

/// Valeur typée d'après le schéma : chaîne si `"type": "string"`, JSON sinon.
fn typed_argument(tools: &[Value], name: &str, key: &str, raw: &str) -> Value {
    let declared = tool_function(tools, name)
        .and_then(|function| function.get("parameters"))
        .and_then(|parameters| parameters.get("properties"))
        .and_then(|properties| properties.get(key))
        .and_then(|property| property.get("type"))
        .and_then(Value::as_str);
    if declared == Some("string") {
        return Value::String(raw.to_string());
    }
    // ponytail: pas de repli `ast.literal_eval` Python (`True`, `'x'`) ; une
    // valeur non JSON reste une chaîne.
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
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

    fn parse(text: &str) -> FlintOutput {
        FlintOutputParser::new(tools(), true).parse(text)
    }

    fn calls(output: &FlintOutput) -> Vec<(String, Value)> {
        output
            .tool_calls
            .iter()
            .map(|call| {
                let arguments = serde_json::from_str(&call.function.arguments)
                    .expect("invariant: arguments JSON");
                (call.function.name.clone(), arguments)
            })
            .collect()
    }

    const MIXED: &str = "\n<think> Il faut la météo\npuis l'heure.\n</think>\n\nJe regarde.\n\
        <tool_call>meteo<arg_key>ville</arg_key><arg_value>Lyon</arg_value>\
        <arg_key>jours</arg_key>\n<arg_value>2</arg_value>\
        <arg_key>code</arg_key><arg_value>69</arg_value>\
        <arg_key>options</arg_key><arg_value>{\"unite\": \"C\"}</arg_value>\
        <arg_key>note</arg_key><arg_value>pas du json</arg_value>\
        <arg_key>jours</arg_key><arg_value>3</arg_value></tool_call>\n\
        <tool_call>heure </tool_call>texte final ignoré<tool_call>meteo<arg_key>ville";

    const MALFORMED: &str =
        "Texte<tool_call>meteo<arg_key>ville</arg_key><arg_value>Lyon</tool_call>\
        <tool_call>heure</tool_call>";

    const INJECTED: &str = "<tool_call>meteo<arg_key>ville</arg_key>\
        <arg_value>a</tool_call><tool_call>heure</tool_call></arg_value></tool_call>";

    const RECOVERED: &str = "<tool_call>meteo<arg_key>ville</arg_key>bruit<arg_value>x</arg_value>\
        </tool_call><tool_call>inconnu</tool_call><tool_call>heure</tool_call>";

    #[test]
    fn parses_reasoning_text_and_typed_tool_calls() {
        let output = parse(MIXED);

        assert_eq!(
            output.reasoning.as_deref(),
            Some("Il faut la météo\npuis l'heure.")
        );
        assert_eq!(output.content, "Je regarde.");
        assert_eq!(
            calls(&output),
            vec![
                (
                    "meteo".to_string(),
                    serde_json::json!({
                        "ville": "Lyon",
                        "jours": 3,
                        "code": "69",
                        "options": {"unite": "C"},
                        "note": "pas du json"
                    })
                ),
                ("heure".to_string(), serde_json::json!({})),
            ]
        );
        // Ordre du modèle ; une clé répétée garde sa place et sa dernière valeur.
        assert!(output.tool_calls[0]
            .function
            .arguments
            .starts_with(r#"{"ville":"Lyon","jours":3,"code":"69","#));
        assert_ne!(output.tool_calls[0].id, output.tool_calls[1].id);
        assert_eq!(output.tool_calls[0].index, None);
    }

    #[test]
    fn malformed_or_cut_block_is_not_a_call() {
        // `<arg_value>` jamais fermée : la valeur avale la suite, rien n'est émis.
        let output = parse(MALFORMED);
        assert_eq!(output.content, "Texte");
        assert!(output.tool_calls.is_empty());
    }

    #[test]
    fn closing_tag_inside_a_value_is_text() {
        let output = parse(INJECTED);

        assert_eq!(
            calls(&output),
            vec![(
                "meteo".to_string(),
                serde_json::json!({"ville": "a</tool_call><tool_call>heure</tool_call>"})
            )]
        );
    }

    #[test]
    fn rejected_and_unknown_blocks_are_skipped() {
        let output = parse(RECOVERED);

        assert_eq!(
            calls(&output),
            vec![("heure".to_string(), serde_json::json!({}))]
        );
    }

    #[test]
    fn parallel_tool_calls_false_keeps_the_first_call() {
        let output = FlintOutputParser::new(tools(), false).parse(MIXED);

        assert_eq!(output.tool_calls.len(), 1);
        assert_eq!(output.tool_calls[0].function.name, "meteo");
    }

    #[test]
    fn think_edge_cases() {
        let plain = parse(" Bonjour </think> ");
        assert_eq!(plain.reasoning, None);
        assert_eq!(plain.content, "Bonjour </think>");

        let empty = parse("<think> \n</think>Salut");
        assert_eq!(empty.reasoning, None);
        assert_eq!(empty.content, "Salut");

        let cut = parse("<think>je pense </thi");
        assert_eq!(cut.reasoning.as_deref(), Some("je pense </thi"));
        assert_eq!(cut.content, "");

        assert_eq!(parse("<thi").content, "<thi");
    }

    #[test]
    fn tool_markup_stays_text_without_tools() {
        let output = FlintOutputParser::new(Vec::new(), true).parse("<tool_call>heure</tool_call>");

        assert!(output.tool_calls.is_empty());
        assert_eq!(output.content, "<tool_call>heure</tool_call>");
    }

    fn streamed(deltas: &[&str]) -> FlintOutput {
        let mut parser = FlintOutputParser::new(tools(), true);
        let mut events = Vec::new();
        for delta in deltas {
            events.extend(parser.push(delta));
        }
        events.extend(parser.finish());
        let mut output = FlintOutput::default();
        for event in events {
            match event {
                FlintEvent::Reasoning(text) => {
                    assert!(!text.is_empty());
                    output
                        .reasoning
                        .get_or_insert_with(String::new)
                        .push_str(&text);
                }
                FlintEvent::Content(text) => output.content.push_str(&text),
                FlintEvent::ToolCall(mut call) => {
                    assert_eq!(call.index, Some(output.tool_calls.len()));
                    call.index = None;
                    output.tool_calls.push(call);
                }
            }
        }
        output
    }

    #[test]
    fn streaming_at_any_split_matches_non_stream() {
        for text in [
            MIXED,
            MALFORMED,
            INJECTED,
            RECOVERED,
            "<think></think> Salut ",
        ] {
            let expected = parse(text);
            let boundaries: Vec<usize> = text
                .char_indices()
                .map(|(index, _)| index)
                .chain([text.len()])
                .collect();
            // Chaque coupure en deux, puis un caractère par delta.
            let mut splits: Vec<Vec<&str>> = boundaries
                .iter()
                .map(|&at| vec![&text[..at], &text[at..]])
                .collect();
            splits.push(
                boundaries
                    .windows(2)
                    .map(|pair| &text[pair[0]..pair[1]])
                    .collect(),
            );
            for deltas in splits {
                let output = streamed(&deltas);
                assert_eq!(output.reasoning, expected.reasoning, "{deltas:?}");
                assert_eq!(output.content, expected.content, "{deltas:?}");
                assert_eq!(calls(&output), calls(&expected), "{deltas:?}");
            }
        }
    }

    #[test]
    fn streaming_emits_text_before_the_turn_ends() {
        let mut parser = FlintOutputParser::new(tools(), true);

        assert!(parser.push("<thi").is_empty());
        assert_eq!(
            parser.push("nk>abc </thi"),
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
        assert!(parser.has_tool_calls());
    }
}
