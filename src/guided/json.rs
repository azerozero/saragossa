//! Automate JSON byte-level pour sorties structurées JSON.

use std::sync::{Arc, Mutex};

use crate::{InferError, Result, RustTokenizer};

use super::TokenConstraint;

/// Catalogue immutable des bytes visibles de chaque token.
#[derive(Debug)]
pub struct JsonTokenCatalog {
    tokens: Vec<TokenBytes>,
}

impl JsonTokenCatalog {
    /// Construit le catalogue depuis le tokenizer du modèle.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un token ne peut pas être décodé.
    pub fn from_tokenizer(tokenizer: &RustTokenizer) -> Result<Self> {
        let mut tokens = Vec::with_capacity(tokenizer.vocab_size());
        for token in 0..tokenizer.vocab_size() {
            let token_id = u32::try_from(token).map_err(|_| {
                InferError::Config(format!("token id hors u32 pour le tokenizer: {token}"))
            })?;
            tokens.push(TokenBytes {
                bytes: tokenizer.decode_token_bytes(token_id)?,
            });
        }
        Ok(Self { tokens })
    }

    fn token_bytes(&self, token: usize) -> Option<&[u8]> {
        self.tokens.get(token).map(|token| token.bytes.as_slice())
    }

    #[cfg(test)]
    fn from_token_bytes_for_tests(tokens: &[&[u8]]) -> Self {
        Self {
            tokens: tokens
                .iter()
                .map(|bytes| TokenBytes {
                    bytes: bytes.to_vec(),
                })
                .collect(),
        }
    }
}

#[derive(Debug)]
struct TokenBytes {
    bytes: Vec<u8>,
}

/// Contraint un ou plusieurs objets JSON racine complets.
#[derive(Debug)]
pub struct JsonTokenConstraint {
    catalog: Arc<JsonTokenCatalog>,
    eot_tokens: Vec<usize>,
    state: Mutex<JsonAutomaton>,
}

impl JsonTokenConstraint {
    /// Crée une contrainte JSON pour une requête.
    ///
    /// `repeat=false` impose un seul objet racine (`json_object`). `repeat=true`
    /// autorise une suite NDJSON saragossa (`json_lines`) séparée par `\n`.
    #[must_use]
    pub fn new(catalog: Arc<JsonTokenCatalog>, eot_token_ids: &[usize], repeat: bool) -> Self {
        let mut eot_tokens = eot_token_ids.to_vec();
        eot_tokens.sort_unstable();
        eot_tokens.dedup();
        Self {
            catalog,
            eot_tokens,
            state: Mutex::new(JsonAutomaton::new(repeat)),
        }
    }

    fn state_snapshot(&self) -> Result<JsonAutomaton> {
        self.state
            .lock()
            .map_err(|_| InferError::Config("structured output JSON: état empoisonné".to_string()))
            .map(|state| state.clone())
    }

    fn is_eot(&self, token: usize) -> bool {
        self.eot_tokens.binary_search(&token).is_ok()
    }
}

impl TokenConstraint for JsonTokenConstraint {
    fn mask_logits(&self, logits: &mut [f32]) -> Result<()> {
        let state = self.state_snapshot()?;
        let first_bytes = state.allowed_first_bytes();
        let final_state = state.is_final();
        let mut allowed = 0_usize;

        for (token, logit) in logits.iter_mut().enumerate() {
            let token_allowed = if self.is_eot(token) {
                final_state
            } else {
                self.catalog
                    .token_bytes(token)
                    .filter(|bytes| !bytes.is_empty())
                    .is_some_and(|bytes| {
                        first_bytes[usize::from(bytes[0])] && state.can_accept_bytes(bytes)
                    })
            };
            if token_allowed {
                allowed = allowed.saturating_add(1);
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }

        if allowed == 0 {
            return Err(InferError::Config(
                "structured output JSON: aucun token admissible pour prolonger le préfixe"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn accept_token(&self, token: usize) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| {
            InferError::Config("structured output JSON: état empoisonné".to_string())
        })?;
        if self.is_eot(token) {
            if state.is_final() {
                return Ok(());
            }
            return Err(InferError::Config(
                "structured output JSON: EOT refusé avant l'objet racine fermé".to_string(),
            ));
        }
        let Some(bytes) = self.catalog.token_bytes(token) else {
            return Err(InferError::Config(format!(
                "structured output JSON: token {token} hors catalogue"
            )));
        };
        if bytes.is_empty() || !state.advance_bytes(bytes) {
            return Err(InferError::Config(format!(
                "structured output JSON: token {token} inadmissible"
            )));
        }
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.is_final())
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug)]
struct JsonAutomaton {
    mode: Mode,
    stack: Vec<Container>,
    repeat: bool,
    completed_root: bool,
}

impl JsonAutomaton {
    fn new(repeat: bool) -> Self {
        Self {
            mode: Mode::BeforeRoot,
            stack: Vec::new(),
            repeat,
            completed_root: false,
        }
    }

    fn is_final(&self) -> bool {
        if self.mode == Mode::AfterRoot && self.stack.is_empty() {
            return true;
        }
        self.repeat && self.completed_root && self.mode == Mode::BeforeRoot && self.stack.is_empty()
    }

    fn can_accept_bytes(&self, bytes: &[u8]) -> bool {
        let mut state = self.clone();
        state.advance_bytes(bytes)
    }

    fn advance_bytes(&mut self, bytes: &[u8]) -> bool {
        for byte in bytes {
            if !self.consume_byte(*byte) {
                return false;
            }
        }
        true
    }

    fn allowed_first_bytes(&self) -> [bool; 256] {
        let mut allowed = [false; 256];
        for byte in u8::MIN..=u8::MAX {
            let mut state = self.clone();
            allowed[usize::from(byte)] = state.consume_byte(byte);
        }
        allowed
    }

    fn consume_byte(&mut self, byte: u8) -> bool {
        let mut current = Some(byte);
        while let Some(byte) = current.take() {
            match self.mode {
                Mode::BeforeRoot => {
                    if is_ws(byte) {
                        if self.repeat {
                            return false;
                        }
                        continue;
                    }
                    if byte == b'{' {
                        self.open_container(Container::Object);
                        continue;
                    }
                    return false;
                }
                Mode::AfterRoot => {
                    if self.repeat {
                        if byte == b'\n' {
                            self.mode = Mode::BeforeRoot;
                            continue;
                        }
                        return false;
                    }
                    if is_ws(byte) {
                        continue;
                    }
                    return false;
                }
                Mode::ObjectKeyOrEnd => {
                    if is_ws(byte) {
                        continue;
                    }
                    match byte {
                        b'}' => {
                            if !self.close_container(Container::Object) {
                                return false;
                            }
                        }
                        b'"' => {
                            self.mode = Mode::String {
                                after: StringAfter::ObjectKey,
                                escape: StringEscape::None,
                            }
                        }
                        _ => return false,
                    }
                }
                Mode::ObjectKey => {
                    if is_ws(byte) {
                        continue;
                    }
                    if byte != b'"' {
                        return false;
                    }
                    self.mode = Mode::String {
                        after: StringAfter::ObjectKey,
                        escape: StringEscape::None,
                    };
                }
                Mode::ObjectColon => {
                    if is_ws(byte) {
                        continue;
                    }
                    if byte != b':' {
                        return false;
                    }
                    self.mode = Mode::ObjectValue;
                }
                Mode::ObjectValue => {
                    if !self.consume_value_start(byte) {
                        return false;
                    }
                }
                Mode::ObjectCommaOrEnd => {
                    if is_ws(byte) {
                        continue;
                    }
                    match byte {
                        b',' => self.mode = Mode::ObjectKey,
                        b'}' => {
                            if !self.close_container(Container::Object) {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
                Mode::ArrayValueOrEnd => {
                    if is_ws(byte) {
                        continue;
                    }
                    if byte == b']' {
                        if !self.close_container(Container::Array) {
                            return false;
                        }
                        continue;
                    }
                    if !self.consume_value_start(byte) {
                        return false;
                    }
                }
                Mode::ArrayValue => {
                    if !self.consume_value_start(byte) {
                        return false;
                    }
                }
                Mode::ArrayCommaOrEnd => {
                    if is_ws(byte) {
                        continue;
                    }
                    match byte {
                        b',' => self.mode = Mode::ArrayValue,
                        b']' => {
                            if !self.close_container(Container::Array) {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
                Mode::String { after, escape } => {
                    if !self.consume_string_byte(byte, after, escape) {
                        return false;
                    }
                }
                Mode::Number(number) => match consume_number_byte(number, byte) {
                    NumberConsume::Continue(next) => self.mode = Mode::Number(next),
                    NumberConsume::FinishAndReprocess => {
                        self.finish_value();
                        current = Some(byte);
                    }
                    NumberConsume::Invalid => return false,
                },
                Mode::Literal { kind, pos } => {
                    let literal = kind.bytes();
                    let Some(expected) = literal.get(pos).copied() else {
                        return false;
                    };
                    if byte != expected {
                        return false;
                    }
                    let next = pos.saturating_add(1);
                    if next == literal.len() {
                        self.finish_value();
                    } else {
                        self.mode = Mode::Literal { kind, pos: next };
                    }
                }
            }
        }
        true
    }

    fn consume_value_start(&mut self, byte: u8) -> bool {
        if is_ws(byte) {
            return true;
        }
        match byte {
            b'{' => self.open_container(Container::Object),
            b'[' => self.open_container(Container::Array),
            b'"' => {
                self.mode = Mode::String {
                    after: StringAfter::Value,
                    escape: StringEscape::None,
                };
            }
            b'-' | b'0'..=b'9' => {
                let NumberConsume::Continue(next) = consume_number_byte(NumberState::Start, byte)
                else {
                    return false;
                };
                self.mode = Mode::Number(next);
            }
            b't' => {
                self.mode = Mode::Literal {
                    kind: LiteralKind::True,
                    pos: 1,
                }
            }
            b'f' => {
                self.mode = Mode::Literal {
                    kind: LiteralKind::False,
                    pos: 1,
                }
            }
            b'n' => {
                self.mode = Mode::Literal {
                    kind: LiteralKind::Null,
                    pos: 1,
                }
            }
            _ => return false,
        }
        true
    }

    fn consume_string_byte(&mut self, byte: u8, after: StringAfter, escape: StringEscape) -> bool {
        match escape {
            StringEscape::None => match byte {
                b'"' => match after {
                    StringAfter::ObjectKey => self.mode = Mode::ObjectColon,
                    StringAfter::Value => self.finish_value(),
                },
                b'\\' => {
                    self.mode = Mode::String {
                        after,
                        escape: StringEscape::Escaped,
                    };
                }
                0x00..=0x1F => return false,
                _ => {}
            },
            StringEscape::Escaped => match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.mode = Mode::String {
                        after,
                        escape: StringEscape::None,
                    };
                }
                b'u' => {
                    self.mode = Mode::String {
                        after,
                        escape: StringEscape::Unicode {
                            value: 0,
                            remaining: 4,
                            expect_low: false,
                        },
                    };
                }
                _ => return false,
            },
            StringEscape::Unicode {
                value,
                remaining,
                expect_low,
            } => {
                let Some(digit) = hex_value(byte) else {
                    return false;
                };
                let value = (value << 4) | u16::from(digit);
                let remaining = remaining.saturating_sub(1);
                let escape = if remaining > 0 {
                    StringEscape::Unicode {
                        value,
                        remaining,
                        expect_low,
                    }
                } else if expect_low {
                    // Complète une paire : la deuxième moitié doit être un
                    // surrogate bas, sinon serde_json (l'oracle de validité)
                    // refuse la chaîne.
                    if !is_low_surrogate(value) {
                        return false;
                    }
                    StringEscape::None
                } else if is_high_surrogate(value) {
                    StringEscape::PairBackslash
                } else if is_low_surrogate(value) {
                    // Surrogate bas isolé : invalide hors paire.
                    return false;
                } else {
                    StringEscape::None
                };
                self.mode = Mode::String { after, escape };
            }
            StringEscape::PairBackslash => {
                if byte != b'\\' {
                    return false;
                }
                self.mode = Mode::String {
                    after,
                    escape: StringEscape::PairMarker,
                };
            }
            StringEscape::PairMarker => {
                if byte != b'u' {
                    return false;
                }
                self.mode = Mode::String {
                    after,
                    escape: StringEscape::Unicode {
                        value: 0,
                        remaining: 4,
                        expect_low: true,
                    },
                };
            }
        }
        true
    }

    fn open_container(&mut self, container: Container) {
        self.stack.push(container);
        self.mode = match container {
            Container::Object => Mode::ObjectKeyOrEnd,
            Container::Array => Mode::ArrayValueOrEnd,
        };
    }

    fn close_container(&mut self, expected: Container) -> bool {
        if self.stack.pop() != Some(expected) {
            return false;
        }
        self.finish_value();
        true
    }

    fn finish_value(&mut self) {
        self.mode = match self.stack.last().copied() {
            Some(Container::Object) => Mode::ObjectCommaOrEnd,
            Some(Container::Array) => Mode::ArrayCommaOrEnd,
            None => {
                self.completed_root = true;
                Mode::AfterRoot
            }
        };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    BeforeRoot,
    ObjectKeyOrEnd,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectCommaOrEnd,
    ArrayValueOrEnd,
    ArrayValue,
    ArrayCommaOrEnd,
    String {
        after: StringAfter,
        escape: StringEscape,
    },
    Number(NumberState),
    Literal {
        kind: LiteralKind,
        pos: usize,
    },
    AfterRoot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StringAfter {
    ObjectKey,
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StringEscape {
    None,
    Escaped,
    /// Collecte les 4 hexa d'un `\uXXXX`. `expect_low` impose que la valeur
    /// obtenue soit un surrogate bas (complétion d'une paire déjà ouverte).
    Unicode {
        value: u16,
        remaining: u8,
        expect_low: bool,
    },
    /// Après un surrogate haut : exige le `\` d'un `\uXXXX` bas apparié.
    PairBackslash,
    /// Après le `\` d'une paire : exige le `u`.
    PairMarker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiteralKind {
    True,
    False,
    Null,
}

impl LiteralKind {
    fn bytes(self) -> &'static [u8] {
        match self {
            Self::True => b"true",
            Self::False => b"false",
            Self::Null => b"null",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumberState {
    Start,
    AfterMinus,
    Zero,
    Int,
    AfterDot,
    Frac,
    AfterExp,
    AfterExpSign,
    Exp,
}

enum NumberConsume {
    Continue(NumberState),
    FinishAndReprocess,
    Invalid,
}

fn consume_number_byte(state: NumberState, byte: u8) -> NumberConsume {
    match state {
        NumberState::Start => match byte {
            b'-' => NumberConsume::Continue(NumberState::AfterMinus),
            b'0' => NumberConsume::Continue(NumberState::Zero),
            b'1'..=b'9' => NumberConsume::Continue(NumberState::Int),
            _ => NumberConsume::Invalid,
        },
        NumberState::AfterMinus => match byte {
            b'0' => NumberConsume::Continue(NumberState::Zero),
            b'1'..=b'9' => NumberConsume::Continue(NumberState::Int),
            _ => NumberConsume::Invalid,
        },
        NumberState::Zero => match byte {
            b'.' => NumberConsume::Continue(NumberState::AfterDot),
            b'e' | b'E' => NumberConsume::Continue(NumberState::AfterExp),
            _ if is_value_delimiter(byte) => NumberConsume::FinishAndReprocess,
            _ => NumberConsume::Invalid,
        },
        NumberState::Int => match byte {
            b'0'..=b'9' => NumberConsume::Continue(NumberState::Int),
            b'.' => NumberConsume::Continue(NumberState::AfterDot),
            b'e' | b'E' => NumberConsume::Continue(NumberState::AfterExp),
            _ if is_value_delimiter(byte) => NumberConsume::FinishAndReprocess,
            _ => NumberConsume::Invalid,
        },
        NumberState::AfterDot => match byte {
            b'0'..=b'9' => NumberConsume::Continue(NumberState::Frac),
            _ => NumberConsume::Invalid,
        },
        NumberState::Frac => match byte {
            b'0'..=b'9' => NumberConsume::Continue(NumberState::Frac),
            b'e' | b'E' => NumberConsume::Continue(NumberState::AfterExp),
            _ if is_value_delimiter(byte) => NumberConsume::FinishAndReprocess,
            _ => NumberConsume::Invalid,
        },
        NumberState::AfterExp => match byte {
            b'+' | b'-' => NumberConsume::Continue(NumberState::AfterExpSign),
            b'0'..=b'9' => NumberConsume::Continue(NumberState::Exp),
            _ => NumberConsume::Invalid,
        },
        NumberState::AfterExpSign => match byte {
            b'0'..=b'9' => NumberConsume::Continue(NumberState::Exp),
            _ => NumberConsume::Invalid,
        },
        NumberState::Exp => match byte {
            b'0'..=b'9' => NumberConsume::Continue(NumberState::Exp),
            _ if is_value_delimiter(byte) => NumberConsume::FinishAndReprocess,
            _ => NumberConsume::Invalid,
        },
    }
}

fn is_value_delimiter(byte: u8) -> bool {
    is_ws(byte) || matches!(byte, b',' | b']' | b'}')
}

fn is_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_high_surrogate(value: u16) -> bool {
    (0xD800..=0xDBFF).contains(&value)
}

fn is_low_surrogate(value: u16) -> bool {
    (0xDC00..=0xDFFF).contains(&value)
}

#[cfg(test)]
#[path = "json_tests.rs"]
mod tests;
