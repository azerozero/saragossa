//! Curseurs persistants : aucun préfixe déjà lu n'est rescanné entre deltas.
use super::ToolFamily;

#[derive(Default)]
pub(super) struct FrameScan {
    cursor: usize,
    mode: FrameMode,
    #[cfg(test)]
    pub(super) steps: usize,
}

#[derive(Default)]
enum FrameMode {
    #[default]
    Body,
    ParameterHeader,
    ParameterValue,
    GemmaString,
}

impl FrameScan {
    pub(super) fn end(&mut self, text: &str, family: ToolFamily, close: &str) -> Option<usize> {
        while self.cursor < text.len() {
            #[cfg(test)]
            {
                self.steps += 1;
            }
            let rest = &text[self.cursor..];
            let marker = match self.mode {
                FrameMode::ParameterHeader => ">",
                FrameMode::ParameterValue => "</parameter>",
                FrameMode::GemmaString => "<|\"|>",
                FrameMode::Body => {
                    if rest.starts_with(close) {
                        return Some(self.cursor);
                    }
                    if close.starts_with(rest) {
                        return None;
                    }
                    if family == ToolFamily::Qwen {
                        "<parameter="
                    } else {
                        "<|\"|>"
                    }
                }
            };
            if rest.starts_with(marker) {
                self.cursor += marker.len();
                self.mode = match self.mode {
                    FrameMode::Body if family == ToolFamily::Qwen => FrameMode::ParameterHeader,
                    FrameMode::Body => FrameMode::GemmaString,
                    FrameMode::ParameterHeader => FrameMode::ParameterValue,
                    _ => FrameMode::Body,
                };
            } else if marker.starts_with(rest) {
                return None;
            } else {
                self.cursor += rest.chars().next()?.len_utf8();
            }
        }
        None
    }
}

/// Contexte littéral Markdown conservé même si les délimiteurs sont fractionnés.
/// Les citations ambiguës restent du texte (choix conservateur).
#[derive(Default)]
pub(super) struct LiteralContext {
    code: Option<(char, usize, bool)>,
    run: Option<(char, usize, bool)>,
    quote: Option<char>,
    line_content: bool,
    indentation: usize,
    block_quote: bool,
    escaped: bool,
    previous_word: bool,
    closing_fence: bool,
}

impl LiteralContext {
    fn settle_run(&mut self, next: char) {
        if let Some((ch, count, line_start)) = self.run {
            if next == ch {
                return;
            }
            match self.code {
                Some((delimiter, size, true))
                    if delimiter == ch && count >= size && line_start && next.is_whitespace() =>
                {
                    if next == '\n' {
                        self.code = None;
                    } else {
                        self.closing_fence = true;
                    }
                }
                Some((delimiter, size, false)) if delimiter == ch && count == size => {
                    self.code = None;
                }
                None if ch == '`' || (count >= 3 && line_start) => {
                    self.code = Some((ch, count, count >= 3 && line_start));
                }
                _ => {}
            }
            self.run = None;
        }
    }

    pub(super) fn is_literal_before(&mut self, next: &str) -> bool {
        if let Some(ch) = next.chars().next() {
            self.settle_run(ch);
        }
        self.code.is_some()
            || self.quote.is_some()
            || self.block_quote
            || self.indentation >= 4
            || self.escaped
    }

    pub(super) fn push(&mut self, ch: char) {
        self.settle_run(ch);
        if self.closing_fence {
            if ch == '\n' {
                self.code = None;
            }
            if ch == '\n' || !ch.is_whitespace() {
                self.closing_fence = false;
            }
        }
        if ch == '\n' {
            self.line_content = false;
            self.indentation = 0;
            self.block_quote = false;
            self.previous_word = false;
            self.escaped = false;
            return;
        }
        if self.escaped {
            self.escaped = false;
        } else if ch == '\\' {
            self.escaped = true;
        } else if matches!(ch, '`' | '~') && self.quote.is_none() {
            let (count, line_start) = self.run.map_or(
                (0, !self.line_content && self.indentation < 4),
                |(_, count, line_start)| (count, line_start),
            );
            self.run = Some((ch, count.saturating_add(1), line_start));
        } else if self.code.is_none() {
            if self.quote == Some(ch) {
                self.quote = None;
            } else if self.quote.is_none() {
                self.quote = match ch {
                    '"' | '“' => Some(if ch == '“' { '”' } else { '"' }),
                    '«' => Some('»'),
                    '\'' | '‘' if !self.previous_word => {
                        Some(if ch == '‘' { '’' } else { '\'' })
                    }
                    _ => None,
                };
            }
        }
        if !self.line_content {
            match ch {
                ' ' => self.indentation = self.indentation.saturating_add(1),
                '\t' => self.indentation = self.indentation.saturating_add(4),
                '>' => {
                    self.block_quote = true;
                    self.line_content = true;
                }
                _ => self.line_content = true,
            }
        }
        self.previous_word = ch.is_alphanumeric();
    }
}
