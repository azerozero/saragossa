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
/// Les préfixes espaces/tabulations/listes/citations n'empêchent jamais l'ouverture.
/// Toute indentation et toute clôture ambiguë restent littérales.
#[derive(Default)]
pub(super) struct LiteralContext {
    code: Option<(char, usize, bool)>,
    run: Option<(char, usize, bool)>,
    quote: Option<char>,
    line_content: bool,
    indented: bool,
    raw_indent: usize,
    container_prefix: bool,
    indentation_tab: bool,
    block_quote: bool,
    escaped: bool,
    previous_word: bool,
    closing_fence: bool,
    list_prefix: ListPrefix,
    list_marker: bool,
    quote_depth: usize,
    fence_quote_depth: usize,
}

// Préfixes conservateurs, sans calcul des colonnes ni des conteneurs CommonMark.
#[derive(Default)]
enum ListPrefix {
    #[default]
    None,
    Digits,
}

impl LiteralContext {
    fn settle_run(&mut self, next: char) {
        if let Some((ch, count, line_start)) = self.run {
            if next == ch {
                return;
            }
            match self.code {
                Some((delimiter, size, true))
                    if delimiter == ch
                        && count >= size
                        && line_start
                        && self.raw_indent <= 3
                        && !self.indentation_tab
                        && !self.list_marker
                        && self.quote_depth == self.fence_quote_depth
                        && matches!(next, ' ' | '\n') =>
                {
                    // Un marqueur de liste ou une autre profondeur de citation
                    // peut être le contenu du bloc, pas sa clôture.
                    if next == '\n' {
                        self.code = None;
                    } else {
                        self.closing_fence = true;
                    }
                }
                None | Some((_, _, false)) if count >= 3 && line_start => {
                    self.code = Some((ch, count, true));
                    self.fence_quote_depth = self.quote_depth;
                    self.quote = None;
                }
                Some((delimiter, size, false)) if delimiter == ch && count == size => {
                    self.code = None;
                }
                None if ch == '`' => {
                    self.code = Some((ch, count, false));
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
            || self.indented
            || self.list_marker
            || self.escaped
    }

    pub(super) fn push(&mut self, ch: char) {
        self.settle_run(ch);
        if self.closing_fence {
            if ch == '\n' {
                self.code = None;
            }
            if ch != ' ' {
                self.closing_fence = false;
            }
        }
        if ch == '\n' {
            self.list_prefix = ListPrefix::None;
            self.line_content = false;
            self.indented = false;
            self.raw_indent = 0;
            self.container_prefix = false;
            self.indentation_tab = false;
            self.list_marker = false;
            self.block_quote = false;
            self.quote_depth = 0;
            self.previous_word = false;
            self.escaped = false;
            return;
        }
        if self.escaped {
            self.escaped = false;
        } else if ch == '\\' {
            self.escaped = true;
        } else if matches!(ch, '`' | '~')
            && (self.quote.is_none()
                || !self.line_content
                || self.run.is_some_and(|(_, _, line_start)| line_start))
        {
            let (count, line_start) = self
                .run
                .map_or((0, !self.line_content), |(_, count, line_start)| {
                    (count, line_start)
                });
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
            // Compter les espaces bruts avant tout conteneur, sans retirer
            // son indentation. Une tabulation du préfixe interdit la clôture,
            // mais ces contraintes ne doivent jamais empêcher l'ouverture.
            match ch {
                ' ' if !self.container_prefix => {
                    self.raw_indent = self.raw_indent.saturating_add(1);
                }
                '\t' => self.indentation_tab = true,
                ' ' => {}
                _ => self.container_prefix = true,
            }
            self.list_prefix = match (&self.list_prefix, ch) {
                (ListPrefix::None, ' ' | '\t') => {
                    // Toute indentation est ambiguë : ne jamais promouvoir son contenu.
                    self.indented = true;
                    ListPrefix::None
                }
                (ListPrefix::None, '>') => {
                    self.block_quote = true;
                    self.quote_depth += 1;
                    ListPrefix::None
                }
                (ListPrefix::None, '-' | '+' | '*') => {
                    self.list_marker = true;
                    ListPrefix::None
                }
                (ListPrefix::None | ListPrefix::Digits, '0'..='9') => ListPrefix::Digits,
                (ListPrefix::Digits, '.' | ')') => {
                    self.list_marker = true;
                    ListPrefix::None
                }
                _ => {
                    self.line_content = true;
                    ListPrefix::None
                }
            };
        }
        self.previous_word = ch.is_alphanumeric();
    }
}
