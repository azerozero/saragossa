//! Rendu ChatML Qwen pour le backend Rust pur.

/// Marqueur d'ouverture de message ChatML Qwen.
pub const QWEN_IM_START: &str = "<|im_start|>";

/// Marqueur de fermeture de message ChatML Qwen.
pub const QWEN_IM_END: &str = "<|im_end|>";

/// Bloc de raisonnement vide injecté quand le thinking Qwen est désactivé.
pub const QWEN_EMPTY_THINK_BLOCK: &str = "<think>\n\n</think>\n\n";

/// Message minimal consommé par le rendu ChatML Qwen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateMessage {
    /// Rôle ChatML (`system`, `user`, `assistant`, `tool`).
    pub role: String,
    /// Contenu textuel du message.
    pub content: Option<String>,
}

impl ChatTemplateMessage {
    /// Construit un message ChatML.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
        }
    }
}

/// Rend des messages en ChatML Qwen.
pub fn render_qwen_chatml(
    messages: &[ChatTemplateMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    for message in messages {
        out.push_str(QWEN_IM_START);
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(message.content.as_deref().unwrap_or(""));
        out.push_str(QWEN_IM_END);
        out.push('\n');
    }
    if add_generation_prompt {
        out.push_str(QWEN_IM_START);
        out.push_str("assistant\n");
        if !enable_thinking {
            out.push_str(QWEN_EMPTY_THINK_BLOCK);
        }
    }
    out
}

/// Prépare un tour assistant passé pour préserver le préfixe Qwen.
pub fn qwen_assistant_history_content(text: &str, enable_thinking: bool) -> String {
    if enable_thinking || text.contains("</think>") {
        text.to_string()
    } else {
        format!("{QWEN_EMPTY_THINK_BLOCK}{text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_system_user_with_generation_prompt() {
        let messages = vec![
            ChatTemplateMessage::new("system", "Réponds en français."),
            ChatTemplateMessage::new("user", "Bonjour"),
        ];

        let prompt = render_qwen_chatml(&messages, true, true);

        assert_eq!(
            prompt,
            "<|im_start|>system\nRéponds en français.<|im_end|>\n\
             <|im_start|>user\nBonjour<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn thinking_disabled_injects_empty_think_block() {
        let messages = vec![ChatTemplateMessage::new("user", "Salut")];

        let prompt = render_qwen_chatml(&messages, true, false);

        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn thinking_enabled_leaves_assistant_turn_open() {
        let messages = vec![ChatTemplateMessage::new("user", "Salut")];

        let prompt = render_qwen_chatml(&messages, true, true);

        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(!prompt.contains("<think>"));
    }

    #[test]
    fn assistant_history_preserves_existing_think_block() {
        let content = qwen_assistant_history_content("<think>\ntrace\n</think>\nRéponse", false);

        assert_eq!(content, "<think>\ntrace\n</think>\nRéponse");
    }

    #[test]
    fn assistant_history_prefixes_empty_think_when_disabled() {
        let content = qwen_assistant_history_content("Réponse", false);

        assert_eq!(content, "<think>\n\n</think>\n\nRéponse");
    }
}
