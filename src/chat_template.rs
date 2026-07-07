//! Rendu des templates de chat (ChatML Qwen, tours Gemma) du backend Rust pur.

/// Marqueur d'ouverture de message ChatML Qwen.
pub const QWEN_IM_START: &str = "<|im_start|>";

/// Marqueur de fermeture de message ChatML Qwen.
pub const QWEN_IM_END: &str = "<|im_end|>";

/// Marqueur d'ouverture de tour Gemma.
pub const GEMMA_START_OF_TURN: &str = "<start_of_turn>";

/// Marqueur de fermeture de tour Gemma.
pub const GEMMA_END_OF_TURN: &str = "<end_of_turn>";

/// Marqueur d'ouverture de tour Gemma 4.
pub const GEMMA4_START_OF_TURN: &str = "<|turn>";

/// Marqueur de fermeture de tour Gemma 4.
pub const GEMMA4_END_OF_TURN: &str = "<turn|>";

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

/// Rend des messages au format chat Gemma (`<start_of_turn>…<end_of_turn>`).
///
/// Suit le chat_template officiel Gemma 3 : un message `system` en tête est
/// absorbé comme préfixe (suivi d'une ligne vide) du premier tour rendu, le
/// rôle `assistant` est rendu `model`, les contenus des tours sont trimmés
/// (le préfixe system ne l'est pas, comme le template Jinja). Le BOS n'est
/// PAS rendu ici : il vient du post-processor du tokenizer
/// ([`crate::ModelAssets::encode_prompt_with_special`]), là où mlx_lm
/// l'insère via `{{ bos_token }}` puis encode sans tokens spéciaux.
pub fn render_gemma_chat(messages: &[ChatTemplateMessage], add_generation_prompt: bool) -> String {
    let (first_user_prefix, turns) = match messages.split_first() {
        Some((first, rest)) if first.role == "system" => {
            let prefix = format!("{}\n\n", first.content.as_deref().unwrap_or(""));
            (prefix, rest)
        }
        _ => (String::new(), messages),
    };
    let mut out = String::new();
    for (index, message) in turns.iter().enumerate() {
        let role = if message.role == "assistant" {
            "model"
        } else {
            message.role.as_str()
        };
        out.push_str(GEMMA_START_OF_TURN);
        out.push_str(role);
        out.push('\n');
        if index == 0 {
            out.push_str(&first_user_prefix);
        }
        out.push_str(message.content.as_deref().unwrap_or("").trim());
        out.push_str(GEMMA_END_OF_TURN);
        out.push('\n');
    }
    if add_generation_prompt {
        out.push_str(GEMMA_START_OF_TURN);
        out.push_str("model\n");
    }
    out
}

/// Rend des messages au format chat Gemma 4.
///
/// Suit le `chat_template.jinja` des checkpoints Gemma 4 MLX : BOS littéral,
/// tours `<|turn>role\n...<turn|>`, rôle assistant rendu `model`, et canal
/// thought vide quand le thinking est désactivé.
pub fn render_gemma4_chat(
    messages: &[ChatTemplateMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::from("<bos>");
    for message in messages {
        let role = if message.role == "assistant" {
            "model"
        } else {
            message.role.as_str()
        };
        out.push_str(GEMMA4_START_OF_TURN);
        out.push_str(role);
        out.push('\n');
        out.push_str(message.content.as_deref().unwrap_or("").trim());
        out.push_str(GEMMA4_END_OF_TURN);
        out.push('\n');
    }
    if add_generation_prompt {
        out.push_str(GEMMA4_START_OF_TURN);
        out.push_str("model\n");
        if !enable_thinking {
            out.push_str("<|channel>thought\n<channel|>");
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
    fn gemma_renders_user_turn_with_generation_prompt() {
        let messages = vec![ChatTemplateMessage::new("user", "Bonjour")];

        let prompt = render_gemma_chat(&messages, true);

        assert_eq!(
            prompt,
            "<start_of_turn>user\nBonjour<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_absorbs_system_message_into_first_turn() {
        let messages = vec![
            ChatTemplateMessage::new("system", "Réponds en français."),
            ChatTemplateMessage::new("user", "Bonjour"),
        ];

        let prompt = render_gemma_chat(&messages, true);

        assert_eq!(
            prompt,
            "<start_of_turn>user\nRéponds en français.\n\nBonjour<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_renders_assistant_history_as_model_and_trims_contents() {
        let messages = vec![
            ChatTemplateMessage::new("user", "  Bonjour  "),
            ChatTemplateMessage::new("assistant", "Salut !\n"),
            ChatTemplateMessage::new("user", "Ça va ?"),
        ];

        let prompt = render_gemma_chat(&messages, true);

        assert_eq!(
            prompt,
            "<start_of_turn>user\nBonjour<end_of_turn>\n\
             <start_of_turn>model\nSalut !<end_of_turn>\n\
             <start_of_turn>user\nÇa va ?<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_without_generation_prompt_closes_last_turn() {
        let messages = vec![ChatTemplateMessage::new("user", "Bonjour")];

        let prompt = render_gemma_chat(&messages, false);

        assert_eq!(prompt, "<start_of_turn>user\nBonjour<end_of_turn>\n");
    }

    #[test]
    fn gemma4_renders_turns_with_bos_and_empty_thought_channel() {
        let messages = vec![
            ChatTemplateMessage::new("system", "Réponds en français."),
            ChatTemplateMessage::new("user", "Bonjour"),
        ];

        let prompt = render_gemma4_chat(&messages, true, false);

        assert_eq!(
            prompt,
            "<bos><|turn>system\nRéponds en français.<turn|>\n\
             <|turn>user\nBonjour<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>"
        );
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
