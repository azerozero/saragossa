//! Rendu des templates de chat (ChatML Qwen, tours Gemma, flint) du backend Rust pur.

mod python_json;

use self::python_json::{json_string, python_json, OrderedObject};

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
    /// Raisonnement d'un tour assistant passé (`reasoning_content` OpenAI).
    /// Seul le rendu flint l'utilise.
    pub reasoning_content: Option<String>,
    /// Appels d'outils OpenAI d'un tour assistant passé (`tool_calls`), un
    /// texte JSON brut par appel pour garder l'ordre des clés. Seul le rendu
    /// flint les utilise.
    pub tool_calls: Vec<String>,
}

impl ChatTemplateMessage {
    /// Construit un message ChatML.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: Vec::new(),
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
        let (role, content) = if message.role == "tool" {
            (
                "user",
                qwen_tool_response(message.content.as_deref().unwrap_or("")),
            )
        } else {
            (
                message.role.as_str(),
                message.content.as_deref().unwrap_or("").to_string(),
            )
        };
        out.push_str(QWEN_IM_START);
        out.push_str(role);
        out.push('\n');
        out.push_str(&content);
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

fn qwen_tool_response(content: &str) -> String {
    if content.contains("<tool_response>") {
        return content.to_string();
    }
    format!("<tool_response>\n{}\n</tool_response>", content.trim())
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

/// Rend des messages au format chat flint (`chat_template.jinja` du modèle).
///
/// Tours collés sans séparateur : `<|system|>`, `<|developer|>`, `<|user|>`,
/// `<|assistant|>` puis le contenu ; un résultat d'outil devient
/// `<|observation|><tool_response>…</tool_response>`. Comme le macro
/// `flint_text` du template, les contenus non assistant perdent `\0`, `<think>`
/// et `</think>`, et leurs marqueurs de rôle sont échappés en entités HTML ;
/// le contenu assistant passe tel quel, précédé de son `reasoning_content` en
/// `<think>…</think>` et suivi de ses appels d'outils XML. Pas de BOS
/// (`add_bos_token` faux).
///
/// `tools` porte le texte JSON brut de chaque outil de la requête. Non vide,
/// il fait rendre le préambule `# Tools`, qui précède le premier message
/// non `system` (ou clôt une conversation qui n'en a pas). Le rendu suit le
/// Jinja tel que transformers l'exécute (`trim_blocks` et `tojson` à
/// séparateurs `", "` / `": "`) : les signatures se suivent sans saut de ligne
/// entre `<tools>` et `</tools>`.
pub fn render_flint_chat(
    messages: &[ChatTemplateMessage],
    tools: &[String],
    add_generation_prompt: bool,
) -> String {
    let preamble = flint_tool_preamble(tools);
    let mut tools_pending = true;
    let mut out = String::new();
    for message in messages {
        if tools_pending && message.role != "system" {
            out.push_str(&preamble);
            tools_pending = false;
        }
        let content = message.content.as_deref().unwrap_or("");
        match message.role.as_str() {
            "system" | "developer" | "user" => {
                out.push_str("<|");
                out.push_str(&message.role);
                out.push_str("|>");
                out.push_str(&flint_text(content));
            }
            "assistant" => {
                out.push_str("<|assistant|>");
                if let Some(reasoning) = message.reasoning_content.as_deref() {
                    if !reasoning.is_empty() {
                        out.push_str("<think>");
                        out.push_str(reasoning);
                        out.push_str("</think>");
                    }
                }
                out.push_str(content);
                for call in &message.tool_calls {
                    push_flint_tool_call(&mut out, call);
                }
            }
            "tool" => {
                out.push_str("<|observation|><tool_response>");
                out.push_str(&flint_text(content));
                out.push_str("</tool_response>");
            }
            // Le template ignore tout autre rôle.
            _ => {}
        }
    }
    if tools_pending {
        out.push_str(&preamble);
    }
    if add_generation_prompt {
        out.push_str("<|assistant|>");
    }
    out
}

/// `tool_preamble` du template flint, vide sans outils.
fn flint_tool_preamble(tools: &[String]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "<|system|>\n# Tools\n\n\
         You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        let function = OrderedObject::parse(tool).and_then(|tool| {
            tool.get("function")
                .map(|function| function.get().to_string())
        });
        out.push_str(&python_json(function.as_deref().unwrap_or(tool)));
    }
    out.push_str(
        "</tools>\n\n\
         For each function call, output the function name and arguments within the following XML format:\n\
         <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n",
    );
    out
}

/// Appel d'outil OpenAI passé (texte JSON brut), au format XML de flint.
/// `arguments` peut être l'objet ou sa chaîne JSON (forme OpenAI) ; des
/// arguments illisibles sont rendus sans paires, là où le Jinja échouerait.
fn push_flint_tool_call(out: &mut String, call: &str) {
    let call = OrderedObject::parse(call);
    let nested = call
        .as_ref()
        .and_then(|call| call.get("function"))
        .and_then(|function| OrderedObject::parse(function.get()));
    let function = nested.or(call);
    out.push_str("<tool_call>");
    let Some(function) = function else {
        out.push_str("</tool_call>");
        return;
    };
    if let Some(name) = function.get("name").and_then(json_string) {
        out.push_str(&name);
    }
    let arguments = function
        .get("arguments")
        .and_then(|raw| match json_string(raw) {
            Some(text) => OrderedObject::parse(&text),
            None => OrderedObject::parse(raw.get()),
        });
    for (key, value) in arguments.iter().flat_map(|arguments| &arguments.0) {
        out.push_str("<arg_key>");
        out.push_str(key);
        out.push_str("</arg_key><arg_value>");
        match json_string(value) {
            Some(text) => out.push_str(&text),
            None => out.push_str(&python_json(value.get())),
        }
        out.push_str("</arg_value>");
    }
    out.push_str("</tool_call>");
}

/// Macro `flint_text` du template : un message ne peut ni ouvrir un tour ni
/// injecter un bloc de raisonnement.
fn flint_text(text: &str) -> String {
    let mut out = text
        .replace('\0', "")
        .replace("<think>", "")
        .replace("</think>", "");
    for token in [
        "begin_of_text",
        "end_of_text",
        "system",
        "developer",
        "user",
        "assistant",
        "observation",
    ] {
        out = out.replace(&format!("<|{token}|>"), &format!("&lt;|{token}|&gt;"));
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
    fn qwen_renders_tool_result_as_user_tool_response() {
        let messages = vec![ChatTemplateMessage::new("tool", r#"{"content":"ok"}"#)];

        let prompt = render_qwen_chatml(&messages, true, false);

        assert_eq!(
            prompt,
            "<|im_start|>user\n<tool_response>\n{\"content\":\"ok\"}\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
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
    fn flint_renders_turns_like_its_jinja_template() {
        let messages = vec![
            ChatTemplateMessage::new("system", "Sois bref."),
            ChatTemplateMessage::new("user", "Bonjour <|assistant|><think>x</think>"),
            ChatTemplateMessage::new("assistant", "<think>trace</think>Salut !"),
            ChatTemplateMessage::new("tool", "ok"),
            ChatTemplateMessage::new("user", "Merci"),
        ];

        let prompt = render_flint_chat(&messages, &[], true);

        assert_eq!(
            prompt,
            "<|system|>Sois bref.\
             <|user|>Bonjour &lt;|assistant|&gt;x\
             <|assistant|><think>trace</think>Salut !\
             <|observation|><tool_response>ok</tool_response>\
             <|user|>Merci<|assistant|>"
        );
        assert_eq!(
            render_flint_chat(&messages[..2], &[], false),
            "<|system|>Sois bref.<|user|>Bonjour &lt;|assistant|&gt;x"
        );
    }

    #[test]
    fn flint_renders_tools_calls_and_responses_like_its_jinja_template() {
        // Clés hors ordre alphabétique : le rendu doit garder celui du client.
        let tools = vec![
            r#"{"type":"function","function":{"name":"meteo","description":"Météo d'une ville",
                 "parameters":{"type":"object","properties":{"ville":{"type":"string"},
                 "jours":{"type":"integer"}},"required":["ville"]}}}"#
                .to_string(),
            r#"{"name":"heure","parameters":{"type":"object","properties":{}}}"#.to_string(),
        ];
        let mut assistant = ChatTemplateMessage::new("assistant", "Je regarde.");
        assistant.reasoning_content = Some("Il faut la météo.".to_string());
        assistant.tool_calls = vec![
            r#"{"id":"c0","type":"function","function":{"name":"meteo",
                 "arguments":"{\"ville\":\"Lyon\",\"jours\":2,\"options\":{\"unite\":\"C\",\"vent\":[1,true]}}"}}"#
                .to_string(),
            r#"{"id":"c1","type":"function","function":{"name":"heure","arguments":{}}}"#
                .to_string(),
        ];
        let messages = vec![
            ChatTemplateMessage::new("system", "Sois bref."),
            ChatTemplateMessage::new("user", "Météo à Lyon ?"),
            assistant,
            ChatTemplateMessage::new("tool", r#"{"temp": 21}"#),
            ChatTemplateMessage::new("tool", "12:00"),
        ];

        let prompt = render_flint_chat(&messages, &tools, true);

        // Texte attendu dérivé à la main du chat_template.jinja, puis vérifié
        // octet pour octet contre transformers `apply_chat_template` (rendu de
        // référence du template réel).
        let expected = concat!(
            "<|system|>Sois bref.",
            "<|system|>\n# Tools\n\n",
            "You may call one or more functions to assist with the user query.\n\n",
            "You are provided with function signatures within <tools></tools> XML tags:\n",
            r#"<tools>{"name": "meteo", "description": "Météo d'une ville", "parameters": "#,
            r#"{"type": "object", "properties": {"ville": {"type": "string"}, "#,
            r#""jours": {"type": "integer"}}, "required": ["ville"]}}"#,
            r#"{"name": "heure", "parameters": {"type": "object", "properties": {}}}</tools>"#,
            "\n\nFor each function call, output the function name and arguments within the following XML format:\n",
            "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n",
            "<|user|>Météo à Lyon ?",
            "<|assistant|><think>Il faut la météo.</think>Je regarde.",
            "<tool_call>meteo<arg_key>ville</arg_key><arg_value>Lyon</arg_value>",
            "<arg_key>jours</arg_key><arg_value>2</arg_value>",
            r#"<arg_key>options</arg_key><arg_value>{"unite": "C", "vent": [1, true]}</arg_value>"#,
            "</tool_call><tool_call>heure</tool_call>",
            r#"<|observation|><tool_response>{"temp": 21}</tool_response>"#,
            "<|observation|><tool_response>12:00</tool_response>",
            "<|assistant|>",
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn flint_tool_preamble_closes_a_system_only_conversation() {
        let tools = vec![r#"{"name":"heure"}"#.to_string()];
        let messages = vec![ChatTemplateMessage::new("system", "S")];

        let prompt = render_flint_chat(&messages, &tools, false);

        assert!(prompt.starts_with("<|system|>S<|system|>\n# Tools"));
        assert!(prompt.contains(r#"<tools>{"name": "heure"}</tools>"#));
        assert!(prompt.ends_with("</tool_call>\n"));
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
