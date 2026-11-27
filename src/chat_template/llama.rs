//! Templates de chat Llama 3.1+ et Mistral Instruct v0.3, sans outils.

use super::{ChatTemplateMessage, TemplateError};

/// BOS littéral Llama 3, rendu comme `{{ bos_token }}` du template HF.
pub const LLAMA3_BOS: &str = "<|begin_of_text|>";
/// Fin de tour Llama 3.
pub const LLAMA3_EOT: &str = "<|eot_id|>";
/// Fin de message avec appel d'outil Llama 3.1+.
pub const LLAMA3_EOM: &str = "<|eom_id|>";

fn header(out: &mut String, role: &str) {
    out.push_str("<|start_header_id|>");
    out.push_str(role);
    out.push_str("<|end_header_id|>\n\n");
}

/// Rend des messages au format chat Llama 3.1 / 3.2 / 3.3 sans outils.
///
/// Suit le template HF : BOS littéral, bloc système toujours présent avec les
/// dates (`date` remplace `strftime_now("%d %b %Y")`), contenus trimmés.
///
/// # Errors
///
/// Refuse un rôle autre que `system` en tête, `user` ou `assistant` : les
/// tours d'outils (`tool`, `ipython`) ne sont pas rendus ici.
pub fn render_llama3_chat(
    messages: &[ChatTemplateMessage],
    add_generation_prompt: bool,
    date: &str,
) -> Result<String, TemplateError> {
    let (system, turns) = match messages.split_first() {
        Some((first, rest)) if first.role == "system" => {
            (first.content.as_deref().unwrap_or("").trim(), rest)
        }
        _ => ("", messages),
    };
    let mut out = String::from(LLAMA3_BOS);
    header(&mut out, "system");
    out.push_str("Cutting Knowledge Date: December 2023\nToday Date: ");
    out.push_str(date);
    out.push_str("\n\n");
    out.push_str(system);
    out.push_str(LLAMA3_EOT);
    for message in turns {
        if !matches!(message.role.as_str(), "user" | "assistant") || !message.tool_calls.is_empty()
        {
            return Err(TemplateError::UnsupportedRole(message.role.clone()));
        }
        header(&mut out, &message.role);
        out.push_str(message.content.as_deref().unwrap_or("").trim());
        out.push_str(LLAMA3_EOT);
    }
    if add_generation_prompt {
        header(&mut out, "assistant");
    }
    Ok(out)
}

/// Date figée par défaut des templates Llama 3.1 et 3.3.
pub const LLAMA3_FIXED_DATE: &str = "26 Jul 2024";

/// Date du jour au format `%d %b %Y` (UTC), celle de `strftime_now` du template.
///
/// NOTE: transformers lit l'heure locale ; l'écart ne porte que sur l'heure
/// autour de minuit.
#[must_use]
pub fn llama3_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    // Algorithme civil_from_days de Howard Hinnant (calendrier grégorien proleptique).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month = MONTHS[usize::try_from(month - 1).unwrap_or(0)];
    format!("{day:02} {month} {year}")
}

/// Rend des messages au format Mistral Instruct v0.3 sans outils.
///
/// Suit le template HF : `<s>` littéral, `[INST] …[/INST]`, message système
/// optionnel préfixé au dernier message, s'il est user (`loop.last` : sinon il disparaît), réponses ` …</s>` trimmées. Le
/// template n'a pas de prompt de génération : le dernier `[/INST]` en tient lieu.
///
/// # Errors
///
/// Refuse une alternance user/assistant rompue et tout rôle d'outil.
pub fn render_mistral_chat(messages: &[ChatTemplateMessage]) -> Result<String, TemplateError> {
    let (system, turns) = match messages.split_first() {
        Some((first, rest)) if first.role == "system" => (first.content.as_deref(), rest),
        _ => (None, messages),
    };
    let mut out = String::from("<s>");
    for (index, message) in turns.iter().enumerate() {
        if !message.tool_calls.is_empty() {
            return Err(TemplateError::UnsupportedRole(message.role.clone()));
        }
        let expected = if index % 2 == 0 { "user" } else { "assistant" };
        if !matches!(message.role.as_str(), "user" | "assistant") {
            return Err(TemplateError::UnsupportedRole(message.role.clone()));
        }
        if message.role != expected {
            return Err(TemplateError::RoleAlternation);
        }
        let content = message.content.as_deref().unwrap_or("");
        if message.role == "user" {
            out.push_str("[INST] ");
            if let (Some(system), true) = (system, index + 1 == turns.len()) {
                out.push_str(system);
                out.push_str("\n\n");
            }
            out.push_str(content);
            out.push_str("[/INST]");
        } else {
            out.push(' ');
            out.push_str(content.trim());
            out.push_str("</s>");
        }
    }
    Ok(out)
}

/// Rend le prompt de génération d'un Llama 3.x ou d'un Mistral, `None` pour
/// une autre famille. Point d'entrée commun du CLI, du REPL et de `serve`.
///
/// # Errors
///
/// Propage les refus de rôle et d'alternance des deux templates.
pub fn render_llama_or_mistral_chat(
    config: &crate::ModelConfig,
    messages: &[ChatTemplateMessage],
) -> Option<Result<String, TemplateError>> {
    if config.is_llama() {
        // NOTE: Llama 3.1/3.3 figent `date_string` à « 26 Jul 2024 » ; seul 3.2 appelle
        // `strftime_now`. La config ne porte pas la version : le facteur de scaling
        // llama3 (32 pour 3.2 1B/3B, 8 pour 3.1/3.3) la distingue.
        let date = if config
            .rope_llama3_scaling()
            .is_some_and(|scaling| scaling.factor == 32.0)
        {
            llama3_today()
        } else {
            LLAMA3_FIXED_DATE.to_string()
        };
        Some(render_llama3_chat(messages, true, &date))
    } else if config.is_mistral() {
        Some(render_mistral_chat(messages))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatTemplateMessage {
        ChatTemplateMessage::new(role, content)
    }

    #[test]
    fn llama3_renders_system_block_dates_and_generation_prompt() {
        let rendered = render_llama3_chat(
            &[msg("system", " Sois bref. "), msg("user", " Bonjour ")],
            true,
            "26 Jul 2024",
        )
        .expect("invariant: rôles valides");
        assert_eq!(
            rendered,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n\
             Sois bref.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             Bonjour<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert!(matches!(
            render_llama3_chat(&[msg("tool", "{}")], true, "x"),
            Err(TemplateError::UnsupportedRole(_))
        ));
    }

    #[test]
    fn mistral_prefixes_system_to_last_user_turn() {
        let rendered = render_mistral_chat(&[
            msg("system", "Sys"),
            msg("user", "A"),
            msg("assistant", " B "),
            msg("user", "C"),
        ])
        .expect("invariant: alternance valide");
        assert_eq!(rendered, "<s>[INST] A[/INST] B</s>[INST] Sys\n\nC[/INST]");
        // `loop.last` : sans dernier tour user, le système disparaît du rendu.
        assert_eq!(
            render_mistral_chat(&[
                msg("system", "Sys"),
                msg("user", "A"),
                msg("assistant", "B")
            ])
            .as_deref(),
            Ok("<s>[INST] A[/INST] B</s>")
        );
        assert_eq!(
            render_mistral_chat(&[msg("assistant", "x")]),
            Err(TemplateError::RoleAlternation)
        );
    }

    #[test]
    fn today_has_strftime_shape() {
        let today = llama3_today();
        let parts: Vec<&str> = today.split(' ').collect();
        assert_eq!(parts.len(), 3, "{today}");
        assert_eq!(parts[0].len(), 2);
        assert!(parts[2].parse::<u32>().is_ok_and(|year| year >= 2026));
    }

    #[test]
    fn llama_date_follows_template_version() {
        let config = |factor: f32| {
            let mut config = crate::qwen_loader::test_fixtures::test_config();
            config.model_type = "llama".to_string();
            config.rope_scaling = Some(crate::RopeScalingConfig {
                rope_type: Some("llama3".to_string()),
                factor: Some(factor),
                low_freq_factor: Some(1.0),
                high_freq_factor: Some(4.0),
                original_max_position_embeddings: Some(8192),
                ..Default::default()
            });
            config
        };
        let render = |factor| {
            render_llama_or_mistral_chat(&config(factor), &[msg("user", "A")])
                .expect("invariant: famille Llama")
                .expect("invariant: rôles valides")
        };
        assert!(render(8.0).contains("Today Date: 26 Jul 2024"));
        assert!(!render(32.0).contains("Today Date: 26 Jul 2024"));
    }
}
