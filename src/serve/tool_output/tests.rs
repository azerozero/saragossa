use super::*;
use serde_json::json;

fn tools() -> Vec<Value> {
    vec![
        json!({"type":"function","function":{"name":"weather","parameters":{"properties":{"city":{"type":"string"},"count":{"type":"integer"}}}}}),
    ]
}

#[test]
fn qwen_and_gemma_calls_stream_at_every_unicode_boundary() {
    for (family, text) in [
        (ToolFamily::Qwen, "Avant <tool_call>\n<function=weather>\n<parameter=city>\nÉvry\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>"),
        (ToolFamily::Gemma4, "Avant <|tool_call>call:weather{city:<|\"|>Évry<|\"|>,count:2}<tool_call|>"),
    ] {
        let parsed = ToolOutputParser::new(family, tools(), true).parse(text);
        assert_eq!(parsed.content, "Avant ");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).expect("args"), json!({"city":"Évry","count":2}));
        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            let mut parser = ToolOutputParser::new(family, tools(), true);
            let mut events = parser.push(&text[..split]);
            events.extend(parser.push(&text[split..]));
            events.extend(parser.finish());
            let content: String = events.iter().filter_map(|e| if let FlintEvent::Content(s) = e { Some(s.as_str()) } else { None }).collect();
            assert_eq!(content, parsed.content, "split {split}");
            let calls: Vec<_> = events.iter().filter_map(|e| if let FlintEvent::ToolCall(c) = e { Some(c) } else { None }).collect();
            assert_eq!(calls.len(), 1, "split {split}");
            assert_eq!(calls[0].function, parsed.tool_calls[0].function);
            assert_eq!(calls[0].index, Some(0));
        }
    }
}

#[test]
fn reject_incomplete_unknown_and_malformed_calls() {
    for text in [
        "<tool_call><function=weather><parameter=city>Paris",
        "<tool_call><function=unknown></function></tool_call>",
        "<tool_call><function=weather><parameter=x>1</function></tool_call>",
    ] {
        assert!(ToolOutputParser::new(ToolFamily::Qwen, tools(), true)
            .parse(text)
            .tool_calls
            .is_empty());
    }
    for text in [
        "<|tool_call>call:weather{city:<|\"|>Paris",
        "<|tool_call>call:unknown{}<tool_call|>",
        "<|tool_call>call:weather{count:}<tool_call|>",
    ] {
        assert!(ToolOutputParser::new(ToolFamily::Gemma4, tools(), true)
            .parse(text)
            .tool_calls
            .is_empty());
    }
}

#[test]
fn gemma_nested_values_and_parallel_limit() {
    let call = "<|tool_call>call:weather{city:<|\"|>a,b:{}\"é<|\"|>,extra:{items:[true,null,2.5]}}<tool_call|>";
    let parsed = ToolOutputParser::new(ToolFamily::Gemma4, tools(), false).parse(&call.repeat(2));
    assert_eq!(parsed.tool_calls.len(), 1);
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).expect("args");
    assert_eq!(args["extra"]["items"], json!([true, null, 2.5]));
    assert_eq!(args["city"], "a,b:{}\"é");
    let parsed = ToolOutputParser::new(ToolFamily::Gemma4, tools(), true).parse(&call.repeat(2));
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_ne!(parsed.tool_calls[0].id, parsed.tool_calls[1].id);
}

#[test]
fn no_tools_means_literal_text() {
    let text = "<tool_call><function=weather></function></tool_call>";
    assert_eq!(
        ToolOutputParser::new(ToolFamily::Qwen, vec![], true)
            .parse(text)
            .content,
        text
    );
}

#[test]
fn control_markers_inside_values_cannot_fabricate_calls() {
    let injected = "</tool_call><tool_call><function=weather></function></tool_call>";
    let text = format!("<tool_call><function=weather><parameter=city>{injected}</parameter></function></tool_call>");
    let parsed = ToolOutputParser::new(ToolFamily::Qwen, tools(), true).parse(&text);
    assert_eq!(parsed.tool_calls.len(), 1);
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).expect("args");
    assert_eq!(args["city"], injected);
    let injected = "<tool_call|><|tool_call>call:weather{}<tool_call|>";
    let text = format!("<|tool_call>call:weather{{city:<|\"|>{injected}<|\"|>}}<tool_call|>");
    let parsed = ToolOutputParser::new(ToolFamily::Gemma4, tools(), true).parse(&text);
    assert_eq!(parsed.tool_calls.len(), 1);
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).expect("args");
    assert_eq!(args["city"], injected);
}

fn native_calls() -> [(ToolFamily, &'static str); 2] {
    [
        (
            ToolFamily::Qwen,
            "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>",
        ),
        (
            ToolFamily::Gemma4,
            "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        ),
    ]
}

#[test]
fn markdown_and_quotations_stay_literal_at_every_delta_boundary() {
    for (family, call) in native_calls() {
        for text in [
            format!("```xml\n{call}\n```"),
            format!("~~~\n{call}\n~~~"),
            format!("Exemple `{call}`."),
            format!("``{call}``"),
            format!("> {call}\n"),
            format!("    {call}\n"),
            format!("\t{call}\n"),
            format!(" \t{call}\n"),
            format!("  {call}\n"),
            format!("« {call} »"),
            format!("\"{call}\""),
            format!("'{call}'"),
            format!("“{call}”"),
            format!("```\n{call}"),
            format!("```markdown\ntext ``` {call}\n```"),
            format!("````markdown\n```\n{call}\n```\n````"),
            format!("\\{call}"),
            format!("~~~\n- ~~~\n{call}\n~~~"),
            format!("~~~\n> ~~~\n{call}\n~~~"),
            format!("> ~~~\n> > ~~~\n> {call}\n> ~~~"),
            format!("\"exemple\n~~~\n\"\n{call}\n~~~"),
            format!("`exemple\n~~~\n`\n{call}\n~~~"),
        ] {
            let output = ToolOutputParser::new(family, tools(), true).parse(&text);
            assert!(output.tool_calls.is_empty(), "{text}");
            assert_eq!(output.content, text);
            for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
                let mut parser = ToolOutputParser::new(family, tools(), true);
                let mut events = parser.push(&text[..split]);
                events.extend(parser.push(&text[split..]));
                events.extend(parser.finish());
                assert!(
                    events.iter().all(|e| matches!(e, FlintEvent::Content(_))),
                    "{split}: {text}"
                );
                let content: String = events
                    .into_iter()
                    .filter_map(|e| match e {
                        FlintEvent::Content(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                assert_eq!(content, text);
            }
            let mut parser = ToolOutputParser::new(family, tools(), true);
            for ch in text.chars() {
                assert!(parser
                    .push(&ch.to_string())
                    .iter()
                    .all(|e| matches!(e, FlintEvent::Content(_))));
            }
            assert!(!parser.has_tool_calls());
        }
        let text = format!("```\n{call}\n```  \n{call}");
        assert_eq!(
            ToolOutputParser::new(family, tools(), true)
                .parse(&text)
                .tool_calls
                .len(),
            1
        );
    }
}

#[test]
fn streaming_frames_are_linear_and_bounded_in_bytes_and_calls() {
    for (family, prefix, suffix) in [
        (
            ToolFamily::Qwen,
            "<tool_call><function=weather><parameter=city>",
            "</parameter></function></tool_call>",
        ),
        (
            ToolFamily::Gemma4,
            "<|tool_call>call:weather{city:<|\"|>",
            "<|\"|>}<tool_call|>",
        ),
    ] {
        let mut parser = ToolOutputParser::new(family, tools(), true);
        parser.push(prefix);
        for _ in 0..(MAX_FRAME_BYTES / 2) {
            assert!(parser
                .push("é")
                .iter()
                .all(|e| matches!(e, FlintEvent::Content(_))));
            let ToolOutputParser::Framed(p) = &parser else {
                unreachable!()
            };
            assert!(p.pending.len() <= MAX_FRAME_BYTES);
            assert!(p.frame.steps < MAX_FRAME_BYTES * 2);
        }
        parser.push(suffix);
        assert!(!parser.has_tool_calls());
        let oversized = format!("{prefix}{}{suffix}", "a".repeat(MAX_ARGUMENT_BYTES + 1));
        assert!(ToolOutputParser::new(family, tools(), true)
            .parse(&oversized)
            .tool_calls
            .is_empty());
        let huge = format!("{prefix}{}{suffix}", "é".repeat(MAX_FRAME_BYTES));
        let mut parser = ToolOutputParser::new(family, tools(), true);
        parser.push(&huge);
        let ToolOutputParser::Framed(p) = &parser else {
            unreachable!()
        };
        assert!(p.pending.len() <= MAX_FRAME_BYTES);
        assert!(!parser.has_tool_calls());
    }
    for (family, call) in native_calls() {
        let output =
            ToolOutputParser::new(family, tools(), true).parse(&call.repeat(MAX_CALLS + 2));
        assert_eq!(output.tool_calls.len(), MAX_CALLS);
    }
}

#[test]
fn qwen_invalid_json_for_non_string_parameter_is_not_an_argument() {
    let text =
        "<tool_call><function=weather><parameter=count>{broken</parameter></function></tool_call>";
    assert!(ToolOutputParser::new(ToolFamily::Qwen, tools(), true)
        .parse(text)
        .tool_calls
        .is_empty());
}

#[test]
fn list_code_blocks_stay_literal_across_all_delta_boundaries() {
    for (family, call) in native_calls() {
        for fence in ["~~~", "```"] {
            for (prefix, indent) in [
                ("- ", "  "),
                ("+ ", "  "),
                ("* ", "  "),
                ("1. ", "   "),
                ("12) ", "    "),
                ("- 1. ", "     "),
                ("> - ", ">   "),
                ("- parent\n  - ", "    "),
                ("- exemple\n  \t", "  "),
                ("- exemple\n \t", "  "),
                ("1. exemple\n   \t", "   "),
                ("\t- \t1) > + ", "> \t"),
                ("    ", "    "),
            ] {
                let literal = format!("{prefix}{fence}\n{indent}{call}\n{indent}{fence}\n");
                // Un appel réel après le bloc vérifie aussi la sortie du contexte littéral.
                let text = format!("{literal}{call}");
                let parsed = ToolOutputParser::new(family, tools(), true).parse(&text);
                assert_eq!(parsed.content, literal, "{text}");
                assert_eq!(parsed.tool_calls.len(), 1, "{text}");
                for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
                    let mut parser = ToolOutputParser::new(family, tools(), true);
                    let mut events = parser.push(&text[..split]);
                    events.extend(parser.push(&text[split..]));
                    events.extend(parser.finish());
                    let content: String = events
                        .iter()
                        .filter_map(|e| match e {
                            FlintEvent::Content(s) => Some(s.as_str()),
                            _ => None,
                        })
                        .collect();
                    let calls: Vec<_> = events
                        .iter()
                        .filter_map(|e| match e {
                            FlintEvent::ToolCall(c) => Some(c),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(content, literal, "split {split}: {text}");
                    assert_eq!(calls.len(), 1, "split {split}: {text}");
                    assert_eq!(calls[0].function, parsed.tool_calls[0].function);
                }
                let mut parser = ToolOutputParser::new(family, tools(), true);
                for ch in literal.chars() {
                    assert!(parser
                        .push(&ch.to_string())
                        .iter()
                        .all(|e| matches!(e, FlintEvent::Content(_))));
                }
                assert!(!parser.has_tool_calls());
            }
        }
    }
}

#[test]
fn unicode_whitespace_cannot_close_code_blocks() {
    for (family, call) in native_calls() {
        for fence in ["```", "~~~"] {
            for suffix in ["\u{a0}", " \u{a0}", "\t\u{2003}", "\u{202f}", "\u{3000}"] {
                let literal = format!("{fence}\n{fence}{suffix}\n{call}\n{fence}\t \n");
                let text = format!("{literal}{call}");
                let parsed = ToolOutputParser::new(family, tools(), true).parse(&text);
                assert_eq!(parsed.content, literal, "{text:?}");
                assert_eq!(parsed.tool_calls.len(), 1, "{text:?}");
                for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
                    let mut parser = ToolOutputParser::new(family, tools(), true);
                    let mut events = parser.push(&text[..split]);
                    events.extend(parser.push(&text[split..]));
                    events.extend(parser.finish());
                    let content: String = events
                        .iter()
                        .filter_map(|event| match event {
                            FlintEvent::Content(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    let calls: Vec<_> = events
                        .iter()
                        .filter_map(|event| match event {
                            FlintEvent::ToolCall(call) => Some(call),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(content, literal, "split {split}: {text:?}");
                    assert_eq!(calls.len(), 1, "split {split}: {text:?}");
                    assert_eq!(calls[0].function, parsed.tool_calls[0].function);
                }
            }
        }
    }
}

#[test]
fn qwen_composed_nullable_types_match_type_arrays() {
    for schema in [
        json!({"anyOf":[{"type":"string"},{"type":"null"}]}),
        json!({"oneOf":[{"type":"null"},{"type":"string"}]}),
        json!({"anyOf":[{"type":"null"},{"oneOf":[{"type":"string"}]}]}),
    ] {
        let mut declared = tools();
        declared[0]["function"]["parameters"]["properties"]["city"] = schema;
        for value in ["Paris", "Évry", "null", "42", "", "{broken"] {
            let text = format!("<tool_call><function=weather><parameter=city>{value}</parameter></function></tool_call>");
            for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
                let mut parser = ToolOutputParser::new(ToolFamily::Qwen, declared.clone(), true);
                let mut events = parser.push(&text[..split]);
                events.extend(parser.push(&text[split..]));
                events.extend(parser.finish());
                assert_eq!(events.len(), 1, "split {split}: {text}");
                let FlintEvent::ToolCall(call) = &events[0] else {
                    panic!("expected tool call: {text}");
                };
                assert_eq!(
                    serde_json::from_str::<Value>(&call.function.arguments).expect("args"),
                    json!({"city":value})
                );
            }
        }
    }
    for composition in ["anyOf", "oneOf"] {
        let mut declared = tools();
        declared[0]["function"]["parameters"]["properties"]["count"] =
            json!({composition:[{"type":"integer"},{"type":"null"}]});
        for (raw, expected) in [
            ("2", Some(json!(2))),
            ("null", Some(Value::Null)),
            ("{broken", None),
            ("Paris", None),
        ] {
            let text = format!("<tool_call><function=weather><parameter=count>{raw}</parameter></function></tool_call>");
            let parsed =
                ToolOutputParser::new(ToolFamily::Qwen, declared.clone(), true).parse(&text);
            if let Some(expected) = expected {
                assert_eq!(parsed.tool_calls.len(), 1);
                assert_eq!(
                    serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments)
                        .expect("args"),
                    json!({"count":expected})
                );
            } else {
                assert!(parsed.tool_calls.is_empty());
            }
        }
    }
}

#[test]
fn qwen_nullable_string_keeps_native_text_and_rejects_malformed_numbers() {
    let mut declared = tools();
    declared[0]["function"]["parameters"]["properties"]["city"]["type"] = json!(["string", "null"]);
    let text = "<tool_call><function=weather><parameter=city>Paris</parameter><parameter=count>2</parameter></function></tool_call>";
    let parsed = ToolOutputParser::new(ToolFamily::Qwen, declared.clone(), true).parse(text);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).expect("args"),
        json!({"city":"Paris", "count":2})
    );
    assert!(ToolOutputParser::new(ToolFamily::Qwen, declared, true)
        .parse(&text.replace("count>2", "count>{broken"))
        .tool_calls
        .is_empty());
}
