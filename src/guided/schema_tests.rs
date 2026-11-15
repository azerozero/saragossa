use super::*;
use serde_json::json;

fn byte_constraint(schema: Value) -> SchemaTokenConstraint {
    let mut bytes: Vec<_> = (0..=255).map(|b| vec![b]).collect();
    bytes.extend([Vec::new(), Vec::new()]);
    let catalog = Arc::new(SchemaTokenCatalog::from_bytes(bytes, &[256, 257]).expect("catalog"));
    SchemaTokenConstraint::new(catalog, &JsonSchema::compile(&schema).expect("schema"))
        .expect("constraint")
}

fn consume(constraint: &SchemaTokenConstraint, bytes: &[u8]) {
    for &byte in bytes {
        let mut logits = vec![0.0; 260];
        constraint.mask_logits(&mut logits).expect("mask");
        assert!(
            logits[usize::from(byte)].is_finite(),
            "masked byte {byte} in {bytes:?}"
        );
        assert_eq!(logits[259], f32::NEG_INFINITY);
        constraint.accept_token(usize::from(byte)).expect("accept");
    }
}

#[test]
fn compiler_rejects_invalid_unsupported_and_excessive_schemas() {
    for schema in [
        json!({"type":"array"}),
        json!({"type":"object", "properties": []}),
        json!({"type":"object", "required":["missing"]}),
        json!({"type":"object", "properties":{"x":{}}, "required":["x","x"]}),
        json!({"type":"object", "properties":{"x":{"enum":[]}}}),
        json!({"type":"object", "properties":{"x":{"type":"invalid"}}}),
        json!({"type":"object", "properties":{"x":{"type":["string","string"]}}}),
        json!({"type":"object", "items": []}),
        json!({"type":"object", "additionalProperties": 1}),
        json!({"type":"object", "properties":{"x":{"pattern":".*"}}}),
        json!({"type":"object", "x-guidance":{"lenient":true}}),
        json!({"type":"object", "$ref":"https://example.invalid/schema"}),
        json!({"type":"object", "properties":{"x":false}, "required":["x"]}),
    ] {
        assert!(JsonSchema::compile(&schema).is_err(), "accepted {schema}");
    }
    let mut nested = json!({});
    for _ in 0..34 {
        nested = json!({"type":"object", "properties":{"x": nested}});
    }
    assert!(JsonSchema::compile(&nested).is_err());
    assert!(
        JsonSchema::compile(&json!({"type":"object", "description":"x".repeat(65536)})).is_err()
    );
}

#[test]
fn required_enum_nested_array_and_additional_properties_masks() {
    let schema = json!({"type":"object", "properties":{
        "data":{"type":"array", "items":{"type":"integer"}},
        "kind":{"enum":["ok","non"]},
        "meta":{"type":"object", "additionalProperties":{"type":"boolean"}}
    }, "required":["data","kind","meta"], "additionalProperties":false});
    let c = byte_constraint(schema);
    assert!(c.accept_token(256).is_err());
    consume(&c, b"{");
    let mut logits = vec![0.0; 258];
    c.mask_logits(&mut logits).expect("mask");
    assert_eq!(logits[b'}' as usize], f32::NEG_INFINITY);
    assert!(c.accept_token(b'}' as usize).is_err());
    consume(&c, br#""data":[1,-2],"kind":"ok","meta":{"enabled":true}}"#);
    assert!(c.is_finished());
    logits.fill(0.0);
    c.mask_logits(&mut logits).expect("terminal mask");
    assert_eq!(logits.iter().filter(|v| v.is_finite()).count(), 2);
    assert!(logits[256].is_finite() && logits[257].is_finite());
    assert!(c.accept_token(b' ' as usize).is_err());
}

#[test]
fn unicode_and_invalid_escapes_preserve_json_syntax() {
    let schema = json!({"type":"object", "properties":{"s":{"type":"string"}}, "required":["s"], "additionalProperties":false});
    for text in [r#"{"s":"été𐀀"}"#, r#"{"s":"\u000a"}"#] {
        let c = byte_constraint(schema.clone());
        consume(&c, text.as_bytes());
        assert!(c.is_finished());
        assert!(serde_json::from_str::<Value>(text).is_ok());
    }
    let c = byte_constraint(
        json!({"type":"object", "properties":{"s":{}}, "required":["s"], "additionalProperties":false}),
    );
    consume(&c, br#"{"s":"\u"#);
    let mut logits = vec![0.0; 258];
    c.mask_logits(&mut logits).expect("mask");
    assert_eq!(logits[b'D' as usize], f32::NEG_INFINITY);
    assert_eq!(logits[b'"' as usize], f32::NEG_INFINITY);
}

#[test]
fn multi_byte_tokens_cannot_skip_schema_checks_and_requests_are_isolated() {
    let bytes = [br#"{"ok":true}"#.as_slice(), b"{}", br#"{"ok":false}"#, b""]
        .iter()
        .map(|b| b.to_vec())
        .collect();
    let catalog = Arc::new(SchemaTokenCatalog::from_bytes(bytes, &[3]).expect("catalog"));
    for expected in [true, false, true] {
        let schema = JsonSchema::compile(&json!({"type":"object", "properties":{"ok":{"enum":[expected]}}, "required":["ok"], "additionalProperties":false})).expect("schema");
        let c = SchemaTokenConstraint::new(Arc::clone(&catalog), &schema).expect("constraint");
        let mut logits = vec![0.0; 5];
        c.mask_logits(&mut logits).expect("mask");
        let valid = if expected { 0 } else { 2 };
        assert_eq!(logits.iter().position(|v| v.is_finite()), Some(valid));
        assert_eq!(logits.iter().filter(|v| v.is_finite()).count(), 1);
        c.accept_token(valid).expect("accept");
        assert!(c.is_finished());
    }
}

#[test]
fn scalar_types_optional_properties_and_additional_schema_are_enforced() {
    for (schema, candidates) in [
        (
            json!({"type":"object", "properties":{"x":{"type":"integer"}}, "required":["x"], "additionalProperties":false}),
            vec![
                (r#"{"x":1}"#, true),
                (r#"{"x":1.5}"#, false),
                (r#"{"x":"1"}"#, false),
                ("{}", false),
            ],
        ),
        (
            json!({"type":"object", "properties":{"x":{"type":["string","null"]}}, "additionalProperties":false}),
            vec![
                ("{}", true),
                (r#"{"x":null}"#, true),
                (r#"{"x":"v"}"#, true),
                (r#"{"x":false}"#, false),
                (r#"{"z":null}"#, false),
            ],
        ),
        (
            json!({"type":"object", "additionalProperties":{"type":"number"}}),
            vec![
                ("{}", true),
                (r#"{"other":1.5}"#, true),
                (r#"{"other":"v"}"#, false),
            ],
        ),
        (
            json!({"type":"object", "properties":{"x":{"type":"array", "items":false}}, "required":["x"], "additionalProperties":false}),
            vec![(r#"{"x":[]}"#, true), (r#"{"x":[1]}"#, false)],
        ),
    ] {
        let mut bytes: Vec<_> = candidates
            .iter()
            .map(|(text, _)| text.as_bytes().to_vec())
            .collect();
        let eos = bytes.len();
        bytes.push(Vec::new());
        let catalog = Arc::new(SchemaTokenCatalog::from_bytes(bytes, &[eos]).expect("catalog"));
        let c = SchemaTokenConstraint::new(catalog, &JsonSchema::compile(&schema).expect("schema"))
            .expect("constraint");
        let mut logits = vec![0.0; eos + 1];
        c.mask_logits(&mut logits).expect("mask");
        for (id, (text, expected)) in candidates.iter().enumerate() {
            assert_eq!(logits[id].is_finite(), *expected, "{schema}: {text}");
        }
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(48))]
    #[test]
    fn every_sampled_completion_satisfies_schema(seed in proptest::prelude::any::<u64>()) {
        let c = byte_constraint(json!({"type":"object", "properties":{
            "choice":{"enum":["oui", "non"]}, "ok":{"type":"boolean"}
        }, "required":["choice", "ok"], "additionalProperties":false}));
        let mut rng = seed;
        let mut bytes = Vec::new();
        for _ in 0..128 {
            if c.is_finished() { break; }
            let mut logits = vec![0.0; 258];
            c.mask_logits(&mut logits).expect("mask");
            let choices: Vec<_> = logits.iter().enumerate().filter(|(id,v)| v.is_finite() && *id < 256 && !matches!(*id, 9 | 10 | 13 | 32)).map(|(id,_)| id).collect();
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let token = choices[(rng as usize) % choices.len()];
            c.accept_token(token).expect("selected token");
            bytes.push(token as u8);
        }
        proptest::prop_assert!(c.is_finished());
        let value: Value = serde_json::from_slice(&bytes).expect("JSON");
        proptest::prop_assert_eq!(value.as_object().expect("object").len(), 2);
        proptest::prop_assert!(matches!(value["choice"].as_str(), Some("oui" | "non")));
        proptest::prop_assert!(value["ok"].is_boolean());
    }
}
