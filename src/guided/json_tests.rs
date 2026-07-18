use super::*;

const EOT: usize = 99;

#[test]
fn accepts_valid_root_objects() {
    for json in [
        br#"{}"#.as_slice(),
        br#" { "a": 1, "b": [true, false, null] } "#.as_slice(),
        br#"{"nested":{"n":1e-5,"x":-0.5}}"#.as_slice(),
    ] {
        let mut automaton = JsonAutomaton::new(false);
        assert!(automaton.advance_bytes(json), "{json:?}");
        assert!(automaton.is_final(), "{json:?}");
        serde_json::from_slice::<serde_json::Value>(json).expect("invariant: cas JSON valide");
    }
}

#[test]
fn rejects_non_object_root() {
    let mut automaton = JsonAutomaton::new(false);

    assert!(!automaton.advance_bytes(br#"[]"#));
    assert!(!automaton.is_final());
}

#[test]
fn rejects_bad_numbers() {
    for json in [
        br#"{"n":01}"#.as_slice(),
        br#"{"n":1.}"#.as_slice(),
        br#"{"n":1e}"#.as_slice(),
    ] {
        let mut automaton = JsonAutomaton::new(false);
        assert!(!automaton.advance_bytes(json), "{json:?}");
    }
}

#[test]
fn accepts_escapes_and_unicode_sequences() {
    let json = br#"{"s":"quote:\" slash:\\ newline:\n unicode:\u00e9"}"#;
    let mut automaton = JsonAutomaton::new(false);

    assert!(automaton.advance_bytes(json));
    assert!(automaton.is_final());
}

#[test]
fn accepts_paired_surrogates_but_rejects_lone_ones() {
    let paired = br#"{"s":"\uD83D\uDE00"}"#; // \uD83D\uDE00 en paire de surrogates
    let mut automaton = JsonAutomaton::new(false);
    assert!(automaton.advance_bytes(paired));
    assert!(automaton.is_final());
    serde_json::from_slice::<serde_json::Value>(paired)
        .expect("invariant: paire de surrogates valide");

    for json in [
        br#"{"s":"\uD800"}"#.as_slice(),  // surrogate haut isolé
        br#"{"s":"\uDC00"}"#.as_slice(),  // surrogate bas isolé
        br#"{"s":"\uD83DA"}"#.as_slice(), // haut + \u non-bas
        br#"{"s":"\uD800A"}"#.as_slice(), // haut non complété par \u
    ] {
        let mut automaton = JsonAutomaton::new(false);
        assert!(!automaton.advance_bytes(json), "{json:?}");
        assert!(
            serde_json::from_slice::<serde_json::Value>(json).is_err(),
            "oracle serde_json refuse aussi: {json:?}"
        );
    }
}

#[test]
fn rejects_raw_newline_in_string_and_bad_escape() {
    for json in [b"{\"s\":\"a\nb\"}".as_slice(), br#"{"s":"\x"}"#.as_slice()] {
        let mut automaton = JsonAutomaton::new(false);
        assert!(!automaton.advance_bytes(json), "{json:?}");
    }
}

#[test]
fn accepts_utf8_bytes_split_across_tokens_inside_string() {
    let mut automaton = JsonAutomaton::new(false);

    assert!(automaton.advance_bytes(br#"{"s":""#));
    assert!(automaton.advance_bytes(&[0xC3]));
    assert!(automaton.advance_bytes(&[0xA9]));
    assert!(automaton.advance_bytes(br#""}"#));

    assert!(automaton.is_final());
}

#[test]
fn json_lines_accepts_multiple_objects_and_eot_boundaries() {
    let ndjson = r#"{"a":1}
{"b":2}
{"c":3}"#;
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[
        br#"{"a":1}"#,
        b"\n",
        br#"{"b":2}"#,
        br#"{"c":3}"#,
        b"{",
        b"}",
    ]));
    let constraint = JsonTokenConstraint::new(Arc::clone(&catalog), &[EOT], true);

    assert_eot_masked(&constraint);
    constraint
        .accept_token(0)
        .expect("invariant: premier objet admissible");
    assert_eot_allowed(&constraint);
    constraint
        .accept_token(1)
        .expect("invariant: séparateur NDJSON admissible");
    assert_eot_allowed(&constraint);
    constraint
        .accept_token(2)
        .expect("invariant: deuxième objet admissible");
    assert_eot_allowed(&constraint);
    constraint
        .accept_token(1)
        .expect("invariant: deuxième séparateur admissible");
    constraint
        .accept_token(3)
        .expect("invariant: troisième objet admissible");
    assert!(constraint.is_finished());

    for line in ndjson.split('\n') {
        let value: serde_json::Value =
            serde_json::from_str(line).expect("invariant: ligne NDJSON parseable");
        assert!(value.is_object());
    }

    let partial = JsonTokenConstraint::new(catalog, &[EOT], true);
    partial
        .accept_token(4)
        .expect("invariant: début d'objet admissible");
    assert_eot_masked(&partial);
    partial
        .accept_token(EOT)
        .expect_err("invariant: EOT refusé pendant un objet");
}

#[test]
fn json_lines_rejects_non_newline_separators() {
    let mut comma = JsonAutomaton::new(true);
    assert!(comma.advance_bytes(br#"{"a":1}"#));
    assert!(!comma.advance_bytes(b","));

    let mut glued = JsonAutomaton::new(true);
    assert!(glued.advance_bytes(br#"{}"#));
    assert!(!glued.advance_bytes(br#"{}"#));

    let mut blank_line = JsonAutomaton::new(true);
    assert!(blank_line.advance_bytes(br#"{}"#));
    assert!(blank_line.advance_bytes(b"\n"));
    assert!(!blank_line.advance_bytes(b"\n"));

    let mut spaced_line = JsonAutomaton::new(true);
    assert!(spaced_line.advance_bytes(br#"{}"#));
    assert!(spaced_line.advance_bytes(b"\n"));
    assert!(!spaced_line.advance_bytes(b" "));
}

#[test]
fn json_object_newline_does_not_repeat_after_root() {
    let mut automaton = JsonAutomaton::new(false);

    assert!(automaton.advance_bytes(br#"{}"#));
    assert!(automaton.is_final());
    assert!(automaton.advance_bytes(b"\n"));
    assert!(automaton.is_final());
    assert!(!automaton.advance_bytes(br#"{}"#));
}

#[test]
fn eot_is_allowed_only_after_root_object_closed() {
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[
        b"{", b"}", b"\"a\"", b":", b"1",
    ]));
    let constraint = JsonTokenConstraint::new(catalog, &[EOT], false);
    let mut logits = vec![0.0; 100];

    constraint
        .mask_logits(&mut logits)
        .expect("invariant: début contraignable");
    assert!(logits[EOT].is_infinite() && logits[EOT].is_sign_negative());

    for token in [0, 2, 3, 4, 1] {
        constraint
            .accept_token(token)
            .expect("invariant: token test admissible");
    }
    let mut logits = vec![0.0; 100];
    constraint
        .mask_logits(&mut logits)
        .expect("invariant: état final contraignable");
    assert!(logits[EOT].is_finite());
    assert!(constraint.is_finished());
}

#[test]
fn eot_lookup_accepts_only_configured_ids() {
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[b"{}"]));
    let constraint = JsonTokenConstraint::new(catalog, &[42, 7, 42], false);

    assert!(constraint.is_eot(7));
    assert!(constraint.is_eot(42));
    assert!(!constraint.is_eot(0));
    assert!(!constraint.is_eot(41));
}

#[test]
fn mask_rejects_tokens_that_break_the_prefix() {
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[
        b"{", b"[", b"\"a\"", b":", b"true", b"}",
    ]));
    let constraint = JsonTokenConstraint::new(catalog, &[EOT], false);
    let mut logits = vec![0.0; 100];

    constraint
        .mask_logits(&mut logits)
        .expect("invariant: début contraignable");

    assert!(logits[0].is_finite());
    assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
}

#[test]
fn reports_empty_candidate_set() {
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[b"[", b"]"]));
    let constraint = JsonTokenConstraint::new(catalog, &[], false);
    let mut logits = vec![0.0; 2];

    let error = constraint
        .mask_logits(&mut logits)
        .expect_err("invariant: aucun token ne peut ouvrir l'objet racine");

    assert!(error.to_string().contains("aucun token admissible"));
}

fn assert_eot_allowed(constraint: &JsonTokenConstraint) {
    let mut logits = vec![0.0; 100];
    constraint
        .mask_logits(&mut logits)
        .expect("invariant: état final contraignable");
    assert!(logits[EOT].is_finite());
}

fn assert_eot_masked(constraint: &JsonTokenConstraint) {
    let mut logits = vec![0.0; 100];
    constraint
        .mask_logits(&mut logits)
        .expect("invariant: état non final contraignable");
    assert!(logits[EOT].is_infinite() && logits[EOT].is_sign_negative());
}
