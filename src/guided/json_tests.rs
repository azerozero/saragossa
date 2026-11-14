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
fn closed_object_allows_only_model_stops_even_with_higher_whitespace_logits() {
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[
        b"{}",
        b" ",
        b"\n\n  \n",
        b"\t",
        b"\r\n",
        b"",
        b"<eos>",
        b"<end_of_turn>",
    ]));
    // Un EOS ou plusieurs fins de tour : la règle ne dépend pas de la famille.
    for stops in [&[6][..], &[7, 6][..]] {
        let constraint = JsonTokenConstraint::new(Arc::clone(&catalog), stops, false);
        constraint.accept_token(0).expect("objet fermé");
        let mut logits = vec![100.0; 8];
        for &stop in stops {
            logits[stop] = -10.0;
        }
        constraint
            .mask_logits(&mut logits)
            .expect("masque terminal");
        for (token, logit) in logits.iter().enumerate() {
            assert_eq!(logit.is_finite(), stops.contains(&token), "token {token}");
        }
        for token in 0..6 {
            assert!(constraint.accept_token(token).is_err(), "token {token}");
        }
        for &stop in stops {
            constraint.accept_token(stop).expect("stop du modèle");
        }
    }
}

#[test]
fn closing_token_may_include_whitespace_but_next_token_must_stop() {
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&[
        b"{\n",
        b"\"key\":",
        b"{}",
        b"}\n  ",
        b" ",
    ]));
    let constraint = JsonTokenConstraint::new(catalog, &[EOT], false);
    for token in [0, 1, 2] {
        constraint.accept_token(token).expect("préfixe JSON");
    }
    assert!(
        !constraint.is_finished(),
        "l'objet imbriqué ne ferme pas la racine"
    );
    constraint
        .accept_token(4)
        .expect("espace avant fermeture racine");
    constraint
        .accept_token(3)
        .expect("fermeture et espaces dans un même token");
    let mut logits = vec![0.0; EOT + 1];
    constraint
        .mask_logits(&mut logits)
        .expect("masque terminal");
    assert_eq!(logits.iter().filter(|x| x.is_finite()).count(), 1);
    assert!(logits[EOT].is_finite());
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

/// Oracle historique : clone indépendant par candidat, sans préfiltre.
#[test]
fn reused_automaton_preserves_every_mask_bit() -> Result<()> {
    let mut tokens: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
    tokens.extend([
        Vec::new(),
        b"{}".to_vec(),
        b"\"x\": [".to_vec(),
        b"true,".to_vec(),
        b"null}".to_vec(),
        b"-1.2e+3}".to_vec(),
        b"\\uD83D\\uDE00\"}".to_vec(),
        b"\n{".to_vec(),
        vec![0xff, b'}'],
        b"]}}".to_vec(),
    ]);
    let slices: Vec<&[u8]> = tokens.iter().map(Vec::as_slice).collect();
    let catalog = Arc::new(JsonTokenCatalog::from_token_bytes_for_tests(&slices));
    let eot = tokens.len();
    for repeat in [false, true] {
        for document in [
            br#"{"a":[{"b":true},null,-1.25e+3],"s":"\uD83D\uDE00"}"#.as_slice(),
            "{\"é\":\"😀\"}\n{}".as_bytes(),
        ] {
            let constraint = JsonTokenConstraint::new(catalog.clone(), &[eot], repeat);
            let mut state = JsonAutomaton::new(repeat);
            for offset in 0..=document.len() {
                *constraint.state.lock().expect("invariant: mutex sain") = state.clone();
                let mut logits: Vec<f32> = (0..=eot + 1)
                    .map(|i| match i % 5 {
                        0 => f32::NAN,
                        1 => f32::NEG_INFINITY,
                        2 => f32::INFINITY,
                        3 => -0.0,
                        _ => 0.0,
                    })
                    .collect();
                let mut expected = logits.clone();
                let mut allowed = 0;
                for (token, logit) in expected.iter_mut().enumerate() {
                    let admissible = if token == eot {
                        state.is_final()
                    } else if state.is_final() && !repeat {
                        false
                    } else {
                        tokens.get(token).is_some_and(|bytes| {
                            !bytes.is_empty() && state.clone().advance_bytes(bytes)
                        })
                    };
                    if admissible {
                        allowed += 1;
                    } else {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                assert_eq!(constraint.mask_logits(&mut logits).is_ok(), allowed > 0);
                assert_eq!(
                    logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "repeat={repeat}, offset={offset}"
                );
                if offset == document.len() || !state.advance_bytes(&document[offset..offset + 1]) {
                    break;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn clone_from_reuses_nested_stack_after_rejected_candidates() {
    let mut source = JsonAutomaton::new(false);
    assert!(source.advance_bytes(br#"{"a":[{"b":["#));
    let mut scratch = source.clone();
    scratch.stack.reserve(16);
    let pointer = scratch.stack.as_ptr();
    let capacity = scratch.stack.capacity();
    for candidate in [b"]}]}garbage".as_slice(), b"[[]]", b"null", b"invalid"] {
        assert_eq!(
            source.can_accept_bytes(candidate, &mut scratch),
            source.clone().advance_bytes(candidate)
        );
        assert_eq!(scratch.stack.as_ptr(), pointer);
        assert_eq!(scratch.stack.capacity(), capacity);
    }
}
