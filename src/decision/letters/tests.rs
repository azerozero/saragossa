use super::*;

#[test]
fn letter_prompt_follows_shim_rendering() -> Result<()> {
    let labels: Vec<String> = ["A", "B", "C"].map(String::from).to_vec();
    let noul = options(&json!({"type":"noul","criteria":{"yes":"Y","false":""}}), 3)?;
    assert_eq!(noul.names, ["true", "false"]);
    assert_eq!(noul.descriptions, ["Y", "it does not"]);
    assert_eq!(
        options(&json!({"type":"noul","criteria":[]}), 3)?.descriptions,
        ["the statement holds", "it does not"]
    );
    assert!(options(&json!({"type":"noul","criteria":"x"}), 3).is_err());
    assert!(options(&json!({"type":"box"}), 3).is_err());
    assert!(options(&json!({"type":null}), 3).is_err());
    assert!(options(&json!({"type":"score","criteria":[]}), 3).is_err());
    assert!(options(&json!({"criteria":{"a":1,"b":2,"c":3,"d":4}}), 3).is_err());
    let (user, choice) = user_turn(
        "{\"k\":1}",
        &json!({"instructions":{"q":"é"},"criteria":{"a":null,"b":["x",2]}}),
        &labels,
        "D",
    )?;
    assert_eq!(
        user,
        "record: {\"k\":1}\nquestion: {\n \"q\": \"\\u00e9\"\n}\noptions:\nA) a: a\nB) b: [\"x\", 2]"
    );
    assert_eq!(choice.kind, Kind::Choice);
    let score = json!({"type":"score","instructions":"  ","criteria":["lo"]});
    assert!(user_turn("", &score, &labels, "Default.")?
        .0
        .contains("question: Default.\n"));
    let null = json!({"type":"noul","instructions":null});
    assert!(user_turn("", &null, &labels, "D")?
        .0
        .contains("question: null\n"));
    assert_eq!(
        python_indent1(&json!({"e":"😀","l":[]}))?,
        "{\n \"e\": \"\\ud83d\\ude00\",\n \"l\": []\n}"
    );
    Ok(())
}

#[test]
fn letter_answers_and_modes_follow_shim() -> Result<()> {
    let names = |n: &[&str]| n.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let choice = Options {
        kind: Kind::Choice,
        names: names(&["x", "y"]),
        descriptions: names(&["x", "y"]),
    };
    let a = answer(&choice, &[0.25, 0.75], 0.0, None);
    assert_eq!(
        (a["choice"].as_str(), a["confidence"].as_f64()),
        (Some("y"), Some(0.5))
    );
    assert_eq!(answer(&choice, &[0.5, 0.5], 0.0, None)["choice"], "x");
    let noul = Options {
        kind: Kind::Noul,
        names: names(&["true", "false"]),
        descriptions: names(&["y", "n"]),
    };
    let n = answer(&noul, &[0.625, 0.375], 0.8, Some(0.9));
    assert_eq!(
        (n["noul"].as_f64(), n["decision"].as_bool()),
        (Some(0.8), Some(false))
    );
    assert_eq!(answer(&noul, &[0.375, 0.625], 0.0, None)["noul"], 0.375);
    let score = Options {
        kind: Kind::Score,
        names: names(&["0", "1", "2"]),
        descriptions: names(&["lo", "mid", "hi"]),
    };
    let s = answer(&score, &[0.25, 0.5, 0.25], 0.0, None);
    assert_eq!(
        (s["score"].as_f64(), s["confidence"].as_f64()),
        (Some(1.0), Some(0.5))
    );
    assert_eq!(s["legend"]["2"], "hi");
    assert_eq!(
        threshold(&json!({"type":"noul","noul_threshold":0.3})),
        Some(0.3)
    );
    assert_eq!(threshold(&json!({"type":"noul","threshold":2})), None);
    assert_eq!(threshold(&json!({"type":"choice","threshold":0.3})), None);
    let one = AttentionMode::Causal;
    assert_eq!(
        AttentionMode::parse(Some(&json!("Two-Way")), one)?,
        AttentionMode::Bidirectional
    );
    assert_eq!(AttentionMode::parse(None, one)?, one);
    assert!(AttentionMode::parse(Some(&json!(1)), one).is_err());
    assert_eq!(common_prefix(&[&[1, 2, 3], &[1, 2, 4]]), 2);
    assert_eq!(common_prefix(&[&[1, 2], &[1, 2, 3]]), 1);
    assert!(Calibration::parse(&json!({"choice":1,"noul":1}), &json!(0)).is_err());
    assert!(Calibration::parse(&json!(1), &json!(0.4)).is_err());
    assert_eq!(
        Calibration::parse(&json!(0.7), &json!(0))?.temperature(Kind::Score),
        0.7
    );
    Ok(())
}

/// Écart maximal entre lectures et référence, après contrôle des formes.
fn drift(actual: &[Vec<f32>], expected: &Value) -> f64 {
    let expected: Vec<Vec<f32>> =
        serde_json::from_value(expected.clone()).expect("invariant: probabilités de référence");
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    for (a, b) in actual.iter().zip(&expected) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(a.is_finite() && b.is_finite());
            maximum = maximum.max(f64::from((a - b).abs()));
        }
    }
    maximum
}

/// Référence f32 Transformers (deux modes), cas longs bidirectionnels, puis
/// cache de préfixe causal contre prefills complets sur un état de 4000 tokens.
#[test]
#[ignore = "poids H2O-Lightning-4B locaux, référence f32 et verrou GPU requis"]
fn decision_letters_h2o_real_weight_oracle() -> Result<()> {
    let path = crate::runtime_flags::env_var_os("SARAGOSSA_DECISION_H2O_MODEL")
        .ok_or_else(|| invalid("SARAGOSSA_DECISION_H2O_MODEL requis"))?;
    let model = LetterModel::load(std::path::PathBuf::from(path), true)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decision/h2o");
    let tolerance = read_json(&root.join("tolerance.json"))?["max_abs_probability"]
        .as_f64()
        .ok_or_else(|| invalid("tolérance bf16 mesurée requise"))?;
    // Seuil figé sur la référence avant la première comparaison Rust.
    assert!((tolerance - FROZEN_TOLERANCE).abs() < 1e-12);
    let reference = read_json(&root.join("reference-f32.json"))?;
    let mut maximum = 0.0_f64;
    for record in reference["records"].as_array().expect("invariant: records") {
        let request = &record["request"];
        let mode = model.mode(request)?;
        let reads = model.encode(request)?;
        let sequences = record["sequences"]
            .as_array()
            .expect("invariant: séquences");
        assert_eq!(reads.len(), sequences.len());
        for (read, sequence) in reads.iter().zip(sequences) {
            assert_eq!(
                read.text,
                sequence["text"].as_str().expect("invariant: texte")
            );
            assert_eq!(json!(read.tokens), sequence["tokens"]);
        }
        maximum = maximum.max(drift(
            &model.probabilities(&reads, mode, true)?,
            &record["probabilities"],
        ));
        let ours = model.decide(request)?;
        assert_eq!(ours["attention_mode"], mode.name());
        for (key, expected) in record["answers"].as_object().expect("invariant: réponses") {
            for (field, value) in expected.as_object().expect("invariant: réponse") {
                if !value.is_number() && field != "probabilities" {
                    assert_eq!(&ours["answers"][key][field], value, "{key}.{field}");
                }
            }
        }
    }
    for case in read_json(&root.join("reference-long.json"))?
        .as_array()
        .expect("invariant: cas longs")
    {
        let read = LetterRead {
            text: String::new(),
            tokens: serde_json::from_value(case["tokens"].clone()).expect("invariant: tokens"),
            kind: Kind::Noul,
            count: 2,
        };
        let p = model.probabilities(&[read], AttentionMode::Bidirectional, true)?;
        maximum = maximum.max(drift(&p, &case["probabilities"]));
    }
    let cache = read_json(&root.join("reference-cache.json"))?;
    let mut request = cache["request"].clone();
    request["attention_mode"] = json!("causal");
    let reads = model.encode(&request)?;
    let prefix: Vec<usize> =
        serde_json::from_value(cache["prefix_tokens"].clone()).expect("invariant: préfixe");
    let suffixes = cache["suffixes"].as_array().expect("invariant: suffixes");
    let reads = &reads[..suffixes.len()];
    for (read, suffix) in reads.iter().zip(suffixes) {
        let suffix: Vec<usize> =
            serde_json::from_value(suffix.clone()).expect("invariant: suffixe");
        assert_eq!(read.tokens, [&prefix[..], &suffix[..]].concat());
    }
    let shared = model.probabilities(reads, AttentionMode::Causal, true)?;
    let cold = model.probabilities(reads, AttentionMode::Causal, false)?;
    let cache_drift = drift(&shared, &cache["probabilities"]);
    let cold_drift = drift(&cold, &cache["probabilities"]);
    let (mut equal, mut gap) = (0, 0.0_f32);
    for (a, b) in shared.iter().flatten().zip(cold.iter().flatten()) {
        equal += usize::from(a.to_bits() == b.to_bits());
        gap = gap.max((a - b).abs());
    }
    let total = shared.iter().map(Vec::len).sum::<usize>();
    eprintln!(
        "letters h2o oracle max_abs={maximum:.9} cache_ref={cache_drift:.9} cold_ref={cold_drift:.9} \
         cache_vs_cold bit_equal={equal}/{total} max={gap:.9} prefix={} tolerance={tolerance:.9}",
        prefix.len()
    );
    assert!(
        maximum.max(cache_drift).max(cold_drift) <= tolerance,
        "écart > bruit bf16 admis {tolerance}"
    );
    assert_eq!(
        equal, total,
        "cache de préfixe : écart {gap} avec les prefills complets"
    );
    Ok(())
}

const FROZEN_TOLERANCE: f64 = 0.021_488_593_650_817_872;
