use super::*;

#[test]
fn readout_calibration_uses_only_valid_rows() -> Result<()> {
    let w = Tensor::from_vec(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1000.0, 1000.0])?;
    let p = calibrated_readout(&[2.0, 0.0], &w, 2, 2.0)?;
    assert!((p[0] - 0.7310586).abs() < 1e-6);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert_eq!(calibrated_readout(&[2.0, 0.0], &w, 1, 1.0)?, [1.0]);
    assert!(calibrated_readout(&[2.0], &w, 2, 1.0).is_err());
    assert!(calibrated_readout(&[2.0, 0.0], &w, 2, 0.0).is_err());
    assert!(calibrated_readout(&[f32::NAN, 0.0], &w, 2, 1.0).is_err());
    Ok(())
}

#[test]
fn answer_semantics_and_validation() -> Result<()> {
    for empty in [
        json!(null),
        json!(false),
        json!(0),
        json!(""),
        json!([]),
        json!({}),
    ] {
        assert!(!truthy(&empty));
    }
    let (_, defaults) = options(&json!({"type":"noul","criteria":{"false":[],"true":{}}}))?;
    assert_eq!(defaults, [json!("No / false"), json!("Yes / true")]);
    let choice = json!({"type":"choice","criteria":{"z":"last alphabetically","a":null}});
    let a = format_answer(&choice, &[0.75, 0.25])?;
    assert_eq!(a["choice"], "z");
    assert_eq!(a["confidence"], 0.5);
    assert_eq!(
        format_answer(&json!({"type":"noul"}), &[0.25, 0.75])?["noul"],
        0.75
    );
    let score = json!({"type":"score","criteria":["bad","neutral","good"]});
    let s = format_answer(&score, &[0.25, 0.5, 0.25])?;
    assert_eq!(s["score"], 1.0);
    assert_eq!(s["confidence"], 0.25);
    assert!(options(&json!({"type":"score","criteria":["alone"]})).is_err());
    assert!(validate_request(&json!({"state":"s","questions":{},"think":true})).is_err());
    assert!(validate_request(&json!({"state":"s","questions":{"q":choice},"images":[]})).is_err());
    assert_eq!(describe(&json!({"x":[true,"é"]})), "{\"x\": [true, \"é\"]}");
    Ok(())
}

#[test]
fn decision_mask_scope_is_nested_and_thread_local() {
    assert!(!crate::runtime_flags::decision_noncausal());
    crate::runtime_flags::with_decision_noncausal(|| {
        assert!(crate::runtime_flags::decision_noncausal());
        crate::runtime_flags::with_decision_noncausal(|| {
            assert!(crate::runtime_flags::decision_noncausal())
        });
        assert!(crate::runtime_flags::decision_noncausal());
        assert!(
            !std::thread::spawn(crate::runtime_flags::decision_noncausal)
                .join()
                .unwrap()
        );
    });
    assert!(!crate::runtime_flags::decision_noncausal());
}

#[test]
#[ignore = "poids THX-02 locaux, référence f32 et verrou GPU requis"]
fn decision_real_weight_oracle() -> Result<()> {
    let path = crate::runtime_flags::env_var_os("SARAGOSSA_DECISION_MODEL")
        .ok_or_else(|| invalid("SARAGOSSA_DECISION_MODEL requis"))?;
    let model = DecisionModel::load(std::path::PathBuf::from(path), true)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decision");
    let reference = read_json(&root.join("reference-f32.json"))?;
    let tolerance = read_json(&root.join("tolerance.json"))?["max_abs_probability"]
        .as_f64()
        .ok_or_else(|| invalid("tolérance bf16 mesurée requise"))?;
    // Seuil établi sur la référence avant la première comparaison Rust.
    assert!((tolerance - 0.012_218_342_853_546_142).abs() < 1e-12);
    let records = reference["records"]
        .as_array()
        .ok_or_else(|| invalid("référence vide"))?;
    let mut maximum = 0.0_f64;
    for record in records {
        let encoded = model.encode(&record["request"])?;
        assert_eq!(encoded.text, record["text"].as_str().unwrap());
        assert_eq!(
            serde_json::to_value(&encoded.tokens).unwrap(),
            record["tokens"]
        );
        assert_eq!(
            serde_json::to_value(&encoded.markers).unwrap(),
            record["markers"]
        );
        let actual = model.probabilities(&encoded.tokens, &encoded.markers, &encoded.counts)?;
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(record["probabilities"].clone()).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.len(), b.len());
        }
        for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!(a.is_finite() && b.is_finite());
            maximum = maximum.max(f64::from((a - b).abs()));
        }
    }
    let long = read_json(&root.join("reference-long.json"))?;
    for record in long
        .as_array()
        .ok_or_else(|| invalid("oracle long absent"))?
    {
        let tokens: Vec<usize> = serde_json::from_value(record["tokens"].clone()).unwrap();
        let markers: Vec<usize> = serde_json::from_value(record["markers"].clone()).unwrap();
        let counts: Vec<usize> = serde_json::from_value(record["counts"].clone()).unwrap();
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(record["probabilities"].clone()).unwrap();
        let actual = model.probabilities(&tokens, &markers, &counts)?;
        for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!(a.is_finite() && b.is_finite());
            maximum = maximum.max(f64::from((a - b).abs()));
        }
    }
    eprintln!("decision oracle max_abs={maximum:.9} tolerance={tolerance:.9}");
    assert!(
        maximum <= tolerance,
        "écart {maximum} > bruit bf16 admis {tolerance}"
    );
    Ok(())
}
