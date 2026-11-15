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
    let a = format_answer(&choice, &[0.75, 0.25], false)?;
    assert_eq!(a["choice"], "z");
    assert_eq!(a["confidence"], 0.5);
    assert_eq!(
        format_answer(&json!({"type":"noul"}), &[0.25, 0.75], false)?["noul"],
        0.75
    );
    let score = json!({"type":"score","criteria":["bad","neutral","good"]});
    let s = format_answer(&score, &[0.25, 0.5, 0.25], false)?;
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
fn d3_answer_semantics_and_validation() -> Result<()> {
    assert!(check_d3(&json!({"type":"score","criteria":["a"]})).is_err());
    assert!(check_d3(&json!({"type":"score","criteria":(0..11).collect::<Vec<_>>()})).is_err());
    assert!(check_d3(&json!({"type":"score","criteria":["a","a"]})).is_err());
    assert!(check_d3(&json!({"type":"score","criteria":[1,"1"]})).is_err());
    assert!(check_d3(&json!({"type":"noul","criteria":{"maybe":"x"}})).is_err());
    assert!(check_d3(&json!({"type":"noul","criteria":[]})).is_err());
    assert!(check_d3(&json!({"type":"choice","criteria":{"":"x"}})).is_err());
    check_d3(&json!({"type":"noul","criteria":null}))?;
    let choice = json!({"type":"choice","criteria":{"a":null,"b":null}});
    assert_eq!(
        format_answer(&choice, &[0.5, 0.5], true)?["confidence"],
        0.0
    );
    assert_eq!(
        format_answer(&choice, &[1.0, 0.0], true)?["confidence"],
        1.0
    );
    let score = json!({"type":"score","criteria":["low",{"z":1,"a":[true,"é"]}]});
    let s = format_answer(&score, &[0.25, 0.75], true)?;
    assert_eq!(s["legend"]["1"], "{\"a\":[true,\"é\"],\"z\":1}");
    assert_eq!(s["score"], 0.75);
    Ok(())
}

/// Compare saragossa à une référence f32 figée : prompts, tokens, positions,
/// probabilités et, si la référence les fournit, champs non numériques des
/// réponses. Renvoie l'écart absolu maximal.
fn real_weight_oracle(env: &str, fixtures: &str, frozen: f64) -> Result<f64> {
    let path =
        crate::runtime_flags::env_var_os(env).ok_or_else(|| invalid(format!("{env} requis")))?;
    let model = DecisionModel::load(std::path::PathBuf::from(path), true)?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(fixtures);
    let reference = read_json(&root.join("reference-f32.json"))?;
    let tolerance = read_json(&root.join("tolerance.json"))?["max_abs_probability"]
        .as_f64()
        .ok_or_else(|| invalid("tolérance bf16 mesurée requise"))?;
    // Seuil établi sur la référence avant la première comparaison Rust.
    assert!((tolerance - frozen).abs() < 1e-12);
    let records = reference["records"]
        .as_array()
        .ok_or_else(|| invalid("référence vide"))?;
    let mut maximum = 0.0_f64;
    let mut compare = |actual: &[Vec<f32>], expected: &Value| {
        let expected: Vec<Vec<f32>> = serde_json::from_value(expected.clone()).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert!(a.is_finite() && b.is_finite());
                maximum = maximum.max(f64::from((a - b).abs()));
            }
        }
    };
    for record in records {
        // THX : une séquence au niveau du record ; d3 : une par question.
        let sequences = record["sequences"]
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![record.clone()]);
        let encoded = model.encode(&record["request"])?;
        assert_eq!(encoded.len(), sequences.len());
        let mut actual = Vec::new();
        for (e, s) in encoded.iter().zip(&sequences) {
            assert_eq!(e.text, s["text"].as_str().unwrap());
            assert_eq!(json!(e.tokens), s["tokens"]);
            if let Some(markers) = s.get("markers") {
                assert_eq!(&json!(e.markers), markers);
            }
            actual.extend(model.probabilities(&e.tokens, &e.markers, &e.counts)?);
        }
        compare(&actual, &record["probabilities"]);
        if let Some(answers) = record["answers"].as_object() {
            let ours = model.decide(&record["request"])?;
            for (key, reference) in answers {
                for (field, value) in reference.as_object().unwrap() {
                    if !value.is_number() && field != "probabilities" {
                        assert_eq!(&ours["answers"][key][field], value, "{key}.{field}");
                    }
                }
            }
        }
    }
    let long = read_json(&root.join("reference-long.json"))?;
    for record in long
        .as_array()
        .ok_or_else(|| invalid("oracle long absent"))?
    {
        let tokens: Vec<usize> = serde_json::from_value(record["tokens"].clone()).unwrap();
        let markers: Vec<usize> = record
            .get("markers")
            .map_or(Ok(vec![tokens.len() - 1]), |m| {
                serde_json::from_value(m.clone())
            })
            .unwrap();
        let counts: Vec<usize> = serde_json::from_value(record["counts"].clone()).unwrap();
        compare(
            &model.probabilities(&tokens, &markers, &counts)?,
            &record["probabilities"],
        );
    }
    eprintln!("decision oracle {fixtures} max_abs={maximum:.9} tolerance={tolerance:.9}");
    assert!(
        maximum <= tolerance,
        "écart {maximum} > bruit bf16 admis {tolerance}"
    );
    Ok(maximum)
}

#[test]
#[ignore = "poids THX-02 locaux, référence f32 et verrou GPU requis"]
fn decision_real_weight_oracle() -> Result<()> {
    real_weight_oracle(
        "SARAGOSSA_DECISION_MODEL",
        "tests/fixtures/decision",
        0.012_218_342_853_546_142,
    )
    .map(drop)
}

#[test]
#[ignore = "poids d3-flash locaux, référence f32 et verrou GPU requis"]
fn decision_d3_real_weight_oracle() -> Result<()> {
    real_weight_oracle(
        "SARAGOSSA_DECISION_D3_MODEL",
        "tests/fixtures/decision/d3",
        0.018_397_973_609_924_317,
    )
    .map(drop)
}
