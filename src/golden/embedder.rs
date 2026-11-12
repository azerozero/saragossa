use super::external::*;
use crate::text_embedder::TextEmbedder;

fn vectors() -> TestResult<Vec<f32>> {
    let (header, _) = read("embedder/e5")?;
    assert_eq!(header.dtype, "f32");
    assert_eq!(header.shape, [20, 384]);
    // Même lecteur little-endian que les goldens TTS/Whisper historiques.
    let (_, vectors) = super::read_f32(&root().join("embedder/e5").to_string_lossy())?;
    Ok(vectors)
}

#[test]
fn external_embedder_fixture_parse() -> TestResult {
    for row in vectors()?.chunks_exact(384) {
        assert!(row.iter().all(|v| v.is_finite()));
        let norm = row.iter().map(|v| v * v).sum::<f32>();
        assert!((norm - 1.0).abs() < 1e-5, "norme={norm}");
    }
    Ok(())
}

fn compare() -> TestResult<Option<Vec<String>>> {
    let Some(path) = local("SARAGOSSA_EMBEDDER_DIR")? else {
        return Ok(None);
    };
    check_model("embedder/e5", &path)?;
    let model = TextEmbedder::load_local(path)?;
    let header: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root().join("embedder/e5.json"))?)?;
    let expected = vectors()?;
    let cases = header["cases"].as_array().ok_or("cas absents")?;
    assert_eq!(cases.len(), 20);
    let mut failures = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let text = case["text"].as_str().ok_or("texte absent")?;
        let actual = if case["kind"] == "query" {
            model.embed_query(text)?
        } else {
            model.embed_passage(text)?
        };
        assert_eq!(actual.len(), 384);
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{index}: vecteur non fini"
        );
        let reference = &expected[index * 384..(index + 1) * 384];
        let dot = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>();
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let cosine = dot / (norm(&actual) * norm(reference));
        let max_abs = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("embedder[{index}]: cosine={cosine:.9} max_abs={max_abs:.9}");
        if !cosine.is_finite() || cosine < 0.9999 || !max_abs.is_finite() || max_abs > 0.001 {
            failures.push(format!(
                "{index}/{}",
                case["kind"].as_str().ok_or("type absent")?
            ));
        }
    }
    Ok(Some(failures))
}

#[test]
#[ignore = "CPU réel: SARAGOSSA_EMBEDDER_DIR; écart connu 8/query (requête vide); tools/goldens/README.md#suivi-des-ecarts"]
fn external_embedder_known_differences_are_stable() -> TestResult {
    if let Some(differences) = compare()? {
        assert_known_differences(&differences, &["8/query"]);
    }
    Ok(())
}

#[test]
#[ignore = "parité stricte CPU: SARAGOSSA_EMBEDDER_DIR; requête vide divergente; tools/goldens/README.md#suivi-des-ecarts"]
fn external_embedder_strict_parity_matches_sentence_transformers() -> TestResult {
    if let Some(differences) = compare()? {
        assert_known_differences(&differences, &[]);
    }
    Ok(())
}
