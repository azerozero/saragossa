use super::*;

fn fixture(root: &Path) -> (CausalDecoder, Option<Request>) {
    let model = CausalDecoder::from_tensors(
        super::super::tests::test_weights(),
        CausalDecoderConfig::default(),
    )
    .unwrap();
    let request = Request {
        root: root.to_owned(),
        key: Key {
            version: 1,
            prompt_sha256: "test".into(),
            tokens: 2,
            sources: Vec::new(),
            binary_sha256: "build-a".into(),
            mode: "ar".into(),
            config: "test".into(),
            kv_dtype: "mixed".into(),
            runtime_sha256: "test".into(),
            little_endian: cfg!(target_endian = "little"),
        },
    };
    (model, Some(request))
}

#[test]
fn roundtrip_and_reject_corruption_without_weights() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let (model, request) = fixture(dir.path());
    let (mut cache, final_state) = model.prefill_cache_state(&[0, 1])?;
    // Le transport doit conserver les bits bf16, le f32 et les états CPU,
    // indépendamment du chemin qui a produit ce cache synthétique.
    cache.layers[0].keys = kv_rows::KvRows::default();
    let len = cache.layers[0].values.len();
    cache.layers[0].keys.push_owned_bf16(vec![0x8000; len]);
    cache.layers[0].linear.conv = vec![1.0, -0.0];
    cache.layers[0].linear.conv_dim = Some(2);
    cache.layers[0].linear.ssm = vec![3.0, 4.0];
    cache.layers[0].linear.ssm_shape = Some((1, 1, 2));
    let captures = PrefixCaptures::new(&[0], 1, final_state.clone());
    let mut history = cache.layers[0].clone();
    history.truncate(1)?;
    model.snapshot_save(
        &request,
        &cache,
        &final_state,
        Some(&history),
        Some(&captures),
    )?;
    let request = request.unwrap();
    let loaded = model.snapshot_read(&request)?;
    assert_eq!(loaded.cache.position, 2);
    assert_eq!(loaded.cache.layers[0].keys.snapshot_segments()[0].0, "bf16");
    assert_eq!(
        loaded.cache.layers[0].keys.snapshot_segments()[0].1,
        cache.layers[0].keys.snapshot_segments()[0].1
    );
    assert_eq!(
        loaded.cache.layers[0].linear.conv[1].to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(loaded.cache.layers[0].linear.ssm, vec![3.0, 4.0]);
    assert_eq!(loaded.final_state.data(), final_state.data());
    assert_eq!(loaded.history.unwrap().keys.to_vec(), history.keys.to_vec());
    assert_eq!(
        loaded.captures.unwrap().hidden.data(),
        captures.hidden.data()
    );
    let mut other = Request {
        root: dir.path().into(),
        key: request.key.clone(),
    };
    other.key.binary_sha256 = "build-b".into();
    assert!(model
        .snapshot_read(&other)
        .err()
        .unwrap()
        .to_string()
        .contains("binaire différent"));
    other.key = request.key.clone();
    other.key.tokens += 1;
    assert!(model.snapshot_read(&other).is_err());
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(dir.path().join("manifest.json")).unwrap()).unwrap();
    let blob = dir.path().join(&manifest.layers[0].keys.segments[0].file);
    std::fs::write(blob, [0u8; 1]).unwrap();
    assert!(model.snapshot_read(&request).is_err());
    Ok(())
}

#[test]
fn publication_incomplete_keeps_previous_manifest() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::new(dir.path())?;
    let blob = writer.blob("bf16", &[0, 128, 128, 63])?;
    writer.publish(&vec![&blob])?;
    let previous = std::fs::read(dir.path().join("manifest.json")).unwrap();
    {
        let mut interrupted = Writer::new(dir.path())?;
        interrupted.f32(&[42.0])?;
    }
    assert_eq!(
        previous,
        std::fs::read(dir.path().join("manifest.json")).unwrap()
    );
    assert_eq!(blob.read(dir.path())?, [0, 128, 128, 63]);
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn identity_with_weights(mtp: bool) -> Result<()> {
    let dir = crate::runtime_flags::env_var_os("SARAGOSSA_TEST_MODEL")
        .ok_or_else(|| invalid("SARAGOSSA_TEST_MODEL requis, aucun téléchargement"))?;
    let assets = crate::ModelAssets::load_local(dir)?;
    let prompt: Vec<usize> = assets
        .encode_prompt("Explique pourquoi le ciel est bleu en quelques phrases.")?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let root = tempfile::tempdir().unwrap();
    let original = crate::runtime_flags::state_snapshot_dir();
    // Test ignoré, à lancer seul sous verrou GPU : chaque phase recharge le
    // décodeur pour exclure un hit du cache RAM ou un état GPU resté vivant.
    let run = || -> Result<Vec<usize>> {
        let mut model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
        if mtp {
            let path = assets
                .mtp
                .path
                .as_ref()
                .ok_or_else(|| invalid("tête MTP absente"))?;
            model = model.with_mtp_sidecar(path)?;
            Ok(model
                .generate_greedy_mtp_core(&prompt, 24, &GenerationOptions::default(), 2, |_| true)?
                .tokens)
        } else {
            Ok(model
                .generate_greedy_timed_with_options(&prompt, 24, &GenerationOptions::default())?
                .tokens)
        }
    };
    std::env::remove_var("SARAGOSSA_RUST_STATE_SNAPSHOT");
    let result = (|| -> Result<()> {
        let baseline = run()?;
        std::env::set_var("SARAGOSSA_RUST_STATE_SNAPSHOT", root.path());
        let cold = run()?;
        assert!(std::fs::read_dir(root.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .any(|entry| entry.path().join("manifest.json").is_file()));
        HITS.set(0);
        let warm = run()?;
        assert_eq!(
            HITS.get(),
            1,
            "le prefill doit être sauté, sans repli silencieux"
        );
        assert_eq!(baseline, cold, "écriture : ids greedy différents");
        assert_eq!(baseline, warm, "relecture : ids greedy différents");
        Ok(())
    })();
    match original {
        Some(path) => std::env::set_var("SARAGOSSA_RUST_STATE_SNAPSHOT", path),
        None => std::env::remove_var("SARAGOSSA_RUST_STATE_SNAPSHOT"),
    }
    result
}

#[test]
#[ignore = "poids locaux et verrou GPU requis, AR"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn identity_weights_ar() -> Result<()> {
    identity_with_weights(false)
}

#[test]
#[ignore = "poids locaux et verrou GPU requis, MTP"]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn identity_weights_mtp() -> Result<()> {
    identity_with_weights(true)
}

#[test]
fn restored_cpu_decode_matches_complete_run() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let (model, request) = fixture(dir.path());
    let (cache, final_state) = model.prefill_cache_state(&[0, 1])?;
    model.snapshot_save(&request, &cache, &final_state, None, None)?;
    let mut restored = model.snapshot_read(request.as_ref().unwrap())?.cache;
    let mut baseline = cache;
    for token in [2, 0, 1, 2] {
        assert_eq!(
            model.next_final_state_cached(&mut baseline, token)?.data(),
            model.next_final_state_cached(&mut restored, token)?.data()
        );
    }
    Ok(())
}

#[test]
fn rows_preserve_sliding_base_and_mixed_dtypes() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut rows = kv_rows::KvRows::default();
    assert!(rows.set_base(128));
    rows.push_owned_bf16(vec![0x3f80, 0x8000]);
    rows.push_owned(vec![1.000_001, -0.0]);
    let mut writer = Writer::new(dir.path())?;
    let stored = Rows::write(&rows, &mut writer)?;
    let restored = stored.read(dir.path())?;
    assert_eq!(restored.base(), 128);
    assert_eq!(restored.len(), 132);
    assert_eq!(restored.snapshot_segments(), rows.snapshot_segments());
    Ok(())
}
