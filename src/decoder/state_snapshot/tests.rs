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
            version: 2,
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
    let (mut model, request) = fixture(dir.path());
    let AttentionBlock::Full(attention) = &model.layers[0].attention else {
        panic!("fixture full-attention attendue");
    };
    let identity = Linear::new(
        Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?,
        None,
    )?;
    model.mtp = Some(MtpHead {
        pre_fc_norm_embedding: model.final_norm.clone(),
        pre_fc_norm_hidden: model.final_norm.clone(),
        fc: Linear::new(Tensor::from_vec(vec![2, 4], vec![0.0; 8])?, None)?,
        layer: MtpLayer {
            input_norm: model.final_norm.clone(),
            attention: attention.as_ref().clone(),
            post_attention_norm: model.final_norm.clone(),
            mlp: FeedForward::Dense(Box::new(GatedMlp::new(
                identity.clone(),
                identity.clone(),
                identity,
            ))),
        },
        norm: model.final_norm.clone(),
    });
    let (mut cache, final_state) = model.prefill_cache_state(&[0, 1])?;
    // Le transport doit conserver les bits bf16, le f32 et les états CPU,
    // indépendamment du chemin qui a produit ce cache synthétique.
    cache.layers[0].keys = kv_rows::KvRows::default();
    let len = cache.layers[0].values.len();
    cache.layers[0].keys.push_owned_bf16(vec![0x8000; len]);
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
    // Le flag est immuable : baseline, écriture et hit vivent dans trois
    // processus distincts, chacun recharge ses poids et ses OnceLock.
    let phase = std::env::var("SARAGOSSA_SNAPSHOT_ORACLE_PHASE").ok();
    let Some(phase) = phase else {
        let root = tempfile::tempdir().unwrap();
        let name = if mtp {
            "identity_weights_mtp"
        } else {
            "identity_weights_ar"
        };
        let mut outputs = Vec::new();
        for phase in ["baseline", "cold", "warm"] {
            let output_path = root.path().join(format!("{phase}.json"));
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    &format!("decoder::state_snapshot::tests::{name}"),
                    "--ignored",
                    "--nocapture",
                ])
                .env("SARAGOSSA_SNAPSHOT_ORACLE_PHASE", phase)
                .env("SARAGOSSA_SNAPSHOT_ORACLE_OUTPUT", &output_path)
                .env_remove("SARAGOSSA_RUST_STATE_SNAPSHOT")
                .env_remove("RETI_RUST_STATE_SNAPSHOT");
            if phase != "baseline" {
                command.env(
                    "SARAGOSSA_RUST_STATE_SNAPSHOT",
                    root.path().join("snapshots"),
                );
            }
            assert!(command.status().unwrap().success());
            outputs.push(std::fs::read(output_path).unwrap());
        }
        assert_eq!(outputs[0], outputs[1], "écriture : ids greedy différents");
        assert_eq!(outputs[0], outputs[2], "relecture : ids greedy différents");
        return Ok(());
    };
    let dir = crate::runtime_flags::env_var_os("SARAGOSSA_TEST_MODEL")
        .ok_or_else(|| invalid("SARAGOSSA_TEST_MODEL requis, aucun téléchargement"))?;
    let assets = crate::ModelAssets::load_local(dir)?;
    let prompt: Vec<usize> = assets
        .encode_prompt("Explique pourquoi le ciel est bleu en quelques phrases.")?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let mut model = crate::load_causal_decoder(&assets)?.with_metal_runtime()?;
    HITS.set(0);
    let tokens = if mtp {
        let path = assets
            .mtp
            .path
            .as_ref()
            .ok_or_else(|| invalid("tête MTP absente"))?;
        model = model.with_mtp_sidecar(path)?;
        model
            .generate_greedy_mtp_core(&prompt, 24, &GenerationOptions::default(), 2, |_| true)?
            .tokens
    } else {
        model
            .generate_greedy_timed_with_options(&prompt, 24, &GenerationOptions::default())?
            .tokens
    };
    assert_eq!(HITS.get(), usize::from(phase == "warm"));
    let output = std::env::var_os("SARAGOSSA_SNAPSHOT_ORACLE_OUTPUT").unwrap();
    std::fs::write(output, serde_json::to_vec(&tokens).unwrap()).unwrap();
    Ok(())
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

fn read_manifest(root: &Path) -> Manifest {
    serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap()
}

fn write_manifest(root: &Path, manifest: &Manifest) {
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
}

// Chaque test d'environnement tourne dans son propre processus : aucun OnceLock
// ni réglage numérique d'un autre test du harness ne peut le contaminer.
fn isolated(name: &str, enabled: bool, run: impl FnOnce(&Path)) {
    if let Some(root) = std::env::var_os("SARAGOSSA_SNAPSHOT_TEST_CHILD") {
        run(Path::new(&root));
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            &format!("decoder::state_snapshot::tests::{name}"),
            "--nocapture",
        ])
        .env("SARAGOSSA_SNAPSHOT_TEST_CHILD", dir.path())
        .env_remove("RETI_RUST_STATE_SNAPSHOT")
        .env_remove("SARAGOSSA_RUST_STATE_SNAPSHOT")
        .env("SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK", "4096");
    if enabled {
        command.env(
            "SARAGOSSA_RUST_STATE_SNAPSHOT",
            dir.path().join("snapshots"),
        );
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn sources_frozen_at_load_reject_replacement_and_skip_writes() {
    isolated(
        "sources_frozen_at_load_reject_replacement_and_skip_writes",
        true,
        |root| {
            let path = root.join("model.safetensors");
            let bytes = safetensors::serialize(
                super::super::tests::tensor_views(super::super::tests::test_weights()),
                None,
            )
            .unwrap();
            std::fs::write(&path, &bytes).unwrap();
            let model =
                CausalDecoder::from_safetensors(&path, CausalDecoderConfig::default()).unwrap();
            let request = model.snapshot_request(&[0, 1], "ar").unwrap();
            let (cache, final_state) = model.prefill_cache_state(&[0, 1]).unwrap();
            assert!(model.snapshot_load(&request).is_some());
            // Même taille et mtime : seul le remplacement inode/dev révèle B.
            let replacement = root.join("replacement");
            std::fs::write(&replacement, &bytes).unwrap();
            std::fs::File::options()
                .write(true)
                .open(&replacement)
                .unwrap()
                .set_modified(std::fs::metadata(&path).unwrap().modified().unwrap())
                .unwrap();
            std::fs::rename(replacement, &path).unwrap();
            #[cfg(unix)]
            {
                assert!(model.snapshot_request(&[0, 1], "ar").unwrap().is_none());
                assert!(model.snapshot_load(&request).is_none());
                let manifest = request.as_ref().unwrap().root.join("manifest.json");
                let previous = std::fs::read(&manifest).unwrap();
                model
                    .snapshot_save(&request, &cache, &final_state, None, None)
                    .unwrap();
                assert_eq!(std::fs::read(&manifest).unwrap(), previous);
                let fresh =
                    CausalDecoder::from_safetensors(&path, CausalDecoderConfig::default()).unwrap();
                assert!(fresh
                    .snapshot_load(&fresh.snapshot_request(&[0, 1], "ar").unwrap())
                    .is_none());
                assert_eq!(
                    model.prefill_cache_state(&[0, 1]).unwrap().1.data(),
                    final_state.data()
                );
            }
            // Une modification de taille est détectée sur toutes les plateformes.
            std::fs::write(&path, b"changed").unwrap();
            assert!(model.snapshot_request(&[0, 1], "ar").unwrap().is_none());
        },
    );
}

#[test]
fn disabled_flag_skips_sources_and_stays_frozen() {
    isolated(
        "disabled_flag_skips_sources_and_stays_frozen",
        false,
        |root| {
            assert!(crate::runtime_flags::state_snapshot_dir().is_none());
            std::env::set_var("SARAGOSSA_RUST_STATE_SNAPSHOT", root);
            let sources = SnapshotSources::capture(std::iter::from_fn(|| -> Option<&Path> {
                panic!("les chemins ne doivent même pas être parcourus sans flag")
            }));
            assert!(sources.0.is_none());
            let (model, _) = fixture(root);
            assert!(model.snapshot_request(&[0, 1], "ar").unwrap().is_none());
            #[cfg(feature = "devtools")]
            assert!(model
                .snapshot_request_dflash(&[0, 1], "dflash", None)
                .unwrap()
                .is_none());
            assert!(std::fs::read_dir(root).unwrap().next().is_none());
        },
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn snapshot_key_uses_effective_chunk_after_environment_change() {
    isolated(
        "snapshot_key_uses_effective_chunk_after_environment_change",
        true,
        |root| {
            assert_eq!(crate::runtime_flags::prefill_resident_chunk_tokens(), 4096);
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let before = model.snapshot_request(&[0, 1], "ar").unwrap().unwrap().key;
            std::env::set_var("SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK", "512");
            let after = model.snapshot_request(&[0, 1], "ar").unwrap().unwrap().key;
            assert_eq!(before, after);
            let mut settings = crate::runtime_flags::state_snapshot_settings();
            let chunk = settings
                .iter_mut()
                .find(|(name, _)| *name == "prefill_resident_chunk_tokens")
                .unwrap();
            assert_eq!(chunk.1, "4096");
            chunk.1 = "512".into();
            settings.sort_by_key(|(name, _)| *name);
            assert_ne!(
                after.runtime_sha256,
                storage::digest(&serde_json::to_vec(&settings).unwrap())
            );
        },
    );
}

#[test]
fn oversized_sparse_blob_rejected_before_any_blob_read() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let (model, request) = fixture(dir.path());
    let (cache, final_state) = model.prefill_cache_state(&[0, 1])?;
    model.snapshot_save(&request, &cache, &final_state, None, None)?;
    let mut manifest = read_manifest(dir.path());
    // La première donnée manque : le diagnostic doit viser la géométrie de la
    // dernière, prouvant que toutes les tailles sont contrôlées avant les I/O.
    std::fs::remove_file(dir.path().join(&manifest.layers[0].keys.segments[0].file)).unwrap();
    let blob = &mut manifest.final_state.data;
    blob.bytes = 1usize << 40;
    std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join(&blob.file))
        .unwrap()
        .set_len(blob.bytes as u64)
        .unwrap();
    write_manifest(dir.path(), &manifest);
    let error = model
        .snapshot_read(request.as_ref().unwrap())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("dimensions attendues"), "{error}");
    assert!(model.snapshot_load(&request).is_none());
    Ok(())
}

#[test]
fn split_kv_rows_with_valid_checksums_fall_back_to_prefill() {
    isolated(
        "split_kv_rows_with_valid_checksums_fall_back_to_prefill",
        true,
        |root| {
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let request = model.snapshot_request(&[0, 1], "ar").unwrap();
            let (cache, final_state) = model.prefill_cache_state(&[0, 1]).unwrap();
            for bf16 in [false, true] {
                for keys in [true, false] {
                    let mut broken = cache.clone();
                    assert_eq!(broken.layers[0].kv_dim, Some(2));
                    let rows = if keys {
                        &mut broken.layers[0].keys
                    } else {
                        &mut broken.layers[0].values
                    };
                    *rows = kv_rows::KvRows::default();
                    if bf16 {
                        rows.push_owned_bf16(vec![0x3f80]);
                        rows.push_owned_bf16(vec![0x3f80; 3]);
                    } else {
                        rows.push_owned(vec![1.0]);
                        rows.push_owned(vec![2.0, 3.0, 4.0]);
                    }
                    model
                        .snapshot_save(&request, &broken, &final_state, None, None)
                        .unwrap();
                    let error = model
                        .snapshot_read(request.as_ref().unwrap())
                        .err()
                        .unwrap()
                        .to_string();
                    assert!(error.contains("lignes entières"), "{error}");
                    HITS.set(0);
                    let (mut fresh, restored) = model.prefill_cache_state(&[0, 1]).unwrap();
                    assert_eq!(HITS.get(), 0, "le snapshot invalide doit être refusé");
                    assert_eq!(restored.data(), final_state.data());
                    model.next_final_state_cached(&mut fresh, 2).unwrap();
                    assert!(
                        model.snapshot_load(&request).is_some(),
                        "le prefill normal remplace le snapshot invalide"
                    );
                }
            }
        },
    );
}

#[test]
fn recurrent_cpu_geometry_roundtrip_and_oversized_ssm_rejected() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let (_, request) = fixture(dir.path());
    let model = CausalDecoder::from_tensors(
        super::super::tests::hybrid_weights(2, 2),
        super::super::tests::hybrid_config(2),
    )?;
    let (cache, final_state) = model.prefill_cache_state(&[0, 1])?;
    model.snapshot_save(&request, &cache, &final_state, None, None)?;
    let restored = model.snapshot_read(request.as_ref().unwrap())?;
    assert_eq!(
        cache.layers[0].linear.conv,
        restored.cache.layers[0].linear.conv
    );
    assert_eq!(
        cache.layers[0].linear.ssm,
        restored.cache.layers[0].linear.ssm
    );
    let mut manifest = read_manifest(dir.path());
    manifest.layers[0].ssm.bytes += 4;
    write_manifest(dir.path(), &manifest);
    assert!(model.snapshot_load(&request).is_none());
    Ok(())
}
