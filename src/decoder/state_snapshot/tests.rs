use super::SnapshotMode::Ar;
use super::*;
use crate::runtime_flags::SNAPSHOT_ABLATION_FLAGS;

fn fixture(root: &Path) -> (CausalDecoder, Option<Request>) {
    let model = CausalDecoder::from_tensors(
        super::super::tests::test_weights(),
        CausalDecoderConfig::default(),
    )
    .unwrap();
    let request = Request {
        root: root.to_owned(),
        key: Key {
            version: FORMAT_VERSION,
            prompt_sha256: "test".into(),
            tokens: 2,
            sources: Vec::new(),
            binary_sha256: "build-a".into(),
            mode: Ar,
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
    );
    let request = request.unwrap();
    let loaded = model.snapshot_read(&request)?;
    assert_eq!(loaded.cache.position, 2);
    assert_eq!(
        loaded.cache.layers[0].keys.snapshot_segments()[0].0,
        BlobDtype::Bf16
    );
    assert_eq!(
        loaded.cache.layers[0].keys.snapshot_segments()[0].1,
        cache.layers[0].keys.snapshot_segments()[0].1
    );
    assert_eq!(loaded.final_state.data(), final_state.data());
    let restored_history = loaded.history.expect("historique restauré");
    assert_eq!(
        restored_history.keys.snapshot_segments(),
        history.keys.snapshot_segments()
    );
    assert_eq!(
        restored_history.values.snapshot_segments(),
        history.values.snapshot_segments()
    );
    assert!(!super::super::mtp_history::compatible(&restored_history));
    super::super::mtp_history::with_storage(Some(16), || {
        assert!(super::super::mtp_history::compatible(&restored_history));
    });
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
    std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("manifest.json"))
        .unwrap()
        .set_len(MAX_MANIFEST_BYTES + 1)
        .unwrap();
    assert!(model
        .snapshot_read(&request)
        .err()
        .unwrap()
        .to_string()
        .contains("manifeste trop grand"));
    assert!(storage::f32_values(&[0; 3]).is_err());
    assert!(storage::bf16_values(&[0]).is_err());
    Ok(())
}

#[test]
fn publication_incomplete_keeps_previous_manifest() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::new(dir.path())?;
    let blob = writer.blob(BlobDtype::Bf16, &[0, 128, 128, 63])?;
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
    let phase = std::env::var("SNAPSHOT_ORACLE_PHASE").ok();
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
                .env("SNAPSHOT_ORACLE_PHASE", phase)
                .env("SNAPSHOT_ORACLE_OUTPUT", &output_path)
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
    let output = std::env::var_os("SNAPSHOT_ORACLE_OUTPUT").unwrap();
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
    model.snapshot_save(&request, &cache, &final_state, None, None);
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
fn isolated(name: &str, enabled: bool, run: impl FnOnce(&Path)) -> String {
    if let Some(root) = std::env::var_os("SARAGOSSA_SNAPSHOT_TEST_CHILD") {
        run(Path::new(&root));
        return String::new();
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
    String::from_utf8(output.stderr).unwrap()
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
            let request = model.snapshot_request(&[0, 1], Ar).unwrap();
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
                assert!(model.snapshot_request(&[0, 1], Ar).unwrap().is_none());
                assert!(model.snapshot_load(&request).is_none());
                let manifest = request.as_ref().unwrap().root.join("manifest.json");
                let previous = std::fs::read(&manifest).unwrap();
                model.snapshot_save(&request, &cache, &final_state, None, None);
                assert_eq!(std::fs::read(&manifest).unwrap(), previous);
                let fresh =
                    CausalDecoder::from_safetensors(&path, CausalDecoderConfig::default()).unwrap();
                assert!(fresh
                    .snapshot_load(&fresh.snapshot_request(&[0, 1], Ar).unwrap())
                    .is_none());
                assert_eq!(
                    model.prefill_cache_state(&[0, 1]).unwrap().1.data(),
                    final_state.data()
                );
            }
            // Une modification de taille est détectée sur toutes les plateformes.
            std::fs::write(&path, b"changed").unwrap();
            assert!(model.snapshot_request(&[0, 1], Ar).unwrap().is_none());
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
            assert!(model.snapshot_request(&[0, 1], Ar).unwrap().is_none());
            #[cfg(feature = "devtools")]
            assert!(model
                .snapshot_request_dflash(&[0, 1], SnapshotMode::DflashReference, None)
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
            // Le premier flag capture aussi les variables inconnues et les alias.
            std::env::set_var("SARAGOSSA_FUTURE_SETTING", "initial");
            std::env::set_var("RETI_FUTURE_SETTING", "legacy");
            let mut initial: Vec<_> = std::env::vars_os()
                .filter(|(key, _)| {
                    let key = key.as_encoded_bytes();
                    key.starts_with(b"SARAGOSSA_") || key.starts_with(b"RETI_")
                })
                .collect();
            initial.sort_unstable();
            let mut hash = Sha256::new();
            for (key, value) in initial {
                for field in [key, value] {
                    let field = field.as_encoded_bytes();
                    hash.update((field.len() as u64).to_le_bytes());
                    hash.update(field);
                }
            }
            let environment = format!("{:x}", hash.finalize());
            assert_eq!(crate::runtime_flags::prefill_resident_chunk_tokens(), 4096);
            // Aucune demande de clé avant la mutation : c'est le premier flag
            // qui doit avoir capturé l'environnement initial entier.
            std::env::set_var("SARAGOSSA_FUTURE_SETTING", "changed");
            std::env::remove_var("RETI_FUTURE_SETTING");
            std::env::set_var("SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK", "512");
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let before = model.snapshot_request(&[0, 1], Ar).unwrap().unwrap().key;
            std::env::set_var("SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK", "1024");
            let after = model.snapshot_request(&[0, 1], Ar).unwrap().unwrap().key;
            assert_eq!(before, after);
            assert_eq!(
                after.runtime_sha256,
                storage::digest(
                    format!(
                        "{environment}:None:{}",
                        crate::runtime_flags::state_snapshot_overrides()
                    )
                    .as_bytes()
                )
            );
            assert_eq!(crate::runtime_flags::prefill_resident_chunk_tokens(), 4096);
            assert_eq!(environment, crate::runtime_flags::environment_sha256());
        },
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal", feature = "devtools"))]
fn snapshot_key_tracks_pairwise_overrides_without_environment_changes() {
    isolated(
        "snapshot_key_tracks_pairwise_overrides_without_environment_changes",
        true,
        |root| {
            use crate::runtime_flags::pairwise::{Case, Guard};
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let request = || model.snapshot_request(&[0, 1], Ar).unwrap();
            let baseline = request();
            let (cache, state) = model.prefill_cache_state(&[0, 1]).unwrap();
            model.snapshot_save(&baseline, &cache, &state, None, None);
            let environment = crate::runtime_flags::environment_sha256();
            {
                let case = Case::default();
                let _guard = Guard::set(case, false).unwrap();
                let active = request();
                assert_ne!(baseline.as_ref().unwrap().key, active.as_ref().unwrap().key);
                model.snapshot_save(&active, &cache, &state, None, None);
                assert!(model.snapshot_load(&active).is_some());
                // Même mode, même prompt, même environnement : chaque surcharge
                // doit refuser l'état du cas précédent, notamment les fusions p1.
                for variant in [
                    Case {
                        p1: !case.p1,
                        ..case
                    },
                    Case {
                        p2: !case.p2,
                        ..case
                    },
                    Case {
                        p3: !case.p3,
                        ..case
                    },
                    Case {
                        hidden: !case.hidden,
                        ..case
                    },
                    Case {
                        embedding: !case.embedding,
                        ..case
                    },
                    Case {
                        overlap: !case.overlap,
                        ..case
                    },
                    Case { lookup: 1, ..case },
                    Case { cap: 2, ..case },
                ] {
                    let _variant = Guard::set(variant, false).unwrap();
                    let changed = request();
                    assert_ne!(active.as_ref().unwrap().key, changed.as_ref().unwrap().key);
                    assert!(model.snapshot_load(&changed).is_none());
                    assert_eq!(environment, crate::runtime_flags::environment_sha256());
                }
                assert_eq!(active.as_ref().unwrap().key, request().unwrap().key);
                assert!(model.snapshot_load(&active).is_some());
            }
            assert_eq!(baseline.as_ref().unwrap().key, request().unwrap().key);
            assert!(model.snapshot_load(&baseline).is_none());
        },
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn snapshot_key_tracks_forced_resident_linear() {
    isolated("snapshot_key_tracks_forced_resident_linear", true, |root| {
        std::env::set_var("SARAGOSSA_RUST_DECODE_RESIDENT_FULL_LINEAR", "0");
        assert!(!crate::runtime_flags::decode_resident_full_linear_enabled());
        let path = root.join("weights");
        std::fs::write(&path, b"synthetic").unwrap();
        let (model, _) = fixture(root);
        let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
        let before = model.snapshot_request(&[0, 1], Ar).unwrap();
        let (cache, state) = model.prefill_cache_state(&[0, 1]).unwrap();
        model.snapshot_save(&before, &cache, &state, None, None);
        assert!(model.snapshot_load(&before).is_some());
        let environment = crate::runtime_flags::environment_sha256();
        crate::runtime_flags::force_resident_full_linear_decode();
        assert!(crate::runtime_flags::decode_resident_full_linear_enabled());
        let after = model.snapshot_request(&[0, 1], Ar).unwrap();
        assert_ne!(before.as_ref().unwrap().key, after.as_ref().unwrap().key);
        assert!(model.snapshot_load(&after).is_none());
        assert_eq!(environment, crate::runtime_flags::environment_sha256());
    });
}

// NOTE: NTFS ne crée pas de fichier creux par défaut : le blob géant y occupe
// réellement le disque. Le contrôle de taille testé ici est indépendant de l'OS.
#[test]
#[cfg(unix)]
fn oversized_sparse_blob_rejected_before_any_blob_read() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let (model, request) = fixture(dir.path());
    let (cache, final_state) = model.prefill_cache_state(&[0, 1])?;
    model.snapshot_save(&request, &cache, &final_state, None, None);
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
            let request = model.snapshot_request(&[0, 1], Ar).unwrap();
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
                    model.snapshot_save(&request, &broken, &final_state, None, None);
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
    model.snapshot_save(&request, &cache, &final_state, None, None);
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

#[test]
fn content_identity_covers_shards_auxiliaries_and_sidecars() {
    isolated(
        "content_identity_covers_shards_auxiliaries_and_sidecars",
        true,
        |root| {
            use std::io::{Seek, SeekFrom, Write};
            let draft = root.join("draft");
            std::fs::create_dir(&draft).unwrap();
            let paths = [
                root.join("shard-1.safetensors"),
                root.join("shard-2.safetensors"),
                root.join("model.safetensors.index.json"),
                root.join("config.json"),
                root.join("mtp.safetensors"),
                draft.join("model.safetensors"),
                draft.join("config.json"),
            ];
            for path in &paths {
                std::fs::write(path, b"synthetic A").unwrap();
            }
            let capture = || {
                let mut sources =
                    SnapshotSources::capture([paths[0].as_path(), paths[1].as_path()]);
                sources.append(SnapshotSources::capture([paths[4].as_path()]));
                sources.append(SnapshotSources::capture([paths[5].as_path()]));
                sources
            };
            for path in &paths {
                let original = capture();
                let (model, mut request) = fixture(&root.join("snapshots"));
                let (cache, state) = model.prefill_cache_state(&[0, 1]).unwrap();
                request.as_mut().unwrap().key.sources = original.verified().unwrap().to_vec();
                let model = model.with_snapshot_sources(original.clone());
                model.snapshot_save(&request, &cache, &state, None, None);
                assert!(model.snapshot_load(&request).is_some());
                let manifest = request.as_ref().unwrap().root.join("manifest.json");
                let previous = std::fs::read(&manifest).unwrap();
                let metadata = std::fs::metadata(path).unwrap();
                let mut file = std::fs::File::options().write(true).open(path).unwrap();
                file.seek(SeekFrom::End(-1)).unwrap();
                file.write_all(b"B").unwrap();
                file.set_modified(metadata.modified().unwrap()).unwrap();
                let after = file.metadata().unwrap();
                assert_eq!(metadata.len(), after.len());
                assert_eq!(metadata.modified().unwrap(), after.modified().unwrap());
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    assert_eq!(metadata.ino(), after.ino());
                    assert_eq!(metadata.dev(), after.dev());
                    // Le contrôle post-chargement refuse aussi une course à mtime restauré.
                    assert!(original.clone().finish_loading().verified().is_err());
                    assert!(model.snapshot_request(&[0, 1], Ar).unwrap().is_none());
                    assert!(model.snapshot_load(&request).is_none());
                    model.snapshot_save(&request, &cache, &state, None, None);
                    assert_eq!(previous, std::fs::read(&manifest).unwrap());
                }
                let changed = capture();
                let old_sources = original.0.as_ref().unwrap().as_ref().unwrap();
                let new_sources = changed.verified().unwrap();
                // Vérifier le hash lui-même : le ctime ne doit pas masquer une
                // régression vers une identité purement fondée sur les métadonnées.
                let hashes = |sources: &[Source]| {
                    sources
                        .iter()
                        .filter(|source| source.path() == path)
                        .map(|source| {
                            serde_json::to_value(source).unwrap()["content_sha256"].clone()
                        })
                        .collect::<Vec<_>>()
                };
                assert!(!hashes(old_sources).is_empty());
                assert_ne!(hashes(old_sources), hashes(new_sources));
                let (fresh, mut fresh_request) = fixture(&root.join("snapshots"));
                fresh_request.as_mut().unwrap().key.sources = new_sources.to_vec();
                let fresh = fresh.with_snapshot_sources(changed);
                assert!(fresh.snapshot_load(&fresh_request).is_none());
            }
        },
    );
}

#[test]
fn diagnostic_ablations_disable_reads_writes_and_hashing() {
    let flags: Vec<_> = SNAPSHOT_ABLATION_FLAGS
        .iter()
        .zip(["0", "", "full_layer", "1", "true"])
        .flat_map(|(flag, value)| {
            [
                (flag.to_string(), value),
                (flag.replacen("SARAGOSSA_", "RETI_", 1), value),
            ]
        })
        .collect();
    let Some(root) = std::env::var_os("SARAGOSSA_SNAPSHOT_TEST_CHILD") else {
        for (flag, value) in &flags {
            let dir = tempfile::tempdir().unwrap();
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            for (name, _) in &flags {
                command.env_remove(name);
            }
            let output = command.args([
                "--exact", "decoder::state_snapshot::tests::diagnostic_ablations_disable_reads_writes_and_hashing",
                "--nocapture",
            ])
                .env("SARAGOSSA_SNAPSHOT_TEST_CHILD", dir.path())
                .env("SARAGOSSA_RUST_STATE_SNAPSHOT", dir.path().join("snapshots"))
                .env(flag, value).output().unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "{flag}: {}\n{stderr}",
                String::from_utf8_lossy(&output.stdout)
            );
            assert!(stderr.contains("instantané désactivé"), "{stderr}");
            assert!(!stderr.contains("state_snapshot hash_ms="), "{stderr}");
        }
        return;
    };
    let root = Path::new(&root);
    // Le calcul lit le flag avant toute demande d'instantané. Son retrait ne
    // doit pas blanchir un état qui survivrait dans le cache de préfixe mémoire.
    let (active, _) = flags
        .iter()
        .find(|(flag, _)| std::env::var_os(flag).is_some())
        .unwrap();
    let canonical = active.replacen("RETI_", "SARAGOSSA_", 1);
    assert!(crate::runtime_flags::env_var(&canonical).is_some());
    std::env::remove_var(active);
    assert_eq!(
        crate::runtime_flags::state_snapshot_ablation(),
        Some(canonical.as_str())
    );
    let sources = SnapshotSources::capture(std::iter::from_fn(|| -> Option<&Path> {
        panic!("aucune lecture source sous ablation");
    }));
    assert!(sources.0.unwrap().is_err());
    let (model, request) = fixture(&root.join("snapshots"));
    assert!(model.snapshot_request(&[0, 1], Ar).unwrap().is_none());
    #[cfg(feature = "devtools")]
    assert!(model
        .snapshot_request_dflash(&[0, 1], SnapshotMode::DflashReference, None)
        .unwrap()
        .is_none());
    assert!(model.snapshot_load(&request).is_none());
    let (cache, state) = model.prefill_cache_state(&[0, 1]).unwrap();
    model.snapshot_save(&request, &cache, &state, None, None);
    assert!(!root.join("snapshots").exists());
}

#[test]
fn late_moe_ablation_refuses_an_existing_snapshot() {
    isolated(
        "late_moe_ablation_refuses_an_existing_snapshot",
        true,
        |root| {
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let request = model.snapshot_request(&[0, 1], Ar).unwrap();
            let (cache, state) = model.prefill_cache_state(&[0, 1]).unwrap();
            assert!(model.snapshot_load(&request).is_some());
            let manifest = request.as_ref().unwrap().root.join("manifest.json");
            let previous = std::fs::read(&manifest).unwrap();
            std::env::set_var(SNAPSHOT_ABLATION_FLAGS[0], "1");
            assert!(model.snapshot_load(&request).is_none());
            std::env::remove_var(SNAPSHOT_ABLATION_FLAGS[0]);
            assert!(model.snapshot_load(&request).is_none());
            assert!(model.snapshot_request(&[0, 1], Ar).unwrap().is_none());
            model.snapshot_save(&request, &cache, &state, None, None);
            assert_eq!(std::fs::read(&manifest).unwrap(), previous);
            // Le préfixe peuplé avant l'ablation est réutilisable en mémoire,
            // mais ne doit plus pouvoir être publié après retrait du flag.
            assert!(!model.prefix_cache.lock().unwrap().entries.is_empty());
            std::fs::remove_dir_all(request.as_ref().unwrap().root.clone()).unwrap();
            model.prefill_cache_state(&[0, 1]).unwrap();
            assert!(!manifest.exists());
        },
    );
}

#[test]
fn missing_entry_is_absent_but_missing_blob_is_refused() {
    let log = isolated(
        "missing_entry_is_absent_but_missing_blob_is_refused",
        true,
        |root| {
            let (model, request) = fixture(&root.join("snapshots"));
            assert!(model.snapshot_load(&request).is_none());
            let (cache, state) = model.prefill_cache_state(&[0, 1]).unwrap();
            model.snapshot_save(&request, &cache, &state, None, None);
            let manifest = read_manifest(&root.join("snapshots"));
            std::fs::remove_file(root.join("snapshots").join(&manifest.final_state.data.file))
                .unwrap();
            assert!(model.snapshot_load(&request).is_none());
        },
    );
    if std::env::var_os("SARAGOSSA_SNAPSHOT_TEST_CHILD").is_none() {
        assert_eq!(
            log.matches("state_snapshot absent mode=ar tokens=2")
                .count(),
            1,
            "{log}"
        );
        assert!(log.contains("state_snapshot refusé : I/O"), "{log}");
    }
}

#[test]
fn warmup_skips_snapshots_and_restores_scope_after_error() {
    isolated(
        "warmup_skips_snapshots_and_restores_scope_after_error",
        true,
        |root| {
            use crate::runtime_flags::{state_snapshot_warmup, StateSnapshotWarmup};
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let prompt = vec![0; 32];
            let request = model.snapshot_request(&prompt, Ar).unwrap();
            let entry = &request.as_ref().unwrap().root;
            let warmup = || -> Result<()> {
                let _outer = StateSnapshotWarmup::enter();
                {
                    let _inner = StateSnapshotWarmup::enter();
                    assert!(model.snapshot_request(&prompt, Ar)?.is_none());
                }
                assert!(state_snapshot_warmup());
                let (cache, state) = model.prefill_cache_state(&prompt)?;
                model.snapshot_save(&request, &cache, &state, None, None);
                assert!(!entry.exists());
                // L'erreur doit aussi restaurer la portée du vrai prompt.
                Err(invalid("fin de chauffe synthétique"))
            };
            assert!(warmup().is_err());
            assert!(!state_snapshot_warmup());
            model.prefill_cache_state(&prompt).unwrap();
            assert!(model.snapshot_load(&request).is_some());
            let manifest = entry.join("manifest.json");
            let previous = std::fs::read(&manifest).unwrap();
            {
                let _warmup = StateSnapshotWarmup::enter();
                HITS.set(0);
                assert!(model.snapshot_load(&request).is_none());
                model.prefill_cache_uncached(&prompt).unwrap();
                assert_eq!(HITS.get(), 0);
                assert_eq!(std::fs::read(&manifest).unwrap(), previous);
            }
            assert!(model.snapshot_load(&request).is_some());
        },
    );
}

#[test]
#[cfg(unix)]
fn unwritable_snapshot_keeps_generation_without_manifest() {
    let log = isolated(
        "unwritable_snapshot_keeps_generation_without_manifest",
        true,
        |root| {
            use std::os::unix::fs::PermissionsExt;
            let path = root.join("weights");
            std::fs::write(&path, b"synthetic").unwrap();
            let (model, _) = fixture(root);
            let model = model.with_snapshot_sources(SnapshotSources::capture([path.as_path()]));
            let generate = || {
                model.generate_greedy_timed_with_options(&[0, 1], 4, &GenerationOptions::default())
            };
            let baseline = {
                let _warmup = crate::runtime_flags::StateSnapshotWarmup::enter();
                generate().unwrap().tokens
            };
            let request = model.snapshot_request(&[0, 1], Ar).unwrap().unwrap();
            std::fs::create_dir_all(&request.root).unwrap();
            let permissions = std::fs::metadata(&request.root).unwrap().permissions();
            std::fs::set_permissions(&request.root, std::fs::Permissions::from_mode(0o555))
                .unwrap();
            let generated = generate();
            std::fs::set_permissions(&request.root, permissions).unwrap();
            assert_eq!(generated.unwrap().tokens, baseline);
            assert!(!request.root.join("manifest.json").exists());
            assert_eq!(std::fs::read_dir(&request.root).unwrap().count(), 0);
            // Échec tardif du renommage, après écriture des blobs : même garantie.
            std::fs::create_dir(request.root.join("manifest.json")).unwrap();
            assert_eq!(generate().unwrap().tokens, baseline);
            assert_eq!(std::fs::read_dir(&request.root).unwrap().count(), 1);
        },
    );
    if std::env::var_os("SARAGOSSA_SNAPSHOT_TEST_CHILD").is_none() {
        assert_eq!(log.matches("; sans écriture").count(), 2, "{log}");
    }
}
