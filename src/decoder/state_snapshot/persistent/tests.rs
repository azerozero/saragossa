fn private_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    dir
}

use super::*;

fn fixture(store: &Store, id: u8) -> (CausalDecoder, Request, CausalDecoderPromptState) {
    let model = CausalDecoder::from_tensors(
        crate::decoder::tests::test_weights(),
        CausalDecoderConfig::default(),
    )
    .unwrap();
    let state = model.prefill_prompt_state_uncached(&[0, 1]).unwrap();
    let request = Request {
        root: store.root.join(format!("entry-{}", storage::digest(&[id]))),
        key: Key {
            version: FORMAT_VERSION,
            prompt_sha256: storage::digest(&[0, 1]),
            tokens: 2,
            sources: vec![],
            binary_sha256: storage::binary_sha256().unwrap().into(),
            mode: SnapshotMode::Ar,
            config: "synthetic".into(),
            kv_dtype: "f32".into(),
            runtime_sha256: "test".into(),
            little_endian: cfg!(target_endian = "little"),
        },
    };
    (model, request, state)
}
fn payload(model: &CausalDecoder, request: &Request, state: &CausalDecoderPromptState) -> Payload {
    let mut writer = Writer::buffered(&request.root, u64::MAX);
    let layers = state
        .cache
        .layers
        .iter()
        .map(|layer| Layer::write(layer, &mut writer, model).unwrap())
        .collect();
    let manifest = Manifest {
        key: request.key.clone(),
        position: state.position(),
        layers,
        final_state: Dense::write(&state.final_state, &mut writer).unwrap(),
        history: None,
        captures: None,
    };
    writer.finish_buffered(&manifest).unwrap()
}
fn wait(store: &Store) {
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while store.busy.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "worker bloqué");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
fn publish(store: &Store, request: &Request, payload: Payload) {
    store
        .submit(request, payload, store.reserve().unwrap())
        .unwrap();
    wait(store);
}

#[test]
fn restart_preserves_bits_and_greedy_tokens() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    publish(&store, &request, payload(&model, &request, &state));
    drop(store);
    let restarted = Store::open(root.path(), 1 << 20, 0).unwrap();
    let restored = restarted.read(&model, &request).unwrap();
    assert_eq!(
        state
            .final_state
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        restored
            .final_state
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    for (before, after) in state.cache.layers.iter().zip(&restored.cache.layers) {
        assert_eq!(
            before.keys.snapshot_segments(),
            after.keys.snapshot_segments()
        );
        assert_eq!(
            before.values.snapshot_segments(),
            after.values.snapshot_segments()
        );
    }
    let generate = |state| {
        model
            .generate_greedy_timed_from_prompt_state_with_options(
                state,
                std::time::Duration::ZERO,
                12,
                &GenerationOptions::default(),
            )
            .unwrap()
            .tokens
    };
    assert_eq!(
        generate(state),
        generate(CausalDecoderPromptState::new(
            restored.cache,
            restored.final_state
        ))
    );
}

#[test]
fn corruption_and_incompatible_headers_are_deleted() {
    for failure in [
        "blob",
        "checksum",
        "version",
        "tokens",
        "dtype",
        "model",
        "oversized",
    ] {
        let root = private_dir();
        let store = Store::open(root.path(), 1 << 20, 0).unwrap();
        let (model, request, state) = fixture(&store, 1);
        let mut data = payload(&model, &request, &state);
        match failure {
            "blob" | "checksum" => {}
            _ => {
                let mut manifest: serde_json::Value =
                    serde_json::from_slice(&data.manifest).unwrap();
                match failure {
                    "version" => manifest["key"]["version"] = 999.into(),
                    "tokens" => manifest["key"]["tokens"] = 999.into(),
                    "dtype" => manifest["key"]["kv_dtype"] = "wrong".into(),
                    "model" => manifest["key"]["config"] = "wrong".into(),
                    "oversized" => {
                        manifest["layers"][0]["keys"]["segments"][0]["bytes"] = u64::MAX.into()
                    }
                    _ => unreachable!(),
                }
                data.manifest = serde_json::to_vec(&manifest).unwrap();
            }
        }
        publish(&store, &request, data);
        if failure == "blob" {
            let manifest: Manifest =
                serde_json::from_slice(&fs::read(request.root.join("manifest.json")).unwrap())
                    .unwrap();
            let blob = &manifest.layers[0].keys.segments[0];
            let path = request.root.join(&blob.file);
            let mut bytes = fs::read(&path).unwrap();
            bytes[0] ^= 1;
            fs::write(path, bytes).unwrap();
        }
        if failure == "checksum" {
            fs::write(request.root.join("manifest.sha256"), "0".repeat(64)).unwrap();
        }
        assert!(store.read(&model, &request).is_none(), "{failure}");
        assert!(!request.root.exists(), "{failure}");
    }
}

#[test]
fn budget_priority_and_lru_survive_restart() {
    let root = private_dir();
    let initial = Store::open(root.path(), 1 << 20, 2).unwrap();
    let (model, first, state) = fixture(&initial, 1);
    let size = payload(&model, &first, &state).bytes();
    drop(initial);
    let store = Store::open(root.path(), size * 2, 2).unwrap();
    let (_, mut second, _) = fixture(&store, 2);
    let (_, mut third, _) = fixture(&store, 3);
    // La priorité est une décision du serveur ; le contenu n'est pas chargé ici.
    second.key.tokens = 3;
    third.key.tokens = 3;
    publish(&store, &first, payload(&model, &first, &state));
    publish(&store, &second, payload(&model, &second, &state));
    publish(&store, &third, payload(&model, &third, &state));
    assert!(first.root.exists());
    assert!(!second.root.exists());
    assert!(third.root.exists());
    assert!(tree_bytes(root.path()).is_err()); // profondeur supérieure au format d'une entrée
    let used: u64 = store
        .disk
        .lock()
        .unwrap()
        .entries
        .iter()
        .map(|e| e.bytes)
        .sum();
    assert!(used <= size * 2);
    drop(store);
    let restarted = Store::open(root.path(), size, 2).unwrap();
    assert!(first.root.exists());
    assert!(!third.root.exists());
    assert_eq!(restarted.disk.lock().unwrap().entries.len(), 1);
}

#[test]
fn lru_touch_changes_the_victim() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (model, first, state) = fixture(&store, 1);
    let (_, second, _) = fixture(&store, 2);
    let (_, third, _) = fixture(&store, 3);
    let size = payload(&model, &first, &state).bytes();
    publish(&store, &first, payload(&model, &first, &state));
    publish(&store, &second, payload(&model, &second, &state));
    assert!(store.read(&model, &first).is_some());
    let mut disk = store.disk.lock().unwrap();
    disk.write(
        &store.root,
        size * 2,
        &Job {
            tokens: third.key.tokens,
            name: third.root.file_name().unwrap().to_str().unwrap().into(),
            priority: false,
            payload: payload(&model, &third, &state),
            _permit: store.reserve().unwrap(),
        },
    )
    .unwrap();
    assert!(first.root.exists());
    assert!(!second.root.exists());
    assert!(third.root.exists());
}

#[test]
fn admission_is_bounded_and_read_does_not_wait_for_writer() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    publish(&store, &request, payload(&model, &request, &state));
    let permit = store.reserve().unwrap();
    assert!(store.reserve().is_none());
    let guard = store.disk.lock().unwrap();
    assert!(store.read(&model, &request).is_none());
    drop(guard);
    drop(permit);
    assert!(store.reserve().is_some());
    assert!(Store::open(root.path(), 1 << 20, 0).is_err());
}

#[test]
fn interrupted_publication_and_oversized_entry() {
    let root = private_dir();
    fs::create_dir(root.path().join("pending")).unwrap();
    fs::write(root.path().join("pending/partial"), b"interrupted").unwrap();
    let store = Store::open(root.path(), 1, 0).unwrap();
    assert!(!root.path().join("pending").exists());
    let (model, request, state) = fixture(&store, 1);
    publish(&store, &request, payload(&model, &request, &state));
    assert!(!request.root.exists());
    let mut writer = Writer::buffered(&request.root, 1);
    assert!(writer.f32(&[1.0]).is_err());
}

#[test]
#[cfg(unix)]
fn symlinks_and_public_directory_are_refused() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let root = private_dir();
    let linked = root.path().join("linked");
    symlink(root.path(), &linked).unwrap();
    assert!(Store::open(&linked, 1 << 20, 0).is_err());
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(Store::open(root.path(), 1 << 20, 0).is_err());
}

#[test]
fn session_scope_is_hashed_nested_and_thread_local() {
    let before = session();
    let scope = PrefixDiskSession::enter(Some("../../network/path"));
    assert_eq!(session().len(), 64);
    let outer = session();
    {
        let _nested = PrefixDiskSession::enter(Some("other"));
        assert_ne!(session(), outer);
    }
    assert_eq!(session(), outer);
    assert!(std::thread::spawn(session).join().unwrap().is_empty());
    drop(scope);
    assert_eq!(session(), before);
}

#[test]
fn hybrid_recurrence_survives_background_publication() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (_, request, _) = fixture(&store, 1);
    let model = CausalDecoder::from_tensors(
        crate::decoder::tests::hybrid_weights(2, 2),
        crate::decoder::tests::hybrid_config(2),
    )
    .unwrap();
    let state = model.prefill_prompt_state_uncached(&[0, 1]).unwrap();
    publish(&store, &request, payload(&model, &request, &state));
    let restored = store.read(&model, &request).unwrap();
    for (before, after) in state.cache.layers.iter().zip(&restored.cache.layers) {
        assert_eq!(
            before
                .linear
                .conv
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            after
                .linear
                .conv
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            before
                .linear
                .ssm
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            after
                .linear
                .ssm
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    let generate = |state| {
        model
            .generate_greedy_timed_from_prompt_state_with_options(
                state,
                std::time::Duration::ZERO,
                12,
                &GenerationOptions::default(),
            )
            .unwrap()
            .tokens
    };
    assert_eq!(
        generate(state),
        generate(CausalDecoderPromptState::new(
            restored.cache,
            restored.final_state
        ))
    );
}

#[test]
fn process_restart_uses_persistent_hooks() {
    process_oracle(false);
}

#[test]
#[cfg(feature = "devtools")]
fn process_restart_dflash_preserves_captures() {
    process_oracle(true);
}

fn process_oracle(dflash: bool) {
    let phase = std::env::var("SSD_TEST_PHASE").ok();
    if let Some(phase) = phase {
        let root = PathBuf::from(std::env::var_os("SSD_TEST_ROOT").unwrap());
        if phase != "baseline" {
            configure_prefix_disk_cache(&root.join("cache"), 1 << 20, 0).unwrap();
        }
        let model = CausalDecoder::from_tensors(
            crate::decoder::tests::test_weights(),
            CausalDecoderConfig::default(),
        )
        .unwrap()
        .with_snapshot_sources(SnapshotSources::capture([root.join("weights").as_path()]));
        let _scope = PrefixDiskSession::enter(Some("session"));
        let generate_ar = || {
            model
                .generate_greedy_timed_with_options(&[0, 1], 12, &GenerationOptions::default())
                .unwrap()
                .tokens
        };
        #[cfg(feature = "devtools")]
        let tokens = if dflash {
            let mut draft = crate::decoder::tests::tiny_dflash_draft();
            draft.snapshot_sources = SnapshotSources::capture([root.join("weights").as_path()]);
            model
                .generate_greedy_dflash_batched_with_options(
                    &[0, 1],
                    12,
                    &GenerationOptions::default(),
                    &draft,
                    2,
                )
                .unwrap()
                .tokens
        } else {
            generate_ar()
        };
        #[cfg(not(feature = "devtools"))]
        let tokens = generate_ar();
        if let Some(store) = store() {
            wait(store);
        }
        assert_eq!(super::super::HITS.get(), usize::from(phase == "warm"));
        if !dflash && phase != "baseline" {
            model.clear_prefix_cache();
            let prefix = model
                .load_persistent_prefix(&[0, 1, 2], 1, 0)
                .unwrap()
                .unwrap();
            assert_eq!(prefix.position(), 2, "frontière SSD après éviction RAM");
            assert!(model
                .load_persistent_prefix(&[0, 1, 2], 1, 2)
                .unwrap()
                .is_none());
        }
        fs::write(
            root.join(format!("{phase}.json")),
            serde_json::to_vec(&tokens).unwrap(),
        )
        .unwrap();
        return;
    }
    let root = private_dir();
    fs::write(root.path().join("weights"), b"synthetic weights identity").unwrap();
    for phase in ["baseline", "cold", "warm"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                if dflash { "decoder::state_snapshot::persistent::tests::process_restart_dflash_preserves_captures" } else { "decoder::state_snapshot::persistent::tests::process_restart_uses_persistent_hooks" },
                "--nocapture",
            ])
            .env("SSD_TEST_ROOT", root.path())
            .env("SSD_TEST_PHASE", phase)
            .env_remove("SARAGOSSA_RUST_STATE_SNAPSHOT")
            .env_remove("RETI_RUST_STATE_SNAPSHOT")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(
        fs::read(root.path().join("baseline.json")).unwrap(),
        fs::read(root.path().join("cold.json")).unwrap()
    );
    assert_eq!(
        fs::read(root.path().join("baseline.json")).unwrap(),
        fs::read(root.path().join("warm.json")).unwrap()
    );
}

#[test]
fn concurrent_producers_admit_only_one_snapshot() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let ready = std::sync::Barrier::new(8);
    let hold = std::sync::Barrier::new(8);
    let admitted = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                ready.wait();
                let permit = store.reserve();
                if permit.is_some() {
                    admitted.fetch_add(1, Ordering::Relaxed);
                }
                hold.wait();
                drop(permit);
            });
        }
    });
    assert_eq!(admitted.load(Ordering::Relaxed), 1);
    assert!(store.reserve().is_some());
}
