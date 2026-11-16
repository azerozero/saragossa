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
            position_sha256: None,
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
        rope_delta: 0,
        vision_positions: None,
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
            _memory: None,
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
fn startup_ignores_entries_that_cannot_be_removed() {
    use std::os::unix::fs::{symlink, PermissionsExt};

    for failure in [
        "pending",
        "not_directory",
        "tree",
        "missing_manifest",
        "oversized",
        "header",
    ] {
        let root = private_dir();
        let budget = 1 << 20;
        let initial = Store::open(root.path(), budget, 0).unwrap();
        let (model, request, state) = fixture(&initial, 1);
        publish(&initial, &request, payload(&model, &request, &state));
        drop(initial);
        let rejected = root.path().join(if failure == "pending" {
            "pending".into()
        } else {
            format!("entry-{}", storage::digest(&[2]))
        });
        if failure == "not_directory" {
            fs::write(&rejected, b"invalid").unwrap();
        } else {
            fs::create_dir(&rejected).unwrap();
            fs::write(rejected.join("manifest.json"), b"{}").unwrap();
            match failure {
                "tree" => symlink("manifest.json", rejected.join("linked")).unwrap(),
                "missing_manifest" => fs::remove_file(rejected.join("manifest.json")).unwrap(),
                "oversized" => File::create(rejected.join("large"))
                    .unwrap()
                    .set_len(budget + 1)
                    .unwrap(),
                "header" => fs::write(rejected.join("manifest.sha256"), b"invalid").unwrap(),
                _ => {}
            }
        }
        // Le verrou existe déjà. La racine reste lisible, mais ses enfants ne
        // peuvent pas être supprimés : l'ouverture doit conserver l'entrée saine.
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o500)).unwrap();
        let restarted = Store::open(root.path(), budget, 0);
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let restarted = restarted.unwrap_or_else(|error| panic!("{failure}: {error}"));
        assert!(
            rejected.exists(),
            "{failure}: suppression réellement refusée"
        );
        assert_eq!(restarted.disk.lock().unwrap().entries.len(), 1, "{failure}");
        assert!(restarted.read(&model, &request).is_some(), "{failure}");
    }
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
            configure_prefix_disk_cache(&root.join("cache"), 64 << 20, 0).unwrap();
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
            {
                let _other = PrefixDiskSession::enter(Some("session-B"));
                assert!(
                    model
                        .load_persistent_prefix(&[0, 1, 2], 1, 0)
                        .unwrap()
                        .is_none(),
                    "la frontière de session A doit être invisible à B"
                );
            }
            assert!(
                model
                    .load_persistent_prefix(&[0, 1, 2], 1, 0)
                    .unwrap()
                    .is_some(),
                "le retour en session A doit retrouver la même frontière"
            );
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

#[test]
fn default_staging_admits_32k_and_128k_on_128_gib_host() {
    // Tailles seules : aucune allocation de plusieurs Gio ni accès GPU.
    let host = 128 << 30;
    let staging = crate::runtime_flags::prefix_ssd_staging_bytes(None, Some(host), None).unwrap();
    assert_eq!(staging, 32 << 30);
    let guard = crate::memory_guard::MemoryGuard::from_limits(
        true,
        crate::memory_guard::MemoryLimits {
            static_cap: Some(host * 9 / 10),
            host_memory: Some(host),
            headroom_bytes: 2 << 30,
            metal_cap: None,
        },
        Arc::new(|| Some(48 << 30)),
    );
    for scale in [1, 4] {
        // 5,3 Gio à 32k, puis quatre fois plus à 128k ; inclure la marge
        // manifeste et 1 Gio de temporaires supplémentaires dans le pic RAM.
        let blobs = (53 * (1u64 << 30) / 10) * scale;
        let disk = blobs + MAX_MANIFEST_BYTES + 64;
        let ram = blobs + MAX_MANIFEST_BYTES * 8 + (1 << 30);
        assert!(check_staging_limits(disk, ram, 32 << 30, staging, 64 << 30).is_ok());
        let reservation = guard.reserve_allocation(ram).unwrap();
        drop(reservation);
    }
    assert!(check_staging_limits(1, staging + 1, 32 << 30, staging, 64 << 30).is_err());
}

#[test]
fn staging_budget_respects_memory_cap_override_and_fallback() {
    use crate::runtime_flags::prefix_ssd_staging_bytes as limit;
    assert_eq!(
        limit(None, Some(128 << 30), Some(64 << 30)).unwrap(),
        16 << 30
    );
    assert_eq!(
        limit(None, Some(128 << 30), Some(256 << 30)).unwrap(),
        32 << 30
    );
    assert_eq!(limit(None, None, Some(64 << 30)).unwrap(), 16 << 30);
    assert_eq!(limit(None, None, None).unwrap(), 512 << 20);
    let explicit = limit(Some("512"), Some(128 << 30), None).unwrap();
    assert_eq!(explicit, 512 << 20);
    assert!(check_staging_limits(5 << 30, 6 << 30, 32 << 30, explicit, 64 << 30).is_err());
    assert_eq!(
        limit(Some("32768"), Some(16 << 30), Some(8 << 30)).unwrap(),
        32 << 30
    );
    for invalid in ["0", "", "-1", "abc", "18446744073709551615"] {
        assert!(limit(Some(invalid), Some(128 << 30), None).is_err());
    }
}

#[test]
fn staging_reclaims_lru_space_without_counting_protected_entries() {
    let root = private_dir();
    let initial = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (model, first, state) = fixture(&initial, 1);
    let size = payload(&model, &first, &state).bytes();
    drop(initial);
    let mut store = Store::open(root.path(), size * 2, 0).unwrap();
    let (_, second, _) = fixture(&store, 2);
    let (_, mut candidate, _) = fixture(&store, 3);
    publish(&store, &first, payload(&model, &first, &state));
    publish(&store, &second, payload(&model, &second, &state));
    store.disk.lock().unwrap().entries[0].used = SystemTime::UNIX_EPOCH;

    // Le volume est plein, mais l'éviction précédant l'écriture libère le LRU.
    let reservation = store
        .admit_staging_with_space(&candidate, size, 1, 0)
        .unwrap();
    publish(&store, &candidate, payload(&model, &candidate, &state));
    drop(reservation);
    assert!(!first.root.exists());
    assert!(second.root.exists());
    assert!(store.read(&model, &candidate).is_some());

    // Isoler la capacité physique de la limite du cache : même avec un budget
    // disponible, un candidat ordinaire ne peut pas récupérer les prioritaires.
    store.budget = size * 4;
    for entry in &mut store.disk.lock().unwrap().entries {
        entry.priority = true;
    }
    let error = store
        .admit_staging_with_space(&candidate, size, 1, 0)
        .unwrap_err();
    assert!(error.to_string().contains("espace libre"));
    candidate.key.tokens = 0;
    assert!(store
        .admit_staging_with_space(&candidate, size, 1, 0)
        .is_ok());
    assert!(store
        .admit_staging_with_space(&candidate, size * 3, 1, 0)
        .is_err());
    assert!(store
        .admit_staging_with_space(&candidate, size, 1, u64::MAX)
        .is_ok());
}

#[test]
fn staging_limits_and_memory_guard_refuse_before_copy() {
    assert!(check_staging_limits(11, 20, 10, 30, 100).is_err());
    assert!(check_staging_limits(10, 20, 10, 30, 9).is_err());
    assert!(check_staging_limits(10, 31, 10, 30, 100).is_err());
    assert!(check_staging_limits(10, 30, 10, 30, 10).is_ok());
    let root = private_dir();
    let mut store = Store::open(root.path(), 64 << 20, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    let (disk, ram) = staging_estimate(&state.cache, &state.final_state, None, None);
    assert!(disk >= payload(&model, &request, &state).bytes());
    store.memory_guard = crate::memory_guard::MemoryGuard::from_limits(
        true,
        crate::memory_guard::MemoryLimits {
            static_cap: Some(ram - 1),
            host_memory: None,
            headroom_bytes: 0,
            metal_cap: None,
        },
        Arc::new(|| Some(0)),
    );
    assert!(store.admit_staging(&request, disk, ram).is_err());
    assert!(!request.root.exists());
    assert!(!store.busy.load(Ordering::Acquire));
}

#[test]
fn sealing_failure_is_memoized_until_expiry() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    let mut data = payload(&model, &request, &state);
    data.manifest = b"{invalid".to_vec();
    publish(&store, &request, data);

    assert!(!request.root.exists());
    assert!(!store.root.join("pending").exists());
    assert!(store.reserve().is_some());
    for _ in 0..3 {
        assert!(store.skip_save(&request));
    }
    {
        let mut disk = store.disk.lock().unwrap();
        assert!(!disk.write_failed);
        assert_eq!(disk.refused.len(), 1);
        assert!(disk
            .refusal(&request.root, Instant::now())
            .unwrap()
            .contains("key must be a string"));
        assert!(disk
            .refusal(&request.root, Instant::now() + REFUSAL_TTL)
            .is_none());
    }
    assert!(!store.skip_save(&request));
    publish(&store, &request, payload(&model, &request, &state));
    assert!(store.read(&model, &request).is_some());
}

#[test]
fn negative_memo_is_bounded_expires_and_covers_worker_refusals() {
    let root = private_dir();
    let store = Store::open(root.path(), 1, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    publish(&store, &request, payload(&model, &request, &state));
    assert!(store.skip_save(&request));
    let now = Instant::now();
    {
        let mut disk = store.disk.lock().unwrap();
        assert!(disk
            .refusal(&request.root, now)
            .unwrap()
            .contains("budget SSD"));
        assert!(disk.refusal(&request.root, now + REFUSAL_TTL).is_none());
        for index in 0..REFUSAL_CAPACITY + 10 {
            disk.remember_refusal(&PathBuf::from(format!("entry-{index}")), "priority", now);
        }
        assert_eq!(disk.refused.len(), REFUSAL_CAPACITY);
        assert!(disk.refusal(Path::new("entry-0"), now).is_none());
        disk.write_failed = true;
    }
    assert!(
        store.skip_save(&request),
        "écrivain arrêté, même après expiration"
    );
}

#[test]
fn priority_refusal_is_memoized_before_staging() {
    let root = private_dir();
    let store = Store::open(root.path(), 64 << 20, 2).unwrap();
    let (model, first, state) = fixture(&store, 1);
    publish(&store, &first, payload(&model, &first, &state));
    let (_, mut candidate, _) = fixture(&store, 2);
    candidate.key.tokens = 3;
    let error = store
        .admit_staging(&candidate, store.budget, 1)
        .unwrap_err();
    assert!(error.to_string().contains("prioritaires"));
    store.refuse(&candidate, &error.to_string());
    for _ in 0..5 {
        assert!(store.skip_save(&candidate));
    }
    assert!(!candidate.root.exists());
    assert_eq!(store.disk.lock().unwrap().refused.len(), 1);
}

/// Sans poids externes : le KV est arrondi une seule fois puis placé en
/// mémoire Metal partagée. Baseline et reprise voient les mêmes bits bf16.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "device Metal et verrou GPU requis, modèle hybride synthétique"]
fn synthetic_metal_bf16_greedy_survives_restart() -> Result<()> {
    crate::force_resident_full_linear_decode();
    let root = private_dir();
    let store = Store::open(root.path(), 64 << 20, 0)?;
    let (_, request, _) = fixture(&store, 1);
    let load = || {
        let mut weights = crate::decoder::tests::hybrid_weights(2, 2);
        for layer in 0..2 {
            weights.insert(
                format!("layers.{layer}.post_attention_layernorm.weight"),
                Tensor::from_vec(vec![2], vec![1.0; 2])?,
            );
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                weights.insert(
                    format!("layers.{layer}.mlp.{projection}.weight"),
                    Tensor::from_vec(vec![2, 2], vec![0.25, 0.0, 0.0, 0.25])?,
                );
            }
        }
        for norm in ["q_norm", "k_norm"] {
            weights.insert(
                format!("layers.1.self_attn.{norm}.weight"),
                Tensor::from_vec(vec![2], vec![1.0; 2])?,
            );
        }
        CausalDecoder::from_tensors(
            weights,
            CausalDecoderConfig {
                head_dim: Some(2),
                rope_theta: Some(10_000.0),
                ..crate::decoder::tests::hybrid_config(2)
            },
        )?
        .with_metal_runtime()
    };
    let model = load()?;
    let mut state = model.prefill_prompt_state_uncached(&[0, 1])?;
    let metal = model.forward_runtime().metal_executor().unwrap();
    assert!(state.cache.layers[0].linear.metal_state().is_some());
    for layer in &mut state.cache.layers {
        for rows in [&mut layer.keys, &mut layer.values] {
            if rows.len() == 0 {
                continue;
            }
            let bits: Vec<u16> = rows
                .to_vec()
                .iter()
                .map(|value| {
                    let bits = value.to_bits();
                    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
                })
                .collect();
            let buffer = metal.weight_buffer_from_bf16_bytes(bytemuck::cast_slice(&bits))?;
            // SAFETY: buffer partagé de bits.len() u16 initialisés, alignés ;
            // aucune écriture GPU en vol ni autre vue, la vue possède le buffer.
            #[expect(unsafe_code, reason = "vue partagée sur les bits bf16 de la fixture")]
            let shared =
                unsafe { crate::metal_backend::SharedBf16Rows::new(buffer, 0, bits.len())? };
            *rows = kv_rows::KvRows::default();
            rows.push_prefill(crate::metal_backend::PrefillKvRows::SharedBf16(shared));
            assert!(rows.adoptable_bf16_prefix().is_some());
        }
    }
    let data = payload(&model, &request, &state);
    let (disk_estimate, _) = staging_estimate(&state.cache, &state.final_state, None, None);
    assert!(disk_estimate >= data.bytes());
    publish(&store, &request, data);
    let generate = |model: &CausalDecoder, mut state: CausalDecoderPromptState| {
        assert!(model.supports_resident_full_decode());
        assert!(model.setup_resident_full_decode(&mut state.cache, 12, 0, true)?);
        crate::decode_resident::with_kv_format_override(
            crate::decode_resident::KvFormat::Bf16,
            || {
                model
                    .generate_greedy_timed_from_prompt_state_with_options(
                        state,
                        std::time::Duration::ZERO,
                        12,
                        &GenerationOptions::default(),
                    )
                    .map(|output| output.tokens)
            },
        )
    };
    let expected_linear = state.cache.layers[0]
        .linear
        .metal_state()
        .unwrap()
        .disk_parts(metal)?;
    let expected_kv = state.cache.layers[1].keys.snapshot_segments()[0].1.to_vec();
    let expected = generate(&model, state)?;
    assert_eq!(expected.len(), 12);
    drop(model);
    drop(store);
    let restarted = Store::open(root.path(), 64 << 20, 0)?;
    let model = load()?;
    let restored = restarted
        .read(&model, &request)
        .expect("invariant: snapshot Metal valide");
    let restored_linear = restored.cache.layers[0]
        .linear
        .metal_state()
        .expect("invariant: récurrence reconstruite sur Metal")
        .disk_parts(model.forward_runtime().metal_executor().unwrap())?;
    assert_eq!(expected_linear.1, restored_linear.1);
    assert_eq!(expected_linear.2, restored_linear.2);
    assert!(restored.cache.layers[1].keys.has_bf16());
    assert_eq!(
        expected_kv,
        restored.cache.layers[1].keys.snapshot_segments()[0].1
    );
    assert_eq!(
        expected,
        generate(
            &model,
            CausalDecoderPromptState::new(restored.cache, restored.final_state)
        )?
    );
    Ok(())
}

#[test]
fn missing_blob_removes_entry_from_disk_and_index() {
    let root = private_dir();
    let store = Store::open(root.path(), 1 << 20, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    publish(&store, &request, payload(&model, &request, &state));
    let manifest = verify_manifest(&request.root).unwrap();
    fs::remove_file(request.root.join(&manifest.final_state.data.file)).unwrap();
    assert!(verify_manifest(&request.root).is_ok());

    assert!(store.read(&model, &request).is_none());
    assert!(!request.root.exists());
    assert!(store.disk.lock().unwrap().entries.is_empty());
    assert!(store.boundaries(2, 1, 0).is_empty());
    assert!(store.read(&model, &request).is_none());
    assert!(!store.skip_save(&request));
}

#[test]
#[cfg(unix)]
fn transient_read_error_keeps_entry_for_retry() {
    use std::os::unix::fs::PermissionsExt;

    let root = private_dir();
    let store = Store::open(root.path(), 64 << 20, 0).unwrap();
    let (model, request, state) = fixture(&store, 1);
    publish(&store, &request, payload(&model, &request, &state));
    let checksum = request.root.join("manifest.sha256");
    let permissions = fs::metadata(&checksum).unwrap().permissions();
    let bytes = store.disk.lock().unwrap().entries[0].bytes;
    fs::set_permissions(&checksum, fs::Permissions::from_mode(0o000)).unwrap();
    let inaccessible = fs::read(&checksum).is_err();
    let restored = store.read(&model, &request);
    fs::set_permissions(&checksum, permissions).unwrap();
    // Sous root, les permissions ne provoquent pas de refus de lecture.
    if inaccessible {
        assert!(restored.is_none());
    }
    assert!(request.root.exists());
    assert_eq!(store.disk.lock().unwrap().entries[0].bytes, bytes);
    assert!(store.read(&model, &request).is_some());
    for kind in [
        std::io::ErrorKind::PermissionDenied,
        std::io::ErrorKind::Interrupted,
    ] {
        assert!(!invalidates_entry(&InferError::Io {
            path: checksum.clone(),
            source: std::io::Error::from(kind),
        }));
    }
    assert!(!invalidates_entry(&InferError::Metal(
        "device temporairement indisponible".into()
    )));
    let json_io =
        serde_json::from_reader::<_, serde_json::Value>(std::io::Read::take(UnavailableReader, 1))
            .unwrap_err();
    assert!(!invalidates_entry(&InferError::Json {
        path: "manifest.json".into(),
        source: json_io
    }));
}

struct UnavailableReader;
impl std::io::Read for UnavailableReader {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
    }
}

#[test]
fn repeated_persistent_hits_do_not_copy_refused_states() {
    if let Ok(failure) = std::env::var("SSD_REFUSAL_CHILD") {
        let root = private_dir();
        let source = root.path().join("weights");
        fs::write(&source, b"synthetic identity").unwrap();
        configure_prefix_disk_cache(
            &root.path().join("cache"),
            if failure == "budget" { 1 } else { 64 << 20 },
            0,
        )
        .unwrap();
        let store = store().unwrap();
        let model = CausalDecoder::from_tensors(
            crate::decoder::tests::test_weights(),
            CausalDecoderConfig::default(),
        )
        .unwrap()
        .with_snapshot_sources(SnapshotSources::capture([source.as_path()]));
        let _session = PrefixDiskSession::enter(Some("refused"));
        let state = {
            let _warmup = crate::runtime_flags::StateSnapshotWarmup::enter();
            model.prefill_prompt_state_uncached(&[0, 1]).unwrap()
        };
        let request = model
            .snapshot_request(&[0, 1], SnapshotMode::Ar)
            .unwrap()
            .unwrap();
        if failure == "writer" {
            store.disk.lock().unwrap().write_failed = true;
        } else if failure == "priority" {
            store.disk.lock().unwrap().entries.push(Entry {
                path: store.root.join("protected"),
                tokens: 1,
                bytes: store.budget,
                priority: true,
                used: SystemTime::now(),
            });
        }
        storage::BUFFERED_BYTES.set(0);
        for _ in 0..4 {
            model.persist_prompt(&[0, 1], &state).unwrap();
        }
        assert_eq!(
            storage::BUFFERED_BYTES.get(),
            0,
            "aucun octet copié pour {failure}"
        );
        assert!(!request.root.exists());
        if failure != "writer" {
            let disk = store.disk.lock().unwrap();
            assert_eq!(disk.refused.len(), 1);
            assert!(disk.refused[0].reason.contains(if failure == "budget" {
                "budget"
            } else {
                "prioritaires"
            }));
        }
        return;
    }
    for failure in ["budget", "priority", "writer"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "decoder::state_snapshot::persistent::tests::repeated_persistent_hits_do_not_copy_refused_states", "--nocapture"])
            .env("SSD_REFUSAL_CHILD", failure)
            .env_remove("SARAGOSSA_RUST_STATE_SNAPSHOT")
            .env_remove("RETI_RUST_STATE_SNAPSHOT")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{failure}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
