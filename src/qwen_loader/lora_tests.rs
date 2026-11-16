//! Oracle de la fusion LoRA au chargement, sur un modèle synthétique.

use super::lora::tests::{f32_view, lora_config, write_adapter, TestModule};
use super::test_fixtures::*;
use super::*;

type Module = TestModule<'static>;

fn modules() -> Vec<Module> {
    let values = |n: usize, phase: f32| -> Vec<f32> {
        (0..n)
            .map(|i| 0.3 * ((i as f32 + phase) * 1.7).sin())
            .collect()
    };
    vec![
        (
            "model.layers.0.self_attn.q_proj",
            values(4, 0.1),
            vec![2, 2],
            values(4, 0.5),
            vec![2, 2],
        ),
        (
            "model.layers.0.self_attn.o_proj",
            values(4, 1.1),
            vec![2, 2],
            values(4, 1.9),
            vec![2, 2],
        ),
        (
            "model.layers.0.mlp.gate_proj",
            values(4, 2.3),
            vec![2, 2],
            values(8, 2.9),
            vec![4, 2],
        ),
        (
            "model.layers.0.mlp.down_proj",
            values(8, 3.7),
            vec![2, 4],
            values(4, 4.1),
            vec![2, 2],
        ),
    ]
}

/// Réécrit `base` avec `W + s·B·A` calculé en f64, indépendamment du chargeur.
fn write_merged_by_hand(base: &Path, out: &Path, scale: f64, modules: &[Module]) {
    let bytes = std::fs::read(base).expect("invariant: lecture base");
    let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("invariant: base lisible");
    let mut views = Vec::new();
    for (name, view) in tensors.tensors() {
        let mut values: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if let Some((_, a, a_shape, b, _)) = modules
            .iter()
            .find(|module| format!("{}.weight", module.0) == name)
        {
            let (rank, inputs) = (a_shape[0], a_shape[1]);
            for (o, row) in values.chunks_mut(inputs).enumerate() {
                for (i, w) in row.iter_mut().enumerate() {
                    let delta: f64 = (0..rank)
                        .map(|k| f64::from(b[o * rank + k]) * f64::from(a[k * inputs + i]))
                        .sum();
                    *w = (f64::from(*w) + scale * delta) as f32;
                }
            }
        }
        views.push((name, f32_view(view.shape().to_vec(), &values)));
    }
    let bytes = safetensors::serialize(views, None).expect("invariant: sérialisable");
    std::fs::write(out, bytes).expect("invariant: écriture");
}

fn load(path: &Path, lora: Option<&LoraAdapter>) -> Result<CausalDecoder> {
    let shards = [path.to_path_buf()];
    let catalog = WeightCatalog::from_shards(&shards)?;
    let context = DecoderLoadContext::cpu();
    let context = match lora {
        Some(lora) => context.with_lora(lora),
        None => context,
    };
    load_from_shards_with_context(&test_config(), &shards, &catalog, &context)
}

#[test]
fn fused_logits_match_hand_merged_checkpoint() {
    let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
    let base = dir.path().join("base.safetensors");
    write_safetensors_with_mlp(&base);
    let modules = modules();
    // r = 2, alpha = 3 : s = 1,5.
    write_adapter(dir.path(), &lora_config(2, 3.0), &modules);
    let adapter = LoraAdapter::load(dir.path()).expect("invariant: adaptateur valide");
    let merged = dir.path().join("merged.safetensors");
    write_merged_by_hand(&base, &merged, 1.5, &modules);

    let fused = load(&base, Some(&adapter)).expect("invariant: fusion chargeable");
    let oracle = load(&merged, None).expect("invariant: oracle chargeable");
    let plain = load(&base, None).expect("invariant: base chargeable");
    for prompt in [&[0_usize][..], &[2, 1, 0], &[1, 1, 2, 0]] {
        let got = fused
            .next_logits(prompt)
            .expect("invariant: forward fusionné");
        let want = oracle
            .next_logits(prompt)
            .expect("invariant: forward oracle");
        let base_logits = plain.next_logits(prompt).expect("invariant: forward base");
        for (g, w) in got.data().iter().zip(want.data()) {
            assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} contre {w}");
        }
        assert_ne!(
            got.data(),
            base_logits.data(),
            "l'adaptateur doit changer les logits"
        );
    }
}

#[test]
fn zero_b_keeps_base_logits_bit_exact() {
    let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
    let base = dir.path().join("base.safetensors");
    write_safetensors_with_mlp(&base);
    let zeroed: Vec<Module> = modules()
        .into_iter()
        .map(|(name, a, a_shape, b, b_shape)| (name, a, a_shape, vec![0.0; b.len()], b_shape))
        .collect();
    write_adapter(dir.path(), &lora_config(2, 3.0), &zeroed);
    let adapter = LoraAdapter::load(dir.path()).expect("invariant: adaptateur valide");
    let fused = load(&base, Some(&adapter)).expect("invariant: fusion chargeable");
    let plain = load(&base, None).expect("invariant: base chargeable");
    let prompt = [2, 0, 1];
    assert_eq!(
        fused
            .next_logits(&prompt)
            .expect("invariant: forward")
            .data(),
        plain
            .next_logits(&prompt)
            .expect("invariant: forward")
            .data()
    );
}

#[test]
fn adapter_without_base_target_is_refused() {
    let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
    let base = dir.path().join("base.safetensors");
    write_safetensors_with_mlp(&base);
    write_adapter(
        dir.path(),
        &lora_config(1, 1.0),
        &[(
            "model.layers.0.self_attn.x_proj",
            vec![1.0, 1.0],
            vec![1, 2],
            vec![1.0, 1.0],
            vec![2, 1],
        )],
    );
    let adapter = LoraAdapter::load(dir.path()).expect("invariant: adaptateur lisible");
    let error = load(&base, Some(&adapter)).expect_err("cible absente refusée");
    assert!(error.to_string().contains("x_proj"), "{error}");
}

#[test]
fn quantized_base_target_is_refused() {
    let dir = tempfile::tempdir().expect("invariant: répertoire temporaire");
    let base = dir.path().join("base.safetensors");
    write_quantized_q_proj_safetensors(&base);
    write_adapter(
        dir.path(),
        &lora_config(1, 1.0),
        &[(
            "model.layers.0.self_attn.q_proj",
            vec![1.0; 4],
            vec![1, 4],
            vec![1.0; 4],
            vec![4, 1],
        )],
    );
    let adapter = LoraAdapter::load(dir.path()).expect("invariant: adaptateur lisible");
    let shards = [base.clone()];
    let catalog = WeightCatalog::from_shards(&shards).expect("invariant: catalog");
    let mut config = test_config();
    config.hidden_size = 4;
    config.head_dim = Some(4);
    config.quantization = Some(QuantConfig {
        group_size: Some(4),
        bits: Some(8),
        quant_method: Some("mx".to_string()),
        fmt: None,
        extra: HashMap::new(),
    });
    let error = load_from_shards_with_context(
        &config,
        &shards,
        &catalog,
        &DecoderLoadContext::cpu().with_lora(&adapter),
    )
    .expect_err("base quantifiée refusée");
    assert!(error.to_string().contains("quantifiée"), "{error}");
}

/// Adaptateur thinker de THX-02 (r = 32, alpha = 32) sur sa base bf16 : chaque
/// module trouve sa cible, `W'·x` suit `W·x + s·B·(A·x)` calculé à part, et le
/// modèle fusionné se charge sur Metal. `mtp.safetensors` n'est pas lu.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
#[ignore = "requiert les poids THX-02 (SARAGOSSA_THX02_DIR)"]
fn thx02_thinker_adapter_fuses_on_metal() {
    let root = std::env::var_os("SARAGOSSA_THX02_DIR").map_or_else(
        || PathBuf::from("/Users/ludwig/workspace/ai_model/THX-02"),
        PathBuf::from,
    );
    let adapter = LoraAdapter::load(root.join("thinker")).expect("invariant: adaptateur thinker");
    assert_eq!(adapter.module_count(), 200);
    let top: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("config.json")).expect("invariant: config.json"),
    )
    .expect("invariant: config JSON");
    let raw: crate::config::RawModelConfig =
        serde_json::from_value(top["backbone_config"].clone()).expect("invariant: backbone");
    let config = raw.resolve().expect("invariant: config résolue");
    let shards = [root.join("model.safetensors")];
    let catalog = WeightCatalog::from_shards(&shards).expect("invariant: catalog");

    let headers = read_shard_headers(&shards).expect("invariant: en-têtes");
    let entries = index_shard_entries(&headers).expect("invariant: index");
    for name in [
        "language_model.layers.3.self_attn.q_proj.weight",
        "language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.layers.31.mlp.down_proj.weight",
    ] {
        let entry = &entries[name];
        let bytes = quantized::read_entry_bytes(&headers[entry.shard_index], &entry.entry)
            .expect("invariant: payload");
        let base = bytes_to_dense_f32(&bytes, entry.entry.dtype, name).expect("invariant: bf16");
        let mut fused = base.clone();
        let module = adapter.module(name).expect("invariant: module ciblé");
        module
            .merge_into(name, &entry.entry.shape, &mut fused)
            .expect("invariant: fusion");
        let (outputs, inputs) = (entry.entry.shape[0], entry.entry.shape[1]);
        let x: Vec<f64> = (0..inputs).map(|i| ((i as f64) * 0.37).sin()).collect();
        let ax: Vec<f64> = (0..module.rank())
            .map(|k| {
                (0..inputs)
                    .map(|i| f64::from(module.a()[k * inputs + i]) * x[i])
                    .sum()
            })
            .collect();
        let mut worst = 0.0_f64;
        for o in (0..outputs).step_by(97) {
            let dot = |w: &[f32]| -> f64 {
                (0..inputs)
                    .map(|i| f64::from(w[o * inputs + i]) * x[i])
                    .sum()
            };
            let low_rank: f64 = (0..module.rank())
                .map(|k| f64::from(module.b()[o * module.rank() + k]) * ax[k])
                .sum();
            let expected = dot(&base) + f64::from(module.scale()) * low_rank;
            let scale = expected.abs().max(1.0);
            worst = worst.max((dot(&fused) - expected).abs() / scale);
        }
        eprintln!("{name} : écart relatif max {worst:e}");
        assert!(worst <= 1e-5, "{name} : {worst:e}");
    }

    let executor = crate::MetalExecutor::new().expect("invariant: Metal");
    let decoder = crate::runtime_flags::with_decision_headless(|| {
        load_from_shards_with_context(
            &config,
            &shards,
            &catalog,
            &DecoderLoadContext::metal(&executor).with_lora(&adapter),
        )
    });
    decoder.expect("invariant: modèle fusionné chargeable");
}
