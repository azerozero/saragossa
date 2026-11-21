//! Oracle de bout en bout des poids mmap sans copie contre la copie.

use super::*;

const CHILD_OUT: &str = "SARAGOSSA_MMAP_TEST_OUT";
const TEST: &str = "qwen_loader::mmap_weights_tests::mmap_weights_logits_match_copy_bitwise";

/// Logits identiques bit à bit entre `SARAGOSSA_RUST_MMAP_WEIGHTS=0` et `=1`
/// sur le modèle réel `SARAGOSSA_MMAP_TEST_MODEL_DIR` (dense, MoE à expert
/// partagé, shards réalignés ou non). Le flag est figé par OnceLock : chaque
/// variante tourne dans un processus neuf, qui rapporte aussi le nombre de vues
/// mmap créées (0 attendu sans le flag ; 0 aussi sur un shard MLX dont
/// l'en-tête désaligne tous les tenseurs).
#[test]
#[ignore = "poids réels requis (SARAGOSSA_MMAP_TEST_MODEL_DIR) et device Metal"]
fn mmap_weights_logits_match_copy_bitwise() {
    let Some(model_dir) = std::env::var_os("SARAGOSSA_MMAP_TEST_MODEL_DIR").map(PathBuf::from)
    else {
        eprintln!("test mmap sauté : définir SARAGOSSA_MMAP_TEST_MODEL_DIR");
        return;
    };
    if let Some(out) = std::env::var_os(CHILD_OUT).map(PathBuf::from) {
        let assets = ModelAssets::load_local(&model_dir).expect("invariant: assets du modèle");
        let executor = crate::MetalExecutor::new().expect("invariant: device Metal");
        let model = crate::load_causal_decoder_for_metal(&assets, &executor)
            .expect("invariant: modèle chargeable en Metal")
            .with_metal_executor(executor)
            .expect("invariant: runtime Metal attaché");
        let mut tokens: Vec<usize> = assets
            .tokenizer
            .encode("Le cache LRU évince l'entrée la moins récemment utilisée, car")
            .expect("invariant: tokenisation")
            .into_iter()
            .map(|id| id as usize)
            .collect();
        let mut bytes = Vec::new();
        // Prefill puis quelques pas greedy : chaque passe relit tous les poids.
        for _ in 0..4 {
            let logits = model.next_logits(&tokens).expect("invariant: forward");
            let data = logits.data();
            bytes.extend(data.iter().flat_map(|value| value.to_bits().to_le_bytes()));
            let argmax = data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .expect("invariant: logits non vides");
            tokens.push(argmax);
        }
        std::fs::write(&out, bytes).expect("invariant: logits écrits");
        let views =
            crate::metal_backend::MMAP_WEIGHT_VIEWS.load(std::sync::atomic::Ordering::Relaxed);
        std::fs::write(out.with_extension("views"), views.to_string())
            .expect("invariant: compteur écrit");
        return;
    }
    let dir = tempfile::tempdir().expect("invariant: dossier temporaire");
    let mut runs = Vec::new();
    for flag in ["0", "1"] {
        let out = dir.path().join(format!("logits-{flag}.bin"));
        let status = std::process::Command::new(std::env::current_exe().expect("invariant: exe"))
            .args(["--exact", TEST, "--ignored", "--nocapture"])
            .env(CHILD_OUT, &out)
            .env("SARAGOSSA_RUST_MMAP_WEIGHTS", flag)
            .status()
            .expect("invariant: processus enfant lancé");
        assert!(status.success(), "flag={flag}: {status}");
        let views: usize = std::fs::read_to_string(out.with_extension("views"))
            .expect("invariant: compteur lu")
            .parse()
            .expect("invariant: compteur entier");
        eprintln!("mmap flag={flag} vues={views}");
        runs.push((std::fs::read(&out).expect("invariant: logits lus"), views));
    }
    assert_eq!(runs[0].1, 0, "vues mmap sans le flag");
    assert!(!runs[0].0.is_empty());
    assert!(
        runs[0].0 == runs[1].0,
        "logits différents entre copie et mmap"
    );
}
