//! Test cross-longueur du coût froid de prefill (diagnostic, mesure GPU idle).
//!
//! Question : le coût froid de la 1ʳᵉ passe de prefill est-il payé UNE FOIS par
//! process, ou à CHAQUE nouvelle longueur de prompt ? `--prefill-repeat` ne teste
//! qu'une longueur fixe ; ici on enchaîne, dans un SEUL process après un seul
//! warmup, deux longueurs A et B différentes, plusieurs fois, et on regarde le
//! motif chaud/froid.
//!
//! Lecture :
//! - A#0 (1ʳᵉ passe réelle) ≈ A#1 après un warmup suffisant → cold-start amorti.
//! - B#0 (autre longueur, 1ʳᵉ fois) : ≈ B#1 chaud → coût froid UNE FOIS PAR
//!   PROCESS, pas par-longueur (mesuré 2026-07-24 : c'est le cas) ; ≈ froid →
//!   coût PAR LONGUEUR (allocation keyée par longueur).
//!
//! ATTENTION — ce coût froid once-per-process est DISTINCT du déficit prefill
//! reti-vs-oMLX. Ce dernier est un intercept ~1 s qui PERSISTE à chaud (slopes
//! per-token à parité, cf mémoire project_reti_prefill_qmv_gap, brick TEST
//! SERVEUR-CHAUD) : ce test-ci ne le mesure pas et ne le réfute pas. reti reste
//! derrière oMLX au prefill (~0,68×@8k → 0,92×@32k).
//!
//! Usage :
//!   cargo run -p saragossa --release --features metal --example prefill_cross_length -- <model_dir> [A] [B]

use std::time::Instant;

use saragossa::{load_causal_decoder, MetalExecutor, ModelAssets};

fn synth_ids(len: usize) -> Vec<usize> {
    // Ids valides et variés (routing MoE non dégénéré), déterministes.
    (0..len).map(|i| (i * 7 + 13) % 30_000 + 1).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: prefill_cross_length <model_dir> [A] [B]")?;
    let len_a: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let len_b: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4096);

    eprintln!("chargement {model_dir} …");
    let assets = ModelAssets::load_local(&model_dir)?;
    // IMPÉRATIF : rattacher un MetalExecutor persistant, comme le CLI
    // (main.rs, load_decoder_with_runtime). Sans lui, le prefill tombe sur le
    // chemin per-op dégénéré (mesuré : warmup 333 s, 1ʳᵉ passe >10 min) au lieu du
    // résident (~1-2 s). Tout harnais de bench prefill DOIT le rattacher.
    let executor = MetalExecutor::new()?;
    let decoder = load_causal_decoder(&assets)?.with_metal_executor(executor);

    // Warmup : paie le coût unique de compilation JIT/shaders (~27 s), hors mesure,
    // exactement comme un serveur au boot. On warm à une 3ᵉ longueur (256) pour ne
    // privilégier ni A ni B.
    eprintln!("warmup (256 tokens, 2 passes) …");
    let warm = synth_ids(256);
    let t = Instant::now();
    let _ = decoder.prefill_cache_uncached(&warm)?;
    let _ = decoder.prefill_cache_uncached(&warm)?;
    eprintln!("warmup fait en {} ms", t.elapsed().as_millis());

    let ids_a = synth_ids(len_a);
    let ids_b = synth_ids(len_b);

    // Séquence conçue pour disséquer froid/chaud × longueur.
    let plan: &[(&str, &[usize])] = &[
        ("A#0", &ids_a),
        ("A#1", &ids_a),
        ("B#0", &ids_b), // <-- LE TEST : autre longueur, 1ʳᵉ fois
        ("B#1", &ids_b),
        ("A#2", &ids_a), // retour à A : encore chaud ?
        ("B#2", &ids_b),
    ];

    println!("# label\tlen\tprefill_ms\ttok_s");
    for (label, ids) in plan {
        let started = Instant::now();
        let _ = decoder.prefill_cache_uncached(ids)?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let tok_s = ids.len() as f64 / (ms / 1000.0);
        println!("{label}\t{}\t{ms:.0}\t{tok_s:.0}", ids.len());
    }
    Ok(())
}
