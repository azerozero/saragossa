//! Micro-bench CPU du tirage : distribution top-k/top-p, softmax dense et
//! tirage cb0 TTS, sur des logits synthétiques reproductibles.
//!
//! `cargo run --release --example sampling_micro`

use saragossa::sampling::{token_distribution_top_k_top_p, DeterministicSampler};
use saragossa::{sample_token_top_k_top_p, softmax};
use std::time::Instant;

fn logits(vocab: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..vocab)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state % 10_000) as f32 / 10_000.0;
            // Queue de type logits LLM : bruit gaussien grossier + quelques pics.
            let peak = if i % 9973 == 0 { 12.0 } else { 0.0 };
            noise * 8.0 - 4.0 + peak
        })
        .collect()
}

fn median_us(mut work: impl FnMut(), reps: usize) -> f64 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        work();
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    samples[reps / 2]
}

fn main() -> saragossa::Result<()> {
    let llm = logits(248_320, 7);
    let tts = logits(3072, 11);
    let mut sampler = DeterministicSampler::new(1);
    let reps = 61;
    let distribution = median_us(
        || {
            let d = token_distribution_top_k_top_p(&llm, 0.7, 0.95, 20);
            std::hint::black_box(d.ok());
        },
        reps,
    );
    let dense = median_us(
        || {
            std::hint::black_box(softmax(&llm, 0.7));
        },
        reps,
    );
    let copy = median_us(
        || {
            std::hint::black_box(llm.to_vec());
        },
        reps,
    );
    let cb0 = median_us(
        || {
            let t = sample_token_top_k_top_p(&tts, 0.9, 1.0, 50, &mut sampler);
            std::hint::black_box(t.ok());
        },
        1001,
    );
    println!("llm_distribution_topk20_topp095_us={distribution:.1}");
    println!("llm_softmax_dense_us={dense:.1}");
    println!("llm_copy_logits_us={copy:.1}");
    println!("tts_cb0_sample_topk50_us={cb0:.2}");
    Ok(())
}
