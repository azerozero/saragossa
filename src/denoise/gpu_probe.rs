//! Sonde de latence GPU par hop : même graphe de dépendances que DeepFilterNet3.
//!
//! Ce n'est pas un port Metal du modèle : chaque couche est remplacée par un
//! produit matrice-vecteur de même nombre de MAC, enchaîné en série dans un
//! command buffer par hop. La mesure borne par en dessous le coût d'un port
//! GPU réel (dispatch, attente et exécution d'environ 40 noyaux dépendants).

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

/// (lignes, colonnes) des produits équivalents d'un pas DFN3, dans l'ordre du graphe.
const LAYERS: &[(u64, u64)] = &[
    (2048, 9),  // erb_conv0
    (1024, 67), // erb_conv1 (dw + 1×1)
    (512, 67),  // erb_conv2
    (512, 67),  // erb_conv3
    (6144, 9),  // df_conv0 groupée
    (6144, 64), // df_conv0 1×1
    (3072, 67), // df_conv1
    (512, 96),  // df_fc_emb
    (256, 32),  // enc lin_in
    (768, 256), // enc GRU ih
    (768, 256), // enc GRU hh
    (512, 16),  // enc lin_out
    (256, 32),  // erb lin_in
    (768, 256), // erb GRU l0 ih
    (768, 256), // erb GRU l0 hh
    (768, 256), // erb GRU l1 ih
    (768, 256), // erb GRU l1 hh
    (512, 16),  // erb lin_out
    (512, 65),  // conv3p + convt3
    (512, 64),  // conv2p
    (1024, 67), // convt2
    (1024, 64), // conv1p
    (2048, 67), // convt1
    (2048, 64), // conv0p
    (32, 192),  // conv0_out
    (256, 64),  // df lin_in
    (768, 256), // df GRU l0 ih
    (768, 256), // df GRU l0 hh
    (768, 256), // df GRU l1 ih
    (768, 256), // df GRU l1 hh
    (256, 32),  // df_skip
    (960, 16),  // df_out
    (960, 160), // df_convp groupée
    (960, 10),  // df_convp 1×1
    (481, 32),  // masque ERB + filtre profond
];

const SRC: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void matvec(device const float* w [[buffer(0)]], device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]], constant uint2& dims [[buffer(3)]],
                   uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (row >= dims.x) return;
    float acc = 0.0f;
    for (uint c = lane; c < dims.y; c += 32) acc += w[row * dims.y + c] * x[c];
    acc = simd_sum(acc);
    if (lane == 0) y[row] = tanh(acc);
}
";

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[(v.len() - 1) * 99 / 100])
}

#[test]
#[ignore = "mesure GPU ; lancer sous le verrou GPU"]
fn dfn3_gpu_hop_latency_probe() {
    let device = Device::system_default().expect("invariant: GPU Metal présent");
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(SRC, &opts).unwrap();
    let f = lib.get_function("matvec", None).unwrap();
    let pso = device.new_compute_pipeline_state_with_function(&f).unwrap();
    let queue = device.new_command_queue();
    let shared = MTLResourceOptions::StorageModeShared;
    let weights: Vec<_> = LAYERS
        .iter()
        .map(|&(r, c)| {
            let data = vec![0.01_f32; (r * c) as usize];
            device.new_buffer_with_data(data.as_ptr().cast(), r * c * 4, shared)
        })
        .collect();
    let a = device.new_buffer(8192 * 4, shared);
    let b = device.new_buffer(8192 * 4, shared);
    let macs: u64 = LAYERS.iter().map(|&(r, c)| r * c).sum();

    let (mut empty, mut wall, mut gpu) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..1200 {
        let t0 = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
        if i >= 200 {
            empty.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    for i in 0..1200 {
        let t0 = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (k, (&(r, c), w)) in LAYERS.iter().zip(&weights).enumerate() {
            let (x, y) = if k % 2 == 0 { (&a, &b) } else { (&b, &a) };
            let dims = [r as u32, c as u32];
            enc.set_buffer(0, Some(w), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 8, dims.as_ptr().cast());
            enc.dispatch_thread_groups(MTLSize::new(r, 1, 1), MTLSize::new(32, 1, 1));
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if i >= 200 {
            wall.push(t0.elapsed().as_secs_f64() * 1e6);
            gpu.push(crate::metal_backend::command_buffer_gpu_ns(cb) as f64 / 1e3);
        }
    }
    let ((e50, e99), (w50, w99), (g50, g99)) = (stats(empty), stats(wall), stats(gpu));
    eprintln!(
        "dfn3 GPU sonde ({} noyaux, {:.2} MMAC/hop) : CB vide p50 {e50:.0} µs p99 {e99:.0} ; \
         hop mur p50 {w50:.0} µs p99 {w99:.0} ; temps GPU p50 {g50:.0} µs p99 {g99:.0}",
        LAYERS.len(),
        macs as f64 / 1e6
    );
}
