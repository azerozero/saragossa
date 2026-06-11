use super::utils::write_f32_at;
use super::*;

// Kernels minimaux pour la chaîne synthétique : `scale` (out = in*s) et
// `add` (out = a+b). Suffisent à exercer ping-pong + scratch + accumulateur
// persistant dans UN seul command buffer (squelette de 1c).
const RESIDENT_TEST_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void resident_scale_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    output[i] = input[i] * scale;
}

kernel void resident_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) { return; }
    output[i] = a[i] + b[i];
}
"#;

/// Renvoie l'état résident, ou `None` proprement si aucun device Metal n'est
/// disponible (miroir des tests `metal_backend`, pour la CI sans GPU). Une
/// erreur de compilation des kernels FAIT échouer le test (pas de skip).
fn try_state() -> Result<Option<DecodeResidentState>> {
    match Device::system_default() {
        Some(device) => Ok(Some(DecodeResidentState::new(device)?)),
        None => Ok(None),
    }
}

fn build_pipeline(device: &Device, name: &str) -> ComputePipelineState {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(RESIDENT_TEST_KERNELS, &options)
        .expect("invariant: kernels de test compilent");
    let function = library
        .get_function(name, None)
        .expect("invariant: fonction de test présente");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("invariant: pipeline de test valide")
}

fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: f32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<f32>() as u64,
        std::ptr::from_ref(&value).cast::<c_void>(),
    );
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        std::ptr::from_ref(&value).cast::<c_void>(),
    );
}

#[allow(
    unsafe_code,
    reason = "écriture d'un MTLBuffer partagé avant commit (test)"
)]
fn seed_f32(tensor: &GpuTensor, data: &[f32]) {
    assert_eq!(tensor.len(), data.len(), "seed: longueur incohérente");
    let ptr = tensor.buffer().contents().cast::<f32>();
    assert!(!ptr.is_null(), "MTLBuffer partagé sans pointeur CPU");
    // SAFETY: buffer en StorageModeShared, dimensionné pour `data.len()` f32,
    // écrit avant tout commit ; copie de longueur exacte, sans chevauchement.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
}

#[allow(
    unsafe_code,
    reason = "lecture d'un MTLBuffer partagé après wait_until_completed (test)"
)]
fn read_f32(tensor: &GpuTensor) -> Vec<f32> {
    let ptr = tensor.buffer().contents().cast::<f32>();
    assert!(!ptr.is_null(), "MTLBuffer partagé sans pointeur CPU");
    let mut out = vec![0.0_f32; tensor.len()];
    // SAFETY: buffer en StorageModeShared dont le command buffer a terminé ;
    // copie de `len` f32 vers `out` (même longueur), sans chevauchement.
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), tensor.len());
    }
    out
}

fn dispatch(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    len: usize,
) {
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(width, 1, 1));
}

// Chaîne synthétique de référence (oracle CPU) : pour chaque couche `s`,
//   next = current*s + current ;  acc += next ;  current = next.
fn cpu_oracle(input: &[f32], scales: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut current = input.to_vec();
    let mut acc = vec![0.0_f32; input.len()];
    for &scale in scales {
        let next: Vec<f32> = current.iter().map(|&x| x * scale + x).collect();
        for (slot, &value) in acc.iter_mut().zip(next.iter()) {
            *slot += value;
        }
        current = next;
    }
    (current, acc)
}

/// Dérisquage 1c : 3 « couches » résidentes (ping-pong + scratch + accumulateur
/// persistant) chaînées dans UN seul command buffer, un commit/wait, les bails
/// tenus vivants jusqu'au wait → résultat identique à l'oracle CPU.
#[test]
fn resident_three_layer_chain_matches_cpu() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let scales = [2.0_f32, 0.5, 3.0];
    let n = input.len();
    let (oracle_current, oracle_acc) = cpu_oracle(&input, &scales);

    let scale_pipeline = build_pipeline(state.device(), "resident_scale_f32");
    let add_pipeline = build_pipeline(state.device(), "resident_add_f32");

    // Ping-pong + accumulateur : buffers persistants distincts (jamais aliasés).
    let buf_a = state.persistent(n, GpuElement::F32)?;
    let buf_b = state.persistent(n, GpuElement::F32)?;
    let acc = state.persistent(n, GpuElement::F32)?;
    seed_f32(&buf_a, &input);
    seed_f32(&acc, &vec![0.0_f32; n]);

    let queue = state.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Les bails sont conservés vivants jusqu'APRÈS le wait : discipline de
    // liveness du chemin résident (un scratch encodé ne doit pas être rendu
    // au pool tant que le GPU ne l'a pas consommé).
    let mut leases = Vec::new();
    let mut current = &buf_a;
    let mut other = &buf_b;
    for &scale in &scales {
        let tmp = state.scratch().lease(n, GpuElement::F32)?;
        // tmp = current * scale
        encoder.set_compute_pipeline_state(&scale_pipeline);
        encoder.set_buffer(0, Some(current.buffer()), 0);
        encoder.set_buffer(1, Some(tmp.tensor().buffer()), 0);
        set_f32(encoder, 2, scale);
        set_u32(encoder, 3, n as u32);
        dispatch(encoder, &scale_pipeline, n);
        // other = tmp + current  (résiduel → buffer ping-pong opposé)
        encoder.set_compute_pipeline_state(&add_pipeline);
        encoder.set_buffer(0, Some(tmp.tensor().buffer()), 0);
        encoder.set_buffer(1, Some(current.buffer()), 0);
        encoder.set_buffer(2, Some(other.buffer()), 0);
        set_u32(encoder, 3, n as u32);
        dispatch(encoder, &add_pipeline, n);
        // acc = acc + other  (accumulateur persistant, in-place a==out)
        encoder.set_compute_pipeline_state(&add_pipeline);
        encoder.set_buffer(0, Some(acc.buffer()), 0);
        encoder.set_buffer(1, Some(other.buffer()), 0);
        encoder.set_buffer(2, Some(acc.buffer()), 0);
        set_u32(encoder, 3, n as u32);
        dispatch(encoder, &add_pipeline, n);

        leases.push(tmp);
        std::mem::swap(&mut current, &mut other);
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let gpu_current = read_f32(current);
    let gpu_acc = read_f32(&acc);
    drop(leases);

    for (idx, (gpu, cpu)) in gpu_current.iter().zip(oracle_current.iter()).enumerate() {
        assert!(
            (gpu - cpu).abs() <= 1.0e-5,
            "current[{idx}] gpu={gpu} cpu={cpu}"
        );
    }
    for (idx, (gpu, cpu)) in gpu_acc.iter().zip(oracle_acc.iter()).enumerate() {
        assert!(
            (gpu - cpu).abs() <= 1.0e-5,
            "acc[{idx}] gpu={gpu} cpu={cpu}"
        );
    }
    // 3 bails simultanément vivants → 3 slots physiques distincts (anti-aliasing).
    assert_eq!(state.scratch().slot_count(), 3);
    Ok(())
}

/// Réserve B : deux bails simultanément vivants pointent vers des buffers
/// PHYSIQUEMENT DISTINCTS ; un bail libéré rend son slot réutilisable.
#[test]
fn scratch_leases_never_alias_while_live() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let pool = state.scratch();

    let a = pool.lease(16, GpuElement::F32)?;
    let b = pool.lease(16, GpuElement::F32)?;
    assert_ne!(
        a.tensor().buffer().gpu_address(),
        b.tensor().buffer().gpu_address(),
        "deux bails vivants ne doivent jamais partager de buffer"
    );
    assert_eq!(pool.slot_count(), 2);

    // Libère `a` : son slot redevient réutilisable.
    drop(a);
    let c = pool.lease(16, GpuElement::F32)?;
    assert_eq!(
        pool.slot_count(),
        2,
        "un bail séquentiel réutilise le slot libéré (pas de nouvelle alloc)"
    );
    assert_ne!(
        b.tensor().buffer().gpu_address(),
        c.tensor().buffer().gpu_address(),
        "`c` ne doit pas aliaser `b` (toujours vivant)"
    );
    drop(b);
    drop(c);
    Ok(())
}

/// Une taille (ou un type) différent n'est jamais servi par un slot existant.
#[test]
fn scratch_distinct_sizes_use_distinct_slots() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let pool = state.scratch();
    let small = pool.lease(8, GpuElement::F32)?;
    drop(small);
    // Même slot réutilisé pour la même taille/type.
    let small_again = pool.lease(8, GpuElement::F32)?;
    assert_eq!(pool.slot_count(), 1);
    // Taille différente → nouveau slot.
    let big = pool.lease(32, GpuElement::F32)?;
    assert_eq!(pool.slot_count(), 2);
    // Type différent à taille d'octets identique → nouveau slot.
    let typed = pool.lease(8, GpuElement::U32)?;
    assert_eq!(pool.slot_count(), 3);
    drop(small_again);
    drop(big);
    drop(typed);
    Ok(())
}

/// Les buffers persistants sont toujours des allocations distinctes.
#[test]
fn persistent_buffers_are_distinct() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let first = state.persistent(16, GpuElement::F32)?;
    let second = state.persistent(16, GpuElement::F32)?;
    assert_ne!(
        first.buffer().gpu_address(),
        second.buffer().gpu_address(),
        "deux buffers persistants ne doivent jamais partager d'allocation"
    );
    Ok(())
}

// --- 1b.1 : KV-cache full-attn résident (FullAttentionMetalState) ---

/// 1b.1 — seed (prefill) puis append (decode) : le buffer KV GPU est
/// bit-identique au `Vec<f32>` CPU append-only (l'oracle `LayerKvCache`), sur
/// les **dimensions réelles** Qwen3.6-35B-A3B full-attn (q=16, kv=2, hd=256).
/// Couvre la transition prefill → premier decode et les offsets non nuls.
#[test]
fn kv_seed_then_append_matches_cpu() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (16, 2, 256);
    let kv_dim = kv_heads * head_dim; // 512
    let capacity = 6;
    let mut kv = state.full_attention(capacity, q_heads, kv_heads, head_dim)?;
    assert_eq!(kv.kv_dim(), kv_dim);

    let mut cpu_keys: Vec<f32> = Vec::new();
    let mut cpu_values: Vec<f32> = Vec::new();

    // Seed prefill : 3 lignes.
    let seed_rows = 3;
    let mut seed_k = Vec::new();
    let mut seed_v = Vec::new();
    for r in 0..seed_rows {
        for c in 0..kv_dim {
            seed_k.push((r * kv_dim + c) as f32 * 0.001);
            seed_v.push((r * kv_dim + c) as f32 * -0.002 + 1.0);
        }
    }
    kv.seed(&seed_k, &seed_v, seed_rows)?;
    cpu_keys.extend_from_slice(&seed_k);
    cpu_values.extend_from_slice(&seed_v);
    assert_eq!(kv.len(), seed_rows);

    // Premiers decodes : append une ligne à la fois (offsets seed_rows·kv_dim, …).
    for r in seed_rows..(seed_rows + 2) {
        let k: Vec<f32> = (0..kv_dim)
            .map(|c| (r * kv_dim + c) as f32 * 0.003)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|c| (r * kv_dim + c) as f32 * 0.004 - 0.5)
            .collect();
        kv.append_row(&k, &v)?;
        cpu_keys.extend_from_slice(&k);
        cpu_values.extend_from_slice(&v);
    }
    assert_eq!(kv.len(), seed_rows + 2);

    // Relire les buffers résidents et comparer aux Vec CPU (lignes valides).
    let gpu_keys = read_f32(kv.keys());
    let gpu_values = read_f32(kv.values());
    let valid = kv.len() * kv_dim;
    assert_eq!(&gpu_keys[..valid], cpu_keys.as_slice());
    assert_eq!(&gpu_values[..valid], cpu_values.as_slice());
    Ok(())
}

/// 1b.1 / R5 — l'append écrit au BON offset (les helpers 1a.5 n'écrivaient
/// qu'au début du buffer) : la 2e ligne ne clobbe pas la 1re.
#[test]
fn kv_append_writes_at_nonzero_offset() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (4, 2, 2); // jouet, kv_dim = 4
    let mut kv = state.full_attention(4, q_heads, kv_heads, head_dim)?;
    let row0_k = [1.0, 2.0, 3.0, 4.0];
    let row0_v = [5.0, 6.0, 7.0, 8.0];
    kv.append_row(&row0_k, &row0_v)?; // offset 0
    let row1_k = [10.0, 20.0, 30.0, 40.0];
    let row1_v = [50.0, 60.0, 70.0, 80.0];
    kv.append_row(&row1_k, &row1_v)?; // offset kv_dim = 4 (NON nul)

    let gpu_keys = read_f32(kv.keys());
    let gpu_values = read_f32(kv.values());
    assert_eq!(&gpu_keys[0..4], &row0_k);
    assert_eq!(&gpu_keys[4..8], &row1_k);
    assert_eq!(&gpu_values[0..4], &row0_v);
    assert_eq!(&gpu_values[4..8], &row1_v);
    Ok(())
}

/// 1b.1 / R5 — overflow : capacité == prefill_len, puis append → `InferError`
/// (pas de corruption silencieuse).
#[test]
fn kv_overflow_at_capacity_errors() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (4, 2, 2);
    let kv_dim = kv_heads * head_dim;
    let prefill_len = 3;
    let mut kv = state.full_attention(prefill_len, q_heads, kv_heads, head_dim)?;
    let seed: Vec<f32> = (0..prefill_len * kv_dim).map(|i| i as f32).collect();
    kv.seed(&seed, &seed, prefill_len)?;
    assert_eq!(kv.len(), kv.capacity());
    let row = vec![0.0_f32; kv_dim];
    assert!(
        kv.append_row(&row, &row).is_err(),
        "append au-delà de la capacité doit échouer"
    );
    Ok(())
}

/// 1b.1 / R4 — la GQA exige q_heads multiple de kv_heads.
#[test]
fn full_attention_state_rejects_non_multiple_gqa() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    assert!(
        state.full_attention(4, 3, 2, 8).is_err(),
        "q_heads=3 non multiple de kv_heads=2 doit être rejeté"
    );
    Ok(())
}

// --- 1b.2 : kernel d'attention decode single-query (attention_decode) ---

/// Données pseudo-aléatoires déterministes dans `[-0.1, 0.1]` (pas de crate
/// rand ; reproductible). Garde des produits scalaires modérés (softmax sain).
fn pseudo(seed: usize) -> f32 {
    let x = (seed.wrapping_mul(2_654_435_761) ^ 0x9E37_79B9) % 1000;
    (x as f32 / 1000.0 - 0.5) * 0.2
}

/// Oracle CPU : réplique EXACTEMENT `cached_attention_one` (decoder.rs:2158) —
/// GQA `kv_head = q_head / (q_heads/kv_heads)`, scale `1/√head_dim`, softmax
/// causal sur `0..len`, somme pondérée des valeurs.
fn cpu_attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    len: usize,
) -> Vec<f32> {
    let kv_dim = kv_heads * head_dim;
    let group = q_heads / kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_heads * head_dim];
    for qh in 0..q_heads {
        let kvh = qh / group;
        let qs = qh * head_dim;
        let ks = kvh * head_dim;
        let mut scores = vec![0.0_f32; len];
        for (r, score) in scores.iter_mut().enumerate() {
            let kstart = r * kv_dim + ks;
            let mut dot = 0.0_f32;
            for c in 0..head_dim {
                dot += q[qs + c] * keys[kstart + c];
            }
            *score = dot * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for score in scores.iter_mut() {
            *score = (*score - max).exp();
            sum += *score;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for score in scores.iter_mut() {
            *score *= inv;
        }
        for (r, &prob) in scores.iter().enumerate() {
            let vstart = r * kv_dim + ks;
            for c in 0..head_dim {
                out[qs + c] += prob * values[vstart + c];
            }
        }
    }
    out
}

/// Différentiel GPU vs CPU pour une longueur de KV donnée, sur les dimensions
/// RÉELLES Qwen3.6-35B-A3B full-attn (q=16, kv=2, hd=256). Tolérance numérique
/// (le kernel change l'ordre de réduction f32 du produit scalaire — réserve E).
fn run_attention_case(len: usize) -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (16, 2, 256);
    let kv_dim = kv_heads * head_dim;
    let mut kv = state.full_attention(len, q_heads, kv_heads, head_dim)?;
    let keys: Vec<f32> = (0..len * kv_dim).map(pseudo).collect();
    let values: Vec<f32> = (0..len * kv_dim).map(|i| pseudo(i + 7)).collect();
    kv.seed(&keys, &values, len)?;

    let q_dim = q_heads * head_dim;
    let q_data: Vec<f32> = (0..q_dim).map(|i| pseudo(i + 99)).collect();

    let gpu = kv.attention_decode(&q_data)?;
    let cpu = cpu_attention(&q_data, &keys, &values, q_heads, kv_heads, head_dim, len);
    assert_eq!(gpu.len(), cpu.len());

    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let delta = (g - c).abs();
        max_abs = max_abs.max(delta);
        sum_abs += delta;
    }
    let mean_abs = sum_abs / gpu.len() as f32;
    // Résidus mesurés (Apple GPU, ce matériel) : len=1 → 0 (bit-exact) ;
    // len=64 → max 7.5e-9 ; len=257 → max 4.7e-9. Seuls l'ordre de réduction
    // f32 du produit scalaire diffère du CPU (réserve E). Bornes ci-dessous =
    // garde-fou de régression (≈4 ordres sous un vrai bug), robuste cross-GPU.
    assert!(max_abs <= 1.0e-4, "len={len}: max_abs={max_abs:e} > 1e-4");
    assert!(
        mean_abs <= 1.0e-5,
        "len={len}: mean_abs={mean_abs:e} > 1e-5"
    );
    Ok(())
}

/// 1b.2 / R5 — len=1 (la requête n'attend qu'elle-même : softmax d'un seul
/// score = 1 → contexte = la valeur du token courant).
#[test]
fn attention_decode_matches_cpu_len1() -> Result<()> {
    run_attention_case(1)
}

/// 1b.2 — len=64 (cas nominal court).
#[test]
fn attention_decode_matches_cpu_len64() -> Result<()> {
    run_attention_case(64)
}

/// 1b.2 / R5 — len=257 : AU-DELÀ du plafond `seq ≤ 256` du kernel de prefill ;
/// prouve que le kernel decode gère un KV non borné (scores en device-scratch).
#[test]
fn attention_decode_matches_cpu_len257() -> Result<()> {
    run_attention_case(257)
}

// --- Micro-jalon R3/R1 : mécaniques du chaînage 40-couches (réserves Codex) ---

fn enc_scale(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input: &Buffer,
    output: &Buffer,
    scale: f32,
    n: usize,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    set_f32(encoder, 2, scale);
    set_u32(encoder, 3, n as u32);
    dispatch(encoder, pipeline, n);
}

fn enc_add(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    output: &Buffer,
    n: usize,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(output), 0);
    set_u32(encoder, 3, n as u32);
    dispatch(encoder, pipeline, n);
}

/// R3 — append device-side (écriture à un OFFSET par un dispatch) PUIS lecture
/// par le dispatch SUIVANT dans le MÊME compute encoder = correct SANS barrière
/// explicite : le hazard-tracking Metal par défaut (ressources tracked,
/// StorageModeShared) ordonne la lecture après l'écriture. Mime exactement
/// l'append KV[len] device-side puis l'attention qui lit KV[0..=len] (le cas
/// signalé par Codex pour 1c). À exécuter avec `MTL_DEBUG_LAYER=1` (Metal API
/// Validation) → aucune erreur attendue.
#[test]
fn resident_device_append_then_read_same_encoder() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let scale_pipeline = build_pipeline(state.device(), "resident_scale_f32");
    let add_pipeline = build_pipeline(state.device(), "resident_add_f32");
    let width = 4;
    let capacity_rows = 3;
    let append_row = 1; // offset NON nul (16 octets)

    let kv = state.persistent(capacity_rows * width, GpuElement::F32)?;
    let row0 = [1.0_f32, 2.0, 3.0, 4.0];
    write_f32_at(&kv, 0, &row0)?;
    let src = state.persistent(width, GpuElement::F32)?;
    let src_data = [10.0_f32, 20.0, 30.0, 40.0];
    write_f32_at(&src, 0, &src_data)?;
    let result = state.persistent(width, GpuElement::F32)?;
    let scale = 0.5_f32;
    let offset = (append_row * width * std::mem::size_of::<f32>()) as u64;

    let queue = state.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    // dispatch 1 (APPEND) : kv[append_row] = src * scale (écriture device à offset).
    encoder.set_compute_pipeline_state(&scale_pipeline);
    encoder.set_buffer(0, Some(src.buffer()), 0);
    encoder.set_buffer(1, Some(kv.buffer()), offset);
    set_f32(encoder, 2, scale);
    set_u32(encoder, 3, width as u32);
    dispatch(encoder, &scale_pipeline, width);
    // dispatch 2 (LECTURE) : result = kv[append_row] + kv[0] — lit la ligne juste
    // écrite par le dispatch précédent, dans le MÊME encoder, sans barrière.
    encoder.set_compute_pipeline_state(&add_pipeline);
    encoder.set_buffer(0, Some(kv.buffer()), offset);
    encoder.set_buffer(1, Some(kv.buffer()), 0);
    encoder.set_buffer(2, Some(result.buffer()), 0);
    set_u32(encoder, 3, width as u32);
    dispatch(encoder, &add_pipeline, width);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = read_f32(&result);
    for (idx, value) in got.iter().enumerate() {
        let expected = src_data[idx] * scale + row0[idx];
        assert!(
            (value - expected).abs() <= 1.0e-5,
            "R3 read-after-write: idx={idx} got={value} exp={expected}"
        );
    }
    Ok(())
}

/// R1 — drop d'un scratch en fin de « couche » + REUSE du slot par la couche
/// suivante AVANT la fin GPU, dans UN encoder = correct (==CPU). Le pool POSSÈDE
/// le buffer (drop = libère le slot, ne désalloue pas) ; les 4 dispatches sur le
/// buffer réutilisé sont ordonnés par le hazard-tracking → la valeur de la
/// couche 0 (lue par son `add`) n'est pas clobbée par le `scale` de la couche 1.
/// C'est la mécanique EXACTE de l'assemblage 40-couches (BLOQUANT 3 Codex).
#[test]
fn resident_scratch_drop_reuse_within_one_encoder() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let scale_pipeline = build_pipeline(state.device(), "resident_scale_f32");
    let add_pipeline = build_pipeline(state.device(), "resident_add_f32");
    let n = 4;
    let x_data = [1.0_f32, 2.0, 3.0, 4.0];
    let scales = [2.0_f32, 5.0];

    let x = state.persistent(n, GpuElement::F32)?;
    write_f32_at(&x, 0, &x_data)?;
    let acc = state.persistent(n, GpuElement::F32)?;
    write_f32_at(&acc, 0, &[0.0_f32; 4])?;
    let pool = state.scratch();

    let queue = state.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Couche 0 : tmp0 = x*scale0 ; acc += tmp0 ; DROP tmp0 (slot libéré).
    {
        let tmp0 = pool.lease(n, GpuElement::F32)?;
        enc_scale(
            encoder,
            &scale_pipeline,
            x.buffer(),
            tmp0.tensor().buffer(),
            scales[0],
            n,
        );
        enc_add(
            encoder,
            &add_pipeline,
            acc.buffer(),
            tmp0.tensor().buffer(),
            acc.buffer(),
            n,
        );
    }
    assert_eq!(
        pool.slot_count(),
        1,
        "un seul slot physique après la couche 0"
    );

    // Couche 1 : tmp1 REUSE le slot de tmp0 (même buffer) ; tmp1 = x*scale1 ; acc += tmp1.
    let tmp1 = pool.lease(n, GpuElement::F32)?;
    assert_eq!(
        pool.slot_count(),
        1,
        "le slot est RÉUTILISÉ (aucune nouvelle alloc)"
    );
    enc_scale(
        encoder,
        &scale_pipeline,
        x.buffer(),
        tmp1.tensor().buffer(),
        scales[1],
        n,
    );
    enc_add(
        encoder,
        &add_pipeline,
        acc.buffer(),
        tmp1.tensor().buffer(),
        acc.buffer(),
        n,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(tmp1); // après le wait

    let got = read_f32(&acc);
    for (idx, value) in got.iter().enumerate() {
        let expected = x_data[idx] * scales[0] + x_data[idx] * scales[1];
        assert!(
            (value - expected).abs() <= 1.0e-5,
            "R1 drop/reuse: idx={idx} got={value} exp={expected}"
        );
    }
    Ok(())
}

// --- 1c.1b / BLOQUANT 1 : gate q_proj (split q/gate GPU + attn_gate) ---

/// `split_q_gate` GPU == `split_attention_gate` CPU (decoder.rs) : désinterleave
/// la projection q_proj gated `[2·q_dim]` (layout par tête `[q | gate]`) sur les
/// dimensions RÉELLES (16 têtes, head_dim 256). Bit-exact (pur gather, aucune
/// arithmétique flottante). Sans lui, 1c devrait relire q/gate sur CPU au milieu.
#[test]
fn split_q_gate_matches_cpu() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (num_heads, head_dim) = (16, 256);
    let q_dim = num_heads * head_dim;
    let proj: Vec<f32> = (0..2 * q_dim)
        .map(|i| (i as f32 - q_dim as f32) * 0.001)
        .collect();
    let (gpu_q, gpu_gate) = state.split_q_gate(&proj, num_heads, head_dim)?;

    let mut cpu_q = Vec::with_capacity(q_dim);
    let mut cpu_gate = Vec::with_capacity(q_dim);
    for head in 0..num_heads {
        let base = head * 2 * head_dim;
        cpu_q.extend_from_slice(&proj[base..base + head_dim]);
        cpu_gate.extend_from_slice(&proj[base + head_dim..base + 2 * head_dim]);
    }
    assert_eq!(gpu_q, cpu_q, "q désinterleavé != split_attention_gate");
    assert_eq!(
        gpu_gate, cpu_gate,
        "gate désinterleavé != split_attention_gate"
    );
    Ok(())
}

/// `apply_attn_gate` GPU == `ctx · σ(gate)` CPU (le gate de sortie full-attn,
/// appliqué après l'attention). Tolérance (fast-math exp vs std exp).
#[test]
fn apply_attn_gate_matches_cpu() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let n = 4096; // q_dim réel Qwen3.6 full-attn
    let ctx: Vec<f32> = (0..n).map(|i| (i as f32 % 7.0 - 3.0) * 0.1).collect();
    let gate: Vec<f32> = (0..n).map(|i| (i as f32 % 5.0 - 2.0) * 0.5).collect();
    let gpu = state.apply_attn_gate(&ctx, &gate)?;

    let mut max_abs = 0.0_f32;
    for i in 0..n {
        let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
        max_abs = max_abs.max((gpu[i] - ctx[i] * sigmoid).abs());
    }
    assert!(max_abs <= 1.0e-6, "attn_gate vs CPU: max_abs={max_abs:e}");
    Ok(())
}

// --- 1c.2 : RoPE-decode position-aware + append KV device-side ---

/// `rms_norm_rope_decode` GPU == rms_norm+RoPE CPU À LA POSITION du token (pas
/// l'index de ligne — le kernel de prefill roterait à 0). Reproduit
/// `rms_norm_rope_heads_at` (decoder.rs).
#[test]
fn rms_norm_rope_decode_matches_cpu() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (num_heads, head_dim, rope_dims) = (2, 8, 8);
    let dim = num_heads * head_dim;
    let position = 5;
    let (eps, theta) = (1.0e-6_f32, 10_000.0_f32);
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 % 11.0 - 5.0) * 0.1).collect();
    let weight: Vec<f32> = (0..head_dim).map(|i| 0.5 + i as f32 * 0.1).collect();
    let gpu = state.rms_norm_rope_decode(
        &input, &weight, num_heads, head_dim, rope_dims, position, eps, theta,
    )?;

    let mut cpu = vec![0.0_f32; dim];
    for head in 0..num_heads {
        let start = head * head_dim;
        let sumsq: f32 = (0..head_dim)
            .map(|c| input[start + c] * input[start + c])
            .sum();
        let inv_rms = 1.0 / ((sumsq / head_dim as f32) + eps).sqrt();
        let normed: Vec<f32> = (0..head_dim)
            .map(|c| input[start + c] * inv_rms * weight[c])
            .collect();
        for c in 0..head_dim {
            cpu[start + c] = if c < rope_dims {
                let pair = c / 2;
                let exponent = (2 * pair) as f32 / rope_dims as f32;
                let angle = position as f32 / theta.powf(exponent);
                let (even, odd) = (normed[pair * 2], normed[pair * 2 + 1]);
                if c == pair * 2 {
                    even * angle.cos() - odd * angle.sin()
                } else {
                    even * angle.sin() + odd * angle.cos()
                }
            } else {
                normed[c]
            };
        }
    }
    let mut max_abs = 0.0_f32;
    for i in 0..dim {
        max_abs = max_abs.max((gpu[i] - cpu[i]).abs());
    }
    assert!(max_abs <= 1.0e-5, "rope decode vs CPU: max_abs={max_abs:e}");
    Ok(())
}

/// `encode_append_kv` (device-side, offset) == append CPU : la ligne K/V du token
/// courant atterrit à `KV[len]` sans clobber `KV[0]`, et `len` avance. Metal API
/// Validation activée.
#[test]
fn encode_append_kv_device_matches_cpu() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (4, 2, 4);
    let kv_dim = kv_heads * head_dim;
    let mut kv = state.full_attention(4, q_heads, kv_heads, head_dim)?;
    let row0: Vec<f32> = (0..kv_dim).map(|i| i as f32).collect();
    kv.seed(&row0, &row0, 1)?;
    let k_data: Vec<f32> = (0..kv_dim).map(|i| 100.0 + i as f32).collect();
    let v_data: Vec<f32> = (0..kv_dim).map(|i| 200.0 + i as f32).collect();
    let k_buf = state.persistent(kv_dim, GpuElement::F32)?;
    write_f32_at(&k_buf, 0, &k_data)?;
    let v_buf = state.persistent(kv_dim, GpuElement::F32)?;
    write_f32_at(&v_buf, 0, &v_data)?;

    let queue = state.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kv.encode_append_kv(encoder, k_buf.buffer(), v_buf.buffer())?;
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    assert_eq!(kv.len(), 2);

    let gpu_keys = read_f32(kv.keys());
    let gpu_values = read_f32(kv.values());
    assert_eq!(&gpu_keys[0..kv_dim], &row0[..], "K row0 clobbé");
    assert_eq!(
        &gpu_keys[kv_dim..2 * kv_dim],
        &k_data[..],
        "K row1 device append"
    );
    assert_eq!(&gpu_values[0..kv_dim], &row0[..], "V row0 clobbé");
    assert_eq!(
        &gpu_values[kv_dim..2 * kv_dim],
        &v_data[..],
        "V row1 device append"
    );
    Ok(())
}
