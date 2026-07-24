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

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

fn bf16_round_to_f32(value: f32) -> f32 {
    f32::from_bits(u32::from(f32_to_bf16_bits(value)) << 16)
}

fn assert_close_eps(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len(), "longueurs différentes");
    let mut max_abs = 0.0_f32;
    let mut mean_abs = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        let delta = (actual - expected).abs();
        max_abs = max_abs.max(delta);
        mean_abs += delta;
    }
    mean_abs /= actual.len().max(1) as f32;
    assert!(
        max_abs <= eps,
        "écart max {max_abs:e} > {eps:e} (mean {mean_abs:e})"
    );
}

#[allow(
    unsafe_code,
    reason = "lecture d'un MTLBuffer partagé après wait_until_completed (test)"
)]
fn read_bf16_as_f32(tensor: &GpuTensor) -> Vec<f32> {
    let ptr = tensor.buffer().contents().cast::<u16>();
    assert!(!ptr.is_null(), "MTLBuffer partagé sans pointeur CPU");
    let mut out = Vec::with_capacity(tensor.len());
    // SAFETY: buffer en StorageModeShared dont les écritures CPU/GPU ont fini
    // pour les tests ; lecture de `len` u16 dans les bornes du tenseur bf16.
    unsafe {
        for index in 0..tensor.len() {
            let bits = u32::from(ptr.add(index).read()) << 16;
            out.push(f32::from_bits(bits));
        }
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

#[test]
fn pipeline_scratch_slots_are_physically_disjoint() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    state.prepare_pipeline_scratch(2);

    let base_address = state
        .scratch()
        .lease(16, GpuElement::F32)?
        .tensor()
        .buffer()
        .gpu_address();
    let slot_0_address = {
        let _guard = install_pipeline_scratch_slot(0);
        state
            .scratch()
            .lease(16, GpuElement::F32)?
            .tensor()
            .buffer()
            .gpu_address()
    };
    let slot_1_address = {
        let _guard = install_pipeline_scratch_slot(1);
        state
            .scratch()
            .lease(16, GpuElement::F32)?
            .tensor()
            .buffer()
            .gpu_address()
    };

    assert_ne!(base_address, slot_0_address);
    assert_ne!(base_address, slot_1_address);
    assert_ne!(slot_0_address, slot_1_address);
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
    let mut kv = state.full_attention(capacity, q_heads, kv_heads, head_dim, false)?;
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

/// C1B — seed prefill puis append decode en KV bf16 : les lignes valides sont
/// arrondies bf16 (RNE), la longueur logique reste en lignes et la taille mémoire
/// est divisée par deux.
#[test]
fn kv_bf16_seed_then_append_rounds_to_bf16() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (16, 2, 256);
    let kv_dim = kv_heads * head_dim;
    let capacity = 4;
    let mut kv = state.full_attention_bf16_for_test(capacity, q_heads, kv_heads, head_dim)?;
    assert_eq!(kv.keys().element(), GpuElement::Bf16);
    assert_eq!(kv.values().element(), GpuElement::Bf16);
    assert_eq!(kv.keys().byte_len(), capacity * kv_dim * 2);

    let seed_rows = 2;
    let seed_k: Vec<f32> = (0..seed_rows * kv_dim)
        .map(|i| (i as f32 - 333.0) * 0.0017)
        .collect();
    let seed_v: Vec<f32> = (0..seed_rows * kv_dim)
        .map(|i| (i as f32 + 17.0) * -0.0023)
        .collect();
    kv.seed(&seed_k, &seed_v, seed_rows)?;

    let row_k: Vec<f32> = (0..kv_dim).map(|i| 1.0 + i as f32 * 0.0009).collect();
    let row_v: Vec<f32> = (0..kv_dim).map(|i| -1.0 + i as f32 * 0.0011).collect();
    kv.append_row(&row_k, &row_v)?;
    assert_eq!(kv.len(), seed_rows + 1);

    let mut cpu_k = seed_k;
    cpu_k.extend_from_slice(&row_k);
    let mut cpu_v = seed_v;
    cpu_v.extend_from_slice(&row_v);
    let valid = kv.len() * kv_dim;
    let gpu_k = read_bf16_as_f32(kv.keys());
    let gpu_v = read_bf16_as_f32(kv.values());
    for (idx, (&gpu, &cpu)) in gpu_k[..valid].iter().zip(cpu_k.iter()).enumerate() {
        let expected = bf16_round_to_f32(cpu);
        assert_eq!(gpu, expected, "K bf16[{idx}] gpu={gpu} expected={expected}");
    }
    for (idx, (&gpu, &cpu)) in gpu_v[..valid].iter().zip(cpu_v.iter()).enumerate() {
        let expected = bf16_round_to_f32(cpu);
        assert_eq!(gpu, expected, "V bf16[{idx}] gpu={gpu} expected={expected}");
    }
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
    let mut kv = state.full_attention(4, q_heads, kv_heads, head_dim, false)?;
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
    let mut kv = state.full_attention(prefill_len, q_heads, kv_heads, head_dim, false)?;
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
        state.full_attention(4, 3, 2, 8, false).is_err(),
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

fn gemma_tail_value(seed: usize, index: usize, scale: f32) -> f32 {
    let mixed = seed
        .wrapping_mul(1_103_515_245)
        .wrapping_add(index.wrapping_mul(12_345))
        ^ 0xA5A5_9669;
    (((mixed % 2001) as f32 / 1000.0) - 1.0) * scale
}

fn gemma_tail_weight(rows: usize, cols: usize, seed: usize, scale: f32) -> Result<crate::Tensor> {
    crate::Tensor::from_vec(
        vec![rows, cols],
        (0..rows * cols)
            .map(|idx| gemma_tail_value(seed, idx, scale))
            .collect(),
    )
}

fn gemma_tail_norm(hidden: usize, seed: usize) -> Result<crate::Tensor> {
    crate::Tensor::from_vec(
        vec![hidden],
        (0..hidden)
            .map(|idx| 0.75 + gemma_tail_value(seed, idx, 0.18).abs())
            .collect(),
    )
}

fn gemma_tail_affine_weight(
    rows: usize,
    cols: usize,
    seed: usize,
    scale: f32,
) -> Result<crate::AffineQuantizedTensor> {
    let bits = 8_usize;
    let values_per_word = 32 / bits;
    if cols % values_per_word != 0 {
        return Err(InferError::Dimension(format!(
            "fixture affine cols={cols} non multiple de {values_per_word}"
        )));
    }
    let packed_cols = cols / values_per_word;
    let group_size = cols;
    let groups = cols / group_size;
    let mut packed = Vec::with_capacity(rows * packed_cols);
    for row in 0..rows {
        for word in 0..packed_cols {
            let mut value = 0_u32;
            for lane in 0..values_per_word {
                let col = word * values_per_word + lane;
                let quant = ((seed + row * 17 + col * 31 + lane * 7) % 23 + 3) as u32;
                value |= quant << (lane * bits);
            }
            packed.push(value);
        }
    }
    let scales = crate::Tensor::from_vec(
        vec![rows, groups],
        (0..rows * groups)
            .map(|idx| bf16_round_to_f32(scale * (0.7 + ((seed + idx) % 5) as f32 * 0.05)))
            .collect(),
    )?;
    let biases = crate::Tensor::from_vec(
        vec![rows, groups],
        (0..rows * groups)
            .map(|idx| bf16_round_to_f32(gemma_tail_value(seed + 97, idx, scale * 3.0)))
            .collect(),
    )?;
    crate::AffineQuantizedTensor::new(
        &[rows, packed_cols],
        packed,
        scales,
        biases,
        group_size,
        bits,
    )
}

fn gemma_moe_experts(
    experts: usize,
    hidden: usize,
    inter: usize,
    seed: usize,
) -> Result<Vec<crate::GatedMlp>> {
    let mut out = Vec::with_capacity(experts);
    for expert in 0..experts {
        let gate = gemma_tail_affine_weight(inter, hidden, seed + expert * 101 + 1, 0.0035)?;
        let up = gemma_tail_affine_weight(inter, hidden, seed + expert * 101 + 17, 0.003)?;
        let down = gemma_tail_affine_weight(hidden, inter, seed + expert * 101 + 41, 0.0025)?;
        out.push(
            crate::GatedMlp::new(
                crate::Linear::new_quantized(gate, None)?,
                crate::Linear::new_quantized(up, None)?,
                crate::Linear::new_quantized(down, None)?,
            )
            .with_activation(crate::Activation::GeluTanh),
        );
    }
    Ok(out)
}

fn gemma_moe_router_weight(experts: usize, hidden: usize, seed: usize) -> Result<crate::Tensor> {
    crate::Tensor::from_vec(
        vec![experts, hidden],
        (0..experts * hidden)
            .map(|idx| {
                let expert = idx / hidden;
                let col = idx % hidden;
                let diagonal = if col == expert % hidden { 0.11 } else { 0.0 };
                gemma_tail_value(seed + expert * 13, col, 0.18) + diagonal - expert as f32 * 0.007
            })
            .collect(),
    )
}

#[derive(Clone, Copy)]
struct GemmaTailCase {
    rows: usize,
    hidden: usize,
    inter: usize,
    seed: usize,
    embed_scale: f32,
    layer_scalar: f32,
}

fn run_gemma_dense_tail_case(case: GemmaTailCase) -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let executor = MetalExecutor::new()?;
    let eps = 1.0e-6_f32;
    let hidden_data: Vec<f32> = (0..case.rows * case.hidden)
        .map(|idx| gemma_tail_value(case.seed, idx, 0.7) * case.embed_scale)
        .collect();
    let hidden = crate::Tensor::from_vec(vec![case.rows, case.hidden], hidden_data.clone())?;
    let pre_norm = gemma_tail_norm(case.hidden, case.seed + 11)?;
    let post_norm = gemma_tail_norm(case.hidden, case.seed + 23)?;
    let gate_weight = gemma_tail_weight(case.inter, case.hidden, case.seed + 31, 0.09)?;
    let up_weight = gemma_tail_weight(case.inter, case.hidden, case.seed + 43, 0.08)?;
    let down_weight = gemma_tail_weight(case.hidden, case.inter, case.seed + 59, 0.07)?;
    let mlp = crate::GatedMlp::new(
        crate::Linear::new(gate_weight.clone(), None)?,
        crate::Linear::new(up_weight.clone(), None)?,
        crate::Linear::new(down_weight.clone(), None)?,
    )
    .with_activation(crate::Activation::GeluTanh);

    let ffn_input = crate::rms_norm(&hidden, &pre_norm, eps)?;
    let ffn_out = mlp.forward_with_runtime(&ffn_input, crate::ForwardRuntime::metal(&executor))?;
    let ffn_normed = crate::rms_norm(&ffn_out, &post_norm, eps)?;
    let oracle = hidden
        .add(&ffn_normed)?
        .map(|value| value * case.layer_scalar);

    let hidden_buf = state.persistent(case.rows * case.hidden, GpuElement::F32)?;
    write_f32_at(&hidden_buf, 0, &hidden_data)?;
    let out_buf = state.persistent(case.rows * case.hidden, GpuElement::F32)?;
    let pre_norm_buf = executor.cached_buffer_from_f32(pre_norm.data(), "gemma_tail_pre_norm")?;
    let post_norm_buf =
        executor.cached_buffer_from_f32(post_norm.data(), "gemma_tail_post_norm")?;
    let gate_proj = executor.resolve_linear_weight_buffers(
        &crate::LinearWeight::Dense(gate_weight),
        "gemma_tail_gate",
    )?;
    let up_proj = executor
        .resolve_linear_weight_buffers(&crate::LinearWeight::Dense(up_weight), "gemma_tail_up")?;
    let down_proj = executor.resolve_linear_weight_buffers(
        &crate::LinearWeight::Dense(down_weight),
        "gemma_tail_down",
    )?;

    let command_buffer = state.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    state.encode_gemma_dense_tail(
        &executor,
        encoder,
        &mut owned,
        hidden_buf.buffer(),
        out_buf.buffer(),
        case.rows,
        case.hidden,
        eps,
        &gate_proj,
        &up_proj,
        &down_proj,
        GemmaDenseTailWeights {
            pre_feedforward_norm: &pre_norm_buf,
            post_feedforward_norm: &post_norm_buf,
            layer_scalar: Some(case.layer_scalar),
        },
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32(&out_buf);
    // Near-tie documenté : l'oracle per-op garde RMSNorm et gelu_tanh côté CPU,
    // alors que cette brique résidente les évalue dans Metal (réductions par
    // threadgroup + fast-math `tanh`). Les écarts attendus restent au niveau ULP.
    assert_close_eps(&gpu, oracle.data(), 2.0e-5);
    Ok(())
}

/// Phase 1 Gemma : le tail dense résident reproduit l'oracle per-op générique.
#[test]
fn gemma_dense_tail_resident_matches_per_op_oracle() -> Result<()> {
    for case in [
        GemmaTailCase {
            rows: 1,
            hidden: 4,
            inter: 6,
            seed: 3,
            embed_scale: 1.75,
            layer_scalar: 0.95,
        },
        GemmaTailCase {
            rows: 2,
            hidden: 8,
            inter: 10,
            seed: 17,
            embed_scale: 2.25,
            layer_scalar: 1.1,
        },
        GemmaTailCase {
            rows: 3,
            hidden: 16,
            inter: 12,
            seed: 29,
            embed_scale: 0.85,
            layer_scalar: 0.8,
        },
    ] {
        run_gemma_dense_tail_case(case)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct GemmaMoeTailCase {
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
    seed: usize,
    embed_scale: f32,
    layer_scalar: f32,
}

fn run_gemma_moe_tail_case(case: GemmaMoeTailCase) -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let executor = MetalExecutor::new()?;
    let eps = 1.0e-6_f32;
    let hidden_data: Vec<f32> = (0..case.hidden)
        .map(|idx| gemma_tail_value(case.seed, idx, 0.55) * case.embed_scale)
        .collect();
    let hidden = crate::Tensor::from_vec(vec![1, case.hidden], hidden_data.clone())?;
    let pre_norm = gemma_tail_norm(case.hidden, case.seed + 211)?;
    let post_norm = gemma_tail_norm(case.hidden, case.seed + 223)?;
    let router_weight = gemma_moe_router_weight(case.experts, case.hidden, case.seed + 239)?;
    let router = crate::Linear::new(router_weight, None)?;
    let experts = gemma_moe_experts(case.experts, case.hidden, case.inter, case.seed + 251)?;
    let moe = crate::MoeMlp::new(router.clone(), experts.clone(), None, None, case.top_k)?;

    let ffn_input = crate::rms_norm(&hidden, &pre_norm, eps)?;
    let moe_out = moe.forward_with_runtime(&ffn_input, crate::ForwardRuntime::cpu())?;
    let ffn_normed = crate::rms_norm(&moe_out, &post_norm, eps)?;
    let oracle = hidden
        .add(&ffn_normed)?
        .map(|value| value * case.layer_scalar);

    let hidden_buf = state.persistent(case.hidden, GpuElement::F32)?;
    write_f32_at(&hidden_buf, 0, &hidden_data)?;
    let out_buf = state.persistent(case.hidden, GpuElement::F32)?;
    let pre_norm_buf =
        executor.cached_buffer_from_f32(pre_norm.data(), "gemma_moe_tail_pre_norm")?;
    let post_norm_buf =
        executor.cached_buffer_from_f32(post_norm.data(), "gemma_moe_tail_post_norm")?;
    let moe_weights = executor.resolve_moe_routed_weights(&router, &experts)?;

    let command_buffer = state.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    state.encode_gemma_moe_tail(
        &executor,
        encoder,
        &mut owned,
        hidden_buf.buffer(),
        out_buf.buffer(),
        case.hidden,
        eps,
        GemmaMoeTailWeights {
            pre_feedforward_norm: &pre_norm_buf,
            post_feedforward_norm: &post_norm_buf,
            layer_scalar: Some(case.layer_scalar),
            moe: &moe_weights,
            top_k: case.top_k,
        },
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    let gpu = read_f32(&out_buf);
    // Near-tie documenté : l'oracle per-op générique garde le routage MoE et
    // GeGLU en CPU f32 ; le résident évalue top-k softmax, GeGLU tanh et RMSNorm
    // dans Metal (fast-math + réductions threadgroup, scales/biases bf16).
    assert_close_eps(&gpu, oracle.data(), 5.0e-5);
    Ok(())
}

/// Phase 1b Gemma : le tail MoE routed résident reproduit l'oracle per-op.
#[test]
fn gemma_moe_tail_resident_matches_per_op_oracle() -> Result<()> {
    for case in [
        GemmaMoeTailCase {
            hidden: 4,
            inter: 4,
            experts: 3,
            top_k: 2,
            seed: 5,
            embed_scale: 1.6,
            layer_scalar: 0.92,
        },
        GemmaMoeTailCase {
            hidden: 8,
            inter: 4,
            experts: 4,
            top_k: 2,
            seed: 23,
            embed_scale: 2.05,
            layer_scalar: 1.08,
        },
        GemmaMoeTailCase {
            hidden: 8,
            inter: 8,
            experts: 5,
            top_k: 3,
            seed: 41,
            embed_scale: 0.9,
            layer_scalar: 0.83,
        },
    ] {
        run_gemma_moe_tail_case(case)?;
    }
    Ok(())
}

/// Le tail parallèle Gemma 4 du decode résident reproduit la référence CPU.
#[test]
fn gemma_parallel_tail_decode_resident_matches_cpu() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    const HIDDEN: usize = 4;
    const DENSE_INTER: usize = 6;
    const MOE_INTER: usize = 4;
    const EXPERTS: usize = 2;
    const TOP_K: usize = 1;
    const EPS: f32 = 1.0e-6;
    const LAYER_SCALE: f32 = 0.93;

    let executor = MetalExecutor::new()?;
    let hidden = crate::Tensor::row(vec![0.8, -0.4, 0.3, -0.2])?;
    let pre_ffn = gemma_tail_norm(HIDDEN, 311)?;
    let post_ffn_1 = gemma_tail_norm(HIDDEN, 313)?;
    let pre_ffn_2 = crate::Tensor::from_vec(vec![HIDDEN], vec![0.05, 4.0, 0.05, 0.05])?;
    let post_ffn_2 = gemma_tail_norm(HIDDEN, 317)?;
    let post_ffn = gemma_tail_norm(HIDDEN, 331)?;
    let router_norm = crate::Tensor::from_vec(vec![HIDDEN], vec![1.0; HIDDEN])?;
    let per_expert_scale = crate::Tensor::from_vec(vec![EXPERTS], vec![1.0, 1.0])?;
    let dense_gate_weight = gemma_tail_weight(DENSE_INTER, HIDDEN, 337, 0.09)?;
    let dense_up_weight = gemma_tail_weight(DENSE_INTER, HIDDEN, 347, 0.08)?;
    let dense_down_weight = gemma_tail_weight(HIDDEN, DENSE_INTER, 349, 0.07)?;
    let dense = crate::GatedMlp::new(
        crate::Linear::new(dense_gate_weight.clone(), None)?,
        crate::Linear::new(dense_up_weight.clone(), None)?,
        crate::Linear::new(dense_down_weight.clone(), None)?,
    )
    .with_activation(crate::Activation::GeluTanh);
    let router = crate::Linear::new(
        crate::Tensor::from_vec(
            vec![EXPERTS, HIDDEN],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        )?,
        None,
    )?;
    let experts = gemma_moe_experts(EXPERTS, HIDDEN, MOE_INTER, 353)?;
    let moe = crate::MoeMlp::new(router.clone(), experts.clone(), None, None, TOP_K)?
        .with_router_norm(router_norm.clone(), EPS)
        .with_per_expert_scale(per_expert_scale.clone());

    let dense_input = crate::rms_norm(&hidden, &pre_ffn, EPS)?;
    let dense_raw = dense.forward_with_runtime(&dense_input, crate::ForwardRuntime::cpu())?;
    let dense_out = crate::rms_norm(&dense_raw, &post_ffn_1, EPS)?;
    let moe_input = crate::rms_norm(&hidden, &pre_ffn_2, EPS)?;
    let moe_raw =
        moe.forward_with_router_source(&moe_input, &hidden, crate::ForwardRuntime::cpu())?;
    let moe_out = crate::rms_norm(&moe_raw, &post_ffn_2, EPS)?;
    let ffn_out = dense_out.add(&moe_out)?;
    let ffn_normed = crate::rms_norm(&ffn_out, &post_ffn, EPS)?;
    let oracle = hidden.add(&ffn_normed)?.map(|value| value * LAYER_SCALE);

    let hidden_buf = state.persistent(HIDDEN, GpuElement::F32)?;
    write_f32_at(&hidden_buf, 0, hidden.data())?;
    let out_buf = state.persistent(HIDDEN, GpuElement::F32)?;
    let pre_ffn_buf = executor.cached_buffer_from_f32(pre_ffn.data(), "decode_parallel_pre_ffn")?;
    let post_ffn_1_buf =
        executor.cached_buffer_from_f32(post_ffn_1.data(), "decode_parallel_post_ffn_1")?;
    let pre_ffn_2_buf =
        executor.cached_buffer_from_f32(pre_ffn_2.data(), "decode_parallel_pre_ffn_2")?;
    let post_ffn_2_buf =
        executor.cached_buffer_from_f32(post_ffn_2.data(), "decode_parallel_post_ffn_2")?;
    let post_ffn_buf =
        executor.cached_buffer_from_f32(post_ffn.data(), "decode_parallel_post_ffn")?;
    let router_norm_buf =
        executor.cached_buffer_from_f32(router_norm.data(), "decode_parallel_router_norm")?;
    let per_expert_scale_buf = executor
        .cached_buffer_from_f32(per_expert_scale.data(), "decode_parallel_per_expert_scale")?;
    let dense_gate_proj = executor.resolve_linear_weight_buffers(
        &crate::LinearWeight::Dense(dense_gate_weight),
        "decode_parallel_dense_gate",
    )?;
    let dense_up_proj = executor.resolve_linear_weight_buffers(
        &crate::LinearWeight::Dense(dense_up_weight),
        "decode_parallel_dense_up",
    )?;
    let dense_down_proj = executor.resolve_linear_weight_buffers(
        &crate::LinearWeight::Dense(dense_down_weight),
        "decode_parallel_dense_down",
    )?;
    let moe_weights = executor.resolve_moe_routed_weights(&router, &experts)?;

    let command_buffer = state.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let mut owned = Vec::new();
    state.encode_gemma_parallel_moe_tail(
        &executor,
        encoder,
        &mut owned,
        hidden_buf.buffer(),
        out_buf.buffer(),
        1,
        HIDDEN,
        EPS,
        GemmaParallelMoeTailWeights {
            dense_gate_proj: &dense_gate_proj,
            dense_up_proj: &dense_up_proj,
            dense_down_proj: &dense_down_proj,
            pre_feedforward_norm: &pre_ffn_buf,
            post_feedforward_norm_1: &post_ffn_1_buf,
            moe: &moe_weights,
            top_k: TOP_K,
            router_norm: Some((&router_norm_buf, EPS)),
            per_expert_scale: Some(&per_expert_scale_buf),
            pre_feedforward_norm_2: &pre_ffn_2_buf,
            post_feedforward_norm_2: &post_ffn_2_buf,
            post_feedforward_norm: &post_ffn_buf,
            layer_scalar: Some(LAYER_SCALE),
            dense_inter_dim: DENSE_INTER,
        },
    )?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;

    assert_close_eps(&read_f32(&out_buf), oracle.data(), 3.0e-5);
    Ok(())
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
    cpu_attention_windowed(q, keys, values, q_heads, kv_heads, head_dim, len, 0)
}

#[allow(
    clippy::too_many_arguments,
    reason = "oracle CPU de test: tenseurs, dimensions et fenêtre doivent rester explicites"
)]
fn cpu_attention_windowed(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    len: usize,
    window_start: usize,
) -> Vec<f32> {
    let kv_dim = kv_heads * head_dim;
    let group = q_heads / kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_heads * head_dim];
    for qh in 0..q_heads {
        let kvh = qh / group;
        let qs = qh * head_dim;
        let ks = kvh * head_dim;
        let mut scores = vec![0.0_f32; len - window_start];
        for (offset, score) in scores.iter_mut().enumerate() {
            let r = window_start + offset;
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
        for (offset, &prob) in scores.iter().enumerate() {
            let r = window_start + offset;
            let vstart = r * kv_dim + ks;
            for c in 0..head_dim {
                out[qs + c] += prob * values[vstart + c];
            }
        }
    }
    out
}

#[allow(
    clippy::too_many_arguments,
    reason = "oracle CPU de test: tenseurs et paramètres RoPE doivent rester explicites"
)]
fn cpu_rms_norm_rope_heads_at(
    input: &[f32],
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    rope_dims: usize,
    position: usize,
    eps: f32,
    theta: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; num_heads * head_dim];
    let pairs = rope_dims / 2;
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
            out[start + c] = if c < rope_dims {
                let pair = if c < pairs { c } else { c - pairs };
                let exponent = (2 * pair) as f32 / rope_dims as f32;
                let angle = position as f32 / theta.powf(exponent);
                let (first, second) = (normed[pair], normed[pair + pairs]);
                if c < pairs {
                    first * angle.cos() - second * angle.sin()
                } else {
                    first * angle.sin() + second * angle.cos()
                }
            } else {
                normed[c]
            };
        }
    }
    out
}

fn run_gemma4_global_attention_decode_case(prefill_len: usize, seed: usize) -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (4_usize, 2_usize, 16_usize);
    let rope_dims = head_dim / 4;
    let kv_dim = kv_heads * head_dim;
    let (eps, theta) = (1.0e-6_f32, 1_000_000.0_f32);
    let q_norm: Vec<f32> = (0..head_dim)
        .map(|i| 0.7 + pseudo(seed + 11 * i).abs())
        .collect();
    let k_norm: Vec<f32> = (0..head_dim)
        .map(|i| 0.8 + pseudo(seed + 17 * i).abs())
        .collect();

    let mut prefill_keys = Vec::with_capacity(prefill_len * kv_dim);
    let mut prefill_values = Vec::with_capacity(prefill_len * kv_dim);
    for row in 0..prefill_len {
        let k_raw: Vec<f32> = (0..kv_dim).map(|i| pseudo(seed + row * 131 + i)).collect();
        let v_raw: Vec<f32> = (0..kv_dim)
            .map(|i| pseudo(seed + 10_000 + row * 137 + i))
            .collect();
        prefill_keys.extend(cpu_rms_norm_rope_heads_at(
            &k_raw, &k_norm, kv_heads, head_dim, rope_dims, row, eps, theta,
        ));
        prefill_values.extend(v_raw);
    }

    let position = prefill_len;
    let q_raw: Vec<f32> = (0..q_heads * head_dim)
        .map(|i| pseudo(seed + 20_000 + i))
        .collect();
    let k_raw: Vec<f32> = (0..kv_dim).map(|i| pseudo(seed + 30_000 + i)).collect();
    let v_raw: Vec<f32> = (0..kv_dim).map(|i| pseudo(seed + 40_000 + i)).collect();
    let q_cpu = cpu_rms_norm_rope_heads_at(
        &q_raw, &q_norm, q_heads, head_dim, rope_dims, position, eps, theta,
    );
    let k_cpu = cpu_rms_norm_rope_heads_at(
        &k_raw, &k_norm, kv_heads, head_dim, rope_dims, position, eps, theta,
    );
    let q_gpu = state.rms_norm_rope_decode(
        &q_raw, &q_norm, q_heads, head_dim, rope_dims, position, eps, theta,
    )?;
    let k_gpu = state.rms_norm_rope_decode(
        &k_raw, &k_norm, kv_heads, head_dim, rope_dims, position, eps, theta,
    )?;

    let mut kv = state.full_attention(prefill_len + 1, q_heads, kv_heads, head_dim, false)?;
    kv.seed(&prefill_keys, &prefill_values, prefill_len)?;
    kv.append_row(&k_gpu, &v_raw)?;
    let gpu = kv.attention_decode(&q_gpu)?;

    let mut keys = prefill_keys;
    keys.extend(k_cpu);
    let mut values = prefill_values;
    values.extend(v_raw);
    let cpu = cpu_attention(
        &q_cpu,
        &keys,
        &values,
        q_heads,
        kv_heads,
        head_dim,
        prefill_len + 1,
    );
    assert_eq!(gpu.len(), cpu.len());

    let mut max_abs = 0.0_f32;
    let mut mean_abs = 0.0_f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let delta = (g - c).abs();
        max_abs = max_abs.max(delta);
        mean_abs += delta;
    }
    mean_abs /= gpu.len() as f32;
    // Le chemin résident utilise fast-math Metal et une réduction SDPA GPU dont
    // l'ordre diffère de l'oracle CPU. Les cas ci-dessous restent typiquement
    // sous 1e-6 ; 1e-4/1e-5 laisse de la marge cross-GPU sans couvrir une erreur
    // de position RoPE, de GQA ou de rope_dims (observée à >1e-2).
    assert!(
        max_abs <= 1.0e-4,
        "gemma4 global prefill={prefill_len} seed={seed}: max_abs={max_abs:e} > 1e-4"
    );
    assert!(
        mean_abs <= 1.0e-5,
        "gemma4 global prefill={prefill_len} seed={seed}: mean_abs={mean_abs:e} > 1e-5"
    );
    Ok(())
}

/// Phase 2a Gemma 4 : attention globale decode résidente avec `head_dim` réduit
/// et RoPE partiel 0.25, KV seedé depuis un préfill court.
#[test]
fn gemma4_global_attention_decode_resident_matches_per_op_oracle() -> Result<()> {
    for (prefill_len, seed) in [(1_usize, 3_usize), (4, 19), (9, 41)] {
        run_gemma4_global_attention_decode_case(prefill_len, seed)?;
    }
    Ok(())
}

fn run_gemma4_sliding_attention_decode_case(position: usize, seed: usize) -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim, window) = (4_usize, 2_usize, 256_usize, 4_usize);
    let len = position + 1;
    let window_start = len.saturating_sub(window);
    let kv_dim = kv_heads * head_dim;
    let (eps, theta) = (1.0e-6_f32, 10_000.0_f32);
    let q_norm: Vec<f32> = (0..head_dim)
        .map(|i| 0.75 + pseudo(seed + 7 * i).abs())
        .collect();
    let k_norm: Vec<f32> = (0..head_dim)
        .map(|i| 0.8 + pseudo(seed + 11 * i).abs())
        .collect();
    let q_raw: Vec<f32> = (0..q_heads * head_dim)
        .map(|i| pseudo(seed + 20_000 + i))
        .collect();
    let q = state.rms_norm_rope_decode(
        &q_raw, &q_norm, q_heads, head_dim, head_dim, position, eps, theta,
    )?;

    let mut keys = Vec::with_capacity(len * kv_dim);
    let mut values = Vec::with_capacity(len * kv_dim);
    for row in 0..len {
        let k_raw: Vec<f32> = (0..kv_dim).map(|i| pseudo(seed + row * 521 + i)).collect();
        keys.extend(cpu_rms_norm_rope_heads_at(
            &k_raw, &k_norm, kv_heads, head_dim, head_dim, row, eps, theta,
        ));
        values.extend((0..kv_dim).map(|i| {
            let base = pseudo(seed + 50_000 + row * 523 + i);
            if row < window_start {
                base + 2.0
            } else {
                base
            }
        }));
    }

    let mut kv = state.full_attention(len, q_heads, kv_heads, head_dim, false)?;
    kv.seed(&keys, &values, len)?;
    let gpu = kv.attention_decode_windowed(&q, window_start)?;
    let oracle = cpu_attention_windowed(
        &q,
        &keys,
        &values,
        q_heads,
        kv_heads,
        head_dim,
        len,
        window_start,
    );
    let unwindowed = cpu_attention(&q, &keys, &values, q_heads, kv_heads, head_dim, len);
    let max_abs = gpu
        .iter()
        .zip(&oracle)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let mean_abs = gpu
        .iter()
        .zip(&oracle)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / gpu.len() as f32;
    // La réduction Metal additionne les 256 composantes en arbre, tandis que
    // l'oracle les additionne en série. Cette différence f32 atteint ~1e-5 sur
    // les seeds ci-dessous ; 2e-4 (max) / 2e-5 (moyenne) couvre les near-ties
    // cross-GPU sans masquer l'inclusion d'une ancienne ligne (écart > 1e-2).
    assert!(
        max_abs <= 2.0e-4,
        "sliding position={position} seed={seed}: max_abs={max_abs:e}"
    );
    assert!(
        mean_abs <= 2.0e-5,
        "sliding position={position} seed={seed}: mean_abs={mean_abs:e}"
    );
    if window_start > 0 {
        let excluded_delta = oracle
            .iter()
            .zip(&unwindowed)
            .map(|(windowed, full)| (windowed - full).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            excluded_delta > 1.0e-2,
            "fixture non discriminante: delta={excluded_delta:e}"
        );
    }
    Ok(())
}

/// Phase 2b Gemma 4 : GQA, q/k norm, RoPE complet et KV plus long que la fenêtre.
#[test]
fn gemma4_sliding_attention_decode_resident_matches_windowed_oracle() -> Result<()> {
    for (position, seed) in [(1_usize, 5_usize), (4, 23), (8, 47)] {
        run_gemma4_sliding_attention_decode_case(position, seed)?;
    }
    Ok(())
}

/// `window_start=0` emprunte exactement le chemin historique flash/2-pass.
#[test]
fn attention_decode_window_start_zero_is_byte_exact() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    for seed in [13_usize, 37, 71] {
        let (q_heads, kv_heads, head_dim, len) = (4_usize, 2_usize, 256_usize, 7_usize);
        let kv_dim = kv_heads * head_dim;
        let keys: Vec<f32> = (0..len * kv_dim).map(|i| pseudo(seed + i)).collect();
        let values: Vec<f32> = (0..len * kv_dim)
            .map(|i| pseudo(seed + 10_000 + i))
            .collect();
        let q: Vec<f32> = (0..q_heads * head_dim)
            .map(|i| pseudo(seed + 20_000 + i))
            .collect();
        let mut kv = state.full_attention(len, q_heads, kv_heads, head_dim, false)?;
        kv.seed(&keys, &values, len)?;
        assert_eq!(
            kv.attention_decode(&q)?,
            kv.attention_decode_windowed(&q, 0)?
        );
    }
    Ok(())
}

/// Non-régression Qwen : head_dim/RoPE uniformes (`rope_dims=head_dim`) sur le
/// chemin full-attn résident. Avec une seule ligne KV, softmax vaut exactement 1,
/// donc le contexte doit être byte-identique à l'oracle.
#[test]
fn qwen_uniform_attention_decode_resident_is_byte_exact_single_key() -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (4_usize, 2_usize, 16_usize);
    let rope_dims = head_dim;
    let kv_dim = kv_heads * head_dim;
    let (eps, theta, position) = (1.0e-6_f32, 10_000.0_f32, 7_usize);
    let q_norm = vec![1.0_f32; head_dim];
    let k_norm = vec![1.0_f32; head_dim];
    let q_raw: Vec<f32> = (0..q_heads * head_dim)
        .map(|i| pseudo(50_000 + i))
        .collect();
    let k_raw: Vec<f32> = (0..kv_dim).map(|i| pseudo(60_000 + i)).collect();
    let value: Vec<f32> = (0..kv_dim).map(|i| pseudo(70_000 + i)).collect();
    let q = state.rms_norm_rope_decode(
        &q_raw, &q_norm, q_heads, head_dim, rope_dims, position, eps, theta,
    )?;
    let k = state.rms_norm_rope_decode(
        &k_raw, &k_norm, kv_heads, head_dim, rope_dims, position, eps, theta,
    )?;

    let mut kv = state.full_attention(1, q_heads, kv_heads, head_dim, false)?;
    kv.append_row(&k, &value)?;
    let gpu = kv.attention_decode(&q)?;
    let cpu = cpu_attention(&q, &k, &value, q_heads, kv_heads, head_dim, 1);
    assert_eq!(gpu, cpu);
    Ok(())
}

/// Différentiel GPU vs CPU pour une longueur de KV donnée, sur les dimensions
/// RÉELLES Qwen3.6-35B-A3B full-attn (q=16, kv=2, hd=256). Tolérance numérique
/// (le kernel change l'ordre de réduction f32 du produit scalaire — réserve E).
fn run_attention_case(len: usize) -> Result<()> {
    run_attention_case_with_dims(len, 16, 2, 256)
}

fn run_attention_case_with_dims(
    len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let kv_dim = kv_heads * head_dim;
    let mut kv = state.full_attention(len, q_heads, kv_heads, head_dim, false)?;
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

/// Même différentiel que `run_attention_case`, mais avec KV résident bf16. L'oracle
/// CPU lit des K/V arrondis bf16, donc l'écart toléré couvre seulement l'ordre de
/// réduction GPU/CPU et le fast-math de l'exponentielle.
fn run_attention_case_bf16(len: usize) -> Result<()> {
    run_attention_case_bf16_with_dims(len, 16, 2, 256)
}

fn run_attention_case_bf16_with_dims(
    len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let kv_dim = kv_heads * head_dim;
    let mut kv = state.full_attention_bf16_for_test(len, q_heads, kv_heads, head_dim)?;
    let keys: Vec<f32> = (0..len * kv_dim).map(pseudo).collect();
    let values: Vec<f32> = (0..len * kv_dim).map(|i| pseudo(i + 7)).collect();
    kv.seed(&keys, &values, len)?;

    let q_dim = q_heads * head_dim;
    let q_data: Vec<f32> = (0..q_dim).map(|i| pseudo(i + 99)).collect();

    let gpu = kv.attention_decode(&q_data)?;
    let keys_bf16: Vec<f32> = keys.iter().copied().map(bf16_round_to_f32).collect();
    let values_bf16: Vec<f32> = values.iter().copied().map(bf16_round_to_f32).collect();
    let cpu = cpu_attention(
        &q_data,
        &keys_bf16,
        &values_bf16,
        q_heads,
        kv_heads,
        head_dim,
        len,
    );
    assert_eq!(gpu.len(), cpu.len());

    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let delta = (g - c).abs();
        max_abs = max_abs.max(delta);
        sum_abs += delta;
    }
    let mean_abs = sum_abs / gpu.len() as f32;
    assert!(
        max_abs <= 2.0e-3,
        "bf16 len={len}: max_abs={max_abs:e} > 2e-3"
    );
    assert!(
        mean_abs <= 2.0e-4,
        "bf16 len={len}: mean_abs={mean_abs:e} > 2e-4"
    );
    Ok(())
}

/// Microbench AUTONOME du kernel `attention_decode` (SDPA résident decode) aux
/// dimensions réelles 35B full-attn (q16/kv2/hd256), à plusieurs longueurs de KV,
/// SANS prefill (KV synthétique seedé) → itération en secondes, pas 34 min.
/// Mesure ms/appel et **GB/s effectif** (octets K+V UNIQUES lus une fois ;
/// le déficit GQA gonfle le trafic réel mais on veut la lecture unique).
/// Activé par `RETI_RUST_BENCH_SDPA` ; `RETI_RUST_FLASH_SDPA=0` = kernel naïf. Run :
/// `RETI_RUST_BENCH_SDPA=1 cargo test --release -p saragossa attention_decode_bench -- --nocapture`
#[test]
fn attention_decode_bench() -> Result<()> {
    if std::env::var("RETI_RUST_BENCH_SDPA").is_err() {
        return Ok(());
    }
    let Some(state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (16usize, 2usize, 256usize);
    let kv_dim = kv_heads * head_dim;
    let q_dim = q_heads * head_dim;
    let q_data: Vec<f32> = (0..q_dim)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
        .collect();
    let iters: usize = std::env::var("RETI_RUST_BENCH_SDPA_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    for &len in &[1024usize, 4096, 16384, 32768] {
        let mut kv = state.full_attention(len, q_heads, kv_heads, head_dim, false)?;
        let keys: Vec<f32> = (0..len * kv_dim)
            .map(|i| ((i % 101) as f32 - 50.0) * 0.01)
            .collect();
        let values: Vec<f32> = (0..len * kv_dim)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        kv.seed(&keys, &values, len)?;
        for _ in 0..5 {
            let _ = kv.attention_decode(&q_data)?;
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = kv.attention_decode(&q_data)?;
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let bytes = (len * kv_dim * 4 * 2) as f64;
        let gbps = bytes / (ms / 1000.0) / 1.0e9;
        eprintln!("bench_sdpa len={len} ms_per_call={ms:.4} gbps_effective={gbps:.1}");
    }
    Ok(())
}

/// SDPA 2-passes split-K (len ≥ seuil 2048) vs CPU — len=4096 (blocks=32).
#[test]
fn attention_decode_2pass_matches_cpu_len4096() -> Result<()> {
    run_attention_case(4096)
}

/// C1B — SDPA 2-passes split-K avec KV bf16 vs oracle CPU bf16.
#[test]
fn attention_decode_2pass_bf16_matches_cpu_len4096() -> Result<()> {
    run_attention_case_bf16(4096)
}

/// SDPA 2-passes vs CPU — len=8193 (impair, blocks=64, dernier bloc partiel).
#[test]
fn attention_decode_2pass_matches_cpu_len8193() -> Result<()> {
    run_attention_case(8193)
}

/// SDPA 2-passes vs CPU — len=33000 (≈32k, blocks=256), cas cible long-contexte.
#[test]
fn attention_decode_2pass_matches_cpu_len33000() -> Result<()> {
    run_attention_case(33000)
}

/// D-30B — SDPA 2-passes d128 vs CPU — len=4096 (Qwen3-30B : q32/kv4/hd128).
#[test]
fn attention_decode_2pass_d128_matches_cpu_len4096() -> Result<()> {
    run_attention_case_with_dims(4096, 32, 4, 128)
}

/// D-30B — SDPA 2-passes d128 + KV bf16 vs oracle CPU bf16.
#[test]
fn attention_decode_2pass_d128_bf16_matches_cpu_len4096() -> Result<()> {
    run_attention_case_bf16_with_dims(4096, 32, 4, 128)
}

/// D-30B — SDPA 2-passes d128 vs CPU — len=8193.
#[test]
fn attention_decode_2pass_d128_matches_cpu_len8193() -> Result<()> {
    run_attention_case_with_dims(8193, 32, 4, 128)
}

/// D-30B — SDPA 2-passes d128 vs CPU — len=33000.
#[test]
fn attention_decode_2pass_d128_matches_cpu_len33000() -> Result<()> {
    run_attention_case_with_dims(33000, 32, 4, 128)
}

/// D-30B — SDPA 2-passes d128 + KV bf16 vs oracle CPU bf16 — long contexte.
#[test]
fn attention_decode_2pass_d128_bf16_matches_cpu_len33000() -> Result<()> {
    run_attention_case_bf16_with_dims(33000, 32, 4, 128)
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

/// C1B — SDPA single-pass flash avec KV bf16 vs oracle CPU bf16.
#[test]
fn attention_decode_bf16_matches_cpu_len64() -> Result<()> {
    run_attention_case_bf16(64)
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
    let (num_heads, head_dim) = (2, 8);
    // rope_dims 8 = RoPE plein ; 4 = partiel (cas 35B hybride, rope_dims < head_dim).
    for rope_dims in [8_usize, 4] {
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
            let pairs = rope_dims / 2;
            for c in 0..head_dim {
                cpu[start + c] = if c < rope_dims {
                    // Rotate-half : paire (pair, pair+pairs), exposant 2·pair/rope_dims.
                    let pair = if c < pairs { c } else { c - pairs };
                    let exponent = (2 * pair) as f32 / rope_dims as f32;
                    let angle = position as f32 / theta.powf(exponent);
                    let (first, second) = (normed[pair], normed[pair + pairs]);
                    if c < pairs {
                        first * angle.cos() - second * angle.sin()
                    } else {
                        first * angle.sin() + second * angle.cos()
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
        assert!(
            max_abs <= 1.0e-5,
            "rope decode vs CPU (rope_dims={rope_dims}): max_abs={max_abs:e}"
        );
    }
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
    let mut kv = state.full_attention(4, q_heads, kv_heads, head_dim, false)?;
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

/// C1B — `encode_append_kv` f32→bf16 écrit la ligne device au bon offset et
/// arrondit exactement comme le seed CPU bf16.
#[test]
fn encode_append_kv_device_bf16_roundtrip() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let (q_heads, kv_heads, head_dim) = (4, 2, 4);
    let kv_dim = kv_heads * head_dim;
    let mut kv = state.full_attention_bf16_for_test(4, q_heads, kv_heads, head_dim)?;
    let row0: Vec<f32> = (0..kv_dim).map(|i| (i as f32 - 4.0) * 0.125).collect();
    kv.seed(&row0, &row0, 1)?;
    let k_data: Vec<f32> = (0..kv_dim).map(|i| 100.125 + i as f32 * 0.03125).collect();
    let v_data: Vec<f32> = (0..kv_dim).map(|i| -200.25 + i as f32 * 0.0625).collect();
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

    let gpu_keys = read_bf16_as_f32(kv.keys());
    let gpu_values = read_bf16_as_f32(kv.values());
    for i in 0..kv_dim {
        assert_eq!(gpu_keys[i], bf16_round_to_f32(row0[i]), "K row0[{i}]");
        assert_eq!(
            gpu_keys[kv_dim + i],
            bf16_round_to_f32(k_data[i]),
            "K row1[{i}]"
        );
        assert_eq!(gpu_values[i], bf16_round_to_f32(row0[i]), "V row0[{i}]");
        assert_eq!(
            gpu_values[kv_dim + i],
            bf16_round_to_f32(v_data[i]),
            "V row1[{i}]"
        );
    }
    Ok(())
}

/// Oracle BIT-EXACT dé-fusion du gate o_proj (gate E2.2 light-batch) :
/// `attn_gate` (ctx · σ(gate)) suivi du qmv == le kernel FUSIONNÉ
/// `affine_qmv_gated_input_fast` (chemin solo prod de l'o_proj), en bits —
/// l'expression sigmoïde est identique dans les deux kernels, compilés avec les
/// mêmes options fast-math.
#[test]
fn attn_gate_then_qmv_bitwise_matches_fused_gated_o_proj() -> Result<()> {
    let Some(mut state) = try_state()? else {
        return Ok(());
    };
    let executor = MetalExecutor::new()?;
    // Shape o_proj du 35B prod : out=hidden 2048, in=q_dim 4096.
    let (out_dim, in_dim) = (2048_usize, 4096_usize);
    for bits in [4_usize, 8] {
        let values_per_word = 32 / bits;
        let packed_cols = in_dim / values_per_word;
        let groups = in_dim / 64;
        let mut packed = Vec::with_capacity(out_dim * packed_cols);
        for row in 0..out_dim {
            for word in 0..packed_cols {
                let mut value = 0_u32;
                for lane in 0..values_per_word {
                    value |= (((row * 5 + word * 11 + lane) % (1 << bits)) as u32) << (lane * bits);
                }
                packed.push(value);
            }
        }
        let scales = crate::Tensor::from_vec(
            vec![out_dim, groups],
            (0..out_dim * groups)
                .map(|i| 0.003 + 0.000_2 * ((i % 5) as f32))
                .collect(),
        )?;
        let biases = crate::Tensor::from_vec(
            vec![out_dim, groups],
            (0..out_dim * groups)
                .map(|i| -0.02 + 0.001 * ((i % 9) as f32))
                .collect(),
        )?;
        let affine = crate::AffineQuantizedTensor::new(
            &[out_dim, packed_cols],
            packed,
            scales,
            biases,
            64,
            bits,
        )?;
        let weight = executor.resolve_linear_weight_buffers(
            &crate::LinearWeight::AffineQuantized(affine),
            "gate_bits_weight",
        )?;

        let ctx: Vec<f32> = (0..in_dim)
            .map(|i| ((((i * 29) % 97) as f32) - 48.0) / 53.0)
            .collect();
        let gate: Vec<f32> = (0..in_dim)
            .map(|i| ((((i * 41 + 3) % 89) as f32) - 44.0) / 17.0)
            .collect();
        let ctx_buf = state.persistent(in_dim, GpuElement::F32)?;
        write_f32_at(&ctx_buf, 0, &ctx)?;
        let gate_buf = state.persistent(in_dim, GpuElement::F32)?;
        write_f32_at(&gate_buf, 0, &gate)?;
        let gated = state.persistent(in_dim, GpuElement::F32)?;
        let out_fused = state.persistent(out_dim, GpuElement::F32)?;
        let out_split = state.persistent(out_dim, GpuElement::F32)?;

        let command_buffer = state.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let fused = executor.encode_full_attn_o_proj_gated_buffers(
            encoder,
            ctx_buf.buffer(),
            gate_buf.buffer(),
            in_dim,
            &weight,
            out_fused.buffer(),
        )?;
        assert_eq!(
            fused,
            Some(out_dim),
            "le chemin o_proj fusionné doit s'appliquer en {bits} bits"
        );
        state.encode_attn_gate(
            encoder,
            ctx_buf.buffer(),
            gate_buf.buffer(),
            gated.buffer(),
            in_dim,
        )?;
        let split_out = executor.encode_matmul_weight_buffers(
            encoder,
            gated.buffer(),
            1,
            in_dim,
            &weight,
            out_split.buffer(),
            false,
        )?;
        assert_eq!(split_out, out_dim);
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;

        let fused_values = read_f32_buffer(out_fused.buffer(), out_dim)?;
        let split_values = read_f32_buffer(out_split.buffer(), out_dim)?;
        for (idx, (a, b)) in split_values.iter().zip(fused_values.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "o_proj gated {bits} bits: bits divergents à l'index {idx} (split={a:e} fused={b:e})"
            );
        }
    }
    Ok(())
}
