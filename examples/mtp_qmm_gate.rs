//! ÉTAPE 1 (GATE MTP) : microbench V_batch/D — coût d'un verify batché M lignes
//! vs un decode M=1, sur les shapes denses du tronc 27B. Décide si le verify
//! spéculatif est rentable (cible V_batch/D ≤ 1.3 ; >1.5 ⇒ qmm petit-M dédié).
//!
//! Compare, par shape, le temps mur :
//!   (A) M=1  : `affine_qmv_fast_aligned_u4_gs64_f32`  (chemin decode prod)
//!   (B) M=2  : `affine_matmul_rhs_t_u32_f32` batch=2  (chemin batché existant)
//!   (C) M=4  : idem batch=4
//! V_batch/D = t(B|C) / t(A). Régime DRAM froid (rotation 8 jeux de poids), bf16 scales.
//!
//! cargo run --release -p saragossa --example mtp_qmm_gate --features metal

use std::time::Instant;

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

const KERNELS: &str = include_str!("../src/kernels.metal");

/// qmm petit-M (M=2) : aligned fast-qmv étendu à 2 lignes d'activation, **poids lu
/// une seule fois** par mot (réutilisé pour les 2 lignes) → V_batch ≈ D.
const QMM2_SRC: &str = r#"
kernel void affine_qmm2_fast_aligned_u4_gs64_f32(
    device const float* lhs [[buffer(0)]],
    device const uint* packed [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]
) {
    const uint out_dim = dims.x;
    const uint in_dim = dims.y;
    const uint packed_cols = dims.z;
    const uint groups = dims.w;
    const uint rps = 4u;
    const uint simdgroups = 2u;
    const uint vpt = 16u;
    const uint block_size = vpt * 32u;
    const uint row_base = tile.y * (simdgroups * rps) + simd_gid * rps;
    const uint row_bytes = packed_cols * 4u;
    const device uchar* ws = ((const device uchar*)packed) + row_base * row_bytes + simd_lid * 8u;
    const device bfloat* scale_base = scales + row_base * groups + simd_lid / 4u;
    const device bfloat* bias_base = biases + row_base * groups + simd_lid / 4u;
    const device float* x0 = lhs + simd_lid * vpt;
    const device float* x1 = lhs + in_dim + simd_lid * vpt;
    float r0[4] = {0,0,0,0};
    float r1[4] = {0,0,0,0};
    for (uint k = 0u; k < in_dim; k += block_size) {
        float xt0[16]; float s0 = 0.0f;
        float xt1[16]; float s1 = 0.0f;
        for (uint i = 0u; i < vpt; i += 4u) {
            float a0=x0[i],a1=x0[i+1u],a2=x0[i+2u],a3=x0[i+3u];
            s0 += a0+a1+a2+a3; xt0[i]=a0; xt0[i+1u]=a1/16.0f; xt0[i+2u]=a2/256.0f; xt0[i+3u]=a3/4096.0f;
            float c0=x1[i],c1=x1[i+1u],c2=x1[i+2u],c3=x1[i+3u];
            s1 += c0+c1+c2+c3; xt1[i]=c0; xt1[i+1u]=c1/16.0f; xt1[i+2u]=c2/256.0f; xt1[i+3u]=c3/4096.0f;
        }
        for (uint row = 0u; row < rps; ++row) {
            const device ushort* w16 = (const device ushort*)(ws + row * row_bytes);
            const float scale = scale_base[row * groups];
            const float bias = bias_base[row * groups];
            float ac0=0.0f, ac1=0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                const ushort w = w16[i];
                ac0 += xt0[4u*i]*float(w&0x000fu); ac0 += xt0[4u*i+1u]*float(w&0x00f0u); ac0 += xt0[4u*i+2u]*float(w&0x0f00u); ac0 += xt0[4u*i+3u]*float(w&0xf000u);
                ac1 += xt1[4u*i]*float(w&0x000fu); ac1 += xt1[4u*i+1u]*float(w&0x00f0u); ac1 += xt1[4u*i+2u]*float(w&0x0f00u); ac1 += xt1[4u*i+3u]*float(w&0xf000u);
            }
            r0[row] += scale*ac0 + s0*bias;
            r1[row] += scale*ac1 + s1*bias;
        }
        ws += 256u; scale_base += 8u; bias_base += 8u; x0 += block_size; x1 += block_size;
    }
    for (uint row = 0u; row < rps; ++row) {
        const float v0 = simd_sum(r0[row]);
        const float v1 = simd_sum(r1[row]);
        if (simd_lid == 0u) {
            out[row_base + row] = v0;
            out[out_dim + row_base + row] = v1;
        }
    }
}
"#;

const WARMUP: usize = 8;
const CBS: usize = 60;
const REPS: usize = 48;
const ROTATE: usize = 8;

struct Shape {
    name: &'static str,
    out_dim: usize,
    in_dim: usize,
}

fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn main() {
    let device = Device::system_default().expect("device Metal");
    let queue = device.new_command_queue();
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let src = format!("{KERNELS}\n{QMM2_SRC}");
    let lib = device
        .new_library_with_source(&src, &opts)
        .expect("compile kernels.metal + qmm2");
    let pso = |n: &str| {
        let f = lib.get_function(n, None).expect("get_function");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pso")
    };
    let qmv_aligned = pso("affine_qmv_fast_aligned_u4_gs64_f32");
    let rhs_t = pso("affine_matmul_rhs_t_u32_f32");
    let qmm2 = pso("affine_qmm2_fast_aligned_u4_gs64_f32");

    let shared = MTLResourceOptions::StorageModeShared;
    let bits = 4usize;
    let gs = 64usize;

    println!(
        "=== ÉTAPE 1 GATE — V_batch/D (verify batché M vs decode M=1), 27B dense, froid ===\n"
    );

    for sh in [
        Shape {
            name: "down  ",
            out_dim: 5120,
            in_dim: 17408,
        },
        Shape {
            name: "gate/up",
            out_dim: 17408,
            in_dim: 5120,
        },
        Shape {
            name: "in_proj",
            out_dim: 16480,
            in_dim: 5120,
        },
    ] {
        let out_dim = sh.out_dim;
        let in_dim = sh.in_dim;
        let packed_cols = in_dim / (32 / bits);
        let groups = in_dim / gs;

        // Rotation froide de poids (working set ≫ SLC).
        let packed_bufs: Vec<metal::Buffer> = (0..ROTATE)
            .map(|bk| {
                let p: Vec<u32> = (0..out_dim * packed_cols)
                    .map(|i| {
                        (i as u32)
                            .wrapping_mul(2654435761)
                            .wrapping_add(bk as u32 * 0x9E37)
                            | 0x1111_1111
                    })
                    .collect();
                device.new_buffer_with_data(p.as_ptr().cast(), (p.len() * 4) as u64, shared)
            })
            .collect();
        let sc: Vec<u16> = vec![f32_to_bf16(0.0025); out_dim * groups];
        let bs: Vec<u16> = vec![f32_to_bf16(0.0); out_dim * groups];
        let scales: Vec<metal::Buffer> = (0..ROTATE)
            .map(|_| device.new_buffer_with_data(sc.as_ptr().cast(), (sc.len() * 2) as u64, shared))
            .collect();
        let biases: Vec<metal::Buffer> = (0..ROTATE)
            .map(|_| device.new_buffer_with_data(bs.as_ptr().cast(), (bs.len() * 2) as u64, shared))
            .collect();

        // lhs M lignes max (M=4).
        let lhs: Vec<f32> = (0..in_dim * 4)
            .map(|i| ((i % 43) as f32 - 21.0) / 64.0)
            .collect();
        let lhs_buf =
            device.new_buffer_with_data(lhs.as_ptr().cast(), (lhs.len() * 4) as u64, shared);
        let out_buf = device.new_buffer((out_dim * 4 * 4) as u64, shared);

        // (A) M=1 aligned : dims=[out,in,packed_cols,groups], tg=(1,out/8,1) threads=(64,1,1)
        let dims_a: [u32; 4] = [
            out_dim as u32,
            in_dim as u32,
            packed_cols as u32,
            groups as u32,
        ];
        let tg_a = MTLSize::new(1, (out_dim as u64).div_ceil(8), 1);
        let th_a = MTLSize::new(64, 1, 1);

        // (B/C) rhs_t batch=M : dims=[batch,out,in,packed_cols] quant=[gs,bits,groups,0]
        //        tg=(out, batch, 1) threads=(32,1,1)
        let quant: [u32; 4] = [gs as u32, bits as u32, groups as u32, 0];

        let run_aligned = || -> f64 {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&qmv_aligned);
            enc.set_buffer(0, Some(&lhs_buf), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_bytes(5, 16, dims_a.as_ptr().cast());
            for r in 0..REPS {
                let bk = r % ROTATE;
                enc.set_buffer(1, Some(&packed_bufs[bk]), 0);
                enc.set_buffer(2, Some(&scales[bk]), 0);
                enc.set_buffer(3, Some(&biases[bk]), 0);
                enc.dispatch_thread_groups(tg_a, th_a);
            }
            enc.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            (t0.elapsed().as_secs_f64() * 1.0e6) / REPS as f64
        };
        let run_rhs_t = |m: usize| -> f64 {
            let dims_b: [u32; 4] = [m as u32, out_dim as u32, in_dim as u32, packed_cols as u32];
            let tg_b = MTLSize::new(out_dim as u64, m as u64, 1);
            let th_b = MTLSize::new(32, 1, 1);
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&rhs_t);
            enc.set_buffer(0, Some(&lhs_buf), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_bytes(5, 16, dims_b.as_ptr().cast());
            enc.set_bytes(6, 16, quant.as_ptr().cast());
            for r in 0..REPS {
                let bk = r % ROTATE;
                enc.set_buffer(1, Some(&packed_bufs[bk]), 0);
                enc.set_buffer(2, Some(&scales[bk]), 0);
                enc.set_buffer(3, Some(&biases[bk]), 0);
                enc.dispatch_thread_groups(tg_b, th_b);
            }
            enc.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            (t0.elapsed().as_secs_f64() * 1.0e6) / REPS as f64
        };

        // (D) qmm2 dédié M=2 : grid identique au M=1 aligned (1, out/8, 1) threads (64,1,1)
        let run_qmm2 = || -> f64 {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&qmm2);
            enc.set_buffer(0, Some(&lhs_buf), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_bytes(5, 16, dims_a.as_ptr().cast());
            for r in 0..REPS {
                let bk = r % ROTATE;
                enc.set_buffer(1, Some(&packed_bufs[bk]), 0);
                enc.set_buffer(2, Some(&scales[bk]), 0);
                enc.set_buffer(3, Some(&biases[bk]), 0);
                enc.dispatch_thread_groups(tg_a, th_a);
            }
            enc.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            (t0.elapsed().as_secs_f64() * 1.0e6) / REPS as f64
        };

        // CORRECTNESS : qmm2[ligne m] doit être bit-identique à aligned(lhs ligne m).
        {
            let aligned_row = |off_rows: u64| -> Vec<f32> {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&qmv_aligned);
                enc.set_buffer(0, Some(&lhs_buf), off_rows * (in_dim as u64) * 4);
                enc.set_buffer(1, Some(&packed_bufs[0]), 0);
                enc.set_buffer(2, Some(&scales[0]), 0);
                enc.set_buffer(3, Some(&biases[0]), 0);
                enc.set_buffer(4, Some(&out_buf), 0);
                enc.set_bytes(5, 16, dims_a.as_ptr().cast());
                enc.dispatch_thread_groups(tg_a, th_a);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let p = out_buf.contents() as *const f32;
                (0..out_dim).map(|i| unsafe { *p.add(i) }).collect()
            };
            let ref0 = aligned_row(0);
            let ref1 = aligned_row(1);
            // qmm2 → out[0..out_dim]=ligne0, out[out_dim..2*out_dim]=ligne1
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&qmm2);
            enc.set_buffer(0, Some(&lhs_buf), 0);
            enc.set_buffer(1, Some(&packed_bufs[0]), 0);
            enc.set_buffer(2, Some(&scales[0]), 0);
            enc.set_buffer(3, Some(&biases[0]), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_bytes(5, 16, dims_a.as_ptr().cast());
            enc.dispatch_thread_groups(tg_a, th_a);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let p = out_buf.contents() as *const f32;
            let q: Vec<f32> = (0..2 * out_dim).map(|i| unsafe { *p.add(i) }).collect();
            let maxdiff0 = (0..out_dim)
                .map(|i| (q[i] - ref0[i]).abs())
                .fold(0.0f32, f32::max);
            let maxdiff1 = (0..out_dim)
                .map(|i| (q[out_dim + i] - ref1[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                maxdiff0 == 0.0 && maxdiff1 == 0.0,
                "{} qmm2 ≠ aligned : maxdiff ligne0={maxdiff0:e} ligne1={maxdiff1:e}",
                sh.name
            );
        }

        for _ in 0..WARMUP {
            run_aligned();
            run_rhs_t(2);
            run_qmm2();
        }
        let mut a = Vec::new();
        let mut b2 = Vec::new();
        let mut d2 = Vec::new();
        for i in 0..CBS {
            match i % 3 {
                0 => {
                    a.push(run_aligned());
                    b2.push(run_rhs_t(2));
                    d2.push(run_qmm2());
                }
                1 => {
                    b2.push(run_rhs_t(2));
                    d2.push(run_qmm2());
                    a.push(run_aligned());
                }
                _ => {
                    d2.push(run_qmm2());
                    a.push(run_aligned());
                    b2.push(run_rhs_t(2));
                }
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let ma = med(&mut a);
        let m2 = med(&mut b2);
        let md = med(&mut d2);
        println!(
            "{} out={:>5} in={:>5} | A(M=1) {:7.2} µs | B(M=2 rhs_t) {:7.2} (V/D={:.2}) | **D(M=2 qmm2) {:7.2} (V/D={:.2})**",
            sh.name, out_dim, in_dim, ma, m2, m2 / ma, md, md / ma
        );
    }
    println!("\nV_batch/D = t(M)/t(M=1). Cible ≤1.3 (gain spéculatif). Si ~M (≈2.0/≈4.0) ⇒ rhs_t relit les poids/ligne ⇒ qmm petit-M dédié requis.");
}
