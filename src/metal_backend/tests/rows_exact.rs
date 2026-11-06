// Oracle O1 de l'identité greedy : une ligne calculée dans un lot de R lignes
// (vérification spéculative) doit rendre les bits du même calcul mono-ligne
// (decode AR), pour R = 1..9, kernel par kernel, aux formes du 27B dense et du
// 35B MoE (poids synthétiques). Comparaison par `to_bits()` (zéros signés et
// NaN compris). Les écarts connus sont listés dans `ROWS_EXACT_EXCEPTIONS` :
// une exception qui redevient exacte fait échouer le test, pour que la liste
// suive le code. Lancement explicite (`--ignored`) : échoue sans device Metal.
// Voir docs/perf/greedy-identity.md.

/// Écarts connus (préfixe « kernel cas », R minimal, écart relatif max) : le
/// matmul u4 gs64 passe au kernel skinny (MMA bf16 hi+lo) à partir de 5 lignes,
/// et tout bloc qui en contient un hérite de l'écart. Bornes : 1e-5 pour un
/// matmul (mesuré 2,4 à 4,6e-6, x2 et x3) et 5e-5 pour un bloc qui en enchaîne
/// plusieurs (mesuré ≤ 1,7e-5). En mode strict (skinny coupé, mêmes lots),
/// aucune exception ne s'applique.
const ROWS_EXACT_EXCEPTIONS: &[(&str, usize, f32)] = &[
    ("matmul u4", 5, 1.0e-5),
    ("attn QKV u4", 5, 1.0e-5),
    ("linear bloc", 5, 5.0e-5),
    ("linear S1 projection", 5, 1.0e-5),
    ("queue MoE", 5, 5.0e-5),
    ("queue dense", 5, 5.0e-5),
];

thread_local! {
    /// Vrai pendant la passe stricte d'un oracle (skinny coupé).
    static ROWS_EXACT_STRICT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Lance `body` trois fois : réglages par défaut (exceptions connues, bornées),
/// skinny coupé (qmv répliqué sur les mêmes lots), puis vérification greedy
/// stricte (qmv-T à partir de 3 lignes), ces deux dernières en égalité stricte.
fn rows_exact_both_modes(body: fn() -> Result<()>) -> Result<()> {
    /// Rétablit le mode, panique comprise.
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            ROWS_EXACT_STRICT.with(|cell| cell.set(false));
        }
    }
    body()?;
    with_qmm_skinny_min_batch_override(usize::MAX, || {
        let _restore = Restore;
        ROWS_EXACT_STRICT.with(|cell| cell.set(true));
        body()
    })?;
    let _greedy = install_verify_greedy_scope(true);
    with_verify_strict_override(true, || {
        let _restore = Restore;
        ROWS_EXACT_STRICT.with(|cell| cell.set(true));
        body()
    })
}

fn rows_exact_executor() -> Result<MetalExecutor> {
    test_executor()?.ok_or_else(|| {
        InferError::Metal("oracle en lignes : aucun device Metal, test non exécuté".to_string())
    })
}

/// Nombre de sorties dont les bits diffèrent (longueurs vérifiées).
fn rows_exact_ndiff(mono: &[f32], rows: &[f32], label: &str) -> usize {
    assert_eq!(mono.len(), rows.len(), "{label}: longueurs");
    mono.iter()
        .zip(rows)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// Vérifie le lot de `rows` lignes contre les lignes mono de largeur `width`,
/// selon la liste d'exceptions.
fn rows_exact_check(
    kernel: &str,
    case: &str,
    rows: usize,
    width: usize,
    mono: &[f32],
    batch: &[f32],
) {
    rows_exact_check_lot(kernel, case, rows, rows, width, mono, batch);
}

/// Comme [`rows_exact_check`], `lot` (taille du lot vérifié, pour la liste
/// d'exceptions) distinct du nombre de tranches comparées (`slices`) : état
/// final d'un lot comparé d'un seul tenant.
fn rows_exact_check_lot(
    kernel: &str,
    case: &str,
    lot: usize,
    slices: usize,
    width: usize,
    mono: &[f32],
    batch: &[f32],
) {
    let rows = lot;
    let label = format!("{kernel} {case} R={rows}");
    let bad: Vec<(usize, usize)> = (0..slices)
        .map(|r| {
            let range = r * width..(r + 1) * width;
            (
                r,
                rows_exact_ndiff(&mono[range.clone()], &batch[range], &label),
            )
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    let full = format!("{kernel} {case}");
    let strict = ROWS_EXACT_STRICT.with(std::cell::Cell::get);
    let bound = ROWS_EXACT_EXCEPTIONS
        .iter()
        .filter(|(key, from, _)| !strict && full.starts_with(key) && rows >= *from)
        .map(|(_, _, bound)| *bound)
        .reduce(f32::max);
    if let Some(bound) = bound {
        assert!(
            !bad.is_empty(),
            "{label}: exception obsolète, les bits sont égaux (mettre à jour ROWS_EXACT_EXCEPTIONS)"
        );
        let span = slices * width;
        assert!(
            batch[..span].iter().all(|x| x.is_finite()),
            "{label}: sortie non finie"
        );
        let scale = mono[..span]
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()))
            .max(1e-30);
        let err = mono[..span]
            .iter()
            .zip(&batch[..span])
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= bound * scale,
            "{label}: écart {:.2e} relatif au-delà de la borne {bound:e}",
            err / scale
        );
    } else {
        assert!(
            bad.is_empty(),
            "{label}: lignes ≠ mono (ligne, sorties) {bad:?}"
        );
    }
}

fn rows_exact_cb(
    executor: &MetalExecutor,
    encode: impl FnOnce(&ComputeCommandEncoderRef) -> Result<()>,
) -> Result<()> {
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encode(encoder)?;
    encoder.end_encoding();
    commit_and_wait(command_buffer)
}

/// `rows` lignes d'activations distinctes, amplitude `scale`.
fn rows_exact_lhs(rows: usize, dim: usize, variant: usize, scale: f32) -> Vec<f32> {
    (0..rows)
        .flat_map(|r| {
            varied_row(dim, variant + r * 7)
                .into_iter()
                .map(move |x| x * scale)
        })
        .collect()
}

fn rows_exact_u4(out_dim: usize, in_dim: usize, seed: u64) -> Result<Linear> {
    Linear::from_weight(
        LinearWeight::AffineQuantized(test_affine_random_u4(out_dim, in_dim, seed)?),
        None,
    )
}

/// Matmuls résidents de chaque format, R = 1..9, deux graines et deux
/// amplitudes d'activation.
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_matmul_formats() -> Result<()> {
    rows_exact_both_modes(rows_exact_matmul_formats_body)
}

fn rows_exact_matmul_formats_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    let dense = |o: usize, i: usize| -> Result<LinearWeight> {
        Ok(LinearWeight::Dense(Tensor::from_vec(
            vec![o, i],
            (0..o * i)
                .map(|k| ((k as f32) * 0.37).sin() * 0.02)
                .collect(),
        )?))
    };
    let cases: Vec<(&str, &str, LinearWeight)> = vec![
        (
            "matmul u4",
            "35B in_proj 12352x2048",
            LinearWeight::AffineQuantized(test_affine_random_u4(12352, 2048, 7)?),
        ),
        (
            "matmul u4",
            "35B down partagé 2048x512",
            LinearWeight::AffineQuantized(test_affine_random_u4(2048, 512, 9)?),
        ),
        (
            "matmul u4",
            "27B in_proj 16480x5120",
            LinearWeight::AffineQuantized(test_affine_random_u4(16480, 5120, 10)?),
        ),
        (
            "matmul u4",
            "27B kv 1024x5120",
            LinearWeight::AffineQuantized(test_affine_random_u4(1024, 5120, 15)?),
        ),
        (
            "matmul u4",
            "27B down 5120x17408",
            LinearWeight::AffineQuantized(test_affine_random_u4(5120, 17408, 16)?),
        ),
        (
            "matmul u4",
            "sortie large 65536x2048",
            LinearWeight::AffineQuantized(test_affine_random_u4(65536, 2048, 17)?),
        ),
        (
            "matmul u8",
            "35B routeur 256x2048",
            LinearWeight::AffineQuantized(test_affine_varied_u8_group(256, 2048, 64)?),
        ),
        (
            "matmul u8",
            "35B gate partagé 1x2048",
            LinearWeight::AffineQuantized(test_affine_varied_u8_group(1, 2048, 64)?),
        ),
        ("matmul dense", "48x5120", dense(48, 5120)?),
    ];
    for (kernel, case, weight) in &cases {
        let in_dim = match weight {
            LinearWeight::AffineQuantized(w) => w.shape()[1],
            LinearWeight::Dense(t) => t.shape()[1],
        };
        let buffers = executor.resolve_linear_weight_buffers(weight, "rexact_mm")?;
        let out_dim = executor.linear_weight_out_dim(&buffers);
        let run = |rows: usize, lhs: &[f32]| -> Result<Vec<f32>> {
            let lhs_buf = executor.buffer_from_slice(lhs, "rexact_mm_lhs")?;
            let out = executor.uncached_f32_buffer(rows * out_dim, "rexact_mm_out")?;
            rows_exact_cb(&executor, |enc| {
                executor
                    .encode_matmul_weight_buffers(
                        enc, &lhs_buf, rows, in_dim, &buffers, &out, false,
                    )
                    .map(|_| ())
            })?;
            read_f32_buffer(&out, rows * out_dim)
        };
        for (variant, scale) in [(3usize, 1.0f32), (500, 37.0)] {
            let lhs = rows_exact_lhs(9, in_dim, variant, scale);
            let mono = (0..9)
                .map(|r| run(1, &lhs[r * in_dim..(r + 1) * in_dim]))
                .collect::<Result<Vec<_>>>()?
                .concat();
            for rows in 1..=9 {
                let batch = run(rows, &lhs[..rows * in_dim])?;
                rows_exact_check(
                    kernel,
                    &format!("{case} graine {variant}"),
                    rows,
                    out_dim,
                    &mono[..rows * out_dim],
                    &batch,
                );
            }
        }
    }
    Ok(())
}

/// Normes d'entrée : `rms_norm_rows`, `add_rms_norm_rows` et la norme en
/// miroir du prologue fusionné du mono (`encode_rms_norm_rows_matching_solo`)
/// contre le prologue lui-même, projection comprise, à 2048 et 5120.
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_norms_and_prologue() -> Result<()> {
    let executor = rows_exact_executor()?;
    for hidden in [2048usize, 5120] {
        let weight = executor.buffer_from_slice(&varied_row(hidden, 900), "rexact_nw")?;
        let a = rows_exact_lhs(9, hidden, 50, 1.0);
        let b = rows_exact_lhs(9, hidden, 70, 3.0);
        let proj = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(test_affine_random_u4(1024, hidden, 31)?),
            "rexact_np",
        )?;
        let run = |rows: usize, a: &[f32], b: &[f32]| -> Result<[Vec<f32>; 3]> {
            let a_buf = executor.buffer_from_slice(a, "rexact_na")?;
            let b_buf = executor.buffer_from_slice(b, "rexact_nb")?;
            let n1 = executor.uncached_f32_buffer(rows * hidden, "rexact_n1")?;
            let s = executor.uncached_f32_buffer(rows * hidden, "rexact_ns")?;
            let n2 = executor.uncached_f32_buffer(rows * hidden, "rexact_n2")?;
            rows_exact_cb(&executor, |enc| {
                executor.encode_rms_norm_rows(enc, &a_buf, &weight, &n1, rows, hidden, 1e-6)?;
                executor.encode_add_rms_norm_rows(
                    enc, &a_buf, &b_buf, &weight, &s, &n2, rows, hidden, 1e-6,
                )
            })?;
            Ok([
                read_f32_buffer(&n1, rows * hidden)?,
                read_f32_buffer(&s, rows * hidden)?,
                read_f32_buffer(&n2, rows * hidden)?,
            ])
        };
        let mono: Vec<[Vec<f32>; 3]> = (0..9)
            .map(|r| {
                run(
                    1,
                    &a[r * hidden..(r + 1) * hidden],
                    &b[r * hidden..(r + 1) * hidden],
                )
            })
            .collect::<Result<_>>()?;
        // Prologue fusionné du mono (ou norme + matmul s'il refuse).
        let mut prologue = Vec::new();
        for r in 0..9 {
            let x = executor.buffer_from_slice(&a[r * hidden..(r + 1) * hidden], "rexact_px")?;
            let o = executor.uncached_f32_buffer(1024, "rexact_po")?;
            let n = executor.uncached_f32_buffer(hidden, "rexact_pn")?;
            rows_exact_cb(&executor, |enc| {
                let fused = executor
                    .encode_matmul_weight_buffers_rms_prologue(
                        enc, &x, &weight, 1e-6, hidden, &proj, &o,
                    )?
                    .is_some();
                if !fused {
                    executor.encode_rms_norm_rows(enc, &x, &weight, &n, 1, hidden, 1e-6)?;
                    executor.encode_matmul_weight_buffers(enc, &n, 1, hidden, &proj, &o, false)?;
                }
                Ok(())
            })?;
            prologue.extend(read_f32_buffer(&o, 1024)?);
        }
        for rows in 1..=9 {
            let out = run(rows, &a[..rows * hidden], &b[..rows * hidden])?;
            for (k, name) in ["rms_norm_rows", "add_rms_norm somme", "add_rms_norm normée"]
                .iter()
                .enumerate()
            {
                let mono_k: Vec<f32> = mono[..rows].iter().flat_map(|m| m[k].clone()).collect();
                rows_exact_check(
                    "normes",
                    &format!("{name} h{hidden}"),
                    rows,
                    hidden,
                    &mono_k,
                    &out[k],
                );
            }
            // Miroir du prologue : seuls les R ≤ 4 évitent le skinny de la projection.
            if rows <= 4 {
                let x = executor.buffer_from_slice(&a[..rows * hidden], "rexact_mx")?;
                let n = executor.uncached_f32_buffer(rows * hidden, "rexact_mn")?;
                let o = executor.uncached_f32_buffer(rows * 1024, "rexact_mo")?;
                rows_exact_cb(&executor, |enc| {
                    executor.encode_rms_norm_rows_matching_solo(
                        enc, &x, &weight, &n, rows, hidden, 1e-6, &proj, false,
                    )?;
                    executor
                        .encode_matmul_weight_buffers(enc, &n, rows, hidden, &proj, &o, false)
                        .map(|_| ())
                })?;
                rows_exact_check(
                    "normes",
                    &format!("prologue en miroir h{hidden}"),
                    rows,
                    1024,
                    &prologue[..rows * 1024],
                    &read_f32_buffer(&o, rows * 1024)?,
                );
            }
        }
    }
    Ok(())
}

/// Poids et état synthétiques d'une couche linear-attn (16 têtes de clé).
struct RowsExactLinear {
    spec: LinearAttentionStepSpec,
    dims: LinearAttnResidentDims,
    weights: MetalLinearAttnResidentDenseWeights,
    conv_seed: Vec<f32>,
    ssm_seed: Vec<f32>,
}

impl RowsExactLinear {
    fn new(executor: &MetalExecutor, hidden: usize, value_heads: usize) -> Result<Self> {
        let spec = LinearAttentionStepSpec {
            num_key_heads: 16,
            num_value_heads: value_heads,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
            rms_eps: 1e-6,
            l2_qk_sigmoid_gate: false,
        };
        let key_dim = 16 * 128;
        let value_dim = value_heads * 128;
        let conv_dim = 2 * key_dim + value_dim;
        let conv_weight = Tensor::from_vec(
            vec![conv_dim, 4, 1],
            (0..conv_dim * 4)
                .map(|i| ((i as f32) * 0.013).sin() * 0.3)
                .collect(),
        )?;
        let a_log: Vec<f32> = (0..value_heads).map(|h| 0.1 + 0.01 * h as f32).collect();
        let dt_bias: Vec<f32> = (0..value_heads).map(|h| -0.3 + 0.02 * h as f32).collect();
        let norm_weight: Vec<f32> = (0..128).map(|i| 1.0 + 0.003 * i as f32).collect();
        let (qkv, z, b, a, out) = (
            rows_exact_u4(conv_dim, hidden, 21)?,
            rows_exact_u4(value_dim, hidden, 22)?,
            rows_exact_u4(value_heads, hidden, 23)?,
            rows_exact_u4(value_heads, hidden, 24)?,
            rows_exact_u4(hidden, value_dim, 25)?,
        );
        let weights =
            executor.resolve_linear_attn_resident_dense_weights(LinearAttnResidentWeights {
                in_proj_qkv: &qkv,
                in_proj_z: &z,
                in_proj_b: &b,
                in_proj_a: &a,
                out_proj: &out,
                conv_weight: &conv_weight,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm_weight: &norm_weight,
            })?;
        let conv_len = 3 * conv_dim;
        let ssm_len = value_heads * 128 * 128;
        Ok(Self {
            spec,
            dims: LinearAttnResidentDims {
                in_dim: hidden,
                conv_dim,
                value_dim,
                key_dim,
            },
            weights,
            conv_seed: (0..conv_len)
                .map(|i| ((i as f32) * 0.07).sin() * 0.5)
                .collect(),
            ssm_seed: (0..ssm_len)
                .map(|i| ((i as f32) * 0.011).cos() * 0.05)
                .collect(),
        })
    }

    fn state(&self, executor: &MetalExecutor) -> Result<LinearAttentionMetalState> {
        let mut state = None;
        executor.ensure_linear_attention_metal_state(
            &mut state,
            &self.conv_seed,
            &self.ssm_seed,
            self.dims.conv_dim,
            self.conv_seed.len(),
            self.ssm_seed.len(),
            self.spec,
        )?;
        state.ok_or_else(|| InferError::Metal("état linear-attn absent".into()))
    }
}

/// Bloc linear-attn de production : R pas mono contre un passage en lignes,
/// sorties puis états conv et SSM finaux.
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_linear_block() -> Result<()> {
    rows_exact_both_modes(rows_exact_linear_block_body)
}

fn rows_exact_linear_block_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    for (model, hidden, value_heads) in [("35B", 2048usize, 32usize), ("27B", 5120, 48)] {
        let layer = RowsExactLinear::new(&executor, hidden, value_heads)?;
        let norm = executor.buffer_from_slice(&varied_row(hidden, 33), "rexact_la_norm")?;
        let x = rows_exact_lhs(9, hidden, 400, 1.0);
        for rows in 1..=9 {
            let mono_state = layer.state(&executor)?;
            let mut mono = Vec::new();
            for r in 0..rows {
                let input = executor
                    .buffer_from_slice(&x[r * hidden..(r + 1) * hidden], "rexact_la_in1")?;
                let out = executor.uncached_f32_buffer(hidden, "rexact_la_out1")?;
                rows_exact_cb(&executor, |enc| {
                    executor.encode_linear_attn_resident_dense_buffers(
                        enc,
                        &input,
                        Some((&norm, 1e-6)),
                        &out,
                        &layer.weights,
                        &mono_state,
                        layer.spec,
                        layer.dims,
                    )
                })?;
                mono.extend(read_f32_buffer(&out, hidden)?);
            }
            let rows_state = layer.state(&executor)?;
            let input = executor.buffer_from_slice(&x[..rows * hidden], "rexact_la_inr")?;
            let out = executor.uncached_f32_buffer(rows * hidden, "rexact_la_outr")?;
            rows_exact_cb(&executor, |enc| {
                executor.encode_linear_attn_resident_dense_buffers_rows(
                    enc,
                    &input,
                    Some((&norm, 1e-6)),
                    &out,
                    rows,
                    &layer.weights,
                    &rows_state,
                    None,
                    layer.spec,
                    layer.dims,
                )
            })?;
            rows_exact_check(
                "linear bloc",
                &format!("{model} sorties"),
                rows,
                hidden,
                &mono,
                &read_f32_buffer(&out, rows * hidden)?,
            );
            let (conv_len, ssm_len) = (layer.conv_seed.len(), layer.ssm_seed.len());
            rows_exact_check_lot(
                "linear bloc",
                &format!("{model} état conv+SSM"),
                rows,
                1,
                conv_len + ssm_len,
                &[
                    read_f32_buffer(&mono_state.conv, conv_len)?,
                    read_f32_buffer(&mono_state.ssm, ssm_len)?,
                ]
                .concat(),
                &[
                    read_f32_buffer(&rows_state.conv, conv_len)?,
                    read_f32_buffer(&rows_state.ssm, ssm_len)?,
                ]
                .concat(),
            );
        }
    }
    Ok(())
}

/// Étapes du bloc linear-attn, mêmes entrées dans le mono (séquence du decode
/// résident à projection concaténée) et dans le chemin batché : projection,
/// normes q/k et gates, scan gated-delta (y et état), rms-gate.
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_linear_stages() -> Result<()> {
    rows_exact_both_modes(rows_exact_linear_stages_body)
}

fn rows_exact_linear_stages_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    for (model, hidden, vh) in [("35B", 2048usize, 32usize), ("27B", 5120, 48)] {
        let layer = RowsExactLinear::new(&executor, hidden, vh)?;
        let spec = layer.spec;
        let LinearAttnResidentDims {
            conv_dim,
            value_dim,
            key_dim,
            ..
        } = layer.dims;
        let stride = conv_dim + value_dim + 2 * vh;
        let full = layer
            .weights
            .full
            .as_ref()
            .ok_or_else(|| InferError::Config("projection concaténée absente".into()))?;
        let rms_w = executor.buffer_from_slice(&varied_row(hidden, 33), "rexact_st_w")?;
        let x = rows_exact_lhs(9, hidden, 400, 1.0);
        let f4 = |v: usize| (v * 4) as u64;
        for rows in 1..=9 {
            // Projection : prologue fusionné du mono contre norme en miroir + matmul.
            let mut mono_proj = Vec::new();
            for r in 0..rows {
                let xr =
                    executor.buffer_from_slice(&x[r * hidden..(r + 1) * hidden], "rexact_st_xr")?;
                let o = executor.uncached_f32_buffer(stride, "rexact_st_p1")?;
                let n = executor.uncached_f32_buffer(hidden, "rexact_st_n1")?;
                rows_exact_cb(&executor, |enc| {
                    let fused = executor
                        .encode_matmul_weight_buffers_rms_prologue(
                            enc,
                            &xr,
                            &rms_w,
                            1e-6,
                            hidden,
                            &full.in_proj,
                            &o,
                        )?
                        .is_some();
                    if !fused {
                        executor.encode_rms_norm_rows(enc, &xr, &rms_w, &n, 1, hidden, 1e-6)?;
                        executor.encode_matmul_weight_buffers(
                            enc,
                            &n,
                            1,
                            hidden,
                            &full.in_proj,
                            &o,
                            false,
                        )?;
                    }
                    Ok(())
                })?;
                mono_proj.extend(read_f32_buffer(&o, stride)?);
            }
            let xb = executor.buffer_from_slice(&x[..rows * hidden], "rexact_st_x")?;
            let proj = executor.uncached_f32_buffer(rows * stride, "rexact_st_pr")?;
            let normed = executor.uncached_f32_buffer(rows * hidden, "rexact_st_nr")?;
            rows_exact_cb(&executor, |enc| {
                executor.encode_rms_norm_rows_matching_solo(
                    enc,
                    &xb,
                    &rms_w,
                    &normed,
                    rows,
                    hidden,
                    1e-6,
                    &full.in_proj,
                    false,
                )?;
                executor
                    .encode_matmul_weight_buffers(
                        enc,
                        &normed,
                        rows,
                        hidden,
                        &full.in_proj,
                        &proj,
                        false,
                    )
                    .map(|_| ())
            })?;
            rows_exact_check(
                "linear S1 projection",
                model,
                rows,
                stride,
                &mono_proj,
                &read_f32_buffer(&proj, rows * stride)?,
            );
            // Normes q/k, beta, decay (le conv_out des voies q/k n'est écrit que
            // par le mono : seules les sorties consommées sont comparées).
            let sm = layer.state(&executor)?;
            let mut m = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for r in 0..rows {
                let co = executor.uncached_f32_buffer(conv_dim, "rexact_st_co")?;
                let q = executor.uncached_f32_buffer(key_dim, "rexact_st_q")?;
                let k = executor.uncached_f32_buffer(key_dim, "rexact_st_k")?;
                let be = executor.uncached_f32_buffer(vh, "rexact_st_be")?;
                let de = executor.uncached_f32_buffer(vh, "rexact_st_de")?;
                rows_exact_cb(&executor, |enc| {
                    executor.encode_linear_attn_conv_with_offset(
                        enc,
                        &proj,
                        f4(r * stride),
                        &full.conv_weight,
                        &sm.conv,
                        &co,
                        conv_dim,
                        4,
                    )?;
                    executor.encode_linear_attn_norm_gates_with_offsets(
                        enc,
                        &co,
                        &proj,
                        f4(r * stride + conv_dim + value_dim),
                        &proj,
                        f4(r * stride + conv_dim + value_dim + vh),
                        &full.a_log,
                        &full.dt_bias,
                        &q,
                        &k,
                        &be,
                        &de,
                        spec,
                    )
                })?;
                m[0].extend(read_f32_buffer(&co, conv_dim)?[2 * key_dim..].to_vec());
                m[1].extend(read_f32_buffer(&q, key_dim)?);
                m[2].extend(read_f32_buffer(&k, key_dim)?);
                m[3].extend(read_f32_buffer(&be, vh)?);
                m[4].extend(read_f32_buffer(&de, vh)?);
            }
            let sb = layer.state(&executor)?;
            let co = executor.uncached_f32_buffer(rows * conv_dim, "rexact_st_cor")?;
            let q = executor.uncached_f32_buffer(rows * key_dim, "rexact_st_qr")?;
            let k = executor.uncached_f32_buffer(rows * key_dim, "rexact_st_kr")?;
            let be = executor.uncached_f32_buffer(rows * vh, "rexact_st_ber")?;
            let de = executor.uncached_f32_buffer(rows * vh, "rexact_st_der")?;
            let layout = LinearAttnBatchLayout {
                qkv_stride: stride,
                beta_offset: f4(conv_dim + value_dim),
                gate_offset: f4(conv_dim + value_dim + vh),
                gate_stride: stride,
            };
            rows_exact_cb(&executor, |enc| {
                executor.encode_linear_attn_conv_norm_gates_k4_dk128_batch(
                    enc,
                    &proj,
                    &proj,
                    &proj,
                    &full.conv_weight,
                    &sb.conv,
                    &full.a_log,
                    &full.dt_bias,
                    &co,
                    &q,
                    &k,
                    &be,
                    &de,
                    rows,
                    spec,
                    layout,
                    false,
                    None,
                )
            })?;
            let b_conv = read_f32_buffer(&co, rows * conv_dim)?;
            let b_v: Vec<f32> = (0..rows)
                .flat_map(|r| b_conv[r * conv_dim + 2 * key_dim..(r + 1) * conv_dim].to_vec())
                .collect();
            rows_exact_check(
                "linear S2",
                &format!("{model} conv v"),
                rows,
                value_dim,
                &m[0],
                &b_v,
            );
            rows_exact_check(
                "linear S2",
                &format!("{model} q"),
                rows,
                key_dim,
                &m[1],
                &read_f32_buffer(&q, rows * key_dim)?,
            );
            rows_exact_check(
                "linear S2",
                &format!("{model} k"),
                rows,
                key_dim,
                &m[2],
                &read_f32_buffer(&k, rows * key_dim)?,
            );
            rows_exact_check(
                "linear S2",
                &format!("{model} beta"),
                rows,
                vh,
                &m[3],
                &read_f32_buffer(&be, rows * vh)?,
            );
            rows_exact_check(
                "linear S2",
                &format!("{model} decay"),
                rows,
                vh,
                &m[4],
                &read_f32_buffer(&de, rows * vh)?,
            );
            // Scan : entrées = sorties batchées de l'étape précédente.
            let b_q = read_f32_buffer(&q, rows * key_dim)?;
            let b_k = read_f32_buffer(&k, rows * key_dim)?;
            let b_be = read_f32_buffer(&be, rows * vh)?;
            let b_de = read_f32_buffer(&de, rows * vh)?;
            let sm = layer.state(&executor)?;
            let mut m_y = Vec::new();
            for r in 0..rows {
                let c1 = executor
                    .buffer_from_slice(&b_conv[r * conv_dim..(r + 1) * conv_dim], "rexact_st_c1")?;
                let q1 = executor
                    .buffer_from_slice(&b_q[r * key_dim..(r + 1) * key_dim], "rexact_st_q1")?;
                let k1 = executor
                    .buffer_from_slice(&b_k[r * key_dim..(r + 1) * key_dim], "rexact_st_k1")?;
                let be1 =
                    executor.buffer_from_slice(&b_be[r * vh..(r + 1) * vh], "rexact_st_be1")?;
                let de1 =
                    executor.buffer_from_slice(&b_de[r * vh..(r + 1) * vh], "rexact_st_de1")?;
                let y1 = executor.uncached_f32_buffer(value_dim, "rexact_st_y1")?;
                rows_exact_cb(&executor, |enc| {
                    executor.encode_linear_attn_gated_delta(
                        enc,
                        &c1,
                        &q1,
                        &k1,
                        &be1,
                        &de1,
                        &sm.ssm,
                        sm.ssm_bf16,
                        &y1,
                        spec,
                    )
                })?;
                m_y.extend(read_f32_buffer(&y1, value_dim)?);
            }
            let sb = layer.state(&executor)?;
            let y = executor.uncached_f32_buffer(rows * value_dim, "rexact_st_yr")?;
            rows_exact_cb(&executor, |enc| {
                executor.encode_linear_attn_gated_delta_seq_captures_dk128(
                    enc,
                    &co,
                    &q,
                    &k,
                    &be,
                    &de,
                    &sb.ssm,
                    &y,
                    rows,
                    &[],
                    false,
                    spec,
                )
            })?;
            rows_exact_check(
                "linear S3",
                &format!("{model} y"),
                rows,
                value_dim,
                &m_y,
                &read_f32_buffer(&y, rows * value_dim)?,
            );
            let ssm_len = layer.ssm_seed.len();
            rows_exact_check_lot(
                "linear S3",
                &format!("{model} état SSM"),
                rows,
                1,
                ssm_len,
                &read_f32_buffer(&sm.ssm, ssm_len)?,
                &read_f32_buffer(&sb.ssm, ssm_len)?,
            );
            // Rms-gate : y batché et z de la projection en lignes.
            let mut m_g = Vec::new();
            for r in 0..rows {
                let g1 = executor.uncached_f32_buffer(value_dim, "rexact_st_g1")?;
                rows_exact_cb(&executor, |enc| {
                    executor.encode_linear_attn_rms_gate_with_offsets(
                        enc,
                        &y,
                        f4(r * value_dim),
                        &proj,
                        f4(r * stride + conv_dim),
                        &full.norm_weight,
                        &g1,
                        0,
                        spec,
                    )
                })?;
                m_g.extend(read_f32_buffer(&g1, value_dim)?);
            }
            let g = executor.uncached_f32_buffer(rows * value_dim, "rexact_st_gr")?;
            rows_exact_cb(&executor, |enc| {
                executor.encode_linear_attn_rms_gate_batch(
                    enc,
                    &y,
                    &proj,
                    f4(conv_dim),
                    stride,
                    &full.norm_weight,
                    &g,
                    rows,
                    spec,
                )
            })?;
            rows_exact_check(
                "linear S4",
                &format!("{model} rms-gate"),
                rows,
                value_dim,
                &m_g,
                &read_f32_buffer(&g, rows * value_dim)?,
            );
        }
    }
    Ok(())
}

/// Routeur u8 gs64 pseudo-aléatoire (top-8 non dégénéré).
fn rows_exact_router_u8(out_dim: usize, in_dim: usize) -> Result<AffineQuantizedTensor> {
    let mut state = 0x9e37_79b9_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let groups = in_dim / 64;
    let packed = (0..out_dim * in_dim / 4)
        .map(|_| {
            let lanes: Vec<u32> = (0..4).map(|_| (next() * 256.0) as u32 & 255).collect();
            pack_lanes(&lanes, 8)
        })
        .collect();
    let scales: Vec<f32> = (0..out_dim * groups)
        .map(|_| bf16_round((0.0005 + 0.002 * next()) as f32))
        .collect();
    let biases = scales.iter().map(|s| bf16_round(-s * 127.5)).collect();
    AffineQuantizedTensor::new(
        &[out_dim, in_dim / 4],
        packed,
        Tensor::from_vec(vec![out_dim, groups], scales)?,
        Tensor::from_vec(vec![out_dim, groups], biases)?,
        64,
        8,
    )
}

/// Queue MoE shared de vérification contre la queue mono du decode résident
/// (35B : 256 experts u4 512, routeur u8, top-8, partagé avec gate u8).
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_moe_tail() -> Result<()> {
    rows_exact_both_modes(rows_exact_moe_tail_body)
}

fn rows_exact_moe_tail_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    let (hidden, inter, experts_n, top_k) = (2048usize, 512usize, 256usize, 8usize);
    let base: Vec<GatedMlp> = (0..8)
        .map(|e| {
            Ok(GatedMlp::new(
                rows_exact_u4(inter, hidden, 100 + 3 * e)?,
                rows_exact_u4(inter, hidden, 101 + 3 * e)?,
                rows_exact_u4(hidden, inter, 102 + 3 * e)?,
            ))
        })
        .collect::<Result<_>>()?;
    let experts: Vec<GatedMlp> = (0..experts_n).map(|e| base[e % 8].clone()).collect();
    let shared = GatedMlp::new(
        rows_exact_u4(inter, hidden, 200)?,
        rows_exact_u4(inter, hidden, 201)?,
        rows_exact_u4(hidden, inter, 202)?,
    );
    let router = Linear::from_weight(
        LinearWeight::AffineQuantized(rows_exact_router_u8(experts_n, hidden)?),
        None,
    )?;
    let shared_gate = Linear::from_weight(
        LinearWeight::AffineQuantized(test_affine_varied_u8_group(1, hidden, 64)?),
        None,
    )?;
    let weights = executor.resolve_moe_shared_weights(&router, &experts, &shared, &shared_gate)?;
    let x = rows_exact_lhs(9, hidden, 600, 1.0);
    let res = rows_exact_lhs(9, hidden, 800, 1.0);
    let mut mono = Vec::new();
    for r in 0..9 {
        let input =
            executor.buffer_from_slice(&x[r * hidden..(r + 1) * hidden], "rexact_moe_in1")?;
        let residual =
            executor.buffer_from_slice(&res[r * hidden..(r + 1) * hidden], "rexact_moe_res1")?;
        let out = executor.uncached_f32_buffer(hidden, "rexact_moe_out1")?;
        rows_exact_cb(&executor, |enc| {
            executor.encode_moe_shared_buffers(
                enc,
                &mut Vec::new(),
                &input,
                Some(&residual),
                &out,
                hidden,
                &weights,
                top_k,
            )
        })?;
        mono.extend(read_f32_buffer(&out, hidden)?);
    }
    for rows in 1..=9 {
        let input = executor.buffer_from_slice(&x[..rows * hidden], "rexact_moe_inr")?;
        let residual = executor.buffer_from_slice(&res[..rows * hidden], "rexact_moe_resr")?;
        let out = executor.uncached_f32_buffer(rows * hidden, "rexact_moe_outr")?;
        rows_exact_cb(&executor, |enc| {
            executor.encode_moe_shared_buffers_verify_rows(
                enc,
                &mut Vec::new(),
                &input,
                &residual,
                &out,
                rows,
                hidden,
                &weights,
                top_k,
            )
        })?;
        rows_exact_check(
            "queue MoE",
            "35B",
            rows,
            hidden,
            &mono[..rows * hidden],
            &read_f32_buffer(&out, rows * hidden)?,
        );
    }
    Ok(())
}

/// Queue dense du 27B (5120 → 17408 → 5120) : séquence mono du decode
/// résident contre la séquence en lignes, recopiées de
/// `decode_resident/dense_tail.rs` (`encode_dense_tail`, `_rows`).
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_dense_tail() -> Result<()> {
    rows_exact_both_modes(rows_exact_dense_tail_body)
}

fn rows_exact_dense_tail_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    let (hidden, inter) = (5120usize, 17408usize);
    let resolve = |w: AffineQuantizedTensor, label: &'static str| {
        executor.resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(w), label)
    };
    let gate = resolve(test_affine_random_u4(inter, hidden, 300)?, "rexact_dt_gate")?;
    let up = resolve(test_affine_random_u4(inter, hidden, 301)?, "rexact_dt_up")?;
    let down = resolve(test_affine_random_u4(hidden, inter, 302)?, "rexact_dt_down")?;
    let score = executor.buffer_from_slice(&[1.0f32], "rexact_dt_score")?;
    let x = rows_exact_lhs(9, hidden, 1000, 1.0);
    let summed = rows_exact_lhs(9, hidden, 1100, 1.0);
    let mut mono = Vec::new();
    for r in 0..9 {
        let xb = executor.buffer_from_slice(&x[r * hidden..(r + 1) * hidden], "rexact_dt_x1")?;
        let sb =
            executor.buffer_from_slice(&summed[r * hidden..(r + 1) * hidden], "rexact_dt_s1")?;
        let sw = executor.uncached_f32_buffer(inter, "rexact_dt_sw1")?;
        let g = executor.uncached_f32_buffer(inter, "rexact_dt_g1")?;
        let u = executor.uncached_f32_buffer(inter, "rexact_dt_u1")?;
        let d = executor.uncached_f32_buffer(hidden, "rexact_dt_d1")?;
        let o = executor.uncached_f32_buffer(hidden, "rexact_dt_o1")?;
        rows_exact_cb(&executor, |enc| {
            let mut owned = Vec::new();
            if !executor.encode_gate_up_swiglu_fast_buffers(enc, &xb, &gate, &up, &sw, hidden)? {
                executor.encode_matmul_weight_buffers(enc, &xb, 1, hidden, &gate, &g, false)?;
                executor.encode_matmul_weight_buffers(enc, &xb, 1, hidden, &up, &u, false)?;
                executor.encode_swiglu(enc, &mut owned, &g, &u, &sw, inter)?;
            }
            if !executor.encode_qmv_u4_residual_buffers(enc, &sw, &down, &sb, &o, inter, hidden)? {
                executor.encode_matmul_weight_buffers(enc, &sw, 1, inter, &down, &d, false)?;
                executor.encode_weighted_sum_add_topk(
                    enc, &mut owned, &d, &score, &sb, &o, 1, hidden,
                )?;
            }
            Ok(())
        })?;
        mono.extend(read_f32_buffer(&o, hidden)?);
    }
    for rows in 2..=9 {
        let xb = executor.buffer_from_slice(&x[..rows * hidden], "rexact_dt_xr")?;
        let sb = executor.buffer_from_slice(&summed[..rows * hidden], "rexact_dt_sr")?;
        let g = executor.uncached_f32_buffer(rows * inter, "rexact_dt_gr")?;
        let u = executor.uncached_f32_buffer(rows * inter, "rexact_dt_ur")?;
        let sw = executor.uncached_f32_buffer(rows * inter, "rexact_dt_swr")?;
        let d = executor.uncached_f32_buffer(rows * hidden, "rexact_dt_dr")?;
        let o = executor.uncached_f32_buffer(rows * hidden, "rexact_dt_or")?;
        rows_exact_cb(&executor, |enc| {
            let mut owned = Vec::new();
            executor.encode_matmul_weight_buffers(enc, &xb, rows, hidden, &gate, &g, false)?;
            executor.encode_matmul_weight_buffers(enc, &xb, rows, hidden, &up, &u, false)?;
            executor.encode_swiglu(enc, &mut owned, &g, &u, &sw, rows * inter)?;
            executor.encode_matmul_weight_buffers(enc, &sw, rows, inter, &down, &d, false)?;
            executor.encode_add_scaled(enc, &mut owned, &sb, &d, &o, 1.0, rows * hidden)
        })?;
        rows_exact_check(
            "queue dense",
            "27B",
            rows,
            hidden,
            &mono[..rows * hidden],
            &read_f32_buffer(&o, rows * hidden)?,
        );
    }
    Ok(())
}

/// Argmax du lm_head : avec des égalités exactes réparties dans des blocs
/// différents, la tête en lignes et la tête mono fusionnée rendent toutes deux
/// le plus petit indice, pour R = 1..9 (pas d'écart de départage).
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_argmax_ties() -> Result<()> {
    let executor = rows_exact_executor()?;
    let (vocab, hidden) = (4096usize, 2048usize);
    let tied = [300usize, 1000, 3000];
    let groups = hidden / 64;
    let mut packed = Vec::with_capacity(vocab * hidden / 8);
    let mut scales = Vec::with_capacity(vocab * groups);
    for row in 0..vocab {
        let strong = tied.contains(&row);
        for word in 0..hidden / 8 {
            let lanes: Vec<u32> = (0..8)
                .map(|lane| {
                    if strong {
                        15
                    } else {
                        ((row * 7 + word * 3 + lane) % 13) as u32
                    }
                })
                .collect();
            packed.push(pack_lanes(&lanes, 4));
        }
        scales.extend((0..groups).map(|g| {
            bf16_round(if strong {
                0.01
            } else {
                0.002 + 0.0001 * (g % 5) as f32
            })
        }));
    }
    let weight = AffineQuantizedTensor::new(
        &[vocab, hidden / 8],
        packed,
        Tensor::from_vec(vec![vocab, groups], scales)?,
        Tensor::from_vec(vec![vocab, groups], vec![0.0; vocab * groups])?,
        64,
        4,
    )?;
    let lm_head = executor
        .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(weight), "rexact_lm")?;
    // Activations positives : les lignes à q = 15 dominent.
    let x: Vec<f32> = rows_exact_lhs(9, hidden, 5, 1.0)
        .into_iter()
        .map(|v| v.abs() + 0.01)
        .collect();
    let input = executor.buffer_from_slice(&x, "rexact_lm_x")?;
    for rows in 1..=9 {
        let mono_idx = executor.uncached_u32_buffer(rows, "rexact_lm_mono")?;
        let rows_idx = executor.uncached_u32_buffer(rows, "rexact_lm_rows")?;
        rows_exact_cb(&executor, |enc| {
            for r in 0..rows {
                executor.encode_lm_head_argmax_buffers_with_index_offset(
                    enc,
                    &mut Vec::new(),
                    &input,
                    &lm_head,
                    &mono_idx,
                    (r * hidden * 4) as u64,
                    (r * 4) as u64,
                    hidden,
                )?;
            }
            executor.encode_lm_head_argmax_rows_buffers(
                enc, &input, &lm_head, &rows_idx, 0, 0, rows, hidden,
            )
        })?;
        let mono = read_u32_buffer(&mono_idx, rows)?;
        let batch = read_u32_buffer(&rows_idx, rows)?;
        assert!(
            mono.iter().all(|&t| t == 300),
            "R={rows}: argmax mono {mono:?}, attendu 300"
        );
        assert_eq!(batch, mono, "R={rows}: argmax en lignes ≠ mono");
    }
    Ok(())
}

/// QKV concaténé du full-attn : le mono fusionne la norme d'entrée au matmul
/// (`encode_full_attn_qkv_split_rms_buffers`, sorties q, gate et k|v) ; les
/// lignes normalisent par le miroir (`encode_rms_norm_rows_matching_solo`,
/// prédicat `full_attn_qkv_split_rms_applies`) puis projettent. Formes : dense
/// 27B (5120, 24/4 têtes de 256), MoE 35B (2048, 16/2), et u8 gs64 (miroir
/// u8, actif en test).
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_full_attn_qkv_prologue() -> Result<()> {
    rows_exact_both_modes(rows_exact_full_attn_qkv_prologue_body)
}

fn rows_exact_full_attn_qkv_prologue_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    let head_dim = 256usize;
    // Le cas u3 couvre le miroir `rms_norm_simd_u3_rows_f32` ; le cas u8 ne
    // fusionne que sous `SARAGOSSA_RUST_FULL_QKV_SPLIT_RMS_U8` (sinon miroir
    // `rms_norm_rows`, prédicat vérifié ci-dessous).
    let cases: [(&str, usize, usize, usize, usize); 4] = [
        ("attn QKV u4", 5120, 24, 4, 4),
        ("attn QKV u4", 2048, 16, 2, 4),
        ("attn QKV u8", 2048, 16, 2, 8),
        ("attn QKV u3", 2048, 16, 2, 3),
    ];
    for (kernel, hidden, q_heads, kv_heads, bits) in cases {
        let q_gate_dim = 2 * q_heads * head_dim;
        let out_dim = q_gate_dim + 2 * kv_heads * head_dim;
        let weight = match bits {
            8 => test_affine_varied_u8_group(out_dim, hidden, 64)?,
            3 => test_affine_mlx_u3(out_dim, hidden)?,
            _ => test_affine_random_u4(out_dim, hidden, 41)?,
        };
        let proj = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(weight),
            "rexact_qkv_w",
        )?;
        let norm = executor.buffer_from_slice(&varied_row(hidden, 61), "rexact_qkv_n")?;
        let applies = executor.full_attn_qkv_split_rms_applies(&proj, hidden);
        let x = rows_exact_lhs(9, hidden, 900, 1.0);
        // Mono : ligne complète reconstituée (q et gate entrelacés par tête, puis k|v).
        let mut mono = Vec::new();
        for r in 0..9 {
            let xr =
                executor.buffer_from_slice(&x[r * hidden..(r + 1) * hidden], "rexact_qkv_x1")?;
            let full = executor.uncached_f32_buffer(out_dim, "rexact_qkv_full1")?;
            let q = executor.uncached_f32_buffer(q_heads * head_dim, "rexact_qkv_q1")?;
            let gate = executor.uncached_f32_buffer(q_heads * head_dim, "rexact_qkv_g1")?;
            let normed = executor.uncached_f32_buffer(hidden, "rexact_qkv_n1")?;
            let mut fused = false;
            rows_exact_cb(&executor, |enc| {
                fused = executor
                    .encode_full_attn_qkv_split_rms_buffers(
                        enc, &xr, &norm, 1e-6, hidden, &proj, &full, &q, &gate, q_heads, head_dim,
                    )?
                    .is_some();
                if !fused {
                    executor.encode_rms_norm_rows(enc, &xr, &norm, &normed, 1, hidden, 1e-6)?;
                    executor.encode_matmul_weight_buffers(
                        enc, &normed, 1, hidden, &proj, &full, false,
                    )?;
                }
                Ok(())
            })?;
            assert_eq!(fused, applies, "{kernel} h{hidden}: prédicat ≠ encodeur");
            let mut row = read_f32_buffer(&full, out_dim)?;
            if fused {
                let (qv, gv) = (
                    read_f32_buffer(&q, q_heads * head_dim)?,
                    read_f32_buffer(&gate, q_heads * head_dim)?,
                );
                for pair in 0..q_heads {
                    for col in 0..head_dim {
                        row[pair * 2 * head_dim + col] = qv[pair * head_dim + col];
                        row[pair * 2 * head_dim + head_dim + col] = gv[pair * head_dim + col];
                    }
                }
            }
            mono.extend(row);
        }
        for rows in 1..=9 {
            let xb = executor.buffer_from_slice(&x[..rows * hidden], "rexact_qkv_xr")?;
            let normed = executor.uncached_f32_buffer(rows * hidden, "rexact_qkv_nr")?;
            let out = executor.uncached_f32_buffer(rows * out_dim, "rexact_qkv_or")?;
            rows_exact_cb(&executor, |enc| {
                if applies {
                    executor.encode_rms_norm_rows_matching_solo(
                        enc, &xb, &norm, &normed, rows, hidden, 1e-6, &proj, true,
                    )?;
                } else {
                    executor.encode_rms_norm_rows(enc, &xb, &norm, &normed, rows, hidden, 1e-6)?;
                }
                executor
                    .encode_matmul_weight_buffers(enc, &normed, rows, hidden, &proj, &out, false)
                    .map(|_| ())
            })?;
            rows_exact_check(
                kernel,
                &format!("h{hidden} fusionné={applies}"),
                rows,
                out_dim,
                &mono[..rows * out_dim],
                &read_f32_buffer(&out, rows * out_dim)?,
            );
        }
    }
    Ok(())
}

/// Ancre du contrat qmv u4 gs64 sur ses trois variantes :
/// - alignée (`out_dim` multiple de 8, `in_dim` multiple de 512) ;
/// - non alignée, `out_dim` = 61 : le dernier simdgroup ne garde qu'une sortie
///   sur quatre (`row_base + row < out_dim`) ;
/// - `align64`, `in_dim` = 1088 (multiple de 64, pas de 512) : deux blocs pleins
///   puis la queue de 64 valeurs servie par 4 voies, en `out_dim` 64 et 61.
///
/// Trois égalités en bits, par graine :
/// 1. les partiels de chaque voie avant `simd_sum`, relevés sur une copie du
///    kernel de `kernels.metal` dont seule la réduction finale est redirigée,
///    égalent une réplique CPU f32 de la forme canonique (32 voies × 16
///    activations par bloc de 512, `sum` en sous-sommes `((x0 + x1) + x2) + x3`,
///    produits en `mul_add`, `r = scale·accum + (sum·bias + r)` fusionnés) ;
/// 2. ces partiels réduits par le même `simd_sum` égalent la sortie du kernel
///    intact compilé dans la même bibliothèque ;
/// 3. cette sortie égale celle du pipeline de production (`MetalExecutor`),
///    pour les formes que la production sert par ce kernel (`align64` exige
///    `out_dim` multiple de 8 : le cas 61 s'arrête au kernel intact).
///
/// Portée : la preuve vaut pour les compilations exercées (fast-math, ce
/// compilateur et ce GPU). `contract(off)` ne coupe pas la réassociation
/// fast-math de la somme ; c'est ce test qui garde l'ordre, pas le pragma.
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_qmv_contract_cpu_oracle() -> Result<()> {
    let executor = rows_exact_executor()?;
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels.metal"))
        .map_err(|error| InferError::Config(format!("kernels.metal: {error}")))?;
    let kernel_text = |name: &str| -> Result<String> {
        let head = format!("kernel void {name}(");
        let start = source
            .find(&head)
            .ok_or_else(|| InferError::Config(format!("{name} absent")))?;
        let end = start
            + source[start..]
                .find("\n}\n")
                .ok_or_else(|| InferError::Config(format!("fin de {name}")))?
            + 3;
        Ok(source[start..end].to_string())
    };
    let partial_store =
        "            out[(tile.x * out_dim + row_base + row) * 32u + simd_lid] = result[row];\n";
    let masked = "        if (simd_lid == 0u && row_base + row < out_dim) {\n";
    let kernels = [
        (
            "affine_qmv_fast_aligned_u4_gs64_f32",
            "        if (simd_lid == 0u) {\n",
        ),
        ("affine_qmv_fast_u4_gs64_f32", masked),
        ("affine_qmv_fast_u4_gs64_align64_f32", masked),
    ];
    // (kernel, out_dim, in_dim, comparé à la production)
    let cases = [
        ("affine_qmv_fast_aligned_u4_gs64_f32", 64usize, 2048usize, true),
        ("affine_qmv_fast_u4_gs64_f32", 61, 2048, true),
        ("affine_qmv_fast_u4_gs64_align64_f32", 64, 1088, true),
        ("affine_qmv_fast_u4_gs64_align64_f32", 61, 1088, false),
    ];
    let mut library_source = String::from(
        "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n\
         kernel void anchor_reduce(device const float* p [[buffer(0)]], device float* out [[buffer(1)]],\n\
             uint gid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_simdgroup]]) {\n\
             const float r = simd_sum(p[gid * 32u + lid]);\n\
             if (lid == 0u) { out[gid] = r; }\n\
         }\n",
    );
    for (name, guard) in kernels {
        let intact = kernel_text(name)?;
        let reduce = format!(
            "        const float reduced = simd_sum(result[row]);\n{guard}            out[tile.x * out_dim + row_base + row] = reduced;\n        }}\n"
        );
        let instrumented = intact
            .replace(
                &reduce,
                &format!("        if (row_base + row < out_dim) {{\n{partial_store}        }}\n"),
            )
            .replace(&format!("kernel void {name}("), &format!("kernel void {name}_partials("));
        assert!(
            instrumented.contains(partial_store),
            "{name} : réduction finale introuvable"
        );
        library_source.push_str(&intact);
        library_source.push_str(&instrumented);
    }
    let library = {
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(true);
        executor
            .device
            .new_library_with_source(&library_source, &options)
            .map_err(InferError::Metal)?
    };
    let pipeline = |name: &str| -> Result<ComputePipelineState> {
        let function = library.get_function(name, None).map_err(InferError::Metal)?;
        executor
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(InferError::Metal)
    };
    let reduce_pipeline = pipeline("anchor_reduce")?;
    for (name, out_dim, in_dim, compare_production) in cases {
        let intact_pipeline = pipeline(name)?;
        let partial_pipeline = pipeline(&format!("{name}_partials"))?;
        let weight = test_affine_random_u4(out_dim, in_dim, 77)?;
        let groups = in_dim / 64;
        let packed = weight.packed_data().to_vec();
        let scales = weight.scales_f32().into_owned();
        let biases = weight.biases_f32().into_owned();
        let buffers = executor
            .resolve_linear_weight_buffers(&LinearWeight::AffineQuantized(weight), "rexact_anchor_w")?;
        let MetalLinearWeightBuffers::AffineQuantized {
            packed: packed_buf,
            scales: scales_buf,
            biases: biases_buf,
            packed_cols,
            ..
        } = &buffers
        else {
            return Err(InferError::Config("poids affine attendu".into()));
        };
        let dims = [out_dim as u32, in_dim as u32, *packed_cols as u32, groups as u32];
        for (salt, scale_x) in [(9usize, 1.0f32), (40, 300.0)] {
            let x: Vec<f32> = varied_row(in_dim, salt)
                .into_iter()
                .map(|v| v * scale_x)
                .collect();
            let x_buf = executor.buffer_from_slice(&x, "rexact_anchor_x")?;
            let partials = executor.uncached_f32_buffer(out_dim * 32, "rexact_anchor_partials")?;
            let reduced = executor.uncached_f32_buffer(out_dim, "rexact_anchor_reduced")?;
            let intact = executor.uncached_f32_buffer(out_dim, "rexact_anchor_intact")?;
            let production = executor.uncached_f32_buffer(out_dim, "rexact_anchor_prod")?;
            rows_exact_cb(&executor, |enc| {
                for (kernel, out) in [(&partial_pipeline, &partials), (&intact_pipeline, &intact)] {
                    enc.set_compute_pipeline_state(kernel);
                    enc.set_buffer(0, Some(&x_buf), 0);
                    enc.set_buffer(1, Some(packed_buf), 0);
                    enc.set_buffer(2, Some(scales_buf), 0);
                    enc.set_buffer(3, Some(biases_buf), 0);
                    enc.set_buffer(4, Some(out), 0);
                    set_u32_bytes(enc, 5, &dims, "rexact_anchor_dims")?;
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, out_dim.div_ceil(8) as u64, 1),
                        MTLSize::new(64, 1, 1),
                    );
                }
                enc.set_compute_pipeline_state(&reduce_pipeline);
                enc.set_buffer(0, Some(&partials), 0);
                enc.set_buffer(1, Some(&reduced), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(out_dim as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
                executor
                    .encode_matmul_weight_buffers(enc, &x_buf, 1, in_dim, &buffers, &production, false)
                    .map(|_| ())
            })?;
            let gpu = read_f32_buffer(&partials, out_dim * 32)?;
            // Réplique CPU : mots u4 lus comme le kernel (octets little-endian).
            let row_words = in_dim / 8;
            let ushort_at = |o: usize, byte: usize| -> u32 {
                let word = packed[o * row_words + byte / 4];
                (word >> ((byte % 4) * 8)) & 0xffff
            };
            let mut bad = 0usize;
            for o in 0..out_dim {
                for lane in 0..32usize {
                    let mut result = 0.0f32;
                    for block in 0..in_dim.div_ceil(512) {
                        let base = block * 512 + lane * 16;
                        // Queue `align64` : seules les voies couvrant des valeurs
                        // réelles accumulent.
                        if base >= in_dim {
                            continue;
                        }
                        let mut sum = 0.0f32;
                        let mut xt = [0.0f32; 16];
                        for i in (0..16).step_by(4) {
                            let (x0, x1, x2, x3) = (
                                x[base + i],
                                x[base + i + 1],
                                x[base + i + 2],
                                x[base + i + 3],
                            );
                            sum += ((x0 + x1) + x2) + x3;
                            xt[i] = x0;
                            xt[i + 1] = x1 / 16.0;
                            xt[i + 2] = x2 / 256.0;
                            xt[i + 3] = x3 / 4096.0;
                        }
                        let group = block * 8 + lane / 4;
                        let (scale, bias) = (scales[o * groups + group], biases[o * groups + group]);
                        let mut accum = 0.0f32;
                        for i in 0..4usize {
                            let word = ushort_at(o, block * 256 + lane * 8 + i * 2);
                            for (j, mask) in
                                [0x000fu32, 0x00f0, 0x0f00, 0xf000].into_iter().enumerate()
                            {
                                accum = xt[4 * i + j].mul_add((word & mask) as f32, accum);
                            }
                        }
                        result = scale.mul_add(accum, sum.mul_add(bias, result));
                    }
                    bad += usize::from(result.to_bits() != gpu[o * 32 + lane].to_bits());
                }
            }
            assert_eq!(
                bad,
                0,
                "{name} {out_dim}x{in_dim} : {bad}/{} partiels ≠ réplique CPU (graine {salt})",
                out_dim * 32
            );
            let reduced = read_f32_buffer(&reduced, out_dim)?;
            let intact = read_f32_buffer(&intact, out_dim)?;
            let production = read_f32_buffer(&production, out_dim)?;
            assert_eq!(
                rows_exact_ndiff(&intact, &reduced, name),
                0,
                "{name} {out_dim}x{in_dim} : simd_sum des partiels ≠ kernel intact (graine {salt})"
            );
            if !compare_production {
                continue;
            }
            assert_eq!(
                rows_exact_ndiff(&intact, &production, name),
                0,
                "{name} {out_dim}x{in_dim} : kernel intact ≠ pipeline de production (graine {salt})"
            );
        }
    }
    Ok(())
}

/// Prologues rms fusionnés au qmv u8 (gs64 et gs128) : le mono fusionne la norme
/// au qmv, les lignes passent par le miroir `rms_norm_simd_rows_f32` puis le qmv
/// u8 en lignes ; égalité stricte en bits, R = 1..9.
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_rms_prologue_u8() -> Result<()> {
    rows_exact_both_modes(rows_exact_rms_prologue_u8_body)
}

fn rows_exact_rms_prologue_u8_body() -> Result<()> {
    let executor = rows_exact_executor()?;
    let hidden = 2048usize;
    for (case, out_dim, group) in [("u8 gs64 512x2048", 512usize, 64usize), ("u8 gs128 512x2048", 512, 128)] {
        let proj = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(test_affine_varied_u8_group(out_dim, hidden, group)?),
            "rexact_rmsu8_w",
        )?;
        let norm = executor.buffer_from_slice(&varied_row(hidden, 67), "rexact_rmsu8_n")?;
        let x = rows_exact_lhs(9, hidden, 950, 1.0);
        let mut mono = Vec::new();
        for r in 0..9 {
            let xr = executor.buffer_from_slice(&x[r * hidden..(r + 1) * hidden], "rexact_rmsu8_x1")?;
            let out = executor.uncached_f32_buffer(out_dim, "rexact_rmsu8_o1")?;
            let mut fused = false;
            rows_exact_cb(&executor, |enc| {
                fused = executor
                    .encode_matmul_weight_buffers_rms_prologue(enc, &xr, &norm, 1e-6, hidden, &proj, &out)?
                    .is_some();
                Ok(())
            })?;
            assert!(fused, "{case} : prologue rms u8 non fusionné");
            mono.extend(read_f32_buffer(&out, out_dim)?);
        }
        for rows in 1..=9 {
            let xb = executor.buffer_from_slice(&x[..rows * hidden], "rexact_rmsu8_xr")?;
            let normed = executor.uncached_f32_buffer(rows * hidden, "rexact_rmsu8_nr")?;
            let out = executor.uncached_f32_buffer(rows * out_dim, "rexact_rmsu8_or")?;
            rows_exact_cb(&executor, |enc| {
                executor.encode_rms_norm_rows_matching_solo(
                    enc, &xb, &norm, &normed, rows, hidden, 1e-6, &proj, false,
                )?;
                executor
                    .encode_matmul_weight_buffers(enc, &normed, rows, hidden, &proj, &out, false)
                    .map(|_| ())
            })?;
            rows_exact_check(
                "prologue rms u8",
                case,
                rows,
                out_dim,
                &mono[..rows * out_dim],
                &read_f32_buffer(&out, rows * out_dim)?,
            );
        }
    }
    Ok(())
}

/// Mode strict, matmuls u4 gs64 isolés : chaque lot de 2 à 16 lignes, puis 17
/// et 32, rend en bits les lignes du qmv mono, sur les trois variantes du
/// qmv-T (device 1024×5120, threadgroup S = 8 en 5120×17408, S = 16 en
/// 65536×2048 ; R = 15 au plafond threadgroup, R = 16 en device seule). Le
/// qmv-T s'engage de 3 à 16 lignes quand le mono prend le qmv rapide
/// (`SARAGOSSA_RUST_QMV_FAST`), jamais sur les paires (qmm2) ni au-delà de 16
/// lignes (qmv répliqué : ni skinny ni GEMM NA en strict).
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_qmvt_engaged_in_strict_mode() -> Result<()> {
    let executor = rows_exact_executor()?;
    let max_rows = 32usize;
    for (out_dim, in_dim) in [(1024usize, 5120usize), (5120, 17408), (65536, 2048)] {
        let buffers = executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(test_affine_random_u4(out_dim, in_dim, 21)?),
            "rexact_qmvt",
        )?;
        let x = rows_exact_lhs(max_rows, in_dim, 3, 1.0);
        let lhs = executor.buffer_from_slice(&x, "rexact_qmvt_lhs")?;
        let out = executor.uncached_f32_buffer(max_rows * out_dim, "rexact_qmvt_out")?;
        let mut mono = Vec::with_capacity(max_rows * out_dim);
        for r in 0..max_rows {
            let xr = executor.buffer_from_slice(&x[r * in_dim..(r + 1) * in_dim], "rexact_qmvt_x1")?;
            let o = executor.uncached_f32_buffer(out_dim, "rexact_qmvt_o1")?;
            rows_exact_cb(&executor, |enc| {
                executor
                    .encode_matmul_weight_buffers(enc, &xr, 1, in_dim, &buffers, &o, false)
                    .map(|_| ())
            })?;
            mono.extend(read_f32_buffer(&o, out_dim)?);
        }
        let fast = super::flags::fast_affine_qmv_enabled(out_dim);
        for rows in (2..=16).chain([17, 32]) {
            let count = || super::matmul_encode::QMVT_DISPATCHES.with(std::cell::Cell::get);
            let before = count();
            let _greedy = install_verify_greedy_scope(true);
            with_verify_strict_override(true, || {
                rows_exact_cb(&executor, |enc| {
                    executor
                        .encode_matmul_weight_buffers(enc, &lhs, rows, in_dim, &buffers, &out, false)
                        .map(|_| ())
                })
            })?;
            let engaged = count() > before;
            assert_eq!(
                engaged,
                fast && (3..=16).contains(&rows),
                "qmv-T {out_dim}x{in_dim} R={rows} : engagé = {engaged}"
            );
            assert_eq!(
                rows_exact_ndiff(
                    &mono[..rows * out_dim],
                    &read_f32_buffer(&out, rows * out_dim)?,
                    "qmv-T"
                ),
                0,
                "strict {out_dim}x{in_dim} R={rows} : lot ≠ mono"
            );
        }
    }
    Ok(())
}

/// Mode strict, matmuls enchaînés dans un même encodeur : deux projections
/// 5120×5120 (même scratch device `xt`/`sum`, réécrit par le second prep après
/// lecture par le premier qmv-T), puis up et down 5120×17408 en threadgroup ;
/// chaque ligne du lot égale la même chaîne en mono, en bits. Le nombre de qmv-T
/// suit `SARAGOSSA_RUST_QMV_FAST` (0, `q`, `o`, `kv` : le mono et le lot gardent
/// alors leur kernel commun).
#[test]
#[ignore = "oracle GPU en bits : lancer avec --ignored"]
fn rows_exact_qmvt_chained_in_one_encoder() -> Result<()> {
    let executor = rows_exact_executor()?;
    let hidden = 5120usize;
    let inter = 17408usize;
    let resolve = |out_dim: usize, in_dim: usize, seed: u64| {
        executor.resolve_linear_weight_buffers(
            &LinearWeight::AffineQuantized(test_affine_random_u4(out_dim, in_dim, seed)?),
            "rexact_qmvt_chain",
        )
    };
    let (first, second, up, down) = (
        resolve(hidden, hidden, 31)?,
        resolve(hidden, hidden, 32)?,
        resolve(inter, hidden, 33)?,
        resolve(hidden, inter, 34)?,
    );
    let chain = |rows: usize, x: &[f32]| -> Result<Vec<f32>> {
        let lhs = executor.buffer_from_slice(x, "rexact_chain_x")?;
        let a = executor.uncached_f32_buffer(rows * hidden, "rexact_chain_a")?;
        let b = executor.uncached_f32_buffer(rows * hidden, "rexact_chain_b")?;
        let c = executor.uncached_f32_buffer(rows * inter, "rexact_chain_c")?;
        let d = executor.uncached_f32_buffer(rows * hidden, "rexact_chain_d")?;
        rows_exact_cb(&executor, |enc| {
            executor.encode_matmul_weight_buffers(enc, &lhs, rows, hidden, &first, &a, false)?;
            executor.encode_matmul_weight_buffers(enc, &a, rows, hidden, &second, &b, false)?;
            executor.encode_matmul_weight_buffers(enc, &b, rows, hidden, &up, &c, false)?;
            executor
                .encode_matmul_weight_buffers(enc, &c, rows, inter, &down, &d, false)
                .map(|_| ())
        })?;
        read_f32_buffer(&d, rows * hidden)
    };
    let x = rows_exact_lhs(16, hidden, 5, 1.0);
    let mut mono = Vec::with_capacity(16 * hidden);
    for r in 0..16 {
        mono.extend(chain(1, &x[r * hidden..(r + 1) * hidden])?);
    }
    for rows in [3usize, 9, 15, 16] {
        let before = super::matmul_encode::QMVT_DISPATCHES.with(std::cell::Cell::get);
        let _greedy = install_verify_greedy_scope(true);
        let got = with_verify_strict_override(true, || chain(rows, &x[..rows * hidden]))?;
        // Le qmv-T suit la garde du qmv rapide du mono (`SARAGOSSA_RUST_QMV_FAST`).
        let expected = [hidden, hidden, inter, hidden]
            .into_iter()
            .filter(|&out_dim| super::flags::fast_affine_qmv_enabled(out_dim))
            .count();
        assert_eq!(
            super::matmul_encode::QMVT_DISPATCHES.with(std::cell::Cell::get) - before,
            expected,
            "chaîne R={rows} : {expected} qmv-T attendus"
        );
        assert_eq!(
            rows_exact_ndiff(&mono[..rows * hidden], &got, "chaîne qmv-T"),
            0,
            "chaîne strict R={rows} : lot ≠ mono"
        );
    }
    Ok(())
}
