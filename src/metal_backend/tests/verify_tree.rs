/// Profondeurs d'un arbre donné par ses parents (ordre topologique).
fn tree_depths(parents: &[u32]) -> Vec<u32> {
    let mut depths = vec![0u32; parents.len()];
    for row in 1..parents.len() {
        depths[row] = depths[parents[row] as usize] + 1;
    }
    depths
}

/// Lignes du chemin racine → `row`.
fn tree_path(parents: &[u32], row: usize) -> Vec<usize> {
    let mut path = vec![row];
    let mut node = row;
    while node != 0 {
        node = parents[node] as usize;
        path.push(node);
    }
    path.reverse();
    path
}

fn gather_rows(data: &[f32], cols: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&row| data[row * cols..(row + 1) * cols].iter().copied())
        .collect()
}

/// Kernels linéaires de la vérification en arbre : pour chaque ligne d'arbres
/// variés (chaîne comprise), conv/normes/gates, sortie `y`, fenêtre conv et état
/// rejoué le long du chemin égalent AU BIT PRÈS les kernels en chaîne appliqués
/// aux seules lignes de ce chemin (kernels validés contre la boucle par pas par
/// `linear_attn_gated_delta_seq_capture_dk128_matches_step_loop_states`). L'état
/// d'entrée reste intact après le scan arbre.
#[test]
fn linear_attn_tree_kernels_match_chain_per_path() -> Result<()> {
    let Some(executor) = test_executor()? else {
        return Ok(());
    };
    let spec = LinearAttentionStepSpec {
        num_key_heads: 2,
        num_value_heads: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        rms_eps: 1.0e-6,
    };
    // Entrées pour 16 lignes (journal plein, B = 15) ; chaque arbre en prend
    // ses premières lignes.
    let max_rows = 16_usize;
    let heads = spec.num_value_heads;
    let key_dim = spec.num_key_heads * spec.key_head_dim;
    let value_dim = heads * spec.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let state_len = heads * spec.value_head_dim * spec.key_head_dim;
    let qkv: Vec<f32> = (0..max_rows * conv_dim)
        .map(|idx| ((idx as f32) * 0.017).sin() * 0.8)
        .collect();
    let beta_in: Vec<f32> = (0..max_rows * heads)
        .map(|idx| ((idx as f32) * 0.37).cos())
        .collect();
    let gate_in: Vec<f32> = (0..max_rows * heads)
        .map(|idx| ((idx as f32) * 0.29).sin() - 0.5)
        .collect();
    let conv_weight: Vec<f32> = (0..conv_dim * 4)
        .map(|idx| ((idx as f32) * 0.011).cos() * 0.5)
        .collect();
    let conv_state0: Vec<f32> = (0..3 * conv_dim)
        .map(|idx| ((idx as f32) * 0.007).sin() * 0.3)
        .collect();
    let a_log: Vec<f32> = (0..heads).map(|h| -1.0 + 0.2 * h as f32).collect();
    let dt_bias: Vec<f32> = (0..heads).map(|h| 0.1 * h as f32).collect();
    let state_seed: Vec<f32> = (0..state_len)
        .map(|idx| ((idx as f32) * 0.009).cos() * 0.1)
        .collect();
    let upload = |data: &[f32], label: &'static str| executor.upload_f32_buffer(data, label);
    let conv_weight_buffer = upload(&conv_weight, "tree_conv_weight")?;
    let conv_state_buffer = upload(&conv_state0, "tree_conv_state0")?;
    let a_log_buffer = upload(&a_log, "tree_a_log")?;
    let dt_bias_buffer = upload(&dt_bias, "tree_dt_bias")?;
    let layout = LinearAttnBatchLayout::contiguous(spec);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();

    // Sorties d'un passage (chaîne ou arbre) sur `count` lignes.
    struct Pass {
        conv_out: Buffer,
        q_norm: Buffer,
        k_norm: Buffer,
        beta: Buffer,
        decay: Buffer,
        y: Buffer,
        ssm: Buffer,
    }
    let new_pass = |count: usize| -> Result<Pass> {
        Ok(Pass {
            conv_out: executor.uncached_f32_buffer(count * conv_dim, "tree_conv_out")?,
            q_norm: executor.uncached_f32_buffer(count * key_dim, "tree_q_norm")?,
            k_norm: executor.uncached_f32_buffer(count * key_dim, "tree_k_norm")?,
            beta: executor.uncached_f32_buffer(count * heads, "tree_beta")?,
            decay: executor.uncached_f32_buffer(count * heads, "tree_decay")?,
            y: executor.uncached_f32_buffer(count * value_dim, "tree_y")?,
            ssm: executor.upload_f32_buffer(&state_seed, "tree_ssm")?,
        })
    };

    for parents in [
        vec![0u32, 0, 1, 2, 3, 4, 5, 6, 7],
        vec![0u32, 0, 0, 1, 1, 2, 3, 3, 5],
        vec![0u32, 0, 1, 2, 1, 4, 0, 6, 3],
        vec![0u32, 0, 0, 1, 1, 2, 3, 3, 5, 4, 6, 8, 9, 7, 12, 13],
        (0..16u32).map(|row| row.saturating_sub(1)).collect(),
    ] {
        let rows = parents.len();
        let depths = tree_depths(&parents);
        let mut tree_data = parents.clone();
        tree_data.extend_from_slice(&depths);
        let qkv_buffer = upload(&qkv, "tree_qkv")?;
        let beta_buffer = upload(&beta_in, "tree_beta_in")?;
        let gate_buffer = upload(&gate_in, "tree_gate_in")?;
        let tree = new_pass(rows)?;
        let journal = executor.uncached_f32_buffer(state_len, "tree_journal")?;
        let conv_captures = (0..rows)
            .map(|_| executor.uncached_f32_buffer(3 * conv_dim, "tree_conv_capture"))
            .collect::<Result<Vec<_>>>()?;
        let command_buffer = executor.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        executor.encode_linear_attn_conv_norm_gates_k4_dk128_batch(
            encoder,
            &qkv_buffer,
            &beta_buffer,
            &gate_buffer,
            &conv_weight_buffer,
            &conv_state_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            &tree.conv_out,
            &tree.q_norm,
            &tree.k_norm,
            &tree.beta,
            &tree.decay,
            rows,
            spec,
            layout,
            false,
            Some(&tree_data),
        )?;
        executor.encode_linear_attn_gated_delta_tree_journal_dk128(
            encoder,
            &tree.conv_out,
            &tree.q_norm,
            &tree.k_norm,
            &tree.beta,
            &tree.decay,
            &tree.ssm,
            &tree.y,
            &journal,
            &parents,
            spec,
        )?;
        for (row, capture) in conv_captures.iter().enumerate() {
            executor.encode_linear_attn_conv_state_tree(
                encoder,
                &qkv_buffer,
                conv_dim,
                &conv_state_buffer,
                capture,
                conv_dim,
                row,
                &tree_data,
            )?;
        }
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        assert_eq!(
            bits(&read_f32_buffer(&tree.ssm, state_len)?),
            bits(&state_seed),
            "scan arbre : état d'entrée modifié"
        );
        let read_row = |buffer: &Buffer, cols: usize, row: usize| -> Result<Vec<f32>> {
            Ok(read_f32_buffer(buffer, (row + 1) * cols)?[row * cols..].to_vec())
        };

        for (row, conv_capture) in conv_captures.iter().enumerate() {
            let path = tree_path(&parents, row);
            let len = path.len();
            let label = format!("arbre {parents:?} ligne {row} chemin {path:?}");
            // Référence : kernels en chaîne sur les seules lignes du chemin.
            let chain = new_pass(len)?;
            let chain_qkv = upload(&gather_rows(&qkv, conv_dim, &path), "chain_qkv")?;
            let chain_beta = upload(&gather_rows(&beta_in, heads, &path), "chain_beta")?;
            let chain_gate = upload(&gather_rows(&gate_in, heads, &path), "chain_gate")?;
            let chain_conv_state = executor.uncached_f32_buffer(3 * conv_dim, "chain_conv")?;
            let replayed = upload(&state_seed, "tree_replayed")?;
            let path_u32 = path.iter().map(|&r| r as u32).collect::<Vec<_>>();
            let command_buffer = executor.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            executor.encode_linear_attn_conv_norm_gates_k4_dk128_batch(
                encoder,
                &chain_qkv,
                &chain_beta,
                &chain_gate,
                &conv_weight_buffer,
                &conv_state_buffer,
                &a_log_buffer,
                &dt_bias_buffer,
                &chain.conv_out,
                &chain.q_norm,
                &chain.k_norm,
                &chain.beta,
                &chain.decay,
                len,
                spec,
                layout,
                false,
                None,
            )?;
            executor.encode_linear_attn_gated_delta_seq_captures_dk128(
                encoder,
                &chain.conv_out,
                &chain.q_norm,
                &chain.k_norm,
                &chain.beta,
                &chain.decay,
                &chain.ssm,
                &chain.y,
                len,
                &[],
                false,
                spec,
            )?;
            executor.encode_linear_attn_conv_state_shift(
                encoder,
                &chain_qkv,
                conv_dim,
                &conv_state_buffer,
                &chain_conv_state,
                conv_dim,
                len,
            )?;
            // Rollback arbre : replay en place le long du chemin.
            executor.encode_linear_attn_gated_delta_replay_dk128(
                encoder,
                &replayed,
                &journal,
                &replayed,
                heads,
                spec.value_head_dim,
                &path_u32,
            )?;
            encoder.end_encoding();
            commit_and_wait(command_buffer)?;
            let last = len - 1;
            for (name, tree_buffer, chain_buffer, cols) in [
                ("conv_out", &tree.conv_out, &chain.conv_out, conv_dim),
                ("q_norm", &tree.q_norm, &chain.q_norm, key_dim),
                ("k_norm", &tree.k_norm, &chain.k_norm, key_dim),
                ("beta", &tree.beta, &chain.beta, heads),
                ("decay", &tree.decay, &chain.decay, heads),
                ("y", &tree.y, &chain.y, value_dim),
            ] {
                assert_eq!(
                    bits(&read_row(tree_buffer, cols, row)?),
                    bits(&read_row(chain_buffer, cols, last)?),
                    "{label}: {name}"
                );
            }
            assert_eq!(
                bits(&read_f32_buffer(conv_capture, 3 * conv_dim)?),
                bits(&read_f32_buffer(&chain_conv_state, 3 * conv_dim)?),
                "{label}: fenêtre conv"
            );
            assert_eq!(
                bits(&read_f32_buffer(&replayed, state_len)?),
                bits(&read_f32_buffer(&chain.ssm, state_len)?),
                "{label}: état rejoué"
            );
        }
    }
    // Garde-fous : replay hors journal, fenêtre conv en place, conv arbre mal formé.
    let buffer = executor.uncached_f32_buffer(state_len, "tree_guard")?;
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    assert!(executor
        .encode_linear_attn_gated_delta_replay_dk128(
            encoder,
            &buffer,
            &buffer,
            &buffer,
            heads,
            spec.value_head_dim,
            &[0, 16],
        )
        .is_err());
    assert!(executor
        .encode_linear_attn_conv_state_tree(
            encoder,
            &buffer,
            conv_dim,
            &buffer,
            &buffer,
            conv_dim,
            0,
            &[0, 0],
        )
        .is_err());
    encoder.end_encoding();
    Ok(())
}
