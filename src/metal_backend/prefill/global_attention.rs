//! Primitives propres à l'attention globale Gemma 4.

use super::resident::{ResidentPrefillChunk, ResidentPrefillEncodeContext};
use super::*;

const RMS_HEADS_TG_WIDTH: u64 = 256;

impl MetalExecutor {
    #[inline]
    #[expect(
        clippy::too_many_arguments,
        reason = "phase attention résidente: buffers, dimensions et index restent explicites"
    )]
    pub(super) fn encode_resident_attention_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        attention: PrefillAttentionLayer<'_>,
        attention_scratch: PrefillResidentAttentionScratch,
        normed_buffer: &BufferRef,
        layer_cache: &PrefillResidentLayerCacheBuffer,
        chunk: ResidentPrefillChunk,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
        q_dim: usize,
        layer_index: usize,
        kv_only: bool,
    ) -> Result<Buffer> {
        match (attention, attention_scratch, layer_cache) {
            (
                PrefillAttentionLayer::Full {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    gated,
                    ..
                },
                PrefillResidentAttentionScratch::Full(full_scratch),
                PrefillResidentLayerCacheBuffer::Full {
                    key: key_cache,
                    value: value_cache,
                    kv_dim,
                    bf16_mirror,
                    bf16_store,
                    window,
                    ..
                },
            ) => {
                let kv_chunk_len = checked_len(spec.seq, *kv_dim, "resident kv morceau")?;
                // Ligne du buffer où écrire le morceau : la position absolue
                // (RoPE, causalité) reste `chunk.start`. Couche bornée qui
                // déborderait : la fenêtre est d'abord recopiée au début.
                let (kv_row, compaction) = match window {
                    Some(window) => window_kv_row(window, chunk)?,
                    None => (chunk.start, None),
                };
                let kv_cache_offset = checked_len(kv_row, *kv_dim, "resident kv offset")?
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| InferError::Shape("resident kv offset déborde".to_string()))?;
                let PrefillResidentFullAttentionScratch {
                    q_norm: q_norm_buffer,
                    k_norm: k_norm_buffer,
                    q2: q2_buffer,
                    gate: attn_gate_buffer,
                    q: q_buffer,
                    k: k_buffer,
                    v: v_buffer,
                    q_rope: q_rope_buffer,
                    k_rope: k_rope_buffer,
                    context: context_buffer,
                    gated_context: gated_context_buffer,
                    o: o_buffer,
                    sinks: sinks_buffer,
                } = full_scratch;
                self.run_resident_prefill_section(context, "qkv", |encoder, owned| {
                    if let Some((from, keep)) = compaction {
                        for cache in [key_cache, value_cache] {
                            self.encode_kv_copy(
                                encoder,
                                cache,
                                checked_len(from, *kv_dim, "K/V fenêtré compaction")?,
                                cache,
                                0,
                                checked_len(keep, *kv_dim, "K/V fenêtré compaction")?,
                                *bf16_store,
                            )?;
                        }
                    }
                    if kv_only {
                        // Seul le K/V de ces positions sert : Q et la suite sont sautés.
                    } else if gated {
                        let (Some(q2_buffer), Some(attn_gate_buffer)) =
                            (q2_buffer.as_ref(), attn_gate_buffer.as_ref())
                        else {
                            return Err(InferError::Dimension(format!(
                                "prefill résident full gated scratch incomplet couche {layer_index}"
                            )));
                        };
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            normed_buffer,
                            spec.seq,
                            hidden_dim,
                            q_proj.weight(),
                            q2_buffer,
                        )?;
                        self.encode_split_q_gate_rows(
                            encoder,
                            q2_buffer,
                            &q_buffer,
                            attn_gate_buffer,
                            spec.seq,
                            spec.q_heads,
                            spec.head_dim,
                        )?;
                    } else {
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            normed_buffer,
                            spec.seq,
                            hidden_dim,
                            q_proj.weight(),
                            &q_buffer,
                        )?;
                    }
                    self.encode_matmul_weight(
                        encoder,
                        owned,
                        normed_buffer,
                        spec.seq,
                        hidden_dim,
                        k_proj.weight(),
                        &k_buffer,
                    )?;
                    if spec.k_eq_v {
                        self.encode_copy(
                            encoder,
                            &k_buffer,
                            &v_buffer,
                            checked_len(
                                spec.seq,
                                checked_len(spec.kv_heads, spec.head_dim, "resident alias kv_dim")?,
                                "resident alias kv_len",
                            )?,
                        )?;
                    } else {
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            normed_buffer,
                            spec.seq,
                            hidden_dim,
                            v_proj.weight(),
                            &v_buffer,
                        )?;
                    }
                    if spec.value_norm {
                        self.encode_rms_norm_heads_no_scale(encoder, &v_buffer, &v_buffer, spec)?;
                    }
                    if !kv_only {
                        self.encode_rms_norm_rope_heads(
                            encoder,
                            &q_buffer,
                            &q_norm_buffer,
                            &q_rope_buffer,
                            spec,
                            spec.q_heads,
                            chunk.start,
                        )?;
                    }
                    self.encode_rms_norm_rope_heads(
                        encoder,
                        &k_buffer,
                        &k_norm_buffer,
                        &k_rope_buffer,
                        spec,
                        spec.kv_heads,
                        chunk.start,
                    )?;
                    if *bf16_store {
                        // Même `bfloat(x)` que la conversion du NAX ou du
                        // miroir : K/V lus à l'identique.
                        let kv_start = checked_len(kv_row, *kv_dim, "resident kv bf16")?;
                        for (chunk_kv, cache) in
                            [(&k_rope_buffer, key_cache), (&v_buffer, value_cache)]
                        {
                            self.encode_f32_to_bf16_offsets(
                                encoder,
                                chunk_kv,
                                0,
                                cache,
                                kv_start,
                                kv_chunk_len,
                            )?;
                        }
                        return Ok(());
                    }
                    self.encode_copy_with_offsets(
                        encoder,
                        &k_rope_buffer,
                        0,
                        key_cache,
                        kv_cache_offset,
                        kv_chunk_len,
                    )?;
                    self.encode_copy_with_offsets(
                        encoder,
                        &v_buffer,
                        0,
                        value_cache,
                        kv_cache_offset,
                        kv_chunk_len,
                    )
                })?;
                if kv_only {
                    return Ok(o_buffer);
                }
                self.run_resident_prefill_section(
                    context,
                    if spec.window.is_some() {
                        "windowed_attention"
                    } else {
                        "causal_attention"
                    },
                    |encoder, _owned| {
                        if let Some(sinks_buffer) = sinks_buffer.as_ref() {
                            // Sinks (flint) : K/V lu dans son format de
                            // stockage, f32 ou bf16. Morceau ou préfixe : les
                            // requêtes finissent à `chunk.start + chunk.rows`.
                            return self.encode_attention_prefill_sink(
                                encoder,
                                &q_rope_buffer,
                                key_cache,
                                value_cache,
                                &context_buffer,
                                sinks_buffer,
                                spec,
                                kv_row + chunk.rows,
                                *bf16_store,
                            );
                        }
                        if chunk.rows < chunk.total {
                            let kv_rows = chunk.start + chunk.rows;
                            let bf16_kv = match bf16_mirror {
                                _ if *bf16_store => {
                                    Some([key_cache.as_ref(), value_cache.as_ref()])
                                }
                                Some(mirror) => {
                                    // Préfixe et lignes des morceaux `kv_only`
                                    // compris : tout ce qui n'est pas encore converti.
                                    let from =
                                        checked_len(mirror.rows.get(), *kv_dim, "miroir kv")?;
                                    let to = checked_len(kv_rows, *kv_dim, "miroir kv")?;
                                    for (f32_cache, bf16_cache) in
                                        [(key_cache, &mirror.key), (value_cache, &mirror.value)]
                                    {
                                        self.encode_f32_to_bf16_at(
                                            encoder,
                                            f32_cache,
                                            bf16_cache,
                                            from,
                                            to - from,
                                        )?;
                                    }
                                    mirror.rows.set(kv_rows);
                                    Some([mirror.key.as_ref(), mirror.value.as_ref()])
                                }
                                None => None,
                            };
                            self.encode_causal_attention_prefill_chunk(
                                encoder,
                                &q_rope_buffer,
                                key_cache,
                                value_cache,
                                &context_buffer,
                                spec,
                                kv_rows,
                                chunk.total,
                                bf16_kv,
                            )
                        } else if *bf16_store {
                            // Exclu par l'éligibilité : seul le NAX par
                            // morceaux lit un K/V bf16.
                            Err(InferError::Config(
                                "K/V bf16 hors attention par morceaux".to_string(),
                            ))
                        } else if spec.window.is_some() {
                            self.encode_windowed_attention_prefill(
                                encoder,
                                &q_rope_buffer,
                                key_cache,
                                value_cache,
                                &context_buffer,
                                spec,
                            )
                        } else {
                            self.encode_causal_attention_prefill(
                                encoder,
                                &q_rope_buffer,
                                key_cache,
                                value_cache,
                                &context_buffer,
                                spec,
                            )
                        }
                    },
                )?;
                let context_for_o = if gated {
                    let (Some(attn_gate_buffer), Some(gated_context_buffer)) =
                        (attn_gate_buffer.as_ref(), gated_context_buffer.as_ref())
                    else {
                        return Err(InferError::Dimension(format!(
                            "prefill résident full gated output scratch incomplet couche {layer_index}"
                        )));
                    };
                    self.run_resident_prefill_section(context, "attn_gate", |encoder, _owned| {
                        self.encode_attn_gate_rows(
                            encoder,
                            &context_buffer,
                            attn_gate_buffer,
                            gated_context_buffer,
                            checked_len(spec.seq, q_dim, "resident gated context")?,
                        )
                    })?;
                    gated_context_buffer
                } else {
                    &context_buffer
                };
                self.run_resident_prefill_section(context, "o_proj", |encoder, owned| {
                    self.encode_matmul_weight(
                        encoder,
                        owned,
                        context_for_o,
                        spec.seq,
                        q_dim,
                        o_proj.weight(),
                        &o_buffer,
                    )?;
                    Ok(())
                })?;
                Ok(o_buffer)
            }
            (
                PrefillAttentionLayer::Linear {
                    weights,
                    spec: linear_spec,
                    dims,
                },
                PrefillResidentAttentionScratch::Linear(PrefillResidentLinearAttentionScratch {
                    output,
                }),
                PrefillResidentLayerCacheBuffer::Linear { state },
            ) => {
                self.run_resident_prefill_section(
                    context,
                    "linear_attention",
                    |encoder, owned| {
                        self.encode_linear_attn_batch_resident(
                            encoder,
                            owned,
                            normed_buffer,
                            &output,
                            spec.seq,
                            weights,
                            state,
                            linear_spec,
                            dims,
                        )
                    },
                )?;
                Ok(output)
            }
            _ => Err(InferError::Dimension(format!(
                "prefill résident scratch attention incohérent couche {layer_index}"
            ))),
        }
    }

    pub(super) fn encode_rms_norm_heads_no_scale(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
    ) -> Result<()> {
        self.encode_rms_norm_heads_no_scale_rows(
            encoder,
            input_buffer,
            output_buffer,
            spec.seq,
            spec.kv_heads,
            spec.head_dim,
            spec.eps,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "normalisation V par tête: encoder, buffers et dimensions explicites"
    )]
    pub(crate) fn encode_rms_norm_heads_no_scale_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        output_buffer: &BufferRef,
        rows: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let dims = [
            checked_u32(rows, "value norm rows")?,
            checked_u32(heads, "value norm heads")?,
            checked_u32(head_dim, "value norm head_dim")?,
        ];
        encoder.set_compute_pipeline_state(&self.rms_norm_heads_no_scale_f32);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        set_u32_bytes(encoder, 2, &dims, "value_norm_dims")?;
        set_f32_bytes(encoder, 3, &[eps], "value_norm_eps")?;
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                checked_nsuint(heads, "value norm heads")?,
                checked_nsuint(rows, "value norm rows")?,
                1,
            ),
            MTLSize::new(RMS_HEADS_TG_WIDTH, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }
}

/// Ligne du buffer borné où écrire `chunk`, et la compaction à encoder avant
/// (`(première ligne de la fenêtre dans le buffer, lignes gardées)`) si le
/// morceau déborderait. Le kernel à sinks relit alors exactement les mêmes
/// lignes : sa fenêtre `[abs + 1 - keep, abs]`, ramenée au buffer, commence
/// à `rel + 1 - keep ≥ 0` dès que `rel ≥ keep - 1` lignes précèdent la
/// requête, ce que garantit la queue de `keep` lignes.
fn window_kv_row(
    window: &PrefillWindowKv,
    chunk: ResidentPrefillChunk,
) -> Result<(usize, Option<(usize, usize)>)> {
    let base = window.base.get();
    let row = chunk.start.checked_sub(base).ok_or_else(|| {
        InferError::Shape(format!(
            "K/V fenêtré: morceau {} avant la base {base}",
            chunk.start
        ))
    })?;
    if row + chunk.rows <= window.capacity {
        return Ok((row, None));
    }
    // Débordement : `row > capacity - rows ≥ 2 × keep`, la source
    // `[row - keep, row)` ne recouvre pas la destination `[0, keep)`.
    let from = row - window.keep.min(row);
    if from < window.keep || window.keep + chunk.rows > window.capacity {
        return Err(InferError::Shape(format!(
            "K/V fenêtré: compaction {from}..{row} sur {} lignes gardées, capacité {}",
            window.keep, window.capacity
        )));
    }
    window.base.set(chunk.start - window.keep);
    Ok((window.keep, Some((from, window.keep))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(capacity: usize, base: usize) -> PrefillWindowKv {
        PrefillWindowKv {
            keep: 513,
            capacity,
            base: Cell::new(base),
        }
    }

    fn chunk(start: usize, rows: usize) -> ResidentPrefillChunk {
        ResidentPrefillChunk {
            start,
            rows,
            total: usize::MAX,
        }
    }

    /// Fenêtre flint (513) : lignes d'écriture, compactions disjointes et
    /// base absolue, en morceaux de 4096 (dernier plus court), de 100 (plus
    /// courts que la fenêtre) et après la queue d'un préfixe.
    #[test]
    fn window_kv_row_compacts_disjoint_tail_at_window_513() -> Result<()> {
        let big = window(2 * 513 + 4096, 0);
        assert_eq!(window_kv_row(&big, chunk(0, 4096))?, (0, None));
        assert_eq!(
            window_kv_row(&big, chunk(4096, 4096))?,
            (513, Some((3583, 513)))
        );
        assert_eq!(big.base.get(), 3583);
        assert_eq!(
            window_kv_row(&big, chunk(8192, 1000))?,
            (513, Some((4096, 513)))
        );
        assert_eq!(big.base.get(), 8192 - 513);
        let small = window(2 * 513 + 100, 0);
        let mut compactions = Vec::new();
        for start in (0..3000).step_by(100) {
            let (row, compaction) = window_kv_row(&small, chunk(start, 100))?;
            assert_eq!(start - small.base.get(), row, "début {start}");
            assert!(row + 100 <= small.capacity);
            if let Some((from, keep)) = compaction {
                assert!(from >= keep, "compaction chevauchante à {start}");
                compactions.push(start);
            }
        }
        assert_eq!(compactions.first(), Some(&1100));
        assert!(compactions.len() > 3);
        // Tour 2 : queue de 513 lignes du préfixe, morceau qui tient.
        let tail = window(513 + 300, 10_000 - 513);
        assert_eq!(window_kv_row(&tail, chunk(10_000, 300))?, (513, None));
        assert!(window_kv_row(&tail, chunk(9_000, 1)).is_err());
        Ok(())
    }
}
