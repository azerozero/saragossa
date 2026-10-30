//! Orchestration du prefill résident Qwen MoE.

use super::buffers::embedding_indices;
use super::*;
use crate::decode_resident::KvSegment;

pub(super) struct ResidentPrefillEncodeContext<'a> {
    profile_sections: bool,
    section_profile: &'a mut PrefillSectionProfile,
    kernel_timing: &'a mut Option<PrefillKernelTiming>,
    command_buffer: &'a metal::CommandBufferRef,
    fallback_encoder: Option<&'a ComputeCommandEncoderRef>,
    owned_buffers: &'a mut Vec<Buffer>,
}

/// Morceau de lignes `[start, start + rows)` d'un prefill de `total` lignes.
#[derive(Clone, Copy)]
pub(super) struct ResidentPrefillChunk {
    pub(super) start: usize,
    pub(super) rows: usize,
    pub(super) total: usize,
}

/// Lignes K/V réservées au-delà d'un prefill à froid : la queue d'un prompt
/// servi (moins d'un bloc du cache de préfixe serve, 256 par défaut) s'y
/// prolonge en place. Sans elle, le prolongement recopiait tout le KV f32 du
/// préfixe dans un buffer neuf : +30 Go transitoires à 256k sur le 27B.
// ponytail: réserve minimale, ~34 Mo sur le 27B ; une queue plus longue
// (bloc serve agrandi, prolongement sur plusieurs blocs) retombe sur la copie.
const PREFILL_KV_RESERVE_ROWS: usize = 256;
/// Part du prompt ajoutée à la réserve d'un K/V bf16 adoptable : une recopie
/// tous les `seq / 8` tokens prolongés au plus (≈14k à 115k).
const PREFILL_KV_RESERVE_GROWTH_DIVISOR: usize = 8;

thread_local! {
    /// Réserve demandée par l'appelant du prefill en cours sur ce thread, cf.
    /// [`with_prefill_kv_reserve`].
    static PREFILL_KV_RESERVE_HINT: Cell<usize> = const { Cell::new(0) };
}

/// Exécute `run` avec une réserve K/V d'au moins `rows` lignes derrière tout
/// prefill à froid stocké en bf16 de ce thread (un K/V f32 n'est jamais
/// adopté : sa réserve reste minimale) : la queue servie et le decode qui suivent
/// (`max_tokens` + marge spéculative) y tiennent, et l'arène de decode peut
/// adopter les buffers au lieu de les recopier
/// (`FullAttentionMetalState`). Hors de `run`, la réserve revient à 256.
// ponytail: réglage par thread plutôt qu'un paramètre traversant serve,
// décodeur et prefill ; un prefill lancé ailleurs garde la réserve minimale.
pub(crate) fn with_prefill_kv_reserve<R>(rows: usize, run: impl FnOnce() -> R) -> R {
    /// Rétablit la réserve précédente, panique comprise.
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_KV_RESERVE_HINT.with(|hint| hint.set(self.0));
        }
    }
    // Scopes imbriqués : la plus grande réserve demandée l'emporte.
    let _restore = Restore(PREFILL_KV_RESERVE_HINT.with(|hint| hint.replace(hint.get().max(rows))));
    run()
}

#[cfg(test)]
thread_local! {
    /// Morceaux et seuil du stockage bf16 imposés par un test du décodeur,
    /// cf. [`with_prefill_kv_layout_override`].
    static PREFILL_KV_LAYOUT_OVERRIDE: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// Exécute `run` avec, sur ce thread, des morceaux de prefill d'au plus
/// `chunk_tokens` et le K/V stocké en bf16 dès `bf16_min_rows` lignes (si le
/// NAX existe) : un modèle synthétique de quelques centaines de tokens
/// emprunte alors, via le décodeur, les chemins que la production ne prend
/// qu'au-delà de 4096 tokens et du seuil bf16 (`OnceLock` de
/// `runtime_flags`, non réglables dans le processus de test).
#[cfg(test)]
pub(crate) fn with_prefill_kv_layout_override<R>(
    chunk_tokens: usize,
    bf16_min_rows: usize,
    run: impl FnOnce() -> R,
) -> R {
    /// Rétablit le réglage précédent, panique comprise.
    struct Restore(Option<(usize, usize)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_KV_LAYOUT_OVERRIDE.with(|layout| layout.set(self.0));
        }
    }
    let _restore = Restore(
        PREFILL_KV_LAYOUT_OVERRIDE
            .with(|layout| layout.replace(Some((chunk_tokens, bf16_min_rows)))),
    );
    run()
}

/// Réserve des buffers K/V d'un préfixe pour un prolongement en place
/// jusqu'à `seq` lignes : `in_place` désigne ces buffers, le préfixe en est
/// la vue unique depuis leur début, ils ont la place des lignes nouvelles et,
/// **sous le claim**, aucune vue vivante de leur allocation (snapshot plus
/// long, prolongement concurrent déjà publié) ne couvre la réserve
/// ([`InPlaceClaim::try_after`]). Sinon `None` : l'appelant recopie.
///
/// Seul un préfixe bf16 rend une vue entière `[0, seq)` du même buffer
/// (`PrefillKvPlacement::ExtendedInPlace`), prolongeable en place au tour
/// suivant.
// NOTE: restent hors du prolongement multi-tours en place, chacun correct :
// - un préfixe f32 relit ses lignes nouvelles en `Vec` sous le claim (deux
//   segments, le tour suivant recopie) ;
// - l'état après decode (`release_suffix_arena`) ajoute un segment f32
//   possédé : il n'a plus de vue unique et recopie ;
// - les couches glissantes à sinks bornées à leur fenêtre recopient toujours
//   leur queue dans un buffer neuf (`PrefillWindowKv`) ;
// - une réserve épuisée (queue ou decode plus longs que la réserve) recopie
//   dans un buffer neuf.
fn prefix_kv_in_place(
    in_place: Option<&[Buffer; 2]>,
    key: &[KvSegment<'_>],
    value: &[KvSegment<'_>],
    seq: usize,
    (kv_dim, bf16): (usize, bool),
) -> Result<Option<([Buffer; 2], InPlaceReuse)>> {
    let Some(buffers @ [key_buffer, value_buffer]) = in_place else {
        return Ok(None);
    };
    let needed_len = checked_len(seq, kv_dim, "prolongement en place")?;
    let needed = if bf16 {
        byte_len::<u16>(needed_len)?
    } else {
        byte_len::<f32>(needed_len)?
    };
    if key_buffer.length() < needed || value_buffer.length() < needed {
        return Ok(None);
    }
    // Identité de l'objet, valable aussi sans pointeur CPU (stockage privé).
    let same = |view: &BufferRef, buffer: &Buffer| {
        allocation_identity(view) == allocation_identity(buffer)
    };
    let reuse = match (key, value, bf16) {
        ([KvSegment::SharedBf16(key)], [KvSegment::SharedBf16(value)], true)
            if same(key.buffer(), key_buffer) && same(value.buffer(), value_buffer) =>
        {
            InPlaceClaim::try_after([*key, *value]).map(|claim| InPlaceReuse {
                claim,
                bf16_prefix: Some([(*key).clone(), (*value).clone()]),
            })
        }
        ([KvSegment::Shared(key)], [KvSegment::Shared(value)], false)
            if same(key.buffer(), key_buffer) && same(value.buffer(), value_buffer) =>
        {
            InPlaceClaim::try_after([*key, *value]).map(|claim| InPlaceReuse {
                claim,
                bf16_prefix: None,
            })
        }
        _ => None,
    };
    Ok(reuse.map(|reuse| (buffers.clone(), reuse)))
}

/// Copie GPU d'un segment partagé du préfixe vers le K/V d'un prolongement
/// recopié (offsets en octets pour les copies, en éléments pour la
/// conversion).
enum PrefixCopy {
    Bf16 {
        source: Buffer,
        source_offset: u64,
        target: Buffer,
        target_offset: u64,
        len: usize,
    },
    F32 {
        source: Buffer,
        source_offset: u64,
        target: Buffer,
        target_offset: u64,
        len: usize,
    },
    F32ToBf16 {
        source: Buffer,
        source_start: usize,
        target: Buffer,
        target_start: usize,
        len: usize,
    },
}

/// Entrée du prefill une fois sur le GPU : flux plongé complet, ou ids et
/// table pour plonger chaque morceau.
enum ResidentPrefillSource {
    Hidden(Buffer),
    Tokens {
        weights: MetalEmbeddingWeightBuffers,
        indices: Buffer,
        scale: f32,
        recast_bf16: bool,
    },
}

fn f32_byte_offset(len: usize) -> Result<u64> {
    len.checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| InferError::Shape(format!("offset f32 {len} déborde")))
}

#[expect(
    clippy::too_many_arguments,
    reason = "orchestration prefill: couches, dimensions et caches restent explicites"
)]
impl MetalExecutor {
    #[cfg(test)]
    pub(crate) fn qwen_moe_prefill_resident(
        &self,
        input: &Tensor,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
    ) -> Result<(Tensor, Vec<PrefillResidentLayerCache>)> {
        let (output, caches, _) = self.qwen_moe_prefill_resident_with_captures(
            PrefillInput::Hidden(input),
            layers,
            spec,
            &[],
            0,
            false,
        )?;
        Ok((output.into_tensor(spec.hidden_dim)?, caches))
    }

    pub(crate) fn qwen_moe_prefill_resident_with_captures(
        &self,
        input: PrefillInput<'_>,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
        capture_layer_ids: &[usize],
        chunk_tokens: usize,
        last_row_only: bool,
    ) -> Result<(
        PrefillKvRows,
        Vec<PrefillResidentLayerCache>,
        Option<Tensor>,
    )> {
        self.qwen_moe_prefill_resident_from(
            input,
            layers,
            spec,
            capture_layer_ids,
            chunk_tokens,
            None,
            last_row_only,
        )
    }

    /// Indique si un prefill peut prolonger un préfixe : toutes les attentions
    /// complètes doivent passer par une attention à décalage (Steel/NAX, ou
    /// kernels à sinks de flint).
    pub(crate) fn prefill_resident_extend_supported(&self, layers: &[PrefillMoeLayer<'_>]) -> bool {
        layers.iter().all(|layer| match layer.attention {
            // Les deux kernels à sinks (Steel d128, `attention_prefill_sink_f32`)
            // prennent le décalage des requêtes ; la forme est vérifiée à
            // l'encodage.
            PrefillAttentionLayer::Full { sinks: Some(_), .. } => true,
            PrefillAttentionLayer::Full { .. } => {
                self.causal_attention_prefill_chunk_supported(layer.attention_spec)
            }
            PrefillAttentionLayer::Linear { .. } => true,
        })
    }

    /// Prefill résident avec capture du flux résiduel en sortie des couches
    /// `capture_layer_ids` (toutes les positions). Les captures sont copiées sur
    /// le GPU pendant l'encodage, relues une fois à la fin et rendues en
    /// `[seq, n * hidden]`, la disposition attendue par le drafter DFlash.
    /// `None` quand aucune couche n'est demandée. `chunk_tokens` borne la
    /// taille d'un morceau (0 : un seul morceau), cf.
    /// `prefill_resident_chunk_tokens`. Avec `prefix`, `input` ne porte que les
    /// `spec.seq - prefix.rows` positions nouvelles ; sorties, captures et KV
    /// rendus ne couvrent que celles-ci, l'état linéaire du préfixe avance en
    /// place. `last_row_only` rend la seule dernière ligne `[1, hidden]` : la
    /// dernière couche, si elle est en attention complète et non capturée, ne
    /// calcule alors que son K/V sur les morceaux non finaux (ses autres sorties
    /// n'alimentent rien), et seule cette ligne est relue.
    pub(crate) fn qwen_moe_prefill_resident_from(
        &self,
        input: PrefillInput<'_>,
        layers: &[PrefillMoeLayer<'_>],
        spec: PrefillAttentionSpec,
        capture_layer_ids: &[usize],
        chunk_tokens: usize,
        prefix: Option<&PrefillResidentPrefix<'_>>,
        last_row_only: bool,
    ) -> Result<(
        PrefillKvRows,
        Vec<PrefillResidentLayerCache>,
        Option<Tensor>,
    )> {
        let profile_sections = prefill_profile_sections_enabled();
        if profile_sections {
            reset_prefill_f32_to_bf16_shapes();
        }
        let start = prefix.map_or(0, |prefix| prefix.rows);
        let (rows, hidden_dim) = match input {
            PrefillInput::Hidden(hidden) => hidden.as_matrix()?,
            PrefillInput::Tokens { ids, embedding, .. } => {
                (ids.len(), embedding.shape().get(1).copied().unwrap_or(0))
            }
        };
        if rows.checked_add(start) != Some(spec.seq) || hidden_dim != spec.hidden_dim {
            return Err(InferError::Dimension(format!(
                "prefill résident input=[{rows},{hidden_dim}] après {start}, spec seq={} hidden={}",
                spec.seq, spec.hidden_dim
            )));
        }
        if layers.is_empty() || rows == 0 {
            return Err(InferError::Config(
                "prefill résident sans couche ou sans token".to_string(),
            ));
        }
        if prefix.is_some_and(|prefix| prefix.layers.len() != layers.len())
            || (start > 0 && !self.prefill_resident_extend_supported(layers))
        {
            return Err(InferError::Config(
                "prefill résident prolongé hors périmètre".to_string(),
            ));
        }
        #[cfg(test)]
        let layout_override = PREFILL_KV_LAYOUT_OVERRIDE.with(Cell::get);
        #[cfg(test)]
        let chunk_tokens = layout_override.map_or(chunk_tokens, |(chunk, _)| chunk);
        let chunk_rows = self.resident_prefill_chunk_rows(layers, rows, chunk_tokens);
        let chunk_count = rows.div_ceil(chunk_rows);
        let mut kernel_timing = if profile_sections {
            None
        } else {
            PrefillKernelTiming::try_new(&self.device, layers.len() * chunk_count)
        };
        let profile_total_started = profile_sections.then(std::time::Instant::now);
        let mut section_profile = PrefillSectionProfile::default();
        let trace = !profile_sections && trace_prefill_enabled();
        let total_started = trace.then(std::time::Instant::now);
        let hidden_len = checked_len(rows, hidden_dim, "prefill résident hidden")?;
        let chunk_hidden_capacity =
            checked_len(chunk_rows, hidden_dim, "prefill résident hidden morceau")?;
        let source = match input {
            PrefillInput::Hidden(hidden) => ResidentPrefillSource::Hidden(
                self.upload_f32_buffer(hidden.data(), "resident_input")?,
            ),
            PrefillInput::Tokens {
                ids,
                embedding,
                scale,
                recast_bf16,
            } => ResidentPrefillSource::Tokens {
                weights: self.resolve_embedding_weight_buffers(embedding)?,
                indices: self
                    .upload_u32_buffer(&embedding_indices(embedding, ids)?, "resident_input_ids")?,
                scale,
                recast_bf16,
            },
        };
        let hidden_a = self.private_f32_buffer(chunk_hidden_capacity, "resident_hidden_a")?;
        let hidden_b = self.private_f32_buffer(chunk_hidden_capacity, "resident_hidden_b")?;
        let chunk_input =
            if chunk_count > 1 || matches!(source, ResidentPrefillSource::Tokens { .. }) {
                Some(self.private_f32_buffer(chunk_hidden_capacity, "resident_chunk_input")?)
            } else {
                None
            };
        if let Some(bad) = capture_layer_ids.iter().find(|&&id| id >= layers.len()) {
            return Err(InferError::Config(format!(
                "prefill résident capture couche {bad} hors des {} couches",
                layers.len()
            )));
        }
        let capture_buffer = if capture_layer_ids.is_empty() {
            None
        } else {
            let len = checked_len(
                capture_layer_ids.len(),
                hidden_len,
                "prefill résident captures",
            )?;
            Some(self.uncached_f32_buffer(len, "resident_captures")?)
        };
        let final_output = if (chunk_count > 1 && !last_row_only) || private_scratch_enabled() {
            Some(self.uncached_f32_buffer(hidden_len, "resident_final_output")?)
        } else {
            None
        };
        // NOTE: seuls le KV pleine longueur et l'état linéaire survivent aux
        // morceaux ; tout le reste est un scratch dimensionné sur un morceau.
        // Le miroir ne sert qu'à l'attention par morceaux (plusieurs morceaux
        // ou préfixe), et seulement si elle passe par le NAX bf16.
        let chunked = chunk_count > 1 || start > 0;
        let has_sinks = layers.iter().any(|layer| {
            matches!(
                layer.attention,
                PrefillAttentionLayer::Full { sinks: Some(_), .. }
            )
        });
        // Les couches à sinks (flint) lisent le K/V f32 tel quel : le miroir
        // ne sert qu'au NAX, qu'elles ne prennent pas.
        let bf16_mirror = self.prefill_kv_bf16_mirror
            && chunked
            && !has_sinks
            && self.causal_attention_prefill_steel_nax_d256.is_some();
        // Le stockage bf16 vaut pour toutes les couches : il faut un lecteur
        // bf16 à chacune, le NAX d256 sans sinks ; avec sinks, le flag flint (actif par défaut)
        // puis le Steel d128 bf16 pour une couche pleine, le naïf bf16
        // (toujours compilé) suffisant à une fenêtrée. L'arène d'un prompt de
        // cette longueur sera bf16 : sans sinks, le NAX lisait déjà ces
        // octets ; avec, l'arrondi change les K/V lus (dérive mesurée par
        // `synthetic_flint_prefill_kv_bf16_store_drift` et
        // `flint_int4_prefill_kv_bf16_store_drift`).
        let kv_bf16_readable = layers.iter().all(|layer| match layer.attention {
            PrefillAttentionLayer::Full { sinks: Some(_), .. } => {
                self.prefill_kv_bf16_store_sinks
                    && (layer.attention_spec.window.is_some()
                        || self.attention_prefill_steel_sink_d128_kv_bf16.is_some())
            }
            PrefillAttentionLayer::Full { .. } => {
                self.causal_attention_prefill_steel_nax_d256.is_some()
            }
            PrefillAttentionLayer::Linear { .. } => true,
        });
        let bf16_min_rows = self.prefill_kv_bf16_store_min_rows;
        #[cfg(test)]
        let bf16_min_rows = layout_override.map_or(bf16_min_rows, |(_, min_rows)| Some(min_rows));
        let bf16_store = chunked
            && kv_bf16_readable
            && bf16_min_rows.is_some_and(|min_rows| spec.seq >= min_rows);
        // Réserve d'un K/V écrit dans un buffer neuf (prefill à froid, préfixe
        // recopié) : la demande de l'appelant pour un K/V bf16, adoptable par
        // l'arène, plus un huitième du prompt. Un tour de conversation ajoute
        // des milliers de lignes, bien au-delà de `max_tokens` : sans cette
        // marge géométrique (comme un `Vec`), chaque tour recopiait tout le
        // K/V, la conversation existant deux fois au pic (+8 Go à 115k sur le
        // 27B). Mesuré : les lignes jamais écrites n'entrent pas dans
        // `phys_footprint` (pic des tours 2-3 de 35 à 29 Go à 115k), mais elles
        // comptent dans les plafonds d'octets des caches de préfixe, qui
        // lisent `buffer.length()`. Jamais adopté en f32, il garde la réserve
        // minimale.
        let fresh_reserve = if bf16_store {
            PREFILL_KV_RESERVE_HINT
                .with(Cell::get)
                .max(PREFILL_KV_RESERVE_ROWS)
                .saturating_add(spec.seq / PREFILL_KV_RESERVE_GROWTH_DIVISOR)
        } else {
            PREFILL_KV_RESERVE_ROWS
        };
        // Copies GPU des segments partagés du préfixe, encodées avant le
        // premier morceau : une écriture CPU dans le buffer neuf payait une
        // faute de page par 16 Kio.
        let mut prefix_copies = Vec::new();
        let mut layer_cache_buffers = Vec::with_capacity(layers.len());
        for (layer_index, layer) in layers.iter().enumerate() {
            let (_, kv_dim, layer_shape) =
                self.resident_prefill_layer_shape(layer, layer_index, spec, hidden_dim)?;
            let prefix_layer = prefix.map(|prefix| &prefix.layers[layer_index]);
            // Couche glissante à sinks : K/V borné à la fenêtre, sans réserve
            // ni prolongement en place ; le préfixe n'apporte que sa queue.
            let window_keep = match layer.attention {
                PrefillAttentionLayer::Full { sinks: Some(_), .. } if self.prefill_window_kv => {
                    layer.attention_spec.window.filter(|window| *window > 0)
                }
                _ => None,
            };
            let cache = match (window_keep, prefix_layer, &layer_shape.attention) {
                (
                    Some(keep),
                    None | Some(PrefillResidentPrefixLayer::Full { .. }),
                    PrefillResidentAttentionShape::Full { .. },
                ) => {
                    let prefix_rows = start.min(keep);
                    let capacity = keep
                        .checked_mul(2)
                        .and_then(|rows| rows.checked_add(chunk_rows))
                        .ok_or_else(|| InferError::Shape("K/V fenêtré déborde".to_string()))?
                        .min(prefix_rows + rows);
                    let mut cache = self.allocate_prefill_resident_layer_cache(
                        &layer_shape,
                        capacity,
                        kv_dim,
                        false,
                        bf16_store,
                        None,
                    )?;
                    if let PrefillResidentLayerCacheBuffer::Full { window, .. } = &mut cache {
                        *window = Some(PrefillWindowKv {
                            keep,
                            capacity,
                            base: Cell::new(start - prefix_rows),
                        });
                    }
                    cache
                }
                (_, prefix_layer, attention) => match (prefix_layer, attention) {
                    (
                        Some(PrefillResidentPrefixLayer::Linear { state }),
                        PrefillResidentAttentionShape::Linear { .. },
                    ) => PrefillResidentLayerCacheBuffer::Linear {
                        state: (*state).clone(),
                    },
                    (None, _) => self.allocate_prefill_resident_layer_cache(
                        &layer_shape,
                        // Réserve pour la queue d'un prompt servi, prolongée en
                        // place (cf. `prefix_kv_in_place`).
                        spec.seq.saturating_add(fresh_reserve),
                        kv_dim,
                        bf16_mirror,
                        bf16_store,
                        None,
                    )?,
                    (
                        Some(PrefillResidentPrefixLayer::Full {
                            key,
                            value,
                            in_place,
                            base: 0,
                        }),
                        PrefillResidentAttentionShape::Full { .. },
                    ) => {
                        // Réserve tenue (claim) jusqu'à la relecture ou la
                        // publication des lignes nouvelles, en fin de fonction.
                        let (in_place, reuse) = prefix_kv_in_place(
                            in_place.as_ref(),
                            key,
                            value,
                            spec.seq,
                            (kv_dim, bf16_store),
                        )?
                        .unzip();
                        // Préfixe recopié dans un buffer neuf : s'il est déjà tout en
                        // bf16, le buffer prend la réserve d'un prefill à froid et
                        // remplace le préfixe du cache (vue entière, adoptable et
                        // prolongeable en place au tour suivant). Un préfixe avec du
                        // f32 garde ses valeurs : seules les lignes nouvelles repartent.
                        let replaces_prefix = in_place.is_none()
                            && bf16_store
                            && crate::runtime_flags::prefill_kv_shared_enabled()
                            && [key, value].iter().all(|segments| {
                                segments.iter().all(|segment| {
                                    matches!(
                                        segment,
                                        KvSegment::CpuBf16(_) | KvSegment::SharedBf16(_)
                                    )
                                })
                            });
                        let mut cache = self.allocate_prefill_resident_layer_cache(
                            &layer_shape,
                            if replaces_prefix {
                                spec.seq.saturating_add(fresh_reserve)
                            } else {
                                spec.seq
                            },
                            kv_dim,
                            bf16_mirror,
                            bf16_store,
                            in_place,
                        )?;
                        if let PrefillResidentLayerCacheBuffer::Full {
                            replaces_prefix: replaces,
                            in_place,
                            ..
                        } = &mut cache
                        {
                            *replaces = replaces_prefix;
                            *in_place = reuse;
                        }
                        cache
                    }
                    _ => {
                        return Err(InferError::Config(format!(
                            "préfixe résident incohérent couche {layer_index}"
                        )))
                    }
                },
            };
            if let (
                Some(PrefillResidentPrefixLayer::Full {
                    key,
                    value,
                    base: prefix_base,
                    ..
                }),
                PrefillResidentLayerCacheBuffer::Full {
                    key: key_cache,
                    value: value_cache,
                    bf16_store,
                    in_place,
                    window,
                    ..
                },
            ) = (prefix_layer, &cache)
            {
                // Le préfixe stocke les lignes `[prefix_base, start)` ; une
                // couche bornée n'en recopie que la queue `[window.base, start)`.
                let stored = start.checked_sub(*prefix_base).ok_or_else(|| {
                    InferError::Dimension(format!(
                        "préfixe résident couche {layer_index}: base {prefix_base} > {start}"
                    ))
                })?;
                let first = window
                    .as_ref()
                    .map_or(*prefix_base, |window| window.base.get());
                let prefix_len = checked_len(stored, kv_dim, "préfixe résident kv")?;
                let key_len: usize = key.iter().map(|segment| segment.len()).sum();
                let value_len: usize = value.iter().map(|segment| segment.len()).sum();
                if key_len != prefix_len || value_len != prefix_len || first < *prefix_base {
                    return Err(InferError::Dimension(format!(
                        "préfixe résident KV couche {layer_index}: {key_len}/{value_len} attendu {prefix_len}, lignes {first}.. sous la base {prefix_base}"
                    )));
                }
                let skip = checked_len(first - *prefix_base, kv_dim, "préfixe résident kv")?;
                for (buffer, segments) in [(key_cache, key), (value_cache, value)] {
                    // Prolongement en place : le préfixe est déjà au début.
                    if in_place.is_some() {
                        continue;
                    }
                    let mut offset = 0;
                    let mut skip = skip;
                    for segment in segments {
                        // Segment entièrement sorti de la fenêtre gardée.
                        if skip >= segment.len() {
                            skip -= segment.len();
                            continue;
                        }
                        let from = std::mem::take(&mut skip);
                        let len = segment.len() - from;
                        match (*bf16_store, segment) {
                            (true, KvSegment::SharedBf16(rows)) => {
                                prefix_copies.push(PrefixCopy::Bf16 {
                                    source: rows.buffer().to_owned(),
                                    source_offset: rows.byte_offset() + byte_len::<u16>(from)?,
                                    target: buffer.clone(),
                                    target_offset: byte_len::<u16>(offset)?,
                                    len,
                                });
                            }
                            (true, KvSegment::Shared(rows)) => {
                                prefix_copies.push(PrefixCopy::F32ToBf16 {
                                    source: rows.buffer().to_owned(),
                                    source_start: usize::try_from(rows.byte_offset())
                                        .map_err(|_| InferError::Shape("offset préfixe".into()))?
                                        / std::mem::size_of::<f32>()
                                        + from,
                                    target: buffer.clone(),
                                    target_start: offset,
                                    len,
                                });
                            }
                            (false, KvSegment::Shared(rows)) => {
                                prefix_copies.push(PrefixCopy::F32 {
                                    source: rows.buffer().to_owned(),
                                    source_offset: rows.byte_offset() + byte_len::<f32>(from)?,
                                    target: buffer.clone(),
                                    target_offset: byte_len::<f32>(offset)?,
                                    len,
                                });
                            }
                            (true, KvSegment::CpuBf16(data)) => {
                                write_u16_buffer_at(buffer, offset, &data[from..])?;
                            }
                            // Arrondi RNE, comme `bfloat(x)` sur le GPU.
                            (true, _) => write_u16_buffer_at(
                                buffer,
                                offset,
                                &super::buffers::f32_slice_to_bf16(&segment.as_f32()[from..]),
                            )?,
                            (false, _) => {
                                write_f32_buffer_at(buffer, offset, &segment.as_f32()[from..])?;
                            }
                        }
                        offset += len;
                    }
                }
            }
            layer_cache_buffers.push(cache);
        }
        let mut owned_buffers = Vec::new();
        // Queue publiée de chaque couche bornée : buffers K/V et première
        // ligne absolue.
        let mut window_tails: Vec<Option<([Buffer; 2], usize)>> =
            layer_cache_buffers.iter().map(|_| None).collect();

        let command_buffer = self.queue.new_command_buffer();
        let use_shared_encoder = kernel_timing
            .as_ref()
            .is_none_or(PrefillKernelTiming::uses_dispatch_boundary);
        let encoder = if use_shared_encoder {
            Some(command_buffer.new_compute_command_encoder())
        } else {
            None
        };
        let mut encoder_guard = encoder.map(EncoderEndGuard::new);
        let fallback_encoder = encoder_guard.as_ref().map(EncoderEndGuard::encoder);
        if profile_sections {
            let guard = encoder_guard
                .take()
                .expect("invariant: garde encodeur initialisée");
            guard.end();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            ensure_completed(command_buffer.status())?;
        }
        let encode_started = trace.then(std::time::Instant::now);
        let mut current_buffer = match &source {
            ResidentPrefillSource::Hidden(buffer) => buffer.clone(),
            ResidentPrefillSource::Tokens { .. } => chunk_input
                .clone()
                .expect("invariant: chunk_input alloué en entrée par jetons"),
        };
        {
            let mut encode_context = ResidentPrefillEncodeContext {
                profile_sections,
                section_profile: &mut section_profile,
                kernel_timing: &mut kernel_timing,
                command_buffer,
                fallback_encoder,
                owned_buffers: &mut owned_buffers,
            };
            if !prefix_copies.is_empty() {
                self.run_resident_prefill_section(
                    &mut encode_context,
                    "prefix_copy",
                    |encoder, _owned| {
                        for copy in &prefix_copies {
                            match copy {
                                PrefixCopy::Bf16 {
                                    source,
                                    source_offset,
                                    target,
                                    target_offset,
                                    len,
                                } => self.encode_copy_u16_with_offsets(
                                    encoder,
                                    source,
                                    *source_offset,
                                    target,
                                    *target_offset,
                                    *len,
                                )?,
                                PrefixCopy::F32 {
                                    source,
                                    source_offset,
                                    target,
                                    target_offset,
                                    len,
                                } => self.encode_copy_with_offsets(
                                    encoder,
                                    source,
                                    *source_offset,
                                    target,
                                    *target_offset,
                                    *len,
                                )?,
                                PrefixCopy::F32ToBf16 {
                                    source,
                                    source_start,
                                    target,
                                    target_start,
                                    len,
                                } => self.encode_f32_to_bf16_offsets(
                                    encoder,
                                    source,
                                    *source_start,
                                    target,
                                    *target_start,
                                    *len,
                                )?,
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            for chunk_start in (start..spec.seq).step_by(chunk_rows) {
                let chunk = ResidentPrefillChunk {
                    start: chunk_start,
                    rows: chunk_rows.min(spec.seq - chunk_start),
                    total: spec.seq,
                };
                let chunk_hidden_len =
                    checked_len(chunk.rows, hidden_dim, "prefill résident hidden morceau")?;
                let chunk_offset_bytes = f32_byte_offset(checked_len(
                    chunk.start - start,
                    hidden_dim,
                    "morceau offset",
                )?)?;
                if let Some(chunk_input) = chunk_input.as_ref() {
                    self.run_resident_prefill_section(
                        &mut encode_context,
                        "chunk_input",
                        |encoder, _owned| match &source {
                            ResidentPrefillSource::Hidden(input_buffer) => self
                                .encode_copy_with_offsets(
                                    encoder,
                                    input_buffer,
                                    chunk_offset_bytes,
                                    chunk_input,
                                    0,
                                    chunk_hidden_len,
                                ),
                            ResidentPrefillSource::Tokens {
                                weights,
                                indices,
                                scale,
                                recast_bf16,
                            } => self.encode_embedding_from_index_buffers_with_offset_scaled(
                                encoder,
                                weights,
                                indices,
                                byte_len::<u32>(chunk.start - start)?,
                                chunk_input,
                                hidden_dim,
                                chunk.rows,
                                *scale,
                                *recast_bf16,
                            ),
                        },
                    )?;
                    current_buffer = chunk_input.clone();
                }
                for (layer_index, layer) in layers.iter().enumerate() {
                    let layer_spec = PrefillAttentionSpec {
                        seq: chunk.rows,
                        ..layer.attention_spec
                    };
                    let (q_dim, kv_dim, layer_shape) =
                        self.resident_prefill_layer_shape(layer, layer_index, spec, hidden_dim)?;
                    let kv_only = last_row_only
                        && layer_index + 1 == layers.len()
                        && chunk.start + chunk.rows < spec.seq
                        && matches!(layer.attention, PrefillAttentionLayer::Full { .. })
                        && !capture_layer_ids.contains(&layer_index);
                    // Scratch dimensionné sur `chunk_rows` : le dernier morceau, plus
                    // court, réutilise les mêmes buffers du cache de scratch.
                    let scratch = self.allocate_prefill_resident_layer_scratch(
                        &layer_shape,
                        PrefillAttentionSpec {
                            seq: chunk_rows,
                            ..layer_spec
                        },
                        hidden_dim,
                        q_dim,
                        kv_dim,
                        layer.tail.top_k(),
                    )?;
                    let tail_shape = layer_shape.tail;
                    let PrefillResidentLayerScratch {
                        input_norm: input_norm_buffer,
                        post_norm: post_norm_buffer,
                        normed: normed_buffer,
                        attention: attention_scratch,
                        attention_state: attention_state_buffer,
                        post_normed: post_normed_buffer,
                        tail: tail_scratch,
                    } = scratch;
                    let output_buffer = if layer_index % 2 == 0 {
                        hidden_a.clone()
                    } else {
                        hidden_b.clone()
                    };

                    self.encode_resident_input_norm_phase(
                        &mut encode_context,
                        &current_buffer,
                        &input_norm_buffer,
                        &normed_buffer,
                        layer_spec,
                        hidden_dim,
                    )?;
                    let attention_output_buffer = self.encode_resident_attention_phase(
                        &mut encode_context,
                        layer.attention,
                        attention_scratch,
                        &normed_buffer,
                        &layer_cache_buffers[layer_index],
                        chunk,
                        layer_spec,
                        hidden_dim,
                        q_dim,
                        layer_index,
                        kv_only,
                    )?;
                    if kv_only {
                        continue;
                    }
                    self.encode_resident_residual_phase(
                        &mut encode_context,
                        &current_buffer,
                        &attention_output_buffer,
                        &post_norm_buffer,
                        &attention_state_buffer,
                        &post_normed_buffer,
                        layer.post_norm_before_residual,
                        layer_spec,
                        hidden_dim,
                    )?;
                    self.encode_resident_moe_tail_phase(
                        &mut encode_context,
                        layer.tail,
                        tail_shape,
                        tail_scratch,
                        &post_normed_buffer,
                        &attention_state_buffer,
                        &output_buffer,
                        layer_spec,
                        hidden_dim,
                        chunk_hidden_len,
                        layer_index,
                    )?;
                    current_buffer = output_buffer;
                    if let Some(capture_buffer) = capture_buffer.as_ref() {
                        for slot in capture_layer_ids
                            .iter()
                            .enumerate()
                            .filter_map(|(slot, &id)| (id == layer_index).then_some(slot))
                        {
                            let offset = f32_byte_offset(
                                checked_len(slot, hidden_len, "prefill résident capture offset")?
                                    .checked_add((chunk.start - start) * hidden_dim)
                                    .ok_or_else(|| {
                                        InferError::Shape(
                                            "prefill résident capture offset déborde".to_string(),
                                        )
                                    })?,
                            )?;
                            self.run_resident_prefill_section(
                                &mut encode_context,
                                "capture_copy",
                                |encoder, _owned| {
                                    self.encode_copy_with_offsets(
                                        encoder,
                                        &current_buffer,
                                        0,
                                        capture_buffer,
                                        offset,
                                        chunk_hidden_len,
                                    )
                                },
                            )?;
                        }
                    }
                }
                if let Some(final_output) = final_output.as_ref() {
                    self.run_resident_prefill_section(
                        &mut encode_context,
                        "final_copy",
                        |encoder, _owned| {
                            self.encode_copy_with_offsets(
                                encoder,
                                &current_buffer,
                                0,
                                final_output,
                                chunk_offset_bytes,
                                chunk_hidden_len,
                            )
                        },
                    )?;
                }
            }
            // Couches bornées : la fenêtre finale part dans un buffer à sa
            // taille, que le cache et ses snapshots tiennent seul ; le buffer
            // de travail (2 × fenêtre + morceau) meurt avec ce prefill.
            for (cache, tail) in layer_cache_buffers.iter().zip(window_tails.iter_mut()) {
                let PrefillResidentLayerCacheBuffer::Full {
                    key,
                    value,
                    kv_dim,
                    bf16_store,
                    window: Some(window),
                    ..
                } = cache
                else {
                    continue;
                };
                let stored = spec.seq - window.base.get();
                if stored <= window.keep {
                    continue;
                }
                let from = checked_len(stored - window.keep, *kv_dim, "K/V fenêtré queue")?;
                let len = checked_len(window.keep, *kv_dim, "K/V fenêtré queue")?;
                let buffers = if *bf16_store {
                    [
                        self.uncached_bf16_buffer(len, "resident_window_k_bf16")?,
                        self.uncached_bf16_buffer(len, "resident_window_v_bf16")?,
                    ]
                } else {
                    [
                        self.uncached_f32_buffer(len, "resident_window_k")?,
                        self.uncached_f32_buffer(len, "resident_window_v")?,
                    ]
                };
                self.run_resident_prefill_section(
                    &mut encode_context,
                    "window_tail",
                    |encoder, _owned| {
                        for (source, target) in [(key, &buffers[0]), (value, &buffers[1])] {
                            self.encode_kv_copy(
                                encoder,
                                source,
                                from,
                                target,
                                0,
                                len,
                                *bf16_store,
                            )?;
                        }
                        Ok(())
                    },
                )?;
                *tail = Some((buffers, spec.seq - window.keep));
            }
        }
        let final_output_used = final_output.is_some();
        let final_read_buffer = final_output.unwrap_or(current_buffer);
        let encode_elapsed = encode_started.map(|started| started.elapsed());
        let wait_elapsed = if profile_sections {
            None
        } else {
            if let Some(guard) = encoder_guard.take() {
                guard.end();
            }
            if let Some(timing) = kernel_timing.as_ref() {
                timing.encode_resolve(command_buffer)?;
            }
            let wait_started = trace.then(std::time::Instant::now);
            command_buffer.commit();
            command_buffer.wait_until_completed();
            ensure_completed(command_buffer.status())?;
            if let Some(timing) = kernel_timing.as_ref() {
                if let Err(error) = timing.dump_report() {
                    eprintln!("gpu_timestamps report_error={error}");
                }
            }
            wait_started.map(|started| started.elapsed())
        };

        let read_started = trace.then(std::time::Instant::now);
        let profile_read_started = profile_sections.then(std::time::Instant::now);
        let output = if !last_row_only
            && final_output_used
            && crate::runtime_flags::prefill_kv_shared_enabled()
        {
            // SAFETY: `final_output` est un `uncached_f32_buffer` propre à ce
            // prefill, hors scratch ; son command buffer est terminé et plus
            // rien ne l'écrit.
            #[expect(unsafe_code, reason = "vue en place du hidden final figé")]
            let rows = unsafe { SharedF32Rows::new(final_read_buffer, 0, hidden_len)? };
            PrefillKvRows::Shared(rows)
        } else if last_row_only {
            // Dernière ligne du morceau final, ou de la sortie pleine longueur
            // quand le scratch privé l'impose.
            let last_row = if final_output_used {
                rows - 1
            } else {
                (rows - 1) % chunk_rows
            };
            PrefillKvRows::Owned(read_f32_buffer_at(
                &final_read_buffer,
                checked_len(last_row, hidden_dim, "prefill résident dernière ligne")?,
                hidden_dim,
            )?)
        } else {
            PrefillKvRows::Owned(read_f32_buffer(&final_read_buffer, hidden_len)?)
        };
        let captures = capture_buffer
            .map(|buffer| -> Result<Tensor> {
                let count = capture_layer_ids.len();
                let raw = read_f32_buffer(&buffer, count * hidden_len)?;
                // [n, seq, hidden] sur le GPU -> [seq, n * hidden] attendu par le drafter.
                let mut out = vec![0.0_f32; count * hidden_len];
                for slot in 0..count {
                    for row in 0..rows {
                        let src = slot * hidden_len + row * hidden_dim;
                        let dst = (row * count + slot) * hidden_dim;
                        out[dst..dst + hidden_dim].copy_from_slice(&raw[src..src + hidden_dim]);
                    }
                }
                Tensor::from_vec(vec![rows, count * hidden_dim], out)
            })
            .transpose()?;
        let mut layer_caches = Vec::with_capacity(layer_cache_buffers.len());
        for (cache_buffer, tail) in layer_cache_buffers.into_iter().zip(window_tails) {
            match cache_buffer {
                PrefillResidentLayerCacheBuffer::Full {
                    key,
                    value,
                    kv_dim,
                    bf16_store,
                    window: Some(window),
                    ..
                } => {
                    // Couche bornée : seule la fenêtre `[base, seq)` est
                    // rendue, et remplace les segments d'un préfixe.
                    let ([key, value], base) = tail.unwrap_or(([key, value], window.base.get()));
                    let kv_len = checked_len(spec.seq - base, kv_dim, "K/V fenêtré rendu")?;
                    let (key, value) = if crate::runtime_flags::prefill_kv_shared_enabled() {
                        shared_kv_rows(key, value, kv_len, bf16_store)?
                    } else if bf16_store {
                        (
                            PrefillKvRows::OwnedBf16(read_u16_buffer_at(&key, 0, kv_len)?),
                            PrefillKvRows::OwnedBf16(read_u16_buffer_at(&value, 0, kv_len)?),
                        )
                    } else {
                        (
                            PrefillKvRows::Owned(read_f32_buffer_at(&key, 0, kv_len)?),
                            PrefillKvRows::Owned(read_f32_buffer_at(&value, 0, kv_len)?),
                        )
                    };
                    layer_caches.push(PrefillResidentLayerCache::Full {
                        key,
                        value,
                        kv_dim,
                        placement: if start == 0 {
                            PrefillKvPlacement::Appended
                        } else {
                            PrefillKvPlacement::CopiedPrefix
                        },
                        base,
                    });
                }
                PrefillResidentLayerCacheBuffer::Full {
                    key,
                    value,
                    kv_dim,
                    bf16_store,
                    replaces_prefix,
                    in_place,
                    ..
                } => {
                    // Seules les positions nouvelles repartent au CPU : le préfixe
                    // y est déjà.
                    let kv_len = checked_len(rows, kv_dim, "prefill résident readback kv")?;
                    let kv_offset = checked_len(start, kv_dim, "prefill résident readback kv")?;
                    // Prefill complet : le buffer, que plus rien n'écrit, devient
                    // le stockage CPU du K/V. Extension : il porte aussi le
                    // préfixe, seules les lignes nouvelles sont relues.
                    let mut placement = PrefillKvPlacement::Appended;
                    let (key, value) = if let Some(InPlaceReuse {
                        claim,
                        bf16_prefix: Some([key_prefix, value_prefix]),
                    }) = in_place
                        .as_ref()
                        .filter(|_| crate::runtime_flags::prefill_kv_shared_enabled())
                    {
                        // Prolongement bf16 en place : vue entière `[0, seq)`
                        // dérivée de celle du préfixe (même registre), qui
                        // remplace les segments du cache. Tant qu'elle vit,
                        // aucun autre prolongement du même préfixe n'écrit
                        // plus en place.
                        let whole = checked_len(spec.seq, kv_dim, "prefill résident kv en place")?;
                        placement = PrefillKvPlacement::ExtendedInPlace;
                        extended_kv_rows([key_prefix, value_prefix], claim, whole)?
                    } else if replaces_prefix {
                        // Buffer neuf, préfixe recopié : vue entière de tout le K/V.
                        placement = PrefillKvPlacement::CopiedPrefix;
                        shared_kv_rows(
                            key,
                            value,
                            checked_len(spec.seq, kv_dim, "prefill résident readback kv")?,
                            true,
                        )?
                    } else if start == 0 && crate::runtime_flags::prefill_kv_shared_enabled() {
                        shared_kv_rows(key, value, kv_len, bf16_store)?
                    } else if bf16_store {
                        // Lignes possédées, hors de toute vue : relues sous le
                        // claim d'un prolongement en place f32 ou sans vues
                        // partagées ; un prolongement ultérieur du même préfixe
                        // peut réécrire la réserve.
                        (
                            PrefillKvRows::OwnedBf16(read_u16_buffer_at(&key, kv_offset, kv_len)?),
                            PrefillKvRows::OwnedBf16(read_u16_buffer_at(
                                &value, kv_offset, kv_len,
                            )?),
                        )
                    } else {
                        (
                            PrefillKvRows::Owned(read_f32_buffer_at(&key, kv_offset, kv_len)?),
                            PrefillKvRows::Owned(read_f32_buffer_at(&value, kv_offset, kv_len)?),
                        )
                    };
                    // Fin de la relecture ou de la publication : la réserve
                    // est rendue.
                    // NOTE: le claim (ici, ou au drop sur une sortie d'erreur)
                    // ne survit à aucune écriture GPU parce que rien ne sépare
                    // `commit` de `wait_until_completed` : aucune sortie `?`
                    // entre les deux. Déplacer l'attente (soumission
                    // asynchrone) exigerait d'attendre avant tout drop du claim,
                    // comme `AdoptedKv`.
                    drop(in_place);
                    layer_caches.push(PrefillResidentLayerCache::Full {
                        key,
                        value,
                        kv_dim,
                        placement,
                        base: 0,
                    });
                }
                PrefillResidentLayerCacheBuffer::Linear { state } => {
                    layer_caches.push(PrefillResidentLayerCache::Linear { state });
                }
            }
        }
        let read_elapsed = read_started.map(|started| started.elapsed());
        if let Some(started) = profile_read_started {
            section_profile.add("readback", started.elapsed().as_micros(), 0);
        }
        if let Some(total_started) = total_started {
            eprintln!(
                "prefill_resident profile encode_us={} wait_us={} read_us={} total_us={}",
                encode_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                wait_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                read_elapsed.map_or(0, |elapsed| elapsed.as_micros()),
                total_started.elapsed().as_micros()
            );
        }
        if let Some(profile_total_started) = profile_total_started {
            let conversion_shapes = take_prefill_f32_to_bf16_shapes();
            self.profile_f32_to_bf16_conversions(&mut section_profile, &conversion_shapes)?;
            eprintln!(
                "prefill_section_run seq={} layers={} total_wall_us={}",
                spec.seq,
                layers.len(),
                profile_total_started.elapsed().as_micros()
            );
            section_profile.dump();
        }
        Ok((output, layer_caches, captures))
    }

    /// Lignes par morceau : les `rows` positions à calculer sont coupées en
    /// morceaux égaux d'au plus `target` lignes, si toutes les attentions
    /// complètes passent par le chemin Steel/NAX à décalage ; sinon un seul
    /// morceau.
    fn resident_prefill_chunk_rows(
        &self,
        layers: &[PrefillMoeLayer<'_>],
        rows: usize,
        target: usize,
    ) -> usize {
        // NOTE: les timestamps GPU plafonnent à 8192 échantillons ; un seul
        // morceau garde le profil par couche lisible.
        if target == 0
            || rows <= target
            || crate::runtime_flags::gpu_timestamps_enabled()
            || !self.prefill_resident_extend_supported(layers)
        {
            return rows;
        }
        rows.div_ceil(rows.div_ceil(target))
    }

    fn resident_prefill_layer_shape<'a>(
        &self,
        layer: &PrefillMoeLayer<'a>,
        layer_index: usize,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
    ) -> Result<(usize, usize, PrefillResidentLayerShape<'a>)> {
        let layer_spec = layer.attention_spec;
        if layer_spec.seq != spec.seq
            || layer_spec.hidden_dim != hidden_dim
            || layer_spec.q_heads != spec.q_heads
            || layer_spec.eps != spec.eps
        {
            return Err(InferError::Dimension(format!(
                "prefill résident spec incohérent couche {layer_index}: {layer_spec:?}"
            )));
        }
        let q_dim = checked_len(
            layer_spec.q_heads,
            layer_spec.head_dim,
            "prefill résident q_dim couche",
        )?;
        let kv_dim = checked_len(
            layer_spec.kv_heads,
            layer_spec.head_dim,
            "prefill résident kv_dim couche",
        )?;
        let shape = self.check_prefill_resident_layer_shapes(
            *layer,
            layer_index,
            hidden_dim,
            q_dim,
            kv_dim,
            layer_spec,
        )?;
        Ok((q_dim, kv_dim, shape))
    }

    #[inline]
    pub(super) fn run_resident_prefill_section<F>(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        label: &'static str,
        encode: F,
    ) -> Result<()>
    where
        F: FnOnce(&ComputeCommandEncoderRef, &mut Vec<Buffer>) -> Result<()>,
    {
        if context.profile_sections {
            self.run_prefill_profile_section(context.section_profile, label, encode)
        } else {
            time_prefill_pass(
                context.kernel_timing.as_mut(),
                context.command_buffer,
                context.fallback_encoder,
                label,
                |encoder| encode(encoder, context.owned_buffers),
            )
        }
    }

    #[inline]
    fn encode_resident_input_norm_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        current_buffer: &BufferRef,
        input_norm_buffer: &BufferRef,
        normed_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
    ) -> Result<()> {
        self.run_resident_prefill_section(context, "input_norm", |encoder, _owned| {
            self.encode_rms_norm_rows(
                encoder,
                current_buffer,
                input_norm_buffer,
                normed_buffer,
                spec.seq,
                hidden_dim,
                spec.eps,
            )
        })
    }

    #[inline]
    fn encode_resident_residual_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        current_buffer: &BufferRef,
        attention_output_buffer: &BufferRef,
        post_norm_buffer: &BufferRef,
        attention_state_buffer: &BufferRef,
        post_normed_buffer: &BufferRef,
        post_norm_before_residual: bool,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
    ) -> Result<()> {
        self.run_resident_prefill_section(context, "o_postnorm", |encoder, owned| {
            if post_norm_before_residual {
                self.encode_rms_norm_rows(
                    encoder,
                    attention_output_buffer,
                    post_norm_buffer,
                    post_normed_buffer,
                    spec.seq,
                    hidden_dim,
                    spec.eps,
                )?;
                return self.encode_add_scaled(
                    encoder,
                    owned,
                    current_buffer,
                    post_normed_buffer,
                    attention_state_buffer,
                    1.0,
                    checked_len(spec.seq, hidden_dim, "resident Gemma attention state")?,
                );
            }
            self.encode_add_rms_norm_rows(
                encoder,
                current_buffer,
                attention_output_buffer,
                post_norm_buffer,
                attention_state_buffer,
                post_normed_buffer,
                spec.seq,
                hidden_dim,
                spec.eps,
            )
        })
    }

    #[inline]
    fn encode_resident_moe_tail_phase(
        &self,
        context: &mut ResidentPrefillEncodeContext<'_>,
        tail: PrefillMoeTail<'_>,
        tail_shape: PrefillResidentTailShape,
        tail_scratch: PrefillResidentTailScratch,
        post_normed_buffer: &BufferRef,
        attention_state_buffer: &BufferRef,
        output_buffer: &BufferRef,
        spec: PrefillAttentionSpec,
        hidden_dim: usize,
        hidden_len: usize,
        layer_index: usize,
    ) -> Result<()> {
        match (tail, tail_shape, tail_scratch) {
            (
                PrefillMoeTail::Dense { .. },
                PrefillResidentTailShape::Dense(mlp),
                PrefillResidentTailScratch::Dense(scratch),
            ) => self.run_resident_prefill_section(context, "tail_dense", |encoder, owned| {
                self.encode_resident_dense_mlp(
                    encoder,
                    owned,
                    post_normed_buffer,
                    &mlp,
                    &scratch,
                    spec.seq,
                    hidden_dim,
                )?;
                self.encode_copy(encoder, attention_state_buffer, output_buffer, hidden_len)?;
                self.encode_accumulate_scaled(
                    encoder,
                    owned,
                    &scratch.down,
                    output_buffer,
                    1.0,
                    hidden_len,
                )
            }),
            (
                PrefillMoeTail::GemmaDense { .. },
                PrefillResidentTailShape::GemmaDense {
                    gate_proj,
                    up_proj,
                    down_proj,
                    pre_feedforward_norm,
                    post_feedforward_norm,
                    layer_scalar,
                    inter_dim,
                },
                PrefillResidentTailScratch::GemmaDense {
                    ffn_input,
                    gate,
                    up,
                    geglu,
                    down,
                    ffn_normed,
                },
            ) => {
                self.run_resident_prefill_section(context, "tail_gemma_dense", |encoder, owned| {
                    self.encode_gemma_dense_tail_rows(
                        encoder,
                        owned,
                        attention_state_buffer,
                        output_buffer,
                        spec.seq,
                        hidden_dim,
                        spec.eps,
                        &gate_proj,
                        &up_proj,
                        &down_proj,
                        &pre_feedforward_norm,
                        &post_feedforward_norm,
                        layer_scalar,
                        inter_dim,
                        &ffn_input,
                        &gate,
                        &up,
                        &geglu,
                        &down,
                        &ffn_normed,
                    )
                })
            }
            (
                PrefillMoeTail::GemmaParallel { top_k, .. },
                PrefillResidentTailShape::GemmaParallel {
                    dense_gate_proj,
                    dense_up_proj,
                    dense_down_proj,
                    pre_feedforward_norm,
                    post_feedforward_norm_1,
                    moe,
                    router_norm,
                    per_expert_scale,
                    pre_feedforward_norm_2,
                    post_feedforward_norm_2,
                    post_feedforward_norm,
                    layer_scalar,
                    dense_inter_dim,
                },
                PrefillResidentTailScratch::GemmaParallel {
                    dense_input,
                    dense_gate,
                    dense_up,
                    dense_geglu,
                    dense_down,
                    dense_out,
                    moe_input,
                    moe_out,
                    ffn_out,
                    ffn_normed,
                },
            ) => self.run_resident_prefill_section(
                context,
                "tail_gemma_parallel",
                |encoder, owned| {
                    self.encode_gemma_parallel_tail_rows(
                        encoder,
                        owned,
                        attention_state_buffer,
                        output_buffer,
                        spec.seq,
                        hidden_dim,
                        spec.eps,
                        &dense_gate_proj,
                        &dense_up_proj,
                        &dense_down_proj,
                        &pre_feedforward_norm,
                        &post_feedforward_norm_1,
                        &moe,
                        top_k,
                        router_norm
                            .as_ref()
                            .map(|(weight, eps)| (weight.as_ref(), *eps)),
                        per_expert_scale.as_ref().map(Buffer::as_ref),
                        &pre_feedforward_norm_2,
                        &post_feedforward_norm_2,
                        &post_feedforward_norm,
                        layer_scalar,
                        dense_inter_dim,
                        &dense_input,
                        &dense_gate,
                        &dense_up,
                        &dense_geglu,
                        &dense_down,
                        &dense_out,
                        &moe_input,
                        &moe_out,
                        &ffn_out,
                        &ffn_normed,
                    )
                },
            ),
            (
                PrefillMoeTail::Routed { router, top_k, .. },
                PrefillResidentTailShape::Routed {
                    expert_count,
                    stacked,
                    sigmoid_shared,
                },
                PrefillResidentTailScratch::Routed {
                    router: router_buffer,
                    indices: indices_buffer,
                    scores: scores_buffer,
                    gate: gate_buffer,
                    up: up_buffer,
                    hidden: hidden_buffer,
                    down: down_buffer,
                    sigmoid_shared: sigmoid_shared_scratch,
                },
            ) => {
                let total_topk = checked_len(spec.seq, top_k, "resident topk total")?;
                let inter_dim = stacked.gate.out_dim;
                // flint (expert partagé sans gate) route en sigmoïde ; Qwen en softmax.
                let router_score = if sigmoid_shared.is_some() {
                    crate::RouterScore::Sigmoid
                } else {
                    crate::RouterScore::Softmax
                };
                // Bascule routed coop, défaut ON (cf. `moe_routed_coop_prefill_enabled`,
                // politique near-tie bf16) ; flint a son kill-switch dédié. Repli
                // gather qmv si la quantification ou les pipelines MPP manquent. Le
                // routeur n'est résolu que dans la branche coop.
                if moe_routed_coop_prefill_enabled()
                    && (sigmoid_shared.is_none() || flint_moe_coop_enabled())
                    && MetalMoeRoutedWeights::stacked_coop_compatible(&stacked)
                    && self.routed_coop_pipelines_ready(&stacked)
                {
                    let weights = MetalMoeRoutedWeights {
                        router: self.resolve_linear_weight_buffers(
                            router.weight(),
                            "resident_moe_router",
                        )?,
                        stacked,
                    };
                    self.run_resident_prefill_section(
                        context,
                        "moe_routed_coop",
                        |encoder, owned| {
                            self.encode_moe_routed_rows_coop(
                                encoder,
                                owned,
                                post_normed_buffer,
                                Some(attention_state_buffer),
                                output_buffer,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                &down_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                                router_score,
                            )
                        },
                    )?;
                } else {
                    self.run_resident_prefill_section(context, "router_topk", |encoder, owned| {
                        self.encode_matmul_weight(
                            encoder,
                            owned,
                            post_normed_buffer,
                            spec.seq,
                            hidden_dim,
                            router.weight(),
                            &router_buffer,
                        )?;
                        match router_score {
                            crate::RouterScore::Sigmoid => self.encode_topk_sigmoid_rows(
                                encoder,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                spec.seq,
                                expert_count,
                                top_k,
                            ),
                            crate::RouterScore::Softmax => self.encode_topk_softmax_rows(
                                encoder,
                                &router_buffer,
                                &indices_buffer,
                                &scores_buffer,
                                spec.seq,
                                expert_count,
                                top_k,
                            ),
                        }
                    })?;
                    self.run_resident_prefill_section(context, "moe_gate_up", |encoder, owned| {
                        if !self.encode_gather_gate_up_swiglu(
                            encoder,
                            owned,
                            post_normed_buffer,
                            spec.seq,
                            &stacked.gate,
                            &stacked.up,
                            &indices_buffer,
                            total_topk,
                            &hidden_buffer,
                        )? {
                            self.encode_gather_matmul(
                                encoder,
                                owned,
                                post_normed_buffer,
                                spec.seq,
                                &stacked.gate,
                                &indices_buffer,
                                total_topk,
                                &gate_buffer,
                            )?;
                            self.encode_gather_matmul(
                                encoder,
                                owned,
                                post_normed_buffer,
                                spec.seq,
                                &stacked.up,
                                &indices_buffer,
                                total_topk,
                                &up_buffer,
                            )?;
                            self.encode_swiglu(
                                encoder,
                                owned,
                                &gate_buffer,
                                &up_buffer,
                                &hidden_buffer,
                                checked_len(total_topk, inter_dim, "resident swiglu")?,
                            )?;
                        }
                        Ok(())
                    })?;
                    self.run_resident_prefill_section(context, "moe_down", |encoder, owned| {
                        self.encode_gather_matmul(
                            encoder,
                            owned,
                            &hidden_buffer,
                            total_topk,
                            &stacked.down,
                            &indices_buffer,
                            total_topk,
                            &down_buffer,
                        )
                    })?;
                    self.run_resident_prefill_section(
                        context,
                        "moe_weighted_sum",
                        |encoder, owned| {
                            self.encode_weighted_sum_add_grouped_topk(
                                encoder,
                                owned,
                                &down_buffer,
                                &scores_buffer,
                                attention_state_buffer,
                                output_buffer,
                                spec.seq,
                                top_k,
                                hidden_dim,
                            )
                        },
                    )?;
                }
                match (sigmoid_shared, sigmoid_shared_scratch) {
                    (None, None) => Ok(()),
                    // Expert partagé flint, sans gate : ajouté tel quel au flux
                    // qui porte déjà résiduel et experts routés (les deux chemins
                    // écrivent `output = residual + Σ scores·down`).
                    (Some(mlp), Some(scratch)) => self.run_resident_prefill_section(
                        context,
                        "moe_shared_ungated",
                        |encoder, owned| {
                            self.encode_resident_dense_mlp(
                                encoder,
                                owned,
                                post_normed_buffer,
                                &mlp,
                                &scratch,
                                spec.seq,
                                hidden_dim,
                            )?;
                            self.encode_accumulate_scaled(
                                encoder,
                                owned,
                                &scratch.down,
                                output_buffer,
                                1.0,
                                hidden_len,
                            )
                        },
                    ),
                    _ => Err(InferError::Dimension(format!(
                        "prefill résident expert partagé sigmoïde incohérent couche {layer_index}"
                    ))),
                }
            }
            (
                PrefillMoeTail::Shared { top_k, .. },
                PrefillResidentTailShape::Shared { weights },
                PrefillResidentTailScratch::Shared,
            ) => {
                if moe_coop_enabled() && weights.coop_compatible() {
                    self.run_resident_prefill_section(
                        context,
                        "moe_shared_coop",
                        |encoder, owned| {
                            self.encode_moe_shared_rows_coop(
                                encoder,
                                owned,
                                post_normed_buffer,
                                Some(attention_state_buffer),
                                output_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        },
                    )
                } else {
                    self.run_resident_prefill_section(
                        context,
                        "moe_shared_rows",
                        |encoder, owned| {
                            self.encode_moe_shared_buffers_rows(
                                encoder,
                                owned,
                                post_normed_buffer,
                                Some(attention_state_buffer),
                                output_buffer,
                                spec.seq,
                                hidden_dim,
                                &weights,
                                top_k,
                            )
                        },
                    )
                }
            }
            _ => Err(InferError::Dimension(format!(
                "prefill résident tail MoE incohérent couche {layer_index}"
            ))),
        }
    }
}

impl MetalExecutor {
    /// MLP SwiGLU dense de `seq` lignes `input` vers `scratch.down`, fusionné
    /// ou non selon le scratch alloué.
    #[expect(
        clippy::too_many_arguments,
        reason = "MLP dense résident: buffers, poids et dimensions restent explicites"
    )]
    fn encode_resident_dense_mlp(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        input: &BufferRef,
        mlp: &PrefillResidentDenseMlp,
        scratch: &PrefillResidentDenseScratch,
        seq: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        let inter_dim = mlp.inter_dim;
        let Some((gate, up, hidden)) = &scratch.unfused else {
            return self.encode_fused_dense_mlp(
                encoder,
                input,
                seq,
                hidden_dim,
                &mlp.gate_proj,
                &mlp.up_proj,
                &mlp.down_proj,
                inter_dim,
                &scratch.down,
            );
        };
        let gate_dim = self.encode_matmul_weight_buffers(
            encoder,
            input,
            seq,
            hidden_dim,
            &mlp.gate_proj,
            gate,
            false,
        )?;
        let up_dim = self.encode_matmul_weight_buffers(
            encoder,
            input,
            seq,
            hidden_dim,
            &mlp.up_proj,
            up,
            false,
        )?;
        if gate_dim != inter_dim || up_dim != inter_dim {
            return Err(InferError::Dimension(format!(
                "prefill résident dense gate/up sortent gate={gate_dim} up={up_dim}, attendu {inter_dim}"
            )));
        }
        let inter_len = checked_len(seq, inter_dim, "resident dense inter")?;
        self.encode_swiglu(encoder, owned, gate, up, hidden, inter_len)?;
        let down_dim = self.encode_matmul_weight_buffers(
            encoder,
            hidden,
            seq,
            inter_dim,
            &mlp.down_proj,
            &scratch.down,
            false,
        )?;
        if down_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "prefill résident dense down sort {down_dim}, attendu {hidden_dim}"
            )));
        }
        Ok(())
    }
}

/// Vues entières `[0, whole)` dérivées des vues K et V `prefix` d'un
/// prolongement bf16 en place, sous le `claim` revalidé après elles.
#[expect(unsafe_code, reason = "publication du prolongement en place")]
fn extended_kv_rows(
    [key, value]: [&SharedBf16Rows; 2],
    claim: &InPlaceClaim,
    whole: usize,
) -> Result<(PrefillKvRows, PrefillKvRows)> {
    // SAFETY: les lignes `[start, whole)` ont été écrites par ce prefill sous
    // `claim`, pris puis revalidé après `key` et `value` (`prefix_kv_in_place`),
    // et son command buffer est terminé ; plus rien ne les écrit.
    unsafe {
        Ok((
            PrefillKvRows::SharedBf16(key.extended(claim, whole)?),
            PrefillKvRows::SharedBf16(value.extended(claim, whole)?),
        ))
    }
}

impl MetalExecutor {
    /// Copie GPU de `len` éléments K/V (f32, ou bf16 si `bf16`) de `source`
    /// à partir de l'élément `from` vers `target` à partir de `to` : queue
    /// d'une fenêtre bornée (compaction entre morceaux, publication).
    #[expect(
        clippy::too_many_arguments,
        reason = "copie Metal: buffers, offsets, longueur et format explicites"
    )]
    pub(super) fn encode_kv_copy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        source: &BufferRef,
        from: usize,
        target: &BufferRef,
        to: usize,
        len: usize,
        bf16: bool,
    ) -> Result<()> {
        if bf16 {
            self.encode_copy_u16_with_offsets(
                encoder,
                source,
                byte_len::<u16>(from)?,
                target,
                byte_len::<u16>(to)?,
                len,
            )
        } else {
            self.encode_copy_with_offsets(
                encoder,
                source,
                byte_len::<f32>(from)?,
                target,
                byte_len::<f32>(to)?,
                len,
            )
        }
    }
}

/// Remet les buffers K/V d'un prefill terminé comme vues partagées.
#[expect(
    unsafe_code,
    reason = "les buffers K/V du prefill passent au cache CPU en lecture seule"
)]
fn shared_kv_rows(
    key: metal::Buffer,
    value: metal::Buffer,
    kv_len: usize,
    bf16: bool,
) -> Result<(PrefillKvRows, PrefillKvRows)> {
    // SAFETY: buffers `uncached_f32_buffer` ou `uncached_bf16_buffer` neufs,
    // propres à ce prefill et hors scratch : aucune autre vue n'en existe
    // (première vue, registre créé ici), son command buffer est terminé et
    // l'appelant les cède. Seules les lignes au-delà de toute vue vivante
    // seront encore écrites, sous claim revalidé (`prefix_kv_in_place`,
    // adoption par l'arène).
    unsafe {
        Ok(if bf16 {
            (
                PrefillKvRows::SharedBf16(SharedBf16Rows::new(key, 0, kv_len)?),
                PrefillKvRows::SharedBf16(SharedBf16Rows::new(value, 0, kv_len)?),
            )
        } else {
            (
                PrefillKvRows::Shared(SharedF32Rows::new(key, 0, kv_len)?),
                PrefillKvRows::Shared(SharedF32Rows::new(value, 0, kv_len)?),
            )
        })
    }
}
