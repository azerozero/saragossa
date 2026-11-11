//! Stockage partagé entre temporaires de prefill à vies disjointes.
//!
//! Le cache garde sa clé (label, longueur, élément, namespace) : seules les
//! demandes de même taille partagent un MTLBuffer, toujours à offset zéro.
//! Aucun arrondi, allocation maximale, changement de kernel ou de dispatch.
//!
//! Preuve des dernières lectures dans `global_attention` et `resident` :
//!
//! | Stockage | Attention complète | Attention linéaire | Après attention |
//! |---|---|---|---|
//! | Q_CONTEXT | Q → RoPE, puis contexte → gate/O | Z → RMS gate | — |
//! | ROPE_GATED | Q RoPE → attention, puis contexte gaté → O | gated → O | — |
//! | GATE | gate → contexte gaté | Y → RMS gate | — |
//! | OUTPUT | O → résiduel/postnorm | O → résiduel/postnorm | MLP down → ajout |
//! | NORM | entrée normée → projections QKV | entrée normée → projections | postnorm → MLP |
//!
//! Les deux branches d'attention sont exclusives. Dans chacune, la première
//! écriture suivante dépend déjà du résultat du dernier lecteur précédent :
//! Q → RoPE → attention → contexte → gate ; projections → O → postnorm → MLP.
//! On n'introduit donc pas de dépendance entre dispatchs indépendants. Les
//! encodeurs de `resident` sont sériels (`new_compute_command_encoder`) ; les
//! barrières existantes restent intactes. Le profilage par passe emploie soit
//! ces mêmes encodeurs, soit des encodeurs successifs du même command buffer,
//! soit un commit/wait par section. Aucun encodeur concurrent n'utilise ces
//! labels ; le helper batch GDN n'a que le prefill résident pour appelant.
//!
//! Garder un clone du Buffer dans `owned_buffers` conserve sa durée de vie,
//! pas ses valeurs : aucune lecture différée de ces temporaires n'est encodée.
//! Les sorties ping-pong, le résiduel `attention_state`, les KV et états SSM/
//! conv restent distincts. DFlash copie le ping-pong après le tail ; le semis
//! MTP le lit après toutes les couches. `last_row_only` copie ce même buffer
//! et `kv_only` saute attention/tail sans lire O. Ni les captures, ni les
//! préfixes, ni les vues partagées rendues ne retiennent ces cinq stockages.
//!
//! Le dernier morceau reprend les capacités via `scratch_len_for_rows` ; les
//! kernels continuent d'écrire/lire seulement leurs lignes actives. Les autres
//! formes et namespaces restent isolés. Le decode et les conversions bf16
//! génériques conservent leurs labels : aucun travail ajouté sur leur chemin.
//!
//! Pour Qwen3.8-27B (H=5120, Q=6144, C=4079), le gain logique vaut
//! `4*C*(5*Q+3*H) = 751_841_280` octets : le total de référence passe de
//! 2,534332 Go à 1,782491 Go. Ce calcul ne mesure ni la résidence physique ni
//! le débit. Les QKV/conv GDN de taille différente et le MLP bf16 restent
//! séparés : les partager exigerait des capacités/vues nouvelles, et une
//! preuve supplémentaire pour les kernels génériques et leurs appelants.

pub(in crate::metal_backend) const PHASE_Q_CONTEXT: &str = "resident_phase_q_context_z";
pub(in crate::metal_backend) const PHASE_ROPE_GATED: &str = "resident_phase_rope_gated";
pub(in crate::metal_backend) const PHASE_GATE: &str = "resident_phase_gate_y";
pub(in crate::metal_backend) const PHASE_OUTPUT: &str = "resident_phase_output_down";
pub(in crate::metal_backend) const PHASE_NORM: &str = "resident_phase_norm";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_backend::*;

    #[test]
    fn prefill_phase_scratch_budget_27b() {
        let rows = 261_000_usize.div_ceil(261_000_usize.div_ceil(4096));
        let (h, q) = (5120, 6144);
        // Trois Q de GDN + deux Q de full-attention + deux sorties H et une
        // norme H disparaissent. Les autres allocations restent inchangées.
        let before = rows * (8 * q + 5 * h) * 4;
        let after = rows * (3 * q + 2 * h) * 4;
        assert_eq!(rows, 4079);
        assert_eq!(before - after, rows * (5 * q + 3 * h) * 4);
        assert_eq!(before - after, 751_841_280);
    }

    /// Même stockage réécrit trois fois dans un encodeur sériel, sans barrière
    /// ajoutée : chaque phase publie son résultat avant la réécriture suivante.
    /// Compare les bits à des RMS exécutés dans des buffers tous distincts.
    #[test]
    fn prefill_phase_reuse_is_bit_exact_without_weights() -> Result<()> {
        let executor = match MetalExecutor::new() {
            Ok(executor) => executor,
            Err(InferError::Metal(message)) if message.contains("aucun device") => return Ok(()),
            Err(error) => return Err(error),
        };
        let labels = [
            PHASE_Q_CONTEXT,
            PHASE_ROPE_GATED,
            PHASE_GATE,
            PHASE_OUTPUT,
            PHASE_NORM,
        ];
        let (capacity, width) = (17, 64);
        let weights = executor.uncached_f32_buffer(width, "phase_test_norm")?;
        write_f32_buffer(&weights, &vec![1.0; width])?;
        let buffers = labels
            .iter()
            .map(|label| executor.private_f32_buffer(capacity * width, label))
            .collect::<Result<Vec<_>>>()?;
        let (entries, bytes) = executor.scratch_buffer_stats()?;
        assert_eq!(entries, labels.len());
        assert_eq!(bytes, (labels.len() * capacity * width * 4) as u64);

        let command = executor.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(encoder);
        let mut results = Vec::new();
        let mut inputs = Vec::new();
        for (slot, label) in labels.iter().enumerate() {
            for phase in 0..3 {
                // La dernière phase simule le dernier morceau plus court.
                let rows = if phase == 2 { capacity - 1 } else { capacity };
                let _rows = enter_scratch_row_capacity(rows, capacity);
                let shared = executor.private_f32_buffer(rows * width, label)?;
                assert_eq!(shared.as_ptr(), buffers[slot].as_ptr());
                let values = (0..rows * width)
                    .map(|i| ((i * 17 + phase * 31 + slot * 7) % 257) as f32 / 128.0 - 1.0)
                    .collect::<Vec<_>>();
                // Les uploads ne doivent pas reprendre un label mémoïsé :
                // toutes ces entrées sont consommées après le même commit.
                let input = executor.uncached_f32_buffer(values.len(), "phase_test_input")?;
                write_f32_buffer(&input, &values)?;
                let saved = executor.uncached_f32_buffer(rows * width, "phase_test_saved")?;
                let reference =
                    executor.uncached_f32_buffer(rows * width, "phase_test_reference")?;
                executor.encode_rms_norm_rows(
                    encoder, &input, &weights, &shared, rows, width, 1.0e-6,
                )?;
                executor.encode_copy(encoder, &shared, &saved, rows * width)?;
                executor.encode_rms_norm_rows(
                    encoder, &input, &weights, &reference, rows, width, 1.0e-6,
                )?;
                inputs.push(input);
                results.push((saved, reference, rows * width));
            }
        }
        guard.end();
        commit_and_wait(command)?;
        assert_eq!(executor.scratch_buffer_stats()?, (entries, bytes));
        for (saved, reference, len) in results {
            let bits = |buffer: &BufferRef| -> Result<Vec<u32>> {
                Ok(read_f32_buffer(buffer, len)?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(bits(&saved)?, bits(&reference)?);
        }
        // Ni une autre taille ni un autre flux ne doivent reprendre le slot.
        let different = executor.private_f32_buffer(capacity * width + 1, PHASE_NORM)?;
        assert_ne!(different.as_ptr(), buffers[4].as_ptr());
        let _namespace = install_scratch_namespace(771);
        let other_stream = executor.private_f32_buffer(capacity * width, PHASE_NORM)?;
        assert_ne!(other_stream.as_ptr(), buffers[4].as_ptr());
        Ok(())
    }
}
