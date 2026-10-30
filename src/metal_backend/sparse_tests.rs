//! Tests du K/V sparse. Ceux qui touchent le GPU s'ignorent (retour précoce
//! avec un message) si le device ne gère pas le placement sparse
//! (`supportsPlacementSparse`, macOS 26.4) : les runners GitHub actuels ne le
//! gèrent pas, la preuve de ces tests reste donc locale (M5 Max, macOS 26.5).
//! Les tests du registre de tuiles tournent partout, sans GPU.

use super::*;
use crate::memory_guard::{MemoryGuard, MemoryLimits};
use metal::CommandQueueRef;

const MIB: u64 = 1 << 20;

fn sparse_pool(test: &str) -> Option<(Device, SparseKvPool)> {
    let device = Device::system_default()?;
    if supports(&device).is_none() {
        eprintln!("{test} ignoré : placement sparse indisponible sur ce device");
        return None;
    }
    let pool = SparseKvPool::new(&device, MemoryGuard::disabled())
        .expect("invariant: supports() vient de répondre oui");
    Some((device, pool))
}

/// Preuve signalée par une file classique après ses commandes déjà soumises.
fn signal_after(queue: &CommandQueueRef, event: &metal::SharedEvent, value: u64) -> UseProof {
    let command_buffer = queue.new_command_buffer();
    command_buffer.encode_signal_event(event, value);
    command_buffer.commit();
    UseProof {
        event: event.clone(),
        value,
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn footprint(stage: &str) {
    let mib = crate::memory_guard::process_footprint_bytes().map(|bytes| bytes / MIB);
    eprintln!("phys_footprint {stage} : {mib:?} Mio");
}

#[test]
fn ledger_never_recycles_leaked_tiles() {
    let mut ledger = Ledger::<()>::new(4);
    ledger.add_heap(());
    let mut taken = Vec::new();
    ledger.take(3, &mut taken);
    assert_eq!(
        taken.iter().map(|tile| tile.index).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    // Empoisonnement simulé : deux tuiles au mapping incertain.
    ledger.leak(&taken[..2]);
    ledger.recycle(&taken[2..]);
    assert_eq!((ledger.free_tiles(), ledger.used_tiles()), (2, 2));
    assert_eq!(ledger.leaked_tiles, 2);
    let mut again = Vec::new();
    ledger.take(4, &mut again);
    assert_eq!(
        again.iter().map(|tile| tile.index).collect::<Vec<_>>(),
        [2, 3]
    );
    // Un heap qui porte une tuile perdue n'est jamais détruit ; les heaps
    // vides partent tant que la capacité libre dépasse un heap.
    ledger.recycle(&again);
    ledger.add_heap(());
    ledger.add_heap(());
    assert_eq!(ledger.drain_empty_heaps().len(), 2);
    assert_eq!(ledger.heaps().count(), 1);
    assert!(ledger.slots[0].as_ref().is_some_and(|heap| heap.used == 2));
    // Un seul heap libre reste disponible pour la croissance suivante.
    let mut idle = Ledger::<()>::new(4);
    idle.add_heap(());
    assert!(idle.drain_empty_heaps().is_empty());
}

#[test]
fn ledger_hands_over_only_heaps_with_leaked_tiles() {
    let mut ledger = Ledger::<u8>::new(2);
    ledger.add_heap(0);
    ledger.add_heap(1);
    ledger.add_heap(2);
    let mut taken = Vec::new();
    ledger.take(5, &mut taken);
    // Empoisonnement simulé d'une tuile du heap 2, recyclage du reste.
    assert_eq!(taken[4].slot, 2);
    ledger.leak(&taken[4..]);
    ledger.recycle(&taken[..4]);
    assert_eq!(ledger.take_leaked_heaps(), [2]);
    assert_eq!(ledger.heaps().count(), 2);
    assert!(ledger.take_leaked_heaps().is_empty());
}

#[test]
fn contiguous_runs_split_on_heap_or_gap() {
    let tile = |slot, index| Tile { slot, index };
    let tiles = [tile(0, 4), tile(0, 5), tile(0, 7), tile(1, 8), tile(1, 9)];
    let runs: Vec<_> = contiguous_runs(&tiles)
        .map(|(start, run)| (start, run.len()))
        .collect();
    assert_eq!(runs, [(0, 2), (2, 1), (3, 2)]);
}

#[test]
fn mapped_rows_take_gpu_writes_and_read_back() -> Result<()> {
    let Some((device, pool)) = sparse_pool("mapped_rows_take_gpu_writes_and_read_back") else {
        return Ok(());
    };
    let queue = device.new_command_queue();
    let mut staging = transfer::Staging::new(&device)?;
    let row_bytes = 512;
    let mut kv = pool.new_buffer(1 << 20, row_bytes)?;
    let tile = pool.stats()?.tile_bytes;
    assert_eq!(kv.buffer().length(), 512 * MIB);
    assert_eq!((kv.mapped_bytes(), kv.valid_rows()), (0, 0));
    assert_eq!(kv.capacity_rows(), 1 << 20);
    // Aucun pointeur CPU : l'identité passe par l'objet (`allocation_identity`).
    assert!(kv.buffer().contents().is_null());
    assert!(kv.ensure_rows((1 << 20) + 1).is_err());

    kv.ensure_rows(300)?;
    let tiles = (300 * row_bytes).div_ceil(tile);
    assert_eq!(kv.mapped_bytes(), tiles * tile);
    assert!(kv.mapped_rows() >= 300);
    let stats = pool.stats()?;
    assert_eq!(stats.virtual_bytes, 512 * MIB);
    assert_eq!(stats.mapped_tiles, tiles);
    assert_eq!(stats.heap_bytes, HEAP_BYTES);
    assert_eq!(stats.free_bytes, HEAP_BYTES - tiles * tile);
    assert!(kv.set_valid_rows(kv.mapped_rows() + 1).is_err());

    // Écriture GPU par blit (staging), relecture par blit.
    let data = pattern(300 * row_bytes as usize, 7);
    transfer::write_bytes(&queue, &mut staging, kv.buffer(), 0, &data)?;
    kv.set_valid_rows(300)?;
    let mut back = vec![0u8; data.len()];
    transfer::read_bytes(&queue, &mut staging, kv.buffer(), 0, &mut back)?;
    assert_eq!(back, data);

    // Écriture GPU par kernel, heaps déclarés sur l'encodeur.
    let library = device
        .new_library_with_source(
            "kernel void fill(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = i * 3u + 1u; }",
            &metal::CompileOptions::new(),
        )
        .map_err(InferError::Metal)?;
    let function = library
        .get_function("fill", None)
        .map_err(InferError::Metal)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(InferError::Metal)?;
    let words = 1000u64;
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    kv.use_heaps(encoder)?;
    encoder.set_buffer(0, Some(kv.buffer()), 4096);
    encoder.dispatch_threads(MTLSize::new(words, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    commit_and_wait(command_buffer)?;
    let mut words_back = vec![0u8; words as usize * 4];
    transfer::read_bytes(&queue, &mut staging, kv.buffer(), 4096, &mut words_back)?;
    for (index, word) in words_back.chunks_exact(4).enumerate() {
        let value = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        assert_eq!(value, index as u32 * 3 + 1);
    }

    // Un transfert hors du buffer virtuel est refusé.
    let mut past = [0u8; 8];
    assert!(
        transfer::read_bytes(&queue, &mut staging, kv.buffer(), 512 * MIB - 4, &mut past).is_err()
    );

    let event = device.new_shared_event();
    kv.record_use(signal_after(&queue, &event, 1));
    drop(kv);
    let stats = pool.stats()?;
    assert_eq!(
        (stats.virtual_bytes, stats.mapped_tiles, stats.leaked_tiles),
        (0, 0, 0)
    );
    Ok(())
}

#[test]
fn growth_spans_several_heaps() -> Result<()> {
    let Some((device, pool)) = sparse_pool("growth_spans_several_heaps") else {
        return Ok(());
    };
    let queue = device.new_command_queue();
    let mut staging = transfer::Staging::new(&device)?;
    let mut kv = pool.new_buffer(4096, MIB)?;
    kv.ensure_rows(100)?;
    assert_eq!(pool.stats()?.heap_bytes, HEAP_BYTES);
    // 300 Mio : le premier heap ne suffit plus.
    kv.ensure_rows(300)?;
    let stats = pool.stats()?;
    assert_eq!(stats.heap_bytes, 2 * HEAP_BYTES);
    assert_eq!(stats.mapped_tiles * stats.tile_bytes, 300 * MIB);
    for row in [0u64, 255, 256, 299] {
        let data = pattern(4096, row as u8);
        transfer::write_bytes(&queue, &mut staging, kv.buffer(), row * MIB, &data)?;
        let mut back = vec![0u8; data.len()];
        transfer::read_bytes(&queue, &mut staging, kv.buffer(), row * MIB, &mut back)?;
        assert_eq!(back, data, "ligne {row}");
    }
    let event = device.new_shared_event();
    kv.record_use(signal_after(&queue, &event, 1));
    drop(kv);
    // Capacité libre au-delà d'un heap : le heap vide en trop est détruit.
    let stats = pool.stats()?;
    assert_eq!(
        (stats.heap_bytes, stats.free_bytes),
        (HEAP_BYTES, HEAP_BYTES)
    );
    Ok(())
}

#[test]
fn release_keeps_partial_tile_and_recycles_the_rest() -> Result<()> {
    let Some((device, pool)) = sparse_pool("release_keeps_partial_tile_and_recycles_the_rest")
    else {
        return Ok(());
    };
    let queue = device.new_command_queue();
    let mut staging = transfer::Staging::new(&device)?;
    let tile = pool.stats()?.tile_bytes;
    // Lignes de 1000 octets : les frontières de lignes et de tuiles diffèrent.
    let mut kv = pool.new_buffer(10_000, 1000)?;
    kv.ensure_rows(200)?;
    let mapped = 200_000u64.div_ceil(tile);
    assert_eq!(kv.mapped_bytes(), mapped * tile);
    let data = pattern(70_000, 3);
    transfer::write_bytes(&queue, &mut staging, kv.buffer(), 0, &data)?;
    kv.set_valid_rows(200)?;

    let event = device.new_shared_event();
    kv.release_from(70, signal_after(&queue, &event, 1))?;
    let kept = 70_000u64.div_ceil(tile);
    assert_eq!(kv.mapped_bytes(), kept * tile);
    assert_eq!(kv.valid_rows(), 70);
    let stats = pool.stats()?;
    assert_eq!(stats.mapped_tiles, kept);
    assert_eq!(stats.free_bytes, HEAP_BYTES - kept * tile);
    // La tuile partielle gardée porte toujours ses lignes.
    let mut back = vec![0u8; data.len()];
    transfer::read_bytes(&queue, &mut staging, kv.buffer(), 0, &mut back)?;
    assert_eq!(back, data);

    // Les tuiles rendues servent un autre buffer sans nouveau heap.
    let mut other = pool.new_buffer(10_000, 1000)?;
    other.ensure_rows(200)?;
    assert_eq!(pool.stats()?.heap_bytes, HEAP_BYTES);
    other.record_use(signal_after(&queue, &event, 2));
    kv.record_use(signal_after(&queue, &event, 3));
    drop((kv, other));
    let stats = pool.stats()?;
    assert_eq!((stats.mapped_tiles, stats.leaked_tiles), (0, 0));
    assert_eq!(stats.free_bytes, HEAP_BYTES);
    Ok(())
}

#[test]
fn missing_or_late_proof_never_recycles() -> Result<()> {
    let Some((device, pool)) = sparse_pool("missing_or_late_proof_never_recycles") else {
        return Ok(());
    };
    pool.set_wait_timeout_ms(20)?;
    let mut kv = pool.new_buffer(1000, MIB)?;
    kv.ensure_rows(4)?;
    let mapped = pool.stats()?.mapped_tiles;
    // Preuve jamais signalée : attente bornée, rien n'est rendu.
    let event = device.new_shared_event();
    let late = UseProof {
        event: event.clone(),
        value: 5,
    };
    assert!(kv.release_from(0, late).is_err());
    assert!(kv.is_poisoned());
    assert_eq!(pool.stats()?.mapped_tiles, mapped);
    drop(kv);
    let stats = pool.stats()?;
    assert_eq!(stats.leaked_tiles, mapped);
    assert_eq!(stats.mapped_tiles, mapped);
    assert_eq!(stats.free_bytes, HEAP_BYTES - mapped * stats.tile_bytes);

    // Drop sans preuve connue : tuiles et buffer perdus, jamais recyclés.
    let mut silent = pool.new_buffer(1000, MIB)?;
    silent.ensure_rows(1)?;
    let more = pool.stats()?.mapped_tiles - mapped;
    drop(silent);
    assert_eq!(pool.stats()?.leaked_tiles, mapped + more);
    Ok(())
}

#[test]
fn expired_proof_overrides_an_older_completed_one() -> Result<()> {
    let Some((device, pool)) = sparse_pool("expired_proof_overrides_an_older_completed_one") else {
        return Ok(());
    };
    pool.set_wait_timeout_ms(20)?;
    let queue = device.new_command_queue();
    let event = device.new_shared_event();
    let mut kv = pool.new_buffer(1000, MIB)?;
    kv.ensure_rows(4)?;
    let mapped = pool.stats()?.mapped_tiles;
    // Preuve 1 atteinte et retenue, puis un usage dont la preuve 2 expire.
    let first = signal_after(&queue, &event, 1);
    assert!(wait_signaled(&first.event, 1, 5_000));
    kv.record_use(first);
    let second = UseProof {
        event: event.clone(),
        value: 2,
    };
    assert!(kv.release_from(1, second).is_err());
    assert!(kv.is_poisoned());
    assert_eq!(kv.valid_rows(), 0);
    // Le drop ne se rabat pas sur la preuve 1 : tout est perdu.
    drop(kv);
    let stats = pool.stats()?;
    assert_eq!((stats.leaked_tiles, stats.mapped_tiles), (mapped, mapped));
    Ok(())
}

#[test]
#[expect(unsafe_code, reason = "retainCount du heap observé par le test")]
fn last_pool_keeps_heaps_with_leaked_tiles() -> Result<()> {
    let Some((_device, pool)) = sparse_pool("last_pool_keeps_heaps_with_leaked_tiles") else {
        return Ok(());
    };
    let mut kv = pool.new_buffer(1000, MIB)?;
    kv.ensure_rows(2)?;
    let heap = {
        let state = pool.shared.lock()?;
        let slot = state
            .ledger
            .heaps()
            .next()
            .expect("invariant: un heap vient d'être créé");
        slot.heap.heap.clone()
    };
    drop(kv);
    drop(pool);
    // SAFETY: `heap` est un `id<MTLHeap>` vivant ; `retainCount` rend un
    // `NSUInteger`.
    let retained =
        unsafe { send::<_, NSUInteger>(heap.as_ptr().cast::<Object>(), "retainCount", ())? };
    // Notre clone plus la référence oubliée par le pool détruit.
    assert!(retained >= 2, "retainCount {retained}");
    Ok(())
}

#[test]
fn transfers_span_several_staging_ranges() -> Result<()> {
    let Some((device, pool)) = sparse_pool("transfers_span_several_staging_ranges") else {
        return Ok(());
    };
    let queue = device.new_command_queue();
    let mut staging = transfer::Staging::new(&device)?;
    let mut kv = pool.new_buffer(16, MIB)?;
    kv.ensure_rows(16)?;
    // 9 Mio + 123 octets à un offset non aligné : trois plages de staging.
    let data = pattern(9 * MIB as usize + 123, 11);
    transfer::write_bytes(&queue, &mut staging, kv.buffer(), 4096 + 7, &data)?;
    let mut back = vec![0u8; data.len()];
    transfer::read_bytes(&queue, &mut staging, kv.buffer(), 4096 + 7, &mut back)?;
    assert_eq!(back, data);
    let event = device.new_shared_event();
    kv.record_use(signal_after(&queue, &event, 1));
    Ok(())
}

#[test]
fn poisoned_buffer_refuses_work_and_leaks() -> Result<()> {
    let Some((_device, pool)) = sparse_pool("poisoned_buffer_refuses_work_and_leaks") else {
        return Ok(());
    };
    let mut kv = pool.new_buffer(1000, MIB)?;
    kv.ensure_rows(2)?;
    let mapped = pool.stats()?.mapped_tiles;
    // Injection : comme après un mapping non confirmé.
    {
        let shared = Arc::clone(&kv.pool);
        let mut state = shared.lock()?;
        kv.poison(&mut state);
    }
    assert!(kv.ensure_rows(3).is_err());
    drop(kv);
    let stats = pool.stats()?;
    assert_eq!((stats.leaked_tiles, stats.mapped_tiles), (mapped, mapped));
    Ok(())
}

#[test]
fn refused_budget_is_a_typed_error() -> Result<()> {
    let Some(device) = Device::system_default() else {
        return Ok(());
    };
    if supports(&device).is_none() {
        eprintln!("refused_budget_is_a_typed_error ignoré : placement sparse indisponible");
        return Ok(());
    }
    let limits = MemoryLimits {
        static_cap: Some(MIB),
        host_memory: None,
        headroom_bytes: 0,
        metal_cap: None,
    };
    let guard = MemoryGuard::from_limits(true, limits, Arc::new(|| Some(0)));
    let pool = SparseKvPool::new(&device, guard)?;
    let mut kv = pool.new_buffer(1000, MIB)?;
    assert!(matches!(kv.ensure_rows(1), Err(InferError::MemoryGuard(_))));
    assert!(!kv.is_poisoned());
    let stats = pool.stats()?;
    assert_eq!((stats.heap_bytes, stats.mapped_tiles), (0, 0));
    Ok(())
}

#[test]
fn second_heap_is_refused_while_the_first_is_unseen() -> Result<()> {
    let Some(device) = Device::system_default() else {
        return Ok(());
    };
    if supports(&device).is_none() {
        eprintln!("second_heap_is_refused_while_the_first_is_unseen ignoré : placement sparse indisponible");
        return Ok(());
    }
    // Plafond d'un heap et demi, footprint simulé nul : seule la réservation
    // gardée par le premier heap peut refuser le second.
    let limits = MemoryLimits {
        static_cap: Some(HEAP_BYTES + HEAP_BYTES / 2),
        host_memory: None,
        headroom_bytes: 0,
        metal_cap: None,
    };
    let guard = MemoryGuard::from_limits(true, limits, Arc::new(|| Some(0)));
    let pool = SparseKvPool::new(&device, guard)?;
    let mut kv = pool.new_buffer(1000, MIB)?;
    kv.ensure_rows(100)?;
    let mapped = pool.stats()?.mapped_tiles;
    assert!(matches!(
        kv.ensure_rows(300),
        Err(InferError::MemoryGuard(_))
    ));
    assert!(!kv.is_poisoned());
    let stats = pool.stats()?;
    assert_eq!((stats.heap_bytes, stats.mapped_tiles), (HEAP_BYTES, mapped));
    let queue = device.new_command_queue();
    let event = device.new_shared_event();
    kv.record_use(signal_after(&queue, &event, 1));
    Ok(())
}

/// Mesure informative : `phys_footprint` aux étapes du cycle de vie. Aucune
/// assertion, les autres tests tournent en parallèle dans le même process
/// (`--test-threads=1 --nocapture` pour une lecture propre).
#[test]
fn footprint_through_lifecycle_is_reported() -> Result<()> {
    let Some((device, pool)) = sparse_pool("footprint_through_lifecycle_is_reported") else {
        return Ok(());
    };
    let queue = device.new_command_queue();
    let mut staging = transfer::Staging::new(&device)?;
    footprint("avant");
    // Les command buffers sont autoreleased et retiennent le buffer sparse :
    // sans pool d'autorelease drainé, ni lui ni ses heaps ne partent.
    metal::objc::rc::autoreleasepool(|| -> Result<()> {
        let mut kv = pool.new_buffer(4096, MIB)?;
        footprint("buffer virtuel 4 Gio");
        kv.ensure_rows(400)?;
        footprint("400 Mio mappés (2 heaps)");
        let data = vec![1u8; MIB as usize];
        for row in 0..400 {
            transfer::write_bytes(&queue, &mut staging, kv.buffer(), row * MIB, &data)?;
        }
        footprint("400 Mio écrits");
        let event = device.new_shared_event();
        kv.release_from(0, signal_after(&queue, &event, 1))?;
        footprint("unmap, un heap détruit, pool d'autorelease non drainé");
        drop(kv);
        Ok(())
    })?;
    footprint("pool d'autorelease drainé (un heap libre gardé)");
    drop(pool);
    footprint("pool K/V détruit");
    std::thread::sleep(std::time::Duration::from_millis(500));
    footprint("pool K/V détruit + 500 ms");
    Ok(())
}
