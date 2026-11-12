# Historique MTP bf16 partagé

Le semis committed conserve ses blocs de 1024 lignes, ses normes, projections
et RoPE f32. Seule sa sortie change : chaque bloc est converti avec le même
`bfloat(input[i])` que le semis du KV résident. Le buffer final bf16 réserve
`prompt + max_new_tokens + draft_headroom` lignes ; les scratch f32 restent
bornés au bloc. À 261 000 tokens et `kv_dim=1024`, l'ancien historique f32
occupait 2 138 103 808 octets (K et V, `prompt - 1` lignes).

Le cache de préfixe partage les segments immuables. Le setup MTP adopte leurs
buffers avec le même claim exclusif que le tronc, sans réécrire le préfixe.
Capacité insuffisante, vue tronquée, autre écrivain ou stockage non adoptable
(notamment un instantané disque possédé) entraînent une copie. Les extensions
ajoutent des segments bf16 ; l'arène peut adopter le premier segment et semer
uniquement la queue lorsque sa réserve suffit.

Le contrat f32 reste le défaut : CPU, résident désactivé, tête non supportée,
échantillonnage sans résident ou `SARAGOSSA_RUST_KV_BF16=0`. Une entrée bf16
n'est pas réutilisée par un draft exigeant f32. Si le setup refuse le résident
après le semis bf16, le prefill est rejoué en f32 sans réémettre le premier
token. Le refus de copie sous claim occupé suit le même repli que le tronc.
Les états déjà installés, l'historique local et les entrées bf16 du cache de
préfixe sont libérés avant reconstruction ; l'état final reconstruit accompagne
le nouvel historique f32. Ce repli peut coûter un second prefill. Il ne récupère
pas les bits perdus par élargissement. La clé des instantanés inclut la version du contrat
de stockage ; les formats de blobs f32/bf16 restent inchangés.

Une ancienne entrée f32 reste utilisable en résident bf16 par copie. Le gain
maximal concerne le semis intégré bf16 avec adoption ; il ne s'applique pas
aux copies de repli, ni aux lecteurs qui retiennent encore une ancienne
allocation. L'oracle d'append garde son allocation distincte lorsqu'activé.

## Vérifications sans GPU

Toutes les commandes restent hors ligne. La suite complète contient des tests
Metal non ignorés : `--no-run` compile tout, la commande avec `--exact`
n'exécute que les tests CPU identifiés.

```sh
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo fmt --all -- --check
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo clippy --all-targets --all-features -- -D warnings
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo build
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo test --all-features --lib --no-run
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo test --all-features --lib -- --exact \
  decoder::mtp_history::tests::storage_scope_restores_and_rejects_rounded_history_for_f32 \
  decoder::tests::mtp_bf16_prefix_compatibility_and_shared_accounting_cpu \
  decoder::tests::mtp_f32_fallback_discards_all_bf16_prefix_references_cpu \
  decoder::state_snapshot::tests::roundtrip_and_reject_corruption_without_weights \
  decoder::kv_rows::tests::compact_snapshot_preserves_mixed_rows_and_base_without_reserve \
  decoder::kv_rows::tests::snapshot_shares_prefix_and_truncate_copies_only_split_segment \
  decoder::kv_rows::tests::bf16_segments_widen_count_two_bytes_and_stay_unrounded_on_append \
  decoder::tests::prefix_cache_byte_cap_counts_shared_segments_once \
  decoder::tests::prefix_cache_put_exact_replacement_keeps_metadata \
  decoder::tests::prefix_cache_extends_append_suffix_with_batched_prefill
```

## Validations Metal à lancer explicitement

Ces tests sont ignorés par défaut. Les deux premiers n'ouvrent aucun poids :
ils comparent les octets bf16 du nouveau semis à l'ancien historique f32 semé
dans une arène bf16, contrôlent l'identité des allocations et le repli par
copie sous contention. Ils couvrent les morceaux non alignés, la queue de
903 lignes et les scratch privés.

```sh
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo test --all-features --lib \
  metal_backend::prefill::tests::mtp_seed_bf16_streamed_bit_exact_and_adopted \
  -- --ignored --exact --nocapture
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo test --all-features --lib \
  metal_backend::prefill::tests::mtp_seed_bf16_streamed_32768_bit_exact \
  -- --ignored --exact --nocapture
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true \
  SARAGOSSA_RUST_MTP_HISTORY=committed SARAGOSSA_RUST_PREFIX_CACHE=1 \
  SARAGOSSA_RUST_DECODE_RESIDENT_FULL=1 SARAGOSSA_RUST_KV_BF16=1 \
  SARAGOSSA_RUST_MTP_LOOKUP_DRAFT=0 SARAGOSSA_RUST_MTP_ADAPTIVE_DEPTH=0 \
  cargo test --all-features --lib \
  decoder::tests::mtp_bf16_history_generation_adoption_and_f32_fallback \
  -- --ignored --exact --nocapture
```

Le troisième test utilise un modèle synthétique : ids AR/MTP D1/D2 à froid,
préfixe exact et extension, buffers partagés avec le cache, écriture exclusive,
copie concurrente et refus d'un préfixe bf16 lors du retour au f32 forcé.

Les deux tests suivants couvrent le refus après semis bf16 : claim MTP
occupé avec budget de copie épuisé et nettoyage du tronc ; reconstruction f32
à froid et après extension, état final/historique cohérents, ids AR/MTP D1/D2
et émission unique du premier token. Ils utilisent uniquement des données
synthétiques et restent ignorés par défaut.

```sh
rtk proxy env CARGO_NET_OFFLINE=true \
  SARAGOSSA_RUST_MTP_HISTORY=committed SARAGOSSA_RUST_PREFIX_CACHE=1 \
  SARAGOSSA_RUST_DECODE_RESIDENT_FULL=1 SARAGOSSA_RUST_KV_BF16=1 \
  SARAGOSSA_RUST_PREFILL_KV_SHARED=1 SARAGOSSA_RUST_MTP_PREFILL_RESIDENT=1 \
  SARAGOSSA_RUST_MTP_LOOKUP_DRAFT=0 SARAGOSSA_RUST_MTP_ADAPTIVE_DEPTH=0 \
  SARAGOSSA_RUST_MTP_EARLY_FIRST_TOKEN=1 \
  SARAGOSSA_RUST_MTP_APPEND_KV_ORACLE=0 \
  cargo test --all-features --lib -- --ignored --exact --nocapture \
  decoder::tests::mtp_bf16_busy_claim_memory_guard_cleans_setup \
  decoder::tests::mtp_bf16_post_seed_refusal_rebuilds_f32_and_streams_once
```

## Modèle local, ids greedy et mémoire

Fournir les deux chemins locaux via `MODEL_DIR` et `MTP_SIDECAR`. Aucun
téléchargement n'est effectué. Compiler avant de mesurer pour exclure le
compilateur du pic de mémoire. Le test imprime les ids et les octets réservés
par l'historique, puis exige AR = MTP à froid et à chaud. Par défaut, il répète
un texte technique jusqu'au nombre exact de tokens ;
`SARAGOSSA_MTP_VALIDATION_PROMPT` permet de fournir un fichier texte local.

```sh
rtk proxy taskpolicy -b env CARGO_NET_OFFLINE=true cargo test --release --all-features --lib --no-run
for tokens in 4096 32768 262144; do
  for depth in 1 2; do
    rtk proxy /usr/bin/time -l taskpolicy -b env CARGO_NET_OFFLINE=true \
      SARAGOSSA_RUST_MTP_VERIFIER_TEST_MODEL="$MODEL_DIR" \
      SARAGOSSA_RUST_MTP_SIDECAR="$MTP_SIDECAR" \
      SARAGOSSA_REQUIRE_REAL_MODELS=1 \
      SARAGOSSA_RUST_MTP_HISTORY=committed SARAGOSSA_RUST_PREFIX_CACHE=1 \
      SARAGOSSA_RUST_DECODE_RESIDENT_FULL=1 \
      SARAGOSSA_RUST_DECODE_RESIDENT_FULL_LINEAR=1 SARAGOSSA_RUST_GPU_ARGMAX=1 \
      SARAGOSSA_RUST_PREFILL_KV_SHARED=1 SARAGOSSA_RUST_MTP_PREFILL_RESIDENT=1 \
      SARAGOSSA_RUST_KV_BF16=1 SARAGOSSA_RUST_MTP_LOOKUP_DRAFT=0 \
      SARAGOSSA_RUST_MTP_ADAPTIVE_DEPTH=0 \
      SARAGOSSA_RUST_MTP_APPEND_KV_ORACLE=0 \
      SARAGOSSA_MTP_VALIDATION_TOKENS="$tokens" \
      SARAGOSSA_MTP_VALIDATION_DEPTH="$depth" \
      cargo test --release --all-features --lib \
      decoder::tests::mtp_bf16_real_greedy_ids_and_memory \
      -- --ignored --exact --nocapture
  done
done
```

Relancer avec `SARAGOSSA_RUST_MTP_ADAPTIVE_DEPTH=1` pour l'adaptatif et,
séparément, `SARAGOSSA_RUST_MTP_APPEND_KV_ORACLE=1` pour l'oracle d'append.
Limiter la taille à la fenêtre réellement supportée par le modèle.

Le pic RSS de `time -l` couvre le processus entier (poids, tronc, scratch,
AR puis MTP) : ce n'est pas la taille du seul historique. Les octets réservés
et les pointeurs sont vérifiés par les tests ; le gain physique et les temps
doivent être mesurés sur le matériel cible. Aucune identité numérique GPU ni
économie de RSS n'est déduite des seuls tests CPU.
