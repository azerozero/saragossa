# Qualification du prefill Gemma 4

Le chemin per-op sert de référence. L'activation du résident par défaut exige
l'identité des 200 tokens greedy après les prompts 1k et 4k, l'identité du
decode sur 200 tokens et le passage de l'oracle chat KL existant. Aucun seuil
de cet oracle ne doit être modifié pour qualifier le résident.

Le résident reste **désactivé par défaut** : sur q4a8-ls, le prompt 1k du
corpus versionné diverge au token 161 (index 160) sur 200 tokens générés.
À 4109 tokens, le prefill résident termine sans repli ; la génération diverge
au token 62 (index 61). Ces deux tests restent stricts et échouent tant que
l'identité n'est pas acquise.
Le decode seul est identique sur 200 tokens. L'oracle KL sur le snapshot
4bit passe avec le résident : ratio KL 0,09, ratio p99 Δp 0,34 et top-1
99,61 % (référence bf16 : 98,83 %). Cette qualité ne remplace pas l'identité
greedy exigée pour changer le défaut.

Le kernel causal long couvre désormais les têtes globales d512 : chaque
thread accumule deux colonnes indépendantes. Cette variante est séparée du
kernel historique d128/d256. Elle matérialise le score et le poids en f32
comme le kernel mid pour éviter les contractions de fast-math à ces
frontières ; leur égalité en bits est testée à 256 et 257 tokens. Les oracles
CPU conservent le seuil existant de 5e-4 aux longueurs 2048, 2049 et 4109.
Les erreurs d'exécution du prefill résident Gemma 4 remontent à l'appelant ; un échec ne relance plus
silencieusement le prefill per-op. Le coupe-circuit reste
`SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4=0`.

## Oracles sous verrou GPU

Compiler avant de prendre le verrou :

```sh
cargo test --release --lib --no-run
cargo build --release
```

Définir `SARAGOSSA_GEMMA4_DIR` vers le checkpoint q4a8-ls local pour les tests
d'identité. Ces tests ignorés échouent si le modèle ou le chemin résident
manque. Ils exécutent chaque prefill une fois, sans instantané ni cache de
préfixe, puis exactement 200 tokens sans arrêt anticipé sur EOS. Les prompts
sont construits à partir du corpus KL versionné, tronqué à 1024/4096 tokens
avant l'enveloppe chat. Le decode est comparé séparément au per-op à partir
du même cache et du même état final de prefill.

```sh
python3 -I -B tools/gpulock.py run --owner gemma4-kernel --prio 10 -- \
  cargo test --release --lib causal_attention_prefill_long_d512 \
  -- --nocapture --test-threads=1

python3 -I -B tools/gpulock.py run --owner gemma4-error --prio 10 -- \
  cargo test --release --lib gemma4_resident_prefill_propagates_execution_error \
  -- --nocapture --test-threads=1

python3 -I -B tools/gpulock.py run --owner gemma4-identity --prio 10 -- \
  env SARAGOSSA_RUST_PREFILL_RESIDENT=1 SARAGOSSA_RUST_PREFILL_CHUNK=0 \
  SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0 \
  cargo test --release --lib decoder::tests::gemma4_oracles \
  -- --ignored --nocapture --test-threads=1
```

Pour KL, définir `SARAGOSSA_GEMMA4_DIR` vers le snapshot **4bit de référence**
identifié dans `tests/fixtures/goldens/kld/gemma4.json`. Le q4a8-ls n'a pas la
même provenance : ne pas contourner le contrôle SHA pour l'utiliser ici.
Exécuter les deux configurations dans des processus distincts :

```sh
for resident in 0 1; do
  python3 -I -B tools/gpulock.py run --owner gemma4-kl --prio 10 -- \
    env SARAGOSSA_REQUIRE_REAL_MODELS=1 \
    SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4="$resident" \
    cargo test --release --lib golden::kld::external_kld_gemma4 \
    -- --ignored --exact --nocapture --test-threads=1
done
```

## Mesure A/B à la demande

Reprendre le générateur CLI des sondes kbench (`--prompt-tokens`) et le même
checkpoint q4a8-ls. Désactiver les caches de préfixe et instantanés éventuels
dans l'environnement. Une passe par configuration suffit pour vérifier le
chemin ; les répétitions et l'entrelacement appartiennent à la mesure.

```sh
for tokens in 1024 4096; do
  for resident in 0 1; do
    python3 -I -B tools/gpulock.py run --owner gemma4-ab --prio 10 -- \
      env SARAGOSSA_RUST_STATE_SNAPSHOT= SARAGOSSA_RUST_PREFIX_CACHE=0 \
      SARAGOSSA_RUST_PREFILL_RESIDENT=1 \
      SARAGOSSA_RUST_PREFILL_RESIDENT_GEMMA4="$resident" \
      SARAGOSSA_RUST_TRACE_PREFILL=1 \
      target/release/saragossa --model-dir "$SARAGOSSA_GEMMA4_DIR" \
      --backend metal --prompt-tokens "$tokens" --prefill-only --prefill-repeat 1
  done
done
```

Les durées imprimées par les oracles ne constituent pas un benchmark : elles
incluent les premières allocations et la préparation des pipelines.
