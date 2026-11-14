# Attention prefill causale NAX d128/d256

Le prefill sans sinks accepte tout GQA entier positif (`q_heads % kv_heads == 0`)
en d128 et d256. Le bloc complet utilise NAX au-delà de 256 requêtes ; un
morceau avec préfixe accepte aussi une queue courte. Les fenêtres restent hors
périmètre. La spécialisation d128 fixe 301=true, 302=false et 305=false, sans
buffer de sinks. Sa grille reste `(NQ, H)` : le flag `NAX_GRID` ne concerne que
le dsplit d256.

Les kernels naïfs restent disponibles. `SARAGOSSA_RUST_PREFILL_ATTN_NAX=0`
revient au naïf en d128 et au Steel f32 en d256.
`SARAGOSSA_RUST_PREFILL_ATTN_STEEL_D256=0` désactive les deux dimensions malgré
son nom historique. Si aucun pipeline adapté n'est disponible, le découpage
est désactivé et le bloc complet retrouve son repli naïf.
`SARAGOSSA_RUST_PREFILL_RESIDENT_CHUNK=0` désactive seulement le découpage.
Le miroir et le stockage KV bf16 vérifient le lecteur de chaque couche.

## Oracles

Le seuil est celui du NAX d256 existant, **1e-3**, sans assouplissement. Les
entrées des trois références sont arrondies en bf16, comme dans les oracles
NAX existants, pour isoler l'accumulation et l'arrondi des probabilités. Le
naïf est comparé au NAX sur toute la sortie et les deux sont comparés au CPU
sur des lignes choisies, dont les frontières des tuiles et la dernière ligne.
Les valeurs non finies échouent explicitement.

- Court : d128 32/8, 32/4, 64/8, 15/5, 7/1 et 3/3 à 257 tokens ; d256
  32/8, 32/4 et 64/8 à 319 tokens.
- Long : d128 et d256, 32/8, 32/4, 64/8, longueurs 273, 1069, 2049, 4097,
  8191 et 8192. Une seule exécution par cas, sans répétitions de benchmark.
- Morceaux : queues 1/31/32/33/63/64/65, préfixes non alignés, capacité KV
  supérieure aux lignes valides, miroir bf16 bit-identique à la reconversion.
- Intégration résidente : morceaux 367/367/366, prolongements, dernière ligne
  seule et KV stocké en bf16, avec les assertions existantes inchangées.
- Repli : pipeline d128 absent et chacun des deux kill-switches.

Depuis la racine du dépôt, définir `GPU_LOCK` sur **le verrou partagé avec les
autres utilisateurs du GPU**, puis compiler avant de le prendre :

```sh
rtk cargo test --release --offline --lib --no-run

gpu() {
  rtk proxy python3 -I -B tools/gpulock.py run \
    --owner attention-validation --prio 5 -- "$@"
}

gpu cargo test --release --offline --lib causal_attention_generic_ \
  -- --nocapture --test-threads=1

gpu cargo test --release --offline --lib \
  metal_backend::tests::causal_attention_generic_nax_long_oracle \
  -- --ignored --exact --nocapture --test-threads=1

gpu cargo test --release --offline --lib \
  prefill_resident_d128_chunked_matches_single_chunk -- --nocapture --test-threads=1

gpu env SARAGOSSA_RUST_PREFILL_ATTN_NAX=0 \
  cargo test --release --offline --lib \
  metal_backend::tests::causal_attention_generic_kill_switch \
  -- --ignored --exact --nocapture --test-threads=1

gpu env SARAGOSSA_RUST_PREFILL_ATTN_STEEL_D256=0 \
  cargo test --release --offline --lib \
  metal_backend::tests::causal_attention_generic_kill_switch \
  -- --ignored --exact --nocapture --test-threads=1
```

L'oracle long et les tests des kill-switches échouent si Metal est absent ;
le long échoue aussi si NAX n'a pas compilé. Les tests courts conservent la
convention des tests Metal portables et peuvent être sautés sans périphérique.

Qualification du 14 novembre 2026 : les 45 cas synthétiques courts et longs
passent sur GPU, écart maximal NAX/naïf 3,30e-4. Les oracles CPU, les variantes
d256 et les deux kill-switches passent. Les sorties des prefills résidents
par morceaux d128 et d256 sont bit-identiques au bloc complet. La sonde A/B
ci-dessous est préparée, mais aucune campagne de mesure n'a été lancée.

## Mesure A/B explicite

La sonde reprend les entrées f32 et la médiane de `commit_and_wait` de
`kb_attn_d128`. Elle retire le pipeline d128 pour A (sélecteur naïf de production),
le rétablit pour B (conversions Q/K/V comprises), puis vérifie l'écart sur toute
la sortie avec le même seuil 1e-3. Elle couvre 32/8, 32/4, 64/8 à 2048, 4096
et 8192 tokens. Les sorties sont vérifiées finies. Aucun chiffre de vitesse
n'est imposé par le test.

```sh
gpu env KB_REPS=7 cargo test --release --offline --lib \
  metal_backend::tests::causal_attention_generic_nax_ab \
  -- --ignored --exact --nocapture --test-threads=1
```

Ne pas définir les kill-switches pour cette mesure. `KB_REPS=1` réduit la
sonde à une passe par chemin ; la référence naïve 8k reste lente.

## Identité greedy sur modèles locaux

Pas de modèle local d128 sans sinks lors de la qualification : l'identité
greedy d128 reste **non vérifiée**. Les oracles synthétiques ne la remplacent
pas. Les formes locales d256 24/4 et 16/2 utilisent déjà NAX avant ce changement.
Qualification avant/après la base `59bd5d8` : Qwen3.8-27B-u4ls (24/4/256) et
Qwen3.6-35B-A3B-4bit (16/2/256), prompts effectifs de 2060 et 8204 tokens,
16/16 tokens greedy identiques pour les quatre cas ; logits identiques en bits.
Cette vérification courte ne constitue pas une garantie sur tous les prompts.

Comparer le binaire de base au binaire candidat, avec les mêmes flags, les
mêmes poids et les mêmes prompts. `--kl-record` enregistre de vrais tokens
greedy, sans sampling ni exception de quasi-égalité.

Définir `BASELINE`, `CANDIDATE` et `MODEL` sur les chemins locaux, puis répéter
pour chaque modèle concerné. Les captures restent hors du dépôt public :

```sh
rtk cargo build --release --offline
OUT=$(mktemp -d)
export OUT
for n in 2048 8192; do
  gpu env SARAGOSSA_RUST_MTP_DECODE=0 "$BASELINE" \
    --model-dir "$MODEL" --prompt-tokens "$n" --max-tokens 16 \
    --kl-record "$OUT/before-$n"
  gpu env SARAGOSSA_RUST_MTP_DECODE=0 "$CANDIDATE" \
    --model-dir "$MODEL" --prompt-tokens "$n" --max-tokens 16 \
    --kl-record "$OUT/after-$n"
done
rtk proxy python3 - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ['OUT'])
for n in (2048, 8192):
    a, b = [json.loads((root / f'{label}-{n}' / 'ref.json').read_text())
            for label in ('before', 'after')]
    assert a['prompt_ids'] == b['prompt_ids'], 'prompts différents'
    assert len(a['forced_ids']) == len(b['forced_ids']) == 16
    assert a['forced_ids'] == b['forced_ids'], 'tokens greedy différents'
    print(n, 'greedy identique : 16/16')
PY
```

Le template peut ajouter quelques tokens aux longueurs demandées ; comparer
les `prompt_ids` garantit que les deux exécutions ont reçu la même séquence.
