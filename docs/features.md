# Features de compilation

Le cœur LLM (chargement, tokenizer, génération et caches) est toujours compilé.
Toutes les features ci-dessous sont dans `default` : une compilation habituelle
conserve les capacités et les défauts existants. Les modules restent à leur
emplacement ; les gardes portent sur leurs entrées et leurs dépendances.

| Feature | Périmètre | Dépendances optionnelles directes |
|---|---|---|
| `metal` | Backend GPU Apple Silicon | `dep:metal` (macOS) |
| `devtools` | Diagnostics, conversion, oracles et draft DFlash | Aucune |
| `stt` | Whisper, transcription et sessions incrémentales | `dep:flate2`, `dep:rustfft`, `dep:hound` |
| `tts` | Talker Qwen3, codec, Mimi, clonage et voix | `dep:rustfft`, `dep:hound` |
| `vision` | Entrées internes image/embeddings et oracles M-RoPE | Aucune |
| `denoise` | Nom réservé ; aucun moteur de débruitage dans cette version | Aucune |
| `embed` | Embedder texte e5-small | Aucune |
| `serve` | Serveur HTTP/Unix et commandes `serve`, `bench-serve` | Aucune |
| `decision` | Têtes de décision, lectures empaquetées et classifieurs | Aucune |

Le classifieur spéculatif sur embeddings nécessite `decision` **et** `embed`.
Sans lui, le routage heuristique du cœur LLM reste disponible. `serve` n'active
aucune modalité implicitement. Une route STT/TTS/embeddings non compilée répond
HTTP **501** avec la feature à activer, y compris pour un corps invalide ; une
modalité compilée mais non configurée garde son erreur **400** habituelle.
Les commandes absentes n'apparaissent pas dans l'aide CLI.

`vision` ne livre pas de transport image public ni d'encodeur complet : les
requêtes média restent refusées comme auparavant. Les validations des en-têtes
multimodaux et les métadonnées de cache restent dans le cœur pour charger les
modèles texte et reconnaître les snapshots sans réinterpréter leur format.
Les pipelines Metal génériques restent partagés entre modalités.

## Binaires minimaux

LLM seul, avec CLI et Metal, sans serveur ni outils de développement :

```sh
CARGO_BUILD_JOBS=4 taskpolicy -b cargo build --release --no-default-features --features metal
./target/release/saragossa --help
./target/release/saragossa run ./models/llm --backend metal
```

STT comme seule modalité supplémentaire, exposée par le serveur ; le cœur LLM
reste compilé mais aucun modèle LLM n'est chargé :

```sh
CARGO_BUILD_JOBS=4 taskpolicy -b cargo build --release --no-default-features --features metal,stt,serve
./target/release/saragossa serve --stt-model ./models/whisper
```

Pour intégrer seulement la bibliothèque STT : `--no-default-features --features
metal,stt`. Pour un moteur CPU, omettre `metal`. L'absence de `devtools` ne
supprime plus le binaire. Les poids des exemples doivent être présents localement.

## Vérification des combinaisons

Le job macOS `feature-check` partitionne `cargo hack check --each-feature`
en conservant Metal, vérifie toutes les features sauf une et exécute Clippy
sur le LLM minimal. Les exemples et tests métier déclarent leurs prérequis.
Sans cargo-hack installé, utiliser la boucle locale suivante, sans installer
d'outil ni télécharger de dépendance :

```bash
set -euo pipefail
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=4
features=(devtools stt tts vision denoise embed serve decision)
taskpolicy -b cargo check --all-targets --no-default-features --features metal
for feature in "${features[@]}"; do
  taskpolicy -b cargo check --all-targets --no-default-features --features "metal,$feature"
  selected=metal
  for other in "${features[@]}"; do
    if [[ "$other" != "$feature" ]]; then selected+=",$other"; fi
  done
  taskpolicy -b cargo check --all-targets --no-default-features --features "$selected"
done
```

La [conception des profils TOML](design/configuration.md) décrit l'étape
suivante. `--profile` et le rechargement à chaud ne sont pas implémentés.
