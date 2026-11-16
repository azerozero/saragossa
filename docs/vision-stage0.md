# Vision Qwen — résultat de l'étape 0

Contrat : [vision-spec.md](vision-spec.md). Base : `bdc2dadc3cbc`.
La vision reste explicitement indisponible dans le moteur ; le texte reste
utilisable avec `VISION=0`.

Sous `VISION=0` (défaut), une tour vision non inventoriée, quantifiée ou
incompatible est ignorée avec un avertissement stderr unique par processus.
La validation des fichiers safetensors reste active. Sous `VISION=1`, le
chargement reste refusé tant que le chemin vision n'est pas implémenté.

| Inventaire | Identifiés / attendus | Layout Conv3D |
|---|---:|---|
| Qwen3.8-27B-BF16 | 333 / 333 | OITHW |
| Qwen3.6-35B-A3B-4bit | 333 / 333 | OTHWI → OITHW |
| THX-02 | 297 / 297 | OITHW |
| d3-flash | 333 / 333 | OITHW |

Les 1 296 tenseurs vision sont BF16. Le
[manifeste](../tests/fixtures/vision/manifest.json) contient leurs SHA256,
formes et offsets, les processeurs, configurations et versions. Les fichiers
de poids ont été lus uniquement ; aucun code de checkpoint n'a été exécuté.

Trois images synthétiques sans resize produisent 12 embeddings f32, les tables
d'interpolation Python exactes et six préfixes M-RoPE par modèle, dont trois
multi-images, avec 16 pas imposés. Deltas signés : −56 à −120. Les 42 payloads
binaires ont été vérifiés octet par octet dans l'index Git.

| Contrôle | Résultat |
|---|---|
| Python, images/patchs/embeddings/tables/positions | 4 tests réussis, code 0 |
| Nouveaux tests Rust en release | 9 tests réussis, code 0 |
| `cargo fmt --all -- --check` | code 0 |
| `cargo clippy --all-targets --all-features -- -D warnings` | code 0 |
| `cargo build` | code 0 |
| `cargo test --release --all-features` | 1 282 réussis, 261 ignorés, aucun échec, code 0 |
| Contrats réels Qwen3.8 et Qwen3.6, `--check --backend cpu` | `VISION=0` : code 0 ; `VISION=1` : refus explicite, code 2 |
| Gitleaks, index et commit des fixtures | aucun secret détecté ; SHA256 publics des tokenizers exclus par une règle exacte |

Compilations sous `taskpolicy -b`, Cargo hors ligne ; captures Python CPU en
`-I -B`, sans installation ni téléchargement. Les tests Metal utilisent le
verrou coopératif, propriétaire `vision`, priorité 5.

Limites : torchvision absent, donc aucune porte de resize exécutée. Les logits
et leur calibration BF16 unique restent à l'étape 3. Le wheel Transformers
5.8.1 ne contient pas de SHA Git ; SHA256 des sources installées versionnés.
Les mesures p50/p95 et la qualification Metal vision appartiennent aux étapes
suivantes. L'étape 3 peut commencer avec ces embeddings sans attendre l'étape 2.
