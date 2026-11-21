# Chargement des RMSNorm Qwen3.5 HF

Les checkpoints Hugging Face Qwen3.5 bruts stockent les RMSNorm du décodeur
avec la convention `rms_norm(x) * (1 + weight)`. Le chargeur utilisait
directement `weight`, ce qui déformait les activations sur CPU comme sur Metal
et produisait du texte incohérent.

`mlx_lm/models/qwen3_5.py`, dans `TextModel.sanitize()`, ajoute déjà cet offset
aux normes d'entrée, post-attention, Q/K et finale. Il détecte le format brut
par les convolutions `[canaux, 1, noyau]`, qu'il transpose ensuite en
`[canaux, noyau, 1]`. La norme gated `linear_attn.norm.weight` n'a pas d'offset.
La distinction est également explicite dans les classes Transformers
`Qwen3_5RMSNorm` et `Qwen3_5RMSNormGated`.

Le chargeur applique désormais l'offset en f32 après décodage des poids,
uniquement pour les familles `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe` et
`qwen3_5_moe_text` avec une convolution HF dans le tronc texte chargé. La
détection ne dépend ni du dtype ni du préfixe. Les exports MLX, y compris
non quantifiés, conservent leurs normes déjà converties. Le calcul en f32
évite l'arrondi intermédiaire BF16 effectué lors de certains exports MLX.

Les préfixes HF `model.language_model.*` et `lm_head.weight` étaient déjà
correctement reconnus, de même que `model.*` pour les checkpoints texte.
Les sections vision/MTP ne sont pas chargées dans le décodeur texte. Les deux
layouts conv1d ont le même ordre de valeurs et sont déjà acceptés par les
kernels. `A_log`, `dt_bias` et les projections séparées QKV/Z restent inchangés.

Le défaut concerne notamment Qwen3.8-27B-BF16 et Arete-27B-BF16, tous deux
déclarés `qwen3_5`, ainsi que les autres checkpoints HF de ces familles qui
utilisent cette convention. Ce n'est pas un défaut général du décodage BF16 :
Qwen3 classique et Llama ne doivent pas recevoir cet offset. Qwen3-Next a
également des normes à offset dans Transformers, mais son format HF nécessite
d'autres conversions (projections fusionnées et experts) et n'est pas rendu
compatible par ce correctif.

## Régression synthétique

```sh
cargo test --no-default-features --lib qwen35_sanitize_tests
```

Le test de chargement compare les tenseurs et les logits CPU d'un petit
checkpoint BF16 HF et de son équivalent MLX, avec plusieurs préfixes, embeddings
liés ou séparés, et sections vision/MTP parasites. Sans l'appel au correctif,
les logits HF de cette fixture s'annulent et le test échoue. Un second test
vérifie les catégories de normes, les familles et la préservation des poids MLX.

## Vérification avec poids réels

Définir `MODEL_DIR` vers un checkpoint HF local, puis lancer une génération
greedy courte, sans chauffe ni benchmark :

```sh
cargo build --release
SARAGOSSA_RUST_WARMUP=0 SARAGOSSA_RUST_MTP_DECODE=0 target/release/saragossa \
  --model-dir "$MODEL_DIR" --prompt 'The capital of France is' \
  --max-tokens 12 --temperature 0 --backend metal
```

Le backend `cpu` emprunte le même traitement des normes ; ses matrices BF16
sont développées en f32 et demandent davantage de mémoire.

Validation du 14 novembre 2026 sur Qwen3.8-27B-BF16, backend Metal, plafond de
12 tokens, température zéro, chauffe et MTP désactivés :

```text
The capital of France is **Paris**.
```

Les en-têtes et un échantillon des normes d'Arete-27B-BF16 présentent la même
convention HF ; aucune seconde génération réelle n'a été exécutée.
