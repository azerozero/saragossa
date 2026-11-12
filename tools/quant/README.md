# Quantification mixte par sensibilité

Outils Python qui produisent les quantifications mixtes 3/4 bits publiées pour
ce moteur, et mesurent leur qualité. Annexes au moteur Rust : ils servent à
fabriquer des poids, pas à les exécuter.

| Fichier | Rôle |
|---|---|
| `mixed_quant.py` | Alloue deux largeurs de bits par couche selon la sensibilité, écrit un modèle MLX |
| `kl_eval.py` | Mesure KL(bf16 ‖ quantifié) et l'accord top-1, en teacher forcing |
| `test_allocation.py` | Tests de la logique d'allocation, sans dépendance à MLX |
| `convert_ls.py` | Re-quantifie en u4 gs64 par moindres carrés depuis les poids BF16, sans changer le format (+1,7 % de perplexité contre +2,9 % en min-max sur le 27B, sans données de calibration) |
| `opt_quant.py` | Ajustement d'échelle et de biais par groupe utilisé par `convert_ls.py` |

## Méthode

Pour chaque couche linéaire, on mesure l'erreur relative de la largeur basse :

```text
rel_err(W) = ‖W − dequant(quant(W))‖_F / ‖W‖_F
```

Les couches sont classées par erreur décroissante et promues en largeur haute
tant qu'un budget en paramètres n'est pas épuisé. Aucune donnée de calibration
n'intervient, donc le résultat est déterministe et reproductible.

Sur les architectures hybrides GatedDeltaNet/SSM, ce classement en espace des
poids bat le classement pondéré par activations (façon imatrix) : les
projections de porte récurrentes sont systématiquement sous-protégées par les
proxys d'activation d'entrée. Mesuré sur le 27B : KL 0,065 contre 0,18 à 0,20
pour trois variantes imatrix à bpw égal. Les méthodes à gradient (DWQ) ne
s'appliquent pas ici, MLX n'ayant pas de rétropropagation sur les opérateurs
SSM.

Deux garde-fous structurels comptent autant que le classement :

- `--force-high` épingle hors budget ce dont l'erreur se propage. Le défaut
  suit l'architecture : en dense seuls `embed` et `lm_head` sortent du budget,
  tandis que `--moe` y ajoute la récurrence SSM, le routeur, les experts
  partagés et l'attention pleine. Appliquer la recette MoE à un modèle dense
  épingle plus de la moitié des paramètres et fait grimper le bpw sans que
  rien ne le signale (mesuré sur un 0,6B : 55,7 % épinglés, 4,35 bpw au lieu
  de 4,06).
- `--moe` décide **par trio de couche** (`gate`/`up`/`down` ensemble). Les
  kernels experts fusionnés exigent des largeurs identiques ; un trio
  dépareillé perd tout chemin rapide. Mesuré : 10 couches sur 40 dépareillées
  faisaient tomber le décodage de ≥140 à 95 tok/s, sans la moindre erreur
  visible.

## Prérequis

```bash
pip install mlx mlx-lm
```

Apple Silicon requis. Prévoir la place disque du modèle bf16 source (~52 Go
pour un 27B, ~67 Go pour un 35B) et autant de mémoire unifiée que possible :
`mixed_quant.py` charge le modèle entier avant de le réécrire.

## Reproduire les modèles publiés

Récupérer d'abord le modèle bf16 de référence depuis Hugging Face. Les
commandes ci-dessous s'exécutent depuis ce répertoire (`tools/quant/`) :

```bash
# Dense 27B, mixte 3/4 bits -> 3,85 bpw, 12 Go
# 0,25 est la valeur du modèle publié (9,5 % épinglés + 25,2 % promus = 34,6 %).
python3 mixed_quant.py \
    --input Qwen3.6-27B-bf16 --output Qwen3.6-27B-mixed34 \
    --low-bits 3 --high-bits 4 --high-fraction 0.25

# MoE 35B-A3B : décision par trio, garde-fous structurels -> 3,85 bpw, 16 Go
# 0,28 est la valeur du modèle publié (7,1 % épinglés + 27,9 % promus).
python3 mixed_quant.py \
    --input Qwen3.6-35B-A3B-bf16 --output Qwen3.6-35B-A3B-mixed34 \
    --low-bits 3 --high-bits 4 --high-fraction 0.28 --moe
```

Le second produit 12 couches expertes en 4 bits sur 40, le partage annoncé sur
la carte du modèle.

`--high-fraction` est un budget **hors modules épinglés** : la part promue par
sensibilité, à laquelle s'ajoute celle de `--force-high`. La part totale en
largeur haute est donc la somme des deux, et c'est elle que rapporte la ligne
de synthèse affichée par le script.

Cette convention explique un écart apparent avec la carte du 27B, qui annonce
un « 30 % parameter budget » : le script d'origine comptait `embed` et
`lm_head` **dans** le budget, si bien que 0,30 y produisait 34,6 % de
paramètres en 4 bits. Ici les épinglés sont hors budget, donc `0,25 + 9,5 %`
donne le même 34,6 % et le même modèle. Les deux chiffres décrivent la même
quantification sous deux conventions différentes.

Puis mesurer l'écart au bf16 :

```bash
python3 kl_eval.py \
    --reference Qwen3.6-27B-bf16 --candidate Qwen3.6-27B-mixed34
```

Valeurs attendues : KL ≈ 0,065 pour le 27B, ≈ 0,162 pour le 35B-A3B. Repères
d'interprétation : KL < 0,05 excellent, < 0,15 bon, > 0,3 dégradation notable ;
un accord top-1 > 0,95 vaut quasi-sans-perte.

Le jeu de prompts est figé dans `kl_eval.py` pour que deux exécutions restent
comparables. Le remplacer via `--prompts` change les valeurs absolues, pas
seulement leur bruit : ne comparez que des mesures issues du même jeu.

## Référence KL du moteur Rust

Avec les features `metal,devtools` (actives par défaut), sur deux modèles locaux :

```bash
saragossa run /chemin/reference --prompt "Bonjour" --max-tokens 32 --kl-record /tmp/kl-ref
saragossa run /chemin/candidat --kl-compare /tmp/kl-ref
```

Les mêmes options fonctionnent avec `--model-dir` à la place de `run`.
`--kl-record` et `--kl-compare` sont exclusifs. Aucun téléchargement : le modèle
est local. Le répertoire de sortie doit être nouveau ; il contient `ref.json`
(version 1, prompt, tokens greedy imposés, dimensions, empreintes et transformations)
et `ref.bin` (log-softmax f32 little-endian, matrice `[n, vocab]`). La ligne `i`
prédit `forced_ids[i]` après le prompt et `forced_ids[..i]`. Seule la dernière
position du prefill est évaluée, puis les `n−1` pas de decode GPU résident.
L'enregistrement fait exactement `--max-tokens` pas, y compris après EOS.

Les logits sont pris après softcapping, avant température/top-k/top-p ; ces
options de sampling n'affectent pas le diagnostic. Le KV du decode est f32.
La comparaison relit le prompt et les tokens enregistrés, refuse le repli du
decode résident, les versions inconnues, vocabulaires différents et fichiers
tronqués. Le JSON est borné à 1 Mio, le binaire à 256 Mio et les séquences à
65 536 tokens. L'empreinte SHA-256 de config et celle d'un échantillon de poids
(embeddings exécutés premier/milieu/dernier et norme finale) sont informatives : ce ne
sont ni une preuve d'identité de tous les poids ni un contrôle du tokenizer.
Conserver le même vocabulaire et la même correspondance token/id entre modèles.

`--kl-compare` écrit une ligne JSON : `kl_mean`, `kl_max` en nats,
`top1_agreement`, `forced_delta_p_mean` (écart signé candidat moins référence),
`forced_delta_p_max` (maximum absolu), `forced_abs_delta_p_mean` (moyenne absolue
du token imposé) et `mean_abs_delta_p` (moyenne absolue sur les entrées `[n, vocab]`). Les log-probabilités f32 des deux côtés sont
renormalisées en f64 pour les métriques. La KL couvre tout le vocabulaire ;
ce format est propre au moteur, sans compatibilité binaire llama.cpp.

## Modèles publiés avec ces outils

- [Qwen3.6-27B-mixed-3-4bit-3.85bpw-mlx](https://huggingface.co/destynova002/Qwen3.6-27B-mixed-3-4bit-3.85bpw-mlx)
- [Qwen3.6-35B-A3B-mixed-3-4bit-3.85bpw-mlx](https://huggingface.co/destynova002/Qwen3.6-35B-A3B-mixed-3-4bit-3.85bpw-mlx)

## Tests

```bash
python3 test_allocation.py
```

Les tests couvrent le classement, le budget, les couches épinglées et
l'invariant des trios MoE. Ils n'importent pas `mlx`, donc ils tournent sur
n'importe quelle machine.
