# Qwen3.8-Flash-Next : conversion FP8 → MLX affine « q4e8 »

Étape 0 du portage `qwen4_exp`. Elle produit une sortie au format MLX, mais le
loader ne la charge pas encore : il refuse `output_gate_type=sigmoid`, et
l'hyper-connection, QSA et le PLE ne sont pas portés. Cette étape valide les
tenseurs, pas l'inférence.

## Commande

```bash
saragossa convert-qwen4exp \
  --src  ai_model/Qwen3.8-Flash-Next-FP8 \
  --dst  ai_model/Qwen3.8-Flash-Next-MLX-q4e8 \
  --source-revision 236dfdf285828023ca3bcd3f37366c58a3469b13 \
  --source-repo Qwen/Qwen3.8-Flash-Next-FP8 \
  --converter-commit "$(git rev-parse HEAD)"
```

La source et la destination doivent être deux dossiers distincts, y compris à
travers un lien symbolique.

Options :

- `--plan` produit l'inventaire et le plan sans rien écrire ;
- `--only <unité,…>` traite seulement les unités listées (pilote, sans
  finalisation) ;
- `--verify` impose une vérification sha256 complète des shards déjà
  terminés.

**Reprise.**

- Un shard terminé est sauté sur sa taille. La finalisation rehache tous les
  fichiers ; un shard divergent est retiré du journal, et la commande échoue
  pour qu'une relance le réécrive.
- Les segments n-gram sont toujours rehachés, même sans `--verify`. Un fichier
  absent, tronqué ou altéré entraîne la recopie des segments touchés.
- La reprise est refusée si la révision source, l'index, la config, la
  version des règles ou les paramètres numériques changent.
- Le sha du binaire et le commit du convertisseur sont consignés au
  manifeste, mais ne bloquent pas la reprise.

## Format de sortie

| tenseurs | format |
|---|---|
| experts routés, tronc et MTP (75 264 matrices FP8) | u4 gs64, `switch_mlp.{gate,up,down}_proj` empilés `[512, rows, packed]` |
| matrices denses (525) | u8 gs64 |
| `lm_head` | u6 gs64, flux de bits : `.weight U32 [248320,480]`, scales et biais `[248320,40]` |
| routeurs, gates, `block_inject`, `in_proj_a/b`, normes, conv, `A_log`, `dt_bias` (570) | copie à l'octet |
| buffers I64 du hachage n-gram (3) | I64, copie à l'octet |
| `ngram_embedding.weight_scale` (1) | scalaire BF16, copie à l'octet, repris dans `ngram/meta.json` |
| table n-gram (128 shards) | `ngram/ngram_e4m3.bin`, `[320 001 536, 160]` E4M3FN compact |
| vision (333) | exclue |

Le décodage des experts FP8 suit `w = e4m3 × weight_scale_inv`, avec une
échelle bf16 par bloc de 128×128.

Les codes valent `round((w − b) / s)`, calculés en f64 contre la scale `s` et
le biais `b` déjà arrondis en bf16 :

- le biais est arrondi vers −∞ et la scale vers +∞, pour qu'aucun code ne soit
  écrêté ;
- l'étendue du groupe est calculée en f64 ;
- un groupe dont la restitution sortirait de f32 est une erreur.

Les normes sont copiées à l'octet, mais deux conventions de lecture coexistent
(voir le manifeste) :

- `Qwen4ExpTextRMSNorm` lit un gain `1 + w` : `hc_norm`, `q/k_norm`, normes de
  l'indexeur et du PLE, `mtp.pre_fc_norm_*` ;
- `linear_attn.norm` (`Qwen4ExpTextRMSNormGated`) lit un gain `w` direct.

La couche PLE est lue dans `ple_layer_ids`.

La table n-gram se lit avec `w = f32(e4m3) × weight_scale`, une échelle
per-tensor unique valant `0x3951`, soit ≈ 1,99e-4. Les paramètres de hachage
(seed 1234, multiplicateurs, nombres premiers des têtes, offsets) sont
recalculés et comparés aux buffers du checkpoint.

## Conversion mesurée

Mesures faites sur M5 Max 128 Gio, le 5 nov. 2026, avec les règles
`qwen4_exp-q4e8-v2`.

- **Durée totale** : 720 s, relecture finale et sha256 de chaque fichier
  compris. Une reprise complète sans `--verify` saute les 146 unités en 380 s :
  elle rehache les 128 segments n-gram et tous les fichiers à la
  finalisation, et retrouve des sha identiques.
- **Pic de mémoire** : 515 Mo, mesuré par `/usr/bin/time -l` (`peak memory
  footprint`). L'échantillonnage interne, fait blocs vivants, donne la même
  valeur. Les lectures sont bornées, sans mmap de la source.
- **Taille** : 125,83 Go, soit 117 Gio sur disque.

| catégorie | octets |
|---|---|
| experts u4 (tronc) | 67 947 724 800 |
| experts u4 (MTP) | 1 415 577 600 |
| denses u8 (tronc) | 4 498 227 200 |
| denses u8 (MTP) | 94 699 520 |
| lm_head u6 | 516 505 600 |
| copies | 159 626 778 |
| table n-gram E4M3 | 51 200 245 760 |

Erreur relative `‖W−dq(q(W))‖/‖W‖` des tenseurs, calculée avec les
paramètres bf16 sérialisés :

| tenseurs | erreur relative |
|---|---|
| experts u4, par pile de 512 | 0,0903 à 0,0973 ; pire expert 0,1308 (`layers.5…gate_proj`) |
| denses u8 | 0,0053 à 0,0087 |
| lm_head u6 | 0,0223 |

## Qualité contre la référence FP8

**Version mesurée.** Cette mesure porte sur la sortie `q4e8-v1`, où les codes
étaient arrondis par produit avec l'inverse de la scale. La v2 arrondit par
division : quelques codes changent dans chaque matrice, mais les scales et les
biais restent identiques. Un écart de qualité mesurable n'est pas attendu, et
la mesure n'a pas été refaite.

**Dispositif.** La référence est le checkpoint FP8 déquantifié. Le passage se
fait couche par couche, en f32 sur CPU, avec les modules transformers 5.20.2
de `qwen4_exp` en attention `sdpa`.

- Le MoE est recalculé expert par expert, avec le routage propre à chaque
  pipeline.
- La table n-gram est identique dans les deux pipelines.

**Validation préalable.** Le chemin découpé est comparé aux couches
officielles 0 à 3, aux longueurs 64, 2047, 2048, 2049 et 2052. L'écart relatif
mesuré est inférieur ou égal à 2,5e-8.

**Corpus.** 8 × 512 tokens (code ×3, français ×2, anglais ×2, appels
d'outils), plus 6 000 tokens de contexte long. Les 16 premières positions de
chaque séquence sont exclues. Total : 9 943 positions évaluées.

| mesure | valeur | porte provisoire |
|---|---|---|
| KL(Ref‖Q) moyen | 0,0446 | ≤ 0,05 |
| KL médian (blocs de 64) | 0,0366 | |
| ΔNLL (nat/token) | +0,0135, IC95 [+0,0056 ; +0,0212] | ≤ 0,015 |
| top-1 identique | 91,7 % | |

Sur les séquences courtes seules, le KL vaut 0,0486 et le ΔNLL +0,0152.

Résultats par domaine :

| domaine | KL | ΔNLL | top-1 |
|---|---|---|---|
| code (×3) | 0,023 / 0,023 / 0,050 | +0,002 / −0,011 / +0,007 | 93,5 / 96,2 / 93,9 % |
| français (×2) | 0,038 / 0,047 | +0,028 / +0,037 | 91,9 / 86,7 % |
| anglais (×2) | 0,136 / 0,067 | +0,079 / −0,019 | 90,3 / 92,3 % |
| outils | 0,006 | −0,001 | 98,2 % |
| contexte long 6k | 0,042 | +0,012 | 90,9 % |

**Ablation locale** (poids Q sur l'entrée de la référence) :

- erreur de sortie de couche : médiane 2,2 %, maximum 4,3 % ;
- branche attention ou DeltaNet : 0,7 % à 1,3 % ;
- routage top-10 identique : médiane 98,0 %, minimum 94,7 %.

**Lecture.** Les deux moyennes passent les portes provisoires, de justesse.
L'IC du ΔNLL chevauche le seuil, et les séquences courtes seules le dépassent
légèrement. La dégradation est du même ordre que le mixte 4/8 de pipenetwork
(+0,0128 sur wikitext-2).

La perte se concentre sur la prose, en particulier un texte anglais à KL 0,136.
Le code et les appels d'outils sont quasi intacts.

Pistes, à tester une variable à la fois : calibration de type imatrix ou refit
WLS des experts, et lm_head en u8.

## Changement de comportement des loaders FP8

Le décodeur E4M3 partagé (`safetensor.rs`) refuse désormais les octets `0x7f`
et `0xff`. Ce sont les seuls NaN d'E4M3FN ; ils étaient auparavant décodés en
±480. Les échelles FP8 non finies sont aussi refusées.

Un checkpoint FP8 qui contient ces valeurs, auparavant chargé en silence avec
des poids faux, échoue maintenant au chargement avec une erreur explicite.

La forme des échelles par blocs est contrôlée par un cœur unique
(`check_fp8_block_scale_shape`), pour le loader comme pour le convertisseur.
Une échelle scalaire reste acceptée dans le loader.
