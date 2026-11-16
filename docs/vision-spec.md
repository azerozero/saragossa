# Vision Qwen : contrat de portage, étape 0

Base : `bdc2dadc3cbc6351d23aa4ec672beb9ebd5e8580`. La vision n'est pas
implémentée de bout en bout dans le moteur. Ce document fige les poids, la
référence et les portes nécessaires au portage. Le [prétraitement CPU
image](vision-preprocess.md) est disponible sur entrée RGB déjà décodée ;
il ne constitue pas une parité de l'encodeur ViT Rust.

## Inventaire vérifié

Le [manifeste exhaustif](../tests/fixtures/vision/manifest.json) contient les
noms originaux et canoniques, formes, dtypes, shards, offsets et SHA256 de
**chaque payload vision**, ainsi que les SHA256 des headers, des configurations,
des tokenizers et des processeurs. Tous les headers des safetensors à la racine
sont parcourus, indépendamment de l'index HF. Les poids texte ne sont pas
chargés ni hachés : leur identité complète devra accompagner les logits.

| Checkpoint | Type effectif | ViT profondeur / largeur / MLP | Têtes / dimension | Projection | Tenseurs identifiés |
|---|---|---|---|---|---|
| Qwen3.8-27B-BF16 | qwen3_5 | 27 / 1152 / 4304 | 16 / 72 | 5120 | 333 / 333 |
| Qwen3.6-35B-A3B-4bit | qwen3_5_moe | 27 / 1152 / 4304 | 16 / 72 | 2048 | 333 / 333 |
| THX-02 | thx02 → backbone qwen3_5 | 24 / 1024 / 4096 | 16 / 64 | 2560 | 297 / 297 |
| d3-flash | qwen3_5 | 27 / 1152 / 4304 | 16 / 72 | 4096 | 333 / 333 |

Les **1 296 tenseurs sont BF16**, y compris le ViT du checkpoint Qwen3.6
quantifié. Le contrat attend `12 × depth + 9` tenseurs. Aucun tenseur vision
inconnu, manquant ou dupliqué n'est toléré dans un encodeur présent.

- Qwen3.8 : préfixe `model.visual.` ; THX/d3 : `visual.`.
- Qwen3.6 : `vision_tower.` ; convolution MLX `[out,T,H,W,in]` à transposer
  en `[out,in,T,H,W]` avant le calcul de référence. Les autres convolutions
  sont déjà HF `[out,3,2,16,16]`.
- Toutes les matrices linéaires sont `[out,in]`, QKV fusionné dans cet ordre.
- Position apprise `[2304,hidden]`, soit 48 × 48 ; aucun DeepStack.
- Ne pas confondre la version Transformers déclarée dans un checkpoint avec
  celle qui produit la référence. Les deux figurent au manifeste.

Les profils connus sont validés par le chargeur Rust, avec formes, BF16,
tailles de payload, préfixes et présence exhaustive. Une configuration vision
peut subsister dans un export **texte seul sans aucun tenseur vision** ; ce cas
ne rend pas la vision disponible. La validation porte sur les headers, sans
allouer les poids. Les tests utilisent les quatre inventaires réels et des
mutations négatives (tenseur manquant, forme, dtype, préfixe ou nom inconnu).

## Prétraitement image

RGB uint8, `patch_size=16`, `temporal_patch_size=2`, `merge_size=2`.
Ne jamais reprendre le défaut générique 14. Bornes : 65 536 à 16 777 216
pixels pour Qwen3.8/3.6 ; maximum 1 638 400 pour THX et d3. Le fichier
processeur de d3 annonce un maximum plus élevé : son contrat runtime local
impose 1 638 400. Le SHA de ce fichier source est conservé, sans l'exécuter.

`smart_resize` arrondit d'abord H/32 et W/32 au **demi pair**, comme Python,
puis multiplie par 32. Au-dessus du maximum, employer `floor` après réduction
proportionnelle, avec chaque dimension au moins 32 ; sous le minimum, employer
`ceil` après agrandissement proportionnel. Un ratio supérieur à 200 est refusé.
Le backend de référence est torchvision, bicubique **antialiasé**.

Deux portes distinctes :

1. Sans resize, dimensions multiples de 32 et dans les bornes : max-abs ≤ 1e-6
   sur les pixels normalisés et les patchs. Normalisation mathématique
   `(pixel/255 - 0.5)/0.5` ; le processeur 5.8.1 la fusionne en
   `(pixel - 127.5)/127.5` en f32.
2. Avec resize : au plus **un niveau uint8**, soit ≤ 1/255 après conversion
   de la sortie uint8 vers [0,1]. Cela ne signifie pas 1/255 d'un niveau entier.
   Ne pas appliquer la première porte aux images redimensionnées.

L'image est dupliquée sur T=2. L'ordre des lignes avant le ViT est
`[bloc_h,bloc_w,intra_h,intra_w]` ; chaque ligne contient
`[C,T,patch_h,patch_w]`, soit 1 536 valeurs. La grille est `[1,H/16,W/16]`.
Quatre lignes voisines regroupées 2×2 produisent une ligne LLM.

## Encodeur et positions spatiales

Conv3D biaisée de noyau et stride `(2,16,16)`, équivalente à une projection
des patchs aplatis. Addition de la table apprise interpolée bilinéairement.
Les indices et poids de `fast_pos_embed_interpolate` sont **exportés depuis
la fonction Python installée, par grille**. Ne pas re-dériver `torch.linspace`
en Rust : sa séquence f32 peut différer de 1 ulp. Les tables sont `[4,H×W]`,
avant la permutation 2×2 ; ordre des voisins : bas d'indice h/w, h/w+1,
h+1/w, h+1/w+1. Somme dans l'ordre Python puis permutation des lignes.
Pour une capture BF16 future, les poids d'interpolation sont convertis au
dtype de la table apprise par Python ; les fixtures actuelles sont f32.

**RoPE du ViT** : `head_dim = hidden/heads`, dimension du générateur
`head_dim/2`, θ=10 000. Fréquences `theta^(-arange(0,dim,2)/dim)`.
La concaténation des fréquences h puis w donne `emb` de largeur `head_dim/2`
(moitié h, moitié w), puis `cat(emb,emb)` de largeur `head_dim`.
Appliquer `q*cos + rotate_half(q)*sin`, idem k ; `rotate_half([a,b])=[-b,a]`.
Ce n'est pas une rotation par paires adjacentes. Positions dans l'ordre 2×2
des patchs ; conversion de q/k en f32 pendant la rotation, retour au dtype.

Attention complète **non causale**, sans fenêtre, séparée par grille
temporelle via `cu_seqlens`. LayerNorm avec biais, epsilon 1e-6. MLP biaisé,
GELU-tanh. Merger : LayerNorm **avant** regroupement, concaténation 4×hidden,
projection 4×hidden → 4×hidden, GELU **exacte**, projection → largeur LLM.
La table apprise, le patch embed et le merger font partie de l'identité vision.

## Injection et M-RoPE LLM

Remplacer les embeddings de `<|image_pad|>` / `<|video_pad|>` en conservant
les délimiteurs et l'ordre des médias. Le nombre de placeholders doit être
exactement `T×H×W/4` ; aucun remplissage ou retrait silencieux.
Conserver `mm_token_type_ids` : texte 0, image 1, vidéo 2.

M-RoPE ne concerne que les couches **full-attention, une sur quatre**.
Les couches linéaires n'appliquent pas RoPE. Positions `[3,batch,seq]` :
texte identique sur t/h/w ; image `(t,h,w)` sur la grille fusionnée, décalée
du curseur courant ; le texte suivant reprend après la coordonnée maximale.
Sections entrelacées `[11,11,10]`, θ=1e7, fraction rotative 0,25 du head_dim
texte 256. Le prefill doit accepter des **positions par ligne** : son
`position_offset` scalaire actuel ne peut pas représenter une image.

En decode, les trois axes valent `pos + rope_delta`, avec
`rope_delta=max(position_ids)+1-longueur_non_pad`. Un seul entier suffit par
requête pour MTP, DFlash, lots et instantanés. Le delta peut être **négatif**.
L'adresse KV suit le nombre de tokens, jamais la position RoPE.

L'étape 3 dépend de l'étape 0, **pas de l'encodeur Metal de l'étape 2** :
les embeddings Python exportés sont directement injectables. Six préfixes
par modèle incluent trois cas multi-images ; leurs tokens, spans, positions,
deltas et 16 tokens de decode imposés sont fournis. Ils n'incluent pas encore
les logits ni les seuils BF16. Il faut mesurer l'oracle LLM sur le vocabulaire
entier, en préfill complet/fractionné puis decode forcé.

Calibrer les golden BF16 **une seule fois par modèle**, avant comparaison Rust,
sur les mêmes poids et préfixes que la référence f32 : RMS/max-abs des logits,
max-abs des probabilités ; seuils figés à exactement 2× le bruit observé.
Versionner ces seuils avec le SHA de Transformers et l'identité des poids.
Pour Qwen3.6, déquantifier les mêmes poids texte pour isoler le bruit BF16 de
l'erreur de quantification. Aucun recalibrage pour faire passer Rust.
Les embeddings f32 de cette étape ne prétendent pas calibrer ces logits.

Lever l'interdiction MTP/DFlash dès l'étape 3 après test A/B **avec image**,
acceptation, rejet et rollback ; il n'est pas nécessaire d'attendre l'étape 5.
Qualifier séparément lots, annulation, éviction et réutilisation des slots.

## Cache et instantanés

`BlockHash::chain` accepte le hash du **contenu du média et des paramètres
de traitement** au bloc où commence `<|image_pad|>`. Le producteur inclut
ordre, grilles, timestamps, versions et identité des poids vision. Deux images
différentes avec les mêmes placeholders provoquent un cache miss ; une image
identique permet un hit à une frontière sûre.

Le manifeste d'instantané **v5** conserve `rope_delta`, les positions par ligne,
les KV et les états récurrents. La reprise avant/après un bloc image est testée ;
les frontières internes sont refusées. Pour THX/d3 non causaux,
mettre en cache les embeddings vision, pas un préfixe KV causal.
Le [chemin LLM interne](vision-llm.md) est découplé du ViT ; le transport
média reste fermé.

## Vidéo et décision

La vidéo reste hors exécution de l'étape 0. Contrat à porter : échantillonnage
uniforme `linspace(...).round()` au demi pair, 2 fps, bornes 4–768 frames,
bornes effectives de disponibilité, répétition de la dernière frame si besoin.
Budget resize sur T×H×W : 4 096–25 165 824. Timestamp moyen par paire,
rendu `<x.x seconds>`, suivi d'un bloc vision. Chaque bloc vaut T=1 pour
M-RoPE ; le temps absolu appartient au texte. MVP : 16 frames / 2 048 tokens
vision. Contrat d3 spécifique : 2 fps, 32 frames, 200 704 pixels/frame.
THX lit ses marqueurs joints, d3 le dernier token par question ; préserver
masques non causaux, calibration et ordre des options.

## Référence reproductible et limites de l'étape 0

[vision.py](../tools/goldens/vision.py) s'exécute uniquement avec `-I -B`, sans
pip. Définir `REF_PYTHON` vers l'interpréteur de référence existant,
`MODEL_ROOT` vers le dossier local en lecture seule. Exemple depuis le dépôt :

```sh
"$REF_PYTHON" -I -B tools/goldens/vision.py inventory \
  --model-root "$MODEL_ROOT" --output tests/fixtures/vision
"$REF_PYTHON" -I -B tools/goldens/vision.py capture \
  --model-root "$MODEL_ROOT" --output tests/fixtures/vision
"$REF_PYTHON" -I -B tools/goldens/test_vision.py
```

PyTorch 2.12.0, Transformers 5.8.1, CPU eager f32, quatre threads,
algorithmes déterministes. Les SHA256 des sources installées et leur empreinte
agrégée sont versionnés ; le wheel ne fournit pas de SHA Git, signalé par
`transformers_git_sha: null`. Ne pas présenter le SHA256 de contenu comme un
identifiant de commit amont. Le script de capture est lui-même empreinté.

L'environnement ne contient pas torchvision. Les trois images déterministes
256×256, 224×320 et 320×224 ne nécessitent aucun resize. La normalisation
fusionnée et la permutation du processeur sont reproduites explicitement ;
les tests contrôlent les pixels, la duplication temporelle et l'ordre des
patchs indépendamment. Cette capture initiale n'exécute aucune porte de resize.
Les fixtures du [prétraitement CPU](vision-preprocess.md) ajoutent séparément
la référence PIL et les portes de resize de l'étape 1.

Les 12 embeddings sont calculés par le modèle vision installé, avec les vrais
poids BF16 promus f32 et chargés un par un ; aucun LLM complet, aucun code
présent dans un checkpoint n'est instancié ou exécuté. Les tables sont capturées
dans les variables locales de la fonction installée ; leur rejeu est vérifié
bit pour bit. Images, patchs et sorties sont des `.bin` little-endian avec
formes/dtypes/SHA256 dans des JSON. Aucun poids n'est copié au dépôt.

`VISION=0` est le défaut : chargement texte conservé, poids vision reconnus
et vérifiés sans consommation. `VISION=1` refuse explicitement le chargement
car la vision n'est pas implémentée ; une valeur invalide est refusée.
Les encodeurs de prompt de `ModelAssets` refusent image_pad/video_pad, et
l'API refuse les blocs image/vidéo avant chargement plutôt que de les ignorer.
La primitive bas niveau sur ids/embeddings reste une API textuelle : elle
n'interprète pas les médias. Le flag `VISION` entre dans l'empreinte runtime
des instantanés.
