# Prétraitement image Qwen sur CPU

Le contrat numérique est décrit dans [vision-spec.md](vision-spec.md).
Cette étape couvre le prétraitement, pas l'encodeur ViT.

## Entrée RGB et décodage

`Cargo.lock` ne contient aucun décodeur PNG ou JPEG (`image`, `png`,
`jpeg-decoder` et `zune-jpeg` sont absents). Aucune dépendance n'est ajoutée.
L'appelant doit donc fournir des pixels RGB uint8 déjà décodés, dans l'ordre
HWC, sans padding. Les fichiers PNG/JPEG ne sont pas acceptés directement.
Le décodage reste à intégrer lorsque des crates adaptées seront autorisées.

## Référence du redimensionnement

Les captures Pillow, `Image.Resampling.BICUBIC` sur RGB uint8, sont conservées.
Une seconde référence utilise torchvision 0.27.0 avec PyTorch 2.12.0 sur CPU :
tenseur uint8 NCHW contigu, batch de taille 1, `transforms.v2.functional.resize`,
`BICUBIC`, `antialias=True`, avant conversion f32 et normalisation fusionnée.
C'est le pipeline du [backend HF 5.8.1](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/image_processing_backends.py)
appelé par le [processeur Qwen2VL](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py).
Les tailles proviennent du `smart_resize` déjà capturé ; aucun processeur ni
poids téléchargé depuis un checkpoint n'est exécuté.

La porte spécifiée est d'un niveau uint8 (1/255 dans [0,1]). Le bicubique Rust
reproduit torchvision **exactement** (0 niveau sur les quatre images et les six
sondes) : poids normalisés, quantifiés sur int16 avec la plus grande précision
commune à l'axe qui tienne sous 32 768, arrondis symétriques, passe horizontale
puis verticale, chaque passe arrondie et saturée en uint8. PIL fixe 22 bits ;
l'écart PIL/torchvision atteint 2 niveaux (`ties`, `tiny`). Les captures PIL ne
sont plus qu'informatives : leurs tests n'assertent plus sur le resize.

## Interface et reproduction

Avec la feature Cargo `vision` (activée par défaut),
`saragossa::vision::preprocess` expose `RgbImage::from_rgb8(height, width,
pixels)` et `preprocess(&image, ImageProfile)`. Les profils `Qwen38`, `Qwen36`,
`Thx02` et `D3Flash` fixent les bornes du contrat. La sortie contient
`grid_thw = [1,H/16,W/16]` et `pixel_values`, matrice aplatie
`[H/16 × W/16,1536]` : lignes `[bloc_h,bloc_w,intra_h,intra_w]`, colonnes
`[C,T,patch_h,patch_w]`. La normalisation précède la répétition temporelle ×2.
Les étapes intermédiaires `resized(profile)`, `normalize_and_repeat(profile)`
et `into_patches(profile)` permettent de comparer les captures. Les deux
dernières exigent explicitement un profil et vérifient son plafond de pixels
avant toute allocation de sortie. Un dépassement renvoie
`InferError::VisionPixelBudget { pixels, max_pixels }`, sans allouer de message.
La normalisation alloue 24 octets par pixel ; la permutation fait coexister
deux tampons f32, soit au plus 48 octets par pixel (768 Mio pour Qwen,
75 Mio pour THX/d3), hors entrée RGB déjà fournie par l'appelant.
Le minimum de `smart_resize` ne s'applique pas aux étapes intermédiaires.
Les dimensions nulles, les tampons RGB mal dimensionnés et les patchs non
alignés sont refusés. Une image 8192×8192 est refusée avant allocation f32.

Les [fixtures](../tests/fixtures/vision/preprocess/manifest.json) contiennent
12 images synthétiques, 6 sondes de filtre (forte réduction, axe d'un pixel,
redimensionnement d'un seul axe), 60 dimensions calculées par la fonction
`smart_resize` installée et les 256 niveaux de normalisation f32. Le manifeste
conserve versions et empreintes SHA256 du générateur, de la source Python,
des pixels et des patchs. La section `torchvision_reference` ajoute quatre
images (dont une sans resize) et les six sondes, moins de 1 Mio de RGB au total,
avec versions, paramètres, empreintes et écart PIL/torchvision par capture.
Aucun poids ni appareil GPU n'est nécessaire.

Définir `REF_PYTHON` vers un environnement contenant torch 2.12.0,
torchvision 0.27.0, Pillow 12.2.0 et NumPy (version enregistrée au manifeste).
Le générateur additionnel vérifie les captures PIL existantes sans les modifier :

```sh
taskpolicy -b "$REF_PYTHON" -I -B tests/fixtures/vision/preprocess/generate_torchvision.py
CARGO_BUILD_JOBS=4 taskpolicy -b cargo fmt --all -- --check
CARGO_BUILD_JOBS=4 taskpolicy -b cargo clippy --all-targets --all-features -- -D warnings
CARGO_BUILD_JOBS=4 taskpolicy -b cargo test --release --no-default-features --features vision --lib vision::preprocess
```

Les tests CPU sont non ignorés. Ils vérifient chaque pixel et chaque valeur
de patch, et valident l'ordre contre les empreintes de la permutation PyTorch.
Ils distinguent la porte f32 sans resize de la porte uint8 avec resize.

Mesure locale avec Pillow 12.2.0, PyTorch 2.12.0 et Transformers 5.8.1 :
écart maximal **0 en f32** sur les 256 niveaux et sur les pixels/patchs des
5 fixtures sans resize. Les 60 dimensions et les grilles/ordres des 12 images
correspondent exactement à la référence Python. Contre torchvision 0.27.0, le maximum est **0 niveau uint8** ; contre PIL, il
atteint 2 niveaux sur le resize (informatif). Les captures sans resize gardent
la porte f32 à 1e-6.
