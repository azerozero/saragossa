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

Torchvision n'est pas installé dans l'environnement de référence local.
La référence de repli est Pillow, `Image.Resampling.BICUBIC` sur une image RGB
uint8, avec l'antialias inhérent au rééchantillonnage PIL. La porte reste
un écart absolu maximal d'un niveau uint8 (1/255 dans [0,1]). La parité avec
torchvision n'est pas mesurée ; les arrondis et la saturation intermédiaire
de PIL peuvent différer de ceux du backend tenseur de torchvision.

## Interface et reproduction

Avec la feature Cargo `vision` (activée par défaut),
`saragossa::vision::preprocess` expose `RgbImage::from_rgb8(height, width,
pixels)` et `preprocess(&image, ImageProfile)`. Les profils `Qwen38`, `Qwen36`,
`Thx02` et `D3Flash` fixent les bornes du contrat. La sortie contient
`grid_thw = [1,H/16,W/16]` et `pixel_values`, matrice aplatie
`[H/16 × W/16,1536]` : lignes `[bloc_h,bloc_w,intra_h,intra_w]`, colonnes
`[C,T,patch_h,patch_w]`. La normalisation précède la répétition temporelle ×2.
Les étapes intermédiaires `resized`, `normalize_and_repeat` et `into_patches`
permettent de comparer les captures. Les dimensions nulles, les tampons RGB
mal dimensionnés et les patchs non alignés sont refusés.

Les [fixtures](../tests/fixtures/vision/preprocess/manifest.json) contiennent
12 images synthétiques, 6 sondes de filtre (forte réduction, axe d'un pixel,
redimensionnement d'un seul axe), 60 dimensions calculées par la fonction
`smart_resize` installée et les 256 niveaux de normalisation f32. Le manifeste
conserve versions et empreintes SHA256 du générateur, de la source Python,
des pixels et des patchs. Aucun poids ni appareil GPU n'est nécessaire.

Définir `REF_PYTHON` vers l'interpréteur existant, sans installer de paquet :

```sh
"$REF_PYTHON" -I -B tools/goldens/vision_preprocess.py \
  --output tests/fixtures/vision/preprocess
CARGO_BUILD_JOBS=4 taskpolicy -b cargo fmt --all -- --check
CARGO_BUILD_JOBS=4 taskpolicy -b cargo clippy --all-targets --all-features -- -D warnings
CARGO_BUILD_JOBS=4 taskpolicy -b cargo test --release --no-default-features --features vision --lib vision::preprocess
```

Les tests CPU sont non ignorés. Ils vérifient chaque pixel et chaque valeur
de patch, et valident l'ordre contre les empreintes de la permutation PyTorch.
Ils distinguent la porte f32 sans resize de la porte uint8 avec resize.

Mesure locale avec Pillow 12.2.0, PyTorch 2.12.0 et Transformers 5.8.1 :
écart maximal **0 niveau uint8** sur les 12 fixtures et les 6 sondes ;
écart maximal **0 en f32** sur les 256 niveaux et sur les pixels/patchs des
5 fixtures sans resize. Les 60 dimensions et les grilles/ordres des 12 images
correspondent exactement à la référence Python. Ces mesures ne prétendent
pas quantifier l'écart avec torchvision, absent de cet environnement.
