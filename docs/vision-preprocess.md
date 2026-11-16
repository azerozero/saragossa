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
