# Régressions proptest

Ce répertoire est versionné. Proptest rejoue les graines enregistrées avant
les nouveaux tirages. Conserver les fichiers `.txt` produits après réduction,
avec le test et le correctif correspondants ; ne pas effacer une graine pour
faire passer la suite.

`sampling.txt` est le journal explicitement utilisé par
`tests/sampling_properties.rs`. Il ne contient pas de régression historique
inventée : le fichier initial ne porte que son en-tête. Les propriétés des
modules `src/` utilisent aussi le chemin par défaut `SourceParallel` de
proptest, sous ce même répertoire.

Rejouer sans poids ni réseau :

```sh
cargo test --locked --no-default-features --test sampling_properties
```

Une graine ne constitue pas seule un oracle : garder les assertions de
normalisation, de support et de déterminisme lors de toute correction.
