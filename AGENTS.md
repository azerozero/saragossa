# saragossa — Instructions pour les agents de code

Avant de modifier le dépôt, lire [`CLAUDE.md`](CLAUDE.md) pour l'architecture
et les conventions, puis [`llms.txt`](llms.txt) pour la carte des modules.

## Règles non négociables

1. Travailler sur une branche dédiée. Le flux public est fork, branche, pull
   request, vérifications CI et signature du [`CLA`](CLA.md).
2. Utiliser des Conventional Commits en français avec un corps utile. Ne pas
   ajouter de trailer de génération ni de `Co-Authored-By`.
3. Avant de finir, exécuter `cargo fmt --all -- --check`, `cargo clippy
   --all-targets --all-features`, `cargo build` et les tests concernés.
4. Ne pas ajouter de `unwrap()` ou `panic!` sur un chemin de production.
   Utiliser `Result` et `?`; réserver `expect("invariant: …")` aux invariants
   démontrés.
5. Chaque bloc `unsafe` porte immédiatement un commentaire `// SAFETY:` qui
   expose l'invariant de taille, d'alignement ou de durée de vie Metal.
6. Ne jamais ajouter de poids de modèle, jeton, chemin personnel ou donnée
   privée au dépôt public.
7. Ne jamais pousser, publier sur crates.io ou créer une release sans demande
   explicite du mainteneur.

## Périmètre

saragossa est le moteur d'inférence lui-même. Toute nouvelle dépendance ou
documentation doit rester dans ce périmètre moteur et ses interfaces publiques.
