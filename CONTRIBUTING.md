# Contribuer à saragossa

saragossa est un moteur d'inférence Rust pour Apple Silicon. Les contributions
portent sur le moteur, ses kernels Metal et ses interfaces publiques.
Consultez [CLAUDE.md](CLAUDE.md) pour l'architecture, les conventions Rust et
Metal et les vérifications détaillées, puis [llms.txt](llms.txt) pour la carte
des modules.

## Licence

Le projet est distribué sous [Apache-2.0](LICENSE-APACHE). Les contributions
sont proposées sous cette même licence. N'ajoutez pas de poids de modèles,
de jetons, de chemins personnels ni de données privées au dépôt.

## Préparer son environnement

- macOS sur Apple Silicon pour compiler et tester le backend Metal.
- Rust stable avec `rustfmt` et `clippy` ; la version minimale est déclarée
  dans `Cargo.toml`.
- Xcode et la Metal Toolchain pour les kernels GPU.
- [prek](https://github.com/j178/prek) et Gitleaks pour les hooks locaux.
- `cargo-deny`, `cargo-audit`, `cargo-machete` et `cargo-sweep` pour les
  contrôles de dépendances et l'entretien du cache de compilation.

Après avoir créé votre fork sur GitHub :

```bash
git clone https://github.com/VOTRE_COMPTE/saragossa.git
cd saragossa
git remote add upstream https://github.com/azerozero/saragossa.git
git switch -c fix/sujet
rustup component add rustfmt clippy
prek install
cargo build
```

Les hooks de [prek.toml](prek.toml) vérifient l'hygiène des fichiers, le
formatage, Clippy et les secrets au commit. Au push, ils ajoutent les
contrôles de dépendances et de documentation. Les tests restent à lancer
séparément et sont rejoués en CI.

## Branches et commits

Le flux public est : fork, branche dédiée, pull request vers `main`, revue
et CI verte. Utilisez un nom court comme `feat/sujet`, `fix/sujet` ou
`ci/sujet`. Ne travaillez pas directement sur `main`.

Rédigez les [Conventional Commits](https://www.conventionalcommits.org/)
en français, avec un corps expliquant la raison du changement :

```text
fix(metal): vérifie la taille du buffer avant la copie

Retourne une erreur pour un tenseur tronqué afin de préserver les
bornes du buffer GPU.
```

Les types courants sont `feat`, `fix`, `perf`, `refactor`, `docs`, `test`,
`ci`, `build` et `chore`. Le scope est facultatif. Signalez une rupture
avec `!` ou un pied de message `BREAKING CHANGE:`. N'ajoutez aucun trailer
d'attribution.

Conservez un changement cohérent par commit. Les pull requests sont
fusionnées avec un commit de merge ; leurs commits doivent donc rester
lisibles individuellement.

## Vérifier avant la pull request

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features --document-private-items
cargo build
cargo test --all-features
cargo deny check
cargo audit
cargo machete
```

Les tests exigeant des poids restent ignorés ou activés explicitement. Ne
rajoutez pas de téléchargement de modèles aux contrôles ordinaires. Toute
optimisation GPU doit conserver un oracle CPU ou une référence numérique.
Pour un changement de workflow, lancez également `actionlint`.

## Proposer et faire relire

1. Ouvrez la pull request de votre fork vers `main` avec un titre Conventional
   Commits en français.
2. Décrivez le problème, le comportement obtenu, les vérifications réalisées
   et les limites connues.
3. Attendez le statut `Required checks` et les contrôles concernés, notamment
   GPU synthétique, CodeQL et documentation.
4. Répondez aux remarques du mainteneur et relancez les vérifications après
   correction.

Le workflow Rust regroupe l'hygiène et les dépendances sur Ubuntu, les
contrôles Metal sur macOS, les tests CPU existants, Gitleaks et actionlint.
Les changements purement documentaires évitent les compilations Rust ; le
statut agrégé reste disponible pour le ruleset de protection de `main`.

`release-plz` prépare les versions et le changelog. Ne poussez pas de tag,
ne publiez pas sur crates.io et ne créez pas de release sans demande
explicite du mainteneur.

Pour une vulnérabilité, suivez [SECURITY.md](SECURITY.md) plutôt que d'ouvrir
une issue publique.
