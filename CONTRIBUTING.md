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

## Préparation et regroupement des releases

Le workflow `Release-plz` prépare la PR de version après un succès du workflow
`Rust` sur `main`. Il ignore les succès dont le SHA ne correspond plus au
checkout de `main`. Un lancement manuel est possible sur `main`, mais exige
également une CI `Rust` verte pour ce SHA.

`release-plz release-pr` prépare ou actualise la PR ouverte `release-plz-*`.
Les changements s'y accumulent jusqu'à sa fusion. Le filtre `release_commits`
exige au moins un commit `feat`, `fix`, `refactor` ou `perf` correspondant à
l'expression configurée ; il ne définit aucune fréquence de publication.
Les commits filtrés peuvent néanmoins apparaître dans le changelog.
Voir la [configuration de release-plz](https://release-plz.dev/docs/config#the-release_commits-field)
et le [fonctionnement des PR de release](https://release-plz.dev/docs/usage/release-pr).

**Le label `release` seul n'active pas l'auto-merge.** Le mainteneur ajoute
`release-ready` lorsque le lot est prêt. Le workflow active alors l'auto-merge
avec un commit de merge, après les checks requis. Retirer `release-ready` ou
`release` désactive l'auto-merge tant que la PR n'est pas fusionnée. Ne pas
attribuer le label générique `automerge` aux PR de release.

Sans cette décision, la PR reste ouverte pour regrouper les contributions.
Un auto-merge systématique dès l'ajout de `release` pourrait au contraire
produire une release par merge éligible de `main` : release-plz n'impose ni
délai ni nombre minimal de commits. `release_always = false` réserve la
publication à une PR de release fusionnée ; ce réglage ne crée pas de délai.

La PR de release conserve la CI ordinaire et compile en plus le binaire avec
`--locked --release --target aarch64-apple-darwin --bin saragossa --features metal,devtools`,
puis exécute `--help`. Le job `gpu-synthetic` est sauté sur `release-plz-*` :
ces branches sont réservées aux versions, au changelog et au lock du workspace,
sans changement des dépendances externes ni du code. Les tests nocturnes et
manuels restent actifs.

Seule la fusion d'une PR `release-plz-*` du dépôt, étiquetée `release`, lance
le job de tag, à partir de son commit de merge. Un merge ordinaire de `main`
ne pose aucun tag. Le tag `v*` déclenche ensuite le workflow `Release` qui
recompile, publie l'archive, actualise le tap Homebrew et teste l'installation.
Aucun paquet n'est publié sur crates.io.

### Reprise après une publication partielle

Le workflow `Release` n'est pas idempotent après `gh release create` : une
relance du job `release` échoue si la release existe déjà. Inspecter les
jobs et les assets publiés avant toute reprise.

- Si l'archive, sa somme SHA256, le SBOM et `CHECKSUMS.md` sont présents et
  valides, conserver la release. Si seul `homebrew` ou `homebrew-test` a
  échoué, relancer les jobs échoués depuis l'interface Actions, sans rejouer
  le job `release`. Vérifier d'abord que le tap n'a pas avancé vers une
  version plus récente ; une reprise ancienne ne doit pas le rétrograder.
- Si la release est incomplète ou si le job `release` doit être rejoué,
  supprimer manuellement la release et ses assets après vérification :

  ```bash
  gh release delete vX.Y.Z --repo azerozero/saragossa --yes
  ```

  Remplacer `vX.Y.Z` par le tag concerné. **Conserver le tag Git** : ne pas
  utiliser `--cleanup-tag`, supprimer ou déplacer le tag, ni recréer une PR
  de version. La suppression retire les liens publics de téléchargement
  jusqu'à la nouvelle publication ; si le tap référence déjà cette version,
  ses installations seront indisponibles pendant cette fenêtre.
  Relancer ensuite le job `release` et ses dépendants via l'interface
  Actions. Si l'artefact `release-assets` a expiré, relancer tous les jobs
  pour le reconstruire avant la publication.

Après reprise, contrôler les assets, leurs sommes SHA256 et l'installation
Homebrew. Ces opérations sont manuelles et réservées au mainteneur ; les
commandes ci-dessus ne font pas partie de la préparation d'une contribution.

## Réglages GitHub à appliquer par le mainteneur

- Autoriser les merge commits et l'auto-merge du dépôt.
- Protéger `main` avec le statut agrégé **`Required checks`** du workflow
  `Rust`, sans contournement pour le compte de livraison. Ce job dépend de
  `release-build-check` : son échec bloque la fusion et son statut `skipped`
  sur les PR ordinaires reste non fatal. Ne pas ajouter
  **`Release build (Apple Silicon)`** comme check requis séparé ; retirer
  cette exigence si elle avait déjà été configurée. Conserver les autres
  protections et contrôles concernés (GPU, CodeQL et documentation).
- Créer les labels `release` et `release-ready`. Seul `release` figure dans
  `pr_labels` ; `release-ready` reste une décision du mainteneur. Désactiver
  l'auto-merge des éventuelles PR de release déjà ouvertes avant ce changement
  si leur lot n'est pas prêt.
- Configurer `RELEASE_PLZ_TOKEN` avec les droits de création/mise à jour des
  PR, de fusion et d'écriture des tags dans ce dépôt, ainsi que d'écriture
  dans `azerozero/homebrew-tap`. Utiliser un PAT ou un jeton d'application
  permettant aux PR et tags créés de déclencher les workflows suivants.
- Autoriser les workflows à créer des PR selon la politique du dépôt et de
  l'organisation. Le job de préparation utilise aussi `actions: read` pour
  vérifier la CI lors d'un lancement manuel.

Ces réglages sont externes aux fichiers versionnés. Modifier les workflows
localement ne les applique pas. Le déclencheur `workflow_run` devient actif
une fois le workflow présent sur la branche par défaut.
