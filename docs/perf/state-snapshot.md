# Instantané disque post-prefill

Outil de développement opt-in pour répéter le decode d'un long prompt sans
recalculer son prefill. Aucun téléchargement ni changement des poids.

```sh
export SARAGOSSA_RUST_STATE_SNAPSHOT=/tmp/saragossa-state
# Exécuter ensuite la commande run/serve habituelle, avec les mêmes ids et flags.
```

Le premier passage crée une entrée `entry-*` par empreinte du prompt et mode,
avec `manifest.json` et un répertoire `state-*` de blobs bruts. Le suivant journalise `state_snapshot hit ... (prefill sauté)` et
restaure l'état avant le semis du decode résident. Enlever le flag désactive
entièrement la lecture et l'écriture. Une API construite depuis des tenseurs
sans chemins de poids refuse ce mode diagnostic.

La clé comprend SHA-256 des ids (u64 little-endian), chemins canoniques,
tailles et mtimes des shards chargés, de l'index et de la configuration s'ils
existent, ainsi que les sidecars MTP et le drafter DFlash. Elle inclut le mode
AR/MTP/DFlash, la configuration, le dtype KV et une empreinte des réglages runtime,
l'endianness et SHA-256 du binaire courant. Chaque blob porte aussi son
SHA-256 et sa taille. Un autre build est explicitement refusé. Une clé
incompatible, un manifeste incomplet ou un blob corrompu déclenche un prefill
normal avec diagnostic. Une erreur d'écriture est propagée.

Les KV gardent leurs bits f32/bf16, leurs segments et leur base absolue pour
les fenêtres glissantes. Seules les lignes vivantes sont écrites, jamais la
réserve de decode. Les états conv/SSM CPU et Metal, leur géométrie et le dtype
SSM sont conservés. La restauration Metal réutilise celle du cache de préfixe.
La dernière ligne normalisée sert au premier token ; MTP conserve son
historique déjà semé, DFlash ses captures de couches et, si nécessaire,
l'historique de bascule vers MTP. Aucun semis ne dépend alors d'un hidden de
prompt disparu. Une génération interrompue avant le semis MTP complet ne
publie pas d'instantané incomplet.

Les blobs sont écrits en fichiers temporaires, synchronisés puis renommés ;
le manifeste est publié atomiquement en dernier. Chaque entrée conserve un
seul manifeste actif ; la bascule DFlash vers MTP garde ainsi le prompt initial. Les anciennes générations restent sur disque pour ne
pas invalider un lecteur concurrent ; supprimer le répertoire hors exécution
pour les purger, ou employer un répertoire par expérience. Aucun poids ni
chemin local n'est ajouté au dépôt. Les instantanés contiennent des états
dérivés du prompt et restent des artefacts locaux.

## Limites et coût

Ne pas utiliser ce mode pour mesurer le prefill, le TTFT ou le pic mémoire.
Le temps de « prefill » inclut l'I/O et les checksums ; la relecture alloue des
buffers CPU et GPU différents d'un prefill normal. Les poids restent chargés
normalement. Le cache n'est pas un checkpoint de génération : il ne conserve
ni sampler en cours, ni tokens déjà décodés, ni arène spéculative en cours.
Changer un réglage runtime invalide volontairement la clé, même si ce réglage
n'affecte que le decode. Aucun résultat d'identité avec poids n'est présumé.

Le KV coûte `2 × tokens × somme(kv_heads × head_dim) × octets_par_valeur`.
Pour le Qwen3.8-27B à 262 144 tokens, 16 couches full-attention, 4 têtes KV
et head_dim 256 : **16 Gio en bf16**, **32 Gio en f32**, hors états linéaires
(~144 Mio de SSM f32 pour 48 couches, plus conv). L'historique MTP f32 ajoute
environ 2 Gio si sa couche utilise les mêmes dimensions. Les captures DFlash
ajoutent `lignes conservées × couches capturées × hidden_size × 4` octets.
Les autres entrées et anciennes générations s'ajoutent à ces chiffres. Mesurer le total réel
avec `du -sh "$SARAGOSSA_RUST_STATE_SNAPSHOT"`.

## Vérification

Sans poids : aller-retour f32/bf16 et états récurrents CPU, captures/historique,
rejet d'un autre build, corruption, et publication interrompue.

```sh
cargo test --lib decoder::state_snapshot::tests
```

Les deux tests d'identité suivants sont **ignorés**, à lancer manuellement
sous le verrou GPU habituel. Ils chargent uniquement le modèle local indiqué
(un petit modèle avec tête MTP peut remplacer le 27B), sans téléchargement.
Chaque mode compare 24 ids greedy entre run complet, écriture et relecture,
avec un nouveau décodeur pour chaque phase et assertion du hit disque.

```sh
SARAGOSSA_TEST_MODEL=/chemin/vers/modele \
SARAGOSSA_RUST_MTP_HISTORY=committed \
SARAGOSSA_RUST_MTP_FRESH_CACHE=0 \
cargo test --lib --features metal \
  decoder::state_snapshot::tests::identity_weights \
  -- --ignored --nocapture --test-threads=1
```

Le filtre exécute AR et MTP. Le test échoue si la tête MTP manque. DFlash est
sérialisé et restauré, mais n'est pas couvert par cet oracle avec poids.
