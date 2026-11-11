# Instantané disque post-prefill

Outil de développement opt-in pour répéter le decode d'un long prompt sans
recalculer son prefill. Aucun téléchargement ni changement des poids.

```sh
export SARAGOSSA_RUST_STATE_SNAPSHOT=/tmp/saragossa-state
# Exécuter ensuite la commande run/serve habituelle, avec les mêmes ids et flags.
```

Le premier passage crée une entrée `entry-*` par empreinte du prompt et mode,
avec `manifest.json` et un répertoire `state-*` de blobs bruts. Le suivant
journalise `state_snapshot hit ... (prefill sauté)` et restaure l'état avant le
semis du decode résident. Le flag est figé au premier accès (OnceLock) : le
définir avant le chargement,
puis redémarrer le processus pour le changer. Sans flag au démarrage, le chemin
instantané ne lit aucun poids ; seule l’empreinte de l’environnement est capturée.
Une API construite depuis des tenseurs sans chemins de poids refuse ce mode
diagnostic.

La clé comprend SHA-256 des ids (u64 little-endian), chemins des sources,
tailles, mtimes et, sous Unix, inode/device et ctimes des shards chargés, de
l'index et de la configuration s'ils existent, ainsi que les sidecars MTP et le
drafter DFlash. Chaque fichier porte aussi une empreinte du **contenu entier** :
SHA-256 des blocs de 4 Mio, puis SHA-256 de leur combinaison ordonnée avec la
taille et le découpage. Cette empreinte est indépendante du nombre de threads.
Une modification en place conservant taille, mtime et inode est donc détectée
au prochain chargement ; la limite de l'ancienne identité par métadonnées est levée.

Le hachage est calculé juste avant la lecture des poids, uniquement si le mode
instantané est actif, sans cache disque d'empreintes. Les métadonnées sont
vérifiées avant/après le hachage (chemin et descripteurs ouverts), après le
chargement, à la relecture et avant toute écriture/publication. Le ctime Unix
permet aussi de refuser une écriture concurrente qui restaure le mtime. Un
changement refuse l'instantané avec diagnostic, garde le prefill normal et
interdit son écriture. Les sources doivent rester immuables pendant leur
utilisation ; sur les plateformes sans ctime Unix, la garde de course reste
limitée aux changements de métadonnées observables. Le format courant est identifié par `FORMAT_VERSION` :
les anciens manifestes sont refusés puis remplacés après un prefill complet.

Les diagnostics qui retirent des calculs désactivent toute lecture et écriture
d'instantané, avec le nom du flag dans le diagnostic :

- `SARAGOSSA_RUST_MOE_COOP_SKIP_GEMM` et `SARAGOSSA_RUST_MOE_COOP_SKIP_SHARED`
  sont actifs par leur seule présence, même avec `0` ou une valeur vide ;
- `SARAGOSSA_RUST_VERIFY_SKIP` est refusé par présence, même hors de la portée
  du bench où il altère effectivement les calculs ;
- `SARAGOSSA_RUST_KV_BF16_SIM_KONLY` et `SARAGOSSA_RUST_KV_BF16_SIM_VONLY`
  sont refusés dès qu'une valeur vraie est observée : ils altèrent le seed KV.

Les alias historiques `RETI_*` suivent les mêmes règles. Les contrôles sont
répétés à la demande, à la lecture et à l'écriture : activer tardivement un flag
MoE ne permet pas de réutiliser ou publier un instantané. Les simulations KV
sont surveillées dès la lecture du flag, même par un accesseur OnceLock.
Le premier diagnostic observé actif rend ce refus définitif pour le processus,
même après retrait du flag et pour les états issus du cache de préfixe mémoire.

Les prompts de warmup CLI et serveur (AR et MTP) ne lisent ni n'écrivent
d'instantané. Une entrée absente au premier run journalise
`state_snapshot absent mode=… tokens=…` ; les entrées invalides restent refusées.

La clé inclut aussi le mode
AR/MTP/DFlash, la configuration, le dtype KV et une empreinte de toutes les
variables `SARAGOSSA_*` et `RETI_*`, triées et capturées une seule fois au
premier accès à un flag runtime. Le profil de l'executor Metal est ajouté
séparément. L'endianness et le SHA-256 du binaire courant complètent la clé ;
ce dernier est mémorisé par processus et couvre aussi les valeurs par défaut. Chaque blob porte aussi son
SHA-256 et sa taille. Avant toute allocation de blob, la géométrie complète
(KV, conv/SSM CPU et Metal, dernière ligne, historique MTP et captures DFlash)
est contrôlée contre le modèle et le nombre de tokens, avec arithmétique vérifiée
et budget cumulé. Chaque segment KV contient des lignes entières. Les lectures
ont une longueur fixe : un fichier qui grossit ne peut pas augmenter l'allocation.
Un autre build est explicitement refusé. Une clé
incompatible, un manifeste incomplet ou un blob corrompu déclenche un prefill
normal avec diagnostic. Toute erreur d'écriture journalise
`state_snapshot refusé : …; sans écriture` et laisse la génération continuer.

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

Un lien symbolique échangé A→B→A pendant le chargement n'est pas détecté :
cet outil de développement local ne doit pas pointer vers des chemins modifiés
en cours d'exécution.

Ne pas utiliser ce mode pour mesurer le prefill, le TTFT ou le pic mémoire.
Le temps de « prefill » inclut l'I/O et les checksums ; la relecture alloue des
buffers CPU et GPU différents d'un prefill normal. Les poids restent chargés
normalement. Le cache n'est pas un checkpoint de génération : il ne conserve
ni sampler en cours, ni tokens déjà décodés, ni arène spéculative en cours.
Changer une variable moteur au démarrage invalide volontairement la clé, même
si ce réglage n'affecte que le decode. L'empreinte conserve l'environnement du
premier accès runtime : définir tous les réglages avant ce premier accès et
redémarrer pour les changer. Les diagnostics tardifs restent surveillés. Aucun résultat
d'identité avec poids n'est présumé.

Le hachage ajoute une lecture complète des sources à chaque chargement, y
compris lors d'un hit. Il utilise `std::thread::scope`, au plus huit threads
(selon le parallélisme disponible), un buffer de 4 Mio par thread et 32 octets
par bloc pour la combinaison : pas de copie entière des poids en RAM. Le coût
CPU/I/O dépend du stockage et du cache du système. Chaque fichier haché produit
`state_snapshot hash_ms=… bytes=…` ; additionner ces lignes pour mesurer le coût
sur un modèle de 16 Go, sidecars compris. Aucun coût sur modèle réel n'a été
mesuré ici. Les vérifications ultérieures dans le même chargement comparent les
métadonnées, sans relire tous les poids à chaque prompt.

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
rejet d'un autre build, corruption et publication interrompue. Les régressions
couvrent aussi le remplacement et la modification en place à taille/mtime
identiques (shards, index, config, sidecars et drafter), le hachage déterministe
avec plusieurs nombres de threads, les courses de métadonnées, le refus des
ablations (y compris alias, valeurs vides et `0`), le refus
d'écriture après modification, le court-circuit sans flag, les réglages figés,
les blobs creux surdimensionnés, le repli après des segments KV mal alignés et
la génération dans un répertoire d'instantanés non inscriptible.

```sh
cargo test --lib decoder::state_snapshot
```

Les deux tests d'identité suivants sont **ignorés**, à lancer manuellement
sous le verrou GPU habituel. Ils chargent uniquement le modèle local indiqué
(un petit modèle avec tête MTP peut remplacer le 27B), sans téléchargement.
Chaque mode compare 24 ids greedy entre run complet, écriture et relecture,
avec un processus neuf pour chaque phase et assertion du hit disque.

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
