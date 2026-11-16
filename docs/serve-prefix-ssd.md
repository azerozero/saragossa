# Cache de préfixe persistant de `serve`

Activation explicite, sans effet sur un serveur lancé sans répertoire SSD :

```sh
saragossa serve --model qwen="$MODEL" \
  --prefix-cache-ssd-dir "$CACHE" --prefix-cache-ssd-gib 32 \
  --prefix-cache-ssd-priority-tokens 4096
```

`CACHE` est un répertoire **dédié et privé**, créé en mode 0700 s'il est
absent. Un répertoire existant accessible au groupe ou aux autres est refusé.
Les variables équivalentes sont `SARAGOSSA_SERVE_PREFIX_SSD_DIR`,
`SARAGOSSA_SERVE_PREFIX_SSD_GIB` et
`SARAGOSSA_SERVE_PREFIX_SSD_PRIORITY_TOKENS`. Les options CLI ont priorité.
Le budget est exprimé en Gio entiers, strictement positifs ; 32 par défaut.
Un verrou OS exclusif interdit deux serveurs dans le même répertoire et se
libère après la mort du processus. Ne pas modifier ce répertoire en service.

Le budget porte sur la somme des tailles logiques des fichiers publiés et en
cours de publication, hors métadonnées du système de fichiers. Il est global
au serveur, tous modèles et sessions confondus. Le LRU élimine d'abord les
entrées ordinaires, puis les entrées prioritaires. Les préfixes d'au plus
4096 tokens sont prioritaires : servir un prompt système seul permet de
conserver cette frontière. Zéro désactive cette priorité. Elle ne dépend
d'aucune valeur de priorité reçue du réseau. Une nouvelle entrée ordinaire
ne déloge pas une entrée prioritaire pour se faire une place.

## Chemins de reprise

Le cache AR de `serve` recherche d'abord sa meilleure frontière en RAM, puis
une frontière plus longue sur SSD. Seules les longueurs présentes dans l'index
persistant sont sondées. La frontière retenue retourne en RAM et le suffixe
suit le chemin d'extension existant. Les états nouvellement préremplis ou
étendus sont proposés à l'écriture immédiatement : leur éviction RAM ou un
redémarrage n'impose donc pas de refaire le prefill après publication.
La taille des blocs, le seuil de rentabilité d'extension et les gardes mémoire
restent ceux du cache existant.

MTP et DFlash réutilisent les points d'instantané du décodeur : KV du tronc,
états GatedDeltaNet/conv, dernière ligne cachée, historique MTP committed
s'il existe et captures DFlash. Leur lookup disque porte sur le prompt exact ;
leur cache RAM conserve la recherche de préfixe et l'extension existantes.
Les modes AR, MTP fresh/committed et DFlash ont des clés distinctes.
Les prompts de warmup et les ablations diagnostiques ne sont pas persistés.

Les sessions sont isolées par SHA-256, sans chemin fourni par une requête.
L'absence de session conserve un espace partagé. Cette isolation concerne
le tier disque ; elle ne change pas la politique historique du cache RAM MTP.

## Intégrité et publication

Le format post-prefill version 5 conserve les précisions existantes et ajoute
le delta RoPE signé et le contexte positionnel des oracles vision internes.
Le format v4 est refusé ; les transports média restent fermés.
La clé inclut les tokens exacts, le format KV, les réglages numériques, le
binaire, la configuration et les sources du modèle. Les sources comprennent
**tous les octets des shards**, les sidecars, le tokenizer et sa configuration :
le hachage intégral existant est plus strict qu'un échantillonnage de poids.
Il est effectué au chargement et exclu des mesures de reload/prefill. Les
métadonnées des sources sont revérifiées avant l'écriture et la restauration.
Changer de build ou de réglages numériques crée une autre identité ; les
anciennes entrées incompatibles avec le build/format sont supprimées au démarrage.
L'empreinte exclut `SARAGOSSA_SERVE_PREFIX_SSD_*` et leurs alias `RETI_` :
changer le répertoire, le budget, la priorité ou le staging ne modifie pas l'état.
Elle exclut également cette liste fermée de suffixes, pour les deux préfixes
`SARAGOSSA_` et `RETI_` :

- `SERVE_LOG_PROMPT_FINGERPRINT`, `SERVE_LRU`, `SERVE_MODEL_POOL` ;
- `SERVE_MAX_CONNECTIONS`, `SERVE_QUEUE`, `SERVE_MAX_INFLIGHT_BYTES` ;
- `SERVE_OOM_GUARD`, `SERVE_MEMORY_CAP_BYTES`, `SERVE_MEMORY_HEADROOM_BYTES` ;
- `SERVE_PREFIX_CACHE_BYTES`, `SERVE_PREFIX_CACHE_BLOCKS`,
  `SERVE_PREFIX_BLOCKS_PER_SESSION`.

Ces variables règlent le transport, les diagnostics ou l'admission des caches.
Par prudence, tout autre flag moteur reste dans la clé, notamment les variables
inconnues, le batching et `SERVE_PREFIX_BLOCK_TOKENS`.

Chaque blob et le manifeste portent un SHA-256. Avant toute allocation de
blob, le lecteur valide la clé complète, les dimensions par couche, les
longueurs, les dtypes et le budget géométrique du modèle. Le manifeste est
limité à 16 Mio ; la taille totale de l'entrée est limitée au budget SSD.
Les liens sortant du répertoire et les chemins de blobs arbitraires sont
refusés. Une entrée invalide (format, dimensions, identité, checksum) est
supprimée et le prefill ordinaire reprend. Une erreur d'E/S en lecture, y compris
un fichier momentanément inaccessible, conserve au contraire l'entrée et sa
comptabilisation pour une nouvelle tentative. Les erreurs de ressources Metal
sont également conservées : elles ne prouvent pas que les octets sont corrompus.
Cette politique évite de détruire un cache sain lors d'une indisponibilité ;
aucun état partiellement lu n'est exposé. Le nettoyage de démarrage reste
conservateur et peut éliminer les entrées impossibles à valider à l'ouverture.
Les fichiers ne sont pas chiffrés : ils contiennent des états dérivés des
prompts, à conserver dans un emplacement privé choisi par l'opérateur.

Un seul instantané est admis entre le début de sa préparation et sa publication.
Avant toute copie ou blit, le moteur estime les blobs (KV, récurrence, historique
MTP et captures) et leur mémoire temporaire. Il refuse une estimation dépassant
le budget SSD, l'espace libre actuel ou le plafond RAM de staging. Par défaut,
ce dernier vaut **un quart de la RAM physique**, ou un quart de
`SARAGOSSA_SERVE_MEMORY_CAP_BYTES` si ce budget explicite est inférieur.
Si seule une de ces valeurs est connue, elle sert de base ; si aucune n'est
connue, le repli est 512 Mio. Sur une machine de 128 Gio sans budget explicite,
le plafond est donc **32 Gio** : il admet les instantanés d'environ 5,3 Go à
32k tokens et 21,2 Go à 128k, avec leurs temporaires. Un seul staging peut
occuper ce quart ; le reste demeure disponible pour les autres états moteur.
`SARAGOSSA_SERVE_PREFIX_SSD_STAGING_MIB` remplace ce calcul (Mio entiers
strictement positifs), par exemple `SARAGOSSA_SERVE_PREFIX_SSD_STAGING_MIB=32768`
fixe 32 Gio. Ce plafond RAM inclut une
marge conservatrice de 128 Mio pour les manifestes et métadonnées, ainsi que
les copies temporaires Metal ; le majorant disque réserve 16 Mio de manifeste.
Une réservation `MemoryGuard` avec les limites de `serve` reste détenue jusqu'à
la fin du travail du worker. Le plafond de staging ne garantit donc pas
l'admission si l'empreinte du processus laisse trop peu de mémoire disponible,
même avec un réglage explicite élevé. **La copie admise reste synchrone**. Le contrôle
d'espace libre est préventif : une écriture concurrente externe peut encore
faire échouer la publication.
Le worker calcule les sommes de contrôle et réalise toutes les écritures de
blobs, synchronisations et renommages. Il publie un dossier `pending` complet
par `rename`, après `fsync` des fichiers et dossiers. Un arrêt brutal avant
publication laisse un temporaire supprimé au redémarrage.

Les refus sont mémorisés avant toute nouvelle copie : au plus 1024 noms
hachés et raisons, pendant 60 secondes, sans renouveler le délai sur un hit.
Cela couvre les plafonds, la priorité et les erreurs du worker. Après une
erreur de publication, l'écrivain reste arrêté jusqu'au redémarrage : aucun
hit RAM ne prépare alors un nouvel instantané.

Une file occupée fait abandonner la nouvelle candidature, sans attendre.
Pendant la publication, un lookup disque indisponible repart au prefill ;
aucune attente de l'écrivain n'est imposée au moteur. La persistance est donc
best effort sous charge. Attendre `prefix_ssd written` avant d'arrêter le
serveur pour vérifier une reprise à froid. Une erreur disque est journalisée
et ne fait pas échouer une génération. Le cache complet n'est ni compressé
ni mappé sans copie à la lecture ; les buffers Metal sont reconstruits par
le lecteur d'instantané existant.

## Vérification et mesure

Les tests synthétiques couvrent les bits KV et récurrents, les tokens greedy,
la reprise dans des processus distincts, les checksums, versions, dtypes,
modèles incompatibles, tailles hostiles, interruptions, LRU/priorités,
concurrence et limites d'admission. Les tests existants du format couvrent
aussi l'historique MTP et les captures DFlash. Le test de reprise AR vérifie
l'isolation A/B via `load_persistent_prefix`, puis le retour à A.
L'oracle ignoré `synthetic_metal_bf16_greedy_survives_restart` ne requiert aucun
poids externe : il compare les identifiants greedy sur Metal après réouverture
du store et reconstruction du modèle, avec KV bf16 partagé et récurrence GPU.

```sh
python3 -I -B tools/gpulock.py run --owner prefix-ssd-test --prio 10 -- \
  cargo test --all-features synthetic_metal_bf16_greedy_survives_restart \
  -- --ignored --nocapture
```

```sh
cargo test --all-features decoder::state_snapshot
cargo test --all-features serve::args
cargo build --release --example prefix_ssd
```

Mesure à **32768 et 131072 tokens exacts** sur le modèle local choisi, sous le
verrou GPU du dépôt (aucun téléchargement). Définir `MODEL` sur le répertoire
Qwen3.8-27B-u4ls local. Le staging par défaut sur 128 Gio couvre ces tailles ;
sur une machine plus petite ou avec un budget mémoire réduit, ajuster au besoin
`SARAGOSSA_SERVE_PREFIX_SSD_STAGING_MIB` selon la mémoire disponible.
Vérifier `prefix_ssd written` avant d'interpréter une reprise. La garde mémoire
reste applicable même avec un plafond relevé.
Le dossier de sortie doit être nouveau :

```sh
python3 -I -B tools/gpulock.py run --owner prefix-ssd --prio 10 -- \
  python3 -I -B tools/bench_prefix_ssd.py \
  --model "$MODEL" --mode ar --output /tmp/prefix-ssd-ar

python3 -I -B tools/gpulock.py run --owner prefix-ssd --prio 10 -- \
  python3 -I -B tools/bench_prefix_ssd.py \
  --model "$MODEL" --mode mtp --output /tmp/prefix-ssd-mtp
```

Le script démarre trois processus par longueur : sans SSD, SSD vide, puis
relecture après redémarrage. Il exige l'égalité exacte des 32 identifiants
produits et une réutilisation dans la phase chaude. Il conserve les JSON et
journaux, affiche prefill, prefill avec staging, reload et ratio. Si le binaire
échoue, le script affiche aussi son journal (stdout et stderr) sur stderr,
avec la longueur, la phase et le chemin du journal. Les fichiers
SSD temporaires sont supprimés ensuite. L'oracle MTP désactive un drafter
DFlash implicite pour mesurer l'historique MTP. Aucun gain sur le vrai modèle
n'est présumé par les tests synthétiques.

L'exemple mesure les mêmes APIs de prefill/reprise que `serve`, sans transport
HTTP ni template chat, avec un prompt synthétique répété. Pour mesurer aussi
le transport, lancer le serveur avec les options ci-dessus, puis :

```sh
saragossa bench-serve --model qwen --requests 1 --concurrency 1 \
  --prompt-tokens 32768 --max-tokens 32 --timeout-secs 1800
```

Attendre la publication, redémarrer le serveur avec les mêmes options et
répéter la commande ; recommencer à 131072. Le client HTTP utilise une longueur
approximative : prendre `prompt_tokens` du journal serveur comme référence.
Comparer `prefill_ms` et `prefix_ssd hit ... reload_ms`, pas le chargement des
poids. Le cache de pages de l'OS peut accélérer les relectures : ce protocole
mesure une reprise serveur, pas un SSD physiquement froid. Pour tester une
machine après redémarrage OS, conserver un cache opérateur et les mêmes
réglages/binaire ; ne pas purger les caches système pendant un autre travail.
