# Mémoire Qwen3-TTS

Le talker reçoit des embeddings projetés, jamais des identifiants texte. Son
champ d'embedding générique ne sert donc pas à la synthèse, mais l'arène Metal
l'uploadait quand même. Pour une table `[151936, 2048]`, cela conservait deux
copies inutilisées de 1 244 659 712 octets chacune : CPU et GPU. Une table d'une
ligne remplace ce champ ; la table texte effectivement utilisée reste intacte.

Le codec ne charge plus `encoder.*`. L'encodage des références de clonage
appartient au chargeur Mimi séparé. Après un upload réussi, les poids CPU du
vocodeur sont libérés. Ceux du préfixe le sont aussi si son chemin GPU est actif.
Les codebooks RVQ utilisés sur CPU restent présents. Le premier accès CPU à un
poids libéré le relit depuis le snapshot local et vérifie son SHA-256 avant de
le mémoriser. Une modification du fichier provoque une erreur explicite.
Sans codec GPU, les poids CPU restent chargés. Le chemin glouton recharge son
préfixe CPU historique ; aucun calcul flottant ni kernel n'est remplacé.

## Kill-switches

Chaque option est active par défaut et se décide au chargement du modèle.

| Variable à `0` | Comportement rétabli |
|---|---|
| `SARAGOSSA_TTS_COMPACT_TALKER_EMBEDDING` | Copies CPU et GPU du vocabulaire dans le talker |
| `SARAGOSSA_TTS_CODEC_DECODER_ONLY` | Chargement des poids d'encodeur dans le codec |
| `SARAGOSSA_TTS_CODEC_LAZY_CPU` | Conservation des poids CPU après l'upload Metal |

## Relevés et parité

`SARAGOSSA_TTS_MEMORY_PROFILE=1` journalise une fois chaque phase de chargement,
prefill, création des arènes et premier PCM. `metal_allocated` provient de
`currentAllocatedSize`. Les caches de poids, concaténations et scratch utilisent
`allocatedSize`, avec déduplication des poignées dans chaque executor. Les poids
quantifiés déjà partagés ne sont pas tous dans le cache d'upload : ne pas
assimiler sa somme au total Metal. Les arènes indiquent séparément leurs
allocations persistantes et leurs scratch ; les KV CPU sont comptés à part.
La phase `codec_stream` compte séparément le pool temporaire du codec, ses
contextes de convolution et ses KV avec `allocatedSize`.

`SARAGOSSA_TTS_MEMORY_PROBE=1` ajoute `footprint` à chaque phase. Ce mode perturbe
les temps : le désactiver pour les séries de latence. L'embedder du serveur est
paresseux et n'est pas chargé par une requête de synthèse seule.

L'exemple `tts_pcm_digest` accepte un snapshot local, une voix (ou `-` pour
VoiceDesign) et un corpus JSON associant noms et textes. Il émet les SHA-256 du
PCM **f32 little-endian avant conversion HTTP**, en glouton et en échantillonné
avec la graine 0. Deux textes sur Serena, Ryan et VoiceDesign donnent 12 SHA.
Comparer les sorties avec les trois kill-switches à `0`, puis avec leurs
valeurs par défaut. Les tests ignorés couvrent aussi les références CPU/Metal.

Pour la performance, démarrer un serveur neuf par série, chauffer avec un texte
court puis le premier texte, et mesurer cinq répétitions des deux textes dans
l'ordre ABBA. `time -l` couvre aussi le chargement. Acquérir le verrou GPU et
vérifier `python3 -I -c "import os,sys; sys.exit(0 if os.getloadavg()[0]<2 else 1)"`
avant chaque série, y compris après l'acquisition du verrou. Conserver séparément
pic RSS, pic footprint, RTF et TTFA ; les octets Metal font partie de la mémoire
unifiée du processus et ne s'ajoutent pas au footprint.

## Référence mesurée

Référence `3db0afc`, M5 Max (18 cœurs, 128 Gio), macOS 26.5.1. Un processus
`serve` neuf reproduit 7 613 700 072 octets de pic footprint, sans sondes.
Le relevé détaillé ci-dessous porte sur Serena, C400, graine 0, au premier
PCM ; les sondes sont activées dans un processus distinct. Les valeurs CPU
des tables viennent de leurs capacités et les valeurs GPU des compteurs Metal.

| Poste avant optimisation | Mo décimaux | Observation |
|---|---:|---|
| Table texte CPU utilisée | 1 244,7 | Conservée pour les projections texte |
| Copie de cette table dans le talker, CPU | 1 244,7 | Jamais consultée par la génération TTS |
| Copie de cette table dans le talker, GPU | 1 244,7 | Upload de l'arène générique |
| Talker, représentation des couches et tête côté CPU | 1 239,1 | Inventaire des poids après conversion ; recouvre le stockage partagé |
| Talker, upload des poids quantifiés | 1 151,9 | Delta Metal au chargement ; stockage déjà partagé |
| Code predictor, upload initial | 65,6 | Delta Metal ; tables et têtes supplémentaires ci-dessous |
| Code predictor, couches et projection côté CPU | 70,7 | Inventaire après conversion |
| Code predictor, têtes côté CPU | 27,5 | Inventaire après conversion |
| Embeddings des 15 codebooks, CPU | 251,7 | Utilisés par le chemin de référence |
| Embedding des codes et projection texte, CPU | 32,5 | Tables effectivement utilisées |
| Cache GPU du code predictor actif | 144,8 | Inclut tables et têtes ; hors upload initial |
| QKV concaténé du code predictor | 17,0 | Aucun concat talker dans ce chemin |
| Codec CPU | 681,8 | Inclut l'encodeur audio inutilisé par la synthèse |
| Codec GPU au chargement | 422,6 | Delta Metal |
| KV talker CPU | 58,7 | Cache du prefill, distinct de la réserve GPU |
| KV talker GPU | 184,4 | Réserve bornée par le prompt et le cap serveur |
| KV code predictor GPU | 0,8 | Séquence courte, remise à zéro entre trames |
| Arènes persistantes hors KV | 0,10 | Talker et code predictor |
| Scratch des arènes et du backend | 1,11 | Après la première trame, hors pool du codec |
| Premier chunk PCM f32 | 0,031 | Quatre trames ; hors accumulation de la réponse |
| Tokenizer | 66,1 | Delta footprint du chargement BPE |
| Embedder texte du serveur | 0 | Aucun appel à l'endpoint embeddings |

Ces postes ne forment pas une somme du footprint : les poids unifiés sont
accessibles sur CPU et GPU, et le pilote comme l'allocateur ont leur propre
empreinte. Au premier PCM, `footprint` attribue 3 721,3 Mo à `MALLOC_LARGE`,
452,5 Mo à `MALLOC_SMALL`, 3 259,3 Mo à `IOAccelerator` et 125,8 Mo au graphique
non mappé, pour 7 569,9 Mo physiques. Le pic non instrumenté fait foi pour l'ABBA.
Le contenu PCM accumulé est borné à 6,144 Mo au cap de 800 trames ; la capacité
du `Vec` peut être supérieure avec sa croissance géométrique. Le canal PCM
peut conserver des chunks en attente, dans la même borne de contenu, et les
codes représentent au plus 51,2 ko hors métadonnées des vecteurs. Les arènes et le QKV
restent inchangés : leur coût ne justifie pas de modifier le chemin de calcul
pour corriger les deux copies de vocabulaire de 2 489,3 Mo au total.
