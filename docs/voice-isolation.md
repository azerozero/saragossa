# Isolation et nettoyage de la voix

Ce document couvre trois modèles évalués en novembre 2026 :

| Modèle | Rôle | État dans saragossa |
|---|---|---|
| DeepFilterNet3 | débruitage temps réel, 48 kHz | porté en Rust (CPU), oracle PyTorch, CLI `saragossa enhance`, `POST /v1/audio/enhance` |
| MossFormer2_SS_16K | séparation de deux locuteurs, 16 kHz | spécification et mesures sur la référence PyTorch (CPU et MPS) ; pas encore porté |
| NVIDIA-NemotronLabs-RE-USE | restauration universelle, hors ligne | spécification et estimation de coût ; pas porté |

Les mesures ont été faites sur un M5 Max chargé par d'autres travaux (charge
moyenne 35 à 55) : les médianes sont fiables, les queues (p99) sont
pessimistes.

## Licences

- **DeepFilterNet3** : code et poids `Rikorose/DeepFilterNet`, double licence
  MIT ou Apache-2.0 au choix. Le port Rust de `src/denoise/` suit
  `libDF/src/lib.rs` et `df/deepfilternet3.py`. Redistribuer les poids exige de
  joindre l'une des deux licences et la notice de copyright
  (« Copyright (c) 2021 Hendrik Schröter »).
- **MossFormer2_SS_16K** : poids `alibabasglab/MossFormer2_SS_16K` et code
  ClearerVoice-Studio sous Apache-2.0. Une redistribution doit garder la
  licence et signaler les modifications. Le modèle dépend aussi de
  `rotary-embedding-torch` (MIT) pour les positions rotatives.
- **RE-USE** : `nvidia/NVIDIA-NemotronLabs-RE-USE` sous OpenMDW-1.1, usage
  commercial permis. Garder la licence et les notices d'origine à la
  redistribution ; la licence tombe en cas d'action en contrefaçon contre le
  modèle. Ne pas confondre avec `nvidia/RE-USE` et `nvidia/Real-time_RE-USE`
  (NSCLv1, non commerciaux).

## DeepFilterNet3

### Spécification

Tous les paramètres viennent de `config.ini` du checkpoint officiel
(`model_120.ckpt.best`, 2 167 954 paramètres flottants).

**Signal et STFT** (`libDF`) :

- 48 kHz mono, FFT et fenêtre de 960 échantillons (20 ms), pas de 480 (10 ms) ;
- fenêtre de Vorbis `w[n] = sin(π/2 · sin²(π(n + ½)/480))`, dont la somme des
  carrés vaut 1 à 50 % de recouvrement ;
- STFT « temps réel » : la trame `t` concatène les 480 échantillons du hop
  précédent et ceux du hop `t`, sans centrage. L'analyse multiplie par
  `2H/N² = 1/960` ; la synthèse est une iFFT non normalisée, fenêtrée puis
  sommée par overlap-add ;
- 481 bins.

**Features d'entrée**, avec `α = round(exp(−H/sr/τ), 3) = 0,99` (τ = 1 s) :

- 32 bandes ERB (largeurs `libDF::erb_fb`, au moins deux bins par bande,
  vérifiées contre la banque `erb_fb` du checkpoint). Pour chaque bande :
  puissance moyenne, `10·log10(p + 1e-10)`, puis normalisation par moyenne
  exponentielle `s ← α·s + (1 − α)·x`, `(x − s)/40`, état initial
  `linspace(−60, −90)` ;
- 96 premiers bins complexes, normalisés par `x / sqrt(s)` avec
  `s ← α·s + (1 − α)·|x|`, état initial `linspace(0,001, 0,0001)`.

**Réseau** (activations `[canal][fréquence]` par trame, toutes les
convolutions causales en temps, BatchNorm repliée en évaluation, `eps = 1e-5`) :

| Bloc | Détail |
|---|---|
| `pad_feat` | les features entrent avec deux trames d'avance : le pas réseau τ lit `feat[τ + 2]` |
| `erb_conv0` | conv 3×3 (temps × fréquence), 1 → 64, BN, ReLU |
| `erb_conv1..3` | conv en profondeur 1×3 (stride fréquentiel 2, 2, 1) + 1×1, BN, ReLU : 32 → 16 → 8 → 8 bins |
| `df_conv0` | conv 3×3 en 2 groupes, 2 → 64 + 1×1, BN, ReLU, sur 96 bins |
| `df_conv1` | 1×3 en profondeur stride 2 + 1×1 : 96 → 48 bins |
| `df_fc_emb` | linéaire groupée (32 groupes) 3 072 → 512, ReLU, ajoutée à `e3` aplati `[F][C]` |
| `enc.emb_gru` | linéaire groupée 512 → 256 (16 groupes, ReLU), GRU 256, linéaire 256 → 512 (ReLU) |
| `lsnr_fc` | `sigmoid(W·emb) · 50 − 15` (SNR local, non utilisé pour la sortie) |
| `erb_dec` | GRU 2 couches (même enveloppe), puis U-Net : `convt3(conv3p(e3) + emb)`, `convt2(conv2p(e2) + ·)`, `convt1(conv1p(e1) + ·)`, `conv0_out(conv0p(e0) + ·)` ; `convt1/2` sont des convolutions transposées en profondeur 1×3 stride 2 (padding 1, output_padding 1) + 1×1 ; `conv0_out` finit par une sigmoïde |
| masque | 32 gains ERB étendus à 481 bins par la matrice 0/1 `erb_inv_fb` |
| `df_dec` | linéaire groupée 512 → 256 (8 groupes, ReLU), GRU 2 couches, plus `df_skip` (512 → 256) ; `tanh(df_out)` donne 96 × 10 valeurs, auxquelles s'ajoute `df_convp` (conv 5×1 en 2 groupes sur `c0`, + 1×1, BN, ReLU) |
| deep filtering | pour les 96 premiers bins : `Y[τ, f] = Σₙ X[τ − 2 + n, f] · C[n, τ, f]`, n = 0..4, produit complexe sans conjugaison |
| sortie | bins 96 à 480 : `X[τ, f] · gain[f]` ; bins 0 à 95 : sortie du filtre profond |

Le post-filtre (`mask_pf`) est désactivé dans cette configuration.

**Latence algorithmique.** Un hop d'entrée `t` produit la trame de sortie
`τ = t − 2`, dont l'overlap-add rend les échantillons du hop `τ − 1`. Le retard
entrée → sortie vaut donc `N − H + 2H = 1 440` échantillons, soit **30 ms**,
plus l'attente du hop complet (jusqu'à 10 ms) : 40 ms au total, comme annoncé
par l'article. Le temps de calcul s'ajoute à ces 40 ms.

### Implémentation

`src/denoise/` :

- `dsp.rs` : fenêtre, largeurs ERB, STFT et iSTFT incrémentales (rustfft),
  normalisations ;
- `layers.rs` : conv 2D causale par trame (groupée, stride fréquentiel,
  transposée), linéaire groupée, GRU PyTorch, BatchNorm repliée ;
- `network.rs` : graphe DFN3 ;
- `mod.rs` : `DeepFilterNet3::load`, `enhance` (hors ligne, sans retard,
  équivalent de `df.enhance(pad=True)`) et `DfStream::process_hop` (temps réel,
  un hop de 10 ms à la fois).

Le chargeur vérifie la forme de chaque poids, refuse un tenseur inconnu et
contrôle que la banque ERB du checkpoint correspond aux largeurs de libDF.

Conversion des poids (lecture `weights_only=True`, aucun code du checkpoint
n'est exécuté) :

```bash
python3 -I -B tools/voice/convert_dfn3.py \
  DeepFilterNet3/checkpoints/model_120.ckpt.best dfn3.safetensors
saragossa enhance --model dfn3.safetensors --input bruit.wav --output propre.wav
```

L'entrée est rééchantillonnée à 48 kHz et la sortie reste à 48 kHz.

`saragossa serve --enhance-model dfn3.safetensors` ajoute
`POST /v1/audio/enhance` (multipart, champ `file`), qui renvoie un WAV mono
48 kHz en PCM 16 bits.

### Oracle

`tools/voice/dfn3_ref.py` construit la référence : les classes PyTorch
officielles (`Encoder`, `ErbDecoder`, `DfDecoder`, `DfNet`, `Conv2dNormAct`,
`SqueezedGRU_S`, `GroupedLinearEinsum`, `DF`…) sont extraites telles quelles
par AST du dépôt `Rikorose/DeepFilterNet`, et le state dict est chargé en
`strict=True`. Le paquet `libdf` (extension Rust/PyO3) n'étant pas installé,
ses features et son STFT sont réécrits en numpy à partir de `lib.rs`.

Tolérances figées avant la première comparaison (test
`denoise::tests::dfn3_matches_pytorch_reference`, ignoré sans
`SARAGOSSA_DFN3_DIR`) : features ≤ 1e-4 en absolu, masque ≤ 1e-3, audio
≥ 60 dB de SNR et ≤ 1e-3 en absolu. Justification : tout est en f32, seul
l'ordre des sommes change (huit accumulateurs contre BLAS) sur des produits de
3 072 termes et cinq GRU récurrents sur 300 trames ; l'erreur attendue est de
l'ordre de 1e-5 en relatif, alors qu'une erreur de disposition des poids donne
au mieux −20 dB.

Résultat sur 3 s de parole FLEURS + bruit rose et ronflement à 5 dB :

| Grandeur | Écart max |
|---|---|
| features | 5,7e-7 |
| masque ERB | 1,5e-6 |
| audio de sortie | 6,0e-8 (SNR 134,6 dB contre la référence) |

### Latence par trame de 10 ms

| Chemin | Coût par hop de 10 ms (p50) | Facteur temps réel |
|---|---|---|
| CPU, un thread, port Rust (`dfn3_hop_latency`) | 624 µs (p99 4,5 ms sous charge 55) | 0,062 |
| CPU, fichier entier via `saragossa enhance` (100 clips) | ≈ 500 µs | 0,050 |
| GPU, command buffer vide (commit + attente) | 15 µs | — |
| GPU, sonde à 35 noyaux dépendants de même forme que DFN3 (`dfn3_gpu_hop_latency_probe`) | 306 µs au mur, dont 138 µs de temps GPU | 0,031 |

Sur GPU, le coût n'est donc pas nul. Chaque hop exige environ 35 noyaux
dépendants dans un command buffer, puis une attente : on paie environ 0,3 ms
au mur, soit la moitié du chemin CPU. La sonde ne calcule pas le modèle : elle
reproduit son graphe de dépendances et ses MAC (3,55 M par hop) avec des
produits matrice-vecteur naïfs. Elle donne un ordre de grandeur, pas un port
validé. Dans les deux cas, la latence perçue reste dominée par la latence
algorithmique de 40 ms (30 ms de retard plus le hop en cours de remplissage).
Un port GPU ferait gagner environ 0,3 ms sur 40, et se battrait avec le LLM
pour la file Metal : le CPU reste le bon choix.

### Qualité

Corpus : 50 clips FLEURS (25 en, 25 fr), rééchantillonnés à 48 kHz et mélangés
à un bruit synthétique déterministe (bruit rose plus ronflement à 100 Hz
modulé), à 0 et 5 dB de SNR (`tools/voice/mix_noise.py`). Débruitage par la
CLI Rust. Aucun script PESQ ou DNSMOS n'est disponible localement : la mesure
retenue est le SI-SDR contre le signal propre.

| SNR d'entrée | SI-SDR bruité | SI-SDR débruité | Gain moyen | Gain minimal |
|---|---|---|---|---|
| 0 dB | −0,01 dB | 10,63 dB | +10,64 dB | +4,15 dB |
| 5 dB | 5,00 dB | 13,36 dB | +8,36 dB | −0,24 dB |

WER de Whisper large-v3-turbo (mlx, langue imposée, température 0) avant et
après débruitage, sur les mêmes 100 clips. Le WER de chaque clip est plafonné
à 100 %, car deux sorties débruitées déclenchent des boucles de répétition
(« je ne sais pas… » ×40), qui dominent sinon la moyenne brute.

| SNR | Langue | Propre | Bruité | Débruité DFN3 |
|---|---|---|---|---|
| 0 dB | en | 5,1 % | 21,2 % | 25,9 % |
| 0 dB | fr | 4,5 % | 48,1 % | 69,7 % |
| 5 dB | en | 5,1 % | 8,8 % | 12,5 % |
| 5 dB | fr | 4,5 % | 25,2 % | 35,4 % |

**DFN3 dégrade le WER de Whisper sur ce bruit**, alors qu'il gagne 8 à
10 dB de SI-SDR. Les médianes vont dans le même sens : 24 % → 44 % à 0 dB,
9 % → 18 % à 5 dB. C'est cohérent avec ce qu'on sait des ASR robustes :
Whisper est entraîné sur de l'audio bruité, et les artefacts du masquage
(consonnes atténuées, bandes ERB coupées) le gênent plus que le bruit
stationnaire. Le français souffre davantage. DFN3 sert donc à l'écoute
humaine et à la visio, **pas comme prétraitement de `transcriptions`**.

## MossFormer2_SS_16K

### Spécification

Code de référence : `clearvoice/models/mossformer2_ss/` de ClearerVoice-Studio,
checkpoint `last_best_checkpoint.pt` (55 736 034 paramètres, 1 099 tenseurs,
chargés en `strict=True`).

- **Encodeur** : conv 1D 1 → 512, noyau 16, stride 8, sans biais, ReLU :
  2 000 trames par seconde à 16 kHz.
- **MaskNet** : LayerNorm globale (« ln » sur canaux et temps), conv 1×1
  512 → 512, plus un plongement sinusoïdal mis à l'échelle.
- **24 couches**, chacune composée de :
  1. `FLASH_ShareA_FFConvM` : décalage d'un pas sur la moitié des canaux,
     `to_hidden` (ScaleNorm, linéaire 512 → 2 048, SiLU, `ConvModule` :
     conv en profondeur de noyau 17 avec résiduel) découpé en `v` et `u` de
     1 024 ; `to_qk` 512 → 128, puis `OffsetScale` à quatre têtes (q et k
     quadratiques, q et k linéaires) ; RoPE sur 32 dimensions ; attention
     quadratique par groupes de 256 trames `relu(q·k / 256)²`, plus attention
     linéaire globale `(k·v)/n` ; `(att_u · v) · sigmoid(att_v · u)`, puis
     `to_out` 1 024 → 512 et résiduel ;
  2. `Gated_FSMN_Block_Dilated` : conv 1×1 512 → 256 + PReLU, CLayerNorm,
     FSMN à portes (`to_u` et `to_v` de 256, mémoire FSMN dilatée en conv 2D
     de noyau 39 à connexions denses), CLayerNorm, conv 1×1 256 → 512 et
     résiduel.
- LayerNorm finale, PReLU, conv 1×1 512 → 1 024 (deux locuteurs), porte
  `tanh · sigmoid`, conv 1×1 512 → 512 et ReLU : deux masques.
- **Décodeur** : conv transposée 1D 512 → 1, noyau 16, stride 8.
- Décodage officiel (`decode.py`) : une passe si l'audio tient dans
  `one_time_decode_length`, sinon fenêtres de 2 s avec un pas de 1,5 s sans
  réalignement des locuteurs entre fenêtres. Les sorties sont renormalisées
  au RMS de l'entrée.

Le modèle est non causal : attention linéaire et normalisation globales sur
toute la séquence. Le coût est d'environ 3,3 M de MAC par trame et par couche,
soit environ 160 GMAC par seconde d'audio (0,3 TFLOP).

### Référence et mesures

`tools/voice/moss_ref.py` charge le code officiel tel quel, avec deux
substituts : `rotary_embedding_torch` est le fichier du dépôt
lucidrains/rotary-embedding-torch, et `torchinfo.summary` est un stub sans
effet. Le décodage se fait en une passe (12 paires de 5 à 12 s), suivi de la
renormalisation RMS.

SI-SDR de séparation (meilleure permutation) sur des paires FLEURS
superposées à 0 dB. Exécution sur CPU avec PyTorch, en une passe :

| Paires | SI-SDRi moyen | Remarque |
|---|---|---|
| 6 × en + fr | 20,99 dB (20,24 à 22,01) | |
| 3 × en + en | 6,49 dB (−1,02, 8,80, 11,70) | la paire 8 échoue (−1 dB) : locuteurs probablement proches |
| 3 × fr + fr | 16,97 dB (13,35 à 19,12) | |

Temps de calcul de la référence PyTorch : sur CPU, RTF de 1,24 à 1,48 ; sur
GPU (MPS, f32, une passe, paire 3 de 11,8 s), **0,94 s, soit 80 ms de GPU par
seconde d'audio (RTF 0,080)**, avec le même SI-SDRi (20,76 dB) que sur CPU.

WER de Whisper large-v3-turbo par voix (plafonné à 100 % par clip). Les
références sont les phrases FLEURS entières, alors que le mélange est tronqué
au clip le plus court : le WER « propre » sert donc de plancher (14 %).

| Paires | Voix propre tronquée | Mélange | Voix séparée |
|---|---|---|---|
| en + fr | 16,9 % | 35,1 % | 18,3 % |
| en + en | 15,4 % | 63,9 % | 39,1 % |
| fr + fr | 8,4 % | 61,8 % | 17,4 % |
| toutes | 14,4 % | 49,0 % | 23,2 % |

Hors l'échec en + en (paire 8), la séparation ramène le WER près du plancher.

### Port saragossa

Pas encore fait. Le port demande sur GPU : la LayerNorm globale, la ScaleNorm,
les convolutions en profondeur 1D (noyau 17) et 2D dilatées denses (noyau 39),
l'attention quadratique en groupes de 256 avec `relu²`, l'attention linéaire
globale, la RoPE partielle sur 32 dimensions, PReLU et la conv transposée 1D.
Les GEMM (512 ↔ 2 048, 1 024 → 512) représentent plus de 80 % des MAC et
peuvent réutiliser les matmuls f32/bf16 de `metal_backend`. Estimation : 5 à
8 jours, avec un oracle couche par couche contre `moss_ref.py`.


## RE-USE (NVIDIA-NemotronLabs-RE-USE)

### Spécification

Fichiers locaux lus sans être exécutés : `config.json`, `models/*.py`,
`inference.py` ; `model.safetensors` contient 9 607 747 paramètres f32.

- **STFT à fenêtre fixe (SFI)** : `n_fft = 320 · sr / 8 000`, pas de 40 · sr/8 000
  et fenêtre de Hann centrée, soit 40 ms de fenêtre et 5 ms de pas à toute
  fréquence (1 920 et 240 à 48 kHz, F = 961 bins). Magnitude compressée par
  `log1p`, et phase en entrée ;
- **DenseEncoder** : conv 1×1 2 → 64, InstanceNorm2d et PReLU, DenseBlock de
  4 convs 3×3 dilatées en temps (1, 2, 4, 8), à connexions denses de 64 à 256
  canaux, puis conv 1×3 de stride (4, 2) : 50 trames par seconde et environ F/2 bins ;
- **30 TFMambaBlock** : Mamba bidirectionnel sur l'axe temps pour chaque bin,
  puis sur l'axe fréquence pour chaque trame. Chaque Mamba a `d_model = 64`,
  `expand = 4` (`d_inner = 256`), `d_state = 16`, `d_conv = 4` et
  `dt_rank = 4` ; les deux sens sont concaténés, projetés (128 → 64) et
  normalisés (LayerNorm), avec un résiduel ;
- **décodeurs** magnitude et phase : DenseBlock, deux `SPConvTranspose2d`
  (sous-pixel ×2 en fréquence et ×4 en temps), conv 1×1 ; la phase sort par
  `atan2`. Reconstruction par iSTFT.

### Coût estimé

Par jeton (une case temps-fréquence après l'encodeur), un Mamba coûte environ
71 k MAC (`in_proj` 32 k, `x_proj` 9 k, scan 12 k, `out_proj` 16 k). Un
TFMambaBlock en exécute quatre, soit environ 0,3 M MAC, et les 30 blocs
environ 9 M MAC par jeton.

| sr | Jetons/s (50 trames × F/2) | Mamba | DenseBlocks | Total |
|---|---|---|---|---|
| 16 kHz | 50 × 161 ≈ 8 000 | ≈ 72 GMAC/s | ≈ 30 GMAC/s | ≈ 0,2 TFLOP/s d'audio |
| 48 kHz | 50 × 481 ≈ 24 000 | ≈ 216 GMAC/s | ≈ 90 GMAC/s | ≈ 0,6 TFLOP/s d'audio |

Le scan sélectif est séquentiel le long d'un axe, mais il est parallèle sur
256 × 16 états et sur l'autre axe (481 bins ou trames) ; il tient donc sur le
GPU avec le même schéma par morceaux que le gated delta net de
`linear_attention.rs`. À 2-5 TFLOP f32 effectifs, le RTF attendu est de 0,1 à
0,3 à 48 kHz et de 0,04 à 0,1 à 16 kHz (estimation à mesurer). Le modèle est
non causal : hors ligne seulement, par morceaux pour les fichiers longs
(`inference_chunk.py`).

### État

Pas de référence exécutée : le code dépend de `mamba_ssm` (noyaux CUDA), absent
sur macOS. Un oracle demanderait d'abord un scan Mamba de référence en PyTorch
pur. Estimation du port : 6 à 10 jours (scan bidirectionnel GPU, convs 2D
denses, InstanceNorm, sous-pixel, iSTFT).

