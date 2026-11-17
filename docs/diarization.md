# Diarisation : nvidia/Nemotron-3-Diarization

`saragossa diarize` et `POST /v1/audio/diarization` répondent à « qui parle
quand » sur un enregistrement, jusqu'à 8 locuteurs, hors ligne.

## Licence et notices

Les poids `nvidia/Nemotron-3-Diarization` sont publiés par NVIDIA sous
**OpenMDW-1.1** (<https://openmdw.ai/license/1-1/>) : usage commercial ou non,
modification et redistribution sans restriction, aucune contrainte sur les
sorties. La licence et ses notices doivent accompagner toute redistribution des
poids ; saragossa n'en embarque aucun et ne les télécharge pas. La clause de
résiliation de la licence s'applique en cas d'action en contrefaçon contre le
modèle. Le code de référence suivi est celui de 🤗 Transformers 5.22
(Apache-2.0) ; aucun fichier n'en est copié dans ce dépôt.

## Usage

```bash
saragossa diarize --model-dir /chemin/Nemotron-3-Diarization audio.wav            # RTTM
saragossa diarize --model-dir /chemin/Nemotron-3-Diarization --format json a.wav  # JSON
saragossa serve --diar-model /chemin/Nemotron-3-Diarization ...
curl --unix-socket /tmp/saragossa-serve.sock -F file=@a.wav [-F response_format=rttm] \
     http://localhost/v1/audio/diarization
```

Fichiers lus : `config.json` et `model.safetensors`. Le WAV est replié en mono
et rééchantillonné à 16 kHz (même lecteur que le STT, 500 s au plus). La
réponse JSON vaut `{"duration": s, "segments": [{"speaker", "start", "end"}]}`,
en secondes ; les locuteurs sont numérotés dans leur ordre d'arrivée. API Rust :
`saragossa::diarization::DiarizationModel::{from_model_dir, diarize,
speaker_probabilities}`.

## Spécification

Source : `config.json`, `processor_config.json`, le `model_config.yaml` du
`.nemo`, et le code 🤗 Transformers 5.22
(`modeling_nemotron3_diarization.py`, `processing_nemotron3_diarization.py`,
`feature_extraction_nemotron_asr_streaming.py`).

| Étage | Détail |
|---|---|
| Entrée | mono 16 kHz, f32 |
| Front-end | préaccentuation 0,97 ; STFT 512 points, pas 160 (10 ms), fenêtre Hann symétrique de 400 centrée dans 512, `center=True` à zéros ; puissance ; banc mel Slaney 128 bandes 0-8 kHz (librosa, norm Slaney) ; `ln(x + 2^-24)` ; **aucune normalisation** ; `floor(L/160)` trames valides, la trame de centrage finale est masquée à 0. Pas de dither à l'inférence. |
| Sous-échantillonnage | **pas de conv2D** : `feature_stacking` ×8, 8 trames de 128 bandes concaténées (zéros pour le dernier groupe) puis `Linear 1024 → 512` sans biais. Trame encodeur = 80 ms. |
| Découpage hors ligne | morceaux de 340 trames (27,2 s) + 40 trames d'anticipation, sans retour arrière. Chaque morceau est précédé du cache des locuteurs (≤ 264 trames) et de la FIFO (≤ 40). Positions RoPE remises à 0 à chaque morceau. |
| Encodeur | LayerNorm d'entrée ; 31 couches pré-norme : `x + o(attn(LN1 x))`, `x + fc2(GELU(fc1(LN2 x)))` ; d = 512, 8 têtes de 64, MHA bidirectionnelle (pas de GQA), q/k/v sans biais, o avec biais ; RoPE rotate-half sur toute la tête, θ = 10⁴ ; FFN 2048, GELU exacte (erf) ; LayerNorm finale ; eps 1e-5. 99,2 M paramètres. |
| Tête | `Linear 512 → 192` ; conv1d sous-pixel `192 → 1536`, k = 3, padding 1, relue `[T, 8, 192]` → `[8T, 192]` (retour à 10 ms) ; `ReLU → Linear 192 → 192 → ReLU → Linear 192 → 8` ; sigmoïde = activité par locuteur. |
| Cache (AOSC) | après chaque morceau, les trames du morceau entrent dans la FIFO ; si elle dépasse 40, au moins 300 trames passent au cache ; au-delà de 264 trames, compression par score `ln p − ln(1−p) + Σ ln(1−p) − ln ½` (bornes 0,25), bonus 0,05 aux trames récentes, bonus fort (+2 ln 2) aux 24 et faible (+ln 2) aux 48 meilleures par locuteur, 1 créneau de silence appris par locuteur, 33 créneaux par locuteur. |
| Post-traitement | `p > 0,5` par locuteur et par trame de 10 ms ; segments = suites de trames actives ; **aucune fusion, durée minimale ni marge** ; chevauchements conservés ; 8 locuteurs au plus. |

Le streaming (morceaux de 3 à 9 trames, FIFO 264) réutilise le même cache avec
d'autres tailles ; il n'est pas exposé ici.

## Exécution dans saragossa

- Front-end, empilement, cache et seuillage : CPU (rayon pour le mel).
- 31 couches : encodeur résident de Whisper, un command buffer par morceau,
  GEMM f32, avec un kernel `rope_heads_inplace_f32` appliqué à q et k. Le
  chemin GEMM bf16 de Whisper n'est pas activé pour la diarisation.
- Projection, conv1d (im2col + GEMM) et tête : `Linear` (GEMM Metal per-op).
- La dernière trame encodeur, quand elle ne contient que la trame de centrage
  masquée, reste une requête (la conv1d de la tête lit sa voisine) mais sort
  des clés : Steel d64 avec `kL` réduit, ou K/V tronqués sur CPU.
- Sans Metal (`--cpu`), l'oracle CPU fait le même calcul.

## Mesures (M5 Max, 16 nov. 2026)

Référence : réimplémentation torch f32 CPU du code 🤗 5.22 (transformers 5.8.1
installé ne connaît pas ce modèle), sur 5 clips FLEURS EN concaténés (46,6 s),
5 clips FLEURS FR (45,4 s) et le dialogue à 8 voix de la fiche (97,6 s).

| | Écart max | Écart moyen | Segments |
|---|---|---|---|
| Tolérance figée (2 × bruit bf16 torch) | 0,292 | 1,6e-3 | identiques |
| FLEURS EN | 2,7e-6 | 8e-9 | identiques |
| FLEURS FR | 2,3e-6 | 3e-9 | identiques |
| Dialogue 8 voix | 1,5e-2 | 4,5e-5 | 1 trame de 10 ms basculée (p_ref = 0,502) |

Sur le dialogue, l'écart reste à 4e-6 sur les deux premiers morceaux. Il naît à
la 2ᵉ compression du cache, qui départage deux scores distants de 1e-7 (trames
574 et 587) ; la référence torch elle-même change de sélection et bascule
2 trames sous un bruit de 2e-6 sur les probabilités. C'est une quasi-égalité
du score AOSC, pas un écart du calcul ; le test l'accepte pour les seules
trames à moins de 0,01 du seuil (qualification ajoutée après la mesure).

Débit, 1 h d'audio (dialogue répété), charge < 2 : **5,7 s** de bout en bout
avec Metal (mel CPU 0,05 s). Référence torch f32 (forward seul, mel exclu) :
28,7 s sur CPU, **3,5 s sur MPS**. saragossa est donc 1,6× plus lent que torch
MPS : la tête (projection, conv1d, classifieur) passe par des GEMM per-op avec
aller-retour CPU, et LayerNorm d'entrée et cache restent sur CPU, à chaque
morceau de 27,2 s. TODO: fusionner la tête dans le command buffer de l'encodeur.

Français : la langue n'est pas publiée. Sur les 5 clips FR, chaque clip reçoit
un seul locuteur (pureté 1,0) et le modèle en trouve 3 ; les trois clips
regroupés ont une F0 médiane de 173 à 183 Hz, les deux autres 273 et 108 Hz.
FLEURS ne donne pas l'identité des locuteurs : pas de DER exact, regroupement
cohérent. En anglais, 3 locuteurs pour 5 clips, avec un regroupement de clips
à 127, 155 et 175 Hz, moins évident.

## Proposition : transcription attribuée (non implémentée)

Les segments `verbose_json` de Whisper (PR #231 : `start`, `end`, `text` par
segment, granularité `word` absente) se fusionnent sur CPU, sans relancer de
modèle :

1. Diariser et transcrire le même tableau d'échantillons 16 kHz ; la
   diarisation donne `p[t, k]` toutes les 10 ms.
2. Pour chaque segment Whisper `[s, e)`, sommer `p[t, k]` sur ses trames et
   attribuer le locuteur d'activité cumulée maximale (`speaker`), avec
   `speaker_confidence` = part de cette masse. Sommer les probabilités plutôt
   que les segments seuillés garde une décision quand aucune trame ne dépasse 0,5.
3. Si deux locuteurs dépassent chacun 30 % de la masse, couper le segment au
   changement d'argmax lissé (médiane sur 0,3 s) et répartir le texte au
   prorata des durées, au mot près. Le découpage exact attendra des
   horodatages par mot.
4. Champ additif dans la réponse : `segments[].speaker` (entier dans l'ordre
   d'arrivée) et `speakers[]` en tête ; `response_format=verbose_json` +
   `diarize=true` sur `/v1/audio/transcriptions`, actif seulement avec
   `--diar-model`. Le mode flux reste hors périmètre tant que l'AOSC en
   streaming n'est pas exposé.
