# Embeddings vision fournis au LLM

Le transport média reste fermé tant que le ViT Rust n’est pas disponible.
`decoder::vision::prefill_supplied_vision` est une entrée interne de
qualification : elle prend les ids et les embeddings Python après merger.
Elle ne lit aucun fichier média et ne charge aucun encodeur.

Chaque image fournit son premier placeholder, sa grille T=1/H/W, ses lignes
f32, le SHA256 du contenu et les paramètres canoniques du producteur
(prétraitement, identité des poids vision et timestamps éventuels). Les
placeholders sont comptés strictement, les spans sont ordonnés et les
embeddings non finis sont refusés. Le hash inclut aussi les octets des
embeddings effectivement injectés.

Le prefill transporte une position `(t,h,w)` par ligne. Seules les couches
full-attention appliquent M-RoPE : axes entrelacés [11,11,10], θ=10⁷,
64 dimensions rotées dans une tête de 256, partenaires rotate-half.
Les positions du texte sont identiques sur les trois axes. Le chemin Metal
charge le tableau une fois et lie la tranche correspondant à chaque morceau.

Le cache conserve un compteur de tokens KV et un delta RoPE signé séparés.
`current_pos()` vérifie `tokens + delta` ; le decode utilise cette position
sur les trois axes. Le clone, le rollback et le reset conservent ou effacent
explicitement ce contexte. Les paramètres GPU de SDPA et les adresses KV
restent fondés sur les tokens, même avec un delta négatif.

Le manifeste SSD v5 conserve le delta et les positions. Sa clé authentifie
les positions et les identités média du préfixe couvert. Une frontière
interne à une image est refusée. Le format v4 est incompatible. La primitive
`BlockHash::chain` reçoit les identités média au bloc de leur premier
placeholder ; sans médias, les octets hachés du chemin texte sont inchangés.
Le serveur continue de refuser les médias avant tout lookup multimodal.

## Qualification

- Les 24 prompts exportés contrôlent les positions et l’insertion exacte.
- Le modèle synthétique contrôle prefill entier/fractionné, 16 pas de decode,
  conservation du delta et A/B AR/MTP/DFlash, avec acceptation et rollback.
- Le contrôle Metal synthétique compare le prefill et le decode au CPU.
- `golden::vision::vision_llm_logits_against_frozen_transformers` exige la
  calibration locale et compare tout le vocabulaire, pour six prompts et
  16 tokens imposés, en prefill per-op et résident entier/fractionné.

`tools/goldens/vision_llm.py` capture le Qwen dense BF16 local, couche par
couche, avec les classes Transformers installées et leur cache. Il refuse
un modèle quantifié : sa référence doit être déquantifiée séparément. Il
exécute les mêmes préfixes en f32 puis BF16, enregistre RMS, maximum absolu
et maximum d’erreur de probabilité par pas, puis fige chaque seuil à
**exactement deux fois** le bruit. Les payloads, sources exécutées et poids
sont authentifiés par SHA256. Un manifeste existant interdit le recalibrage.
Les poids restent en lecture seule et le réseau est désactivé.

La capture Qwen dense contient six prompts et 16 pas imposés, avec 851
tenseurs LLM et 146 sources Transformers empreintés. Les bornes RMS figées
vont de 0,04493645 à 0,21593164 selon le prompt et le pas ; chaque état
conserve ses trois seuils propres dans le manifeste.

Exemple, à exécuter sous le verrou GPU du projet avec un ticket de 30 minutes :

```sh
"$REF_PYTHON" -I -B tools/goldens/vision_llm.py \
  --model "$SARAGOSSA_QWEN38_BF16_DIR" \
  --fixtures tests/fixtures/vision/Qwen3.8-27B-BF16
cargo test --release --all-features vision_llm_logits_against_frozen_transformers \
  -- --ignored --nocapture
```

L’A/B synthétique ne qualifie pas les sidecars réels. Les chemins médias
MTP/DFlash restent fermés jusqu’à leur A/B sur les checkpoints concernés ;
le prototype MTP à contrôle GPU refuse explicitement les caches vision.
Le test `vision_decode_ab_real_sidecars` utilise `SARAGOSSA_QWEN36_MOE_DIR`,
`SARAGOSSA_RUST_DFLASH_DRAFT_DIR` et, si la détection standard ne trouve pas
la tête, `SARAGOSSA_RUST_MTP_SIDECAR`.
