# Corpus figé de l'identité greedy (oracle O3)

Prompts du bench concurrents (campagne du 4 nov. 2026), contenu « user » rendu
par le gabarit ChatML Qwen sans réflexion (`render_qwen_chatml`, comme
`serve`). Chaque fichier commence par sa ligne `Request id: <nonce>`.
`MANIFEST.tsv` donne le nonce, l'empreinte sha256 du fichier et la source.

- `prose32k-1.txt` est le **cas fixé** : nonce 9784136873970414, 32 768
  tokens sur Qwen3.8-27B 4 bits. Sur `main` 5896416, MTP et DFlash y
  choisissent « It » (1049) au token généré 182, là où l'AR choisit « The »
  (561), avec une marge AR de 0,0089.
- Les autres prompts ont le nonce 0000000000000000.

Provenance et licences :
- `chat-*` : textes du harnais saragossa.
- `code4k-*` : sources de la bibliothèque standard Rust 1.98.1 (MIT ou
  Apache-2.0).
- `prose32k-1` : articles de WikiText-2 (Wikipedia, CC BY-SA 3.0) : SMS
  Markgraf, Coldrum Long Barrow, Wrapped in Red, Sholay, Loose (Nelly Furtado
  album), Antimony.
