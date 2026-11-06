# Oracles Flash-Next (`qwen4_exp`)

Outils Python qui produisent les sorties de référence de la référence CPU
`src/qwen4_exp/`. Ils sont annexes au moteur : le crate ne les exécute jamais
et n'en dépend pas à l'exécution.

| Fichier | Rôle |
|---|---|
| `make_mini_fixture.py` | Fixture N1 `tests/fixtures/qwen4exp_mini` : petit modèle aléatoire exactement représentable en q4e8, sorties officielles transformers |
| `oracle_dump.py` | Goldens N2 sur les vrais poids q4e8 : couches officielles, routage, scores QSA, `provenance.json` |
| `oracle_common.py` | Lecture des poids q4e8, construction des couches officielles, capture des top-k |

## Dépendances de développement (hors crate)

Ces versions sont épinglées : elles produisent les goldens et les sha figés.

| Paquet | Version |
|---|---|
| torch (CPU) | 2.14.1 |
| transformers (module `qwen4_exp`) | 5.20.2 |
| safetensors | 0.8.0 |
| numpy | 2.5.4 |
| tokenizers | entraîné par transformers |

Installation dans un environnement virtuel jetable, hors du dépôt. Lancer
toujours Python avec `-I` :

```bash
python3 -m venv /tmp/venv-flashnext
/tmp/venv-flashnext/bin/pip install torch==2.14.1 transformers==5.20.2 safetensors==0.8.0 numpy==2.5.4
```

## Régénérer la fixture N1

```bash
/tmp/venv-flashnext/bin/python -I tools/flashnext/make_mini_fixture.py tests/fixtures/qwen4exp_mini
```

Graine par défaut : 20261110, modifiable par `FIXTURE_SEED`. Le script refuse
une graine dont un routeur a une marge inférieure à 1e-4 au seuil du top-k. Il
cherche aussi une entrée QSA à marge nette, supérieure à 1e-3, pour le golden
`qsa_clear.*`.

Après régénération, mettre à jour les sha256 figés dans
`src/qwen4_exp/tests.rs`.

## Produire les goldens N2

```bash
Q=ai_model/Qwen3.8-Flash-Next-MLX-q4e8
/tmp/venv-flashnext/bin/python -I tools/flashnext/oracle_dump.py --layers $Q $Q/tokenizer.json docs/perf/mtp-pairwise.md <sortie>
/tmp/venv-flashnext/bin/python -I tools/flashnext/oracle_dump.py --full   $Q $Q/tokenizer.json <sortie>
SARAGOSSA_QWEN4EXP_DIR=$Q SARAGOSSA_QWEN4EXP_GOLDEN=<sortie> \
  cargo test --release --lib qwen4_exp::tests::n2 -- --ignored --nocapture
```

Coût : environ 2 min pour `--layers` (8000 tokens, 4 couches) et 18 min pour
`--full`. Le pic est d'environ 17 Go, dont les experts f32 d'une couche, soit
10 Go.

Chaque golden a sa propre entrée dans `provenance.json`. Les tests la vérifient
avant l'inférence :
- l'empreinte du golden ;
- les empreintes de `config.json` et de chaque fichier de poids ouvert,
  recalculées sur disque, dont `ngram/meta.json` et la table qu'il désigne ;
- la version de transformers et le backend `sdpa`.

Un même dossier refuse des goldens d'origines différentes. La compatibilité est
vérifiée avant le calcul, puis de nouveau avant une publication atomique : en
cas de refus, les fichiers existants restent intacts. Vérification :
`python -I tools/flashnext/oracle_dump.py --selftest`.
