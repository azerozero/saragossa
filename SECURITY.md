# Politique de sécurité

## Versions prises en charge

Les correctifs de sécurité ciblent la dernière version mineure publiée de
saragossa. Avant de signaler un problème, essayez la
[dernière release](https://github.com/azerozero/saragossa/releases).
Les versions mineures plus anciennes ne reçoivent pas de correctifs.

## Signaler une vulnérabilité

**N'ouvrez pas d'issue publique pour un problème de sécurité.**

Utilisez le [signalement privé GitHub](https://github.com/azerozero/saragossa/security/advisories/new)
ou écrivez à <security@a00.fr>.

Joignez les éléments suivants :

- La description du problème et son impact.
- La version de saragossa, le mode d'installation (binaire, Homebrew ou
  sources), la version de macOS et le matériel Apple Silicon utilisé.
- Une reproduction minimale : commande, configuration, requête vers
  `saragossa serve`, résultat attendu et résultat observé.
- Le niveau de gravité estimé et, si nécessaire, une demande de CVE.

Retirez les jetons, données personnelles, prompts privés et chemins locaux
sensibles des exemples et des journaux.

### Délais visés

| Étape | Objectif |
|-------|----------|
| Accusé de réception | Sous 3 jours ouvrés |
| Qualification et évaluation de la gravité | Sous 7 jours ouvrés |
| Correctif ou plan de mitigation pour une gravité haute ou critique | Sous 30 jours |
| Divulgation coordonnée | Accord avec le déclarant, 90 jours par défaut |

Ces délais sont des objectifs. Le déclarant est crédité dans les notes de
release et l'avis de sécurité, sauf demande d'anonymat.

## Périmètre

Sont concernés :

- Le binaire `saragossa`, sa bibliothèque Rust et leur configuration par défaut.
- Le chargement des modèles et des tenseurs, les buffers et les kernels Metal.
- L'API locale exposée par `saragossa serve` : chat, transcription, synthèse
  vocale et embeddings.
- Les fichiers et métadonnées distribués avec les assets des releases GitHub.
- La formule Homebrew `azerozero/tap/saragossa`.

Les vulnérabilités des fournisseurs de modèles et des dépendances tierces
relèvent de leurs mainteneurs. Signalez aussi leur impact sur saragossa pour
permettre une mise à jour ou une mitigation. La compromission préalable du
compte local et les modifications de binaires par ce compte ne constituent
pas, à elles seules, une vulnérabilité du moteur.

## Données sensibles

Préférez une reproduction avec des données synthétiques. Ne joignez ni poids
privés, ni enregistrements audio personnels, ni jetons d'accès aux modèles.
Pour les éléments sensibles indispensables, utilisez le signalement privé
GitHub et convenez du partage avec le mainteneur.

## Recherche de bonne foi

La recherche de sécurité respectant cette politique est bienvenue. Nous ne
poursuivrons pas les chercheurs qui :

- Laissent un délai raisonnable pour corriger avant la divulgation publique.
- Évitent les atteintes à la vie privée, la destruction de données et la
  dégradation du service pendant les essais.
- Ne récupèrent pas de données au-delà du strict nécessaire pour démontrer
  la vulnérabilité.
