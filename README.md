# fireviewer-geolocation

## Repères documentaires — 19 septembre 2026

- **Rôle :** Production d’hypothèses géographiques par retrieval, perspective, registration, Panoramax et autres références.
- **Statut :** Actif — package v0.1.1.
- **Entrées :** Observations visuelles, position/caméra si disponible, terrain, orthophoto, références géographiques.
- **Sorties :** Candidats géographiques, transformations, confiance/incertitude et preuves associées.
- **Limites :** Toujours fournir un niveau de confiance ou s’abstenir. Ne pas inventer une précision absente des données. Position caméra et position du phénomène sont distinctes.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-geolocation.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

> **Source active FV · public.** Hypothèses géographiques, Panoramax, retrieval et registration. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Geographic hypotheses, retrieval, perspective and registration.

Python package: `fireviewer_geolocation`. Version: `0.1.1`.

## Installation

Install the versioned release wheels (including versioned FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-geolocation==0.1.1
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Canonical repository and rights

Canonical source: [`fireviewer/fireviewer-geolocation`](https://github.com/fireviewer/fireviewer-geolocation). Technical stewardship: FIRE-VIEWER. Repository access: public.

Historical authorship, AGPL-3.0-or-later notices and third-party rights are retained. Technical stewardship and repository placement are not a signed assignment of intellectual-property rights. Any pre-association assets remain subject to their documented licences or agreements.

This repository is the maintained implementation location for the responsibility stated above. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. Older `firewarning_worker` or backend imports remain compatibility adapters where required; they are not alternative locations for new component logic.

## Delivery and qualification

Versioned packages are distributed through the versioned release bundles. Current container locks, reconstruction inputs and dated acceptance records are maintained in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker).

Package installation, CPU/schema tests, service deployment and real-data acceptance are separate checks. CPU/schema tests do not qualify GPU, visual or scientific performance. This documentation update does not publish a package, rebuild an image or change production configuration.

Extraction correspondence and hashes remain in the historical migration dossier. They record the restructuring, not the current deployment state.

## Sources et commandes propres au composant

Commande CPU : `fireviewer-geolocation`. Hypothèses, Panoramax, retrieval et registration sont dans `mvp/localization`. Le téléchargement des modèles et les reprises de corpus restent explicites.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels versionnés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les commandes de reprise et leurs prérequis sont décrits dans [ORGANISATION.md](ORGANISATION.md). Les reçus du dossier de migration restent des preuves historiques, pas une nouvelle qualification.

## Ouverture du code source — 19 septembre 2026

Ce dépôt fait partie du premier lot de huit composants FIRE-VIEWER ouvert au public sur décision du mainteneur. Le code original reste sous **AGPL-3.0-or-later** et la documentation originale sous **CC BY 4.0**, avec les notices et droits tiers existants.

Cette ouverture porte sur le code, son historique et les artefacts de développement déjà associés au dépôt. Les services déployés, comptes, données, corpus, modèles, secrets et autorisations des ressources externes gardent leur propre périmètre. Les sources des sites, du backend, des applications Android et de l’infrastructure restent privées. La visibilité publique ne constitue ni une nouvelle recette fonctionnelle ni un acte de cession des droits.

[Inventaire et périmètre d’ouverture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/OPEN_SOURCE.md).

## Compatibilité des contrats Bonsaï / Jev

La chaîne courante utilise `fireviewer-contracts==0.1.2`. La CI et les images
construisent les dépendances liées depuis les commits immuables de `ci.json`,
puis vérifient les hashes des wheels produites. Les règles métier de ce composant
ne changent pas. Les expériences textuelles Jev sont documentées séparément :
https://github.com/fireviewer/fireviewer-jev-experiments

## Migration par révisions d’incident

La version candidate 0.1.2 aligne les dépendances sur les contrats 0.1.3. Les algorithmes et modèles de ce composant ne changent pas. Les wheels requis sont versionnés dans `vendor/`, vérifiés par SHA-256 dans `ci.json` et utilisés par `python tools/ci.py verify` sans dépôt voisin. Aucun modèle ou corpus n’est incorporé.

## Exécution sans Azure Maps — candidat du 26 septembre 2026

Le service CPU géographique désactive désormais Azure Maps par défaut.
`FIREVIEWER_AZURE_MAPS_ENABLED` absent ou `false` ne requiert ni
`FIREVIEWER_AZURE_MAPS_ACCOUNT_CLIENT_ID` ni `AZURE_CLIENT_ID` et ne crée
aucun client Azure Maps. L'ancien adaptateur reste disponible seulement par
activation explicite avec les deux identifiants requis. Cette modification
ne qualifie pas encore un déploiement de worker ni une autre source de
géocodage : sans donnée spatiale exploitable, le service conserve l'abstention.
