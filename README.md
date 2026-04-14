# Entity-Resolution-Pipeline

Pipeline d'Entity Resolution multi-datasets avec deux axes maintenus:

- une baseline simple `v0`
- une stratégie de recherche `cw_semantic_predictive`, orientée forte réduction de l'espace de comparaison

Le dépôt cible actuellement `abt_buy`, `amazon_google`, `dblp_acm`, `spimbench` et `rexa_dblp`. La direction de recherche vise une méthode benchmark-générique, CPU-first, avec réduction extrême de l'espace candidat avant le matching détaillé.

## Objectif

Le problème principal du pipeline ER classique est qu'il reste `pair-first`:

1. on génère des candidats
2. on matérialise encore beaucoup de paires explicites
3. on score ces paires une par une

Même avec du blocking, du purging ou du meta-blocking, l'objet de calcul reste la paire. La direction de recherche implémentée ici essaie de changer cela.

L'idée centrale est:

- représenter chaque entité par des `witnesses` hétérogènes
- construire des régions cross-source implicites au lieu de comparer directement toutes les paires
- raffiner ces régions par intersections successives
- ne matérialiser des paires explicites qu'au dernier moment, quand le sous-espace est déjà petit

Cette approche est codée dans `pipeline/constraint_witness.py` et exposée dans Member 2 via `candidate_strategy='cw_semantic_predictive'`.

Le matching final n'est plus un simple seuil sur le score direct. Il combine:

- un score de similarité de base (`jaccard`, `tfidf`, `sbert`)
- un reranker contextuel monotone
- une couche `adaptive_rescue` post-reranking, activée seulement quand la surface de matching du benchmark est réellement ambiguë

## Vue d'ensemble

Les modules principaux sont au niveau racine:

- `pipeline/data_ingestion.py`: normalisation et ingestion
- `pipeline/blocking.py`: token blocking initial
- `pipeline/block_processing.py`: réduction et génération de candidats
- `pipeline/matching.py`: scoring et décision de match
- `pipeline/clustering.py`: clustering et fusion d'entités
- `pipeline/constraint_witness.py`: moteur witness-first et rescues génériques

Entrypoints principaux:

- `cli/run_pipeline.py`: implémentation principale du pipeline complet
- `cli/run_member1.py` à `cli/run_member4.py`: implémentations principales par étape
- `cli/run_interactive.py`: implémentation principale du lanceur interactif
- `cli/` est désormais l’unique emplacement des entrypoints exécutables
- `pipeline/` est désormais l’unique emplacement des modules cœur du workflow

Tests:

- `tests/test_data_ingestion.py`
- `tests/test_block_processing.py`
- `tests/test_matching.py`
- `tests/test_constraint_witness.py`

Documentation et rapport:

- `docs/report/`: rapport LaTeX de recherche et artefacts de compilation associés

Espaces de données:

- `data/raw/`: données brutes
- `data/cleaned/<dataset>/`: sorties normalisées
- `output/<dataset>/`: artefacts intermédiaires et finaux
- `output/mock/<dataset>/`: fixtures mock

Pour `rexa_dblp`, l'ingestion attend un dossier brut de type OAEI/RDF dans `data/raw/Rexa-DBLP/`:

- une source RDF pour `dblp`
- une source RDF pour `rexa`
- optionnellement un alignment OAEI (`refalign.rdf`, `alignment.rdf`, `reference.xml`, etc.) ou un CSV à deux colonnes

Les formats RDF/XML, Turtle, N-Triples et N3 sont supportés. Si `rdflib` n'est pas installé, le parseur minimal interne couvre tout de même `RDF/XML`, ce qui suffit pour beaucoup de variantes OAEI.

Pour `dblp_acm`, les fichiers attendus sont maintenant présents dans `data/raw/DBLP-ACM/`:

- `tableA.csv`
- `tableB.csv`
- `train.csv`
- `valid.csv`
- `test.csv`

L'ingestion utilise l'union des paires positives (`label = 1`) de `train/valid/test` comme ground truth.

## Pipeline

La pipeline standard suit toujours cette structure:

1. `Member 1`
   Ingestion, normalisation, token blocking
2. `Member 2`
   Purging, meta-blocking, ou stratégies avancées de génération de candidats
3. `Member 3`
   Matching et agrégation de similarités
4. `Member 4`
   Clustering et fusion

Cette pipeline reste utile comme baseline et comme infrastructure de comparaison.

## Constraint Witness

### Intuition

Le moteur `cw` ne pense plus d'abord en paires, mais en certificats de compatibilité.

Chaque entité est décomposée en familles de `witnesses`, par exemple:

- `rare_token`
- `anchor_bigram`
- `categorical_value`
- `numeric_bucket`
- `token_skeleton`
- `digit_signature`
- `model_code`
- `categorical_model`
- `field_presence`
- `block_id`
- witnesses de titres libres:
  - `rare_pair`
  - `rare_triple`
  - `title_prefix3`
  - `long_pair`
  - `prefix_pair`

Ces témoins servent à construire des régions implicites `source1 x source2` qui sont ensuite raffinées par intersections successives.

### Pipeline `cw_semantic_predictive`

`cw_semantic_predictive` enchaîne:

1. extraction des witnesses et collapse symbolique
2. `semantic rescue` local sur le résidu symboliquement faible
3. `predictive rescue` à prototypes locaux
4. `strong witness rescue`
5. `asymmetric text rescue`
6. `facet rescue`

Puis, côté Member 3:

7. score direct multi-vues (`jaccard`, `tfidf`, `sbert`)
8. reranking contextuel monotone
9. `adaptive rescue` benchmark-générique sur les cas ambigus mais structurellement soutenus

Cette architecture respecte la contrainte CPU:

- le symbolique élimine la masse des non-matches
- le sémantique n'est activé que sur le résidu
- il n'y a pas de matrice complète `n x m`
- l'implémentation actuelle n'introduit pas de dépendance SciPy obligatoire pour la logique de base

## Résultats candidats

Les chiffres ci-dessous mesurent le rappel candidat, pas encore la qualité finale du matching. Ils évaluent la réduction de l'espace de recherche en sortie de Member 2.

- `abt_buy`: `5,439` candidats, `1048 / 1097` vrais matchs couverts, candidate recall `0.955333`, réduction `0.995392`
- `amazon_google`: `11,429` candidats, `1160 / 1300` vrais matchs couverts, candidate recall `0.892308`, réduction `0.997401`

Lecture rapide:

- sur `abt_buy`, la réduction reste supérieure à `99.5%`
- sur `amazon_google`, la réduction reste supérieure à `99.7%`
- le matching final dépend ensuite de Member 3, qui dispose maintenant d'un mode incrémental et reprenable

## Résultats finaux vérifiés

Les chiffres ci-dessous sont ceux de la pipeline réelle complète, avec la couche `adaptive_rescue` active dans Member 3.

| Dataset | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| `abt_buy` | `0.747` | `0.798` | `0.771` | gain réel du matching final après `adaptive_rescue` |
| `amazon_google` | `0.501` | `0.514` | `0.507` | faible gain, mais sans régression globale |
| `dblp_acm` | `0.938` | `0.967` | `0.953` | baseline forte préservée |

Lecture rapide:

- `adaptive_rescue` aide surtout quand la surface de matching est dense autour du seuil
- sur les benchmarks déjà propres comme `dblp_acm`, la couche reste pratiquement inerte
- sur les benchmarks plus ambigus comme `amazon_google`, le gain reste modeste mais positif

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Si `python` n'existe pas sur la machine, utiliser explicitement `python3`.

## Commandes utiles

### Pipeline complète

```bash
./venv/bin/python cli/run_pipeline.py
./venv/bin/python cli/run_pipeline.py --dataset abt_buy
./venv/bin/python cli/run_pipeline.py --dataset amazon_google
./venv/bin/python cli/run_interactive.py
./venv/bin/python cli/run_pipeline.py --dataset dblp_acm
```

`cli/run_interactive.py` pose les questions utiles puis construit et exécute automatiquement la bonne commande.

### Member 1

```bash
./venv/bin/python cli/run_member1.py --dataset abt_buy
./venv/bin/python cli/run_member1.py --dataset amazon_google
./venv/bin/python cli/run_member1.py --mocks-only
```

### Member 2

```bash
./venv/bin/python cli/run_member2.py --dataset abt_buy --candidate-strategy v0
./venv/bin/python cli/run_member2.py --dataset amazon_google --candidate-strategy cw_semantic_predictive
```

Seules `v0` et `cw_semantic_predictive` sont désormais conservées et exposées.

### Member 3 classique

```bash
./venv/bin/python cli/run_member3.py --dataset abt_buy
./venv/bin/python cli/run_member3.py --dataset amazon_google --candidate-strategy cw_semantic_predictive
```

### Member 3 incrémental, reprenable, limité par temps

Le matching ne dépend plus seulement de `--limit`. Il supporte maintenant:

- `--time-limit-minutes`
- `--chunk-size`
- `--online-clustering`
- `--online-cluster-every-n-chunks`
- reprise automatique
- cache d'embeddings
- fichiers d'état

Exemples:

```bash
./venv/bin/python cli/run_member3.py --dataset abt_buy --candidate-strategy cw_semantic_predictive --time-limit-minutes 60
./venv/bin/python cli/run_member3.py --dataset amazon_google --candidate-strategy cw_semantic_predictive --time-limit-minutes 120
./venv/bin/python cli/run_member3.py --dataset abt_buy --candidate-strategy cw_semantic_predictive --time-limit-minutes 60 --no-resume
./venv/bin/python cli/run_member3.py --dataset dblp_acm --candidate-strategy cw_semantic_predictive --time-limit-minutes 60 --online-clustering
./venv/bin/python cli/run_member3.py --dataset amazon_google --candidate-strategy cw_semantic_predictive --time-limit-minutes 120 --online-clustering --online-cluster-every-n-chunks 4
```

Comportement:

- le run s'arrête quand le budget temps est atteint
- il écrit des sorties partielles cohérentes
- un relancement reprend au dernier chunk traité
- les embeddings déjà calculés sont réutilisés

Quand `--online-clustering` est activé:

- les matches courants sont reconstruits à chaque checkpoint
- les clusters courants sont mis à jour pendant Step 3
- les fichiers `clusters_incremental.csv`, `merged_entities_incremental.csv` et `cluster_status.json` deviennent consultables avant la fin complète du matching
- `--online-cluster-every-n-chunks` permet de réduire le surcoût du clustering online en ne recalculant pas les clusters à chaque chunk

### Pipeline complète avec reprise de Step 3

```bash
./venv/bin/python cli/run_pipeline.py --dataset abt_buy --from-step 3 --to-step 4 --candidate-strategy cw_semantic_predictive --time-limit-minutes 60
./venv/bin/python cli/run_pipeline.py --dataset dblp_acm --from-step 3 --to-step 3 --candidate-strategy cw_semantic_predictive --time-limit-minutes 60 --online-clustering
```

Si Step 3 est partielle, Step 4 est sautée automatiquement.

### Member 4

```bash
./venv/bin/python cli/run_member4.py --dataset amazon_google --algorithm center
```

### Tests

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_block_processing.py tests/test_constraint_witness.py -q
python3 -m pytest tests/test_matching.py -v
```

## Sorties

Sorties importantes:

- `data/cleaned/<dataset>/cleaned_source1.csv`
- `data/cleaned/<dataset>/cleaned_source2.csv`
- `data/cleaned/<dataset>/ground_truth.csv`
- `output/<dataset>/blocks.csv`
- `output/<dataset>/candidate_pairs.csv`
- `output/<dataset>/match_results_*.csv`
- `output/<dataset>/clusters.csv`
- `output/<dataset>/merged_entities.csv`
- `output/<dataset>/clusters_incremental.csv`
- `output/<dataset>/merged_entities_incremental.csv`
- `output/<dataset>/matching_status.json`
- `output/<dataset>/cluster_status.json`
- `output/<dataset>/_matching_cache/<strategy>/<run_key>/progress.json`
- `output/<dataset>/_matching_cache/<strategy>/<run_key>/partial_scores.csv`
- `output/<dataset>/_matching_cache/<strategy>/<run_key>/embeddings_left.pkl`
- `output/<dataset>/_matching_cache/<strategy>/<run_key>/embeddings_right.pkl`

Pour `cw_semantic_predictive`, `candidate_pairs.csv` contient les paires retenues après collapse symbolique et rescues hybrides.

Pour le mode incrémental de Member 3:

- `matching_status.json` résume l'état courant du run
- `progress.json` stocke le dernier offset traité
- `partial_scores.csv` contient les scores déjà calculés
- les caches d'embeddings évitent de réencoder les mêmes entités à chaque reprise
- si `--online-clustering` est actif, `cluster_status.json` résume aussi l'état courant des clusters checkpointés

## Datasets

| Dataset | Format | Sources | Ordre de grandeur | Ground truth |
|---|---|---|---|---|
| `abt_buy` | CSV | Abt / Buy | ~1.1k x ~1.1k | 1,097 paires |
| `amazon_google` | CSV | Amazon / Google Shopping | ~1.4k x ~3.2k | 1,300 paires |
| `dblp_acm` | CSV | DBLP / ACM | ~2.6k x ~2.3k | labels train/valid/test |
| `spimbench` | RDF/TTL | split DBpedia | variable | pas de CSV locale |
| `rexa_dblp` | RDF/OAEI | DBLP / Rexa | variable | alignment RDF ou CSV |

Chemins surchargables:

```bash
export ABT_BUY_DIR=/chemin/vers/Abt-Buy
export AMAZON_GOOGLE_DIR=/chemin/vers/Amazon-GoogleProducts
export SPIMBENCH_DIR=/chemin/vers/spimbench/datasets
export REXA_DBLP_DIR=/chemin/vers/Rexa-DBLP
export DBLP_ACM_DIR=/chemin/vers/DBLP-ACM
```

## Fichiers clés

- `pipeline/constraint_witness.py`
  cœur du moteur witness-first et des rescues hybrides
- `pipeline/block_processing.py`
  intégration de `v0` et `cw_semantic_predictive`
- `pipeline/matching.py`
  matching incrémental, cache d'embeddings, reprise, score final contextuel, `adaptive_rescue`
- `pipeline/adaptive_rescue.py`
  couche post-reranking benchmark-générique, activée seulement sur surfaces de matching ambiguës
- optimisation d'exécution:
  - cache local des textes source par fichier nettoyé
  - chargement du modèle SBERT en local d'abord, sans appel réseau si le cache existe
  - fréquence configurable du clustering online
- `tests/test_constraint_witness.py`
  vérification des comportements du moteur
- `tests/test_matching.py`
  vérification du matching, de l'évaluation, et du score final
- `tasks/novel_er_design.md`
  note de design de la nouvelle approche
- `tasks/todo.md`
  suivi du travail de recherche
- `tasks/lessons.md`
  mémoire de travail et contraintes méthodologiques

## Limites actuelles

- `cw_semantic_predictive` utilise aujourd'hui une couche sémantique légère et CPU-first, pas encore une couche ANN plus avancée
- le matching final est maintenant beaucoup plus robuste sur `abt_buy`, mais l'évaluation complète sur gros benchmarks demande encore des runs plus longs côté Member 3
- un `limit` faible peut être trompeur, surtout sur `amazon_google`, si le sous-ensemble tronqué ne contient presque aucun vrai match

## Direction suivante

Les prochaines étapes naturelles sont:

1. évaluer systématiquement `cw_semantic_predictive` sur davantage de benchmarks
2. ajouter des ablations par famille de witnesses
3. pousser les runs longs de Member 3 avec reprise pour mesurer les vrais scores finaux
4. décider si une couche embedding CPU-first plus forte doit être ajoutée au-dessus du résidu symbolique

## Documentation complémentaire

- [README_block_processing.md](https://github.com/Massizelle/Entity-Resolution-Pipeline/blob/pipeline-orchestration/README_block_processing.md)
- [README_matching.md](https://github.com/Massizelle/Entity-Resolution-Pipeline/blob/pipeline-orchestration/README_matching.md)
- [README_pipeline_modes.md](https://github.com/Massizelle/Entity-Resolution-Pipeline/blob/pipeline-orchestration/README_pipeline_modes.md)
