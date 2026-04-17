# Entity-Resolution-Pipeline

Pipeline d'Entity Resolution multi-datasets, orienté CPU-first, centré sur une seule stratégie finale:

- génération de candidats `cw_semantic_predictive`
- matching multi-vues incrémental et reprenable
- clustering par `connected_components`

Le dépôt vise une méthode benchmark-générique: on réduit fortement l'espace candidat avant le matching détaillé, sans branches algorithmiques spécifiques à un dataset.

## Idée du projet

Le pipeline ER classique reste souvent `pair-first`:

1. on génère des candidats
2. on matérialise encore beaucoup de paires explicites
3. on les score une par une

Le projet essaie de changer cela.

L'idée centrale est:

- représenter chaque entité par des `witnesses` hétérogènes
- construire des régions cross-source implicites au lieu d'énumérer directement toutes les paires
- raffiner ces régions par evidence collapse
- ne matérialiser des paires explicites qu'au moment où le sous-espace est déjà petit

Cette logique est implémentée principalement dans `pipeline/constraint_witness.py` et exposée dans Member 2.

## Architecture

Le workflow suit 4 étapes principales, complétées par un baseline externe :

1. `Member 1`
   ingestion, normalisation, token blocking
2. `Member 2`
   purge des blocks, witness-first candidate generation, semantic/predictive rescue
3. `Member 3`
   matching multi-vues (`jaccard`, `tfidf`, `sbert`) + reranking contextuel monotone
4. `Member 4`
   clustering par connected components + fusion d'attributs
5. `Member 5`
   baseline pyJedAI (pipeline standard : StandardBlocking → BlockPurging → BlockFiltering → WeightedEdgePruning → EntityMatching → UniqueMappingClustering)

## Modules principaux

- `pipeline/data_ingestion.py`
  ingestion et normalisation multi-formats
- `pipeline/blocking.py`
  token blocking initial
- `pipeline/block_processing.py`
  génération des candidats à partir du moteur witness-first
- `pipeline/constraint_witness.py`
  extraction des witnesses, collapse, rescues sémantiques et prédictifs
- `pipeline/matching.py`
  scoring, matching incrémental, reprise, cache d'embeddings
- `pipeline/clustering.py`
  clustering final et fusion d'entités
- `pipeline/adaptive_rescue.py`
  couche post-reranking activée sur les surfaces ambiguës
- `pipeline/progress.py`
  barres de progression terminales
- `pipeline/extra_datasets.json`
  configuration des datasets supplémentaires

## Entrypoints

- `cli/run_pipeline.py`
  pipeline complète
- `cli/run_member1.py`
  étape 1 seule
- `cli/run_member2.py`
  étape 2 seule
- `cli/run_member3.py`
  étape 3 seule
- `cli/run_member4.py`
  étape 4 seule
- `cli/run_interactive.py`
  launcher interactif
- `cli/integrate_dataset.py`
  aide à l'intégration de nouveaux datasets tabulaires
- `cli/execution_pyjedai.py`
  baseline pyJedAI (Member 5)

## Datasets

Datasets principaux déjà intégrés (données présentes dans `data/raw/`) :

- `abt_buy`
- `amazon_google`
- `dblp_acm`
- `dblp_scholar`
- `dbpedia_imdb`
- `walmart_amazon`

Dataset supporté par le pipeline custom uniquement (données non incluses) :

- `rexa_dblp` — nécessite un téléchargement externe (sources RDF/OAEI)

Espaces de données:

- `data/raw/`
  données brutes
- `data/cleaned/<dataset>/`
  données normalisées
- `output/<dataset>/`
  artefacts intermédiaires et finaux
- `output/mock/<dataset>/`
  fixtures mock

Pour `dblp_acm`, l'ingestion consomme:

- `tableA.csv`
- `tableB.csv`
- `train.csv`
- `valid.csv`
- `test.csv`

Pour `rexa_dblp`, l'ingestion accepte des sources RDF/OAEI et un alignment RDF/XML ou CSV.

## Résultats représentatifs

### Candidate generation

Les chiffres ci-dessous mesurent le rappel candidat à la sortie de Member 2:

- `abt_buy`: `5,439` candidats, `1048 / 1097` vrais matchs couverts, recall candidat `0.955333`, réduction `0.995392`
- `amazon_google`: `11,429` candidats, `1160 / 1300` vrais matchs couverts, recall candidat `0.892308`, réduction `0.997401`

### Résultats finaux vérifiés

| Dataset | Precision | Recall | F1 |
|---|---:|---:|---:|
| `abt_buy` | `0.747` | `0.798` | `0.771` |
| `amazon_google` | `0.501` | `0.514` | `0.507` |
| `dblp_acm` | `0.938` | `0.967` | `0.953` |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commandes utiles

### Pipeline complète

```bash
./venv/bin/python cli/run_pipeline.py
./venv/bin/python cli/run_pipeline.py --dataset abt_buy
./venv/bin/python cli/run_pipeline.py --dataset amazon_google
./venv/bin/python cli/run_pipeline.py --dataset dblp_acm
./venv/bin/python cli/run_interactive.py
```

### Member 1

```bash
./venv/bin/python cli/run_member1.py --dataset abt_buy
./venv/bin/python cli/run_member1.py --dataset amazon_google
./venv/bin/python cli/run_member1.py --mocks-only
```

### Member 2

```bash
./venv/bin/python cli/run_member2.py --dataset abt_buy
./venv/bin/python cli/run_member2.py --dataset amazon_google
./venv/bin/python cli/run_member2.py --write-stats
```

### Member 3

```bash
./venv/bin/python cli/run_member3.py --dataset abt_buy
./venv/bin/python cli/run_member3.py --dataset amazon_google --time-limit-minutes 120
./venv/bin/python cli/run_member3.py --dataset dblp_acm --time-limit-minutes 60 --online-clustering
./venv/bin/python cli/run_member3.py --dataset abt_buy --time-limit-minutes 60 --no-resume
```

Le matching supporte:

- `--time-limit-minutes`
- `--chunk-size`
- `--online-clustering`
- `--online-cluster-every-n-chunks`
- reprise automatique
- cache d'embeddings

### Member 4

```bash
./venv/bin/python cli/run_member4.py --dataset abt_buy
./venv/bin/python cli/run_member4.py --dataset amazon_google
```

### Member 5 — Baseline pyJedAI

```bash
./venv/bin/python cli/execution_pyjedai.py --dataset abt_buy
./venv/bin/python cli/execution_pyjedai.py --dataset amazon_google
./venv/bin/python cli/execution_pyjedai.py --dataset dblp_acm
./venv/bin/python cli/execution_pyjedai.py --dataset dblp_scholar
./venv/bin/python cli/execution_pyjedai.py --dataset dbpedia_imdb
./venv/bin/python cli/execution_pyjedai.py --dataset walmart_amazon
./venv/bin/python cli/execution_pyjedai.py --dataset all
```

Sorties produites dans `output/<dataset>/` :
- `pyjedai_candidate_pairs.csv`
- `pyjedai_match_results.csv`
- `pyjedai_clusters.csv`
- `pyjedai_metrics.json`

### Pipeline avec reprise à partir de Step 3

```bash
./venv/bin/python cli/run_pipeline.py --dataset abt_buy --from-step 3 --to-step 4 --time-limit-minutes 60
./venv/bin/python cli/run_pipeline.py --dataset dblp_acm --from-step 3 --to-step 3 --time-limit-minutes 60 --online-clustering
```

Si Step 3 est partielle, Step 4 est sautée automatiquement.

## Sorties importantes

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

`candidate_pairs.csv` peut contenir des métadonnées de ranking comme `certificate_score`.

## Tests

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_block_processing.py tests/test_constraint_witness.py -q
python3 -m pytest tests/test_matching.py -v
python3 -m pytest tests/test_clustering.py -v
python3 -m pytest tests/test_data_ingestion.py tests/test_ingestion_paths.py tests/test_mock_paths.py -v
```

Fichiers de test disponibles :

- `tests/test_block_processing.py`
- `tests/test_constraint_witness.py`
- `tests/test_matching.py`
- `tests/test_clustering.py`
- `tests/test_data_ingestion.py`
- `tests/test_ingestion_paths.py`
- `tests/test_mock_paths.py`

## Variables d'environnement utiles

```bash
export ABT_BUY_DIR=/chemin/vers/Abt-Buy
export AMAZON_GOOGLE_DIR=/chemin/vers/Amazon-GoogleProducts
export DBLP_ACM_DIR=/chemin/vers/DBLP-ACM
export REXA_DBLP_DIR=/chemin/vers/Rexa-DBLP
```

## Fichiers clés

- `pipeline/constraint_witness.py`
  cœur du moteur witness-first et des rescues
- `pipeline/block_processing.py`
  intégration de la stratégie finale de génération de candidats
- `pipeline/matching.py`
  matching incrémental, cache d'embeddings, reprise, score final contextuel
- `pipeline/adaptive_rescue.py`
  couche post-reranking activée sur les surfaces ambiguës
- `tests/test_constraint_witness.py`
  vérification du moteur witness-first
- `tests/test_matching.py`
  vérification du matching

## Limites actuelles

- la qualité finale dépend encore fortement du coût SBERT en Step 3
- `amazon_google` reste beaucoup plus ambigu que `abt_buy` ou `dblp_acm`
- un `limit` faible peut donner une vision trompeuse de la qualité finale

## Direction suivante

Les prochaines étapes naturelles sont:

1. comparer les résultats finaux avec des systèmes externes de référence
2. ajouter des graphes plus propres dans le rapport
3. pousser les longs runs de Member 3 sur davantage de benchmarks
4. décider si une couche embedding CPU-first plus forte doit être ajoutée au-dessus du résidu symbolique

