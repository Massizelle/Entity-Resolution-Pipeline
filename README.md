# Entity-Resolution-Pipeline

End-to-end Entity Resolution pipeline sur 3 datasets (Abt-Buy, Amazon-Google, SPIMBench), aligné avec le survey *"End-to-End Entity Resolution for Big Data"* (Christophides et al.).

---

## Architecture du pipeline

```
Raw Data (CSV / RDF-TTL)
    │
    ▼
Member 1 — Ingestion + Token Blocking
    │   data_ingestion.py + blocking.py
    │   → data/cleaned/<dataset>/cleaned_source{1,2}.csv
    │   → output/<dataset>/blocks.csv
    │
    ▼
Member 2 — Block Processing (Purging + Meta-Blocking)
    │   block_processing.py
    │   → output/<dataset>/candidate_pairs.csv
    │
    ▼
Member 3 — Entity Matching  ← ce module
    │   matching.py
    │   → output/<dataset>/match_results_*.csv
    │
    ▼
Member 4 — Clustering + Entity Merging
    │   → output/<dataset>/clusters.csv
    │   → output/<dataset>/merged_entities.csv
    │
    ▼
Member 5 — pyJedAI (indépendant)
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Utilisation

### Lancer le pipeline complet (Member 1 → Member 2 → Member 3)

```bash
# Member 1 — Ingestion + Blocking
python run_member1.py --dataset abt_buy
python run_member1.py --dataset amazon_google

# Member 2 — Block Processing
python run_member2.py --dataset abt_buy
python run_member2.py --dataset amazon_google

# Member 3 — Entity Matching
python run_member3.py
```

### Générer les fichiers mock (sans données brutes)

```bash
python run_member1.py --mocks-only
```

### Lancer les tests

```bash
python -m pytest tests/ -v
```

---

## Member 1 — Ingestion + Token Blocking

**Fichiers** : `data_ingestion.py`, `blocking.py`, `run_member1.py`

**Étapes** :
1. Chargement et normalisation des données brutes (CSV ou RDF/TTL)
2. Construction d'un index inversé de tokens (schema-agnostic)
3. Block purging : suppression des blocs trop grands

**Outputs** :
```
data/cleaned/<dataset>/cleaned_source1.csv
data/cleaned/<dataset>/cleaned_source2.csv
data/cleaned/<dataset>/ground_truth.csv   (si disponible)
output/<dataset>/blocks.csv
```

---

## Member 2 — Block Processing

**Fichiers** : `block_processing.py`, `run_member2.py`

**Étapes** :
1. Block purging (suppression des blocs > `max_block_size`)
2. Meta-blocking : graphe de similarité Jaccard entre entités → filtrage des paires peu prometteuses

**Outputs** :
```
output/<dataset>/candidate_pairs.csv
output/<dataset>/member2_stats.json   (optionnel)
```

---

## Member 3 — Entity Matching

### Fondements théoriques (survey Section 5)

Le matching implémente deux niveaux de similarité :

1. **Value Similarity** (Section 5.1 — attribute-based) : compare les paires via Jaccard, TF-IDF et SBERT. Suffit pour les entités *strongly similar* (valueSim > 0.6).

2. **Neighbor Similarity** (Section 5.2 — collective/relationship-based) : pour les entités *nearly similar* (valueSim < 0.5), exploite la structure des blocs — si les voisins de A matchent les voisins de B, A et B sont probablement un match.

**Score final** : `final_score = 0.6 × value_score + 0.4 × neighbor_score`

### Structure du module (`matching.py`)

| Étape | Fonction | Description |
|-------|----------|-------------|
| 1 | `load_and_prepare()` | Charge les paires et fusionne les textes des entités |
| 2 | `jaccard_sim()` | Similarité Jaccard sur tokens |
| 2 | `tfidf_sim()` | Cosine similarity TF-IDF (par batch) |
| 2 | `sbert_sim()` | Cosine similarity embeddings SBERT |
| 3 | `compute_value_sim()` | Moyenne des 3 scores → `value_score`, `is_match` |
| 4 | `build_neighbor_index()` | Index `{entity_id: set(voisins)}` depuis `blocks.csv` |
| 4 | `compute_neighbor_sim()` | Score = voisins matchés / total voisins |
| 5 | `combine_scores()` | Score final pondéré + décision finale |
| 6 | `evaluate()` | Precision, Recall, F1 vs `ground_truth.csv` |
| 7 | `run_matching()` | Orchestration complète sur un dataset |

### Inputs

| Fichier | Source |
|---------|--------|
| `output/<dataset>/candidate_pairs.csv` | Member 2 |
| `data/cleaned/<dataset>/cleaned_source1.csv` | Member 1 |
| `data/cleaned/<dataset>/cleaned_source2.csv` | Member 1 |
| `output/<dataset>/blocks.csv` | Member 1 |
| `data/cleaned/<dataset>/ground_truth.csv` | Member 1 (optionnel) |

### Outputs

| Fichier | Colonnes | Description |
|---------|----------|-------------|
| `match_results_jaccard.csv` | `id_A, id_B, similarity_score, is_match` | Jaccard seul |
| `match_results_tfidf.csv` | `id_A, id_B, similarity_score, is_match` | TF-IDF seul |
| `match_results_sbert.csv` | `id_A, id_B, similarity_score, is_match` | SBERT seul |
| `match_results_combined.csv` | `id_A, id_B, similarity_score, is_match` | Moyenne value similarity |
| `match_results_collective.csv` | `id_A, id_B, final_score, is_match` | Value + Neighbor (collectif) |

### Contrat de données

```python
# is_match est toujours int (jamais bool)
df[df["is_match"] == 1]    # correct
df[df["is_match"] == True] # incorrect
```

---

## Datasets

| Dataset | Format | Sources | Ground Truth |
|---------|--------|---------|--------------|
| Abt-Buy | CSV | Abt.com / Buy.com (~1K entités chacun) | oui |
| Amazon-Google | CSV | Amazon / Google Shopping (~1.4K / ~3.2K) | oui (1300 paires) |
| SPIMBench | RDF/TTL | DBpedia (split odd/even) | non |

---

## Structure des répertoires

```
Entity-Resolution-Pipeline/
├── data/
│   ├── raw/                        # Données brutes (CSV ou TTL)
│   └── cleaned/<dataset>/          # Données normalisées (Member 1)
├── output/
│   ├── <dataset>/                  # Résultats par dataset
│   └── mock/                       # Fixtures de test pour tous les membres
├── tests/
│   ├── test_block_processing.py    # Tests Member 2
│   └── test_matching.py            # Tests Member 3
├── matching.py                     # Member 3
├── block_processing.py             # Member 2
├── blocking.py                     # Member 1
├── data_ingestion.py               # Member 1
├── run_member1.py                  # CLI Member 1
├── run_member2.py                  # CLI Member 2
├── run_member3.py                  # CLI Member 3
├── create_mocks.py                 # Génération de fixtures de test
└── requirements.txt
```
