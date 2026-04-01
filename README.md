# Entity-Resolution-Pipeline

End-to-end Entity Resolution pipeline sur 3 datasets (Abt-Buy, Amazon-Google, SPIMBench), aligné avec le survey *"End-to-End Entity Resolution for Big Data"* (Christophides et al.).

---

## Architecture du pipeline

```
Member 1  →  Ingestion + Token Blocking       →  blocks.csv
Member 2  →  Candidate Pairs                  →  candidate_pairs.csv
Member 3  →  Entity Matching                   →  match_results_*.csv
Member 4  →  Clustering                       →  clusters.csv
Member 5  →  pyJedAI (indépendant)
```

---

## Member 3 — Entity Matching

### Fondements théoriques (article Section 5)

Le matching implémente deux niveaux de similarité, conformément à l'article :

1. **Value Similarity** (Section 5.1 — attribute-based) : compare les paires indépendamment via Jaccard, TF-IDF et SBERT. Suffit pour les entités *strongly similar* (valueSim > 0.6).

2. **Neighbor Similarity** (Section 5.2 — collective/relationship-based) : pour les entités *nearly similar* (valueSim < 0.5), exploite la structure des blocs — si les voisins de A matchent les voisins de B, alors A et B sont probablement un match. Indispensable pour les données schema-agnostic (SPIMBench).

Score final : `score = 0.6 × valueSim + 0.4 × neighborSim`

### Structure du module (`matching.py`)

```
ÉTAPE 1 — Préparation
  load_and_prepare()         → id_A, id_B, text_A, text_B

ÉTAPE 2 — Value Similarity
  jaccard_sim()              → |A∩B| / |A∪B| sur tokens
  tfidf_sim()                → cosine similarity TF-IDF (sklearn)
  sbert_sim()                → cosine similarity embeddings (sentence-transformers)

ÉTAPE 3 — Matches provisoires
  compute_value_sim()        → score moyen + seuil 0.50

ÉTAPE 4 — Neighbor Similarity (collective ER)
  build_neighbor_index()     → {entity_id: set(voisins)} depuis blocks.csv
  compute_neighbor_sim()     → voisins matchés / total voisins

ÉTAPE 5 — Score final
  combine_scores()           → valueSim + neighborSim → décision finale

ÉTAPE 6 — Évaluation
  evaluate()                 → Precision, Recall, F1 vs ground_truth.csv

ÉTAPE 7 — Orchestration
  run_matching()             → pipeline complet sur les 3 datasets
```

### Inputs / Outputs

| Input | Source |
|---|---|
| `output/<dataset>/candidate_pairs.csv` | Member 2 |
| `data/cleaned/<dataset>/cleaned_source1.csv` | Member 1 |
| `data/cleaned/<dataset>/cleaned_source2.csv` | Member 1 |
| `output/<dataset>/blocks.csv` | Member 1 |

| Output | Contenu |
|---|---|
| `match_results_jaccard.csv` | `id_A, id_B, similarity_score, is_match` |
| `match_results_tfidf.csv` | `id_A, id_B, similarity_score, is_match` |
| `match_results_sbert.csv` | `id_A, id_B, similarity_score, is_match` |
| `match_results_combined.csv` | value similarity seule |
| `match_results_collective.csv` | value + neighbor similarity (final) |

### Contrat de données

```python
# is_match est toujours int (jamais bool)
df[df["is_match"] == 1]   # correct
df[df["is_match"] == True] # incorrect
```

### Commandes

```bash
pip install -r requirements.txt
python matching.py --dataset abt_buy
python matching.py --dataset amazon_google
python matching.py --dataset spimbench
python matching.py --dataset all
```

---

## Datasets

| Dataset | Format | Type ER |
|---|---|---|
| Abt-Buy | CSV | Clean-Clean |
| Amazon-Google | CSV | Clean-Clean |
| SPIMBench | RDF/TTL → CSV | Clean-Clean (schema-agnostic) |
