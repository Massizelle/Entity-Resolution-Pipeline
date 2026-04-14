# Matching Notes

## Scope

This document tracks the behavior of `pipeline/matching.py`, especially how Member 3 consumes ranked candidates from Member 2.

## Current Matching Stack

For every retained candidate pair:

1. merge source records into `text_A` and `text_B`
2. compute value similarity:
   - Jaccard
   - TF-IDF cosine
   - SBERT cosine
3. compute `value_score` as the mean of the three similarities
4. compute neighbor similarity for provisional matches
5. combine scores:

```text
final_score = 0.6 * value_score + 0.4 * neighbor_score
```

## Candidate Consumption Modes

### `v0`

Reads the baseline `candidate_pairs.csv` output from Member 2.

### `cw_semantic_predictive`

Consumes the witness-first candidate set emitted by Member 2. Matching remains full-scoring, but execution can be:

- chunked
- time-budgeted
- resumable
- optionally paired with online clustering checkpoints

## Evaluation Contract

`evaluate()` now treats only rows with `is_match == 1` as predictions.

That matters because the previous behavior accidentally treated every candidate row as a predicted match during evaluation.

## CLI Examples

```bash
# Baseline
./venv/bin/python cli/run_member3.py --dataset amazon_google --candidate-strategy v0

# Current research strategy with resumable matching
./venv/bin/python cli/run_member3.py --dataset amazon_google --candidate-strategy cw_semantic_predictive --time-limit-minutes 120

# Same with online clustering checkpoints
./venv/bin/python cli/run_member3.py --dataset amazon_google --candidate-strategy cw_semantic_predictive --time-limit-minutes 120 --online-clustering
```

## Important Caveats

- SBERT remains the dominant CPU cost.
- If `value_score` never crosses the provisional threshold, neighbor similarity has no chance to help.
- Ranking quality before SBERT is therefore the key lever for CPU-first discovery.
