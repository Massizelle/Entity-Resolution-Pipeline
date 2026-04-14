# Block Processing Notes

## Scope

This document tracks the current design of `pipeline/block_processing.py` and the candidate strategies exposed by Member 2.

## Output Contract

Baseline output is still `candidate_pairs.csv` with at least:

```text
id_A,id_B
```

Experimental modes may append ranking columns such as:

```text
shared_blocks,jaccard_blocks,min_block_size,avg_block_size,rarity_score,promise_score,candidate_rank
```

Member 3 must treat `id_A` and `id_B` as mandatory and all other columns as optional ranking metadata.

## Candidate Strategies

### V0

Current baseline from the survey-inspired pipeline:

- token-block overlap after purging
- Jaccard pruning on block sets
- no ranking metadata

This mode preserves the historical behavior of the repository.

### `cw_semantic_predictive`

Current research strategy:

- witness-first candidate collapse
- semantic rescue on the residual space
- strong-witness rescue
- asymmetric-text rescue
- facet rescue
- predictive rescue

Goal: reduce comparisons aggressively while staying benchmark-generic and CPU-first.

## CLI Examples

```bash
# Baseline
./venv/bin/python cli/run_member2.py --dataset amazon_google --candidate-strategy v0

# Current research strategy
./venv/bin/python cli/run_member2.py --dataset amazon_google --candidate-strategy cw_semantic_predictive
```

## Important Caveats

- `v0` is the historical baseline.
- `cw_semantic_predictive` is the only research strategy currently kept in the CLI.
- The current direction favors benchmark-generic retrieval over dataset-specific heuristics.
