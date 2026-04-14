# Pipeline Modes

## Goal

This file documents the research-oriented pipeline modes added on top of the original end-to-end ER workflow.

## Modes Summary

| Mode | Step 2 behavior | Step 3 behavior | Main purpose |
|------|------------------|-----------------|--------------|
| `v0` | classical candidate generation | classical matching order | stable baseline |
| `cw_semantic_predictive` | witness-first + semantic/predictive rescue | full scoring with resumable execution | benchmark-generic CPU-first retrieval |

## Full Pipeline Examples

```bash
# V0 baseline
./venv/bin/python cli/run_pipeline.py --dataset amazon_google --candidate-strategy v0

# Current research mode
./venv/bin/python cli/run_pipeline.py --dataset amazon_google --candidate-strategy cw_semantic_predictive --time-limit-minutes 120
```

## Research Interpretation

### V0

Reference point for time, recall, and output quality.

### `cw_semantic_predictive`

Tests the current research hypothesis:

> witness-first reduction plus residual semantic and predictive rescue can keep candidate reduction above 99% while preserving strong retrieval quality across heterogeneous benchmarks.

## What To Measure

At minimum, record for each mode:

- candidate count after step 2
- wall-clock time of step 3
- number of confirmed matches
- precision / recall / F1
- online-clustering checkpoints and partial metrics when Step 3 is budgeted
