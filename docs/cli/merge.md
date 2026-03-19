# pmetal merge

Merge two or more models using 12 merge strategies.

Merge multiple models into one using various merge strategies. Supports GPU-accelerated merging, FP8-aware merging, and async double-buffered streaming for large models.

## Usage

```bash
pmetal merge \
  --models <MODEL_A> <MODEL_B> [<MODEL_C>...] \
  --method <METHOD> \
  [OPTIONS]
```

## Examples

```bash
# SLERP merge
pmetal merge \
  --models model-a model-b \
  --method slerp --t 0.5

# TIES merge with sparsification
pmetal merge \
  --models base-model ft-model-1 ft-model-2 \
  --method ties --density 0.5

# DARE-TIES with random pruning
pmetal merge \
  --models model-a model-b \
  --method dare_ties --density 0.7
```

## Strategies

| Method | Description |
|--------|-------------|
| `linear` | Simple weighted averaging |
| `slerp` | Spherical linear interpolation |
| `ties` | Task arithmetic with sparsification and sign consensus |
| `dare_ties` | Random pruning with rescaling (TIES variant) |
| `dare_linear` | Random pruning with rescaling (linear variant) |
| `task_arithmetic` | Task vector arithmetic |
| `della` | Adaptive magnitude-based pruning |
| `della_linear` | Adaptive magnitude pruning (linear variant) |
| `breadcrumbs` | Breadcrumbs merge strategy |
| `model_stock` | Geometric interpolation based on task vector similarity |
| `nearswap` | Near-swap merge strategy |
| `passthrough` | Layer passthrough composition |

Additional library-only strategies: `RamMerge`, `SouperMerge`, `MultiSlerpMerge`.

## See Also

- [Model Merging](/models/merging/) — Detailed merge documentation
