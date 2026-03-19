# Knowledge Distillation

Transfer knowledge from large teacher models to smaller students — online, offline, progressive, and TAID methods.

PMetal supports multiple distillation methods and loss functions for compressing large models into smaller, deployable ones.

## Methods

### Online Distillation
Live teacher inference during training. Highest quality but slowest — both teacher and student must fit in memory.

```bash
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student unsloth/Qwen3.5-0.8B-Base \
  --dataset train.jsonl
```

### Offline Distillation
Pre-cache teacher logits to disk with compression. Faster training at the cost of disk space.

```bash
pmetal distill --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-0.6B --dataset train.jsonl --offline
```

### Progressive Distillation
Gradually increase task difficulty during distillation for curriculum-style knowledge transfer.

### TAID (Temporally Adaptive Interpolated Distillation)
ICLR 2025 SOTA method. Available via `TaidDistiller` in the library API.

### Cross-Vocabulary Distillation
Distill between models with different tokenizers. PMetal handles vocabulary alignment automatically.

## Loss Functions

### Token-Level

| Loss | Description |
|------|-------------|
| KL Divergence | Standard distribution matching |
| Jensen-Shannon | Symmetric divergence |
| Soft Cross-Entropy | Temperature-scaled cross-entropy |
| TVD | Total variation distance |
| Hinge Ranking | Rank-based loss |
| Logistic Ranking | Logistic rank loss |

### Hidden State

| Loss | Description |
|------|-------------|
| MSE | Mean squared error on hidden states |
| Cosine | Cosine similarity matching |
| L1 | L1 distance |

## Reasoning-Aware Distillation

For reasoning models, PMetal supports rationale distillation that preserves the thinking process, not just the final answer.

## See Also

- [pmetal distill](/cli/distill/) — CLI distillation
- [Training Overview](/training/overview/) — All training methods
