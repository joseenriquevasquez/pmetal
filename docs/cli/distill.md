# pmetal distill

Knowledge distillation — transfer knowledge from a teacher model to a smaller student.

Distill knowledge from a larger teacher model into a smaller student model. Supports online, offline, and progressive methods.

## Usage

```bash
pmetal distill \
  --teacher <TEACHER_MODEL> \
  --student <STUDENT_MODEL> \
  --dataset <DATASET> \
  [OPTIONS]
```

## Examples

```bash
# Online distillation (live teacher inference)
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3.5-0.8B-Base \
  --dataset train.jsonl

# Offline distillation (cached logits)
pmetal distill \
  --teacher Qwen/Qwen3-4B \
  --student Qwen/Qwen3.5-0.8B-Base \
  --dataset train.jsonl \
  --offline \
  --offline-cache ./output/distilled/teacher_logits
```

## Methods

| Method | Description |
|--------|-------------|
| Online | Live teacher inference during training — highest quality, slowest |
| Offline | Pre-cached logits with compression — faster, uses more disk |
| Progressive | Gradually increase distillation difficulty |

## Loss Functions

### Token-Level
- **KL Divergence** — Standard distribution matching
- **Jensen-Shannon** — Symmetric divergence
- **Soft Cross-Entropy** — Temperature-scaled cross-entropy
- **TVD** — Total variation distance
- **Hinge Ranking** — Rank-based loss
- **Logistic Ranking** — Logistic rank loss

### Hidden State
- **MSE** — Mean squared error on hidden states
- **Cosine** — Cosine similarity matching
- **L1** — L1 distance

## See Also

- [Training Overview](/training/overview/) — All training methods
- [TAID Distillation](/training/distillation/) — Advanced TAID method (library only)
