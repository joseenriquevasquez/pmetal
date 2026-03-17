# pmetal-trainer

Training loops and optimization strategies for LLM fine-tuning.

## Overview

This crate provides the training infrastructure for PMetal, including various training methods, learning rate scheduling, checkpointing, and callback systems.

## Training Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **SFT** | Supervised Fine-Tuning | General instruction tuning |
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning |
| **DPO** | Direct Preference Optimization | Preference-based alignment |
| **GRPO** | Group Relative Policy Optimization | Efficient PPO alternative |
| **GSPO** | Group Sequence Policy Optimization | Fixes GRPO length bias |
| **DAPO** | Decoupled Clip and Dynamic Sampling PO | ByteDance's 4 GRPO improvements |
| **PPO** | Proximal Policy Optimization | RLHF with reward model |
| **ORPO** | Odds Ratio Preference Optimization | Reference-free alignment |
| **SimPO** | Simple Preference Optimization | Simplified preference learning |
| **KTO** | Kahneman-Tversky Optimization | Unpaired preference data |
| **Online DPO** | Online Direct Preference Optimization | DPO with online sampling |
| **Distillation** | Knowledge distillation | Teacher→student transfer |
| **ANE** | Apple Neural Engine training | Power-efficient on-device training |
| **Diffusion** | LLaDA-style diffusion training | Experimental |

## Usage

### Basic Training Loop

```rust
use pmetal_trainer::{TrainingLoop, TrainingConfig};

let config = TrainingConfig {
    batch_size: 4,
    gradient_accumulation_steps: 4,
    learning_rate: 2e-4,
    epochs: 1,
    max_grad_norm: 1.0,
    ..Default::default()
};

let mut trainer = TrainingLoop::new(model, optimizer, config)?;

// Train with optional callbacks
trainer.train(&dataloader, callbacks)?;
```

### With Checkpointing

```rust
use pmetal_trainer::CheckpointManager;

let checkpoint_mgr = CheckpointManager::new("output/checkpoints");

// Resume from checkpoint if available
if let Some(ckpt) = checkpoint_mgr.latest()? {
    trainer.load_checkpoint(&ckpt)?;
}

// Save checkpoints during training
trainer.train_with_checkpoints(&dataloader, &checkpoint_mgr, save_every: 500)?;
```

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| **AdamW Groups** | AdamW with per-parameter-group learning rates |
| **Adam 8-bit** | Memory-efficient 8-bit Adam optimizer |
| **Schedule-Free** | Optimizer without learning rate schedules |
| **Metal Fused** | GPU-accelerated AdamW parameter updates |

## Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| Constant | Fixed learning rate |
| Linear | Linear warmup and decay |
| Cosine | Cosine annealing |
| Cosine with Restarts | Cosine with periodic warm restarts |
| Polynomial | Polynomial decay |
| WSD | Warmup-Stable-Decay schedule |

## Modules

| Module | Description |
|--------|-------------|
| `training_loop` | Main training orchestration |
| `sft` | Supervised fine-tuning trainer |
| `lora_trainer` | LoRA-specific training |
| `dpo` | Direct Preference Optimization |
| `grpo` | Group Relative Policy Optimization |
| `gspo` | Group Sequence Policy Optimization |
| `dapo` | Decoupled Clip and Dynamic Sampling PO |
| `ane_training` | ANE training loop (feature-gated: `ane`) |
| `ppo` | Proximal Policy Optimization |
| `orpo` | Odds Ratio Preference Optimization |
| `simpo` | Simple Preference Optimization |
| `kto` | Kahneman-Tversky Optimization |
| `online_dpo` | Online DPO with sampling |
| `distillation` | Knowledge distillation orchestration |
| `diffusion` | Diffusion-based training |
| `adamw_groups` | AdamW with parameter groups |
| `adam8bit` | 8-bit Adam optimizer |
| `schedule_free` | Schedule-free optimizer |
| `metal_fused` | Metal-accelerated optimizer |
| `adaptive_lr` | EMA-based adaptive learning rate control |
| `checkpoint` | Checkpoint save/load |
| `checkpointing` | Gradient checkpointing |
| `scheduler` | Learning rate schedulers |
| `callbacks` | Training callbacks (`MetricsJsonCallback`, `StepMetrics`) |
| `param_groups` | Per-layer learning rates |
| `distributed_bridge` | Distributed training sync (feature-gated: `distributed`) |

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Micro-batch size | 4 |
| `gradient_accumulation_steps` | Accumulation steps | 1 |
| `learning_rate` | Initial learning rate | 2e-4 |
| `max_grad_norm` | Gradient clipping | 1.0 |
| `warmup_steps` | LR warmup steps | 0 |
| `weight_decay` | L2 regularization | 0.0 |

## License

MIT OR Apache-2.0
