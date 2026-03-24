//! 1F1B (One Forward, One Backward) pipeline schedule.
//!
//! The classic schedule from PipeDream/Megatron-LM:
//!
//! ```text
//! Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 ... B_last  W
//! Stage 1:    F0 F1 F2 B0 F3 B1 F4 B2 ...  B_last  W
//! Stage 2:       F0 F1 B0 F2 B1 F3 B2 ...   B_last W
//! Stage 3:          F0 B0 F1 B1 F2 B2 ...    B_last W
//! ```
//!
//! Warmup: `(num_stages - stage - 1)` extra forwards before steady state.
//! Steady state: alternating 1 forward + 1 backward.
//! Cooldown: remaining backwards + weight update.
//!
//! Pipeline bubble fraction: `(P-1) / (M+P-1)` where P=stages, M=micro-batches.

/// An action in the 1F1B pipeline schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroBatchAction {
    /// Run forward pass for micro-batch `id`.
    Forward(usize),
    /// Run backward pass for micro-batch `id`.
    Backward(usize),
    /// Send activations to the next stage for micro-batch `id`.
    SendActivation(usize),
    /// Receive activations from the previous stage for micro-batch `id`.
    RecvActivation(usize),
    /// Send gradients to the previous stage for micro-batch `id`.
    SendGradient(usize),
    /// Receive gradients from the next stage for micro-batch `id`.
    RecvGradient(usize),
    /// Apply weight updates after all micro-batches complete.
    WeightUpdate,
}

/// Generate a 1F1B schedule for a given pipeline stage.
///
/// # Arguments
///
/// * `num_stages` — Total number of pipeline stages (P)
/// * `num_micro_batches` — Number of micro-batches (M, must be >= P)
/// * `stage` — This stage's index (0-indexed)
///
/// # Returns
///
/// Ordered list of actions this stage should execute.
///
/// # Panics
///
/// Panics if `num_micro_batches < num_stages`.
pub fn schedule_1f1b(
    num_stages: usize,
    num_micro_batches: usize,
    stage: usize,
) -> Vec<MicroBatchAction> {
    assert!(
        num_micro_batches >= num_stages,
        "1F1B requires num_micro_batches ({}) >= num_stages ({})",
        num_micro_batches,
        num_stages
    );
    assert!(stage < num_stages, "stage {stage} >= num_stages {num_stages}");

    let mut schedule = Vec::new();
    let is_first = stage == 0;
    let is_last = stage == num_stages - 1;

    // Warmup phase: (num_stages - stage - 1) extra forwards.
    let num_warmup = num_stages - stage - 1;
    let num_warmup = num_warmup.min(num_micro_batches);

    for mb in 0..num_warmup {
        if !is_first {
            schedule.push(MicroBatchAction::RecvActivation(mb));
        }
        schedule.push(MicroBatchAction::Forward(mb));
        if !is_last {
            schedule.push(MicroBatchAction::SendActivation(mb));
        }
    }

    // Steady state: alternating 1F 1B.
    let num_steady = num_micro_batches - num_warmup;
    for i in 0..num_steady {
        let fwd_mb = num_warmup + i;
        let bwd_mb = i;

        // Forward for new micro-batch.
        if !is_first {
            schedule.push(MicroBatchAction::RecvActivation(fwd_mb));
        }
        schedule.push(MicroBatchAction::Forward(fwd_mb));
        if !is_last {
            schedule.push(MicroBatchAction::SendActivation(fwd_mb));
        }

        // Backward for earlier micro-batch.
        if !is_last {
            schedule.push(MicroBatchAction::RecvGradient(bwd_mb));
        }
        schedule.push(MicroBatchAction::Backward(bwd_mb));
        if !is_first {
            schedule.push(MicroBatchAction::SendGradient(bwd_mb));
        }
    }

    // Cooldown: remaining backwards.
    for i in num_steady..num_micro_batches {
        let bwd_mb = i;
        if !is_last {
            schedule.push(MicroBatchAction::RecvGradient(bwd_mb));
        }
        schedule.push(MicroBatchAction::Backward(bwd_mb));
        if !is_first {
            schedule.push(MicroBatchAction::SendGradient(bwd_mb));
        }
    }

    // Weight update after all micro-batches.
    schedule.push(MicroBatchAction::WeightUpdate);

    schedule
}

/// Compute the pipeline bubble fraction for 1F1B.
///
/// Returns a value in [0, 1] representing the fraction of time spent idle.
/// Lower is better.
pub fn bubble_fraction(num_stages: usize, num_micro_batches: usize) -> f64 {
    if num_micro_batches == 0 {
        return 1.0;
    }
    (num_stages as f64 - 1.0) / (num_micro_batches as f64 + num_stages as f64 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_counts_match() {
        let num_stages = 4;
        let num_micro_batches = 8;

        for stage in 0..num_stages {
            let schedule = schedule_1f1b(num_stages, num_micro_batches, stage);

            let fwd_count = schedule
                .iter()
                .filter(|a| matches!(a, MicroBatchAction::Forward(_)))
                .count();
            let bwd_count = schedule
                .iter()
                .filter(|a| matches!(a, MicroBatchAction::Backward(_)))
                .count();
            let wu_count = schedule
                .iter()
                .filter(|a| matches!(a, MicroBatchAction::WeightUpdate))
                .count();

            assert_eq!(fwd_count, num_micro_batches, "stage {stage} forward count");
            assert_eq!(bwd_count, num_micro_batches, "stage {stage} backward count");
            assert_eq!(wu_count, 1, "stage {stage} weight update count");
        }
    }

    #[test]
    fn first_stage_no_recv_activation() {
        let schedule = schedule_1f1b(3, 6, 0);
        assert!(
            !schedule.iter().any(|a| matches!(a, MicroBatchAction::RecvActivation(_))),
            "first stage should never RecvActivation"
        );
    }

    #[test]
    fn last_stage_no_send_activation() {
        let schedule = schedule_1f1b(3, 6, 2);
        assert!(
            !schedule.iter().any(|a| matches!(a, MicroBatchAction::SendActivation(_))),
            "last stage should never SendActivation"
        );
    }

    #[test]
    fn last_stage_no_recv_gradient() {
        let schedule = schedule_1f1b(3, 6, 2);
        assert!(
            !schedule.iter().any(|a| matches!(a, MicroBatchAction::RecvGradient(_))),
            "last stage should never RecvGradient"
        );
    }

    #[test]
    fn bubble_fraction_decreases_with_more_microbatches() {
        let bf4 = bubble_fraction(4, 4);
        let bf8 = bubble_fraction(4, 8);
        let bf16 = bubble_fraction(4, 16);
        assert!(bf4 > bf8);
        assert!(bf8 > bf16);
    }

    #[test]
    fn bubble_fraction_known_values() {
        // P=4, M=4: (4-1)/(4+4-1) = 3/7 ≈ 0.4286
        let bf = bubble_fraction(4, 4);
        assert!((bf - 3.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn forward_before_backward_for_same_microbatch() {
        for stage in 0..3 {
            let schedule = schedule_1f1b(3, 6, stage);
            for mb in 0..6 {
                let fwd_pos = schedule
                    .iter()
                    .position(|a| matches!(a, MicroBatchAction::Forward(id) if *id == mb));
                let bwd_pos = schedule
                    .iter()
                    .position(|a| matches!(a, MicroBatchAction::Backward(id) if *id == mb));

                if let (Some(f), Some(b)) = (fwd_pos, bwd_pos) {
                    assert!(
                        f < b,
                        "stage {stage}, mb {mb}: forward at {f} must come before backward at {b}"
                    );
                }
            }
        }
    }
}
