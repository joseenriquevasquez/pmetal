//! ANE/CPU overlap pipeline for training.
//!
//! Provides structured abstractions for overlapping ANE kernel dispatch
//! with CPU work (staging, weight transpose, gradient accumulation).
//!
//! Uses `std::thread::scope` for zero-allocation thread management —
//! no Arc, no channel, no heap allocation per overlap point.

use std::thread;

/// Coordinates ANE kernel dispatch with CPU work overlap.
///
/// Pattern: while ANE executes kernel N, CPU pre-stages data for kernel N+1
/// and/or computes weight gradients from kernel N-1.
///
/// This is a specialized two/three-stage pipeline for the ANE training hot
/// path where predictable, low-jitter overlap matters more than flexibility.
pub struct AnePipeline;

impl AnePipeline {
    /// Execute an ANE kernel while performing CPU work in parallel.
    ///
    /// `ane_work` runs on a spawned thread (ANE dispatch + wait).
    /// `cpu_work` runs on the calling thread (staging, transpose, dW GEMM).
    ///
    /// Returns when BOTH are complete.
    #[inline]
    pub fn overlap<A, C, Ra, Rc>(ane_work: A, cpu_work: C) -> (Ra, Rc)
    where
        A: FnOnce() -> Ra + Send,
        Ra: Send,
        C: FnOnce() -> Rc,
    {
        thread::scope(|s| {
            let ane_handle = s.spawn(ane_work);
            let cpu_result = cpu_work();
            let ane_result = ane_handle
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e));
            (ane_result, cpu_result)
        })
    }

    /// Execute an ANE kernel with TWO independent CPU tasks in parallel.
    ///
    /// Useful when backward has both a dW GEMM and a weight transpose that
    /// can run concurrently with ANE.
    #[inline]
    pub fn overlap_2<A, C1, C2, Ra, Rc1, Rc2>(
        ane_work: A,
        cpu_work_1: C1,
        cpu_work_2: C2,
    ) -> (Ra, Rc1, Rc2)
    where
        A: FnOnce() -> Ra + Send,
        C1: FnOnce() -> Rc1 + Send,
        C2: FnOnce() -> Rc2,
        Ra: Send,
        Rc1: Send,
    {
        thread::scope(|s| {
            let ane_handle = s.spawn(ane_work);
            let cpu1_handle = s.spawn(cpu_work_1);
            let cpu2_result = cpu_work_2();
            let ane_result = ane_handle
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e));
            let cpu1_result = cpu1_handle
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e));
            (ane_result, cpu1_result, cpu2_result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn overlap_runs_both() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();

        let (a, b) = AnePipeline::overlap(
            move || {
                flag2.store(true, Ordering::SeqCst);
                42
            },
            || 99,
        );

        assert_eq!(a, 42);
        assert_eq!(b, 99);
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn overlap_2_runs_all_three() {
        let (a, b, c) = AnePipeline::overlap_2(|| 1, || 2, || 3);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }
}
