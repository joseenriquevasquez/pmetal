//! RAII step profiler for ANE training instrumentation.
//!
//! Provides [`Stopwatch`] for zero-overhead timing accumulation and
//! [`StepProfile`] for structured breakdown of training step costs.
//!
//! All times are in microseconds. Feeds into `tracing::info!` spans
//! and the `StepTimings` struct consumed by `AneTrainingLoop`.

use std::time::Instant;

/// RAII timer that accumulates elapsed microseconds into a target counter.
///
/// Created via [`Stopwatch::start`], automatically records elapsed time on drop.
/// Multiple stopwatches can accumulate into the same counter across a function.
///
/// # Example
///
/// ```ignore
/// let mut total_us = 0u64;
/// {
///     let _t = Stopwatch::start(&mut total_us);
///     expensive_operation();
/// } // elapsed time added to total_us here
/// ```
pub struct Stopwatch<'a> {
    target: &'a mut u64,
    start: Instant,
}

impl<'a> Stopwatch<'a> {
    /// Begin timing, accumulating into `target` on drop.
    #[inline]
    pub fn start(target: &'a mut u64) -> Self {
        Self {
            target,
            start: Instant::now(),
        }
    }
}

impl Drop for Stopwatch<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.target += self.start.elapsed().as_micros() as u64;
    }
}

/// Percentage breakdown of a training step by category.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepBreakdown {
    /// IOSurface write staging (forward + backward).
    pub io_write_pct: f64,
    /// IOSurface readback (forward + backward).
    pub io_read_pct: f64,
    /// ANE forward kernel dispatch.
    pub ane_fwd_pct: f64,
    /// ANE backward kernel dispatch.
    pub ane_bwd_pct: f64,
    /// CPU RMSNorm forward/backward.
    pub rmsnorm_pct: f64,
    /// Weight gradient GEMMs (GPU or CPU cblas).
    pub dw_gemm_pct: f64,
    /// Adam optimizer update.
    pub adam_pct: f64,
    /// Unaccounted overhead (allocation, scheduling, etc.).
    pub overhead_pct: f64,
}

/// Compute a percentage breakdown from raw microsecond counters.
///
/// Takes the 7 category counters and total, returns percentages.
/// Overhead is the residual between sum-of-categories and total.
#[allow(clippy::too_many_arguments)]
pub fn breakdown(
    io_write_us: u64,
    io_read_us: u64,
    ane_fwd_us: u64,
    ane_bwd_us: u64,
    rmsnorm_us: u64,
    dw_gemm_us: u64,
    adam_us: u64,
    total_us: u64,
) -> StepBreakdown {
    let total = total_us.max(1) as f64;
    let accounted =
        (io_write_us + io_read_us + ane_fwd_us + ane_bwd_us + rmsnorm_us + dw_gemm_us + adam_us)
            as f64;

    StepBreakdown {
        io_write_pct: io_write_us as f64 / total * 100.0,
        io_read_pct: io_read_us as f64 / total * 100.0,
        ane_fwd_pct: ane_fwd_us as f64 / total * 100.0,
        ane_bwd_pct: ane_bwd_us as f64 / total * 100.0,
        rmsnorm_pct: rmsnorm_us as f64 / total * 100.0,
        dw_gemm_pct: dw_gemm_us as f64 / total * 100.0,
        adam_pct: adam_us as f64 / total * 100.0,
        overhead_pct: ((total - accounted) / total * 100.0).max(0.0),
    }
}

/// Log a step timing breakdown at `tracing::info` level.
pub fn log_breakdown(step: usize, total_us: u64, b: &StepBreakdown) {
    tracing::info!(
        step,
        total_ms = total_us as f64 / 1000.0,
        "ANE step: io_w {:.0}% io_r {:.0}% ane_f {:.0}% ane_b {:.0}% \
         rms {:.0}% dw {:.0}% adam {:.0}% overhead {:.0}%",
        b.io_write_pct,
        b.io_read_pct,
        b.ane_fwd_pct,
        b.ane_bwd_pct,
        b.rmsnorm_pct,
        b.dw_gemm_pct,
        b.adam_pct,
        b.overhead_pct,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwatch_accumulates() {
        let mut counter = 0u64;
        {
            let _t = Stopwatch::start(&mut counter);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(counter > 0, "expected at least 1us, got {counter}us");
    }

    #[test]
    fn breakdown_sums_to_100() {
        let b = breakdown(100, 200, 300, 400, 50, 150, 100, 1500);
        let sum = b.io_write_pct
            + b.io_read_pct
            + b.ane_fwd_pct
            + b.ane_bwd_pct
            + b.rmsnorm_pct
            + b.dw_gemm_pct
            + b.adam_pct
            + b.overhead_pct;
        assert!((sum - 100.0).abs() < 0.01, "expected 100%, got {sum}%");
    }
}
