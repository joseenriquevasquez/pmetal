#![allow(unsafe_code)]

//! Benchmark utilities for Metal 3 vs Metal 4 kernel comparison.
//!
//! Provides infrastructure for comparative profiling of legacy and MPP kernels
//! on the same workloads.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    buffer::{BufferUsage, MetalBuffer},
    context::MetalContext,
    error::Result,
};

/// Result of a single kernel benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Kernel name.
    pub name: String,
    /// Minimum time across iterations.
    pub min_time: Duration,
    /// Median time across iterations.
    pub median_time: Duration,
    /// Mean time across iterations.
    pub mean_time: Duration,
    /// Number of floating-point operations (for TFLOPS calculation).
    pub flops: u64,
    /// Achieved TFLOPS (based on median time).
    pub tflops: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:40} min={:>8.2}ms  med={:>8.2}ms  mean={:>8.2}ms  {:.2} TFLOPS",
            self.name,
            self.min_time.as_secs_f64() * 1000.0,
            self.median_time.as_secs_f64() * 1000.0,
            self.mean_time.as_secs_f64() * 1000.0,
            self.tflops,
        )
    }
}

/// Benchmark a GPU operation by running it multiple times and measuring wall-clock time.
///
/// Includes warmup iterations to prime caches and pipeline states.
pub fn bench_gpu_op<F>(
    name: &str,
    warmup: usize,
    iterations: usize,
    flops: u64,
    mut op: F,
) -> BenchResult
where
    F: FnMut() -> Result<()>,
{
    // Warmup
    for _ in 0..warmup {
        let _ = op();
    }

    // Timed iterations
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = op();
        times.push(start.elapsed());
    }

    times.sort();
    let min_time = times[0];
    let median_time = times[times.len() / 2];
    let mean_time = Duration::from_secs_f64(
        times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64,
    );
    let tflops = if median_time.as_secs_f64() > 0.0 {
        flops as f64 / median_time.as_secs_f64() / 1e12
    } else {
        0.0
    };

    BenchResult {
        name: name.to_string(),
        min_time,
        median_time,
        mean_time,
        flops,
        tflops,
    }
}

/// Run a comparative benchmark between Metal 3 and Metal 4 implementations.
///
/// Returns a pair of (metal3_result, metal4_result) for comparison.
pub fn bench_comparative<F3, F4>(
    kernel_name: &str,
    m: usize,
    n: usize,
    k: usize,
    warmup: usize,
    iterations: usize,
    metal3_op: F3,
    metal4_op: Option<F4>,
) -> (BenchResult, Option<BenchResult>)
where
    F3: FnMut() -> Result<()>,
    F4: FnMut() -> Result<()>,
{
    // GEMM FLOPS: 2 * M * N * K (multiply + add per output element)
    let flops = 2 * m as u64 * n as u64 * k as u64;

    let metal3 = bench_gpu_op(
        &format!("{} (Metal 3)", kernel_name),
        warmup,
        iterations,
        flops,
        metal3_op,
    );

    let metal4 = metal4_op.map(|op| {
        bench_gpu_op(
            &format!("{} (Metal 4/MPP)", kernel_name),
            warmup,
            iterations,
            flops,
            op,
        )
    });

    (metal3, metal4)
}

/// Print a formatted benchmark comparison report.
pub fn print_comparison(metal3: &BenchResult, metal4: &Option<BenchResult>) {
    println!("{}", metal3);
    if let Some(m4) = metal4 {
        println!("{}", m4);
        let speedup = metal3.median_time.as_secs_f64() / m4.median_time.as_secs_f64();
        println!(
            "{:40} {:.2}x speedup (Metal 4 vs Metal 3)",
            "", speedup
        );
    } else {
        println!(
            "{:40} Metal 4/MPP not available on this device",
            ""
        );
    }
    println!();
}

/// Standard GEMM benchmark configuration.
#[derive(Debug, Clone)]
pub struct GemmBenchConfig {
    /// Problem sizes to benchmark: (M, N, K).
    pub sizes: Vec<(usize, usize, usize)>,
    /// Number of warmup iterations.
    pub warmup: usize,
    /// Number of timed iterations.
    pub iterations: usize,
}

impl Default for GemmBenchConfig {
    fn default() -> Self {
        Self {
            sizes: vec![
                // Decode (M=1): memory-bound
                (1, 4096, 4096),
                (1, 14336, 4096),
                // Small batch: transition region
                (8, 4096, 4096),
                (8, 14336, 4096),
                // Prefill / training: compute-bound
                (64, 4096, 4096),
                (64, 14336, 4096),
                (256, 4096, 4096),
                (512, 4096, 4096),
                // Large GEMM: peak throughput
                (1024, 4096, 4096),
                (2048, 4096, 4096),
            ],
            warmup: 5,
            iterations: 20,
        }
    }
}

/// Allocate random f32 buffer for benchmarking.
pub fn alloc_random_f32(ctx: &Arc<MetalContext>, size: usize) -> Result<MetalBuffer<f32>> {
    let mut data = vec![0.0f32; size];
    // Simple pseudo-random initialization
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i as f32 * 0.618034) % 2.0) - 1.0; // golden ratio hash
    }
    MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)
}
