//! D.2 — per-slot KV cache + batched decode driver.
//!
//! This module bridges the pure-Rust [`crate::continuous_batch`]
//! scheduler and the MLX model forward pass. It owns:
//!
//! - A [`SlotIdxMap`] that tracks which `SlotId` lives in which row of
//!   a [`BatchKVCache`].
//! - A [`ContinuousEngineState`] that bundles the batched KV cache
//!   with a per-slot [`Sampler`] so each slot has its own
//!   temperature / top-p / penalty state.
//! - A [`SlotForward`] trait that decouples the driver from
//!   `DynamicModel` so unit tests can stub the forward pass.
//! - Drivers [`drive_prefill_step`] and [`drive_decode_step`] that the
//!   engine's request pump (D.3) calls with the instructions produced
//!   by [`crate::continuous_batch::ContinuousBatcher::next_instruction`].
//!
//! # Today's batched-decode strategy
//!
//! The driver runs forward **once per slot** inside a single
//! `StreamContext`, schedules `async_eval` on the batch of returned
//! logit tensors, and only then samples. This gives us back-to-back
//! kernel dispatches on the GPU stream without per-slot host syncs.
//! It is *not* yet a single fused batched matmul — that's a future
//! upgrade that requires every architecture to accept a `[N, 1]` input
//! and batched cache, which we'll take as a separate change so it can
//! ship incrementally arch-by-arch.

use crate::continuous_batch::SlotId;
use pmetal_bridge::compat::{Array, Exception};
use pmetal_mlx::kv_cache::{BatchKVCache, FusedBatchKVCache, KVCache, KVCacheConfig};
use pmetal_models::generation::Sampler;
use std::collections::HashMap;

/// Trait that the driver uses to invoke the model's forward pass.
///
/// Kept generic so unit tests can inject a stub without instantiating
/// a full `DynamicModel`. Real call sites pass a closure wrapping
/// `DynamicModel::forward_with_hybrid_cache`.
pub trait SlotForward {
    /// Forward `tokens` (shape `[1, tokens.len()]`) through the model,
    /// extending `cache` with the produced K/V pairs. Returns the
    /// *lazy* logits tensor — the driver batches `async_eval` across
    /// all slots so the caller doesn't pay a host sync per slot.
    fn forward(&mut self, tokens: &[u32], cache: &mut KVCache) -> Result<Array, Exception>;
}

impl<F> SlotForward for F
where
    F: FnMut(&[u32], &mut KVCache) -> Result<Array, Exception>,
{
    fn forward(&mut self, tokens: &[u32], cache: &mut KVCache) -> Result<Array, Exception> {
        self(tokens, cache)
    }
}

/// Trait for fused `[N_active, 1]` batched decode.
///
/// Implementors run one forward per tick over all active slots, writing
/// into the shared [`FusedBatchKVCache`]. Input is `[N_active, 1]` int32
/// token ids; output is `[N_active, 1, vocab_size]` logits.
///
/// The driver picks this path only when the model reports
/// `supports_fused_batched = true` and the engine has allocated a
/// `FusedBatchKVCache`. Serial fallback via [`SlotForward`] is the
/// universal path for all other archs.
pub trait FusedBatchForward {
    fn forward_batched(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut FusedBatchKVCache,
    ) -> Result<Array, Exception>;
}

impl<F> FusedBatchForward for F
where
    F: FnMut(&Array, &[usize], &mut FusedBatchKVCache) -> Result<Array, Exception>,
{
    fn forward_batched(
        &mut self,
        input_ids: &Array,
        active_indices: &[usize],
        cache: &mut FusedBatchKVCache,
    ) -> Result<Array, Exception> {
        self(input_ids, active_indices, cache)
    }
}

/// Bidirectional `SlotId ↔ batch_idx` map.
///
/// The batched KV cache indexes slots by compact integer positions
/// (`batch_idx ∈ [0, max_slots)`), while the scheduler identifies them
/// by opaque `SlotId`s. This map owns the allocation of batch rows,
/// re-using freed rows for new slots to keep the cache compact.
#[derive(Debug, Default)]
pub struct SlotIdxMap {
    by_slot: HashMap<SlotId, usize>,
    by_idx: Vec<Option<SlotId>>,
}

impl SlotIdxMap {
    /// Create a map sized for `capacity` batch rows, all initially free.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            by_slot: HashMap::with_capacity(capacity),
            by_idx: vec![None; capacity],
        }
    }

    /// Return `Some(existing_idx)` if `slot` is already mapped, else
    /// allocate the lowest-numbered free row and return its index.
    /// Returns `None` when every row is occupied.
    pub fn allocate(&mut self, slot: SlotId) -> Option<usize> {
        if let Some(&idx) = self.by_slot.get(&slot) {
            return Some(idx);
        }
        let free = self.by_idx.iter().position(|s| s.is_none())?;
        self.by_idx[free] = Some(slot);
        self.by_slot.insert(slot, free);
        Some(free)
    }

    /// Look up the batch row for a slot, if allocated.
    pub fn get(&self, slot: SlotId) -> Option<usize> {
        self.by_slot.get(&slot).copied()
    }

    /// Free the row held by `slot`, returning its index so the caller
    /// can reset the underlying cache entry.
    pub fn release(&mut self, slot: SlotId) -> Option<usize> {
        let idx = self.by_slot.remove(&slot)?;
        self.by_idx[idx] = None;
        Some(idx)
    }

    /// Number of allocated rows.
    pub fn len(&self) -> usize {
        self.by_slot.len()
    }

    /// Whether any rows are allocated.
    pub fn is_empty(&self) -> bool {
        self.by_slot.is_empty()
    }

    /// Iterate `(SlotId, batch_idx)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (SlotId, usize)> + '_ {
        self.by_slot.iter().map(|(&s, &i)| (s, i))
    }
}

/// Continuous-batch engine state.
///
/// Holds the per-slot KV caches, per-slot samplers, and slot→idx map.
/// Samplers are stored as `Option` so rows can sit "allocated but
/// empty" briefly between release and the next allocate.
pub struct ContinuousEngineState {
    caches: BatchKVCache,
    /// Fused per-layer KV cache used when the model supports fused
    /// batched decode. When `None`, the driver falls back to the
    /// per-slot `caches` path.
    fused_cache: Option<FusedBatchKVCache>,
    samplers: Vec<Option<Sampler>>,
    map: SlotIdxMap,
}

impl ContinuousEngineState {
    /// Create engine state for `max_slots` concurrent slots, each with
    /// an empty KV cache configured by `cache_config`.
    pub fn new(max_slots: usize, cache_config: KVCacheConfig) -> Self {
        let caches = BatchKVCache::new(max_slots, cache_config);
        let samplers = (0..max_slots).map(|_| None).collect();
        let map = SlotIdxMap::with_capacity(max_slots);
        Self {
            caches,
            fused_cache: None,
            samplers,
            map,
        }
    }

    /// Create engine state that also owns a [`FusedBatchKVCache`] for
    /// models that implement fused batched decode. The per-slot
    /// `caches` path is still allocated so prefill (serial) and
    /// fallback archs can share the same state.
    pub fn new_with_fused_cache(
        max_slots: usize,
        cache_config: KVCacheConfig,
        fused_cache: FusedBatchKVCache,
    ) -> Self {
        let mut state = Self::new(max_slots, cache_config);
        state.fused_cache = Some(fused_cache);
        state
    }

    /// Whether this state has a fused KV cache wired in.
    pub fn has_fused_cache(&self) -> bool {
        self.fused_cache.is_some()
    }

    /// Borrow the fused cache mutably if present.
    pub fn fused_cache_mut(&mut self) -> Option<&mut FusedBatchKVCache> {
        self.fused_cache.as_mut()
    }

    /// Total number of rows (active + free).
    pub fn capacity(&self) -> usize {
        self.caches.batch_size()
    }

    /// Number of active slots.
    pub fn active(&self) -> usize {
        self.map.len()
    }

    /// Assign a slot to a free batch row, attaching its sampler.
    ///
    /// Returns the allocated `batch_idx`, or an error if all rows are
    /// in use. Callers must check [`Self::active`] or
    /// [`crate::continuous_batch::BatcherConfig::max_slots`] before
    /// enqueueing to the scheduler; hitting this error indicates a
    /// bookkeeping bug.
    pub fn admit(&mut self, slot: SlotId, sampler: Sampler) -> Result<usize, Exception> {
        let idx = self.map.allocate(slot).ok_or_else(|| {
            Exception::custom(format!(
                "ContinuousEngineState::admit: no free batch row for {:?} (capacity {})",
                slot,
                self.capacity()
            ))
        })?;
        self.samplers[idx] = Some(sampler);
        if let Some(fused) = self.fused_cache.as_mut() {
            fused.admit(idx)?;
        }
        Ok(idx)
    }

    /// Release a slot's row, resetting the cache entry so the next
    /// admit starts with a clean state.
    pub fn retire(&mut self, slot: SlotId) {
        if let Some(idx) = self.map.release(slot) {
            self.samplers[idx] = None;
            self.caches.reset_indices(&[idx]);
            if let Some(fused) = self.fused_cache.as_mut() {
                fused.release(idx);
            }
        }
    }

    /// Look up the batch row for a slot.
    pub fn batch_idx(&self, slot: SlotId) -> Option<usize> {
        self.map.get(slot)
    }

    /// Borrow the per-slot KV cache mutably.
    pub fn cache_for(&mut self, slot: SlotId) -> Option<&mut KVCache> {
        let idx = self.map.get(slot)?;
        self.caches.get_mut(idx)
    }

    /// Borrow the per-slot sampler mutably.
    pub fn sampler_for(&mut self, slot: SlotId) -> Option<&mut Sampler> {
        let idx = self.map.get(slot)?;
        self.samplers.get_mut(idx).and_then(|s| s.as_mut())
    }
}

/// Outcome of a prefill step: `Some(last_position_logits)` on the final
/// chunk, `None` while there are more chunks to process.
///
/// Returned logits are already evaluated (not lazy), so the caller can
/// hand them straight to the slot's sampler.
pub fn drive_prefill_step<F: SlotForward>(
    forward: &mut F,
    state: &mut ContinuousEngineState,
    slot: SlotId,
    chunk: &[u32],
    final_chunk: bool,
) -> Result<Option<Array>, Exception> {
    let cache = state
        .cache_for(slot)
        .ok_or_else(|| Exception::custom(format!("prefill: slot {slot:?} not admitted")))?;
    let logits = forward.forward(chunk, cache)?;

    if final_chunk {
        let last = extract_last_logits(&logits)?;
        Ok(Some(last))
    } else {
        // Non-final chunk: we don't need the logits at all, but we
        // must evaluate the forward to materialize the cache update.
        logits
            .try_eval()
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(None)
    }
}

/// Per-slot decode step output.
#[derive(Debug, Clone)]
pub struct SlotStepOutput {
    pub slot: SlotId,
    pub token: u32,
    pub logits: Array,
}

/// Run a batched decode step across the given
/// `(slot, current_token, history)` tuples.
///
/// `history` is the per-slot full-context window (prompt + generated so
/// far) used by the sampler for repetition / frequency / presence
/// penalties. Penalties are free when the slot's `GenerationConfig` has
/// them disabled — [`Sampler::sample_array_with_penalties`] no-ops in
/// that case.
///
/// The driver:
/// 1. Schedules `forward` for each slot (one token per slot) without
///    evaluating — the returned logits are lazy tensors.
/// 2. Samples the next token for each slot via its own sampler, passing
///    the slot's history so penalties apply.
/// 3. Calls `async_eval` on the batched sample tensors so the GPU can
///    fuse the sampling kernels together.
/// 4. Extracts `u32` tokens from each sampled tensor (one host sync
///    per slot — unavoidable without a single batched sample path).
/// 5. Calls [`Sampler::update_counts`] per slot so frequency penalty
///    state advances with the slot's own token stream.
///
/// All forwards run on the caller's current `StreamContext`.
pub fn drive_decode_step<F: SlotForward>(
    forward: &mut F,
    state: &mut ContinuousEngineState,
    slots: &[(SlotId, u32, &[u32])],
) -> Result<Vec<SlotStepOutput>, Exception> {
    use pmetal_bridge::compat::ops::async_eval;

    // Phase 1: schedule forwards, collect lazy logits per slot.
    let mut lazy: Vec<(SlotId, Array)> = Vec::with_capacity(slots.len());
    for &(slot, token, _) in slots {
        let cache = state
            .cache_for(slot)
            .ok_or_else(|| Exception::custom(format!("decode: slot {slot:?} not admitted")))?;
        let tokens = [token];
        let logits = forward.forward(&tokens, cache)?;
        lazy.push((slot, logits));
    }

    // Phase 2: sample per slot (penalty-aware), also lazy. Keep RAW
    // last-position logits around so the caller can compute per-slot
    // logprobs later.
    let mut samples: Vec<(SlotId, Array, Array)> = Vec::with_capacity(slots.len()); // (slot, y, last_logits)
    for ((slot, full_logits), &(_, _, history)) in lazy.into_iter().zip(slots.iter()) {
        let last = extract_last_logits(&full_logits)?;
        let sampler = state
            .sampler_for(slot)
            .ok_or_else(|| Exception::custom(format!("sample: slot {slot:?} has no sampler")))?;
        let (y, _filtered) = sampler.sample_array_with_penalties(&last, history)?;
        samples.push((slot, y, last));
    }

    // Phase 3: async-eval every (y, last_logits) pair so the GPU
    // schedules them back-to-back.
    {
        let arrays: Vec<&Array> = samples
            .iter()
            .flat_map(|(_, y, last)| [y, last].into_iter())
            .collect();
        async_eval(arrays);
    }

    // Phase 4: per-slot host sync + token extraction + frequency-count
    // update so the next tick's penalties see the full history.
    let mut out = Vec::with_capacity(samples.len());
    for (slot, y, last) in samples {
        let token = y.item::<u32>();
        if let Some(sampler) = state.sampler_for(slot) {
            sampler.update_counts(token);
        }
        out.push(SlotStepOutput {
            slot,
            token,
            logits: last,
        });
    }
    Ok(out)
}

/// Fused variant of [`drive_decode_step`].
///
/// Runs **one** `[N_active, 1]` forward across every active slot against
/// the shared [`FusedBatchKVCache`] held by `state`, then samples per
/// slot (penalty-aware, logprob-capable via the raw last-position logits
/// returned alongside each sampled token).
///
/// Prerequisites:
/// - `state.has_fused_cache()` must be true.
/// - The model powering `forward` must implement [`FusedBatchForward`]
///   (usually a thin closure over `DynamicModel::forward_batched`).
///
/// The `slots` tuple format mirrors [`drive_decode_step`] so the pump
/// can hand the same inputs to either path.
pub fn drive_fused_decode_step<F: FusedBatchForward>(
    forward: &mut F,
    state: &mut ContinuousEngineState,
    slots: &[(SlotId, u32, &[u32])],
) -> Result<Vec<SlotStepOutput>, Exception> {
    use pmetal_bridge::compat::ops::async_eval;

    if slots.is_empty() {
        return Ok(Vec::new());
    }

    // Resolve batch indices in the same order the caller passed.
    let mut active_indices: Vec<usize> = Vec::with_capacity(slots.len());
    let mut input_tokens: Vec<i32> = Vec::with_capacity(slots.len());
    for &(slot, token, _) in slots {
        let idx = state.batch_idx(slot).ok_or_else(|| {
            Exception::custom(format!("fused decode: slot {slot:?} not admitted"))
        })?;
        active_indices.push(idx);
        input_tokens.push(token as i32);
    }

    let n = slots.len() as i32;
    let input_ids = Array::from_i32_slice(&input_tokens).reshape(&[n, 1]);

    let fused = state.fused_cache.as_mut().ok_or_else(|| {
        Exception::custom("fused decode: ContinuousEngineState has no fused cache")
    })?;

    let logits = forward.forward_batched(&input_ids, &active_indices, fused)?;

    // Extract per-slot last-position logits then sample.
    let mut samples: Vec<(SlotId, Array, Array)> = Vec::with_capacity(slots.len());
    for (row, &(slot, _, history)) in slots.iter().enumerate() {
        let idx = Array::from_i32_slice(&[row as i32]);
        // logits: [N, 1, V] → take row → [1, 1, V] → squeeze axis 0 → [1, V]
        let row_logits = logits.take_axis(&idx, 0);
        let last = extract_last_logits(&row_logits)?;
        let sampler = state
            .sampler_for(slot)
            .ok_or_else(|| Exception::custom(format!("sample: slot {slot:?} has no sampler")))?;
        let (y, _filtered) = sampler.sample_array_with_penalties(&last, history)?;
        samples.push((slot, y, last));
    }

    {
        let arrays: Vec<&Array> = samples
            .iter()
            .flat_map(|(_, y, last)| [y, last].into_iter())
            .collect();
        async_eval(arrays);
    }

    let mut out = Vec::with_capacity(samples.len());
    for (slot, y, last) in samples {
        let token = y.item::<u32>();
        if let Some(sampler) = state.sampler_for(slot) {
            sampler.update_counts(token);
        }
        out.push(SlotStepOutput {
            slot,
            token,
            logits: last,
        });
    }
    Ok(out)
}

/// Extract the last-position logits from a `[B, S, V]` tensor as a
/// `[B, V]` slice.
fn extract_last_logits(logits: &Array) -> Result<Array, Exception> {
    let shape = logits.shape();
    if shape.len() < 2 {
        return Err(Exception::custom(format!(
            "extract_last_logits: expected rank >= 2, got shape {shape:?}"
        )));
    }
    let seq_axis = shape.len() - 2;
    let seq_len = shape[seq_axis];
    if seq_len <= 0 {
        return Err(Exception::custom(format!(
            "extract_last_logits: empty sequence axis in {shape:?}"
        )));
    }
    let idx = Array::from_i32_slice(&[seq_len - 1]);
    Ok(logits
        .take_axis(&idx, seq_axis as i32)
        .squeeze_axes(&[seq_axis as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::Array;

    fn make_config() -> KVCacheConfig {
        KVCacheConfig::new(2, 32, 4, 64)
    }

    #[test]
    fn slot_idx_map_allocates_lowest_free_row() {
        let mut m = SlotIdxMap::with_capacity(3);
        assert_eq!(m.allocate(SlotId(1)), Some(0));
        assert_eq!(m.allocate(SlotId(2)), Some(1));
        assert_eq!(m.allocate(SlotId(3)), Some(2));
        // Full.
        assert_eq!(m.allocate(SlotId(4)), None);

        // Free middle row.
        assert_eq!(m.release(SlotId(2)), Some(1));
        assert_eq!(m.allocate(SlotId(4)), Some(1));
    }

    #[test]
    fn slot_idx_map_idempotent_allocate() {
        let mut m = SlotIdxMap::with_capacity(2);
        assert_eq!(m.allocate(SlotId(1)), Some(0));
        // Same slot — same row, no new allocation.
        assert_eq!(m.allocate(SlotId(1)), Some(0));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn continuous_state_admit_retire_roundtrip() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let sampler = Sampler::new(pmetal_models::generation::GenerationConfig::default());
        let idx = state.admit(SlotId(1), sampler).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(state.active(), 1);
        assert_eq!(state.batch_idx(SlotId(1)), Some(0));

        state.retire(SlotId(1));
        assert_eq!(state.active(), 0);
        assert_eq!(state.batch_idx(SlotId(1)), None);
    }

    /// Stub forward that mirrors the real surface: takes tokens,
    /// updates the cache with a dummy K/V of the right shape, returns
    /// a `[1, S, V]` logits tensor whose last-position argmax lets us
    /// verify per-slot routing.
    struct StubForward {
        vocab: i32,
        /// Per-call logits to return, keyed by the first input token.
        /// The caller precomputes which token each slot should sample
        /// next and stashes it here.
        next_token_per_input: HashMap<u32, u32>,
        calls: usize,
    }

    impl SlotForward for StubForward {
        fn forward(&mut self, tokens: &[u32], cache: &mut KVCache) -> Result<Array, Exception> {
            self.calls += 1;
            // Extend the cache a tiny bit so seq_len advances — mirrors
            // the real forward's side effect.
            let (num_heads, head_dim) = {
                let cfg = cache.config();
                (cfg.num_kv_heads as i32, cfg.head_dim as i32)
            };
            let s = tokens.len() as i32;
            let k = Array::zeros_f32(&[1, num_heads, s, head_dim]);
            let v = Array::zeros_f32(&[1, num_heads, s, head_dim]);
            cache.update_and_fetch(0, &k, &v)?;
            cache.update_and_fetch(1, &k, &v)?;

            // Build logits [1, S, V] where the last position scores
            // `next_token_per_input[tokens[0]]` at +10 and 0 elsewhere.
            let mut data = vec![0.0f32; (s * self.vocab) as usize];
            let next = self
                .next_token_per_input
                .get(&tokens[0])
                .copied()
                .unwrap_or(0);
            let last_row_start = ((s - 1) * self.vocab) as usize;
            data[last_row_start + next as usize] = 10.0;
            Ok(Array::from_f32_slice(&data, &[1, s, self.vocab]))
        }
    }

    #[test]
    fn decode_step_routes_per_slot_logits_correctly() {
        let mut state = ContinuousEngineState::new(3, make_config());
        let s1 = SlotId(1);
        let s2 = SlotId(2);
        state
            .admit(
                s1,
                Sampler::new(pmetal_models::generation::GenerationConfig::default()),
            )
            .unwrap();
        state
            .admit(
                s2,
                Sampler::new(pmetal_models::generation::GenerationConfig::default()),
            )
            .unwrap();

        let mut next = HashMap::new();
        next.insert(100u32, 7u32); // slot 1 fed 100 → samples 7
        next.insert(200u32, 13u32); // slot 2 fed 200 → samples 13
        let mut stub = StubForward {
            vocab: 32,
            next_token_per_input: next,
            calls: 0,
        };

        let h1: Vec<u32> = vec![];
        let h2: Vec<u32> = vec![];
        let outs = drive_decode_step(
            &mut stub,
            &mut state,
            &[(s1, 100, h1.as_slice()), (s2, 200, h2.as_slice())],
        )
        .unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(stub.calls, 2, "one forward per slot");

        // Each slot must receive its own sampled token.
        let a = outs.iter().find(|o| o.slot == s1).unwrap().token;
        let b = outs.iter().find(|o| o.slot == s2).unwrap().token;
        assert_eq!(a, 7);
        assert_eq!(b, 13);
    }

    #[test]
    fn prefill_step_non_final_returns_none() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let s = SlotId(10);
        state
            .admit(
                s,
                Sampler::new(pmetal_models::generation::GenerationConfig::default()),
            )
            .unwrap();

        let mut next = HashMap::new();
        next.insert(1u32, 0u32);
        let mut stub = StubForward {
            vocab: 8,
            next_token_per_input: next,
            calls: 0,
        };

        let out = drive_prefill_step(&mut stub, &mut state, s, &[1, 2, 3], false).unwrap();
        assert!(out.is_none());
        assert_eq!(stub.calls, 1);
    }

    #[test]
    fn prefill_step_final_returns_last_position_logits() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let s = SlotId(20);
        state
            .admit(
                s,
                Sampler::new(pmetal_models::generation::GenerationConfig::default()),
            )
            .unwrap();

        let mut next = HashMap::new();
        next.insert(1u32, 0u32);
        let mut stub = StubForward {
            vocab: 8,
            next_token_per_input: next,
            calls: 0,
        };

        let out = drive_prefill_step(&mut stub, &mut state, s, &[1, 2, 3], true).unwrap();
        let logits = out.expect("final chunk must return logits");
        // Shape should be [1, V] after squeezing the 1-length seq axis.
        assert_eq!(logits.shape(), &[1, 8]);
    }

    #[test]
    fn decode_step_errors_for_unadmitted_slot() {
        let mut state = ContinuousEngineState::new(2, make_config());
        let mut stub = StubForward {
            vocab: 8,
            next_token_per_input: HashMap::new(),
            calls: 0,
        };
        let empty: &[u32] = &[];
        let err = drive_decode_step(&mut stub, &mut state, &[(SlotId(99), 1, empty)]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not admitted"), "unexpected error: {msg}");
    }
}
