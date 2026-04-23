# PMetal System Audit

Date: 2026-04-21

Scope: bottom-up dependency audit of the workspace, starting from the leaf crates and moving upward.

## Audit Status

- Completed: `pmetal-core`, `pmetal-gguf`, `pmetal-bridge`, `pmetal-data`, `pmetal-hub`, `pmetal-metal`, `pmetal-mlx`, `pmetal-models`, `pmetal-trainer`, `pmetal-lora`, `pmetal-distill`
- In progress: `pmetal-merge`, `pmetal-distributed`, `pmetal-serve`, `pmetal-py`, `pmetal`
- Next: remaining top-level surfaces and integration paths

## Findings

### Leaf Layer

#### P0

1. `crates/pmetal-bridge/src/compile.rs:141-158`

CompiledFn trampoline can write past the caller-allocated output buffer.

`n_outputs_max` is only documented, not enforced. If the Rust closure returns more arrays than the C++ side reserved, the trampoline blindly advances `outputs.add(i)` and placement-copies past the end of the buffer, which is immediate memory corruption at the FFI boundary.

#### P1

2. `crates/pmetal-bridge/src/compile.rs:148-153`

Placeholder output arrays are leaked on every compiled call.

The code intends to destroy the preinitialized empty output before overwriting it, but `drop_in_place(slot as *mut InlineArray as *mut ())` drops `()`, not `InlineArray`. That means the old `RawBuf` destructor never runs, so every written output slot leaks its placeholder allocation.

3. `crates/pmetal-gguf/src/writer.rs:425-430`

GGUF writer emits dimensions in the opposite order from the reader.

`GgufContent::read` explicitly reverses tensor dimensions because GGUF stores them reversed, but `write_tensor_info` serializes `info.dimensions` as-is. Any tensor written here will round-trip with its axes reversed, which corrupts shapes for non-1D tensors.

4. `crates/pmetal-bridge/src/compat/nn.rs:155-167`

`value_and_grad` masks user errors as NaN tensors.

When the user-supplied loss function returns `Err`, this wrapper converts it into `Array::from_f32(NaN)` and still returns `Ok((loss, grads))`. That hides the real failure, lets training continue with poisoned gradients, and makes downstream failures much harder to diagnose. The same pattern is repeated in `keyed_value_and_grad` below.

5. `crates/pmetal-bridge/src/qwen3_native/generate.rs:565-570`

`CppForwardState` is unsafely marked `Sync` despite holding mutable raw pointers.

The comment says this state is single-threaded, but the type is marked both `Send` and `Sync` while containing `*mut RawBuf` cache pointers and mutable scalar state mirrored back into `NativeCache`. `Sync` permits sharing `&CppForwardState` across threads without synchronization, which makes the raw-pointer invariants unsound.

#### P2

6. `crates/pmetal-gguf/src/reader.rs:173-197`

Reader claims to accept big-endian GGUF but still parses everything as little-endian.

The magic check accepts the big-endian `GGUF` sentinel, but the parser immediately continues with `read_u32::<LittleEndian>` / `read_u64::<LittleEndian>` for version, counts, metadata, and tensor headers. Real big-endian GGUF v3 files will be rejected or misparsed even though this branch advertises support.

7. `crates/pmetal-gguf/src/writer.rs:64-70`

`GgufBuilder` accepts zero alignment and later divides by zero.

`alignment()` stores any `u32`, including `0`, but `write()` ultimately calls `align_offset(offset, self.alignment as u64)`, and `align_offset` performs `offset % alignment`. A caller can therefore trigger a panic just by setting `.alignment(0)`.

### Mid Layer

#### P1

8. `crates/pmetal-data/src/dataloader.rs:173-185`

Distributed `DataLoader` batch counts are computed from the unsharded dataset.

`DataLoader::new` and `reset` shard `self.indices` when `world_size > 1`, but `num_batches()` and `len()` still use `self.dataset.len()` instead of `self.indices.len()`. On distributed runs, every worker therefore overreports its sample count and batch count, which propagates directly into trainer step scheduling via `pmetal-trainer`.

9. `crates/pmetal-hub/src/cache.rs:113-117`

Dataset cache lookup searches the model cache root instead of the datasets cache root.

`find_cached_dataset` builds `datasets--...` paths under `cache_dir()`, even though `datasets_cache_dir()` exists specifically for the Hugging Face datasets cache layout. Cached datasets will typically be missed, forcing unnecessary re-downloads and defeating the local-only fast path.

10. `crates/pmetal-mlx/src/offloading.rs:907-935`

Disk offloading destroys tensor shape and dtype on round-trip.

`save_array_to_disk` serializes only raw f32 values, while `load_array_from_disk` ignores the requested dtype and reconstructs a flat 1-D f32 array. Any embedding, activation, or gradient offloaded to disk comes back flattened and type-erased, which corrupts `OffloadedEmbedding::lookup`, activation reloads, and gradient accumulation. The `unwrap_or_default()` on `to_f32_vec` also means unsupported conversions silently write empty payloads instead of failing.

11. `crates/pmetal-mlx/src/kv_cache/paged.rs:495-505`

Paged KV cache ignores its configured dtype and always allocates float32 blocks.

`PagedKVCacheConfig::dtype` is used for memory accounting, but `ensure_block_allocated` hard-codes `Dtype::Float32` for both key and value storage. Float16/BFloat16 paged caches therefore consume the wrong amount of memory and return cached tensors in a different dtype than the caller configured.

12. `crates/pmetal-trainer/src/training_loop/mod.rs:1494-1541`

Async checkpointing drops the previous in-flight `JoinHandle` when a new checkpoint starts.

`TrainingLoop::spawn_async_checkpoint()` calls `poll_pending_checkpoint()`, but if the previous checkpoint thread is still running that method simply puts the old `JoinHandle` back into `self.pending_checkpoint`. `spawn_async_checkpoint()` then immediately spawns a new thread and overwrites `self.pending_checkpoint = Some(handle)`, dropping the still-running handle anyway. In practice that detaches earlier checkpoint threads, loses any write error or panic reporting for them, and means `Drop` only waits for the most recent checkpoint while older writes may still be mutating files in the background.

13. `crates/pmetal-models/src/architectures/llama4.rs:1070-1082`

`Llama4` accepts a KV cache but ignores it, forcing uncached full-prefix decode.

`Llama4ForCausalLM::forward_with_cache()` accepts `Option<&mut KVCache>` but explicitly ignores it and falls back to `self.forward(input_ids, mask, None)`. This is not a dead compatibility shim: the model dispatcher forwards cache-bearing calls into this method, and higher layers such as generation, benchmarking, and GRPO decoding do pass populated KV caches for incremental decode. For `Llama4`, those call sites therefore recompute the full prefix every step instead of appending to the cache.

14. `crates/pmetal-merge/src/merge.rs:236-315`

Standard model merging silently changes the contributor set on a tensor-by-tensor basis.

`collect_tensor_names()` builds the union of every tensor name across all input models, and `merge_tensor()` then merges each tensor across only the subset of loaders that happen to contain that name. There is no validation that every source model contributes to every merged tensor, and no opt-in “allow missing tensors” mode. A partial or architecture-mismatched checkpoint therefore yields an output model where some tensors are merged across all models while others quietly come from a smaller subset. The inconsistency is especially telling because `AsyncMergePipeline::common_tensor_names()` uses intersection semantics instead.

15. `crates/pmetal-distributed/src/pipeline.rs:104-130,376-399`

Pipeline activation compression is write-only: compressed payloads are never decompressed on receipt.

`PipelineStageRuntime::send_to_next()` applies `compress_activation()` before shipping activations to the next shard, but `PipelineGenerationLoop::run_shard_loop()` passes `msg.data` directly into `forward_fn` without any matching decompression step or dtype adjustment. The wire message still advertises `dtype = self.config.wire_dtype`, so a stage configured for fp32 transport with `codec = Float16` will receive fp16-compressed bytes while the consumer continues to interpret them as raw fp32 activations. The default `Float16` wire dtype avoids the bug only because that path currently skips compression entirely.

#### P2

16. `crates/pmetal-distill/src/offline.rs:349-383`

Offline top-k distillation can panic on `top_k = 0`.

`compress_topk()` computes `let k = self.top_k.min(vocab_size)` and then, when `k < vocab_size`, calls `select_nth_unstable_by(k - 1, ...)`. If the caller sets `top_k` to `0`, `k - 1` underflows and panics. Neither the distillation config layer nor the CLI flag parser validates `offline_top_k > 0`, so this remains a user-triggerable crash path.

17. `crates/pmetal-mlx/src/smart_checkpoint.rs:940-963`

Smart checkpoint disk storage loses tensor metadata and silently flattens reloads.

The single-array checkpoint helpers write only raw f32 bytes and reload them as `Array::from_f32_slice(..., &[len])`, discarding the original shape and dtype. Any activation restored from disk comes back as a flat float32 vector rather than the original tensor, and unsupported conversions quietly serialize as empty arrays via `unwrap_or_default()`.

18. `crates/pmetal-mlx/src/smart_checkpoint.rs:781-823`

Long-context prefetch drops all but one activation per segment and renames it.

`LongContextManager` stores prefetched data as `HashMap<usize, Array>`, so a multi-activation segment is collapsed to `data.into_values().next()` during `prefetch()`. When the segment is later consumed from `prefetch_buffer`, `load_segment()` returns a one-entry map under the synthetic key `"prefetched"`, losing every original activation name except the first arbitrary value.

19. `crates/pmetal/src/tui/tabs/datasets.rs:156-158`

The dataset browser scans the model cache root instead of the Hugging Face datasets cache root.

`DatasetBrowser::scan_datasets()` passes `pmetal_hub::cache_dir()` into `scan_hf_datasets_cache()`, but that scanner explicitly expects a directory layout rooted at `datasets--...`. Since `pmetal_hub` exposes a dedicated `datasets_cache_dir()` for the datasets cache namespace, the TUI currently misses cached datasets in normal Hugging Face layouts and shows an incomplete dataset inventory.

## Cohesion Notes

These are not single-line bugs; they are cross-cutting places where the suite is currently less cohesive than a robust “gestalt” system should be.

1. Contract fidelity is inconsistent across APIs.

Several APIs accept capability-bearing inputs and then silently ignore them or degrade behavior: `Llama4::forward_with_cache()` ignores KV caches, some LoRA/model surfaces accept “uniform” arguments only for compatibility, and the serve layer accepts richer request fields while dropping them on the floor. For a SOTA suite, public API shape should track real capability. If a feature is unsupported, the system should either reject it explicitly or expose a separate capability probe that higher layers actually honor.

2. Round-trip symmetry is a recurring weakness.

The same pattern appears in multiple layers: GGUF write/read dimension order drift, offloading/checkpoint helpers dropping shape and dtype, distributed pipeline compression without decompression, and offline cache compression paths that do not preserve all of the original tensor contract. A cohesive system needs a stronger “every serialization path round-trips metadata and semantics” rule, backed by shared helpers rather than ad hoc per-crate encoders.

3. Error handling often favors silent degradation over explicit failure.

Examples include `value_and_grad` converting errors into NaN tensors, multiple merge/export paths using `to_f32_vec(...).unwrap_or_default()`, and UI/API layers accepting inputs they cannot faithfully serve. This makes the suite feel less trustworthy because failures become latent corruption or misleading behavior instead of actionable errors. For a robust platform, silent fallback should be the rare exception and should always emit structured diagnostics.

4. Cache and storage concepts are not yet unified across surfaces.

The hub layer distinguishes model cache and dataset cache, but higher layers still sometimes hard-code the wrong root. Similar fragmentation exists between runtime KV caches, offline teacher-logit caches, smart-checkpoint storage, and disk offloading. The suite would be more cohesive if cache/storage responsibilities were centralized around a small set of canonical path and metadata helpers used by CLI, TUI, Python, serve, and training code alike.

5. Different execution paths encode different semantics for the same operation.

The clearest example is merge: the standard path uses union-of-tensors semantics while the async path uses intersection-of-tensors semantics. That kind of divergence makes behavior depend on which codepath the caller happens to hit, not on one stable product contract. Similar risks exist anywhere there is a “fast path” and a “reference path”. The suite should define one semantic contract first, then make optimized paths prove equivalence to it.

6. Background work ownership is under-specified.

Async checkpointing currently demonstrates that background threads can outlive the handles meant to supervise them. More broadly, a gestalt system needs explicit lifecycle ownership for background I/O, prefetch, streaming, and transport tasks: who owns them, how errors surface, and what shutdown guarantees exist. Without that, upper layers may look healthy while detached work is still mutating state underneath.

7. Cross-crate integration testing is thinner than unit coverage suggests.

Most audited crates compile cleanly and have strong local tests, but many failures here live at boundaries: FFI trampolines, serialization round-trips, cache-aware inference contracts, distributed wire protocols, and cache-root coordination between hub and UI layers. A more cohesive audit posture would add contract tests that span crate boundaries, not just unit tests within each crate.

8. The suite needs a sharper definition of “authoritative metadata”.

Shape, dtype, alignment, endianness, cache topology, tensor presence, and model capability are all forms of metadata that currently get recomputed, assumed, or partially discarded in different layers. A robust SOTA suite should treat these as first-class invariants: preserved end-to-end, validated at boundaries, and sourced from one authoritative representation wherever possible.

## Verification Notes

- `cargo test -p pmetal-core`
- `cargo test -p pmetal-gguf`
- `cargo check -p pmetal-bridge`
- `cargo test -p pmetal-data`
- `cargo test -p pmetal-hub`
- `cargo test -p pmetal-metal`
- `cargo check -p pmetal-mlx`
- `cargo test -p pmetal-mlx`
- `cargo test -p pmetal-models`
- `cargo check -p pmetal-trainer`
- `cargo test -p pmetal-lora`
- `cargo test -p pmetal-distill`
- `cargo check -p pmetal-merge -p pmetal-distributed -p pmetal-serve -p pmetal-py -p pmetal`

These commands passed, which means the issues above are not currently covered by the existing tests in the audited lower layers.
