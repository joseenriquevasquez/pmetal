# MLlama Dispatcher Integration — Deferred Design Note

**Status:** Deferred (2026-04-24). Standalone `MllamaLoraForCausalLM` adapter exists at `crates/pmetal-lora/src/mllama_lora.rs` and is usable via its own API; it is not reachable through `DynamicLoraModel` / `DynamicQloraModel`.

## Why deferred

MLlama is the first multimodal (vision + text) architecture we've considered wiring into `DynamicModel`. Its forward signature is fundamentally incompatible with the text-LLM dispatcher contract:

```rust
// text archs (dispatch_uniform compatible):
fn forward(&mut self, input_ids: &Array, cache: Option<&mut KVCache>) -> Result<Array, Exception>;

// MLlama:
fn forward(&mut self, input_ids: &Array, pixel_values: Option<&Array>) -> Result<Array, Exception>;
//   no KV cache, pixel_values instead, cross-attention decoder layers
```

The `dispatch_uniform!` macro in `crates/pmetal-models/src/dispatcher.rs` assumes uniform call signatures across variants. Forcing MLlama in today would mean either (a) stubbing cross-attention out — wrong, (b) special-casing every arm — noisy, or (c) building a parallel multimodal-dispatcher lane without design.

## Scope to land full integration

### 1. `pmetal-models`
- Add `ModelArchitecture::Mllama` variant + `Display` + `from_model_type` (`"mllama"`) + `from_architectures` (`"MllamaForConditionalGeneration"`).
- Add `DynamicModel::Mllama(MllamaForConditionalGeneration)` variant.
- Decide dispatcher strategy (see "Design options" below).
- Update `simple_load!`-equivalent path with vision + text + multimodal-projector weight loading (weight-key prefixes: `vision_model.*`, `language_model.*`, `multi_modal_projector.*`, `lm_head.*`).
- Update `eval_module_parameters_batched` if vision tower param counts cause issues.

### 2. `pmetal-lora`
- Add `DynamicLoraModel::Mllama(MllamaLoraForCausalLM)` with all ~15 `TrainableModel` dispatch arms (`supports_kv_cache = true` for self-attn path; cross-attn layers cache differently).
- Add `DynamicQloraModel::Mllama(MllamaQloraForCausalLM)` — requires building `mllama_qlora.rs` first (NF4 on text decoder, vision tower unquantized frozen to keep alignment).
- Update `arch_config.rs` with MLlama target-module defaults.

### 3. `pmetal-py`
- Update `pmetal-py/src/config.rs` bindings.

### 4. Weight loader
- MLlama HF checkpoint has nested prefixes that may not match `load_generic_weights` assumptions — verify / add a sanitizer.

## Design options for the dispatcher

**Option A — Multimodal lane (recommended).** Add a parallel enum `DynamicMultimodalModel` with its own dispatcher that takes optional `pixel_values`. Text-only operations (`forward`, `forward_hidden`, `create_cache`) delegate into the inner text backbone. This establishes the pattern for Llama4-Vision, Qwen2.5-VL, Gemma3-Vision, PaliGemma, MiniCPM-V, etc. Shared trait `MultimodalModel` for the multimodal methods.

**Option B — Extend `DynamicModel` with per-variant forward overrides.** Add `forward_multimodal(input_ids, pixel_values)` method that returns error for non-MM variants, and have the text-only `forward` for MLlama ignore vision. More ergonomic short-term but pollutes the text dispatcher with vision concerns and doesn't scale.

**Option C — Trait-based dispatch.** Replace the variant enums with `Box<dyn TextModel>` / `Box<dyn MultimodalModel>`. Larger refactor but the cleanest long-term. Worth considering when the third multimodal arch lands.

Recommendation: **A now**, revisit **C** at the 3rd multimodal arch.

## Pre-work checklist (pick up here)

- [ ] Audit `MllamaForConditionalGeneration` forward — confirm cross-attn KV cache story (per-decoder-layer cross-attn state cache).
- [ ] Decide whether `DynamicMultimodalModel` lives in `pmetal-models` or a new `pmetal-models-mm` crate.
- [ ] Prototype weight loading end-to-end against `meta-llama/Llama-3.2-11B-Vision-Instruct`.
- [ ] Define the `MultimodalModel` trait surface (forward w/ pixels, encode_images, text-only fallback).
- [ ] Wire `MllamaLoraForCausalLM` + build `mllama_qlora.rs` behind the new dispatcher.
- [ ] Update `pmetal search` / `pmetal-hub` fit estimation with vision-tower memory contribution.

## Known MLlama training constraints

- Vision tower is always frozen during LoRA — alignment matters more than vision adaptation.
- Cross-attention layers are the highest-value LoRA targets (they gate vision → text information flow).
- Self-attention LoRA on text decoder is standard.
- MLlama cross-attention cache is **not** the standard `KVCache` — it's computed once from the image and reused; needs a distinct cache type or cache-mode variant.

## Today's usable surface

Users who need MLlama training right now can construct `MllamaLoraForCausalLM` directly and drive training via its concrete API. It just bypasses the `DynamicLoraModel` auto-detection path.
