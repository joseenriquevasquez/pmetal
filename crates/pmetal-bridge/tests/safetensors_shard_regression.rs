use pmetal_bridge::InlineArray;

const MODEL_SHARD: &str = "/Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/cb9632e46f3232cffd569f81efa81dfceddb2c48/model.safetensors-00001-of-00001.safetensors";
const KNOWN_KEY: &str = "model.language_model.embed_tokens.weight";
const LARGE_MODEL_SHARD: &str = "/Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B/snapshots/61f7d17ddada509622e6f99972930732d61a9e0b/model.safetensors-00009-of-00014.safetensors";
const LARGE_MODEL_KEY: &str = "lm_head.weight";

#[test]
fn shard_loader_returns_known_qwen35_weights() {
    let entries = pmetal_bridge::inline_array::load_safetensors_shard(MODEL_SHARD)
        .expect("failed to load Qwen3.5 shard");

    assert!(!entries.is_empty(), "shard loader returned no tensors");

    let shard_weight = entries
        .iter()
        .find(|(key, _)| key == KNOWN_KEY)
        .map(|(_, array)| array.clone())
        .expect("known embedding weight missing from shard load");

    let single_weight = InlineArray::load_safetensors(MODEL_SHARD, KNOWN_KEY)
        .expect("failed to load known embedding weight directly");

    assert_eq!(shard_weight.ndim(), single_weight.ndim());
    for axis in 0..shard_weight.ndim() {
        assert_eq!(shard_weight.dim(axis), single_weight.dim(axis));
    }
}

#[test]
fn large_shard_loader_returns_qwen35_a3b_lm_head() {
    let weight = InlineArray::load_safetensors(LARGE_MODEL_SHARD, LARGE_MODEL_KEY)
        .expect("failed to load large Qwen3.5-A3B lm_head shard");

    assert_eq!(weight.ndim(), 2);
    assert_eq!(weight.dim(0), 248320);
    assert_eq!(weight.dim(1), 2048);
}
