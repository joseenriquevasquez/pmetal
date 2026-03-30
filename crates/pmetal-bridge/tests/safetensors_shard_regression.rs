use pmetal_bridge::InlineArray;

const MODEL_SHARD: &str = "/Users/nickpaterno/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B/snapshots/cb9632e46f3232cffd569f81efa81dfceddb2c48/model.safetensors-00001-of-00001.safetensors";
const KNOWN_KEY: &str = "model.language_model.embed_tokens.weight";

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
