//! Paged KV cache - block-based memory management for batched inference.

use pmetal_bridge::compat::{Array, Dtype, Exception, ops};
use tracing::warn;

use super::dtype_size;

/// Block size for paged attention (tokens per block).
/// 32 tokens is optimal for Apple Silicon (matches GPU cache lines).
pub const DEFAULT_BLOCK_SIZE: usize = 32;

/// Configuration for paged KV cache.
#[derive(Debug, Clone)]
pub struct PagedKVCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of key-value heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Block size (tokens per block).
    pub block_size: usize,
    /// Maximum number of blocks to allocate.
    pub max_blocks: usize,
    /// Data type for cached tensors.
    pub dtype: Dtype,
}

impl PagedKVCacheConfig {
    /// Create a new paged KV cache configuration.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `num_kv_heads` - Number of KV heads
    /// * `head_dim` - Dimension per head
    /// * `max_seq_len` - Maximum sequence length to support
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let max_blocks = max_seq_len.div_ceil(DEFAULT_BLOCK_SIZE);
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            block_size: DEFAULT_BLOCK_SIZE,
            max_blocks,
            dtype: Dtype::Float16,
        }
    }

    /// Set the block size.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        let previous_capacity = self.max_blocks.saturating_mul(self.block_size);
        let block_size = block_size.max(1);
        self.block_size = block_size;
        self.max_blocks = previous_capacity.div_ceil(block_size);
        self
    }

    /// Set the dtype.
    pub fn with_dtype(mut self, dtype: Dtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the maximum number of blocks.
    pub fn with_max_blocks(mut self, max_blocks: usize) -> Self {
        self.max_blocks = max_blocks;
        self
    }
}

/// Block allocator for managing physical memory blocks.
#[derive(Debug)]
pub struct BlockAllocator {
    /// Free block indices.
    free_blocks: Vec<usize>,
    /// Total number of blocks allocated.
    total_blocks: usize,
    /// Block size in tokens.
    block_size: usize,
    /// Authoritative count of free blocks.
    free_count: usize,
    /// Whether each physical block is currently free.
    is_free: Vec<bool>,
}

impl BlockAllocator {
    /// Create a new block allocator.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            free_blocks: (0..num_blocks).rev().collect(), // Stack-like for LIFO reuse
            total_blocks: num_blocks,
            block_size,
            free_count: num_blocks,
            is_free: vec![true; num_blocks],
        }
    }

    /// Allocate a block, returning its index.
    pub fn allocate(&mut self) -> Option<usize> {
        while let Some(block_idx) = self.free_blocks.pop() {
            if block_idx >= self.total_blocks {
                warn!(block_idx, "ignoring out-of-range free KV cache block");
                continue;
            }
            if !self.is_free[block_idx] {
                warn!(block_idx, "ignoring duplicate free KV cache block entry");
                continue;
            }

            self.is_free[block_idx] = false;
            self.free_count = self.free_count.saturating_sub(1);
            return Some(block_idx);
        }

        None
    }

    /// Allocate multiple blocks.
    pub fn allocate_n(&mut self, n: usize) -> Option<Vec<usize>> {
        if self.free_count < n {
            return None;
        }

        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(block_idx) = self.allocate() {
                blocks.push(block_idx);
            } else {
                self.free_all(&blocks);
                return None;
            }
        }

        Some(blocks)
    }

    /// Free a block.
    pub fn free(&mut self, block_idx: usize) {
        if block_idx >= self.total_blocks {
            warn!(block_idx, "ignoring out-of-range KV cache block free");
            return;
        }
        if self.is_free[block_idx] {
            warn!(block_idx, "ignoring duplicate KV cache block free");
            return;
        }

        self.is_free[block_idx] = true;
        self.free_count = self.free_count.saturating_add(1).min(self.total_blocks);
        self.free_blocks.push(block_idx);
    }

    /// Free multiple blocks.
    pub fn free_all(&mut self, blocks: &[usize]) {
        for &block_idx in blocks {
            self.free(block_idx);
        }
    }

    /// Get the number of free blocks.
    pub fn num_free(&self) -> usize {
        self.free_count
    }

    /// Get the number of allocated blocks.
    pub fn num_allocated(&self) -> usize {
        self.total_blocks - self.free_count
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get total blocks.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }
}

/// Block table mapping logical to physical blocks for a sequence.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical to physical block mapping.
    block_indices: Vec<usize>,
    /// Number of tokens stored.
    pub(crate) num_tokens: usize,
    /// Block size.
    block_size: usize,
}

impl BlockTable {
    /// Create a new block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_indices: Vec::new(),
            num_tokens: 0,
            block_size: block_size.max(1),
        }
    }

    /// Get the number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_indices.len()
    }

    /// Get the number of tokens.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get the block indices.
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }

    /// Add a block to the table.
    pub fn add_block(&mut self, block_idx: usize) {
        self.block_indices.push(block_idx);
    }

    /// Add tokens to the table, returning number of new blocks needed.
    pub fn add_tokens(&mut self, num_tokens: usize) -> usize {
        let old_blocks = self.num_tokens.div_ceil(self.block_size);
        self.num_tokens = self.num_tokens.saturating_add(num_tokens);
        let new_blocks = self.num_tokens.div_ceil(self.block_size);
        new_blocks.saturating_sub(old_blocks)
    }

    /// Get the physical block and offset for a token position.
    pub fn get_block_and_offset(&self, token_pos: usize) -> Option<(usize, usize)> {
        let block_idx = token_pos / self.block_size;
        let offset = token_pos % self.block_size;
        self.block_indices
            .get(block_idx)
            .map(|&phys| (phys, offset))
    }
}

/// Paged KV cache for efficient batched inference.
///
/// This cache uses block-based memory management for:
/// - Memory-efficient variable-length batching
/// - Near-zero memory fragmentation
/// - Efficient block reuse across sequences
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────┐
/// │                    Physical Blocks                          │
/// │ [Block 0][Block 1][Block 2][Block 3][Block 4]...           │
/// │    K+V      K+V      K+V      K+V      K+V                  │
/// └─────────────────────────────────────────────────────────────┘
///                    ↑ ↑ ↑
/// ┌──────────────────┘ │ └──────────────────┐
/// │                    │                    │
/// │ Sequence 0        Sequence 1           Sequence 2          │
/// │ [0, 3]            [1, 4]               [2]                  │
/// │ (64 tokens)       (64 tokens)          (32 tokens)          │
/// └─────────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug)]
pub struct PagedKVCache {
    /// Configuration.
    pub(crate) config: PagedKVCacheConfig,
    /// Block allocator.
    allocator: BlockAllocator,
    /// Physical key blocks per layer [layer][block][kv_heads, block_size, head_dim].
    key_blocks: Vec<Vec<Option<Array>>>,
    /// Physical value blocks per layer [layer][block][kv_heads, block_size, head_dim].
    value_blocks: Vec<Vec<Option<Array>>>,
    /// Block tables per sequence.
    block_tables: std::collections::HashMap<u64, BlockTable>,
    /// Next sequence ID.
    next_seq_id: u64,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    pub fn new(config: PagedKVCacheConfig) -> Self {
        let mut config = config;
        if config.block_size == 0 {
            warn!("paged KV cache block size of zero requested; using block size 1");
            config.block_size = 1;
        }

        let num_layers = config.num_layers;
        let max_blocks = config.max_blocks;

        // Pre-allocate block storage (but not the actual arrays yet - lazy allocation)
        let key_blocks: Vec<Vec<Option<Array>>> =
            (0..num_layers).map(|_| vec![None; max_blocks]).collect();
        let value_blocks: Vec<Vec<Option<Array>>> =
            (0..num_layers).map(|_| vec![None; max_blocks]).collect();

        Self {
            allocator: BlockAllocator::new(max_blocks, config.block_size),
            key_blocks,
            value_blocks,
            block_tables: std::collections::HashMap::new(),
            next_seq_id: 0,
            config,
        }
    }

    /// Allocate a new sequence, returning its ID.
    ///
    /// # Arguments
    /// * `initial_tokens` - Number of tokens to allocate initially (typically prompt length)
    pub fn allocate_sequence(&mut self, initial_tokens: usize) -> Result<u64, Exception> {
        let seq_id = self.next_seq_id;
        let next_seq_id = self
            .next_seq_id
            .checked_add(1)
            .ok_or_else(|| Exception::custom("KV cache sequence ID overflow"))?;

        let num_blocks = initial_tokens.div_ceil(self.config.block_size);
        let blocks = self
            .allocator
            .allocate_n(num_blocks)
            .ok_or_else(|| Exception::custom("Out of KV cache blocks"))?;

        for &block_idx in &blocks {
            if let Err(err) = self.ensure_block_allocated(block_idx) {
                self.allocator.free_all(&blocks);
                return Err(err);
            }
        }

        let mut table = BlockTable::new(self.config.block_size);
        for block_idx in blocks {
            table.add_block(block_idx);
        }
        table.num_tokens = initial_tokens;

        self.next_seq_id = next_seq_id;
        self.block_tables.insert(seq_id, table);
        Ok(seq_id)
    }

    /// Extend a sequence with additional tokens.
    pub fn extend_sequence(&mut self, seq_id: u64, num_tokens: usize) -> Result<(), Exception> {
        let (new_tokens, new_blocks_needed) = {
            let table = self
                .block_tables
                .get(&seq_id)
                .ok_or_else(|| Exception::custom("Sequence not found"))?;
            let new_tokens = table
                .num_tokens
                .checked_add(num_tokens)
                .ok_or_else(|| Exception::custom("KV cache token count overflow"))?;
            let old_blocks = table.num_tokens.div_ceil(self.config.block_size);
            let new_blocks = new_tokens.div_ceil(self.config.block_size);
            (new_tokens, new_blocks.saturating_sub(old_blocks))
        };

        let new_block_indices = self
            .allocator
            .allocate_n(new_blocks_needed)
            .ok_or_else(|| Exception::custom("Out of KV cache blocks"))?;

        for &block_idx in &new_block_indices {
            if let Err(err) = self.ensure_block_allocated(block_idx) {
                self.allocator.free_all(&new_block_indices);
                return Err(err);
            }
        }

        if let Some(table) = self.block_tables.get_mut(&seq_id) {
            table.num_tokens = new_tokens;
            for block_idx in new_block_indices {
                table.add_block(block_idx);
            }
        }

        Ok(())
    }

    /// Free a sequence and return its blocks.
    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(table) = self.block_tables.remove(&seq_id) {
            self.allocator.free_all(table.block_indices());
        }
    }

    /// Update KV cache for a sequence at a specific layer.
    ///
    /// # Arguments
    /// * `seq_id` - Sequence ID
    /// * `layer_idx` - Layer index
    /// * `new_keys` - New keys [batch=1, kv_heads, new_seq, head_dim]
    /// * `new_values` - New values [batch=1, kv_heads, new_seq, head_dim]
    /// * `start_pos` - Starting position in the sequence
    pub fn update(
        &mut self,
        seq_id: u64,
        layer_idx: usize,
        new_keys: &Array,
        new_values: &Array,
        start_pos: usize,
    ) -> Result<(), Exception> {
        self.validate_layer(layer_idx)?;
        self.validate_update_inputs(new_keys, new_values)?;
        let num_new_tokens = new_keys.dim(2) as usize;

        // Collect all block/offset pairs first to avoid holding table borrow
        let block_offsets: Vec<(usize, usize)> = {
            let table = self
                .block_tables
                .get(&seq_id)
                .ok_or_else(|| Exception::custom("Sequence not found"))?;

            (0..num_new_tokens)
                .map(|i| {
                    let token_pos = start_pos.checked_add(i)?;
                    table.get_block_and_offset(token_pos)
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Exception::custom("Token position out of range"))?
        };

        // Now update blocks
        for (i, (block_idx, offset)) in block_offsets.into_iter().enumerate() {
            // Slice the single token from input: [1, heads, 1, dim]
            let kh = new_keys.dim(1) as usize;
            let kd = new_keys.dim(3) as usize;
            let vd = new_values.dim(3) as usize;
            let kb = new_keys.dim(0) as usize;
            let k_token = new_keys.slice(
                &[0, 0, i as i32, 0],
                &[kb as i32, kh as i32, (i + 1) as i32, kd as i32],
            );
            let v_token = new_values.slice(
                &[0, 0, i as i32, 0],
                &[kb as i32, kh as i32, (i + 1) as i32, vd as i32],
            );

            // Update block at offset
            self.update_block_at_offset(layer_idx, block_idx, offset, &k_token, &v_token)?;
        }

        Ok(())
    }

    /// Fetch cached K/V for attention computation.
    ///
    /// Returns concatenated K/V arrays for all tokens in the sequence.
    pub fn fetch(&self, seq_id: u64, layer_idx: usize) -> Result<(Array, Array), Exception> {
        self.validate_layer(layer_idx)?;
        let table = self
            .block_tables
            .get(&seq_id)
            .ok_or_else(|| Exception::custom("Sequence not found"))?;

        let num_tokens = table.num_tokens();
        if num_tokens == 0 {
            return Err(Exception::custom("Empty sequence"));
        }

        // Gather blocks and concatenate
        let mut key_parts: Vec<Array> = Vec::new();
        let mut value_parts: Vec<Array> = Vec::new();

        let block_size = self.config.block_size;
        let mut remaining = num_tokens;

        for &block_idx in table.block_indices().iter() {
            let tokens_in_block = remaining.min(block_size);

            if let (Some(k_block), Some(v_block)) = (
                &self.key_blocks[layer_idx][block_idx],
                &self.value_blocks[layer_idx][block_idx],
            ) {
                // Slice the valid portion of the block [heads, tokens, dim]
                let k_slice = if tokens_in_block < block_size {
                    let h = k_block.dim(0) as usize;
                    let d = k_block.dim(2) as usize;
                    k_block.slice(&[0, 0, 0], &[h as i32, tokens_in_block as i32, d as i32])
                } else {
                    k_block.clone()
                };
                let v_slice = if tokens_in_block < block_size {
                    let h = v_block.dim(0) as usize;
                    let d = v_block.dim(2) as usize;
                    v_block.slice(&[0, 0, 0], &[h as i32, tokens_in_block as i32, d as i32])
                } else {
                    v_block.clone()
                };

                key_parts.push(k_slice);
                value_parts.push(v_slice);
            }

            remaining -= tokens_in_block;
            if remaining == 0 {
                break;
            }
        }

        // Concatenate all blocks along sequence dimension
        if key_parts.is_empty() {
            return Err(Exception::custom("No blocks to fetch"));
        }

        // Concatenate [heads, seq, dim] blocks, then expand to [1, heads, seq, dim]
        let keys = ops::concatenate_owned_axis(&key_parts, 1);
        let values = ops::concatenate_owned_axis(&value_parts, 1);

        // Reshape from [heads, seq, dim] to [1, heads, seq, dim]
        let keys = keys.expand_dims(0);
        let values = values.expand_dims(0);

        Ok((keys, values))
    }

    /// Get the block table for a sequence (for kernel dispatch).
    pub fn get_block_table(&self, seq_id: u64) -> Option<&BlockTable> {
        self.block_tables.get(&seq_id)
    }

    /// Get number of sequences.
    pub fn num_sequences(&self) -> usize {
        self.block_tables.len()
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> PagedCacheMemoryStats {
        let block_elements =
            self.config.num_kv_heads * self.config.block_size * self.config.head_dim;
        let bytes_per_block = block_elements * dtype_size(self.config.dtype) * 2; // K + V

        PagedCacheMemoryStats {
            total_blocks: self.allocator.total_blocks(),
            allocated_blocks: self.allocator.num_allocated(),
            free_blocks: self.allocator.num_free(),
            bytes_per_block,
            total_memory_bytes: self.allocator.total_blocks()
                * bytes_per_block
                * self.config.num_layers,
            used_memory_bytes: self.allocator.num_allocated()
                * bytes_per_block
                * self.config.num_layers,
        }
    }

    /// Reset the cache, freeing all sequences.
    pub fn reset(&mut self) {
        // Free all block tables
        let seq_ids: Vec<u64> = self.block_tables.keys().cloned().collect();
        for seq_id in seq_ids {
            self.free_sequence(seq_id);
        }
        self.next_seq_id = 0;
    }

    /// Ensure a block is allocated (lazy allocation).
    fn ensure_block_allocated(&mut self, block_idx: usize) -> Result<(), Exception> {
        if block_idx >= self.config.max_blocks {
            return Err(Exception::custom("KV cache block index out of range"));
        }

        let shape = [
            self.config.num_kv_heads as i32,
            self.config.block_size as i32,
            self.config.head_dim as i32,
        ];

        for layer_idx in 0..self.config.num_layers {
            if self.key_blocks[layer_idx][block_idx].is_none() {
                self.key_blocks[layer_idx][block_idx] = Some(ops::zeros(&shape, Dtype::Float32));
                self.value_blocks[layer_idx][block_idx] = Some(ops::zeros(&shape, Dtype::Float32));
            }
        }
        Ok(())
    }

    /// Update a block at a specific offset.
    fn update_block_at_offset(
        &mut self,
        layer_idx: usize,
        block_idx: usize,
        offset: usize,
        key: &Array,
        value: &Array,
    ) -> Result<(), Exception> {
        self.validate_layer(layer_idx)?;
        if block_idx >= self.config.max_blocks {
            return Err(Exception::custom("KV cache block index out of range"));
        }
        if offset >= self.config.block_size {
            return Err(Exception::custom("KV cache block offset out of range"));
        }

        // Get or create the block
        let k_block = self.key_blocks[layer_idx][block_idx]
            .take()
            .ok_or_else(|| Exception::custom("Block not allocated"))?;
        let v_block = self.value_blocks[layer_idx][block_idx]
            .take()
            .ok_or_else(|| Exception::custom("Block not allocated"))?;

        // Remove batch dimension from input [1, heads, 1, dim] -> [heads, 1, dim]
        let k_squeezed = key.squeeze(0);
        let v_squeezed = value.squeeze(0);

        // In-place update using slice_set: block[.., offset..offset+1, ..] = squeezed
        let h = k_block.dim(0) as usize;
        let kd = k_block.dim(2) as usize;
        let vd = v_block.dim(2) as usize;

        let new_k = k_block.slice_set(
            &k_squeezed,
            &[0, offset as i32, 0],
            &[h as i32, (offset + 1) as i32, kd as i32],
        );
        let new_v = v_block.slice_set(
            &v_squeezed,
            &[0, offset as i32, 0],
            &[h as i32, (offset + 1) as i32, vd as i32],
        );

        self.key_blocks[layer_idx][block_idx] = Some(new_k);
        self.value_blocks[layer_idx][block_idx] = Some(new_v);

        Ok(())
    }

    fn validate_layer(&self, layer_idx: usize) -> Result<(), Exception> {
        if layer_idx >= self.config.num_layers {
            return Err(Exception::custom("KV cache layer index out of range"));
        }
        Ok(())
    }

    fn validate_update_inputs(
        &self,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<(), Exception> {
        if new_keys.ndim() != 4 || new_values.ndim() != 4 {
            return Err(Exception::custom(
                "Paged KV cache update expects key/value tensors shaped [1, heads, seq, dim]",
            ));
        }

        let key_shape = new_keys.shape();
        let value_shape = new_values.shape();
        if key_shape[0] != 1 || value_shape[0] != 1 {
            return Err(Exception::custom(
                "Paged KV cache update only supports batch size 1",
            ));
        }
        if key_shape[1] != self.config.num_kv_heads as i32
            || value_shape[1] != self.config.num_kv_heads as i32
        {
            return Err(Exception::custom("Paged KV cache head count mismatch"));
        }
        if key_shape[2] != value_shape[2] {
            return Err(Exception::custom(
                "Paged KV cache key/value sequence length mismatch",
            ));
        }
        if key_shape[3] != self.config.head_dim as i32
            || value_shape[3] != self.config.head_dim as i32
        {
            return Err(Exception::custom("Paged KV cache head dimension mismatch"));
        }

        Ok(())
    }
}

/// Memory statistics for paged cache.
#[derive(Debug, Clone)]
pub struct PagedCacheMemoryStats {
    /// Total number of blocks.
    pub total_blocks: usize,
    /// Number of allocated blocks.
    pub allocated_blocks: usize,
    /// Number of free blocks.
    pub free_blocks: usize,
    /// Bytes per block.
    pub bytes_per_block: usize,
    /// Total memory in bytes.
    pub total_memory_bytes: usize,
    /// Used memory in bytes.
    pub used_memory_bytes: usize,
}

impl PagedCacheMemoryStats {
    /// Get memory utilization as a percentage.
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            (self.allocated_blocks as f64 / self.total_blocks as f64) * 100.0
        }
    }
}

/// Convenience function to create a paged KV cache.
pub fn create_paged_cache(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> PagedKVCache {
    PagedKVCache::new(PagedKVCacheConfig::new(
        num_layers,
        num_kv_heads,
        head_dim,
        max_seq_len,
    ))
}
