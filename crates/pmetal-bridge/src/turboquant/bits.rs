//! Bit-packed storage primitives shared across the TurboQuant cache modules.
//!
//! `PackedBits` is the storage backend for variable-width centroid indices
//! (1–8 bits each). `packed_qjl_words` is the canonical D→u32-word reduction
//! used by every QJL-related call site (storage layout, score kernels,
//! attention kernels). Pinning it here as the single source of truth lets the
//! `layout_invariants` test in mod.rs catch silent drift between callers.

/// Bit-packed storage for variable-width unsigned integers (1–8 bits each).
///
/// Values are stored LSB-first in a contiguous byte buffer.  Provides O(1)
/// random read and amortised O(1) append.
#[derive(Debug, Clone)]
pub struct PackedBits {
    bits_per_value: u8,
    len: usize,
    bytes: Vec<u8>,
}

impl PackedBits {
    pub fn new(bits_per_value: u8) -> Self {
        Self {
            bits_per_value,
            len: 0,
            bytes: Vec::new(),
        }
    }

    pub fn extend_from_slice(&mut self, values: &[u16]) {
        if self.bits_per_value == 0 || values.is_empty() {
            self.len += values.len();
            return;
        }

        for &value in values {
            debug_assert!(u32::from(value) < (1u32 << self.bits_per_value));
            let bit_offset = self.len * usize::from(self.bits_per_value);
            let required_bits = bit_offset + usize::from(self.bits_per_value);
            let required_bytes = required_bits.div_ceil(8);
            if self.bytes.len() < required_bytes {
                self.bytes.resize(required_bytes, 0);
            }
            for bit in 0..self.bits_per_value {
                if ((value >> bit) & 1) != 0 {
                    let target_bit = bit_offset + usize::from(bit);
                    self.bytes[target_bit / 8] |= 1u8 << (target_bit % 8);
                }
            }
            self.len += 1;
        }
    }

    pub fn get(&self, index: usize) -> u16 {
        debug_assert!(index < self.len);
        if self.bits_per_value == 0 {
            return 0;
        }
        let bit_offset = index * usize::from(self.bits_per_value);
        let mut value = 0u16;
        for bit in 0..self.bits_per_value {
            let target_bit = bit_offset + usize::from(bit);
            let byte = self.bytes[target_bit / 8];
            if ((byte >> (target_bit % 8)) & 1) != 0 {
                value |= 1u16 << bit;
            }
        }
        value
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        self.len = new_len;
        if self.bits_per_value == 0 {
            return;
        }
        let total_bits = self.len * usize::from(self.bits_per_value);
        let total_bytes = total_bits.div_ceil(8);
        self.bytes.truncate(total_bytes);
        if let Some(last) = self.bytes.last_mut() {
            let used_bits = total_bits % 8;
            if used_bits != 0 {
                *last &= (1u8 << used_bits) - 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

pub(super) fn unpack_all(bits: &PackedBits) -> Vec<u16> {
    (0..bits.len()).map(|i| bits.get(i)).collect()
}

pub(super) fn packed_qjl_words(dim: usize) -> usize {
    dim.div_ceil(32)
}
