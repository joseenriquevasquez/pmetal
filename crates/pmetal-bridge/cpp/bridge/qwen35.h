// Full Qwen3.5 forward pass — single C++ function, zero FFI overhead.
//
// Implements the entire 24-layer decode step (or prefill for T>1) inside C++,
// eliminating ~1800 per-op FFI round trips per decode step.
//
// Weight pointer layout (weight_ptrs, num_weights = 3 + num_layers * WEIGHTS_PER_LAYER):
//   [0]  embed_w          (global)
//   [1]  final_norm_w     (global)
//   [2]  lm_head_w        (global; NULL when tie_word_embeddings=true)
//   Per-layer block at base index 3 + layer_idx * WEIGHTS_PER_LAYER:
//     [+0]  input_ln_w
//     [+1]  post_ln_w
//     [+2]  dense mlp_gate_w / moe_router_w
//     [+3]  dense mlp_up_w   / moe_gate_w
//     [+4]  dense mlp_down_w / moe_up_w
//   GDN-only offsets (+5 .. +15), NULL for attention layers:
//     [+5]  gdn_qkv_w
//     [+6]  gdn_z_w
//     [+7]  gdn_b_w
//     [+8]  gdn_a_w
//     [+9]  gdn_conv_w
//     [+10] gdn_q_nw
//     [+11] gdn_k_nw
//     [+12] gdn_a_log
//     [+13] gdn_dt_bias
//     [+14] gdn_norm_w
//     [+15] gdn_out_w
//   Attention-only offsets (+5 .. +10), NULL for GDN layers:
//     [+5]  attn_q_w
//     [+6]  attn_k_w
//     [+7]  attn_v_w
//     [+8]  attn_o_w
//     [+9]  attn_q_norm_w
//     [+10] attn_k_norm_w
//   MoE-only offsets (+16 .. +20), NULL for dense MLP layers:
//     [+16] moe_down_w
//     [+17] shared_gate_w
//     [+18] shared_up_w
//     [+19] shared_down_w
//     [+20] shared_expert_gate_w
//
// Cache pointer layout (cache_ptrs, num_cache = n_gdn*2 + n_attn*4):
//   For each GDN layer gi = 0..n_gdn-1:
//     [gi*2+0]  conv_state  (in/out; may be NULL on first call → zeros allocated inside)
//     [gi*2+1]  ssm_state   (in/out; may be NULL on first call)
//   For each attention layer ai = 0..n_attn-1:
//     [n_gdn*2 + ai*4+0]  kv_keys    (in/out; may be NULL on first call)
//     [n_gdn*2 + ai*4+1]  kv_values  (in/out; may be NULL on first call)
//
// Scalar cache (passed separately, updated in-place):
//   attn_kv_offsets[n_attn]  — per-attention-layer valid-token count (in/out)
//   rope_offset[1]            — global position counter (in/out)
//
// Config integer layout (config_ints, num_config_ints = 20 + num_layers):
//   [0]  num_layers
//   [1]  hidden_size
//   [2]  model_dtype          (11 = bfloat16)
//   [3]  n_gdn                (# GDN layers)
//   [4]  n_attn               (# attention layers)
//   [5]  gdn_nv
//   [6]  gdn_nk
//   [7]  gdn_dk
//   [8]  gdn_dv
//   [9]  gdn_cd
//   [10] gdn_ck
//   [11] gdn_kd
//   [12] attn_n_heads
//   [13] attn_n_kv
//   [14] attn_head_dim
//   [15] attn_rope_dims
//   [16] full_attn_interval   (every Nth layer is attention, e.g. 4)
//   [17] tie_word_embeddings  (1 = tied, 0 = separate lm_head)
//   [18] moe_top_k
//   [19] moe_norm_topk_prob
//   [20 + i] (i = 0..num_layers-1) layer_i_is_moe
//
// Config float layout (config_floats):
//   [0]                                     final_norm_eps
//   [1]                                     attn_scale
//   [2]                                     attn_rope_base
//   [3]                                     attn_rope_scale
//   [4 + i*2]   (i = 0..num_layers-1)      layer_i input_ln_eps
//   [4 + i*2+1]                             layer_i post_ln_eps
//   [4 + num_layers*2 + gi]  (gi=0..n_gdn-1)  gdn_norm_eps
//   [4 + num_layers*2 + n_gdn + ai*2]       attn_q_norm_eps
//   [4 + num_layers*2 + n_gdn + ai*2+1]     attn_k_norm_eps
//
// Returns logits [B, T, vocab] via placement-new into dst_logits.

#ifndef MLX_INLINE_BRIDGE_QWEN35_H
#define MLX_INLINE_BRIDGE_QWEN35_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QWEN35_WEIGHTS_PER_LAYER 21

void mlx_inline_qwen35_decode_step(
    mlx_inline_array*              dst_logits,
    const mlx_inline_array*        token_ids,          // [B, T] int32
    const mlx_inline_array* const* weight_ptrs,        // flat weight array
    int                            num_weights,
    mlx_inline_array**             cache_ptrs,          // flat cache array (in/out)
    int                            num_cache,
    int*                           attn_kv_offsets,     // [n_attn] in/out
    int*                           rope_offset,         // [1] in/out
    const int*                     config_ints,         // [20 + num_layers]
    int                            num_config_ints,
    const float*                   config_floats,       // see layout above
    int                            num_config_floats
);

#ifdef __cplusplus
}
#endif

#endif
