# Inductor Fusion Catalog (WI6 — Phase 2)

Source: `docs/perf/inductor-triton-dump.cache.py` (1,476,273 bytes)

**Total unique triton kernels:** 92
**Total kernel declarations (including dupes):** 160

## Kernels by kind

| Kind | Count |
|------|------:|
| pointwise | 48 |
| persistent-reduction | 33 |
| reduction | 11 |

## Fusion size distribution (# of ops fused per kernel)

| # ops fused | # kernels |
|-----------:|----------:|
| 2 | 14 |
| 3 | 10 |
| 4 | 1 |
| 5 | 11 |
| 6 | 2 |
| 7 | 6 |
| 8 | 8 |
| 9 | 10 |
| 11 | 4 |
| 12 | 8 |
| 13 | 3 |
| 14 | 5 |
| 17 | 1 |
| 18 | 3 |
| 19 | 2 |
| 21 | 2 |
| 23 | 1 |
| 24 | 1 |

## Top 30 most-fused kernels (by op count)

| # | Kernel | Kind | Fused ops | Loads | Stores | Exps | Sqrts | Body bytes | File |
|--:|--------|------|----------|------:|-------:|-----:|------:|-----------:|------|
| 1 | `triton_per_fused__to_copy__unsafe_view_add_clamp_min_clone_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_slice_squeeze_sum_transpose_view_where_8` | persistent-reduction | `_to, copy, _unsafe, view, add, clamp, min, clone, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, slice, squeeze, sum, transpose, view, where` | 6 | 1 | 0 | 0 | 4,331 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 2 | `triton_per_fused__to_copy__unsafe_view_add_clamp_min_clone_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_squeeze_sum_transpose_view_where_7` | persistent-reduction | `_to, copy, _unsafe, view, add, clamp, min, clone, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, squeeze, sum, transpose, view, where` | 6 | 1 | 0 | 0 | 4,326 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 3 | `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_sum_transpose_view_where_8` | persistent-reduction | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, sum, transpose, view, where` | 4 | 1 | 0 | 0 | 3,966 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 4 | `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_sum_transpose_view_where_9` | persistent-reduction | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, sum, transpose, view, where` | 4 | 1 | 0 | 0 | 3,966 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 5 | `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_scalar_tensor_transpose_view_where_10` | pointwise | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, scalar, tensor, transpose, view, where` | 1 | 1 | 0 | 0 | 2,436 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 6 | `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_scalar_tensor_transpose_view_where_9` | pointwise | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, scalar, tensor, transpose, view, where` | 1 | 1 | 0 | 0 | 2,434 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 7 | `triton_per_fused__scaled_dot_product_efficient_attention_backward__to_copy_add_clamp_div_expand_mul_neg_slice_sum_transpose_view_5` | persistent-reduction | `_scaled, dot, product, efficient, attention, backward, _to, copy, add, clamp, div, expand, mul, neg, slice, sum, transpose, view` | 3 | 2 | 0 | 0 | 3,729 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 8 | `triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clamp_div_expand_mul_neg_slice_sum_transpose_view_4` | persistent-reduction | `_scaled, dot, product, flash, attention, backward, _to, copy, add, clamp, div, expand, mul, neg, slice, sum, transpose, view` | 3 | 2 | 0 | 0 | 3,679 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 9 | `triton_poi_fused__to_copy_add_clamp_div_expand_ge_mul_neg_scalar_tensor_slice_slice_backward_sum_transpose_view_where_5` | pointwise | `_to, copy, add, clamp, div, expand, ge, mul, neg, scalar, tensor, slice, slice, backward, sum, transpose, view, where` | 21 | 1 | 0 | 0 | 7,066 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 10 | `triton_poi_fused__to_copy_add_clamp_div_expand_ge_mul_neg_scalar_tensor_slice_slice_backward_transpose_view_where_6` | pointwise | `_to, copy, add, clamp, div, expand, ge, mul, neg, scalar, tensor, slice, slice, backward, transpose, view, where` | 6 | 1 | 0 | 0 | 4,728 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 11 | `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_17` | persistent-reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 1 | 1 | 0 | 0 | 2,759 | `cx247sakda2h7uyhqs3ofz57feoul6aqsazsygvz` |
| 12 | `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_19` | persistent-reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 1 | 1 | 0 | 0 | 2,759 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 13 | `triton_poi_fused__to_copy__unsafe_view_cat_clamp_min_clone_div_expand_mul_transpose_unsqueeze_view_6` | pointwise | `_to, copy, _unsafe, view, cat, clamp, min, clone, div, expand, mul, transpose, unsqueeze, view` | 4 | 1 | 0 | 0 | 3,684 | `c5otxjwc2jehxnylluhe5jqwwjjawvo2z636i3ky` |
| 14 | `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_16` | reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 5 | 1 | 0 | 0 | 4,457 | `cx247sakda2h7uyhqs3ofz57feoul6aqsazsygvz` |
| 15 | `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_18` | reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 5 | 1 | 0 | 0 | 4,457 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 16 | `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_squeeze_sum_transpose_view_14` | persistent-reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, squeeze, sum, transpose, view` | 1 | 1 | 0 | 0 | 2,749 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 17 | `triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_mul_transpose_unsqueeze_view_6` | pointwise | `_to, copy, _unsafe, view, clamp, min, clone, div, expand, mul, transpose, unsqueeze, view` | 3 | 1 | 0 | 0 | 2,898 | `cjxeciuqn6wez2enzrdddlwzwugk7uytitrukmu5` |
| 18 | `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_squeeze_sum_transpose_view_13` | reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, squeeze, sum, transpose, view` | 5 | 1 | 0 | 0 | 4,439 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 19 | `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_16` | persistent-reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 1 | 1 | 0 | 0 | 2,698 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 20 | `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_19` | persistent-reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 1 | 1 | 0 | 0 | 2,698 | `cx247sakda2h7uyhqs3ofz57feoul6aqsazsygvz` |
| 21 | `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_21` | persistent-reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 1 | 1 | 0 | 0 | 2,698 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 22 | `triton_per_fused__unsafe_view_clamp_clone_div_expand_mul_slice_sum_transpose_unsqueeze_view_8` | persistent-reduction | `_unsafe, view, clamp, clone, div, expand, mul, slice, sum, transpose, unsqueeze, view` | 2 | 2 | 0 | 0 | 3,492 | `cjxeciuqn6wez2enzrdddlwzwugk7uytitrukmu5` |
| 23 | `triton_poi_fused__to_copy__unsafe_view_clone_expand_mul_slice_sub_transpose_unsqueeze_view_9` | pointwise | `_to, copy, _unsafe, view, clone, expand, mul, slice, sub, transpose, unsqueeze, view` | 3 | 1 | 0 | 0 | 3,170 | `cjxeciuqn6wez2enzrdddlwzwugk7uytitrukmu5` |
| 24 | `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_15` | reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 3 | 1 | 0 | 0 | 3,789 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 25 | `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_18` | reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 3 | 1 | 0 | 0 | 3,789 | `cx247sakda2h7uyhqs3ofz57feoul6aqsazsygvz` |
| 26 | `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_20` | reduction | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 3 | 1 | 0 | 0 | 3,789 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 27 | `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_3` | persistent-reduction | `_to, copy, _unsafe, view, add, div, expand, mul, pow, sum, view` | 6 | 1 | 0 | 0 | 4,056 | `cjmxqcbe7ri6b7iwujlv7eawppthdyktlketr7hc` |
| 28 | `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_4` | persistent-reduction | `_to, copy, _unsafe, view, add, div, expand, mul, pow, sum, view` | 6 | 2 | 0 | 0 | 4,122 | `cjte2pvygapfalaktpmteaujkiity4v5twrpyoae` |
| 29 | `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_6` | persistent-reduction | `_to, copy, _unsafe, view, add, div, expand, mul, pow, sum, view` | 6 | 2 | 0 | 0 | 4,122 | `ck4fe6mhjmmt64uvqtitn7izeuwybwxxvw7ars7j` |
| 30 | `triton_poi_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_transpose_view_7` | pointwise | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, transpose, view` | 3 | 1 | 0 | 0 | 3,140 | `c5otxjwc2jehxnylluhe5jqwwjjawvo2z636i3ky` |

## Full kernel catalog (sorted by name)

| Kernel | Kind | # ops | Fused ops | Loads | Stores |
|--------|------|------:|----------|------:|-------:|
| `triton_per_fused__scaled_dot_product_efficient_attention_backward__to_copy_add_clamp_div_expand_mul_neg_slice_sum_transpose_view_5` | persistent-reduction | 18 | `_scaled, dot, product, efficient, attention, backward, _to, copy, add, clamp, div, expand, mul, neg, slice, sum, transpose, view` | 3 | 2 |
| `triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clamp_div_expand_mul_neg_slice_sum_transpose_view_4` | persistent-reduction | 18 | `_scaled, dot, product, flash, attention, backward, _to, copy, add, clamp, div, expand, mul, neg, slice, sum, transpose, view` | 3 | 2 |
| `triton_per_fused__to_copy__unsafe_view_add_clamp_min_clone_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_slice_squeeze_sum_transpose_view_where_8` | persistent-reduction | 24 | `_to, copy, _unsafe, view, add, clamp, min, clone, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, slice, squeeze, sum, transpose, view, where` | 6 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_clamp_min_clone_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_squeeze_sum_transpose_view_where_7` | persistent-reduction | 23 | `_to, copy, _unsafe, view, add, clamp, min, clone, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, squeeze, sum, transpose, view, where` | 6 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_sum_transpose_view_where_8` | persistent-reduction | 21 | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, sum, transpose, view, where` | 4 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_sum_transpose_view_where_9` | persistent-reduction | 21 | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, sum, transpose, view, where` | 4 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_3` | persistent-reduction | 11 | `_to, copy, _unsafe, view, add, div, expand, mul, pow, sum, view` | 6 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_4` | persistent-reduction | 11 | `_to, copy, _unsafe, view, add, div, expand, mul, pow, sum, view` | 6 | 2 |
| `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_6` | persistent-reduction | 11 | `_to, copy, _unsafe, view, add, div, expand, mul, pow, sum, view` | 6 | 2 |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_10` | persistent-reduction | 9 | `_to, copy, _unsafe, view, add, mean, mul, pow, rsqrt` | 3 | 2 |
| `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_2` | persistent-reduction | 9 | `_to, copy, _unsafe, view, add, mean, mul, pow, rsqrt` | 3 | 3 |
| `triton_per_fused__to_copy__unsafe_view_add_mul_sum_view_11` | persistent-reduction | 8 | `_to, copy, _unsafe, view, add, mul, sum, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_mul_sum_view_12` | persistent-reduction | 8 | `_to, copy, _unsafe, view, add, mul, sum, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_add_mul_sum_view_5` | persistent-reduction | 8 | `_to, copy, _unsafe, view, add, mul, sum, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_17` | persistent-reduction | 14 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_19` | persistent-reduction | 14 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_squeeze_sum_transpose_view_14` | persistent-reduction | 13 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, squeeze, sum, transpose, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_16` | persistent-reduction | 12 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_19` | persistent-reduction | 12 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 1 | 1 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_21` | persistent-reduction | 12 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 1 | 1 |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_18` | persistent-reduction | 9 | `_to, copy, add, div, expand, mul, pow, sum, view` | 7 | 1 |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_21` | persistent-reduction | 9 | `_to, copy, add, div, expand, mul, pow, sum, view` | 7 | 1 |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_23` | persistent-reduction | 9 | `_to, copy, add, div, expand, mul, pow, sum, view` | 7 | 1 |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_4` | persistent-reduction | 9 | `_to, copy, add, div, expand, mul, pow, sum, view` | 4 | 1 |
| `triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5` | persistent-reduction | 9 | `_to, copy, add, div, expand, mul, pow, sum, view` | 5 | 1 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | persistent-reduction | 7 | `_to, copy, add, mean, mul, pow, rsqrt` | 2 | 2 |
| `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_3` | persistent-reduction | 7 | `_to, copy, add, mean, mul, pow, rsqrt` | 2 | 2 |
| `triton_per_fused__to_copy_mul_sum_view_3` | persistent-reduction | 5 | `_to, copy, mul, sum, view` | 1 | 1 |
| `triton_per_fused__to_copy_mul_sum_view_4` | persistent-reduction | 5 | `_to, copy, mul, sum, view` | 1 | 1 |
| `triton_per_fused__unsafe_view_clamp_clone_div_expand_mul_slice_sum_transpose_unsqueeze_view_8` | persistent-reduction | 12 | `_unsafe, view, clamp, clone, div, expand, mul, slice, sum, transpose, unsqueeze, view` | 2 | 2 |
| `triton_per_fused__unsafe_view_linalg_vector_norm_transpose_view_3` | persistent-reduction | 7 | `_unsafe, view, linalg, vector, norm, transpose, view` | 1 | 1 |
| `triton_per_fused__unsafe_view_linalg_vector_norm_transpose_view_4` | persistent-reduction | 7 | `_unsafe, view, linalg, vector, norm, transpose, view` | 1 | 1 |
| `triton_per_fused_clamp_div_mul_slice_sum_8` | persistent-reduction | 5 | `clamp, div, mul, slice, sum` | 2 | 2 |
| `triton_poi_fused__to_copy_0` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_1` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_12` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_13` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_17` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_2` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_20` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_22` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_3` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_6` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy_7` | pointwise | 2 | `_to, copy` | 1 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_scalar_tensor_transpose_view_where_10` | pointwise | 19 | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, scalar, tensor, transpose, view, where` | 1 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_scalar_tensor_transpose_view_where_9` | pointwise | 19 | `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, scalar, tensor, transpose, view, where` | 1 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_cat_clamp_min_clone_div_expand_mul_transpose_unsqueeze_view_6` | pointwise | 14 | `_to, copy, _unsafe, view, cat, clamp, min, clone, div, expand, mul, transpose, unsqueeze, view` | 4 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_clamp_min_clone_div_expand_mul_transpose_unsqueeze_view_6` | pointwise | 13 | `_to, copy, _unsafe, view, clamp, min, clone, div, expand, mul, transpose, unsqueeze, view` | 3 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_transpose_view_7` | pointwise | 11 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, transpose, view` | 3 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_clone_expand_mul_slice_sub_transpose_unsqueeze_view_9` | pointwise | 12 | `_to, copy, _unsafe, view, clone, expand, mul, slice, sub, transpose, unsqueeze, view` | 3 | 1 |
| `triton_poi_fused__to_copy__unsafe_view_clone_squeeze_transpose_view_6` | pointwise | 8 | `_to, copy, _unsafe, view, clone, squeeze, transpose, view` | 1 | 1 |
| `triton_poi_fused__to_copy_add_clamp_div_expand_ge_mul_neg_scalar_tensor_slice_slice_backward_sum_transpose_view_where_5` | pointwise | 18 | `_to, copy, add, clamp, div, expand, ge, mul, neg, scalar, tensor, slice, slice, backward, sum, transpose, view, where` | 21 | 1 |
| `triton_poi_fused__to_copy_add_clamp_div_expand_ge_mul_neg_scalar_tensor_slice_slice_backward_transpose_view_where_6` | pointwise | 17 | `_to, copy, add, clamp, div, expand, ge, mul, neg, scalar, tensor, slice, slice, backward, transpose, view, where` | 6 | 1 |
| `triton_poi_fused__to_copy_add_clone_slice_squeeze_sum_transpose_view_7` | pointwise | 9 | `_to, copy, add, clone, slice, squeeze, sum, transpose, view` | 6 | 1 |
| `triton_poi_fused__to_copy_add_slice_sum_view_14` | pointwise | 6 | `_to, copy, add, slice, sum, view` | 6 | 1 |
| `triton_poi_fused__to_copy_add_slice_sum_view_16` | pointwise | 6 | `_to, copy, add, slice, sum, view` | 6 | 1 |
| `triton_poi_fused__to_copy_mul_slice_sub_transpose_view_9` | pointwise | 7 | `_to, copy, mul, slice, sub, transpose, view` | 3 | 1 |
| `triton_poi_fused__to_copy_mul_transpose_1` | pointwise | 4 | `_to, copy, mul, transpose` | 2 | 1 |
| `triton_poi_fused__to_copy_mul_transpose_view_7` | pointwise | 5 | `_to, copy, mul, transpose, view` | 3 | 2 |
| `triton_poi_fused__to_copy_mul_transpose_view_8` | pointwise | 5 | `_to, copy, mul, transpose, view` | 3 | 2 |
| `triton_poi_fused__to_copy_slice_sum_view_15` | pointwise | 5 | `_to, copy, slice, sum, view` | 3 | 1 |
| `triton_poi_fused__to_copy_slice_sum_view_17` | pointwise | 5 | `_to, copy, slice, sum, view` | 3 | 1 |
| `triton_poi_fused__to_copy_t_0` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_t_1` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_t_11` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_t_12` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_t_2` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_t_3` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_t_4` | pointwise | 3 | `_to, copy, t` | 1 | 1 |
| `triton_poi_fused__to_copy_view_0` | pointwise | 3 | `_to, copy, view` | 1 | 1 |
| `triton_poi_fused__unsafe_view_add_14` | pointwise | 3 | `_unsafe, view, add` | 3 | 1 |
| `triton_poi_fused__unsafe_view_add_6` | pointwise | 3 | `_unsafe, view, add` | 2 | 1 |
| `triton_poi_fused__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_5` | pointwise | 8 | `_unsafe, view, cat, clone, expand, transpose, unsqueeze, view` | 2 | 1 |
| `triton_poi_fused__unsafe_view_cat_mul_silu_silu_backward_split_view_1` | pointwise | 9 | `_unsafe, view, cat, mul, silu, silu, backward, split, view` | 5 | 1 |
| `triton_poi_fused__unsafe_view_cat_mul_silu_silu_backward_split_view_2` | pointwise | 9 | `_unsafe, view, cat, mul, silu, silu, backward, split, view` | 5 | 1 |
| `triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_5` | pointwise | 7 | `_unsafe, view, clone, expand, transpose, unsqueeze, view` | 1 | 1 |
| `triton_poi_fused__unsafe_view_mul_silu_split_13` | pointwise | 5 | `_unsafe, view, mul, silu, split` | 2 | 1 |
| `triton_poi_fused__unsafe_view_mul_silu_split_5` | pointwise | 5 | `_unsafe, view, mul, silu, split` | 2 | 1 |
| `triton_poi_fused_cat_view_0` | pointwise | 2 | `cat, view` | 3 | 1 |
| `triton_poi_fused_clone_select_0` | pointwise | 2 | `clone, select` | 1 | 1 |
| `triton_poi_fused_clone_select_1` | pointwise | 2 | `clone, select` | 1 | 1 |
| `triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_10` | reduction | 8 | `_to, copy, _unsafe, view, add, mul, sum, view` | 8 | 2 |
| `triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_11` | reduction | 8 | `_to, copy, _unsafe, view, add, mul, sum, view` | 8 | 2 |
| `triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_4` | reduction | 8 | `_to, copy, _unsafe, view, add, mul, sum, view` | 4 | 1 |
| `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_16` | reduction | 14 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 5 | 1 |
| `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_slice_squeeze_sum_transpose_view_18` | reduction | 14 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, slice, squeeze, sum, transpose, view` | 5 | 1 |
| `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_squeeze_sum_transpose_view_13` | reduction | 13 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, squeeze, sum, transpose, view` | 5 | 1 |
| `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_15` | reduction | 12 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 3 | 1 |
| `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_18` | reduction | 12 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 3 | 1 |
| `triton_red_fused__to_copy__unsafe_view_clamp_min_div_expand_mul_sum_transpose_view_20` | reduction | 12 | `_to, copy, _unsafe, view, clamp, min, div, expand, mul, sum, transpose, view` | 3 | 1 |
| `triton_red_fused__to_copy_mul_sum_view_2` | reduction | 5 | `_to, copy, mul, sum, view` | 3 | 1 |
| `triton_red_fused__to_copy_mul_sum_view_3` | reduction | 5 | `_to, copy, mul, sum, view` | 3 | 1 |

## Most frequently fused ops (per-op frequency across all kernels)

| Op | # kernels fusing it |
|----|-------------------:|
| `view` | 107 |
| `_to` | 77 |
| `copy` | 77 |
| `mul` | 58 |
| `_unsafe` | 45 |
| `sum` | 44 |
| `expand` | 37 |
| `transpose` | 37 |
| `div` | 35 |
| `add` | 33 |
| `clamp` | 27 |
| `min` | 21 |
| `slice` | 20 |
| `clone` | 12 |
| `pow` | 12 |
| `squeeze` | 10 |
| `ge` | 8 |
| `neg` | 8 |
| `scalar` | 8 |
| `tensor` | 8 |
| `where` | 8 |
| `t` | 7 |
| `eq` | 6 |
| `masked` | 6 |
| `fill` | 6 |
| `backward` | 6 |
| `unsqueeze` | 6 |
| `silu` | 6 |
| `cat` | 5 |
| `mean` | 4 |
| `rsqrt` | 4 |
| `split` | 4 |
| `_scaled` | 2 |
| `dot` | 2 |
| `product` | 2 |
| `attention` | 2 |
| `sub` | 2 |
| `linalg` | 2 |
| `vector` | 2 |
| `norm` | 2 |

## Analysis notes

**How to read this catalog:**
- Each `triton_<kind>_fused_<op1>_<op2>_..._<seq>` kernel represents one
  Inductor-generated fused GPU kernel. The op list is the set of ATen/prims
  ops Inductor determined can run as a single kernel — usually a chain of
  pointwise ops, or a pointwise chain terminating in a reduction.
- `kind=poi` = pointwise only. `per` = persistent reduction (small inner dim).
  `red` = reduction with loop. `tem` = matmul template with epilogue fused.

**Phase 2 implication:** every op listed here is ALREADY FUSED by Inductor.
Re-implementing it as a custom HIP kernel buys nothing unless the HIP kernel
beats Inductor's triton output on this specific shape/layout. Do not re-fuse
patterns that appear in this catalog without isolated benchmark proof.

**Kernels to investigate (non-trivial fusions worth understanding):**
1. `triton_per_fused__to_copy__unsafe_view_add_clamp_min_clone_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_slice_squeeze_sum_transpose_view_where_8` — fuses 24 ops: `_to, copy, _unsafe, view, add, clamp, min, clone, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, slice, squeeze, sum, transpose, view, where`
2. `triton_per_fused__to_copy__unsafe_view_add_clamp_min_clone_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_squeeze_sum_transpose_view_where_7` — fuses 23 ops: `_to, copy, _unsafe, view, add, clamp, min, clone, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, squeeze, sum, transpose, view, where`
3. `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_sum_transpose_view_where_8` — fuses 21 ops: `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, sum, transpose, view, where`
4. `triton_per_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_neg_scalar_tensor_sum_transpose_view_where_9` — fuses 21 ops: `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, neg, scalar, tensor, sum, transpose, view, where`
5. `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_scalar_tensor_transpose_view_where_10` — fuses 19 ops: `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, scalar, tensor, transpose, view, where`
6. `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_eq_expand_ge_masked_fill_mul_scalar_tensor_transpose_view_where_9` — fuses 19 ops: `_to, copy, _unsafe, view, add, clamp, min, div, eq, expand, ge, masked, fill, mul, scalar, tensor, transpose, view, where`
7. `triton_per_fused__scaled_dot_product_efficient_attention_backward__to_copy_add_clamp_div_expand_mul_neg_slice_sum_transpose_view_5` — fuses 18 ops: `_scaled, dot, product, efficient, attention, backward, _to, copy, add, clamp, div, expand, mul, neg, slice, sum, transpose, view`
8. `triton_per_fused__scaled_dot_product_flash_attention_backward__to_copy_add_clamp_div_expand_mul_neg_slice_sum_transpose_view_4` — fuses 18 ops: `_scaled, dot, product, flash, attention, backward, _to, copy, add, clamp, div, expand, mul, neg, slice, sum, transpose, view`
9. `triton_poi_fused__to_copy_add_clamp_div_expand_ge_mul_neg_scalar_tensor_slice_slice_backward_sum_transpose_view_where_5` — fuses 18 ops: `_to, copy, add, clamp, div, expand, ge, mul, neg, scalar, tensor, slice, slice, backward, sum, transpose, view, where`
10. `triton_poi_fused__to_copy_add_clamp_div_expand_ge_mul_neg_scalar_tensor_slice_slice_backward_transpose_view_where_6` — fuses 17 ops: `_to, copy, add, clamp, div, expand, ge, mul, neg, scalar, tensor, slice, slice, backward, transpose, view, where`