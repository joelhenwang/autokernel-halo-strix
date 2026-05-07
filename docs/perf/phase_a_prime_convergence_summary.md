# Phase A' convergence diagnostics

Retroactive scoring of existing checkpoints against `halo_training/eval/convergence_stats`.

Gate thresholds (on the last iter-pair's fraction of tokens with cos > 0.95):

  - **GREENLIGHT** Phase B: frac_high_cos >= 0.95
  - **PARTIAL** (aux loss only, no adaptive iter): 0.85 <= frac_high_cos < 0.95
  - **KILL** Phase B: frac_high_cos < 0.85

## Results

| Checkpoint | Step | Model | Looped | iter_k_cos_to_final | frac_high (tau=0.95) | Gate |
|---|---:|---|:---:|---|---|:---:|
| `sprint3-iter2b-lr2_5e3/step_400.pt` | 400 | OdinHalo | yes | 0.281, 0.582 | 0.000, 0.002 | **KILL** |
| `sprint3-s1_3-lr2_5e3-700/step_200.pt` | 200 | OdinHalo | yes | 0.331, 0.613 | 0.000, 0.001 | **KILL** |
| `sprint3-s1_3-lr2_5e3-700/step_400.pt` | 400 | OdinHalo | yes | 0.270, 0.639 | 0.000, 0.002 | **KILL** |
| `sprint3-s1_3-lr2_5e3-700/step_600.pt` | 600 | OdinHalo | yes | 0.040, 0.522 | 0.000, 0.002 | **KILL** |
| `sprint3-s1_3-lr2_5e3-700/step_700.pt` | 700 | OdinHalo | yes | 0.084, 0.412 | 0.000, 0.001 | **KILL** |
| `sprint3-s1_3b-lr2e3-700/step_200.pt` | 200 | OdinHalo | yes | 0.285, 0.558 | 0.000, 0.000 | **KILL** |
| `sprint3-s1_3b-lr2e3-700/step_400.pt` | 400 | OdinHalo | yes | 0.310, 0.571 | 0.000, 0.001 | **KILL** |
| `sprint3-s1_3b-lr2e3-700/step_600.pt` | 600 | OdinHalo | yes | 0.166, 0.472 | 0.000, 0.002 | **KILL** |
| `sprint3-s1_3b-lr2e3-700/step_700.pt` | 700 | OdinHalo | yes | 0.103, 0.418 | 0.000, 0.001 | **KILL** |
| `sprint3-s1_5-odinflat-dolma/step_200.pt` | 200 | OdinFlat | no | - | - | **N/A** |
| `sprint3-s1_5-odinflat-dolma/step_400.pt` | 400 | OdinFlat | no | - | - | **N/A** |

## Gate decision (aggregate)

- 0 GREENLIGHT, 0 PARTIAL, 9 KILL (out of 9 looped checkpoints).
- Aggregate recommendation: **KILL** (majority below 0.85)

## Per-layer convergence (flat or last-iter view)

### sprint3-iter2b-lr2_5e3/step_400.pt

per_layer_cos_to_final: 0.485, 0.543, 0.577, 0.963, 0.986, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.847, 0.993, 1.000
inter_layer_transition_cos: 0.939, 0.953, 0.594, 0.985, 0.986
effective_rank_final: 2.8

### sprint3-s1_3-lr2_5e3-700/step_200.pt

per_layer_cos_to_final: 0.563, 0.654, 0.711, 0.956, 0.985, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.753, 0.995, 1.000
inter_layer_transition_cos: 0.914, 0.950, 0.733, 0.983, 0.985
effective_rank_final: 3.1

### sprint3-s1_3-lr2_5e3-700/step_400.pt

per_layer_cos_to_final: 0.499, 0.558, 0.593, 0.964, 0.986, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.863, 1.000, 1.000
inter_layer_transition_cos: 0.936, 0.949, 0.602, 0.987, 0.986
effective_rank_final: 2.9

### sprint3-s1_3-lr2_5e3-700/step_600.pt

per_layer_cos_to_final: 0.289, 0.326, 0.345, 0.942, 0.976, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.397, 0.983, 1.000
inter_layer_transition_cos: 0.938, 0.933, 0.383, 0.979, 0.976
effective_rank_final: 2.0

### sprint3-s1_3-lr2_5e3-700/step_700.pt

per_layer_cos_to_final: 0.265, 0.305, 0.332, 0.959, 0.981, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.819, 1.000, 1.000
inter_layer_transition_cos: 0.903, 0.906, 0.364, 0.988, 0.981
effective_rank_final: 2.2

### sprint3-s1_3b-lr2e3-700/step_200.pt

per_layer_cos_to_final: 0.595, 0.666, 0.712, 0.953, 0.982, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.693, 0.971, 1.000
inter_layer_transition_cos: 0.932, 0.957, 0.742, 0.982, 0.982
effective_rank_final: 2.8

### sprint3-s1_3b-lr2e3-700/step_400.pt

per_layer_cos_to_final: 0.549, 0.593, 0.623, 0.963, 0.986, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.883, 1.000, 1.000
inter_layer_transition_cos: 0.953, 0.961, 0.636, 0.986, 0.986
effective_rank_final: 2.8

### sprint3-s1_3b-lr2e3-700/step_600.pt

per_layer_cos_to_final: 0.446, 0.477, 0.496, 0.955, 0.984, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.703, 0.990, 1.000
inter_layer_transition_cos: 0.959, 0.957, 0.518, 0.980, 0.984
effective_rank_final: 2.6

### sprint3-s1_3b-lr2e3-700/step_700.pt

per_layer_cos_to_final: 0.375, 0.404, 0.420, 0.947, 0.978, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.502, 0.978, 1.000
inter_layer_transition_cos: 0.953, 0.941, 0.472, 0.981, 0.978
effective_rank_final: 2.4

### sprint3-s1_5-odinflat-dolma/step_200.pt

per_layer_cos_to_final: 0.465, 0.511, 0.540, 0.563, 0.581, 0.594, 0.850, 0.885, 0.908, 0.922, 0.932, 0.939, 0.944, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.006, 0.025, 0.111, 0.261, 0.395, 1.000
inter_layer_transition_cos: 0.951, 0.978, 0.987, 0.991, 0.993, 0.769, 0.991, 0.995, 0.996, 0.997, 0.998, 0.998, 0.944
effective_rank_final: 2.8

### sprint3-s1_5-odinflat-dolma/step_400.pt

per_layer_cos_to_final: 0.494, 0.530, 0.552, 0.568, 0.581, 0.590, 0.830, 0.868, 0.894, 0.913, 0.927, 0.937, 0.943, 1.000
per_layer_frac_high_cos: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.010, 0.016, 0.021, 0.041, 0.140, 0.307, 1.000
inter_layer_transition_cos: 0.964, 0.985, 0.990, 0.993, 0.995, 0.733, 0.989, 0.993, 0.995, 0.996, 0.997, 0.997, 0.943
effective_rank_final: 3.1
