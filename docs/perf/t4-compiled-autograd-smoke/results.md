# T-4 Compiled Autograd Gated Smoke

Gate: continue only if >=2.5-3% net DDP tok/s + no overlap regression.

| Config | tok/s | allreduce_count/step | overlap_est | recompiles | status |
|---|---:|---:|---:|---:|---|
| baseline | 31630 | 1 | 0.000 | 0
0 | PASS |
| compiled_autograd | 31444 | 1 | 0.000 | 0
0 | PASS |
| ca_plus_fused_zloss | 33605 | 1 | 0.000 | 0
0 | PASS |
| ca_plus_ddp_tune | 31459 | 1 | 0.000 | 0
0 | PASS |
