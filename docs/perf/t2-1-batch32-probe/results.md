# T-2.1 batch=32/accum=4 probe

| Config | batch | accum | tok/s | peak_mem_gb | scale | status |
|---|---:|---:|---:|---:|---:|---|
| baseline | 16 | 8 | 31563 | 13.5 | 1.0e+03 | PASS |
| batch32_plain | 32 | 4 | 13476 | 24.6 | 1.0e+03 | PASS |
| batch32_ddp_tune | 32 | 4 | 31504 | 24.6 | 1.0e+03 | PASS |
| batch32_sync_clean | 32 | 4 | 31521 | 25.6 | 1.0e+03 | PASS |
