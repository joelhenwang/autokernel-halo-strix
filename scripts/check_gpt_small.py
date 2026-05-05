import numpy as np
d = np.memmap("datasets/gpt-training-small.bin", dtype=np.uint16, mode="r")
print(f"gpt-training-small.bin: {len(d):,} tokens, max_id={d.max()}, min_id={d.min()}")
