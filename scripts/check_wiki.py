import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
d = np.memmap("datasets/wikitext-103-raw.bin", dtype=np.uint16, mode="r")
print(f"wikitext-103-raw.bin: {len(d):,} tokens, max_id={d.max()}, min_id={d.min()}")
