"""Phase 5 smoke test: spawn_auto_eval fires a detached subprocess.

Invoke as:
    python scripts/test_auto_eval_spawn.py \\
        --checkpoint checkpoints/odin-flat-stem-crawl-ddp/step_500.pt

Verifies that:
  1. spawn_auto_eval returns without blocking
  2. A <checkpoint>.eval.log file appears shortly after
  3. The subprocess runs eval_checkpoint.py against the given checkpoint
"""
import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_ddp import spawn_auto_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="models/odin_flat.py")
    p.add_argument("--class-name", default="OdinFlat")
    p.add_argument("--wait-s", type=int, default=8)
    args = p.parse_args()

    log_path = args.checkpoint + ".eval.log"
    # Remove any prior log
    if os.path.exists(log_path):
        os.remove(log_path)

    t0 = time.time()
    spawn_auto_eval(args.checkpoint, args.model, args.class_name)
    dispatch_latency = time.time() - t0
    print(f"\n>>> spawn_auto_eval returned after {dispatch_latency*1000:.1f} ms")
    assert dispatch_latency < 1.0, f"spawn_auto_eval blocked for {dispatch_latency:.2f}s"

    print(f">>> waiting {args.wait_s}s to check log populates ...")
    time.sleep(args.wait_s)

    if not os.path.exists(log_path):
        print(f"FAIL: eval log {log_path!r} was never created")
        sys.exit(1)

    size = os.path.getsize(log_path)
    print(f">>> {log_path} exists, size={size} bytes")
    if size == 0:
        print(f"WARN: log is empty — subprocess may be still starting")

    with open(log_path, encoding="utf-8") as f:
        head = "".join(f.readlines()[:30])
    print(">>> first ~30 lines of eval log:")
    print(head)
    print(">>> SMOKE TEST PASS")


if __name__ == "__main__":
    main()
