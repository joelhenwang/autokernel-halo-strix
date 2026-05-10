#!/bin/bash
# T-4: Compiled autograd gated smoke per v3 section 6 experiment card E4.
#
# Short (200-500 step) smoke tests to determine whether compiled autograd
# has real net DDP value on our stack. Gate: continue to deeper investment
# only if ≥2.5-3% net DDP tok/s improvement AND no overlap regression AND
# no recompile churn.
#
# Risk: PyTorch DDP docs explicitly warn whole-backward compilation may
# regress allreduce overlap. We must measure DDP overlap via T-0.6 trace.

set -e

cd ~/Desktop/ai_lab/autokernel-halo-strix

mkdir -p docs/perf/t4-compiled-autograd-smoke

RESULTS=docs/perf/t4-compiled-autograd-smoke/results.md
echo "# T-4 Compiled Autograd Gated Smoke" > "$RESULTS"
echo "" >> "$RESULTS"
echo "Gate: continue only if >=2.5-3% net DDP tok/s + no overlap regression." >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| Config | tok/s | allreduce_count/step | overlap_est | recompiles | status |" >> "$RESULTS"
echo "|---|---:|---:|---:|---:|---|" >> "$RESULTS"

run_config() {
  local name="$1"
  local extra_flags="$2"
  local ckpt="checkpoints/t4-${name}"
  rm -rf "$ckpt"
  mkdir -p "$ckpt"
  ssh joelwang-ai-1@10.77.0.2 "mkdir -p ~/Desktop/comfyui-rocm7.12/autokernel-halo-strix/$ckpt"

  echo "=== ${name}: ${extra_flags} ==="

  MODEL=models/odin_flat.py \
  CLASS=OdinFlat \
  DATASET=datasets/dolma-10b-odin32k.bin \
  EPOCHS=1 \
  LR=8e-4 BLOCK=512 BATCH=16 ACCUM=8 \
  WARMUP_STEPS=50 CHECKPOINT_INTERVAL=9999 MAX_GRAD_NORM=1.0 \
  NUM_WORKERS=12 \
  CKPT_DIR="$ckpt" \
  EXTRA_FLAGS="--max-steps 300 --imu1-groups --normuon --lr-2d 5e-3 --lr-1d 8e-4 --intra-doc-mask --value-residuals --head-gating --z-loss 1e-4 --attn-softcap 50.0 --mup --mup-base-width 256 --spectra-post --spectra-clip-norm 1.0 ${extra_flags}" \
  bash scripts/launch_ddp.sh >/dev/null 2>&1

  # Wait for completion (timeout 15 min per config)
  local SECS=0
  while pgrep -f "train_ddp.py" > /dev/null; do
    sleep 5
    SECS=$((SECS + 5))
    if [ $SECS -gt 900 ]; then
      echo "TIMEOUT at 15min for ${name}"
      pkill -f "train_ddp.py" || true
      ssh joelwang-ai-1@10.77.0.2 "pkill -f 'train_ddp.py'" || true
      break
    fi
  done

  # Extract metrics
  if [ -f "$ckpt/rank0.log" ]; then
    local TOK_S=$(grep -oE 'tok/s=[0-9,]+' "$ckpt/rank0.log" | tail -10 \
                  | sed 's/tok\/s=//;s/,//g' | sort -n \
                  | awk '{a[NR]=$1} END{if(NR>0) print a[int(NR/2)]; else print "FAIL"}')
    # DDP trace JSONL: last allreduce_count + overlap_ratio_estimate
    local AR_COUNT=0
    local OVERLAP="0.0"
    if [ -f "$ckpt/ddp_trace_rank0.jsonl" ]; then
      AR_COUNT=$(tail -1 "$ckpt/ddp_trace_rank0.jsonl" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('allreduce_count',0))" 2>/dev/null || echo 0)
      OVERLAP=$(tail -1 "$ckpt/ddp_trace_rank0.jsonl" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"{d.get('overlap_ratio_estimate',0):.3f}\")" 2>/dev/null || echo 0)
    fi
    local RECOMP=$(grep -c "Recompiling function" "$ckpt/rank0.log" 2>/dev/null || echo 0)
    local STATUS="PASS"
    if grep -q "OOM\|nonfinite\|RuntimeError\|Error:" "$ckpt/rank0.log" 2>/dev/null; then
      STATUS="FAIL"
    fi
    echo "| ${name} | ${TOK_S} | ${AR_COUNT} | ${OVERLAP} | ${RECOMP} | ${STATUS} |" >> "$RESULTS"
    echo "Result: tok/s=${TOK_S} ar_count=${AR_COUNT} overlap=${OVERLAP} recompiles=${RECOMP} status=${STATUS}"
  else
    echo "| ${name} | NO_LOG | - | - | - | FAIL |" >> "$RESULTS"
  fi
}

# E4-A baseline: no compiled autograd
run_config "baseline" ""

# E4-B + compiled autograd
# Note: CA activation uses --ak-compiled-autograd flag which sets
# torch._dynamo.config.compiled_autograd=True inside train_ddp.py
# (to be wired). For now use env var:
CONFIG_OVERRIDE="AUTOKERNEL_COMPILED_AUTOGRAD=1"

# E4-B compiled autograd + baseline
run_config "compiled_autograd" "--ak-compiled-autograd"

# E4-C compiled autograd + fused zloss
run_config "ca_plus_fused_zloss" "--ak-compiled-autograd --use-fused-zloss --ak-loss-zloss"

# E4-D compiled autograd + DDP tune
run_config "ca_plus_ddp_tune" "--ak-compiled-autograd --ak-ddp-tune"

echo "=== T-4 DONE ==="
cat "$RESULTS"
