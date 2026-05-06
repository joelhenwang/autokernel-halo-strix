"""Compare Machine A vs Machine B scorecard JSONs for parity."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

def _num(x):
    if x is None:
        return None
    return float(x)

def pct_diff(av, bv):
    if av is None or bv is None:
        return None
    if av == 0:
        return None
    return 100.0 * (bv - av) / av

print(f"Machine A: {a['eval_machine']} ({a['checkpoint_name']})")
print(f"Machine B: {b['eval_machine']} ({b['checkpoint_name']})")
print()

# Per-domain BPB
print("Per-domain BPB:")
for dom in ["wikitext_val", "gpt_small_val", "stem_crawl_val", "dolma_val"]:
    av = a["per_domain_bpb"].get(dom)
    bv = b["per_domain_bpb"].get(dom)
    delta = pct_diff(av, bv)
    if delta is None:
        print(f"  {dom:<18}  A={av}  B={bv}")
    else:
        print(f"  {dom:<18}  A={av:.4f}  B={bv:.4f}  delta={delta:+.2f}%")

# Sampling
print("\nSampling:")
for key in ["distinct_2", "self_ppl", "avg_length"]:
    av = a["sampling"].get(key)
    bv = b["sampling"].get(key)
    delta = pct_diff(av, bv)
    print(f"  {key:<18}  A={av}  B={bv}  delta={delta:+.2f}%" if delta is not None
          else f"  {key:<18}  A={av}  B={bv}")

# Inference profile
print("\nInference:")
for key in ["tok_s_seq512_bs1", "peak_mem_gb_seq512", "tok_s_seq1024_bs1", "peak_mem_gb_seq1024"]:
    av = a["inference_profile"].get(key)
    bv = b["inference_profile"].get(key)
    delta = pct_diff(av, bv)
    print(f"  {key:<22}  A={av}  B={bv}  delta={delta:+.2f}%" if delta is not None
          else f"  {key:<22}  A={av}  B={bv}")

# Sample pack hash
print("\nSample pack:")
print(f"  hash_A={a['sample_pack']['output_hash']}")
print(f"  hash_B={b['sample_pack']['output_hash']}")
print(f"  identical: {a['sample_pack']['output_hash'] == b['sample_pack']['output_hash']}")
