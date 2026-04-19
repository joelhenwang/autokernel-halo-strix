#!/usr/bin/env python3
"""Prepare function-calling data for tool-use SFT.

Converts Locutusque/function-calling-chatml into ChatML format with
<tool_call>/<tool_call> markers, compatible with halo_training's
SFTDataset(format="chatml") and domain-sft tokenizer.

The input format has roles: system, human, gpt, function-call, function-response
We map to ChatML roles: system, user, assistant (with <tool_call> wrapping), tool

Usage:
    python scripts/prepare_tool_data.py --output datasets/swe_prepared/tool_use.jsonl
    python scripts/prepare_tool_data.py --output datasets/swe_prepared/tool_use.jsonl --max-tokens 1536
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    def count_tokens(text): return len(_enc.encode_ordinary(text))
except ImportError:
    def count_tokens(text): return len(text) // 4


def convert_conversation(convs, max_tokens=1536):
    """Convert one conversation from Glaive format to ChatML with tool markers."""
    messages = []

    for turn in convs:
        role = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", ""))

        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})
        elif role == "function-call":
            messages.append({"role": "assistant", "content": f"<tool_call>\n{content}\n</tool_call>"})
        elif role == "function-response":
            messages.append({"role": "tool", "content": content})

    if not messages or len(messages) < 2:
        return None

    # Check token count
    total = sum(count_tokens(m["content"]) for m in messages)
    if total > max_tokens:
        return None

    # Must have at least one assistant response
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_assistant:
        return None

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Prepare tool-use data for SFT")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--input", default="datasets/tool-calling-raw/function_calling.parquet",
                        help="Input parquet file")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Max tokens per conversation")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit output size")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    print(f"Reading {args.input}...")
    table = pq.read_table(args.input)
    print(f"Total rows: {len(table)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    total_tokens = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(table)):
            convs = table.column("conversations")[i].as_py()

            example = convert_conversation(convs, max_tokens=args.max_tokens)

            if example is None:
                stats["filtered"] += 1
                continue

            # Classify the example
            has_tool_call = any("<tool_call>" in m["content"] for m in example["messages"])
            has_tool_response = any(m["role"] == "tool" for m in example["messages"])

            if has_tool_call and has_tool_response:
                stats["with_tool_use"] += 1
            elif has_tool_call:
                stats["call_only"] += 1
            else:
                stats["no_tool"] += 1

            tokens = sum(count_tokens(m["content"]) for m in example["messages"])
            total_tokens += tokens
            stats["total"] += 1

            f.write(json.dumps(example, ensure_ascii=False) + "\n")

            if args.max_examples and stats["total"] >= args.max_examples:
                break

    print(f"\nTool-Use Data Preparation Summary")
    print(f"{'='*50}")
    print(f"Total examples: {stats['total']}")
    print(f"  With tool use (call+response): {stats['with_tool_use']}")
    print(f"  Call only (no response): {stats['call_only']}")
    print(f"  No tool (refusal/chat): {stats['no_tool']}")
    print(f"Filtered (too long): {stats['filtered']}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg tokens/example: {total_tokens // max(stats['total'], 1)}")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
