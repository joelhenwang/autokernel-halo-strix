#!/usr/bin/env python3
"""Prepare Nemotron-Cascade-2 SWE data for small-model SFT.

Streams swe_agentless.jsonl from HuggingFace (23.9 GB), extracts sub-tasks
that fit within a 2048-token context, and outputs ChatML-format JSONL files
compatible with halo_training's SFTDataset(format="chatml").

Sub-tasks extracted:
  1. code_repair   — Given issue + buggy code, produce SEARCH/REPLACE patch
  2. bug_explain   — Given issue + code, produce chain-of-thought reasoning
  3. localize      — Given issue text only, predict buggy file/function

Usage:
  python scripts/prepare_swe_data.py --output-dir datasets/swe_prepared
  python scripts/prepare_swe_data.py --output-dir datasets/swe_prepared --max-rows 1000 --dry-run
  python scripts/prepare_swe_data.py --output-dir datasets/swe_prepared --task code_repair --max-tokens 1536
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except ImportError:
    _enc = None
    print("WARNING: tiktoken not available, using char/4 approximation for token counts")


def count_tokens(text: str) -> int:
    if _enc:
        return len(_enc.encode_ordinary(text))
    return len(text) // 4


# ---------------------------------------------------------------------------
# Parsers for the agentless response format
# ---------------------------------------------------------------------------

def extract_think(assistant_text: str) -> Tuple[str, str]:
    """Split assistant response into (think_content, solution_content)."""
    think = ""
    solution = assistant_text

    think_match = re.search(r'<think>(.*?)</think>', assistant_text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
        solution = assistant_text[think_match.end():].strip()

    return think, solution


def extract_patches(solution_text: str) -> List[Dict[str, str]]:
    """Extract SEARCH/REPLACE patches from the solution.

    Returns list of {file, search, replace} dicts.
    """
    patches = []

    # Pattern: ```python\n### filepath\n<<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE\n```
    blocks = re.split(r'```(?:python|diff)?\s*\n', solution_text)

    for block in blocks:
        block = block.rstrip('`').strip()
        if '<<<<<<< SEARCH' not in block:
            continue

        # Extract filename from ### header
        file_match = re.match(r'###\s*(.+?)(?:\n|$)', block)
        filename = file_match.group(1).strip() if file_match else "unknown"

        # Extract all SEARCH/REPLACE pairs in this block
        pairs = re.findall(
            r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
            block, re.DOTALL
        )
        for search, replace in pairs:
            patches.append({
                "file": filename,
                "search": search.strip(),
                "replace": replace.strip(),
            })

    return patches


def extract_issue_and_code(user_text: str) -> Tuple[str, str, List[Dict[str, str]]]:
    """Parse user message into (issue_text, raw_code_context, [{file, code}])."""
    issue = ""
    code_context = ""
    code_files = []

    # Extract issue
    issue_match = re.search(
        r'--- BEGIN ISSUE ---\s*\n(.*?)\n\s*--- END ISSUE ---',
        user_text, re.DOTALL
    )
    if issue_match:
        issue = issue_match.group(1).strip()

    # Extract code files
    file_blocks = re.findall(
        r'--- BEGIN FILE ---\s*\n```\s*\n###\s*(.+?)\n(.*?)```\s*\n--- END FILE ---',
        user_text, re.DOTALL
    )
    for filename, code in file_blocks:
        code_files.append({"file": filename.strip(), "code": code.strip()})
        code_context += f"### {filename.strip()}\n{code.strip()}\n\n"

    return issue, code_context.strip(), code_files


def find_buggy_file_snippet(code_files: List[Dict], patches: List[Dict],
                             max_tokens: int = 800) -> Optional[str]:
    """Find and extract just the relevant portion of the buggy file.

    Instead of including ALL code files (which can be 100K+ chars),
    find the file that matches the patch and extract a window around
    the buggy region.
    """
    if not patches or not code_files:
        return None

    patch = patches[0]  # Use first patch
    target_file = patch["file"]
    search_text = patch["search"]

    # Find matching code file
    matching = None
    for cf in code_files:
        if cf["file"] == target_file or cf["file"].endswith(target_file):
            matching = cf
            break

    if not matching:
        # Fallback: find file containing the search text
        for cf in code_files:
            if search_text[:50] in cf["code"]:
                matching = cf
                break

    if not matching:
        return None

    code = matching["code"]
    lines = code.split('\n')

    # Find the search text location
    search_start = code.find(search_text[:80])
    if search_start == -1:
        # Return truncated file if search text not found exactly
        truncated = '\n'.join(lines[:60])
        if count_tokens(truncated) <= max_tokens:
            return f"### {matching['file']}\n{truncated}"
        return None

    # Find line number of the match
    line_num = code[:search_start].count('\n')

    # Extract a window: 20 lines before, the match, and 20 lines after
    match_lines = search_text.count('\n') + 1
    start = max(0, line_num - 20)
    end = min(len(lines), line_num + match_lines + 20)

    snippet = '\n'.join(lines[start:end])

    # If still too long, shrink the window
    while count_tokens(snippet) > max_tokens and (end - start) > match_lines + 4:
        start = min(start + 2, line_num - 2)
        end = max(end - 2, line_num + match_lines + 2)
        snippet = '\n'.join(lines[start:end])

    prefix = f"# Lines {start+1}-{end} of {matching['file']}\n" if start > 0 else f"### {matching['file']}\n"
    return prefix + snippet


# ---------------------------------------------------------------------------
# Sub-task builders
# ---------------------------------------------------------------------------

SWE_SYSTEM_PROMPT = "You are an expert software engineer. Given a bug report and relevant code, produce a minimal fix."

CODE_REPAIR_SYSTEM = (
    "You are an expert software engineer. "
    "Given a GitHub issue and the relevant code snippet, produce a SEARCH/REPLACE patch that fixes the bug. "
    "Output ONLY the patch in this format:\n"
    "```\n<<<<<<< SEARCH\nold code\n=======\nnew code\n>>>>>>> REPLACE\n```"
)

BUG_EXPLAIN_SYSTEM = (
    "You are an expert software engineer. "
    "Given a GitHub issue and code, explain the root cause of the bug and how to fix it. "
    "Be concise — focus on the WHY, not a full tutorial."
)

LOCALIZE_SYSTEM = (
    "You are an expert software engineer. "
    "Given a GitHub issue description, identify the most likely file and function where the bug occurs. "
    "Output the file path and function/method name, one per line."
)


def build_code_repair(issue: str, code_files: List[Dict], patches: List[Dict],
                       max_tokens: int = 1536) -> Optional[Dict]:
    """Build a code_repair example: issue + snippet → patch."""
    if not patches:
        return None

    patch = patches[0]
    snippet = find_buggy_file_snippet(code_files, patches, max_tokens=max_tokens // 3)
    if not snippet:
        return None

    # Build the patch output
    patch_text = (
        f"```\n### {patch['file']}\n"
        f"<<<<<<< SEARCH\n{patch['search']}\n"
        f"=======\n{patch['replace']}\n"
        f">>>>>>> REPLACE\n```"
    )

    user_content = f"## Issue\n{issue}\n\n## Code\n{snippet}"
    assistant_content = patch_text

    # Check total tokens
    total = count_tokens(CODE_REPAIR_SYSTEM) + count_tokens(user_content) + count_tokens(assistant_content)
    if total > max_tokens:
        # Try truncating the issue
        max_issue_tokens = max_tokens - count_tokens(snippet) - count_tokens(assistant_content) - count_tokens(CODE_REPAIR_SYSTEM) - 50
        if max_issue_tokens < 30:
            return None
        # Rough truncation by chars
        max_issue_chars = max_issue_tokens * 4
        truncated_issue = issue[:max_issue_chars].rsplit('\n', 1)[0] + "\n..."
        user_content = f"## Issue\n{truncated_issue}\n\n## Code\n{snippet}"
        total = count_tokens(CODE_REPAIR_SYSTEM) + count_tokens(user_content) + count_tokens(assistant_content)
        if total > max_tokens:
            return None

    return {
        "messages": [
            {"role": "system", "content": CODE_REPAIR_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_bug_explain(issue: str, code_files: List[Dict], think: str,
                       patches: List[Dict], max_tokens: int = 1536) -> Optional[Dict]:
    """Build a bug_explain example: issue + snippet → reasoning."""
    if not think or len(think) < 100:
        return None

    snippet = find_buggy_file_snippet(code_files, patches, max_tokens=max_tokens // 3)
    if not snippet:
        return None

    # Truncate thinking to fit
    max_think_tokens = max_tokens - count_tokens(BUG_EXPLAIN_SYSTEM) - count_tokens(snippet) - count_tokens(issue) - 80
    if max_think_tokens < 50:
        return None

    think_truncated = think
    while count_tokens(think_truncated) > max_think_tokens:
        # Cut from the end, try to keep complete sentences
        think_truncated = think_truncated[:len(think_truncated) * 3 // 4].rsplit('.', 1)[0] + '.'

    user_content = f"## Issue\n{issue}\n\n## Code\n{snippet}"
    assistant_content = think_truncated

    total = count_tokens(BUG_EXPLAIN_SYSTEM) + count_tokens(user_content) + count_tokens(assistant_content)
    if total > max_tokens:
        return None

    return {
        "messages": [
            {"role": "system", "content": BUG_EXPLAIN_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_localize(issue: str, code_files: List[Dict], patches: List[Dict],
                    max_tokens: int = 512) -> Optional[Dict]:
    """Build a localize example: issue text → file + function name."""
    if not patches:
        return None

    # Build the target: file paths from patches
    files = list(set(p["file"] for p in patches))
    target = "\n".join(files)

    total = count_tokens(LOCALIZE_SYSTEM) + count_tokens(issue) + count_tokens(target)
    if total > max_tokens:
        # Truncate issue to fit
        max_issue_chars = (max_tokens - count_tokens(LOCALIZE_SYSTEM) - count_tokens(target) - 20) * 4
        if max_issue_chars < 100:
            return None
        issue = issue[:max_issue_chars].rsplit('\n', 1)[0] + "\n..."
        total = count_tokens(LOCALIZE_SYSTEM) + count_tokens(issue) + count_tokens(target)
        if total > max_tokens:
            return None

    return {
        "messages": [
            {"role": "system", "content": LOCALIZE_SYSTEM},
            {"role": "user", "content": issue},
            {"role": "assistant", "content": target},
        ]
    }


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_row(row: dict, tasks: List[str], max_tokens: int) -> Dict[str, Optional[Dict]]:
    """Process one agentless row into sub-task examples."""
    msgs = row["messages"]
    if len(msgs) < 3:
        return {}

    user_text = msgs[1]["content"]
    assistant_text = msgs[2]["content"]

    issue, code_context, code_files = extract_issue_and_code(user_text)
    think, solution = extract_think(assistant_text)
    patches = extract_patches(solution)

    results = {}

    if "code_repair" in tasks:
        results["code_repair"] = build_code_repair(issue, code_files, patches, max_tokens)

    if "bug_explain" in tasks:
        results["bug_explain"] = build_bug_explain(issue, code_files, think, patches, max_tokens)

    if "localize" in tasks:
        results["localize"] = build_localize(issue, code_files, patches, max_tokens=min(max_tokens, 512))

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare Nemotron SWE data for small-model SFT")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed JSONL files")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most N rows (for testing)")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Max tokens per example (default 1536)")
    parser.add_argument("--task", default="all", choices=["all", "code_repair", "bug_explain", "localize"],
                        help="Which sub-task(s) to extract")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    parser.add_argument("--local-file", default=None, help="Local path to swe_agentless.jsonl (skip download)")
    args = parser.parse_args()

    tasks = ["code_repair", "bug_explain", "localize"] if args.task == "all" else [args.task]

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Open output files
    writers = {}
    if not args.dry_run:
        for task in tasks:
            writers[task] = open(output_dir / f"swe_{task}.jsonl", "w", encoding="utf-8")

    # Stats
    stats = {task: Counter() for task in tasks}
    total_processed = 0
    total_skipped = 0

    # Stream the data
    print(f"Streaming swe_agentless.jsonl...")
    if args.local_file:
        f = open(args.local_file, "r", encoding="utf-8")
        print(f"  Source: {args.local_file}")
    else:
        try:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            f = fs.open("datasets/nvidia/Nemotron-Cascade-2-SFT-Data/swe/swe_agentless.jsonl", "r")
            print("  Source: HuggingFace Hub (streaming)")
        except ImportError:
            print("ERROR: Install huggingface_hub: pip install huggingface_hub")
            sys.exit(1)

    try:
        for line_num, line in enumerate(f):
            if args.max_rows and line_num >= args.max_rows:
                break

            if line_num % 1000 == 0 and line_num > 0:
                print(f"  Processed {line_num} rows... "
                      + " ".join(f"{t}={stats[t]['extracted']}" for t in tasks))

            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                total_skipped += 1
                continue

            results = process_row(row, tasks, args.max_tokens)
            total_processed += 1

            for task, example in results.items():
                if example is not None:
                    stats[task]["extracted"] += 1
                    # Count tokens for stats
                    total_tok = sum(count_tokens(m["content"]) for m in example["messages"])
                    stats[task]["total_tokens"] += total_tok
                    if total_tok <= 512:
                        stats[task]["under_512"] += 1
                    elif total_tok <= 1024:
                        stats[task]["under_1024"] += 1
                    else:
                        stats[task]["under_2048"] += 1

                    if not args.dry_run:
                        writers[task].write(json.dumps(example, ensure_ascii=False) + "\n")
                else:
                    stats[task]["filtered"] += 1

    finally:
        f.close()
        for w in writers.values():
            w.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"SWE Data Preprocessing Summary")
    print(f"{'='*60}")
    print(f"Total rows processed: {total_processed}")
    print(f"Total rows skipped (parse error): {total_skipped}")
    print(f"Max tokens per example: {args.max_tokens}")
    print()

    for task in tasks:
        s = stats[task]
        extracted = s["extracted"]
        filtered = s["filtered"]
        total_tok = s["total_tokens"]
        avg_tok = total_tok / extracted if extracted else 0
        print(f"  {task}:")
        print(f"    Extracted: {extracted} ({extracted/(total_processed or 1)*100:.1f}%)")
        print(f"    Filtered:  {filtered}")
        print(f"    Total tokens: {total_tok:,}")
        print(f"    Avg tokens/example: {avg_tok:.0f}")
        print(f"    Distribution: <=512={s['under_512']}, <=1024={s['under_1024']}, <=2048={s['under_2048']}")
        if not args.dry_run:
            out_path = output_dir / f"swe_{task}.jsonl"
            size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
            print(f"    Output: {out_path} ({size_mb:.1f} MB)")
        print()


if __name__ == "__main__":
    main()
