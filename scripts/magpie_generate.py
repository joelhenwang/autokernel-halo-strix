"""Magpie-style synthetic instruction data generation via API.

Generates (prompt, chosen_response, rejected_response) triples for ORPO/SimPO,
or (prompt, response, label) for KTO.

Strategy:
1. Ask the API to generate diverse user instructions
2. Generate a high-quality response (chosen) with strong model
3. Generate a weaker response (rejected) with lower temperature/different prompt
4. Save as JSONL compatible with DPODataset and KTODataset

Usage:
    python scripts/magpie_generate.py --n 50000 --output datasets/alignment/magpie_50k.jsonl
    python scripts/magpie_generate.py --n 1000 --output datasets/alignment/magpie_test.jsonl --provider anthropic
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional

SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant."

INSTRUCTION_CATEGORIES = [
    "general knowledge question",
    "creative writing prompt",
    "math problem",
    "coding task",
    "summarization request",
    "explain a concept simply",
    "give advice about a situation",
    "translate or rephrase text",
    "analyze pros and cons",
    "step-by-step instructions",
    "comparison between two things",
    "brainstorming ideas",
    "factual question",
    "opinion or recommendation request",
    "problem solving",
]

INSTRUCTION_GEN_PROMPT = """Generate {n} diverse user instructions/questions. Each should be a natural message a user might send to an AI assistant.

Requirements:
- Cover different categories: {categories}
- Vary length (short questions to detailed requests)
- Be specific and concrete (not vague)
- Include a mix of difficulties

Return ONLY a JSON array of strings, no other text:
["instruction 1", "instruction 2", ...]"""

REJECT_SYSTEM = "You are a helpful AI assistant. Give brief, somewhat generic responses."


def generate_instructions_anthropic(n: int, model: str = "claude-haiku-4-5-20241022") -> List[str]:
    """Generate instructions using Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()
    instructions = []
    batch_size = min(50, n)

    while len(instructions) < n:
        remaining = n - len(instructions)
        batch = min(batch_size, remaining)
        cats = ", ".join(random.sample(INSTRUCTION_CATEGORIES, min(5, len(INSTRUCTION_CATEGORIES))))

        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": INSTRUCTION_GEN_PROMPT.format(n=batch, categories=cats),
            }],
        )
        text = resp.content[0].text.strip()
        try:
            if text.startswith("["):
                batch_instructions = json.loads(text)
            else:
                start = text.find("[")
                end = text.rfind("]") + 1
                batch_instructions = json.loads(text[start:end])
            instructions.extend(batch_instructions)
            print(f"  Generated {len(instructions)}/{n} instructions")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Parse error, retrying: {e}")
            continue

    return instructions[:n]


def generate_response_anthropic(
    instruction: str,
    model: str = "claude-haiku-4-5-20241022",
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 512,
) -> str:
    """Generate a single response using Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": instruction}],
    )
    return resp.content[0].text.strip()


def generate_instructions_openai(n: int, model: str = "gpt-4o-mini") -> List[str]:
    """Generate instructions using OpenAI API."""
    import openai
    client = openai.OpenAI()
    instructions = []
    batch_size = min(50, n)

    while len(instructions) < n:
        remaining = n - len(instructions)
        batch = min(batch_size, remaining)
        cats = ", ".join(random.sample(INSTRUCTION_CATEGORIES, min(5, len(INSTRUCTION_CATEGORIES))))

        resp = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": INSTRUCTION_GEN_PROMPT.format(n=batch, categories=cats),
            }],
        )
        text = resp.choices[0].message.content.strip()
        try:
            if text.startswith("["):
                batch_instructions = json.loads(text)
            else:
                start = text.find("[")
                end = text.rfind("]") + 1
                batch_instructions = json.loads(text[start:end])
            instructions.extend(batch_instructions)
            print(f"  Generated {len(instructions)}/{n} instructions")
        except (json.JSONDecodeError, ValueError):
            continue

    return instructions[:n]


def generate_response_openai(
    instruction: str,
    model: str = "gpt-4o-mini",
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 512,
) -> str:
    """Generate a single response using OpenAI API."""
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ],
    )
    return resp.choices[0].message.content.strip()


def build_preference_pair(
    instruction: str,
    chosen_response: str,
    rejected_response: str,
) -> dict:
    """Build a preference pair in DPODataset format."""
    return {
        "chosen": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": chosen_response},
        ],
        "rejected": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": rejected_response},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Magpie synthetic data generation")
    parser.add_argument("--n", type=int, default=1000, help="Number of examples")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--chosen-model", default=None, help="Model for chosen responses")
    parser.add_argument("--rejected-model", default=None, help="Model for rejected responses")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.provider == "anthropic":
        gen_model = args.model or "claude-haiku-4-5-20241022"
        chosen_model = args.chosen_model or gen_model
        rejected_model = args.rejected_model or gen_model
        gen_instructions = generate_instructions_anthropic
        gen_response = generate_response_anthropic
    else:
        gen_model = args.model or "gpt-4o-mini"
        chosen_model = args.chosen_model or gen_model
        rejected_model = args.rejected_model or gen_model
        gen_instructions = generate_instructions_openai
        gen_response = generate_response_openai

    existing = 0
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing} existing examples")

    remaining = args.n - existing
    if remaining <= 0:
        print(f"Already have {existing} >= {args.n} examples")
        return

    print(f"Generating {remaining} examples using {args.provider} ({gen_model})")

    # Step 1: Generate instructions
    print("Step 1: Generating instructions...")
    instructions = gen_instructions(remaining, model=gen_model)

    # Step 2: Generate chosen + rejected responses
    print("Step 2: Generating response pairs...")
    mode = "a" if args.resume else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for i, instruction in enumerate(instructions):
            try:
                chosen = gen_response(instruction, model=chosen_model,
                                      system=SYSTEM_PROMPT, max_tokens=512)
                rejected = gen_response(instruction, model=rejected_model,
                                        system=REJECT_SYSTEM, max_tokens=256)
                pair = build_preference_pair(instruction, chosen, rejected)
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(instructions)}] generated")

            except Exception as e:
                print(f"  Error on example {i}: {e}")
                continue

    total = existing + len(instructions)
    print(f"\nDone: {total} examples saved to {output_path}")


if __name__ == "__main__":
    main()
