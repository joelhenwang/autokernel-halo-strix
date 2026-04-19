#!/usr/bin/env python3
"""Generate DPO preference pairs for context-grounding training.

Creates chosen/rejected pairs that teach the model to:
1. Copy exact tool names from the system prompt (not memorized names)
2. Extract exact values from user messages (not approximate/hallucinated)
3. Ground responses on tool results (copy from tool output, not hallucinate)

The key insight: we don't need a reward model. We can construct preferences
algorithmically because we KNOW what the correct answer is for each example.

Output: JSONL with {"chosen": [...messages], "rejected": [...messages]} pairs.

Usage:
    python scripts/generate_dpo_data.py --output datasets/swe_prepared/dpo_grounding.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional


random.seed(42)

# ---------------------------------------------------------------------------
# Type 1: Tool Name Grounding
# The model should use EXACTLY the tool name from the system prompt,
# not a memorized name from training data.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {"name": "web_search", "description": "Search the web for current information",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculator", "description": "Calculate a mathematical expression",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "get_time", "description": "Get the current date and time",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_weather", "description": "Get weather for a location",
     "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
    {"name": "run_python", "description": "Execute Python code",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "translate_text", "description": "Translate text between languages",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}},
    {"name": "lookup_word", "description": "Look up the definition of a word",
     "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}},
    {"name": "convert_units", "description": "Convert between measurement units",
     "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}},
]

WRONG_NAMES = {
    "web_search": ["search_results", "search_news", "get_news_headlines", "google_search"],
    "calculator": ["calculate_tip", "calculate_bmi", "math_eval", "compute"],
    "get_time": ["get_current_time", "get_date_time", "current_time", "fetch_time"],
    "get_weather": ["get_forecast", "weather_api", "check_weather", "fetch_weather"],
    "run_python": ["execute_code", "python_exec", "run_code", "eval_python"],
    "translate_text": ["translate", "language_translate", "convert_language", "translation_api"],
    "lookup_word": ["get_definition", "dictionary_lookup", "define_word", "word_meaning"],
    "convert_units": ["unit_converter", "convert_measurement", "calculate_conversion", "unit_calc"],
}

QUERIES_FOR_TOOL = {
    "web_search": [
        ("What is the population of France?", {"query": "population of France"}),
        ("Find information about the Mars rover", {"query": "Mars rover"}),
        ("Who won the 2024 World Series?", {"query": "2024 World Series winner"}),
        ("Latest news about AI regulation", {"query": "AI regulation news"}),
        ("What is the tallest building in the world?", {"query": "tallest building in the world"}),
    ],
    "calculator": [
        ("What is 15% of 85?", {"expression": "85 * 15 / 100"}),
        ("Calculate 234 times 567", {"expression": "234 * 567"}),
        ("What is the square root of 144?", {"expression": "144 ** 0.5"}),
        ("How much is 1000 divided by 7?", {"expression": "1000 / 7"}),
        ("What is 2 to the power of 10?", {"expression": "2 ** 10"}),
    ],
    "get_weather": [
        ("What's the weather in Tokyo?", {"location": "Tokyo"}),
        ("Is it raining in London?", {"location": "London"}),
        ("Weather forecast for New York City", {"location": "New York City"}),
        ("How hot is it in Dubai right now?", {"location": "Dubai"}),
        ("What's the temperature in Sydney?", {"location": "Sydney"}),
    ],
    "get_time": [
        ("What time is it?", {}),
        ("What's today's date?", {}),
        ("What day of the week is it?", {}),
    ],
    "run_python": [
        ("Write code to print the first 10 fibonacci numbers", {"code": "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b"}),
        ("Generate a list of squares from 1 to 10", {"code": "print([x**2 for x in range(1, 11)])"}),
    ],
    "translate_text": [
        ("Translate 'hello world' to Spanish", {"text": "hello world", "target_language": "Spanish"}),
        ("How do you say 'thank you' in Japanese?", {"text": "thank you", "target_language": "Japanese"}),
    ],
    "lookup_word": [
        ("What does 'ephemeral' mean?", {"word": "ephemeral"}),
        ("Define 'ubiquitous'", {"word": "ubiquitous"}),
    ],
    "convert_units": [
        ("Convert 100 kilometers to miles", {"value": 100, "from_unit": "kilometers", "to_unit": "miles"}),
        ("How many pounds is 50 kilograms?", {"value": 50, "from_unit": "kilograms", "to_unit": "pounds"}),
    ],
}


def build_system_prompt(tools: List[Dict]) -> str:
    tool_defs = "\n".join(json.dumps(t, indent=4) for t in tools)
    return f"You are a helpful assistant with access to the following functions. Use them if required -\n{tool_defs}"


def gen_tool_name_pairs(n: int = 5000) -> List[Dict]:
    """Generate pairs where chosen uses exact tool name, rejected uses memorized name."""
    pairs = []
    for _ in range(n):
        # Pick 1-3 random tools for the system prompt
        num_tools = random.randint(1, 3)
        tools = random.sample(TOOL_SCHEMAS, min(num_tools, len(TOOL_SCHEMAS)))
        tool = random.choice(tools)
        name = tool["name"]

        if name not in QUERIES_FOR_TOOL or name not in WRONG_NAMES:
            continue

        query, correct_args = random.choice(QUERIES_FOR_TOOL[name])
        wrong_name = random.choice(WRONG_NAMES[name])

        system = build_system_prompt(tools)

        chosen_call = f'<tool_call>\n{json.dumps({"name": name, "arguments": json.dumps(correct_args)})}\n</tool_call>'
        rejected_call = f'<tool_call>\n{json.dumps({"name": wrong_name, "arguments": json.dumps(correct_args)})}\n</tool_call>'

        pairs.append({
            "chosen": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": chosen_call},
            ],
            "rejected": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": rejected_call},
            ],
        })
    return pairs


# ---------------------------------------------------------------------------
# Type 2: Value Extraction Grounding
# The model should extract exact values from the user message.
# ---------------------------------------------------------------------------

CALC_TEMPLATES = [
    ("What is {pct}% tip on a ${amount} bill?", "{amount} * {pct} / 100"),
    ("Calculate {pct}% of {amount}", "{amount} * {pct} / 100"),
    ("What is {a} plus {b}?", "{a} + {b}"),
    ("How much is {a} times {b}?", "{a} * {b}"),
    ("What is {a} divided by {b}?", "{a} / {b}"),
    ("What is {a} minus {b}?", "{a} - {b}"),
]


def gen_value_extraction_pairs(n: int = 5000) -> List[Dict]:
    """Generate pairs where chosen extracts exact numbers, rejected uses wrong ones."""
    pairs = []
    calc_tool = next(t for t in TOOL_SCHEMAS if t["name"] == "calculator")

    for _ in range(n):
        template, expr_template = random.choice(CALC_TEMPLATES)
        vals = {
            "amount": random.randint(10, 500),
            "pct": random.choice([10, 12, 15, 18, 20, 25]),
            "a": random.randint(1, 1000),
            "b": random.randint(1, 1000),
        }
        query = template.format(**vals)
        correct_expr = expr_template.format(**vals)

        # Wrong values: perturb the numbers
        wrong_vals = dict(vals)
        key = random.choice(list(wrong_vals.keys()))
        wrong_vals[key] = random.randint(10, 500)
        while wrong_vals[key] == vals[key]:
            wrong_vals[key] = random.randint(10, 500)
        wrong_expr = expr_template.format(**wrong_vals)

        system = build_system_prompt([calc_tool])

        chosen_call = f'<tool_call>\n{json.dumps({"name": "calculator", "arguments": json.dumps({"expression": correct_expr})})}\n</tool_call>'
        rejected_call = f'<tool_call>\n{json.dumps({"name": "calculator", "arguments": json.dumps({"expression": wrong_expr})})}\n</tool_call>'

        pairs.append({
            "chosen": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": chosen_call},
            ],
            "rejected": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": rejected_call},
            ],
        })
    return pairs


# ---------------------------------------------------------------------------
# Type 3: Response Grounding on Tool Output
# The model should quote/use the tool result, not hallucinate.
# ---------------------------------------------------------------------------

def _make_result_templates():
    return [
        {
            "type": "calc",
            "query_tpl": "What is {a} times {b}?",
            "chosen_tpl": "The result of {a} times {b} is {val}.",
            "rejected_tpl": "The result of {a} times {b} is {wrong_val}.",
        },
        {
            "type": "time",
            "query_tpl": "What time is it?",
            "chosen_tpl": "The current time is {time} on {day}, {date}.",
            "rejected_tpl": "The current time is {wrong_time} on {wrong_day}.",
        },
        {
            "type": "weather",
            "query_tpl": "What's the weather in {loc}?",
            "chosen_tpl": "The weather in {loc} is {cond} with a temperature of {temp} degrees.",
            "rejected_tpl": "The weather in {loc} is sunny with a temperature of {wrong_temp} degrees.",
        },
    ]

RESULT_TEMPLATES = _make_result_templates()

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CONDITIONS = ["sunny", "cloudy", "rainy", "partly cloudy", "foggy", "clear", "snowy"]
CITIES = ["Tokyo", "London", "New York", "Paris", "Sydney", "Berlin", "Mumbai", "Seoul", "Toronto", "Dubai"]


def gen_response_grounding_pairs(n: int = 5000) -> List[Dict]:
    """Generate pairs where chosen quotes tool output, rejected hallucinates."""
    pairs = []

    for _ in range(n):
        template = random.choice(RESULT_TEMPLATES)

        if template["type"] == "calc":
            a, b = random.randint(1, 100), random.randint(1, 100)
            val = a * b
            wrong_val = val + random.randint(1, 50) * random.choice([-1, 1])
            fmt = {"a": a, "b": b, "val": val, "wrong_val": wrong_val}
            tool_result = json.dumps({"result": val})
            tool_name = "calculator"
            tools = [next(t for t in TOOL_SCHEMAS if t["name"] == "calculator")]
        elif template["type"] == "time":
            hour, minute = random.randint(0, 23), random.randint(0, 59)
            time_str = f"{hour:02d}:{minute:02d}:00"
            date_str = f"2026-04-{random.randint(1, 28):02d}"
            day = random.choice(DAYS)
            wrong_time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"
            wrong_day = random.choice([d for d in DAYS if d != day])
            fmt = {"date": date_str, "time": time_str, "day": day,
                   "wrong_time": wrong_time, "wrong_day": wrong_day}
            tool_result = json.dumps({"date": date_str, "time": time_str, "day": day})
            tool_name = "get_time"
            tools = [next(t for t in TOOL_SCHEMAS if t["name"] == "get_time")]
        elif template["type"] == "weather":
            loc = random.choice(CITIES)
            temp = random.randint(-10, 45)
            cond = random.choice(CONDITIONS)
            wrong_temp = temp + random.randint(5, 20) * random.choice([-1, 1])
            fmt = {"loc": loc, "temp": temp, "cond": cond, "wrong_temp": wrong_temp}
            tool_result = json.dumps({"temperature": temp, "condition": cond, "location": loc})
            tool_name = "get_weather"
            tools = [next(t for t in TOOL_SCHEMAS if t["name"] == "get_weather")]
        else:
            continue

        system = build_system_prompt(tools)
        query = template["query_tpl"].format(**fmt)
        chosen_resp = template["chosen_tpl"].format(**fmt)
        rejected_resp = template["rejected_tpl"].format(**fmt)

        # Build the full conversation with tool call + tool result + response
        call_content = f'<tool_call>\n{json.dumps({"name": tool_name, "arguments": "{}"})}\n</tool_call>'

        pairs.append({
            "chosen": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": call_content},
                {"role": "tool", "content": tool_result},
                {"role": "assistant", "content": chosen_resp},
            ],
            "rejected": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": call_content},
                {"role": "tool", "content": tool_result},
                {"role": "assistant", "content": rejected_resp},
            ],
        })
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference data for grounding")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--n-tool-name", type=int, default=5000, help="Tool name grounding pairs")
    parser.add_argument("--n-value", type=int, default=5000, help="Value extraction pairs")
    parser.add_argument("--n-response", type=int, default=5000, help="Response grounding pairs")
    args = parser.parse_args()

    print("Generating DPO preference data...")
    pairs = []

    print(f"  Tool name grounding ({args.n_tool_name} pairs)...")
    pairs.extend(gen_tool_name_pairs(args.n_tool_name))

    print(f"  Value extraction ({args.n_value} pairs)...")
    pairs.extend(gen_value_extraction_pairs(args.n_value))

    print(f"  Response grounding ({args.n_response} pairs)...")
    pairs.extend(gen_response_grounding_pairs(args.n_response))

    random.shuffle(pairs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nGenerated {len(pairs)} preference pairs ({size_mb:.1f} MB)")
    print(f"  Tool name: {args.n_tool_name}")
    print(f"  Value extraction: {args.n_value}")
    print(f"  Response grounding: {args.n_response}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
