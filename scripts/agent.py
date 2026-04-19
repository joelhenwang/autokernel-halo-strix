#!/usr/bin/env python3
"""ARGUS Agent — 168M parameter tool-using agent.

Runs an interactive agent loop:
  1. User types a question
  2. Model generates a response (may include <tool_call>)
  3. If tool call detected → execute tool → feed result back → model answers
  4. If no tool call → print response directly

Built-in tools: web_search, calculator, python_exec, get_weather, get_time

Usage:
    python scripts/agent.py --checkpoint checkpoints/argus_prime_mixed_sft/step_1446.pt
    python scripts/agent.py --checkpoint checkpoints/argus_prime_mixed_sft/step_1446.pt --tools web_search,calculator
"""

import argparse
import json
import re
import sys
import time
import importlib.util
from datetime import datetime

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tool Definitions & Executors
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {}


def tool(name, description, parameters):
    """Decorator to register a tool with its JSON schema."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "executor": func,
        }
        return func
    return decorator


@tool(
    name="web_search",
    description="Search the web for current information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo Lite (no API key needed)."""
    try:
        import urllib.request
        import urllib.parse
        url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(url, headers={"User-Agent": "ARGUS-Agent/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Extract text snippets from the lite HTML
        results = []
        # DuckDuckGo Lite returns results in <a> and <td> tags
        snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)
        links = re.findall(r'<a[^>]*rel="nofollow"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        for i, (href, title) in enumerate(links[:5]):
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            if title:
                results.append(f"{i+1}. {title}\n   {snippet}")
        if results:
            return "\n".join(results)
        # Fallback: extract any visible text
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:500] if text else "No results found."
    except Exception as e:
        return f"Search error: {e}"


@tool(
    name="calculator",
    description="Calculate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    import math
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }
    try:
        expr = expression.strip()
        # Basic sanitization
        if any(kw in expr for kw in ["import", "exec", "eval", "__", "open", "os.", "sys."]):
            return "Error: unsafe expression"
        result = eval(expr, {"__builtins__": {}}, allowed)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(
    name="python_exec",
    description="Execute a Python code snippet and return the output",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"]
    }
)
def python_exec(code: str) -> str:
    """Execute Python code in a restricted subprocess."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            output = f"Error: {result.stderr.strip()}"
        return output[:1000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (10s limit)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="get_time",
    description="Get the current date and time",
    parameters={
        "type": "object",
        "properties": {},
    }
)
def get_time() -> str:
    now = datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
    })


# ---------------------------------------------------------------------------
# Model Loading (reuse from generate_text.py)
# ---------------------------------------------------------------------------

def load_model(model_path, class_name):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)()


def generate(model, prompt_ids, max_tokens=300, temperature=0.3, top_k=20,
             repetition_penalty=1.2, frequency_penalty=0.3, stop_tokens=None,
             max_seq_len=2048, vocab_size=50262):
    if stop_tokens is None:
        stop_tokens = set()
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    token_counts = torch.zeros(1, vocab_size, device=device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for _ in range(max_tokens):
            context = ids[:, -max_seq_len:]
            logits = model(context)[:, -1, :]

            if repetition_penalty != 1.0:
                seen = set(ids[0].tolist())
                for tid in seen:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

            if frequency_penalty > 0:
                logits -= frequency_penalty * token_counts

            logits /= max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_token], dim=1)
            tid = next_token.item()
            if tid < vocab_size:
                token_counts[0, tid] += 1
            if tid in stop_tokens:
                break

    return ids


# ---------------------------------------------------------------------------
# Tool Call Detection & Execution
# ---------------------------------------------------------------------------

def parse_tool_call(text: str):
    """Extract tool call JSON from <tool_call>...</tool_call> markers."""
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Training data format: arguments is a single-quoted JSON string
    # e.g. {"name": "calc", "arguments": '{"x": 1}'}
    # Extract name and arguments separately
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    args_match = re.search(r'"arguments"\s*:\s*[\'"](.+?)[\'"](?:\s*})', raw, re.DOTALL)
    if name_match:
        name = name_match.group(1)
        args = "{}"
        if args_match:
            args_str = args_match.group(1)
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                try:
                    args = json.loads(args_str.replace("'", '"'))
                except json.JSONDecodeError:
                    args = {}
        return {"name": name, "arguments": args}

    return None


TOOL_ALIASES = {
    "search_results": "web_search",
    "search_news": "web_search",
    "get_news": "web_search",
    "get_news_headlines": "web_search",
    "search_web": "web_search",
    "web_search_results": "web_search",
    "search": "web_search",
    "calculate_tip": "calculator",
    "calculate_bmi": "calculator",
    "calculate_loan": "calculator",
    "calculate": "calculator",
    "math": "calculator",
    "run_code": "python_exec",
    "execute_code": "python_exec",
    "get_current_time": "get_time",
    "get_date": "get_time",
    "get_stock_price": "web_search",
    "get_movie_details": "web_search",
    "get_recipe": "web_search",
    "get_definition": "web_search",
    "generate_password": "python_exec",
}


def execute_tool(call: dict, available_tools: dict) -> str:
    """Execute a tool call and return the result string."""
    name = call.get("name", "")
    args_raw = call.get("arguments", "{}")

    # Fuzzy match: map training-data tool names to our actual tools
    if name not in available_tools and name in TOOL_ALIASES:
        name = TOOL_ALIASES[name]

    if name not in available_tools:
        return json.dumps({"error": f"Unknown tool: {name}"})

    # Parse arguments — may be a JSON string or already a dict
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            try:
                args = json.loads(args_raw.replace("'", '"'))
            except (json.JSONDecodeError, AttributeError):
                args = {}
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}

    # Adapt arguments for web_search — model may use domain-specific param names
    if name == "web_search" and "query" not in args:
        # Turn any string arg into a search query
        for key in ["ticker", "company", "country", "title", "keyword", "topic", "term", "location"]:
            if key in args:
                args = {"query": str(args[key])}
                break
        else:
            if args:
                args = {"query": " ".join(str(v) for v in args.values())}

    # Adapt arguments for calculator — model may emit domain-specific args
    if name == "calculator" and "expression" not in args:
        if "bill_amount" in args and "tip_percentage" in args:
            args = {"expression": f"{args['bill_amount']} * {args['tip_percentage']} / 100"}
        elif "total_amount" in args and "tip_percentage" in args:
            args = {"expression": f"{args['total_amount']} * {args['tip_percentage']} / 100"}
        elif "base" in args and "exponent" in args:
            args = {"expression": f"{args['base']} ** {args['exponent']}"}
        elif args:
            # Last resort: turn any numeric args into a string expression
            vals = [str(v) for v in args.values() if isinstance(v, (int, float))]
            if vals:
                args = {"expression": " + ".join(vals)}

    executor = available_tools[name]["executor"]
    try:
        result = executor(**args)
        return result
    except TypeError as e:
        # Argument mismatch — try with just the first arg value
        if args:
            try:
                first_val = next(iter(args.values()))
                result = executor(first_val)
                return result
            except Exception:
                pass
        return json.dumps({"error": f"Tool '{name}' argument error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {e}"})


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

def build_system_prompt(tools: dict) -> str:
    """Build the system prompt with available tool definitions."""
    if not tools:
        return "You are a helpful assistant."

    tool_defs = []
    for name, info in tools.items():
        tool_defs.append(json.dumps({
            "name": info["name"],
            "description": info["description"],
            "parameters": info["parameters"],
        }, indent=4))

    tools_json = "\n".join(tool_defs)
    return f"You are a helpful assistant with access to the following functions. Use them if required -\n{tools_json}"


class Agent:
    def __init__(self, model, tokenizer, tools, max_seq_len=2048):
        self.model = model
        self.tok = tokenizer
        self.tools = tools
        self.max_seq_len = max_seq_len
        self.system_prompt = build_system_prompt(tools)
        self.conversation = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _generate_response(self, messages):
        """Generate a single assistant response given conversation history."""
        tokens = self.tok.encode_chatml(messages) + self.tok.start_assistant_turn()
        prompt_ids = torch.tensor([tokens], dtype=torch.long)

        output_ids = generate(
            self.model, prompt_ids,
            max_tokens=300,
            temperature=0.3,
            top_k=20,
            repetition_penalty=1.2,
            frequency_penalty=0.3,
            stop_tokens={self.tok.im_end_id},
            max_seq_len=self.max_seq_len,
            vocab_size=self.tok.vocab_size,
        )

        all_ids = output_ids[0].tolist()
        # Extract only the newly generated tokens (after the prompt)
        prompt_len = len(tokens)
        new_ids = all_ids[prompt_len:]
        # Remove trailing im_end token
        if new_ids and new_ids[-1] == self.tok.im_end_id:
            new_ids = new_ids[:-1]
        response = self.tok.decode(new_ids).strip()
        return response

    def step(self, user_message: str, max_tool_rounds: int = 3) -> str:
        """Process one user message through the full agent loop.

        Returns the final assistant response (after any tool calls).
        """
        self.conversation.append({"role": "user", "content": user_message})

        for round_num in range(max_tool_rounds + 1):
            response = self._generate_response(self.conversation)

            # Check for tool call
            tool_call = parse_tool_call(response)

            if tool_call is None:
                # No tool call — this is the final response
                self.conversation.append({"role": "assistant", "content": response})
                return response

            # Tool call detected — execute it
            tool_name = tool_call.get("name", "unknown")
            print(f"  [tool] Calling {tool_name} (args: {tool_call.get('arguments', '?')})...")

            self.conversation.append({"role": "assistant", "content": response})
            tool_result = execute_tool(tool_call, self.tools)
            print(f"  [tool] Result: {tool_result[:200]}")

            self.conversation.append({"role": "tool", "content": tool_result})

            # Trim conversation if it's getting too long
            total_tokens = sum(len(self.tok.encode(m["content"])) for m in self.conversation)
            if total_tokens > self.max_seq_len - 300:
                # Keep system + last 4 turns
                self.conversation = [self.conversation[0]] + self.conversation[-4:]

        return "(Agent exceeded max tool rounds)"

    def reset(self):
        """Clear conversation history."""
        self.conversation = [
            {"role": "system", "content": self.system_prompt}
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARGUS Agent — 168M tool-using agent")
    parser.add_argument("--model", default="models/argus_prime.py")
    parser.add_argument("--class-name", default="ArgusPrime")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tools", default="all",
                        help="Comma-separated tool names or 'all' (default: all)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--query", default=None,
                        help="Single query (non-interactive mode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    sys.path.insert(0, ".")
    print("Loading model...")
    model = load_model(args.model, args.class_name)
    model = model.to(device)

    # Detect vocab size and build tokenizer
    from halo_training.chat_template import build_tokenizer, resize_embeddings
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_vocab = None
    if "model_state_dict" in ckpt_peek:
        w = ckpt_peek["model_state_dict"].get("tok_embeddings.weight")
        if w is not None:
            ckpt_vocab = w.shape[0]
    del ckpt_peek

    tok_phase = "domain-sft" if ckpt_vocab and ckpt_vocab > 50260 else "sft"
    tokenizer = build_tokenizer(phase=tok_phase)
    model = resize_embeddings(model, tokenizer.vocab_size)

    # Apply autokernel
    try:
        import autokernel
        model = autokernel.optimize(model, training=False)
        print("Applied autokernel.optimize()")
    except Exception as e:
        print(f"autokernel not applied: {e}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("  Warning: loaded with strict=False")
        print(f"  Step: {ckpt.get('step', '?')}")
    else:
        model.load_state_dict(ckpt, strict=False)
    del ckpt

    # Select tools
    if args.tools == "all":
        tools = TOOL_REGISTRY
    else:
        tool_names = [t.strip() for t in args.tools.split(",")]
        tools = {k: v for k, v in TOOL_REGISTRY.items() if k in tool_names}

    print(f"Tools: {', '.join(tools.keys())}")

    # Create agent
    agent = Agent(model, tokenizer, tools, max_seq_len=args.max_seq_len)

    # Single query mode
    if args.query:
        response = agent.step(args.query)
        print(f"\nAssistant: {response}")
        return

    # Interactive mode
    print(f"\nARGUS Agent (168M) — {len(tools)} tools available")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("(conversation reset)")
            continue

        t0 = time.time()
        response = agent.step(user_input)
        elapsed = time.time() - t0

        print(f"\nAssistant: {response}")
        print(f"  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
