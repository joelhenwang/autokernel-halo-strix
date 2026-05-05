# Halo-ML-Intern: Autonomous ML Research Agent

**Date:** 2026-05-05
**Status:** Design approved, pending implementation
**Author:** Joel + Claude

## Overview

Halo-ML-Intern is a personal autonomous ML research agent that combines interactive assistance (Claude Code) with an autonomous daemon for overnight/long-running operations. It handles kernel optimization, architecture search, training monitoring, checkpoint management, data pipelines, and hardware babysitting — all with deep domain knowledge of the autokernel-halo-strix project and AMD Strix Halo hardware.

## Goals

- Full overnight autonomy: kernel opt, training, monitoring, recovery — no human in loop
- Interactive mode for daytime work (integrates with Claude Code via MCP)
- Multi-model routing: premium reasoning, standard code gen, cheap monitoring
- Provider-agnostic: swap LLM backends via config, not code
- Crash-resilient: survives SSH drops, OOM, restarts
- Cost-conscious: budget caps, model cascading, spending alerts

## Non-Goals (v1)

- Web UI (CLI + daemon only)
- Multi-user / auth
- Training own routing model
- Distributed agent coordination (single daemon process)

## Architecture: Event-Driven Pipeline

```
┌─────────────────────────────────────────┐
│  Event Bus (async, priority queue)      │
└────┬────────┬────────┬────────┬─────────┘
     │        │        │        │
┌────▼───┐┌───▼────┐┌──▼───┐┌──▼────────┐
│Planner ││Executor││Monitor││Knowledge  │
│(premium)│(standard)│(cheap)││(retrieval)│
└────┬───┘└───┬────┘└──┬───┘└───────────┘
     │        │        │
     └────────┴────────┘
              │
     ┌────────▼────────┐
     │  Tool Layer     │
     │ MCP + Python    │
     └─────────────────┘
```

Events drive everything. Agents subscribe to event types, process them, emit new events. Creates reactive chains (NaN detected → planner decides → executor restarts training).

## Project Structure

```
halo-ml-intern/
├── core/
│   ├── event_bus.py          # Async priority queue + pub/sub
│   ├── agent_loop.py         # Base agent loop (LLM call → tool → emit)
│   ├── llm.py                # Provider-agnostic LLM abstraction
│   ├── router.py             # Tier-based model routing
│   ├── session.py            # State persistence (JSON files)
│   ├── doom_loop.py          # Cycle detection + corrective injection
│   └── budget.py             # Token/cost tracking per provider
├── agents/
│   ├── planner.py            # Strategic decisions (premium tier)
│   ├── executor.py           # Code gen + tool execution (standard tier)
│   ├── monitor.py            # Continuous log/health watcher (cheap tier)
│   └── base.py               # Shared agent interface
├── knowledge/
│   ├── core_constraints.py   # Always-loaded (~200 tokens)
│   ├── retriever.py          # BM25 on-demand search
│   ├── indexer.py            # Markdown → chunks pipeline
│   └── store/                # Indexed knowledge chunks
├── tools/
│   ├── mcp/                  # MCP servers (ssh, hf_hub, fireworks, notifications)
│   ├── internal/             # Python tools (training, kernel, arch, checkpoint, data, hardware, status)
│   └── registry.py           # Tool discovery + schema
├── events/
│   ├── schema.py             # Event type definitions
│   └── handlers.py           # Event → agent routing rules
├── config/
│   ├── providers.yaml        # LLM endpoints + adapters
│   ├── routing.yaml          # Tier → provider bindings
│   └── budgets.yaml          # Cost caps per tier/day
├── sessions/                  # Persisted state
│   ├── current.json
│   ├── history/              # Completed session logs (JSONL)
│   └── recovery/             # Crash recovery snapshots
└── main.py                   # Entry point (daemon/chat/status/replay/stop)
```

## Event System

### Event Schema

```python
@dataclass
class Event:
    id: str                    # UUID
    type: EventType            # Enum
    priority: Priority         # CRITICAL, HIGH, NORMAL, LOW
    source: str                # Which agent/tool emitted
    payload: dict              # Event-specific data
    timestamp: datetime
    parent_id: str | None      # Chain tracing
```

### Event Types

| Category | Events | Priority |
|----------|--------|----------|
| Training | `train.started`, `train.step`, `train.nan_detected`, `train.diverged`, `train.completed`, `train.checkpoint_saved` | HIGH-CRITICAL |
| Kernel | `kernel.profile_done`, `kernel.bench_result`, `kernel.speedup_achieved`, `kernel.regression` | NORMAL-HIGH |
| Architecture | `arch.search_started`, `arch.candidate_ranked`, `arch.search_complete` | NORMAL |
| Hardware | `hw.oom`, `hw.thermal_throttle`, `hw.ssh_drop`, `hw.process_dead` | CRITICAL |
| Data | `data.mixture_ready`, `data.tokenization_done`, `data.validation_failed` | NORMAL-HIGH |
| Agent | `agent.task_complete`, `agent.stuck`, `agent.budget_exceeded` | HIGH |

### Subscriptions

| Agent | Subscribes To |
|-------|---------------|
| Monitor | `train.step`, `hw.*`, `agent.stuck` |
| Planner | `train.nan_detected`, `train.completed`, `kernel.regression`, `arch.search_complete`, `agent.budget_exceeded` |
| Executor | `planner.action` |

### Reactive Chain Example

```
Monitor sees NaN in train_log.jsonl
  → emits train.nan_detected {step: 4500, loss: nan, last_good: 4200}
    → Planner retrieves training_antipatterns.md
      → emits planner.action {type: "restart", checkpoint: "step_4200", lr_scale: 0.5}
        → Executor stops training, resumes from step 4200 with halved LR
          → emits train.started {resumed_from: 4200, lr: 0.001}
            → Monitor watches again
```

### Crash Recovery

Event bus persists unprocessed events to disk. On restart: load last good state, replay unprocessed events, resume from last completed action.

## Multi-Model Routing

### Provider-Agnostic Design

Agents reference **tiers**, never specific models or providers. Swapping providers = config change only.

```python
class LLMProvider(Protocol):
    async def generate(self, messages, params) -> Response: ...
    async def stream(self, messages, params) -> AsyncIterator[Chunk]: ...
    def estimate_cost(self, input_tokens, output_tokens) -> float: ...
```

Adapters: `BedrockProvider`, `AnthropicDirectProvider`, `FireworksProvider`, `OpenCodeProvider`, `OpenRouterProvider`, `GoogleVertexProvider`.

### Tier System

```yaml
roles:
  planner:
    capability: "strategic reasoning, long context"
    tier: premium
  executor:
    capability: "code generation, tool use"
    tier: standard
  monitor:
    capability: "fast parsing, triage"
    tier: cheap

provider_bindings:
  premium: bedrock/opus
  standard: fireworks/deepseek-v4-pro
  cheap: bedrock/haiku
```

### Cascade Logic

If preferred model fails (rate limit, budget, timeout):
- Premium: opus → sonnet → deepseek-v4-pro
- Standard: deepseek-v4-pro → kimi-k2.6 → glm-5.1
- Cheap: haiku → (retry with backoff)

### Budget System

```yaml
daily_limits:
  premium: $15.00
  standard: $10.00
  cheap: $3.00
  total: $30.00

alerts:
  warn_at: 0.7      # 70% — log warning
  pause_at: 0.9     # 90% — pause non-critical, notify human
  hard_stop: 1.0    # Kill all except CRITICAL events
```

## Tool Layer

### MCP Servers (external integrations)

| Server | Tools | Purpose |
|--------|-------|---------|
| `ssh_remote` | `run_command`, `tail_log`, `check_process`, `transfer_file` | Execute on Strix Halo machines |
| `hf_hub` | `upload_model`, `download_dataset`, `create_repo`, `browse_papers` | HuggingFace Hub operations |
| `fireworks` | `generate`, `batch_generate` | Fireworks.ai model calls |
| `notifications` | `notify`, `request_approval` | Alert human (desktop/email/telegram) |

### Python Tools (internal logic)

| Module | Key Functions |
|--------|--------------|
| `knowledge.py` | `retrieve(query)`, `get_constraints()`, `get_hardware_spec()` |
| `training.py` | `launch_training()`, `resume_training()`, `stop_training()`, `parse_train_log()` |
| `kernel_opt.py` | `profile_model()`, `extract_bottlenecks()`, `bench_kernel()`, `verify_kernel()` |
| `architecture.py` | `list_models()`, `smoke_test()`, `compare_runs()`, `rank_candidates()` |
| `checkpoint.py` | `list_checkpoints()`, `validate_checkpoint()`, `evaluate_checkpoint()`, `prune_checkpoints()` |
| `data_pipeline.py` | `run_climb_search()`, `pretokenize()`, `validate_dataset()` |
| `status.py` | `read_status()`, `update_status()`, `read_report()`, `append_report()` |
| `hardware.py` | `check_gpu_memory()`, `check_thermals()`, `check_disk()`, `check_processes()` |

### Access Control

| Agent | MCP Access | Python Tools |
|-------|-----------|--------------|
| Planner | notifications | knowledge, status, architecture |
| Executor | ssh_remote, hf_hub | ALL internal tools |
| Monitor | ssh_remote (read-only) | training (parse only), hardware, status |

## Knowledge System

### Tier 1: Core Constraints (always in context)

~200 tokens injected into every agent's system prompt. Contains hardware limits, kernel rules, training rules, safety constraints. Never stale — updated manually when project constraints change.

### Tier 2: Deep Knowledge (on-demand retrieval)

BM25 keyword search over pre-chunked markdown from autokernel-halo-strix/knowledge/.

- Source: 32 .md files, ~9K lines
- Chunking: split by `##` headers (natural sections)
- Tags: source file, section name, category, keywords
- Query: BM25 keyword match, optional Haiku re-ranking for ambiguous queries
- Refresh: manual CLI command or event-triggered on file modification

BM25 chosen over vector embeddings because corpus is small and technical terms (gfx1151, FusedQKV, RMSNorm) match better with exact keywords.

## Doom Loop Detection

### Detection

```python
class DoomLoopDetector:
    window: int = 30          # Look back N actions
    min_repeats: int = 3      # Trigger after 3 identical
    max_cycle_len: int = 5    # Detect patterns up to length 5
```

Hashes action signatures (tool_name + normalized args). Distinguishes legitimate polling (same args, different results) from doom loops (same args AND same results).

### Escalation Ladder

1. Attempts 1-2: Inject corrective prompt ("try different approach")
2. Attempts 3-4: Escalate to planner agent
3. Attempt 5: Notify human, pause task

## Session Persistence

### State (persisted every event cycle)

- Event queue (unprocessed)
- Agent states (context summaries)
- Action history (for doom loop detection)
- Budget spent (per-provider)
- Active tasks (in-flight work)

### Recovery Flow

1. Load `sessions/current.json`
2. If crashed/stale (>10 min no update): load `recovery/last_good.json`
3. Replay unprocessed events
4. Emit `agent.recovered` → Planner decides what to resume

### Session Compaction

When action_history > 200 entries: summarize old actions (cheap tier call), keep last 50 raw, store summary as agent context. Prevents unbounded growth on multi-day sessions.

### Session Logging

All events + LLM calls + tool results logged in JSONL format. Compatible with Claude Code traces. Future SFT training data source.

## CLI Interface

```bash
halo-intern daemon --config config/overnight.yaml   # Autonomous mode
halo-intern chat                                     # Interactive mode
halo-intern status                                   # What's running, budget, alerts
halo-intern replay --last 8h                         # What happened while away
halo-intern stop                                     # Graceful shutdown
```

### Daemon Config

```yaml
tasks:
  - type: training
    model: vidar_halo
    dataset: dolma-10b-vidar32k
    flags: ["--compile", "--optimize-kernels", "--ema", "--epochs 1"]
    on_nan: restart_halved_lr
    on_oom: reduce_batch_size
    
  - type: kernel_optimization
    target: models/vidar_halo.py
    goal: "beat current 35K tok/s"
    max_iterations: 20
    
  - type: monitor
    watch: [train_log.jsonl, gpu_thermals, disk_space]
    alert_on: [nan, divergence, oom, disk_below_20gb]

approval_required:
  - delete_checkpoint
  - push_to_hub
  - spend_above: $20
```

### Claude Code Integration

MCP server exposes daemon state to Claude Code interactive sessions:
- `halo_status` — current state
- `halo_submit_task` — add task to daemon queue
- `halo_pause` — pause daemon
- `halo_replay` — what happened since last check
- `halo_approve` — approve pending actions

## Future Considerations (post-v1)

- **Web UI (#1):** Dashboard showing status, event timeline, budget graphs
- **SFT Flywheel (#4):** Use session JSONL logs to fine-tune routing/planning models
- **Multi-machine:** Agent coordinates across multiple Strix Halo machines

## Dependencies

- Python 3.10+
- `asyncio` — event loop
- `pydantic` — event/config schemas
- `litellm` or custom adapters — multi-provider LLM calls
- `fastmcp` — MCP server framework
- `rank_bm25` — knowledge retrieval
- `pyyaml` — config files
- `rich` — CLI output formatting

## Success Criteria

1. Daemon runs overnight (8+ hours) without human intervention
2. Auto-recovers from at least: NaN, OOM, SSH drops
3. Correct model routing saves >40% vs always using premium tier
4. Knowledge retrieval returns relevant context >80% of queries
5. Doom loop detection prevents infinite loops within 5 iterations
6. Budget system never exceeds daily cap
