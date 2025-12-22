# Agent State & Logging

From First Principles (Professor Bone Lab)

⸻

## 1. What We Mean by Agent State

Agent State is the structured representation of everything an agent currently "knows" that is relevant to completing its task right now.

It is:
- Updated at every reasoning step
- Passed between nodes in an orchestration graph
- The only durable substrate the agent reasons over during execution

Agent State is not memory.
Agent State is not logs.
Agent State is not chain-of-thought.

It is working memory + task context, explicitly defined by a schema.

⸻

## 2. Why Agent State Changes Every Step

In an agentic workflow, each step performs one or more of the following:
- Consumes inputs (user input, tool output, retrieved context)
- Updates internal variables (decisions, flags, counters)
- Determines the next action

Each of these operations mutates the state.

This is expected and desirable.

If your agent state is not changing, your agent is not reasoning.

⸻

## 3. The Logging Question (The Common Confusion)

"If the agent state updates at every reasoning step, do we log each update?"

Answer:
Yes, state updates happen every step.
No, we do not log every update.

This distinction is foundational.

⸻

## 4. State vs Logging (Critical Separation of Concerns)

### Agent State
- Exists to enable reasoning
- Optimized for speed and relevance
- Frequently mutated
- Often ephemeral

### Logging
- Exists to observe behavior
- Optimized for debugging, evaluation, compliance
- Selective
- Persistent

Logging observes state — it does not define state.

⸻

## 5. Four Classes of State Updates (What Gets Logged)

Not all state changes are equal.
They fall into four distinct classes.

⸻

### 5.1 Ephemeral Reasoning Updates

(Do NOT Log)

These are internal, high-frequency, short-lived changes.

Examples:
- Intermediate reasoning variables
- Scratchpad calculations
- Partial thoughts
- Token-level reasoning

**Characteristics**
- High volume
- Low long-term value
- Often sensitive
- Lives only in the context window

**Rule**
- ❌ Do not log
- ❌ Do not persist
- ❌ Do not expose

These updates exist only to reach the next step.

⸻

### 5.2 Decision-Relevant Updates

(Log as Events)

These updates change what the agent will do next.

Examples:
- Plan changes
- Branch selection
- Retry decisions
- Confidence threshold crossings

**What to log**
- The decision, not the full state
- The reason, summarized
- The next action

Example (conceptual):

```json
{
  "event": "decision",
  "decision_type": "plan_update",
  "reason": "initial retrieval returned empty",
  "next_action": "broaden_query"
}
```

These logs form the decision trace of the agent.

⸻

### 5.3 External Interaction Updates

(Always Log)

Any state change caused by interaction with the outside world must be logged.

Examples:
- Tool calls
- API requests
- Database reads/writes
- User-visible outputs

**Why this is mandatory**
- Debugging
- Reproducibility
- Evaluation
- Compliance
- Cost tracking

**Rule**
- ✅ Log inputs
- ✅ Log outputs (or hashes/summaries)
- ✅ Log success/failure

⸻

### 5.4 Memory-Qualifying Updates

(Log + Persist)

Some state updates are promotion events—they graduate from short-term state to long-term memory.

Examples:
- User preferences discovered
- Repeated failure patterns
- Successful strategies
- Corrections from the user

These updates:
- Are logged as events
- Are written to a memory store
- Influence future runs

Conceptually:

```json
{
  "event": "memory_promotion",
  "memory_type": "episodic",
  "summary": "User prefers diagram-based explanations"
}
```

⸻

## 6. Event-Based Logging (The Correct Pattern)

### ❌ Anti-Pattern: Full State Dumps

Dumping the entire agent state at every step:
- Creates noise
- Increases storage costs
- Risks leaking sensitive data
- Makes evaluation harder, not easier

### ✅ Best Practice: Event-Based Logging

Log state transitions, not state snapshots.

Think in terms of:
- What changed?
- Why did it change?
- What did it cause?

This yields clean traces that can be:
- Analyzed
- Visualized
- Evaluated
- Audited

⸻

## 7. Where This Lives in the 4-Phase / Layered Model

This design fits cleanly into the layered mental model used throughout Professor Bone Lab:

| Layer | Responsibility |
|-------|---------------|
| Data Layer | Logs, traces, metrics |
| Logic Layer | State mutation rules |
| Knowledge Layer | Long-term memory |
| Orchestration | When to log or persist |
| Execution | Tools and actions |

Key insight:
Logging is orthogonal to reasoning.

⸻

## 8. A Reusable Design Rule

Every reasoning step updates state.
Only meaningful transitions get logged.

Or, more simply:

**Log decisions, not thoughts.**

This rule scales from:
- Single-agent notebooks
- To multi-agent systems
- To production-grade deployments

⸻

## 9. Why This Matters (Big Picture)

Correct state/log separation enables:
- Clean evaluation pipelines
- Safe memory systems
- Debuggable agents
- Scalable multi-agent architectures

Agents that log everything:
- Drown in noise

Agents that log nothing:
- Cannot improve

Good agents remember what mattered.

⸻

## 10. How This Note Is Meant to Be Used

This document:
- Supports the Agent State notebooks
- Informs LangGraph / orchestration design
- Provides conceptual grounding for:
  - Logging
  - Memory
  - Evaluation
  - Observability

It is intentionally framework-agnostic.

⸻

## Next Natural Extensions (Optional)

Future notes can build on this by covering:
- StateEvent schema design
- Checkpointing vs logging
- State ↔ Memory promotion pipelines
- Multi-agent shared state considerations
- Evaluation metrics tied to state transitions
