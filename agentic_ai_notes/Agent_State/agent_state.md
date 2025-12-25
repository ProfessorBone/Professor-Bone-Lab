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

### When does the agent query episodic + procedural memory?

Think of memory retrieval as an **internal tool call** that should happen at **decision points**, not continuously.

The most common (and reliable) retrieval moments are:

1. **Task start (bootstrap retrieval)**

* Right after you parse the user goal / task objective, before you plan.
* **Episodic:** "Have I solved something like this before? What worked/failed?"
* **Procedural:** "Do I have a known playbook/workflow for this task type?"

2. **Before planning a step (pre-action retrieval)**

* Each time you're about to choose the next tool/step.
* **Procedural dominates** here: grab the best-known procedure/checklist for the step you're entering.

3. **When uncertainty is high**

* Triggers: low confidence, missing requirements, ambiguous user intent, conflicting signals.
* **Episodic** can provide "similar past cases" that reduce ambiguity (what questions to ask, what assumptions were wrong last time).

4. **After a failure / exception (recovery retrieval)**

* Tool fails, validation fails, output rejected, or you hit a retry condition.
* **Episodic:** "Last time this API failed, what fallback worked?"
* **Procedural:** "What's the standard retry/backoff/alternative path?"

5. **When a phase changes**

* Example: research → drafting → review.
* Pull phase-specific procedures (procedural) and relevant prior outcomes (episodic).

Rule of thumb:

* **Procedural retrieval = "how to do it"** (playbooks, best practices, checklists).
* **Episodic retrieval = "what happened before"** (cases, outcomes, gotchas, user preferences learned in context).

---

## Concrete "memory write → pointer replace → prune" flow (LangGraph-style) + where query fits

### Tiny state schema (TypedDict)

```python
from typing import TypedDict, Optional, List, Literal, Dict, Any

class AgentState(TypedDict, total=False):
    goal: str
    phase: Literal["bootstrap", "plan", "act", "reflect", "final"]

    # Working set (can bloat if you don't prune)
    working_notes: str
    tool_payload: Dict[str, Any]          # potentially large
    draft_output: str

    # Memory retrieval results (keep small)
    retrieved_procedures: List[Dict[str, Any]]  # summaries/IDs, not full docs
    retrieved_episodes: List[Dict[str, Any]]    # summaries/IDs, not full docs

    # Memory pointers (after commit)
    episodic_memory_ids: List[str]
    procedural_memory_ids: List[str]

    # Telemetry-friendly
    last_decision: str
    confidence: float
```

### Node sequence (high level)

Here's a clean pattern that keeps state lean:

1. **ingest_goal**
2. **memory_query_bootstrap**  ← (query point)
3. **plan_next_step**
4. **act_tool_call**
5. **reflect_and_classify_update**
6. **memory_commit_if_qualifying**  ← (write)
7. **pointer_replace**             ← (replace payload with IDs + summaries)
8. **prune_state**                 ← (evict large fields)
9. loop back to **memory_query_pre_action** when needed  ← (query point)
10. **finalize**

### Pseudocode for the key nodes

#### (A) Memory query nodes (episodic + procedural)

```python
def memory_query_bootstrap(state: AgentState) -> AgentState:
    # INTERNAL TOOL CALLS (conceptually)
    # procedural: get best playbook(s) for this goal/phase
    state["retrieved_procedures"] = [
        {"id": "proc_123", "summary": "Use workflow: parse→plan→tool→validate→summarize", "score": 0.82}
    ]
    # episodic: get similar past task experiences
    state["retrieved_episodes"] = [
        {"id": "epi_998", "summary": "Similar task: tool X failed; fallback to Y worked", "score": 0.71}
    ]
    return state


def memory_query_pre_action(state: AgentState) -> AgentState:
    # Trigger this selectively: low confidence, phase change, or failure
    if state.get("confidence", 1.0) < 0.7 or state.get("last_decision") == "retry":
        # fetch targeted items, keep them small (summaries + IDs)
        state["retrieved_episodes"] = [{"id": "epi_771", "summary": "Ask for missing constraint Z first", "score": 0.76}]
    return state
```

#### (B) Memory commit node (write long-term, then shrink working set)

```python
def memory_commit_if_qualifying(state: AgentState) -> AgentState:
    # Example classification output from reflect step:
    # state["tool_payload"] might be huge; only some parts qualify for memory

    # Suppose we decide this is episodic:
    new_epi_id = "epi_new_001"  # returned by your memory store
    state.setdefault("episodic_memory_ids", []).append(new_epi_id)

    # Suppose we also learned a procedure:
    new_proc_id = "proc_new_010"
    state.setdefault("procedural_memory_ids", []).append(new_proc_id)

    return state
```

#### (C) Pointer replace + prune (anti-bloat)

```python
def pointer_replace(state: AgentState) -> AgentState:
    # Replace heavy payloads with compact references
    if "tool_payload" in state:
        state["tool_payload"] = {
            "stored": True,
            "ref": "blob_555",  # e.g., object storage key
            "summary": "Tool returned 42 records; top 3 relevant saved."
        }
    return state


def prune_state(state: AgentState) -> AgentState:
    # Hard eviction: remove anything not needed for next decision cycle
    # Keep goal, phase, confidence, memory pointers, and small retrieval summaries.
    for k in ["working_notes", "draft_output"]:
        if k in state and len(state[k]) > 2000:
            state[k] = state[k][:500] + " ...[truncated]"
    # Optionally remove large fields entirely once committed
    # del state["tool_payload"]  # if you truly don't need it anymore
    return state
```

---

## Where the query "fits" in the loop (simple rule)

* **Always query at bootstrap.**
* Then query again only on triggers:

  * **phase change**
  * **confidence < threshold**
  * **error/retry**
  * **before a high-cost tool call**

That gives you the benefits of memory without turning state into a junk drawer.

⸻

## Next Natural Extensions (Optional)

Future notes can build on this by covering:
- StateEvent schema design
- Checkpointing vs logging
- State ↔ Memory promotion pipelines
- Multi-agent shared state considerations
- Evaluation metrics tied to state transitions
