# System Prompt Architecture for Agentic AI Systems

## A Practitioner's Framework for Modular Prompt Design, Routing Policy, and State Integration

---

**Author:** Clarence "Faheem" Downs  
**Institution:** Johns Hopkins University — Agentic AI Certificate Program  
**Date:** December 2025  
**Document Type:** Engineering Handbook (Build Guide Integration Module)  
**Status:** Proposed heuristics (requires validation in your deployment context)

---

## Abstract

This document presents a practitioner-oriented framework for designing system prompts in LLM-based agent systems. Rather than claiming formal theoretical contributions, it offers engineering heuristics derived from publicly documented production practices and integrated with the AI Agent Build Guide's workflow (state, orchestration, evaluation).

The framework addresses four practical problems:
1. How to structure system prompts for maintainability and version control
2. How to coordinate system prompts with agent state schemas (contract-driven design)
3. How to incorporate routing/policy logic explicitly (tool use, clarifying questions, handoffs, stop conditions)
4. How to avoid state bloat through disciplined "memory write → pointer replace → prune" patterns

All recommendations are labeled as proposed heuristics requiring empirical validation per system.

**Keywords:** System Prompt, Context Engineering, Prompt Architecture, Agent State, Routing Policy, LangGraph, Agentic AI, Build Workflow

**Scope Limitation:** This framework applies to LLM-based agents orchestrated by workflow frameworks (e.g., LangGraph). It does not cover non-LLM agents, RL agents, or systems without explicit prompt contexts.

---

## 1. Introduction

### 1.1 The Problem This Document Addresses

The AI Agent Build Guide offers strong coverage of state management, memory architecture, orchestration patterns, and evaluation. However, system prompt design is often reduced to a minimal template (useful for Tier 0), which becomes insufficient when systems require:
- Multiple tools with selection logic
- Dynamic behavior based on user context and task phase
- Multi-agent handoffs and coordination contracts
- Prompt versioning, A/B testing, and regression safety
- Tight coupling with state schema and orchestration routes

This module upgrades system prompts from "a string" to a first-class engineered artifact: modular, testable, versioned, and contract-aligned with state and routing.

### 1.2 What This Document Is (and Is Not)

**This document IS:**
- An engineering handbook with proposed heuristics
- A Build Guide integration module for prompt architecture
- A set of patterns observed in documented systems and frameworks
- A practical blueprint designed to be implemented and tested

**This document is NOT:**
- A peer-reviewed academic contribution
- An empirically validated theory of prompt design
- A replacement for testing and evaluation in your deployment
- A universal best-practice for all agents and all models

### 1.3 Methodology and Limitations

This module synthesizes:
1. Publicly documented agent guidance (notably "context engineering" framing)
2. Research on long-context behavior (attention/position effects)
3. Memory-tiering concepts (e.g., memory blocks / paging patterns)
4. Practical orchestration realities (node prompts, routing policy, tool contracts)
5. Integration into the Build Guide's 17-step workflow and tier model

**Limitations:**
- No controlled experiments were conducted
- Survivorship bias (successful systems are documented more than failed ones)
- Rapidly evolving model behavior and tooling
- Context-dependent recommendations that must be validated per system

---

## 2. Background: What the Research Actually Says

This section summarizes findings from verifiable sources and labels inferences explicitly.

### 2.1 Context Engineering vs. Prompt Engineering

Modern engineering practice increasingly frames success as **context engineering**: delivering the right information, tools, and framing at the right time (system prompt is only one component).

**Implication:** System prompt design cannot be separated from:
- State design
- Retrieval design
- Tool schemas
- Routing policy
- Evaluation gates

### 2.2 Attention and Position Effects ("Lost in the Middle")

Long-context models often show degraded recall for information buried in the middle of long inputs.

**Implication:**
- Put identity + non-negotiable constraints near the top
- Put output format + final reminders near the end
- Keep the middle for examples and dynamic injected context
- Keep prompts shorter than you think you can get away with

### 2.3 Memory Blocks and Paging Patterns

Memory-tiering patterns ("fast working memory" vs "slow archival memory") inspire a key production idea:
- Working memory (state) must be bounded and selective
- Long-term memory stores durable facts, experiences, and procedures
- Retrieval determines what gets promoted back into working context

**Implication:** Don't treat "logging everything" as "remembering everything." Design a memory lifecycle.

### 2.4 Multi-Agent Coordination Challenges

Multi-agent systems amplify failures in:
- Role ambiguity
- Handoff format mismatches
- Shared context overload
- Tool/authority confusion
- Lack of stop conditions ("infinite delegation loops")

**Implication:** Prompts must include:
- Role boundaries
- Handoff protocols
- Shared-state visibility rules
- Routing/stop policy in plain language
- State contracts and validation checks

---

## 3. The System Prompt Within the Context Window

### 3.1 Architectural Reality: Reconstruction Per Call

**Fact:** The full context window is reconstructed for every LLM inference call.

Therefore:
1. The LLM has no persistent internal memory across calls
2. "Persistence" lives in the orchestrator (state + memory stores)
3. The system prompt is re-injected each call (as part of context assembly)

### 3.2 Decomposing "Context" Into Components

To avoid confusion, treat the runtime context as four distinct components:

**1. System Prompt (Stable Core)**
- Identity, constraints, policy, capabilities guidance, output format norms

**2. Messages (Conversation Trace)**
- Running interaction history or summaries (often stored in state)

**3. Injected Context (Dynamic)**
- Retrieved docs, task phase, user prefs, current state highlights

**4. Tool Schema**
- Function specs, tool affordances, input/output structure

**Heuristic:** Keep the system prompt stable and minimal; push dynamic task content into injected context.

---

## 4. Prompt Architecture Model: Six Functional Blocks

This taxonomy is a pragmatic decomposition, not a claim about the only correct structure.

### Block 1: Identity

**Purpose:** Establishes who the agent is and how it communicates.
- Role definition
- Domain expertise
- Communication style + collaboration posture
- Non-negotiable identity invariants

### Block 2: Capabilities

**Purpose:** Defines what the agent can do and what tools exist.
- Available tools and intended use
- Tool input/output expectations
- Tool selection criteria (high-level)

### Block 3: Constraints

**Purpose:** Defines boundaries and safety rails.
- Prohibited actions
- Refusal protocols
- Operational constraints (limits, scope boundaries)
- Compliance guardrails (if applicable)

### Block 4: Policy / Routing (NEW)

**Purpose:** Makes routing explicit: what to do next, when, and why.
- When to ask clarifying questions
- When to call tools
- When to handoff to another agent
- Stop conditions (when to finalize)
- Retry rules / fallback modes

This block prevents the prompt from relying on "implied behavior."

### Block 5: Context (Dynamic Injection)

**Purpose:** Provides task-relevant information, injected just-in-time.
- User prefs (session-scoped)
- Task phase (turn-scoped)
- Retrieved evidence (task/turn-scoped)
- Memory recall results (episodic/procedural pointers)

### Block 6: Format

**Purpose:** Specifies output structure and response standards.
- JSON schema / markdown shape
- Citation requirements
- Confidence / uncertainty conventions
- "No extra text" constraints (if structured output required)

---

## 5. Prompt ↔ State Co-Design (and How to Avoid State Bloat)

### 5.1 The Coordination Problem

If the prompt references {user_expertise} but state does not contain user_expertise, the system becomes unreliable.

**Key point:** Prompts and state schemas are coupled artifacts.

### 5.2 Core Heuristics

**Heuristic 5.2.1:** Design prompt + state in the same session.

**Heuristic 5.2.2:** Every {placeholder} must map to a state field or a computed injection product.

**Heuristic 5.2.3:** Every behaviorally significant state value must have corresponding prompt guidance.

### 5.3 The Anti-Bloat Rule (STRONG FORM)

If a value exists only to render a prompt, compute it at injection-time instead of storing it in persistent state.

**Examples:**
- ✅ Store: retrieved_doc_ids, tool_results_refs, memory_ids
- ✅ Compute at injection: retrieved_doc_summaries, ranked_snippets, context_block_text
- ❌ Avoid storing: full raw document chunks in persistent state (unless cross-node reuse requires it)

This is how you keep state from becoming a garbage dump.

---

## 6. Prompt–State Contract (Artifact You Can Enforce)

Treat the contract like code: it is reviewable, testable, enforceable.

### 6.1 Contract Spec Template

**Prompt-State Contract**
- Placeholders
- Source of truth
- Population mechanism
- Required / optional
- Failure behavior
- Invariants

### 6.2 Example Contract (Minimal)

| Placeholder / Input | Source | Required? | Population Mechanism | Failure Behavior |
|---------------------|--------|-----------|---------------------|------------------|
| {task_phase} | state.task_phase | Yes | Updated by router/orchestrator | Hard fail before LLM call |
| {user_expertise} | state.user_expertise | No | Session init (default: "intermediate") | Default value |
| {retrieved_context} | computed | No | Rank → summarize top-k from state.doc_ids | Empty block allowed |
| {episodic_hits} | computed | No | Memory query by state.query → return top hits | Empty block allowed |

**Invariants**
- task_phase ∈ {planning, researching, synthesizing, complete}
- Tool calls must be recorded as events and optionally referenced by IDs in state

---

## 7. System Prompt Templates (Modular)

Below is a modular template that matches the six-block architecture.

### 7.1 Base System Prompt (Composable)

```
# IDENTITY
You are {agent_role}. You help the user accomplish {outcome}.
Your tone: {tone}. Your default: be clear, structured, and practical.

# CAPABILITIES
You can use tools when needed. Prefer direct answers when confident.
Available tools:
{tool_catalog}

# CONSTRAINTS
- Stay within scope: {scope}
- Do not fabricate sources or tool results.
- If uncertain, say so and propose a next step.
- If user request is unsafe/out-of-policy, refuse and suggest safe alternatives.

# POLICY / ROUTING
Follow this decision policy:
1) If requirements are unclear: ask 1–3 targeted questions.
2) If the answer depends on external or changing info: use tools.
3) If retrieved evidence is required: retrieve first, then answer.
4) If this is a multi-agent system: delegate only when a specialist is required.
5) Stop condition: provide final output when success criteria are met.

# CONTEXT (Injected at runtime)
Task phase: {task_phase}
User preferences: {user_prefs}
Relevant retrieved context:
{retrieved_context}
Relevant episodic/procedural memory hits:
{memory_hits}

# FORMAT
Return output as: {output_format}
Do not include extra text outside the required format.
```

---

## 8. Where This Fits in the 17-Step Build Guide Workflow

System prompt architecture is not "one step." It spans multiple steps.

| Build Guide Step | Prompt Architecture Activity |
|------------------|------------------------------|
| Step 1 (PEAS) | Draft Identity + Constraints + Success criteria language |
| Step 8 (State Schema) | Co-design prompt placeholders + state fields |
| Step 9 (Node Functions / Tools) | Define Capabilities + tool selection rules |
| Step 12–13 (Graph + Routing) | Define Policy/Routing block aligned to edges |
| Step 15 (Testing) | Behavioral + format compliance + state injection tests |
| Step 16 (Iteration) | Regression tests, A/B prompt variants, tighten bloat controls |

---

## 9. Context Budgeting (How to Keep Prompts Effective)

A common failure mode is shoving everything into prompt/context.

### 9.1 Context Budget Heuristics
- Keep system prompt stable and relatively short
- Inject only the "minimum sufficient" dynamic context
- Prefer ranked, summarized context over raw dumps
- Use memory pointers and IDs instead of full payloads

### 9.2 Practical Allocation (Rule of Thumb)
- **System prompt:** stable behavior + policies (tight)
- **Injected context:** top-k ranked evidence + short memory hits
- **Messages:** summarize older history; keep recent turns verbatim
- **Tool schema:** keep tool definitions precise and compact

---

## 10. Multi-Agent Prompt Considerations (Supervisor + Workers)

### 10.1 Supervisor Prompt Additions

Supervisor should include:
- Worker registry + when to delegate
- Handoff format contract
- Stop conditions (avoid infinite loops)
- Shared state visibility policy (only inject what each worker needs)

### 10.2 Worker Prompt Additions

Worker should include:
- "Responsibilities" and "Not responsibilities"
- Strict input format and output format
- Tool scope constraints
- Return-control rules ("if blocked, report why and stop")

**Heuristic:** Tier 2+ agents should use per-role prompts, not one mega prompt.

---

## 11. Testing Prompts Like Code

Prompt changes should go through a regression suite.

### 11.1 Test Categories
1. **Behavioral compliance** (identity, policy adherence, refusal correctness)
2. **Format compliance** (JSON schema, no extra text, citations)
3. **State integration** (placeholders filled; missing keys handled correctly)
4. **Tool discipline** (tool calls only when policy conditions trigger)
5. **Bloat resistance** (context remains bounded; payloads not ballooning)

### 11.2 Minimum Regression Gate (Recommended)
- Golden set of 20–100 representative cases
- Must not regress safety
- Must not increase average token usage beyond threshold (unless justified)
- Must improve target metric (accuracy, task success rate, formatting compliance)

---

## 12. Memory Lifecycle: "Write → Pointer Replace → Prune"

This is the anti-bloat memory pattern that keeps state lean.

### 12.1 What Gets Stored Where
- **State:** pointers, IDs, short summaries, control flags (bounded working memory)
- **Logs/Telemetry:** detailed events and traces (observability)
- **Long-term memory store:** episodic/procedural objects retrievable by query

### 12.2 The Pattern
1. Write memory event to long-term store (episodic/procedural)
2. Replace large in-state payload with a pointer (memory_id, doc_id, event_id)
3. Prune state: keep only small summaries + references + routing metadata

---

## 13. Summary and Recommendations

### 13.1 Key Takeaways
1. Context is reconstructed every call; persistence lives in orchestrator + memory stores
2. Prompts and state must be co-designed via enforceable contracts
3. Add an explicit Policy/Routing block to prevent implied behavior failures
4. Prevent state bloat using compute-at-injection, pointers, and pruning
5. Store prompts in version control and test them like code

### 13.2 Quick Checklist
- [ ] Identity block is stable and clear
- [ ] Capabilities describe tools + usage criteria
- [ ] Constraints are non-negotiable and auditable
- [ ] Policy/Routing defines tool use, clarifications, handoffs, stop conditions
- [ ] Context is injected just-in-time and bounded
- [ ] Format matches output schema exactly
- [ ] Prompt-state placeholders have a contract and tests
- [ ] Bloat rule enforced: compute injection-only values, store pointers in state

---

## Appendix A: Minimal Templates

### A.1 Tier 0 (Single-path)

```
You are a {role}. Your job: {outcome}.
Constraints:
- Stay in scope.
- If unsure, say so.

Policy:
- Ask 1 question if needed.
- Otherwise answer directly.

Format:
{format_instructions}
```

### A.2 Tier 1 (RAG + branching)

```
IDENTITY: {role}
CAPABILITIES: retrieve_docs, web_search
CONSTRAINTS: cite sources; don't fabricate
POLICY: retrieve if needed; otherwise answer
CONTEXT: {retrieved_context}
FORMAT: {json_schema}
```

### A.3 Tier 2 (Multi-agent worker)

```
You are the {worker_role}.
Responsibilities: {responsibilities}
Not responsibilities: {boundaries}
Input format: {task_json}
Output format: {result_json}
Constraints: stay in role; return control if blocked.
```

---

## Appendix B: Verified References (Representative)

- Liu et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172
- Packer et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560
- Wang et al. (2023). A Survey on Large Language Model Based Autonomous Agents. arXiv:2308.11432
- Anthropic (2024). Building Effective Agents. (Engineering guidance)
- LangChain (2024). LangGraph Documentation.

---

**Document Version:** 2.1 (Rewritten with routing + contract + anti-bloat upgrades)  
**Last Updated:** December 2025  
**License:** Educational use permitted with attribution
