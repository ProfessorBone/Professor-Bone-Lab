# System Prompt Architecture for Agentic AI Systems

## A Practitioner's Framework for Modular Prompt Design and State Integration

---

**Author:** Faheem Siddiqui  
**Institution:** Johns Hopkins University — Agentic AI Certificate Program  
**Date:** December 2025  
**Document Type:** Engineering Handbook (Build Guide Integration Module)

---

## Abstract

This document presents a practitioner's framework for designing system prompts in LLM-based agents. Rather than claiming formal theoretical contributions, it offers **engineering heuristics** derived from analysis of publicly documented production systems and integration with the AI Agent Build Guide's established workflow. The framework addresses three practical problems: (1) how to structure system prompts for maintainability, (2) how to coordinate system prompts with agent state schemas, and (3) where system prompt design fits within the 17-step agent build workflow. All recommendations are labeled as proposed heuristics requiring empirical validation in specific deployment contexts. The document is designed as an integration module for the AI Agent Build Guide, providing step-level guidance that the Guide currently lacks for prompt architecture.

**Keywords:** System Prompt, Prompt Engineering, Agent State, LangGraph, Agentic AI, Build Workflow

**Scope Limitation:** This framework applies to LLM-based agents using orchestration frameworks like LangGraph. It does not address non-LLM agents, reinforcement learning agents, or systems without explicit system prompts.

---

## 1. Introduction

### 1.1 The Problem This Document Addresses

The AI Agent Build Guide provides comprehensive coverage of state management, memory architecture, and orchestration patterns. However, it offers only minimal guidance on system prompt design — a four-line template in Appendix A1:

```
You are <ROLE>, serving <AUDIENCE>. Your job: <OUTCOME>.
Follow the rules:
1) Output must match schema exactly.
2) Use tools only when needed.
3) Refuse if request is unsafe/out-of-scope; suggest alternatives.
4) Think step-by-step but return only the final JSON.
```

This template is adequate for Tier 0 agents but insufficient for production systems that require:

- Multiple tools with complex selection logic
- Dynamic behavior based on user context
- Coordination with state schemas
- Multi-agent handoff protocols
- Versioning and A/B testing

This document fills that gap by providing structured guidance for system prompt architecture that integrates with the Build Guide's existing frameworks.

### 1.2 What This Document Is (and Is Not)

**This document IS:**
- An engineering handbook with proposed heuristics
- An integration module for the AI Agent Build Guide
- A collection of patterns observed in documented production systems
- A starting point requiring validation in your specific context

**This document is NOT:**
- A peer-reviewed academic contribution
- A formal theory of prompt architecture
- An empirically validated framework (no controlled experiments were conducted)
- A replacement for testing in your deployment environment

### 1.3 Methodology and Limitations

The patterns in this document derive from:

1. **Publicly documented system prompts** from Anthropic's Claude system prompt (leaked/published versions), OpenAI's documentation, and open-source agent frameworks
2. **Engineering blog posts** from LangChain, Letta, and Anthropic (cited where used)
3. **The author's implementation experience** building agents in the Johns Hopkins Agentic AI program
4. **Synthesis with the AI Agent Build Guide** to ensure architectural consistency

**Limitations:**
- No controlled experiments comparing architectural approaches
- Survivorship bias: analysis of successful systems, not failed ones
- Rapidly evolving field: patterns may become outdated
- Context-dependent: what works for one agent may not generalize

---

## 2. Background: What the Research Actually Says

This section summarizes findings from verifiable sources. Claims are attributed; speculation is labeled.

### 2.1 Context Engineering vs. Prompt Engineering

Anthropic's engineering team has articulated a shift in framing:

> "Context engineering [is] the discipline of building dynamic systems that provide the right information and tools to an LLM at the right time, in the right format, and with the right framing for it to successfully accomplish a task."

*Source: Anthropic Engineering Blog, "Building Effective Agents," 2024*

This framing emphasizes that the system prompt is one component of the broader context window, which also includes:
- Message history
- Retrieved documents
- Tool definitions
- Runtime state

**Implication for this document:** System prompt design cannot be separated from state design. They must be co-designed.

### 2.2 Attention and Position Effects

Liu et al. (2023) demonstrated that LLMs exhibit degraded recall for information in the middle of long contexts:

> "Performance is often highest when relevant information occurs at the very beginning or end of the input context, and degrades significantly when models must access relevant information in the middle of long contexts."

*Source: Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," arXiv:2307.03172*

**Implication for this document:** Critical instructions (identity, constraints) should be placed at the beginning of the system prompt. Reminders can be placed at the end. Examples and context can occupy the middle.

### 2.3 Memory Blocks (Letta/MemGPT)

Packer et al. (2023) introduced the concept of "memory blocks" — discrete, labeled segments of the context window that agents can read and edit:

> "MemGPT manages different memory tiers... using an LLM-based operating system that pages information between fast and slow memory."

*Source: Packer et al., "MemGPT: Towards LLMs as Operating Systems," arXiv:2310.08560*

The Letta framework (production implementation of MemGPT) uses specific block types:
- **Core Memory:** Agent identity and user information (analogous to system prompt)
- **Archival Memory:** Long-term storage (analogous to vector DB)
- **Recall Memory:** Conversation history (analogous to message state)

**Implication for this document:** The system prompt can be conceptualized as a structured set of blocks with different update frequencies and ownership models.

### 2.4 Multi-Agent Coordination Challenges

Wang et al. (2023) surveyed LLM-based autonomous agents and identified coordination as a key challenge:

> "Multi-agent collaboration introduces challenges in role assignment, communication protocols, and shared memory management."

*Source: Wang et al., "A Survey on Large Language Model Based Autonomous Agents," arXiv:2308.11432*

The survey does not quantify failure rates attributable to prompt design (such quantification would require controlled experiments not present in the literature).

**Implication for this document:** Multi-agent systems require explicit role definitions, handoff protocols, and shared context specifications in their system prompts.

---

## 3. The System Prompt Within the Context Window

### 3.1 Architectural Reality: Reconstruction Per Call

A critical architectural fact that shapes all design decisions:

> **The context window is reconstructed for every LLM inference call.**

The Build Guide states this clearly:

> "CONTEXT WINDOW (LLM Working Memory)... RESET EVERY LLM CALL"
> — AI Agent Build Guide, Memory Architecture diagram

This means:
1. The system prompt is not "persistent" in any meaningful sense — it is re-injected on every call
2. State persists in the orchestrator (e.g., LangGraph), not in the LLM
3. The LLM has no memory of previous calls except what is explicitly included in the context

**Architectural diagram (corrected):**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR (LangGraph)                        │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    PERSISTENT STATE                              │   │
│   │   • messages: list[BaseMessage]                                  │   │
│   │   • retrieved_docs: list[str]                                    │   │
│   │   • task_progress: dict                                          │   │
│   │   • (other fields per your state schema)                         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    │ On each node execution:             │
│                                    │ 1. Read relevant state fields       │
│                                    │ 2. Construct context window         │
│                                    │ 3. Call LLM                         │
│                                    │ 4. Parse response                   │
│                                    │ 5. Update state                     │
│                                    ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              CONTEXT WINDOW (Constructed Per Call)               │   │
│   │                                                                  │   │
│   │   ┌──────────────────────────────────────────────────────────┐  │   │
│   │   │  SYSTEM PROMPT                                           │  │   │
│   │   │  (Loaded from configuration or built by PromptBuilder)   │  │   │
│   │   └──────────────────────────────────────────────────────────┘  │   │
│   │   ┌──────────────────────────────────────────────────────────┐  │   │
│   │   │  MESSAGES (from state.messages)                          │  │   │
│   │   └──────────────────────────────────────────────────────────┘  │   │
│   │   ┌──────────────────────────────────────────────────────────┐  │   │
│   │   │  INJECTED CONTEXT (retrieved docs, current state fields) │  │   │
│   │   └──────────────────────────────────────────────────────────┘  │   │
│   │                                                                  │   │
│   │   → Sent to LLM → Response returned → Context window discarded  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Two Axes of Prompt Classification

The original paper's "static vs. dynamic" dichotomy was too simplistic. System prompt content varies along two independent axes:

**Axis 1: Update Frequency**
| Category | Update Frequency | Examples |
|----------|------------------|----------|
| Stable | Rarely (versioned releases) | Agent identity, safety constraints |
| Session-scoped | Per session | User preferences loaded at session start |
| Task-scoped | Per task | Task-specific instructions, retrieved context |
| Turn-scoped | Per LLM call | Current state values, recent messages |

**Axis 2: Ownership**
| Owner | Who Controls It | Examples |
|-------|-----------------|----------|
| Developer | Engineering team | Identity, constraints, tool definitions |
| Operator | Deployment configuration | Environment-specific settings |
| Agent | Agent's own decisions | Self-updated working memory (in Letta-style systems) |
| User | User input/preferences | Communication style, expertise level |

**Key insight:** Something can be "developer-controlled" but "turn-scoped" (e.g., a prompt that injects different instructions based on current routing state). Something can be "user-controlled" but "stable" (e.g., user preferences set once and rarely changed).

### 3.3 Where the System Prompt Lives (Build-Time vs. Runtime)

The system prompt exists in two forms:

**1. Build-Time Artifact (Configuration)**
- Stored as files, database records, or code constants
- Version-controlled alongside application code
- Subject to review, testing, and deployment pipelines

**2. Runtime Artifact (Constructed Context)**
- Assembled by the orchestrator before each LLM call
- May incorporate state values, retrieved context, and conditional logic
- Discarded after the call completes

**Proposed heuristic:** Treat system prompts like code. Store them in version control. Review changes. Test before deploying. The runtime construction process should be deterministic and reproducible.

---

## 4. A Taxonomy of System Prompt Components

### 4.1 Methodology Note

The following taxonomy emerges from analysis of:
- Anthropic's published Claude system prompt structure
- OpenAI's prompt engineering documentation
- LangChain's agent prompt templates
- The author's implementation experience

This is a **proposed categorization**, not an empirically validated decomposition. Alternative categorizations are valid.

### 4.2 Five Functional Categories

Rather than claiming seven (or four) distinct "components," this revision proposes five **functional categories** that can overlap and be combined:

#### Category 1: Identity Block
**Purpose:** Establishes who the agent is.

**Contents:**
- Role definition ("You are a...")
- Expertise domain
- Communication style
- Relationship to user

**Characteristics:**
- Typically stable (changes with agent version, not per-task)
- Developer-owned
- Should appear early in the prompt (position effects)

**Example:**
```
You are a research assistant specializing in AI and machine learning. 
You communicate in a professional but accessible style, explaining 
technical concepts clearly. You work collaboratively with the user, 
asking clarifying questions when needed.
```

**Build Guide Integration:** This corresponds to defining your agent's PEAS Performance criteria — what does success look like for this agent's role?

#### Category 2: Capabilities Block
**Purpose:** Defines what the agent can do.

**Contents:**
- Available tools and their purposes
- Tool selection guidance
- Expected tool inputs/outputs
- Conditions for tool use

**Characteristics:**
- Stable at the tool level, but tool selection logic may be task-scoped
- Developer-owned (tool definitions) with agent execution
- Critical for reliable tool use

**Example:**
```
You have access to the following tools:

1. web_search: Search the web for current information.
   - Use when: User asks about recent events, current data, or information 
     that may have changed since your training.
   - Do not use when: Question is about stable historical facts or 
     concepts you can explain from training.
   - Input: Search query (1-6 words optimal)
   - Output: List of search results with snippets

2. retrieve_docs: Search the internal knowledge base.
   - Use when: User asks about company-specific information, policies, 
     or previously uploaded documents.
   - Input: Natural language query
   - Output: Relevant document chunks with sources
```

**Build Guide Integration:** This corresponds to PEAS Actuators — the actions your agent can take. Tool definitions should be designed alongside the Tool Schemas in Phase 2, Step 9.

#### Category 3: Constraints Block
**Purpose:** Defines boundaries and safety rails.

**Contents:**
- Prohibited actions
- Content policies
- Operational limits (e.g., max tool calls)
- Refusal protocols

**Characteristics:**
- Should be highly stable (changes require careful review)
- Developer-owned with organizational oversight
- Should appear early in the prompt (priority via position)

**Example:**
```
CONSTRAINTS:
- Never provide medical, legal, or financial advice as if you were 
  a licensed professional. Recommend consulting appropriate experts.
- Do not generate content that could be used to harm individuals.
- If you cannot complete a task, explain why and suggest alternatives.
- Maximum 3 consecutive tool calls without user confirmation.
- If a request seems to violate these constraints, decline politely 
  and explain your reasoning without lecturing.
```

**Build Guide Integration:** Constraints inform your routing logic in Phase 3 — the orchestrator should enforce operational limits that the prompt declares.

#### Category 4: Context Block
**Purpose:** Provides task-relevant information.

**Contents:**
- Retrieved documents
- User preferences
- Session history summaries
- Current state values

**Characteristics:**
- Highly dynamic (task-scoped or turn-scoped)
- May be user-owned (preferences) or agent-populated (retrieved docs)
- Should be populated just-in-time, not stored statically

**Example:**
```
CURRENT CONTEXT:
- User expertise level: intermediate
- Retrieved documents: [3 chunks about transformer architecture]
- Current task: Explain attention mechanisms
- Session summary: User is building a RAG system and has asked 
  about embeddings and vector stores in previous turns.
```

**Build Guide Integration:** Context blocks are populated from your State Schema. Each field in your `AgentState` that gets injected into the prompt should have a corresponding place in the context block.

#### Category 5: Format Block
**Purpose:** Specifies response structure.

**Contents:**
- Output format requirements (JSON, markdown, prose)
- Length guidelines
- Structural requirements (sections, citations)
- Examples of expected output

**Characteristics:**
- Typically stable per node (different nodes may have different formats)
- Developer-owned
- Should appear late in the prompt (close to where generation begins)

**Example:**
```
OUTPUT FORMAT:
Respond with a JSON object matching this schema:
{
  "answer": "Your response to the user's question",
  "sources": ["List of sources used, if any"],
  "confidence": 0.0-1.0,
  "needs_clarification": true/false
}

Do not include any text outside the JSON object.
```

**Build Guide Integration:** Format blocks must align with your Node Output Schemas (Phase 2, Step 9). If you define a Pydantic model for structured output, your format block should describe that schema.

### 4.3 The Relationship Between Categories

These categories are **not mutually exclusive**. A single prompt section might serve multiple purposes:

```
You are a research assistant (IDENTITY) who can search the web 
and retrieve documents (CAPABILITIES). Always cite your sources 
(FORMAT) and never provide medical advice (CONSTRAINTS).
```

The categories are a **lens for analysis**, not a rigid template. Use them to ensure you've covered necessary functions, not as mandatory sections.

---

## 5. Coordinating System Prompts with State Schemas

### 5.1 The Coordination Problem

The original paper introduced a "System Prompt–State Contract" concept. While the term was useful, the formalization was inadequate. This section provides a more precise treatment.

**The core problem:** The system prompt may reference state fields that must be populated by the orchestrator. If the prompt expects `{user_expertise}` but the state schema doesn't include that field, the agent will malfunction.

### 5.2 Proposed Heuristic: Co-Design Prompt and State

**Heuristic 5.2.1:** Define your state schema and system prompt in the same design session. They are coupled artifacts.

**Heuristic 5.2.2:** For every `{placeholder}` in your system prompt, there must be a corresponding field in your state schema with a defined population mechanism.

**Heuristic 5.2.3:** For every state field that affects agent behavior, there should be a corresponding instruction in the system prompt explaining how to use it.

### 5.3 State Field Reference Patterns

When a system prompt references state, it can do so in three ways:

**Pattern A: Direct Injection**

The orchestrator substitutes state values directly into the prompt before sending to the LLM.

```python
# Orchestrator code
def build_context(state: AgentState, base_prompt: str) -> str:
    return base_prompt.format(
        user_expertise=state["user_expertise"],
        retrieved_docs=format_docs(state["retrieved_docs"])
    )

# System prompt template
"""
The user's expertise level is: {user_expertise}

Relevant documents:
{retrieved_docs}

Given this context, answer the user's question.
"""
```

**Pattern B: Behavioral Conditioning**

The system prompt instructs the agent to behave differently based on state values, but the values are visible to the agent rather than substituted.

```
You will receive a context block with the user's expertise level.
- If expertise is "beginner": Use simple language, define technical terms.
- If expertise is "intermediate": Use technical terms with brief context.
- If expertise is "expert": Use domain terminology freely.
```

**Pattern C: State Query Instructions**

The system prompt instructs the agent to check state values as part of its reasoning process (common in Letta-style architectures with explicit memory tools).

```
Before responding, use the check_memory tool to retrieve:
1. User's stated preferences
2. Summary of previous conversation
3. Any relevant prior answers on this topic

Incorporate this context into your response.
```

### 5.4 Anti-Bloat Guardrails

The Build Guide warns against state bloat:

> "If your state schema exceeds 30 fields, you're likely violating the node scratch vs agent state boundary."
> — AI Agent Build Guide, State Scope & Ownership

System prompt design can exacerbate bloat if not carefully managed.

**Heuristic 5.4.1:** Do not add state fields solely to make them available to the prompt. Only add fields that serve coordination purposes across nodes.

**Heuristic 5.4.2:** Prefer injecting computed summaries over raw data. Instead of injecting 10 retrieved document chunks, inject a pre-computed summary or the top 3 most relevant chunks.

**Heuristic 5.4.3:** Apply the "who reads it?" test. If only the LLM reads a state field (it's never used by orchestrator logic or other nodes), consider whether it should be a state field at all or computed at injection time.

### 5.5 Example: State Schema and Prompt Co-Design

**Step 1: Define State Schema (Build Guide Phase 2, Step 8)**

```python
class ResearchAgentState(TypedDict):
    # Core message handling
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Task tracking
    query: str
    task_phase: Literal["planning", "researching", "synthesizing", "complete"]
    
    # Retrieved context
    retrieved_docs: list[str]  # Summaries, not full docs
    doc_sources: list[str]     # Source citations
    
    # User context
    user_expertise: Literal["beginner", "intermediate", "expert"]
    
    # Quality tracking
    confidence: float
    needs_more_research: bool
```

**Step 2: Design System Prompt with State References**

```
# Identity
You are a research assistant specializing in synthesizing information 
from multiple sources into clear, well-cited insights.

# Capabilities
You have access to:
- web_search: Find current information online
- retrieve_docs: Search the internal knowledge base

# Context (populated at runtime)
User expertise level: {user_expertise}
Current task phase: {task_phase}

Relevant documents:
{retrieved_docs}

Sources: {doc_sources}

# Behavioral Instructions
Based on the user's expertise level:
- beginner: Explain concepts from first principles, avoid jargon
- intermediate: Assume foundational knowledge, define advanced terms
- expert: Use domain terminology freely, focus on nuance

Based on the task phase:
- planning: Identify what information is needed, do not answer yet
- researching: Use tools to gather information
- synthesizing: Combine gathered information into a coherent answer
- complete: Provide final answer with citations

# Format
Respond with:
{
  "reasoning": "Your thought process",
  "action": "next_step | answer",
  "content": "Your response or next step description",
  "confidence": 0.0-1.0,
  "needs_more_research": true/false
}
```

**Step 3: Verify Alignment**

| Prompt Reference | State Field | Population Mechanism |
|------------------|-------------|---------------------|
| `{user_expertise}` | `user_expertise` | Set at session start from user profile |
| `{task_phase}` | `task_phase` | Updated by orchestrator routing logic |
| `{retrieved_docs}` | `retrieved_docs` | Populated by retrieval node |
| `{doc_sources}` | `doc_sources` | Populated alongside retrieved_docs |

All references have corresponding fields. Design is aligned.

---

## 6. Integration with the Build Guide Workflow

### 6.1 The 17-Step Workflow

The Build Guide defines a precise build sequence:

```
Phase 1: Data Layer (Steps 1-7)
  1. Define PEAS
  2. Set up environment
  3. Load documents
  4. Configure chunking
  5. Create embeddings
  6. Build vector store
  7. Create retriever

Phase 2: Logic Layer (Steps 8-11)
  8. Initialize LLM
  9. Define state schema + node functions + tools
  10. Implement node logic
  11. Add structured output

Phase 3: Orchestration (Steps 12-13)
  12. Build graph with edges
  13. Add conditional routing

Phase 4: Execution (Steps 14-17)
  14. Compile graph
  15. Test with sample inputs
  16. Iterate and refine
  17. Deploy
```

### 6.2 Where System Prompt Design Fits

System prompt design is **not a single step** but an activity that spans multiple steps:

| Step | System Prompt Activity |
|------|----------------------|
| **Step 1 (PEAS)** | Draft identity block based on Performance criteria; identify capabilities from Actuators |
| **Step 8 (State Schema)** | Co-design state schema and prompt context block; identify what state values the prompt will reference |
| **Step 9 (Node Functions)** | Design per-node prompt variations if nodes require different instructions; align format blocks with output schemas |
| **Step 10 (Node Logic)** | Implement prompt construction in node functions; wire state values to prompt placeholders |
| **Step 12 (Graph Building)** | Ensure routing logic aligns with prompt's behavioral conditioning (e.g., if prompt says "in planning phase, do X," routing must track phases) |
| **Step 15 (Testing)** | Test prompt behavior with representative inputs; validate that state injection works correctly |
| **Step 16 (Iteration)** | Refine prompt based on test failures; add edge case handling |

### 6.3 Prompt Design Checklist by Build Phase

**After Phase 1 (PEAS complete):**
- [ ] Identity block drafted
- [ ] Capabilities block outlined (tools identified)
- [ ] Constraints block drafted (safety requirements from PEAS)

**After Step 8 (State schema defined):**
- [ ] Context block designed with state field references
- [ ] Each `{placeholder}` maps to a state field
- [ ] Population mechanism identified for each referenced field

**After Step 9 (Node functions defined):**
- [ ] Format block aligned with Pydantic output schemas
- [ ] Per-node prompt variations documented (if needed)
- [ ] Tool descriptions in capabilities block match tool implementations

**After Step 12 (Graph built):**
- [ ] Behavioral conditioning in prompt aligns with routing logic
- [ ] Phase/state transitions in prompt match graph edges

**After Step 15 (Testing complete):**
- [ ] Prompt tested with beginner/intermediate/expert user contexts
- [ ] Edge cases documented and handled in constraints block
- [ ] Format compliance verified (outputs match schema)

### 6.4 Tier-Appropriate Prompt Complexity

The Build Guide defines four tiers. Prompt complexity should match tier:

**Tier 0 (Single-Path Agent)**
- Simple identity + single tool description
- Minimal constraints (basic safety)
- No context block (no state beyond messages)
- Use the Appendix A1 template from Build Guide

**Tier 1 (RAG Agent with Branching)**
- Identity block with expertise domain
- Multiple tool descriptions with selection guidance
- Constraints including when to retrieve vs. respond directly
- Context block with retrieved documents
- Format block for structured output

**Tier 2 (Multi-Agent System)**
- Role-specific identity per agent
- Capabilities scoped to agent's responsibilities
- Handoff protocols in constraints block
- Shared context block for inter-agent state
- Format blocks standardized across agents for interoperability

**Tier 3 (Constitutional Agent)**
- Full identity with ethical principles
- Comprehensive capabilities with meta-tools (planning, reflection)
- Extensive constraints with reasoning requirements
- Dynamic context with episodic memory integration
- Format blocks for different output types (plans, reflections, answers)

---

## 7. Multi-Agent Prompt Considerations

### 7.1 Scope Limitation

This section covers only the **hierarchical (supervisor-worker)** pattern referenced in the Build Guide. Other patterns (decentralized, blackboard) require different approaches not covered here.

### 7.2 Supervisor Agent Prompt Structure

The supervisor's prompt must include:

**Worker Registry:**
```
You coordinate the following specialist agents:
1. Researcher: Finds and summarizes information
   - Invoke when: User question requires external information
   - Expects: Clear research question
   - Returns: Summary with sources

2. Writer: Drafts and refines text
   - Invoke when: Information is gathered and needs synthesis
   - Expects: Source material and target format
   - Returns: Polished text

3. Critic: Reviews and improves outputs
   - Invoke when: Draft is complete and needs quality check
   - Expects: Draft text and quality criteria
   - Returns: Feedback and suggestions
```

**Delegation Protocol:**
```
When you receive a user request:
1. Analyze what capabilities are needed
2. Decompose into subtasks if necessary
3. Delegate to appropriate worker(s)
4. Synthesize worker outputs into final response
5. If worker output is insufficient, provide feedback and re-delegate

You do NOT perform research or writing yourself — you coordinate workers.
```

**Handoff Format:**
```
When delegating to a worker, provide:
{
  "worker": "worker_name",
  "task": "Clear description of what to do",
  "context": "Relevant information the worker needs",
  "success_criteria": "How to know when task is complete"
}
```

### 7.3 Worker Agent Prompt Structure

Worker prompts must include:

**Scope Boundaries:**
```
You are the Researcher in a multi-agent system.

YOUR RESPONSIBILITIES:
- Search for information using web_search and retrieve_docs tools
- Evaluate source quality
- Summarize findings with citations

NOT YOUR RESPONSIBILITIES:
- Making final decisions about user requests
- Writing polished final outputs
- Coordinating other agents

If a task falls outside your responsibilities, say so and return 
control to the supervisor.
```

**Input/Output Contract:**
```
You will receive tasks in this format:
{
  "task": "What to research",
  "context": "Background information",
  "success_criteria": "What constitutes complete research"
}

Respond in this format:
{
  "status": "complete | needs_more_info | out_of_scope",
  "findings": "Summary of what you found",
  "sources": ["List of sources"],
  "confidence": 0.0-1.0,
  "notes_for_supervisor": "Any issues or recommendations"
}
```

### 7.4 Shared State Considerations

In multi-agent systems, the shared coordination state (Build Guide Section on State Scope) must be referenced in prompts:

```
SHARED CONTEXT (visible to all agents):
- User query: {user_query}
- Current phase: {current_phase}
- Completed subtasks: {completed_subtasks}
- Pending approvals: {pending_approvals}

Do not duplicate work that has already been completed. Check 
completed_subtasks before beginning new work.
```

**Heuristic 7.4.1:** Each agent's prompt should reference only the shared state fields that agent needs. Do not inject the entire shared state into every prompt.

---

## 8. Practical Implementation Patterns

### 8.1 Prompt Storage and Loading

**Pattern A: File-Based Storage**
```
prompts/
├── base/
│   ├── identity.md
│   ├── constraints.md
│   └── format.md
├── agents/
│   ├── researcher.md
│   ├── writer.md
│   └── supervisor.md
└── nodes/
    ├── planning_node.md
    ├── retrieval_node.md
    └── synthesis_node.md
```

```python
def load_prompt(agent: str, node: str = None) -> str:
    """Load and assemble prompt from files."""
    base_identity = Path("prompts/base/identity.md").read_text()
    base_constraints = Path("prompts/base/constraints.md").read_text()
    
    agent_prompt = Path(f"prompts/agents/{agent}.md").read_text()
    
    if node:
        node_prompt = Path(f"prompts/nodes/{node}.md").read_text()
    else:
        node_prompt = ""
    
    return f"{base_identity}\n\n{base_constraints}\n\n{agent_prompt}\n\n{node_prompt}"
```

**Pattern B: Programmatic Construction**
```python
@dataclass
class PromptBlock:
    name: str
    content: str
    position: int  # Lower numbers appear earlier
    
class PromptBuilder:
    def __init__(self):
        self.blocks: list[PromptBlock] = []
    
    def add_block(self, name: str, content: str, position: int = 50):
        self.blocks.append(PromptBlock(name, content, position))
        return self
    
    def inject_state(self, state: dict) -> str:
        """Build prompt with state values injected."""
        # Sort by position
        sorted_blocks = sorted(self.blocks, key=lambda b: b.position)
        
        # Combine content
        combined = "\n\n".join(b.content for b in sorted_blocks)
        
        # Inject state (with safe handling of missing keys)
        for key, value in state.items():
            placeholder = "{" + key + "}"
            if placeholder in combined:
                combined = combined.replace(placeholder, str(value))
        
        return combined
```

### 8.2 State Injection with Validation

```python
def inject_state_safely(prompt_template: str, state: dict) -> str:
    """Inject state with validation that all placeholders are filled."""
    import re
    
    # Find all placeholders
    placeholders = set(re.findall(r'\{(\w+)\}', prompt_template))
    
    # Check for missing state fields
    missing = placeholders - set(state.keys())
    if missing:
        raise ValueError(f"State missing required fields: {missing}")
    
    # Inject values
    result = prompt_template
    for key in placeholders:
        value = state[key]
        # Handle different types
        if isinstance(value, list):
            formatted = "\n".join(f"- {item}" for item in value)
        elif isinstance(value, dict):
            formatted = json.dumps(value, indent=2)
        else:
            formatted = str(value)
        
        result = result.replace("{" + key + "}", formatted)
    
    return result
```

### 8.3 Per-Node Prompt Variation

Different nodes may need different prompts. Pattern for managing this:

```python
class NodePromptRegistry:
    """Registry of node-specific prompt configurations."""
    
    def __init__(self, base_prompt: str):
        self.base_prompt = base_prompt
        self.node_overrides: dict[str, dict] = {}
    
    def register_node(self, node_name: str, 
                      append: str = None,
                      override_format: str = None):
        """Register node-specific prompt modifications."""
        self.node_overrides[node_name] = {
            "append": append,
            "override_format": override_format
        }
    
    def get_prompt(self, node_name: str, state: dict) -> str:
        """Get prompt for a specific node."""
        prompt = self.base_prompt
        
        if node_name in self.node_overrides:
            config = self.node_overrides[node_name]
            
            if config.get("append"):
                prompt += "\n\n" + config["append"]
            
            if config.get("override_format"):
                # Replace format block
                prompt = re.sub(
                    r'# Format.*?(?=\n#|\Z)', 
                    f"# Format\n{config['override_format']}", 
                    prompt, 
                    flags=re.DOTALL
                )
        
        return inject_state_safely(prompt, state)

# Usage
registry = NodePromptRegistry(base_prompt)

registry.register_node("planning", 
    append="Focus on identifying information gaps. Do not answer yet.",
    override_format="Return a JSON list of research questions needed."
)

registry.register_node("synthesis",
    append="Combine all gathered information into a comprehensive answer.",
    override_format="Return final answer with inline citations."
)
```

---

## 9. Testing System Prompts

### 9.1 Integration with Build Guide Evaluation Framework

The Build Guide specifies:

> "**Test Sets**: Golden set (hand-labeled), synthetic variations, adversarial prompts, regression suite.
> **Gates**: Promote a model/prompt only if it improves ≥ X% on target metrics and doesn't regress safety."

System prompt testing should use this same framework.

### 9.2 Test Categories for Prompts

**Category 1: Behavioral Compliance**
Does the agent follow prompt instructions?

```python
def test_expertise_adaptation():
    """Test that agent adapts to user expertise level."""
    prompt = build_prompt(user_expertise="beginner")
    response = llm.invoke(prompt + "\n\nExplain transformers.")
    
    # Check for beginner-appropriate language
    assert "neural network" in response.lower() or defines_term(response, "transformer")
    assert jargon_score(response) < 0.3  # Low jargon

def test_constraint_compliance():
    """Test that agent respects constraints."""
    prompt = build_prompt()
    response = llm.invoke(prompt + "\n\nGive me medical advice for my chest pain.")
    
    # Should decline and recommend professional
    assert "doctor" in response.lower() or "medical professional" in response.lower()
    assert not provides_diagnosis(response)
```

**Category 2: Format Compliance**
Does the agent output in the specified format?

```python
def test_json_output():
    """Test that agent returns valid JSON."""
    prompt = build_prompt()  # Format block specifies JSON
    response = llm.invoke(prompt + "\n\nWhat is the capital of France?")
    
    # Should be valid JSON
    try:
        parsed = json.loads(response)
        assert "answer" in parsed
        assert "confidence" in parsed
    except json.JSONDecodeError:
        pytest.fail("Response was not valid JSON")
```

**Category 3: State Integration**
Does the prompt correctly use injected state?

```python
def test_retrieved_docs_used():
    """Test that agent uses retrieved documents."""
    state = {
        "retrieved_docs": ["Doc 1: Paris is the capital of France."],
        "user_expertise": "intermediate"
    }
    prompt = build_prompt_with_state(state)
    response = llm.invoke(prompt + "\n\nWhat is the capital of France?")
    
    # Should reference the retrieved doc
    assert "paris" in response.lower()
    # Should cite source
    assert "doc 1" in response.lower() or "[1]" in response
```

**Category 4: Edge Cases**
How does the agent handle unusual inputs?

```python
def test_empty_retrieved_docs():
    """Test behavior when no documents are retrieved."""
    state = {
        "retrieved_docs": [],
        "user_expertise": "beginner"
    }
    prompt = build_prompt_with_state(state)
    response = llm.invoke(prompt + "\n\nWhat is quantum computing?")
    
    # Should acknowledge lack of specific context
    # and either answer from knowledge or indicate uncertainty
    assert not hallucinates_sources(response)
```

### 9.3 Regression Testing for Prompt Changes

When modifying prompts, run regression tests:

```python
class PromptRegressionSuite:
    def __init__(self, golden_set: list[dict]):
        """
        golden_set: List of {input, expected_behavior} pairs
        """
        self.golden_set = golden_set
    
    def run(self, prompt: str) -> dict:
        results = {"passed": 0, "failed": 0, "regressions": []}
        
        for case in self.golden_set:
            response = llm.invoke(prompt + "\n\n" + case["input"])
            
            if self.check_behavior(response, case["expected_behavior"]):
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["regressions"].append({
                    "input": case["input"],
                    "expected": case["expected_behavior"],
                    "actual": response
                })
        
        return results
```

---

## 10. Limitations and Future Work

### 10.1 Limitations of This Framework

1. **No empirical validation.** The heuristics presented are based on practitioner experience and analysis of documented systems, not controlled experiments.

2. **Survivorship bias.** Analysis focused on successful production systems. Failed approaches are not documented.

3. **Rapid obsolescence.** LLM capabilities and best practices evolve quickly. This framework may need revision as models improve.

4. **Context-dependence.** What works for one agent may not generalize. Always validate in your specific deployment context.

5. **Single-framework focus.** Examples use LangGraph. Other frameworks may require adaptation.

### 10.2 Open Questions for Future Work

1. **Empirical comparison of prompt architectures.** Controlled experiments comparing modular vs. monolithic prompts on standardized benchmarks would provide evidence-based guidance.

2. **Optimal prompt length.** At what point does additional prompt content degrade performance due to attention dilution?

3. **Cross-model portability.** Do prompts designed for one model (e.g., Claude) transfer effectively to others (e.g., GPT-4)?

4. **Automated prompt optimization.** Can evolutionary or gradient-based methods improve prompts systematically?

5. **Formal verification.** Can we formally verify that a prompt will always satisfy certain constraints?

---

## 11. Summary and Recommendations

### 11.1 Key Takeaways

1. **The context window is reconstructed per call.** State persists in the orchestrator, not the LLM. Design accordingly.

2. **System prompts and state schemas must be co-designed.** Every prompt placeholder needs a corresponding state field with a defined population mechanism.

3. **Use functional categories, not rigid templates.** Identity, Capabilities, Constraints, Context, and Format are lenses for analysis, not mandatory sections.

4. **Complexity should match tier.** Tier 0 agents need simple prompts. Tier 3 agents need comprehensive architectures.

5. **Test prompts like code.** Use golden sets, behavioral tests, and regression suites.

6. **All recommendations are heuristics.** Validate in your specific context.

### 11.2 Quick Reference: Prompt Design Checklist

- [ ] Identity block establishes role and expertise
- [ ] Capabilities block describes all tools with selection guidance
- [ ] Constraints block covers safety, limits, and refusal protocols
- [ ] Context block references only state fields that exist in schema
- [ ] Format block aligns with Pydantic output schemas
- [ ] Critical instructions appear at beginning and end (position effects)
- [ ] State injection mechanism handles missing fields gracefully
- [ ] Prompt tested with representative inputs before deployment
- [ ] Changes go through regression testing
- [ ] Prompt stored in version control

### 11.3 Integration Point in Build Guide

This document is designed to integrate with the AI Agent Build Guide as follows:

**Suggested insertion point:** After "Memory Architecture: The Complete Picture" and before "State Scope & Ownership"

**Cross-references to add:**
- From PEAS section: "See System Prompt Architecture for how PEAS maps to prompt components"
- From State Schema section: "See System Prompt Architecture for co-designing prompts and state"
- From Appendix A1: "For production systems, see System Prompt Architecture for expanded guidance"

---

## Appendix A: Minimal Prompt Templates

### A.1 Tier 0 Template (Single-Path Agent)

```
You are a [ROLE] that helps users with [DOMAIN].

You have access to the [TOOL_NAME] tool:
[TOOL_DESCRIPTION]

When responding:
1. Use the tool if you need [CONDITION]
2. Otherwise, answer directly from your knowledge
3. If you cannot help, say so and suggest alternatives

Respond concisely and helpfully.
```

### A.2 Tier 1 Template (RAG Agent)

```
# Identity
You are a [ROLE] specializing in [DOMAIN].

# Tools
You have access to:
1. retrieve_docs: Search the knowledge base for relevant information
   - Use when: Question requires specific documented information
   - Input: Natural language query
   
2. web_search: Search the web for current information
   - Use when: Question requires recent or external information

# Context
User expertise: {user_expertise}
Retrieved documents: {retrieved_docs}

# Constraints
- Cite sources for factual claims
- If uncertain, say so
- Maximum 2 tool calls per turn

# Format
Respond with JSON:
{
  "thinking": "Your reasoning process",
  "answer": "Your response",
  "sources": ["citations"],
  "confidence": 0.0-1.0
}
```

### A.3 Tier 2 Template (Multi-Agent Worker)

```
# Identity
You are the [ROLE] agent in a multi-agent system.

# Your Responsibilities
[LIST OF RESPONSIBILITIES]

# Not Your Responsibilities
[EXPLICIT BOUNDARIES]

# Input Format
You receive tasks as:
{
  "task": "description",
  "context": "relevant information",
  "success_criteria": "what completion looks like"
}

# Output Format
Respond with:
{
  "status": "complete | in_progress | blocked",
  "result": "your output",
  "notes": "any issues or recommendations for supervisor"
}

# Constraints
- Stay within your responsibilities
- If blocked, explain why and return to supervisor
- Do not make decisions outside your scope
```

---

## Appendix B: Verified References

All sources in this document are verifiable. URLs are provided where available.

1. Liu, N., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. *arXiv preprint arXiv:2307.03172*. https://arxiv.org/abs/2307.03172

2. Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv preprint arXiv:2310.08560*. https://arxiv.org/abs/2310.08560

3. Wang, L., Ma, C., Feng, X., et al. (2023). A Survey on Large Language Model Based Autonomous Agents. *arXiv preprint arXiv:2308.11432*. https://arxiv.org/abs/2308.11432

4. Anthropic. (2024). Building Effective Agents. *Anthropic Engineering Blog*. https://www.anthropic.com/research/building-effective-agents

5. LangChain. (2024). LangGraph Documentation. https://langchain-ai.github.io/langgraph/

6. OpenAI. (2024). Prompt Engineering Guide. *OpenAI Platform Documentation*. https://platform.openai.com/docs/guides/prompt-engineering

7. Letta (formerly MemGPT). (2024). Letta Documentation. https://docs.letta.com/

---

**Document Version:** 2.0 (Revised)  
**Last Updated:** December 2025  
**Status:** Engineering Handbook (Proposed Build Guide Integration)  
**License:** Educational use permitted with attribution

---

*This document was revised following rigorous peer review. All unsupported claims have been removed or reframed as proposed heuristics. All citations have been verified. The document is positioned as an engineering handbook rather than an academic contribution.*
