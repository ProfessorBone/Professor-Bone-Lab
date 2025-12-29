# Professor Bone Lab

Professor Bone Lab is a public, first-principles AI systems laboratory focused on understanding **how intelligent systems actually work** — not just how to prompt them.

This lab explores:
- Explicit agent state
- Reasoning as a state transformation
- Evaluation-aware architectures
- Orchestration and multi-agent systems

The emphasis is clarity, structure, and architectural integrity.

---

## 🧠 Lab Philosophy

Modern AI systems fail not because models are weak, but because **state, reasoning, and evaluation are implicit or poorly defined**.

This lab treats those concepts as first-class architectural components.

No hype.  
No framework worship.  
Just understanding.

---

## � AI Agent Build Guide (v1.0)

This repository includes a comprehensive **AI Agent Build Guide** that documents a
production-oriented architecture for building reliable, scalable, and testable
LLM-based agent systems.

The Build guide treats **system prompts as first-class engineering artifacts**
and integrates them explicitly with:

- Agent state schemas
- Memory and context management
- Orchestration and routing logic
- Multi-agent (supervisor–worker) coordination
- Testing, validation, and regression discipline

Unlike ad-hoc prompt engineering approaches, the guide defines **enforceable
contracts and architectural boundaries** that allow agent systems to scale
without prompt drift, context bloat, or coordination failures.

---

### 🧱 Architectural Scope

The Build guide covers the full lifecycle of agent construction, including:

- Modular system prompt architecture
- Prompt–state contracts and placeholder enforcement
- Prompt testing and regression gates
- Context budgeting and anti-bloat guardrails
- Multi-agent prompt standards with explicit handoff contracts
- Loop guards, duplication prevention, and shared-state visibility rules

These patterns are framework-agnostic and apply to orchestrated LLM agents
built with tools such as LangGraph, LangChain, or custom orchestration layers.

---

### 🏷️ Stable Architecture Baseline

The current stable architecture is tagged as:

**`v1.0-agent-architecture-spine`**

This release represents the completed integration of:
- Phase 1: Modular system prompt architecture
- Phase 2: Prompt–state contracts and testing discipline
- Phase 3: Multi-agent prompt standards and coordination protocols

Future changes build on this baseline and do not alter the v1.0 architecture.

---

### 🚀 Getting Started

If you are new to the repository:

1. Start with the **AI Agent Build Guide** for the architectural overview.
2. Review Appendix A1 / A1b for system prompt templates.
3. Use the v1.0 tag as a stable reference point when building or extending agents.

The guide is written for practitioners designing real agent systems—not
toy demos—and emphasizes clarity, enforceability, and long-term maintainability.

---

## �📓 Sections

### 🔹 Agentic AI Notebooks
Hands-on, executable notebooks exploring concepts from first principles.

Path:
- Agent State
- Reasoning
- State ↔ Reasoning Interaction
- Multi-Agent Capstone

---

### 🔹 Agentic AI Notes
Written explanations, distilled insights, and architectural reasoning that support the notebooks.

---

### 🔹 Diagrams
Visual models of state, reasoning flows, and multi-agent coordination.

---

### 🔹 AI Agent Build Guide
See the dedicated section above for the full architectural overview and v1.0 release details.

**Direct link:** [`AI Agent Build Guide.md`](AI%20Agent%20Build%20Guide.md)

---

### 🔹 Books
Deep study and systems translation of foundational books on agentic AI. Each book is treated as a design substrate to be interrogated, stress-tested, and translated into modern agent architectures.

---

### 🔹 Essays/Papers
Original technical writing, architectural analysis, and research documentation exploring agent systems, reasoning patterns, and design principles.

---

### 🔹 References
Curated papers and source material grounding the work.

---

## 🎯 Intended Audience

- AI engineers
- Systems thinkers
- Researchers
- Practitioners building agentic systems
- Learners who care about *why* things work

---

## 🚧 Status

This lab is actively evolving.  
Artifacts are added as understanding deepens.
