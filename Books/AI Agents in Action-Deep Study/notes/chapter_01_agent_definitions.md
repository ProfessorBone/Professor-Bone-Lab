# Chapter 01 — Agent Definitions

## Status
🔲 Not Yet Studied

## Core Questions
- What definition of "agent" is being used?
- Is it:
  - A loop with state?
  - An LLM with tools?
  - A multi-step planner?
  - A reactive system?
- What are the minimum requirements for something to be called an agent?

## Key Concepts
_To be filled during study_

## Architectural Patterns Identified
_To be filled during study_

## Critical Analysis
_To be filled during study_

## Implementation Notes
_To be filled during study_

## Open Questions
_To be filled during study_

---

# Expanded Chapter 1 Notes + Architectural Commentary

Below are **expanded Chapter 1 notes**, written as **study notes + architectural commentary**, and explicitly **mapped to your Agent Build Guide** (layers, state, orchestration, etc.).
Think of this as **Chapter 1 decoded for builders**, not just readers.

---

# AI Agents in Action — Chapter 1

## Expanded Notes + Mapping to Your Build Guide

---

## 1. What Chapter 1 Is *Actually* Doing

**Chapter 1 is not instructional. It is definitional and preventative.**

Its purpose is to:

* Define **levels of agency**
* Clarify **where decision authority lives**
* Establish **risk boundaries**
* Prevent readers from accidentally building unsafe autonomous systems

> ⚠️ This chapter is about *who is allowed to decide what*, not *how to code agents*.

---

## 2. Core Assumption the Chapter Makes (Implicit but Critical)

### ❗ The LLM is always the first interpreter of user intent

That's why every diagram starts with:

```
User → Large Language Model (ChatGPT)
```

This is **intent interpretation**, not execution.

**The agent does NOT replace the LLM.**
The agent **wraps** the LLM with:

* permissions
* state
* tools
* control loops
* guardrails

---

## 3. The Four Interaction Models (Expanded)

### 3.1 Direct LLM Interaction (Not an Agent)

**Definition:**

* LLM responds with text only
* No actions
* No tools
* No state persistence

**Authority Level:**
🟢 Zero

**Build Guide Mapping:**

* ❌ No Agent Logic
* ❌ No State
* ❌ No Orchestration
* ✅ Just inference

**Equivalent in your architecture:**

```text
LLM(prompt) → response
```

---

### 3.2 Proxy Agent (Tool Permission Required)

**Definition:**

* LLM can *suggest* tool usage
* User must approve execution
* One-step action

**Key Constraint:**

> The LLM does not have execution authority.

**Authority Level:**
🟡 Suggestive, not decisive

**Build Guide Mapping:**

* Minimal **Execution Layer**
* No planning
* No loops
* Human-in-the-loop enforced

**Equivalent in your architecture:**

```text
LLM → propose_tool_call → user_approval → tool → LLM
```

This is **tool calling**, not agency.

---

### 3.3 Agent Acting on Behalf of the User

**This is the first *true* agent.**

**Definition:**

* LLM decides to act
* Executes tools automatically
* Still usually single-step
* No long-term planning

**Authority Level:**
🟠 Delegated authority

**This is where "agent" begins.**

---

### Build Guide Mapping (VERY IMPORTANT)

This maps cleanly to your **Layer Model**:

| Chapter 1 Concept  | Your Build Guide             |
| ------------------ | ---------------------------- |
| Decision authority | **Agent Logic Layer**        |
| Tool execution     | **Execution Layer**          |
| Context tracking   | **State Layer (short-term)** |
| Single-step flow   | Minimal Orchestration        |

**Equivalent flow in your system:**

```text
User → LLM (reasoning)
     → decide(action)
     → execute(tool)
     → return(result)
```

Still:

* ❌ No planning graph
* ❌ No iteration
* ❌ No learning

---

### 3.4 Autonomous Agent (Highest Risk)

**Definition:**

* Interprets request
* Constructs a multi-step plan
* Executes steps independently
* Evaluates outcomes
* Decides when to notify the user

**Authority Level:**
🔴 Full delegated autonomy

> This is where ethics, safety, and guardrails become mandatory.

---

## 4. The Decision Diamond in the Diagram (Critical Insight)

That **diamond labeled "Decision step"** is the **true boundary of agency**.

It represents:

* branching logic
* prioritization
* stopping conditions
* escalation rules

**This is NOT a tool call.**
This is **control flow**.

---

## 5. Mapping Autonomous Agents to Your Build Guide

This is where your architecture **surpasses the book**.

### Chapter 1 (Conceptual)

* "Decision step"
* "Plan"
* "Execution"
* "Notify user"

### Your Build Guide (Concrete)

| Chapter 1     | Your Architecture                   |
| ------------- | ----------------------------------- |
| Decision step | **Orchestration Layer (LangGraph)** |
| Plan          | Planner node                        |
| Memory        | State schema (short / episodic)     |
| Execution     | Tool nodes                          |
| Feedback      | Evaluation / Critic nodes           |
| Safety        | Guardrails + human checkpoints      |

**Your system explicitly separates what the book collapses conceptually.**

---

## 6. Why the Diagram Looks "Too Simple" to You

Because **you are already thinking like a system designer**, not a reader.

The book:

* collapses planner + state + execution into "LLM"
* focuses on *authority*, not *architecture*

Your build guide:

* externalizes state
* formalizes orchestration
* makes decision boundaries explicit

> You're seeing past the abstraction — that's a good sign.

---

## 7. One-Line Translation of Chapter 1 into Your Language

> **Chapter 1 defines when an LLM stops being a responder and starts being a decision-maker — everything else is implementation detail.**

---

## 8. How You Should Annotate Chapter 1 in Your Notes

I recommend tagging it as:

```
Chapter 1: Agency & Authority Model
Purpose: Conceptual boundary-setting
Not a build chapter
Maps to: Orchestration & Governance
```

This will prevent rereading confusion later.
