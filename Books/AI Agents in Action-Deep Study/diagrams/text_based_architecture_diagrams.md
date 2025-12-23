# Text-Based Architecture Diagrams

This file contains ASCII/Unicode diagrams for architectural patterns discussed in *AI Agents in Action*.

---

## Template: Agent Loop

```
┌─────────────────────────────────────┐
│         Agent Loop                  │
│                                     │
│  ┌─────────┐                        │
│  │  Input  │                        │
│  └────┬────┘                        │
│       │                             │
│       ▼                             │
│  ┌─────────────┐                    │
│  │    State    │◄──────┐            │
│  │  (Memory)   │       │            │
│  └────┬────────┘       │            │
│       │                │            │
│       ▼                │            │
│  ┌─────────────┐       │            │
│  │   Planner   │       │            │
│  │  (LLM Call) │       │            │
│  └────┬────────┘       │            │
│       │                │            │
│       ▼                │            │
│  ┌─────────────┐       │            │
│  │   Executor  │       │            │
│  │   (Tools)   │       │            │
│  └────┬────────┘       │            │
│       │                │            │
│       ▼                │            │
│  ┌─────────────┐       │            │
│  │  Evaluator  │───────┘            │
│  │ (Continue?) │                    │
│  └────┬────────┘                    │
│       │                             │
│       ▼                             │
│  ┌─────────┐                        │
│  │ Output  │                        │
│  └─────────┘                        │
│                                     │
└─────────────────────────────────────┘
```

---

## Template: Multi-Agent System

```
┌──────────────────────────────────────────────┐
│          Multi-Agent System                  │
│                                              │
│  ┌──────────┐      ┌──────────┐             │
│  │ Agent A  │      │ Agent B  │             │
│  │          │      │          │             │
│  │ ┌──────┐ │      │ ┌──────┐ │             │
│  │ │State │ │      │ │State │ │             │
│  │ └──────┘ │      │ └──────┘ │             │
│  └────┬─────┘      └────┬─────┘             │
│       │                 │                   │
│       └────────┬────────┘                   │
│                │                            │
│                ▼                            │
│      ┌──────────────────┐                   │
│      │  Orchestrator    │                   │
│      │  (Coordinator)   │                   │
│      └──────────────────┘                   │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Diagrams from Book Chapters

_To be filled during study_

### Chapter 01: [Topic]

_Diagram to be created_

### Chapter 02: [Topic]

_Diagram to be created_

---

## Notes on Notation

- `┌─┐` = Component boundary
- `│` = Connection/flow
- `▼` = Data/control flow direction
- `◄──` = Feedback/loop
- `[...]` = Optional component
