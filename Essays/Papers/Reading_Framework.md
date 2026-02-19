# Research Paper Reading Framework (Architect Mode)

> Goal: Convert each paper into (1) system understanding, (2) critique, (3) Build Guide improvements.
> Output required per paper: 1 outline entry + 1 critique paragraph + 1 proposed Guide delta.

---

## 0) Metadata
- Title:
- Authors:
- Year / Venue:
- Domain tags:
- Why I chose it (1 sentence):

---

## 1) Thesis (1 Sentence)
**Claim:**  
- "The paper claims X improves Y by doing Z."

---

## 2) System Outline (Components + Flow)
- Inputs:
- Core components:
- Outputs:
- Feedback signals:
- Loop steps (if any):

---

## 3) Invariants (3)
What must be true for the method to work?

- I1:
- I2:
- I3:

---

## 4) Contracts & Boundaries
List boundary surfaces and expected data shapes.

- Boundary A:
  - Contract:
  - Owner:
  - Versioning needs:

- Boundary B:
  - Contract:
  - Owner:
  - Versioning needs:

---

## 5) Runtime Validation (3)
What checks should fire at runtime?

- V1:
- V2:
- V3:

---

## 6) Observability & Telemetry
What must be measured to detect failure early?

- Key metrics:
- Traces / spans:
- Logs / events:
- Dashboards / alerts:

---

## 7) Evaluation
- What they measured:
- What's missing:
- What I would add (eval harness items):

---

## 8) Failure Modes & Drift
- Failure mode 1 (propagation/amplification risk):
- Failure mode 2 (semantic drift risk):
- Failure mode 3 (governance / safety risk):

---

## 9) Production Translation
If I had to ship this:

- What I'd keep:
- What I'd change:
- What I'd add (contracts/validation/governance):

---

## 10) Build Guide Delta (Mandatory)
- Section(s) impacted:
- Proposed paragraph insert:
- Any new diagram needed:
