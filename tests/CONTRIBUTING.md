# Contributing to LLM-Judge

This document defines the execution process for the LLM-Judge platform. Every team member
must read and follow this before creating any issue, branch, or pull request.

---

## Hierarchy Overview

```
VISION.md (repo root)
    └── L1–L6 Roadmap Levels  (GitHub Project — "LLM-Judge Roadmap")
            └── Capabilities  (GitHub Milestones — CAP-N)
                    └── EPICs (GitHub Issues — label: epic)
                            └── Tasks (GitHub Issues — label: task)
                                    └── Pull Requests (linked via Closes #N)
```

---

## Level Definitions

Every GitHub Project item must be assigned one of the following levels. No ambiguity.

| Level | Name | Time Horizon | Owner | What it is |
|---|---|---|---|---|
| **L1** | Company Bet | 2–3 years | Founder / CTO | A strategic direction the entire product bets on. E.g., "Become the CI/CD gate for AI." |
| **L2** | Annual Theme | 12 months | Head of Engineering | A 12-month focus that advances an L1 bet. E.g., "Achieve L3 platform maturity." |
| **L3** | Quarterly OKR | 3 months | Team Lead | A quarterly objective with measurable key results. Must link to an L2 theme. |
| **L4** | Capability | 6–12 weeks | Senior Engineer | A stable, durable system component. Modeled as a **GitHub Milestone**. |
| **L5** | EPIC | 2–6 weeks | Engineer | A feature or body of work. GitHub Issue with label `epic`, assigned to a Milestone. |
| **L6** | Task | 1–5 days | Any contributor | Atomic unit of work. GitHub Issue with label `task`, tracked by an EPIC. |

### Rules

- **No EPIC without a Milestone.** Every `epic` issue must be assigned to a CAP-N Milestone.
- **No Task without an EPIC.** Every `task` issue must include the EPIC reference in its header.
- **No PR without a closing keyword.** Every PR description must contain `Closes #N` or `Fixes #N`.
- **No branch without a task.** Branches must follow `type/ISSUE_NUMBER-short-description`.
- **Tasks are max 5 working days.** If larger, break it down.

---

## Capabilities (Milestones)

Capabilities are long-lived system components. They are **GitHub Milestones**, not issues.

| Milestone | Capability | Status |
|---|---|---|
| CAP-1 | Dataset Governance | 🔄 In Progress |
| CAP-2 | Baseline Governance | 🔄 In Progress |
| CAP-3 | Rule Governance | ✅ Partially Complete |
| CAP-4 | Artifact Governance | 📋 Planned |
| CAP-5 | Drift Monitoring | 📋 Planned |

Each Milestone's progress bar = EPICs closed / EPICs total.

---

## Branch Naming

All branches must match this pattern: `^(feature|fix|chore|docs|test)/\d+-.+$`

Examples:
- `feature/45-dataset-registry-schema`
- `fix/78-sampling-stability-edge-case`
- `chore/92-update-trivy-deps`
- `docs/103-update-rubric-readme`
- `test/114-add-baseline-diff-coverage`

Branches that do not match this pattern will be **blocked from merge** by CI.

---

## Label Taxonomy

### Type labels
| Label | Hex | Usage |
|---|---|---|
| `epic` | `#8B5CF6` | Parent issues grouping tasks |
| `task` | `#3B82F6` | Atomic units of work |
| `bug` | `#EF4444` | Defects |
| `chore` | `#6B7280` | Maintenance, deps, refactor |

### Capability labels
| Label | Usage |
|---|---|
| `cap:dataset-governance` | CAP-1 work |
| `cap:baseline-governance` | CAP-2 work |
| `cap:rule-governance` | CAP-3 work |
| `cap:artifact-governance` | CAP-4 work |
| `cap:drift-monitoring` | CAP-5 work |

### Priority labels
| Label | Usage |
|---|---|
| `priority:critical` | Blocks maturity level exit criteria |
| `priority:high` | Required for current sprint |
| `priority:medium` | Required for current quarter |
| `priority:low` | Nice to have |

### Status labels
| Label | Usage |
|---|---|
| `status:blocked` | Waiting on external dependency |
| `status:needs-review` | Awaiting team input |

---

## Definition of Done

### Task DoD
- [ ] All acceptance criteria in the issue are met
- [ ] Unit tests cover the new code path
- [ ] No regressions in the full test suite
- [ ] `ruff` and `mypy` pass with no errors
- [ ] PR description contains `Closes #N`
- [ ] At least one reviewer approval

### EPIC DoD
- [ ] All child task issues are closed
- [ ] Overall test coverage ≥ 80%
- [ ] Critical evaluation paths ≥ 95% covered
- [ ] Documentation updated (README, docstrings, or wiki)
- [ ] No HIGH or CRITICAL CVEs in Docker image (Trivy scan)
- [ ] Deterministic regression test suite passes
- [ ] Stakeholder sign-off recorded in EPIC issue comment

---

## Implementation Order (for new contributors)

1. Read `VISION.md` — understand the mission and current maturity level
2. Read this document — understand the hierarchy and rules
3. Find or create a `task` issue — do not start work without one
4. Create your branch: `feature/ISSUE_NUMBER-description`
5. Make your changes, write tests, run `make lint test`
6. Open a PR with `Closes #ISSUE_NUMBER` in the description
7. Request review — CI must be green before review is requested

---

## Escalation

- **Blocked on a task?** Add label `status:blocked`, comment with the blocker, tag the EPIC owner.
- **Scope growing?** Stop. Comment on the task, tag the EPIC owner. Do not expand scope silently.
- **Finding a bug during feature work?** Open a separate `bug` issue. Do not fix it in the feature branch unless it blocks the feature.
