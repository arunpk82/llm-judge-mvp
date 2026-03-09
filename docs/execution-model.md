# Platform Execution Model

This document defines how the LLM-Judge roadmap is executed using GitHub.

---

# Development Hierarchy

The platform development hierarchy follows this structure.

Vision  
↓  
Roadmap Levels  
↓  
Capabilities  
↓  
EPICs  
↓  
Tasks  
↓  
Pull Requests

---

# Mapping to GitHub

| Layer | Implementation |
|------|---------------|
Vision | VISION.md |
Roadmap | docs/roadmap.md |
Capabilities | GitHub Milestones |
EPICs | GitHub Issues (epic label) |
Tasks | GitHub Issues |
Implementation | Pull Requests |

---

# GitHub Workflow

1. Capability defined in roadmap
2. Milestone created for capability
3. EPIC issue created
4. Task issues created under EPIC
5. Pull requests implement tasks
6. CI validates platform stability

---

# Example

Capability

Dataset Governance

EPIC

Dataset Registry

Tasks

• dataset manifest schema  
• dataset validator  
• dataset loader  

PR

Implements dataset schema.

---

# Engineering Principles

All changes must satisfy:

• deterministic evaluation  
• CI validation  
• artifact compatibility  
• rule governance

---

# Release Strategy

Platform evolution follows milestone completion.

Completion of all EPICs under a capability marks the capability as delivered.