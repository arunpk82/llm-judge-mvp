#!/usr/bin/env python3
"""
manage.py — Single entry point for all LLM-Judge GitHub operations.

Single source of truth: epics.yaml

COMMANDS
  setup    Full first-time setup: labels → milestones → vision → project → EPICs
  retag    Sync all project custom fields from epics.yaml (safe to re-run anytime)
  recover  Reopen closed EPICs that have no milestone, fix them, re-close done ones
  status   Print current state of EPICs and project — no changes made

USAGE
  poetry run python tools/manage.py setup
  poetry run python tools/manage.py setup --dry-run
  poetry run python tools/manage.py retag
  poetry run python tools/manage.py retag --dry-run
  poetry run python tools/manage.py recover
  poetry run python tools/manage.py recover --dry-run
  poetry run python tools/manage.py status

TOKEN REQUIREMENTS
  Classic PAT with scopes: repo (full) + project
  Create at: https://github.com/settings/tokens → Tokens (classic)
  export GITHUB_TOKEN=your_token_here
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import requests  # type: ignore[import-untyped]
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

YAML_PATH = Path(__file__).parent.parent / "epics_v2.yaml"
API_BASE  = "https://api.github.com"
GRAPHQL   = "https://api.github.com/graphql"
DELAY     = 0.5

REQUIRED_LABELS = {
    "epic":                    "8B5CF6",
    "task":                    "3B82F6",
    "bug":                     "EF4444",
    "chore":                   "6B7280",
    "cap:dataset-governance":  "1E40AF",
    "cap:baseline-governance": "5B21B6",
    "cap:rule-governance":     "14532D",
    "cap:artifact-governance": "78350F",
    "cap:drift-monitoring":    "831843",
    "priority:critical":       "DC2626",
    "priority:high":           "EA580C",
    "priority:medium":         "CA8A04",
    "priority:low":            "6B7280",
    "status:in-progress":      "16A34A",
    "status:in-review":        "0284C7",
    "status:blocked":          "DC2626",
}

class _FieldDef(TypedDict):
    name: str
    options: list[str]

CUSTOM_FIELDS: list[_FieldDef] = [
    {"name": "Roadmap Level",  "options": ["L1","L2","L3","L4","L5","L6"]},
    {"name": "Capability",     "options": ["CAP-1","CAP-2","CAP-3","CAP-4","CAP-5"]},
    {"name": "Status",         "options": ["Planned","In Progress","In Review","Done","Blocked"]},
    {"name": "Target Quarter", "options": ["2026-Q1","2026-Q2","2026-Q3","2026-Q4",
                                            "2027-Q1","2027-Q2","2027-Q3","2027-Q4","2028-Q4"]},
    {"name": "Mandatory",      "options": ["Yes","No"]},
]

STATUS_MAP = {
    "done":        "Done",
    "in_progress": "In Progress",
    "planned":     "Planned",
    "blocked":     "Blocked",
}

CAP_QUARTER_MAP = {
    "CAP-1": "2026-Q2",
    "CAP-2": "2026-Q2",
    "CAP-3": "2026-Q2",
    "CAP-4": "2026-Q3",
    "CAP-5": "2026-Q3",
}

VISION_TITLE = "[VISION] LLM-Judge Platform — North Star & Roadmap"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("ERROR: GITHUB_TOKEN is not set.")
        print("  export GITHUB_TOKEN=your_token_here")
        print("  Scopes required: repo (full), project")
        print("  Create at: https://github.com/settings/tokens → Tokens (classic)")
        sys.exit(1)
    return token


def rh(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
    }


def _with_retry(fn, retries: int = 4, base_delay: float = 2.0):
    """Retry fn() on network errors and 5xx/429. Backoff: 2s, 4s, 8s, 16s."""
    for attempt in range(retries):
        try:
            return fn()
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"    ⚠️  Network error ({e.__class__.__name__}), retrying in {wait:.0f}s "
                  f"[{attempt + 1}/{retries}]...")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"    ⚠️  HTTP {status}, retrying in {wait:.0f}s [{attempt + 1}/{retries}]...")
                time.sleep(wait)
            else:
                raise


def _rest(method: str, url: str, token: str, **kwargs) -> requests.Response:
    """REST call with timeout and retry."""
    def fn():
        resp = requests.request(method, url, headers=rh(token), timeout=30, **kwargs)
        resp.raise_for_status()
        return resp
    return _with_retry(fn)


def gql(token: str, query: str, variables: dict | None = None) -> dict:
    """GraphQL call with timeout and retry."""
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables

    def fn():
        resp = requests.post(
            GRAPHQL,
            headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        if "errors" in result:
            raise RuntimeError(result["errors"][0]["message"])
        return result.get("data", {})

    return _with_retry(fn)


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

def get_all_issues(repo: str, token: str, label: str | None = None) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    page = 1
    params: dict[str, Any] = {"state": "all", "per_page": 100}
    if label:
        params["labels"] = label
    while True:
        resp = _rest("GET", f"{API_BASE}/repos/{repo}/issues", token, params={**params, "page": page})
        data: list[dict[str, Any]] = resp.json()
        if not data:
            break
        issues.extend(item for item in data if "pull_request" not in item)
        page += 1
    return issues


def get_milestones(repo: str, token: str) -> dict[str, int]:
    """Returns {title: number}"""
    resp = _rest("GET", f"{API_BASE}/repos/{repo}/milestones", token, params={"state": "open", "per_page": 100})
    return {m["title"]: m["number"] for m in resp.json()}


def close_issue_with_comment(repo: str, number: int, comment: str, token: str) -> None:
    _rest("POST", f"{API_BASE}/repos/{repo}/issues/{number}/comments", token, json={"body": comment})
    time.sleep(DELAY)
    _rest("PATCH", f"{API_BASE}/repos/{repo}/issues/{number}", token, json={"state": "closed"})


# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------

def get_viewer(token: str) -> tuple[str, str]:
    data = gql(token, "{ viewer { id login } }")
    return data["viewer"]["id"], data["viewer"]["login"]


def get_issue_node_id(repo: str, number: int, token: str) -> str | None:
    owner, name = repo.split("/")
    try:
        data = gql(token, """
            query($owner: String!, $name: String!, $number: Int!) {
              repository(owner: $owner, name: $name) {
                issue(number: $number) { id }
              }
            }
        """, {"owner": owner, "name": name, "number": number})
        return data.get("repository", {}).get("issue", {}).get("id")
    except Exception:
        return None


def pin_issue(repo: str, issue_number: int, token: str) -> None:
    owner, name = repo.split("/")
    data = gql(token, """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            issue(number: $number) { id }
          }
        }
    """, {"owner": owner, "name": name, "number": issue_number})
    issue_id = data.get("repository", {}).get("issue", {}).get("id")
    if not issue_id:
        return
    try:
        gql(token, """
            mutation($issueId: ID!) {
              pinIssue(input: { issueId: $issueId }) { issue { number } }
            }
        """, {"issueId": issue_id})
        print("  ✅ Vision issue pinned")
    except RuntimeError as e:
        print(f"  ⚠️  Could not pin (may need admin rights): {e}")


# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------

def find_project(login: str, project_name: str, token: str) -> str | None:
    data = gql(token, """
        query($login: String!) {
          user(login: $login) {
            projectsV2(first: 20) { nodes { id title } }
          }
        }
    """, {"login": login})
    for node in data.get("user", {}).get("projectsV2", {}).get("nodes", []):
        if node["title"] == project_name:
            return node["id"]
    return None


def create_project(owner_id: str, project_name: str, token: str) -> str:
    data = gql(token, """
        mutation($ownerId: ID!, $title: String!) {
          createProjectV2(input: { ownerId: $ownerId, title: $title }) {
            projectV2 { id title url }
          }
        }
    """, {"ownerId": owner_id, "title": project_name})
    project = data["createProjectV2"]["projectV2"]
    print(f"  ✅ Created project: {project['title']} — {project['url']}")
    return project["id"]


def get_project_fields(project_id: str, token: str) -> dict[str, dict]:
    """Returns {field_name: {id, options: {name: id}}}"""
    data = gql(token, """
        query($projectId: ID!) {
          node(id: $projectId) {
            ... on ProjectV2 {
              fields(first: 30) {
                nodes {
                  ... on ProjectV2SingleSelectField { id name options { id name } }
                  ... on ProjectV2Field { id name }
                }
              }
            }
          }
        }
    """, {"projectId": project_id})
    result = {}
    for node in data.get("node", {}).get("fields", {}).get("nodes", []):
        name = node.get("name", "")
        if name:
            result[name] = {
                "id":      node["id"],
                "options": {o["name"]: o["id"] for o in node.get("options", [])},
            }
    return result


def ensure_project_fields(project_id: str, token: str) -> dict[str, dict]:
    """Ensure all CUSTOM_FIELDS exist. Returns full field map."""
    existing = get_project_fields(project_id, token)
    result   = dict(existing)
    for field_def in CUSTOM_FIELDS:
        name = field_def["name"]
        if name in existing:
            print(f"    ✅ Field '{name}' already exists")
        else:
            try:
                data = gql(token, """
                    mutation($projectId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]!) {
                      createProjectV2Field(input: {
                        projectId: $projectId, dataType: SINGLE_SELECT,
                        name: $name, singleSelectOptions: $options
                      }) {
                        projectV2Field {
                          ... on ProjectV2SingleSelectField { id name options { id name } }
                        }
                      }
                    }
                """, {
                    "projectId": project_id,
                    "name":      name,
                    "options":   [{"name": o, "color": "GRAY", "description": ""}
                                  for o in field_def["options"]],
                })
                field = data["createProjectV2Field"]["projectV2Field"]
                result[name] = {
                    "id":      field["id"],
                    "options": {o["name"]: o["id"] for o in field.get("options", [])},
                }
                print(f"    ✅ Created field '{name}' ({len(field_def['options'])} options)")
                time.sleep(DELAY)
            except RuntimeError as e:
                print(f"    ⚠️  Could not create field '{name}': {e}")
    return result


def add_issue_to_project(project_id: str, node_id: str, token: str) -> str | None:
    try:
        data = gql(token, """
            mutation($projectId: ID!, $contentId: ID!) {
              addProjectV2ItemById(input: { projectId: $projectId, contentId: $contentId }) {
                item { id }
              }
            }
        """, {"projectId": project_id, "contentId": node_id})
        return data["addProjectV2ItemById"]["item"]["id"]
    except RuntimeError as e:
        print(f"    ⚠️  Could not add to project: {e}")
        return None


def set_field_value(project_id: str, item_id: str, field_id: str,
                    option_id: str, token: str) -> bool:
    try:
        gql(token, """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
              updateProjectV2ItemFieldValue(input: {
                projectId: $projectId, itemId: $itemId, fieldId: $fieldId,
                value: { singleSelectOptionId: $optionId }
              }) { projectV2Item { id } }
            }
        """, {
            "projectId": project_id,
            "itemId":    item_id,
            "fieldId":   field_id,
            "optionId":  option_id,
        })
        return True
    except RuntimeError:
        return False


def tag_item_fields(project_id: str, item_id: str, field_map: dict,
                    values: dict, token: str) -> None:
    """Set multiple field values on a project item. values = {field_name: option_name}"""
    for field_name, option_name in values.items():
        if not option_name:
            continue
        field = field_map.get(field_name)
        if not field:
            continue
        option_id = field["options"].get(option_name)
        if option_id:
            set_field_value(project_id, item_id, field["id"], option_id, token)
            time.sleep(0.2)


def get_project_item_map(project_id: str, token: str) -> dict[str, str]:
    """Returns {issue_node_id: project_item_id}"""
    try:
        data = gql(token, """
            query($projectId: ID!) {
              node(id: $projectId) {
                ... on ProjectV2 {
                  items(first: 100) {
                    nodes { id content { ... on Issue { id } } }
                  }
                }
              }
            }
        """, {"projectId": project_id})
        result = {}
        for node in data.get("node", {}).get("items", {}).get("nodes", []):
            content  = node.get("content", {}) or {}
            issue_id = content.get("id")
            if issue_id:
                result[issue_id] = node["id"]
        return result
    except Exception:
        return {}


def get_project_item_titles(project_id: str, token: str) -> set[str]:
    try:
        data = gql(token, """
            query($projectId: ID!) {
              node(id: $projectId) {
                ... on ProjectV2 {
                  items(first: 100) {
                    nodes { content {
                      ... on DraftIssue { title }
                      ... on Issue      { title }
                    }}
                  }
                }
              }
            }
        """, {"projectId": project_id})
        nodes = data.get("node", {}).get("items", {}).get("nodes", [])
        return {n["content"]["title"] for n in nodes
                if n.get("content") and "title" in n["content"]}
    except Exception:
        return set()


def add_draft_item(project_id: str, title: str, body: str, token: str) -> str | None:
    try:
        data = gql(token, """
            mutation($projectId: ID!, $title: String!, $body: String!) {
              addProjectV2DraftIssue(input: {
                projectId: $projectId, title: $title, body: $body
              }) { projectItem { id } }
            }
        """, {"projectId": project_id, "title": title, "body": body})
        return data["addProjectV2DraftIssue"]["projectItem"]["id"]
    except RuntimeError as e:
        print(f"    ⚠️  Could not add draft item '{title}': {e}")
        return None


def epic_field_values(epic: dict) -> dict:
    """Derive the full set of project field tag values for an EPIC."""
    cap = epic.get("milestone", "")
    return {
        "Roadmap Level":  epic.get("level", "L3"),
        "Capability":     cap,
        "Status":         STATUS_MAP.get(epic.get("status", "planned"), "Planned"),
        "Target Quarter": CAP_QUARTER_MAP.get(cap, ""),
        "Mandatory":      "Yes" if epic.get("mandatory") else "No",
    }


# ---------------------------------------------------------------------------
# Content builders
# ---------------------------------------------------------------------------

def build_epic_body(epic: dict) -> str:
    def fmt_list(items: list) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "- N/A"

    def fmt_ac(items: list) -> str:
        return "\n".join(f"- [ ] {i}" for i in items) if items else "- [ ] N/A"

    def fmt_tasks(items: list) -> str:
        if not items:
            return "- [ ] No tasks defined yet"
        return "\n".join(
            f"- [x] {t}" if ("DONE" in t.upper() or "✅" in t) else f"- [ ] {t}"
            for t in items
        )

    dod = """- [ ] All task issues closed
- [ ] Unit test coverage >= 80%
- [ ] Critical evaluation paths >= 95% covered
- [ ] Documentation updated
- [ ] No HIGH/CRITICAL CVEs in Docker image
- [ ] Deterministic regression suite passes
- [ ] Stakeholder sign-off recorded"""

    cap       = epic.get("milestone", "")
    level     = epic.get("level", "L3")
    mandatory = epic.get("mandatory", False)
    gate_tag  = "🔴 Mandatory (gates level exit)" if mandatory else "🟡 Optional (can defer)"

    return f"""## Context
- **Capability:** {cap} | **Roadmap Level:** {level}
- **Level Gate:** {gate_tag}
- **Vision Objective:** {epic.get('vision_objective', 'N/A')}
- **Effort:** {epic.get('effort', 'N/A')}

## Purpose
{epic.get('purpose', '').strip()}

## Scope

**In scope:**
{fmt_list(epic.get('scope_in', []))}

**Out of scope:**
{fmt_list(epic.get('scope_out', []))}

## Acceptance Criteria
{fmt_ac(epic.get('acceptance_criteria', []))}

## Definition of Done
{dod}

## Task List
{fmt_tasks(epic.get('tasks', []))}

---
_See [Vision issue](../../issues?q=is%3Aissue+VISION) for full roadmap hierarchy._
""".strip()


def build_vision_body(config: dict) -> str:
    roadmap_items = config.get("project", {}).get("roadmap_items", [])
    milestones    = config.get("milestones", {})
    epics         = config.get("epics", [])

    # Index: CAP → EPIC summary lines
    ms_epics: dict[str, list[str]] = defaultdict(list)
    for epic in epics:
        cap    = epic.get("milestone", "")
        status = epic.get("status", "planned")
        icon   = "✅" if status == "done" else ("🔄" if status == "in_progress" else "📋")
        mand   = " 🔴" if epic.get("mandatory") else ""
        ms_epics[cap].append(f"{icon}{mand} {epic['id']}: {epic['title'].replace('[EPIC] ', '')}")

    # Index: level → mandatory_caps
    level_caps: dict[str, list[str]] = {}
    for item in roadmap_items:
        level_caps[item["level"]] = item.get("exit_criteria", {}).get("mandatory_caps", [])

    # Build roadmap section
    horizon_badge = {"active": "🟢 Active", "shaped": "🔵 Shaped", "vision": "⚪ Vision", "done": "✅ Done"}
    roadmap_section = ""
    for item in roadmap_items:
        level   = item["level"]
        horizon = item.get("planning_horizon", "vision")
        badge   = horizon_badge.get(horizon, "⚪")
        caps    = level_caps.get(level, [])
        roadmap_section += f"\n### {badge} {level} — {item['name']}\n"
        roadmap_section += f"**Target:** {item.get('target_quarter', 'TBD')} | **Horizon:** {horizon}\n\n"

        kpis = item.get("exit_criteria", {}).get("kpis", [])
        if kpis:
            roadmap_section += "**Exit KPIs:**\n"
            for kpi in kpis:
                roadmap_section += f"- {kpi}\n"
            roadmap_section += "\n"

        if caps:
            roadmap_section += "**Capabilities:**\n"
            for cap in caps:
                ms    = milestones.get(cap, {})
                title = ms.get("title", cap)
                due   = ms.get("due_date", "TBD")
                total = len(ms_epics.get(cap, []))
                done  = sum(1 for e in ms_epics.get(cap, []) if e.startswith("✅"))
                roadmap_section += f"- **{title}** (due {due}) — {done}/{total} EPICs done\n"
        else:
            roadmap_section += "_Capabilities to be defined when this level becomes active._\n"

    # Build milestone section
    milestone_section = ""
    for cap_key, ms in milestones.items():
        title      = ms.get("title", cap_key)
        due        = ms.get("due_date", "TBD")
        epic_list  = ms_epics.get(cap_key, [])
        done_count = sum(1 for e in epic_list if e.startswith("✅"))
        total      = len(epic_list)
        pct        = int(done_count / total * 100) if total else 0

        milestone_section += f"\n### {title}\n"
        milestone_section += f"**Due:** {due} | **Progress:** {done_count}/{total} EPICs ({pct}%) | [Milestone →](../../milestone)\n\n"
        for epic_line in epic_list:
            milestone_section += f"  {epic_line}\n"

    # L3 progress summary
    l3_epics     = [e for e in epics if e.get("level") == "L3"]
    l3_mandatory = [e for e in l3_epics if e.get("mandatory")]
    l3_done_m    = sum(1 for e in l3_mandatory if e.get("status") == "done")
    l3_total_m   = len(l3_mandatory)
    l3_pct       = int(l3_done_m / l3_total_m * 100) if l3_total_m else 0

    return f"""## Vision

> Build the evaluation and governance operating system for AI systems —
> enabling teams to ship AI with the same confidence that traditional
> CI/CD delivers for conventional code.

**3-Year Targets (2028):**
- ≥ 500 production systems integrated
- Cohen's Kappa ≥ 0.80 on calibration sets
- < 24hr drift detection SLA
- 10,000+ SDK users

---

## L3 Progress (Current Active Level)

| Metric | Value |
|---|---|
| Mandatory EPICs closed | {l3_done_m} / {l3_total_m} ({l3_pct}%) |
| Total EPICs (incl. optional) | {len(l3_epics)} |
| L4 planning trigger | When mandatory EPICs ≥ 80% closed |

---

## How to navigate this repo

| Where | What you'll find |
|---|---|
| This issue | Full hierarchy: Vision → Roadmap → Milestones → EPICs |
| [Milestones](../../milestones) | Per-capability progress |
| [Project board](../../projects) | All EPICs filterable by Level / Capability / Mandatory |
| [Issues](../../issues?q=label%3Aepic) | All EPICs |
| `VISION.md` | Full written vision and maturity model |
| `CONTRIBUTING.md` | Branch naming, Definition of Done |

---

## Roadmap (L1–L6 Rolling Horizon)
{roadmap_section}
---

## Capabilities (Milestones)
{milestone_section}
---

## Legend

| Icon | Meaning |
|---|---|
| ✅ | Done |
| 🔄 | In Progress |
| 📋 | Planned |
| 🔴 | Mandatory — gates level exit |
| 🟢 | Active level (fully planned) |
| 🔵 | Shaped (KPIs known, EPICs TBD) |
| ⚪ | Vision only |

---
_Auto-generated by `tools/manage.py` from `epics.yaml`.
Update `epics.yaml` and re-run `manage.py setup` or `manage.py retag`._
""".strip()


# ---------------------------------------------------------------------------
# COMMAND: setup
# ---------------------------------------------------------------------------

def cmd_setup(config: dict, token: str, dry_run: bool) -> None:
    """Full idempotent setup: labels → milestones → vision → project → EPICs."""
    repo  = config["repo"]
    epics = config["epics"]
    owner = repo.split("/")[0]

    print(f"\n  Repo:  {repo}")
    print(f"  EPICs: {len(epics)}")
    if dry_run:
        print("  MODE:  DRY RUN\n")

    # ------------------------------------------------------------------
    # Step 1 — Labels
    # ------------------------------------------------------------------
    print("\n[1/6] Labels")
    if dry_run:
        print(f"  (dry-run) Would ensure {len(REQUIRED_LABELS)} labels")
    else:
        url      = f"{API_BASE}/repos/{repo}/labels"
        resp     = _rest("GET", url, token, params={"per_page": 100})
        existing_label_names = {lbl["name"] for lbl in resp.json()}
        created  = []
        for name, color in REQUIRED_LABELS.items():
            if name not in existing_label_names:
                r = requests.post(url, headers=rh(token),
                                  json={"name": name, "color": color}, timeout=30)
                if r.status_code == 201:
                    created.append(name)
                elif r.status_code == 422:
                    pass  # already exists — GitHub returns 422 for duplicate labels
                else:
                    r.raise_for_status()
                time.sleep(DELAY)
        if created:
            print(f"  ✅ Created {len(created)} labels: {', '.join(created)}")
        else:
            print(f"  ✅ All {len(REQUIRED_LABELS)} labels already exist")

    # ------------------------------------------------------------------
    # Step 2 — Milestones
    # ------------------------------------------------------------------
    print("\n[2/6] Milestones")
    milestone_map: dict[str, int] = {}
    if dry_run:
        for cap_key, ms in config["milestones"].items():
            print(f"  (dry-run) {cap_key}: {ms['title']} (due {ms.get('due_date','TBD')})")
        milestone_map = {k: 0 for k in config["milestones"]}
    else:
        existing_ms = get_milestones(repo, token)
        created_ms  = []
        for cap_key, ms in config["milestones"].items():
            title = ms["title"]
            if title in existing_ms:
                milestone_map[cap_key] = existing_ms[title]
            else:
                payload: dict = {
                    "title":       title,
                    "description": ms.get("description", ""),
                }
                due = ms.get("due_date")
                if due:
                    payload["due_on"] = f"{due}T00:00:00Z"
                resp = _rest("POST", f"{API_BASE}/repos/{repo}/milestones", token, json=payload)
                milestone_map[cap_key] = resp.json()["number"]
                created_ms.append(title)
                time.sleep(DELAY)
        if created_ms:
            print(f"  ✅ Created: {', '.join(created_ms)}")
        else:
            print("  ✅ All milestones already exist")

    # ------------------------------------------------------------------
    # Step 3 — Vision issue
    # ------------------------------------------------------------------
    print("\n[3/6] Vision Issue")
    if dry_run:
        print(f"  (dry-run) Would create/update '{VISION_TITLE}'")
    else:
        try:
            resp   = _rest("GET", f"{API_BASE}/repos/{repo}/issues", token,
                           params={"state": "open", "per_page": 100})
            found  = next((i for i in resp.json() if i["title"] == VISION_TITLE), None)
            body   = build_vision_body(config)
            if found:
                _rest("PATCH", f"{API_BASE}/repos/{repo}/issues/{found['number']}", token,
                      json={"body": body})
                print(f"  ✅ Vision issue updated (#{found['number']})")
            else:
                resp = _rest("POST", f"{API_BASE}/repos/{repo}/issues", token,
                             json={"title": VISION_TITLE, "body": body, "labels": []})
                resp.raise_for_status()
                num = resp.json()["number"]
                print(f"  ✅ Vision issue created (#{num})")
                pin_issue(repo, num, token)
        except Exception as e:
            print(f"  ⚠️  Vision issue skipped: {e}")

    # ------------------------------------------------------------------
    # Step 4 — Project + fields
    # ------------------------------------------------------------------
    print("\n[4/6] GitHub Project + Custom Fields")
    project_id: str | None = None
    field_map:  dict       = {}
    if dry_run:
        project_name = config.get("project", {}).get("name", "LLM-Judge Roadmap")
        print(f"  (dry-run) Would create/ensure project '{project_name}'")
        print(f"  (dry-run) Would ensure fields: {', '.join(str(f['name']) for f in CUSTOM_FIELDS)}")
    else:
        try:
            project_name = config.get("project", {}).get("name", "LLM-Judge Roadmap")
            user_id, login = get_viewer(token)
            project_id = find_project(login, project_name, token)
            if project_id:
                print(f"  ✅ Project already exists: '{project_name}'")
            else:
                project_id = create_project(user_id, project_name, token)
            time.sleep(DELAY)

            print("  Setting up custom fields...")
            field_map = ensure_project_fields(project_id, token)
            time.sleep(DELAY)

            # Add L1–L6 draft roadmap items
            existing_titles = get_project_item_titles(project_id, token)
            items_added = 0
            for item in config.get("project", {}).get("roadmap_items", []):
                level = item["level"]
                title = f"[{level}] {item['name']}"
                if title in existing_titles:
                    continue
                caps     = item.get("exit_criteria", {}).get("mandatory_caps", [])
                ms_lines = ""
                if caps:
                    ms_lines = "\n**Capabilities:**\n"
                    for cap in caps:
                        ms        = config.get("milestones", {}).get(cap, {})
                        ms_lines += f"- {ms.get('title', cap)} (due {ms.get('due_date', 'TBD')})\n"
                body = (f"## {level} — {item['name']}\n\n"
                        f"**Horizon:** {item.get('planning_horizon','vision')} | "
                        f"**Target:** {item.get('target_quarter','TBD')}\n\n"
                        f"{item.get('summary','').strip()}\n{ms_lines}")
                item_id = add_draft_item(project_id, title, body, token)
                if item_id:
                    tag_item_fields(project_id, item_id, field_map, {
                        "Roadmap Level":  level,
                        "Status":         item.get("status", "Planned"),
                        "Target Quarter": item.get("target_quarter", ""),
                    }, token)
                    items_added += 1
                time.sleep(DELAY)
            if items_added:
                print(f"  ✅ Added {items_added} roadmap items (L1–L6)")
            else:
                print("  ✅ Roadmap items already present")

        except RuntimeError as e:
            msg = str(e)
            if "not accessible by personal access token" in msg or "Resource not accessible" in msg:
                print("  ❌ Project setup failed — token missing 'project' scope.")
                print("     Fix: https://github.com/settings/tokens → edit token → enable 'project'")
            else:
                print(f"  ⚠️  Project setup skipped: {e}")

    # ------------------------------------------------------------------
    # Step 5 — EPIC issues
    # ------------------------------------------------------------------
    print("\n[5/6] EPIC Issues")
    print("-" * 65)

    _issues:      list[dict[str, Any]] = get_all_issues(repo, token) if not dry_run else []
    existing_map: dict[str, dict[str, Any]] = {i["title"]: i for i in _issues}
    created, updated, closed_list, errors = [], [], [], []

    for epic in epics:
        epic_id = epic["id"]
        title   = epic["title"]
        status  = epic.get("status", "planned")
        cap_key = epic.get("milestone", "")
        ms_num  = milestone_map.get(cap_key)
        mandatory_icon = "🔴" if epic.get("mandatory") else "🟡"

        if dry_run:
            icon   = "✅" if status == "done" else ("🔄" if status == "in_progress" else "📋")
            exists = title in existing_map
            tag    = "(skip)" if exists else "(create)"
            print(f"  {icon} {mandatory_icon} {epic_id}  [{epic.get('level','L3')}/{cap_key}]  {tag}")
            continue

        if title in existing_map:
            existing = existing_map[title]
            num      = existing["number"]
            patched  = []

            if not existing.get("milestone") and ms_num:
                _rest("PATCH", f"{API_BASE}/repos/{repo}/issues/{num}", token,
                      json={"milestone": ms_num})
                patched.append("milestone")
                time.sleep(DELAY)

            if project_id and field_map:
                node_id = get_issue_node_id(repo, num, token)
                if node_id:
                    items = get_project_item_map(project_id, token)
                    item_id = items.get(node_id) or add_issue_to_project(project_id, node_id, token)
                    if item_id:
                        tag_item_fields(project_id, item_id, field_map,
                                        epic_field_values(epic), token)
                        patched.append("fields")

            if status == "done" and existing.get("state") == "open":
                close_issue_with_comment(repo, num, epic.get("close_comment", "Completed."), token)
                patched.append("closed")
                closed_list.append(epic_id)

            label = ", ".join(patched) if patched else "up to date"
            print(f"  {mandatory_icon} {epic_id}: #{num} — {label}")
            updated.append(epic_id)
            continue

        # Create new issue
        try:
            payload = {
                "title":  title,
                "body":   build_epic_body(epic),
                "labels": epic.get("labels", []),
            }
            if ms_num:
                payload["milestone"] = ms_num

            resp = _rest("POST", f"{API_BASE}/repos/{repo}/issues", token, json=payload)
            resp.raise_for_status()
            num = resp.json()["number"]
            print(f"  {mandatory_icon} {epic_id}: Created #{num}")
            created.append(f"{epic_id} #{num}")
            time.sleep(DELAY)

            if project_id and field_map:
                node_id = get_issue_node_id(repo, num, token)
                if node_id:
                    item_id = add_issue_to_project(project_id, node_id, token)
                    if item_id:
                        tag_item_fields(project_id, item_id, field_map,
                                        epic_field_values(epic), token)

            if status == "done":
                close_issue_with_comment(repo, num, epic.get("close_comment", "Completed."), token)
                print("       → Closed (status: done)")
                closed_list.append(epic_id)
                time.sleep(DELAY)

        except requests.HTTPError as e:
            print(f"  ❌ {epic_id}: Failed — {e}")
            errors.append(epic_id)

        time.sleep(DELAY)

    # ------------------------------------------------------------------
    # Step 6 — Summary
    # ------------------------------------------------------------------
    print("\n[6/6] Summary")
    print("=" * 65)
    if not dry_run:
        print(f"  ✅ Created:  {len(created)}")
        for c in created:
            print(f"     {c}")
        print(f"  🔧 Updated:  {len(updated)}")
        print(f"  🔒 Closed:   {len(closed_list)} (status: done)")
        if errors:
            print(f"  ❌ Errors:   {len(errors)}: {', '.join(errors)}")
        print()
        print(f"  🔗 Issues:     https://github.com/{repo}/issues")
        print(f"  🔗 Milestones: https://github.com/{repo}/milestones")
        print(f"  🔗 Project:    https://github.com/users/{owner}/projects")
        print(f"  🔗 Vision:     https://github.com/{repo}/issues?q=VISION")
    print("=" * 65)


# ---------------------------------------------------------------------------
# COMMAND: retag
# ---------------------------------------------------------------------------

def cmd_retag(config: dict, token: str, dry_run: bool) -> None:
    """Sync project custom fields for all EPICs from epics.yaml."""
    repo   = config["repo"]
    epics  = config["epics"]
    owner  = repo.split("/")[0]
    pname  = config.get("project", {}).get("name", "LLM-Judge Roadmap")

    print(f"\n  Repo:    {repo}")
    print(f"  Project: {pname}")
    print(f"  EPICs:   {len(epics)}")
    if dry_run:
        print("\n  DRY RUN — field values that would be applied:\n")
        print(f"  {'EPIC':10} {'Level':6} {'CAP':6} {'Mandatory':10} {'Status':12} {'Quarter'}")
        print("  " + "-" * 58)
        for epic in epics:
            vals = epic_field_values(epic)
            print(f"  {epic['id']:10} {vals['Roadmap Level']:6} "
                  f"{vals['Capability']:6} {vals['Mandatory']:10} "
                  f"{vals['Status']:12} {vals['Target Quarter']}")
        return

    print("\n[1/3] Fetching project and fields...")
    _, login   = get_viewer(token)
    project_id = find_project(login, pname, token)
    if not project_id:
        print(f"  ❌ Project '{pname}' not found. Run `manage.py setup` first.")
        sys.exit(1)

    field_map = ensure_project_fields(project_id, token)
    print(f"  ✅ {len(field_map)} fields ready")

    print("\n[2/3] Fetching issues and project items...")
    all_issues    = {i["title"]: i for i in get_all_issues(repo, token, label="epic")}
    project_items = get_project_item_map(project_id, token)
    print(f"  ✅ {len(all_issues)} EPIC issues, {len(project_items)} project items")

    print("\n[3/3] Re-tagging EPICs...")
    print("-" * 65)
    tagged, added, skipped, errors = 0, 0, 0, 0

    for epic in epics:
        eid   = epic["id"]
        issue = all_issues.get(epic["title"])
        if not issue:
            print(f"  ⚠️  {eid}: Not found in GitHub")
            skipped += 1
            continue

        node_id = get_issue_node_id(repo, issue["number"], token)
        if not node_id:
            print(f"  ❌ {eid}: Could not get node ID")
            errors += 1
            continue

        item_id = project_items.get(node_id)
        if not item_id:
            item_id = add_issue_to_project(project_id, node_id, token)
            if item_id:
                project_items[node_id] = item_id
                added += 1
                time.sleep(DELAY)
            else:
                errors += 1
                continue

        tag_item_fields(project_id, item_id, field_map, epic_field_values(epic), token)
        icon = "🔴" if epic.get("mandatory") else "🟡"
        vals = epic_field_values(epic)
        print(f"  {icon} {eid} #{issue['number']:4}  "
              f"[{vals['Roadmap Level']}/{vals['Capability']}]  "
              f"mandatory={vals['Mandatory']}  status={vals['Status']}")
        tagged += 1
        time.sleep(DELAY)

    print("\n" + "=" * 65)
    print(f"  🏷️  Re-tagged:        {tagged}")
    print(f"  ➕ Added to project: {added}")
    print(f"  ⏭️  Skipped:          {skipped}")
    print(f"  ❌ Errors:           {errors}")
    print(f"\n  🔗 Project: https://github.com/users/{owner}/projects")
    print("=" * 65)


# ---------------------------------------------------------------------------
# COMMAND: recover
# ---------------------------------------------------------------------------

def cmd_recover(config: dict, token: str, dry_run: bool) -> None:
    """Reopen closed EPICs missing milestones, fix them, re-close done ones."""
    repo   = config["repo"]
    epics  = config["epics"]
    owner  = repo.split("/")[0]
    pname  = config.get("project", {}).get("name", "LLM-Judge Roadmap")

    epic_by_title = {e["title"]: e for e in epics}

    print("\nFetching current state from GitHub...")
    ms_title_to_num: dict[str, int] = {}
    live_ms = get_milestones(repo, token)
    for cap_key, ms in config["milestones"].items():
        title = ms["title"]
        if title in live_ms:
            ms_title_to_num[cap_key] = live_ms[title]

    all_epics  = get_all_issues(repo, token, label="epic")
    _, login   = get_viewer(token)
    project_id = find_project(login, pname, token)
    field_map  = get_project_fields(project_id, token) if project_id else {}

    print(f"  Found {len(all_epics)} EPICs | "
          f"project: {'✅' if project_id else '❌'} | "
          f"fields: {list(field_map.keys())}")

    candidates = [i for i in all_epics
                  if not i.get("milestone") and
                  (i.get("state") == "closed" or i.get("state") == "open")]

    print(f"  Candidates (no milestone): {len(candidates)}")
    print("-" * 65)

    if not candidates:
        print("  ✅ No orphaned EPICs found — nothing to recover")
        return

    reopened, patched, reclosed, skipped = [], [], [], []

    for issue in candidates:
        title    = issue["title"]
        number   = issue["number"]
        epic_cfg = epic_by_title.get(title)

        if not epic_cfg:
            print(f"  ⚠️  #{number} — not in epics.yaml, skipping")
            skipped.append(number)
            continue

        cap_key   = epic_cfg.get("milestone", "")
        ms_num    = ms_title_to_num.get(cap_key)
        status    = epic_cfg.get("status", "planned")
        is_closed = issue.get("state") == "closed"

        print(f"  #{number:4} {epic_cfg['id']:8} [{cap_key}] status={status}")

        if dry_run:
            actions = []
            if is_closed and status != "done":
                actions.append("reopen")
            actions.append(f"assign {cap_key}" if ms_num else f"⚠️ {cap_key} not found")
            if project_id:
                actions.append("tag project fields")
            if status == "done" and not is_closed:
                actions.append("close")
            print(f"           → would: {', '.join(actions)}")
            continue

        issue_url = f"{API_BASE}/repos/{repo}/issues/{number}"

        if is_closed and status != "done":
            _rest("PATCH", issue_url, token, json={"state": "open"})
            reopened.append(number)
            print("           ✅ Reopened")
            time.sleep(DELAY)

        if ms_num:
            _rest("PATCH", issue_url, token, json={"milestone": ms_num})
            print(f"           ✅ Assigned {cap_key}")
            time.sleep(DELAY)
        else:
            print(f"           ⚠️  Milestone {cap_key} not found in GitHub")

        if project_id:
            node_id = get_issue_node_id(repo, number, token)
            if node_id:
                items   = get_project_item_map(project_id, token)
                item_id = items.get(node_id) or add_issue_to_project(project_id, node_id, token)
                if item_id:
                    tag_item_fields(project_id, item_id, field_map,
                                    epic_field_values(epic_cfg), token)
                    print("           ✅ Tagged project fields")
            time.sleep(DELAY)

        if status == "done":
            close_issue_with_comment(
                repo, number, epic_cfg.get("close_comment", "Completed."), token
            )
            reclosed.append(number)
            print("           ✅ Re-closed (status: done)")
            time.sleep(DELAY)

        patched.append(number)

    print("\n" + "=" * 65)
    print(f"  🔓 Reopened:  {len(reopened)}")
    print(f"  🔧 Patched:   {len(patched)}")
    print(f"  🔒 Re-closed: {len(reclosed)}")
    print(f"  ⏭️  Skipped:   {len(skipped)}")
    print(f"\n  🔗 Milestones: https://github.com/{repo}/milestones")
    print(f"  🔗 Project:    https://github.com/users/{owner}/projects")
    print("=" * 65)


# ---------------------------------------------------------------------------
# COMMAND: status
# ---------------------------------------------------------------------------

def cmd_status(config: dict, token: str) -> None:
    """Show current state of EPICs — no changes made."""
    repo  = config["repo"]
    epics = config["epics"]
    owner = repo.split("/")[0]

    print("\n  Fetching live state from GitHub...")
    # Fetch ALL issues without label filter — closed issues may lose labels,
    # so match by title against epics.yaml instead of relying on label presence.
    epic_titles = {e["title"] for e in epics}
    all_issues  = {i["title"]: i
                   for i in get_all_issues(repo, token)
                   if i["title"] in epic_titles}
    live_ms     = get_milestones(repo, token)

    # Build CAP → milestone number map
    ms_map: dict[str, int] = {}
    for cap_key, ms in config["milestones"].items():
        title = ms["title"]
        if title in live_ms:
            ms_map[cap_key] = live_ms[title]

    print(f"  {len(all_issues)} EPIC issues found in GitHub\n")

    by_cap: dict[str, list] = defaultdict(list)
    missing_in_gh = []

    for epic in epics:
        issue = all_issues.get(epic["title"])
        cap   = epic.get("milestone", "")
        if issue:
            by_cap[cap].append((epic, issue))
        else:
            missing_in_gh.append(epic["id"])

    # L3 progress
    l3_mandatory = [e for e in epics if e.get("level") == "L3" and e.get("mandatory")]
    l3_done      = sum(1 for e in l3_mandatory
                       if all_issues.get(e["title"], {}).get("state") == "closed")
    l3_pct       = int(l3_done / len(l3_mandatory) * 100) if l3_mandatory else 0
    trigger_pct  = 80
    trigger_n    = int(len(l3_mandatory) * trigger_pct / 100)

    print("=" * 65)
    print("  L3 PROGRESS (Mandatory EPICs)")
    print("=" * 65)
    bar_filled = int(l3_pct / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"  [{bar}] {l3_pct}%  ({l3_done}/{len(l3_mandatory)} mandatory closed)")
    print(f"  L4 planning trigger: {l3_done}/{trigger_n} L3 mandatory EPICs closed")
    print(f"  (Trigger = 80% of {len(l3_mandatory)} mandatory L3 EPICs — no L4 EPICs exist yet)")
    if l3_done >= trigger_n:
        print()
        print("  ⚡ TRIGGER REACHED — time to define L4 capabilities and EPICs")
        print("     → Update epics.yaml: add L4 CAPs, set L4 to planning_horizon: active")
    else:
        remaining = trigger_n - l3_done
        print(f"  → {remaining} more L3 mandatory EPICs to close before L4 planning begins")

    for cap_key in sorted(config["milestones"].keys()):
        ms        = config["milestones"].get(cap_key, {})
        ms_title  = ms.get("title", cap_key)
        epic_list = by_cap.get(cap_key, [])
        done      = sum(1 for e, i in epic_list if i.get("state") == "closed")
        total     = len(epic_list)
        m_done    = sum(1 for e, i in epic_list
                        if e.get("mandatory") and i.get("state") == "closed")
        m_total   = sum(1 for e, _ in epic_list if e.get("mandatory"))
        pct       = int(done / total * 100) if total else 0

        print(f"\n  {ms_title} [{pct}% — {done}/{total} EPICs | {m_done}/{m_total} mandatory]")
        for epic, issue in epic_list:
            state    = issue.get("state", "?")
            num      = issue.get("number", "?")
            mand     = "🔴" if epic.get("mandatory") else "🟡"
            # State icon: driven by live GitHub state, not YAML status
            if state == "closed":
                state_icon = "✅"
            elif epic.get("status") == "in_progress":
                state_icon = "🔄"
            else:
                state_icon = "📋"
            ms_ok = "📌" if issue.get("milestone") else "⚠️ no-ms"
            print(f"    {state_icon} {mand} {epic['id']:8} #{num:<5} {ms_ok}  "
                  f"{epic['title'].replace('[EPIC] ','')[:50]}")

    if missing_in_gh:
        print("\n  ⚠️  EPICs in epics.yaml but NOT found in GitHub:")
        for eid in missing_in_gh:
            print(f"    {eid}")
        print("  Run `manage.py setup` to create them.")

    print("\n" + "=" * 65)
    print(f"  🔗 Issues:     https://github.com/{repo}/issues")
    print(f"  🔗 Milestones: https://github.com/{repo}/milestones")
    print(f"  🔗 Project:    https://github.com/users/{owner}/projects")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="LLM-Judge GitHub management — single entry point for all operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  setup    Full idempotent setup: labels, milestones, vision, project, EPICs
  retag    Sync all project custom fields from epics.yaml (safe to re-run anytime)
  recover  Reopen orphaned closed EPICs, assign milestones, re-close done ones
  status   Print current EPIC and L3 progress — no changes made

examples:
  poetry run python tools/manage.py setup
  poetry run python tools/manage.py setup --dry-run
  poetry run python tools/manage.py retag
  poetry run python tools/manage.py status
"""
    )
    parser.add_argument(
        "command",
        choices=["setup", "retag", "recover", "status"],
        help="Operation to run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes (setup/retag/recover)"
    )
    args = parser.parse_args()

    if not YAML_PATH.exists():
        print(f"ERROR: {YAML_PATH} not found.")
        sys.exit(1)

    with open(YAML_PATH) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print(f"  LLM-Judge GitHub Manager  —  {args.command.upper()}")
    print("=" * 65)

    token = get_token()

    if args.command == "setup":
        cmd_setup(config, token, args.dry_run)
    elif args.command == "retag":
        cmd_retag(config, token, args.dry_run)
    elif args.command == "recover":
        cmd_recover(config, token, args.dry_run)
    elif args.command == "status":
        cmd_status(config, token)


if __name__ == "__main__":
    main()
