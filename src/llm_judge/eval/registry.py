from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from llm_judge.paths import state_root

REGISTRY_SCHEMA_VERSION = "1.0"
DEFAULT_REGISTRY_PATH = state_root() / "run_registry.jsonl"


@dataclass(frozen=True)
class RunRegistryEntry:
    schema_version: str
    run_id: str
    created_at_utc: str

    # Core identifiers (governance keys)
    dataset_id: str
    dataset_version: str
    rubric_id: str
    judge_engine: str

    # Run stats
    cases_total: int
    cases_evaluated: int
    sampled: bool

    # Reproducibility
    dataset_hash: str

    # Convenience pointers
    run_dir: str

    # Metrics snapshot (small, denormalized)
    metrics: dict[str, Any]

    # Optional metadata
    git_sha: str | None = None
    policy_gate: dict[str, Any] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                # keep registry robust: skip malformed lines rather than hard-fail list commands
                continue
            if isinstance(obj, dict):
                yield obj


def append_run_registry_entry(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    cases_total: int,
    cases_evaluated: int,
    sampled: bool,
    dataset_id: str,
    dataset_version: str,
    rubric_id: str,
    judge_engine: str,
    dataset_hash: str,
) -> None:
    """
    Append a single run record to reports/run_registry.jsonl.

    This is intentionally "append-only" for auditability and simplicity.
    """
    run_id = run_dir.name
    entry = RunRegistryEntry(
        schema_version=REGISTRY_SCHEMA_VERSION,
        run_id=run_id,
        created_at_utc=_utc_now_iso(),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        rubric_id=rubric_id,
        judge_engine=judge_engine,
        cases_total=cases_total,
        cases_evaluated=cases_evaluated,
        sampled=sampled,
        dataset_hash=dataset_hash,
        run_dir=str(run_dir),
        metrics=metrics if isinstance(metrics, dict) else {},
        git_sha=str(manifest.get("git_sha")) if isinstance(manifest.get("git_sha"), str) else None,
    )

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.__dict__, sort_keys=True) + "\n")


def _print_table(rows: list[dict[str, Any]], *, metric: str = "f1") -> None:
    # Compact tabular view
    # Columns: run_id, dataset@ver, rubric, metric, cases_evaluated, engine, created_at
    header = (
        f"{'RUN ID':<28}  {'DATASET':<16}  {'RUBRIC':<16}  {metric.upper():<8}  "
        f"{'CASES':<6}  {'ENGINE':<14}  {'CREATED_AT(UTC)':<20}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        run_id = str(r.get("run_id", ""))[:28]
        ds = f"{r.get('dataset_id','')}@{r.get('dataset_version','')}"
        rubric = str(r.get("rubric_id", ""))[:16]
        engine = str(r.get("judge_engine", ""))[:14]
        created = str(r.get("created_at_utc", ""))[:20]
        cases = _safe_int(r.get("cases_evaluated"), 0)

        m = r.get("metrics", {})
        mv = ""
        if isinstance(m, dict) and metric in m and isinstance(m.get(metric), (int, float)):
            mv = f"{float(m[metric]):.4f}"
        else:
            mv = "n/a"

        print(f"{run_id:<28}  {ds:<16}  {rubric:<16}  {mv:<8}  {cases:<6}  {engine:<14}  {created:<20}")


def cmd_list(args: argparse.Namespace) -> int:
    path = Path(args.registry)
    entries = list(_iter_jsonl(path))

    # Filters
    if args.dataset_id:
        entries = [e for e in entries if str(e.get("dataset_id")) == args.dataset_id]
    if args.rubric_id:
        entries = [e for e in entries if str(e.get("rubric_id")) == args.rubric_id]
    if args.engine:
        entries = [e for e in entries if str(e.get("judge_engine")) == args.engine]

    # Newest first (best-effort ordering by created_at_utc)
    entries.sort(key=lambda e: str(e.get("created_at_utc", "")), reverse=True)

    if args.limit is not None:
        entries = entries[: int(args.limit)]

    _print_table(entries, metric=args.metric)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = Path(args.registry)
    run_id = args.run_id

    for e in _iter_jsonl(path):
        if str(e.get("run_id")) == run_id:
            print(json.dumps(e, indent=2, sort_keys=True))
            return 0

    print(f"Run not found in registry: {run_id}")
    return 1


def cmd_trend(args: argparse.Namespace) -> int:
    path = Path(args.registry)
    metric = args.metric

    entries = list(_iter_jsonl(path))
    if args.dataset_id:
        entries = [e for e in entries if str(e.get("dataset_id")) == args.dataset_id]
    if args.rubric_id:
        entries = [e for e in entries if str(e.get("rubric_id")) == args.rubric_id]

    # Oldest -> newest for trend readability
    entries.sort(key=lambda e: str(e.get("created_at_utc", "")))

    if args.last is not None:
        entries = entries[-int(args.last) :]

    print(f"Trend for metric='{metric}'")
    for e in entries:
        created = str(e.get("created_at_utc", ""))[:20]
        run_id = str(e.get("run_id", ""))
        m = e.get("metrics", {})
        v: Optional[float] = None
        if isinstance(m, dict) and isinstance(m.get(metric), (int, float)):
            v = float(m[metric])
        print(f"{created}  {run_id}  {('%.6f' % v) if v is not None else 'n/a'}")

    return 0


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run registry / observability utilities for LLM-Judge.")
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH), help="Path to run_registry.jsonl")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List recent runs from the registry")
    p_list.add_argument("--limit", type=int, default=25, help="Max number of rows (default: 25)")
    p_list.add_argument("--metric", default="f1", help="Metric column to display (default: f1)")
    p_list.add_argument("--dataset-id", default=None, help="Filter by dataset_id")
    p_list.add_argument("--rubric-id", default=None, help="Filter by rubric_id")
    p_list.add_argument("--engine", default=None, help="Filter by judge_engine")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show a single run entry as JSON")
    p_show.add_argument("run_id", help="Run id (directory name under reports/runs)")
    p_show.set_defaults(func=cmd_show)

    p_trend = sub.add_parser("trend", help="Print a simple metric trend over time")
    p_trend.add_argument("--metric", required=True, help="Metric name (e.g., f1, accuracy, cohen_kappa)")
    p_trend.add_argument("--dataset-id", default=None, help="Filter by dataset_id")
    p_trend.add_argument("--rubric-id", default=None, help="Filter by rubric_id")
    p_trend.add_argument("--last", type=int, default=20, help="Last N points (default: 20)")
    p_trend.set_defaults(func=cmd_trend)

    return p


def main() -> int:
    args = build_cli().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())