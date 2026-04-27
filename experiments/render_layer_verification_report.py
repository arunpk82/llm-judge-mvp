"""Layer-by-layer verification report renderer (platform-as-harness).

Reads the artifacts produced by a standard batch run invoked with
``--isolate-layer <layer>`` and renders an HTML + Markdown verification
report. **No platform bypass** — the renderer only consumes existing
artifacts written by the Runner / CAP-5 path:

  * ``<batch_run>/cases/<case_id>/manifest.json``  — envelope (proves
    CAP-7 ran)
  * ``<batch_run>/cases/<case_id>/events.jsonl``   — sub-capability
    boundary events (proves L1 boundaries fired)
  * ``<single_eval_root>/<case_id>/manifest.json`` — full CAP-5
    manifest carrying ``verdict.layers_requested``,
    ``verdict.layer_stats`` and ``verdict.sentence_results``

Ground truth is loaded from the registered benchmark
(e.g. ``ragtruth_50``); span annotations are aligned to the same
sentence segmentation CAP-7 used (``_split_sentences`` from
``llm_judge.calibration.hallucination``) so per-sentence labels line
up with ``verdict.sentence_results[i].sentence_idx``.

Verdict logic (PASS / FAIL / PARTIAL):
  * FAIL    — precision below ``--target-precision``
  * PASS    — precision ≥ target AND detection within tolerance
  * PARTIAL — precision ≥ target AND detection outside tolerance
              (algorithm still high-precision, but corpus/segmentation
              shift moved the detection rate; surfaces for review)

The renderer is parameterised by layer ID, target spec, and benchmark.
L2-L5 verification will reuse this same script — only the CLI args
change. The output is written under
``reports/verifications/<layer>_<benchmark>_<timestamp>/``.

Mirrors ``tools/_batch_html_report.py``'s rendering pattern: Rich
``Console(record=True, force_terminal=True)`` + small ``_render_*``
helpers + dual ``console.export_text(clear=False)`` /
``console.export_html()`` export. No Jinja2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_judge.benchmarks import BenchmarkCase
from llm_judge.benchmarks.registry import build
from llm_judge.calibration.hallucination import _split_sentences


# Replicated from tools/_batch_html_report.py to avoid importing through
# tools/run_batch_evaluation.py's sys.path shim. If a third verification
# renderer ever needs the same constants, factor out a helpers module.
_RAG_STYLES = {
    "green": "[green]●[/green] green",
    "amber": "[yellow]●[/yellow] amber",
    "red": "[red]●[/red] red",
}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class CaseEvidence:
    """Per-case evidence assembled from the three artifact sources."""

    case_id: str
    response_text: str
    sentence_count: int
    layers_requested: list[str]
    layer_stats: dict[str, int | float]
    cleared_indices: set[int]
    sub_capability_events: list[dict[str, Any]]
    span_annotations: list[dict[str, Any]]


@dataclass
class CaseClassification:
    """Per-case TP/FP/TN/FN counts (sentence-level)."""

    case_id: str
    sentence_count: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def cleared(self) -> int:
        return self.tp + self.fp


@dataclass
class VerificationMetrics:
    """Aggregated metrics across all cases."""

    total_sentences: int
    total_responses: int
    tp: int
    fp: int
    tn: int
    fn: int
    detection_rate: float
    precision: float
    recall: float
    f1: float


def _load_benchmark_cases(benchmark: str) -> dict[str, BenchmarkCase]:
    """Load benchmark cases keyed by case_id (== response id used by the
    batch driver). Allows O(1) ground-truth lookup per batch case."""
    adapter = build(benchmark)
    cases: dict[str, BenchmarkCase] = {}
    case_iter: Iterator[BenchmarkCase] = adapter.load_cases()
    for case in case_iter:
        cases[case.case_id] = case
    return cases


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts (empty if missing)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gather_case_evidence(
    *,
    case_id: str,
    batch_run_dir: Path,
    single_eval_root: Path,
    benchmark_case: BenchmarkCase,
) -> CaseEvidence:
    """Assemble per-case evidence from batch_run + single_eval artifacts.

    Surfaces a ``ValueError`` with a concrete artifact path on missing
    fields rather than silently coercing — surprising data shapes should
    fail loud, not produce a report with hidden zeros.
    """
    case_dir = batch_run_dir / "cases" / case_id
    events_path = case_dir / "events.jsonl"
    sub_cap_events = [
        e
        for e in _read_jsonl(events_path)
        if e.get("event", "").startswith("sub_capability_")
    ]

    cap5_manifest_path = single_eval_root / case_id / "manifest.json"
    if not cap5_manifest_path.exists():
        raise FileNotFoundError(
            f"renderer: CAP-5 manifest missing at {cap5_manifest_path}; "
            f"the renderer needs verdict data from the single_eval store. "
            f"Was this batch run with case_id={case_id!r}?"
        )
    manifest = _read_json(cap5_manifest_path)
    verdict = manifest.get("verdict", {})
    layer_stats = dict(verdict.get("layer_stats", {}))
    layers_requested = list(verdict.get("layers_requested", []))
    sentence_results = list(verdict.get("sentence_results", []))

    if "total_sentences" not in layer_stats:
        raise ValueError(
            f"renderer: verdict.layer_stats missing 'total_sentences' "
            f"in {cap5_manifest_path}; refresh the batch run after the "
            f"CAP-7 wrapper change that surfaces total_sentences."
        )
    sentence_count = int(layer_stats["total_sentences"])

    cleared_indices = {
        int(sr["sentence_idx"])
        for sr in sentence_results
        if sr.get("resolved_by") == "L1"
    }

    response_text = benchmark_case.request.candidate_answer
    span_annotations = [
        {
            "start": ann.start,
            "end": ann.end,
            "text": ann.text,
            "label_type": ann.label_type,
        }
        for ann in benchmark_case.ground_truth.span_annotations
    ]

    return CaseEvidence(
        case_id=case_id,
        response_text=response_text,
        sentence_count=sentence_count,
        layers_requested=layers_requested,
        layer_stats=layer_stats,
        cleared_indices=cleared_indices,
        sub_capability_events=sub_cap_events,
        span_annotations=span_annotations,
    )


def _classify_case(evidence: CaseEvidence) -> CaseClassification:
    """Compute sentence-level confusion matrix for a single case.

    A sentence is *grounded* when no ground-truth span overlaps it; a
    sentence is *cleared* when it appears in
    ``verdict.sentence_results`` with ``resolved_by == "L1"``. The four
    quadrants follow:

      TP — cleared & grounded     (L1 correctly cleared)
      FP — cleared & ungrounded   (L1 incorrectly cleared) → precision-killer
      TN — uncleared & ungrounded (L1 correctly let the cascade through)
      FN — uncleared & grounded   (L1 missed an easy one)

    Sentence segmentation matches CAP-7 by reusing
    ``_split_sentences``; alignment is positional (sentence_idx ↔ list
    index from spaCy doc.sents with the same length filter).
    """
    sents = _split_sentences(evidence.response_text)
    if len(sents) != evidence.sentence_count:
        # Loud failure: sentence-segmentation drift between CAP-7 run
        # time and renderer would silently miscount the matrix. If this
        # ever fires, the response_text we read from the benchmark
        # adapter has been mutated since the batch run, or spaCy
        # version-skewed mid-run.
        raise ValueError(
            f"renderer: sentence-count mismatch for {evidence.case_id} — "
            f"manifest says {evidence.sentence_count}, "
            f"renderer split returns {len(sents)}. Re-run the batch "
            f"after pinning spaCy / model version."
        )

    # Per-sentence ground-truth labels via span containment.
    # ``span.text`` is the verbatim hallucinated substring; if it
    # appears in a sentence, that sentence is ungrounded. Char offsets
    # are not required because the registered RAGTruth annotations
    # already carry the offending text.
    span_texts = [s["text"] for s in evidence.span_annotations if s["text"]]
    ungrounded_idx: set[int] = set()
    for i, sent in enumerate(sents):
        if any(span and span in sent for span in span_texts):
            ungrounded_idx.add(i)

    tp = fp = tn = fn = 0
    for i in range(evidence.sentence_count):
        cleared = i in evidence.cleared_indices
        ungrounded = i in ungrounded_idx
        if cleared and not ungrounded:
            tp += 1
        elif cleared and ungrounded:
            fp += 1
        elif not cleared and ungrounded:
            tn += 1
        else:
            fn += 1

    return CaseClassification(
        case_id=evidence.case_id,
        sentence_count=evidence.sentence_count,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def _aggregate_metrics(
    classifications: list[CaseClassification],
) -> VerificationMetrics:
    """Sum per-case quadrants and compute detection / precision / recall / F1."""
    tp = sum(c.tp for c in classifications)
    fp = sum(c.fp for c in classifications)
    tn = sum(c.tn for c in classifications)
    fn = sum(c.fn for c in classifications)
    total = tp + fp + tn + fn

    detection_rate = (tp + fp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return VerificationMetrics(
        total_sentences=total,
        total_responses=len(classifications),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        detection_rate=detection_rate,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _compute_verdict(
    metrics: VerificationMetrics,
    *,
    target_detection: float,
    tolerance_detection: float,
    target_precision: float,
) -> tuple[str, list[str]]:
    """Return ``(verdict, findings)``.

    Findings is an ordered list of human-readable lines suitable for
    inclusion in the report's findings panel.
    """
    findings: list[str] = []
    detection_low = target_detection - tolerance_detection
    detection_high = target_detection + tolerance_detection
    detection_in_tolerance = detection_low <= metrics.detection_rate <= detection_high
    precision_at_target = metrics.precision >= target_precision

    if not precision_at_target:
        verdict = "FAIL"
        findings.append(
            f"Precision {metrics.precision:.4f} below target "
            f"{target_precision:.4f} — L1 cleared {metrics.fp} sentence(s) "
            f"that contain hallucination spans (false positives)."
        )
    elif detection_in_tolerance:
        verdict = "PASS"
        findings.append(
            f"Precision {metrics.precision:.4f} ≥ target "
            f"{target_precision:.4f} — every L1-cleared sentence is "
            f"correctly grounded ({metrics.tp} TP, 0 FP)."
        )
        findings.append(
            f"Detection rate {metrics.detection_rate:.4f} within "
            f"tolerance [{detection_low:.4f}, {detection_high:.4f}]."
        )
    else:
        verdict = "PARTIAL"
        findings.append(
            f"Precision {metrics.precision:.4f} ≥ target "
            f"{target_precision:.4f} (algorithm still high-precision)."
        )
        findings.append(
            f"Detection rate {metrics.detection_rate:.4f} OUTSIDE "
            f"tolerance [{detection_low:.4f}, {detection_high:.4f}] — "
            f"corpus / segmentation shift, NOT an algorithm failure. "
            f"Investigate before promoting."
        )
    return verdict, findings


# ----------------------------------------------------------------------
# Rendering — Rich Console pattern, mirrors tools/_batch_html_report.py
# ----------------------------------------------------------------------


def _render_header(
    console: Console,
    *,
    layer: str,
    benchmark: str,
    batch_run_dir: Path,
    output_dir: Path,
    verdict: str,
) -> None:
    verdict_color = {"PASS": "green", "FAIL": "red", "PARTIAL": "yellow"}[verdict]
    body = (
        f"[bold]layer[/bold]         {layer}\n"
        f"[bold]benchmark[/bold]     {benchmark}\n"
        f"[bold]batch_run[/bold]     {batch_run_dir}\n"
        f"[bold]rendered_at[/bold]   {_utc_now_iso()}\n"
        f"[bold]output_dir[/bold]    {output_dir}\n"
        f"[bold]verdict[/bold]       [{verdict_color}]{verdict}[/{verdict_color}]"
    )
    console.print(Panel(body, title="Layer verification report", style="cyan"))


def _render_summary(
    console: Console,
    metrics: VerificationMetrics,
    *,
    target_detection: float,
    tolerance_detection: float,
    target_precision: float,
) -> None:
    detection_low = target_detection - tolerance_detection
    detection_high = target_detection + tolerance_detection
    detection_in_tolerance = detection_low <= metrics.detection_rate <= detection_high
    precision_at_target = metrics.precision >= target_precision

    detection_label = (
        f"[green]{metrics.detection_rate:.4f}[/green]"
        if detection_in_tolerance
        else f"[yellow]{metrics.detection_rate:.4f}[/yellow]"
    )
    precision_label = (
        f"[green]{metrics.precision:.4f}[/green]"
        if precision_at_target
        else f"[red]{metrics.precision:.4f}[/red]"
    )

    table = Table(
        title="Verification summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    table.add_row(
        "Detection rate",
        detection_label,
        f"{target_detection:.4f} ± {tolerance_detection:.4f}",
        _RAG_STYLES["green"] if detection_in_tolerance else _RAG_STYLES["amber"],
    )
    table.add_row(
        "Precision",
        precision_label,
        f"≥ {target_precision:.4f}",
        _RAG_STYLES["green"] if precision_at_target else _RAG_STYLES["red"],
    )
    table.add_row("Recall", f"{metrics.recall:.4f}", "—", "[dim]informational[/dim]")
    table.add_row("F1", f"{metrics.f1:.4f}", "—", "[dim]informational[/dim]")
    console.print(table)


def _render_metrics_detail(
    console: Console, metrics: VerificationMetrics
) -> None:
    body = (
        f"[bold]TP[/bold]               {metrics.tp}\n"
        f"[bold]FP[/bold]               {metrics.fp}\n"
        f"[bold]TN[/bold]               {metrics.tn}\n"
        f"[bold]FN[/bold]               {metrics.fn}\n"
        f"[bold]total_sentences[/bold]  {metrics.total_sentences}\n"
        f"[bold]total_responses[/bold]  {metrics.total_responses}"
    )
    console.print(Panel(body, title="Sentence-level confusion matrix", style="blue"))


def _render_per_case_attribution(
    console: Console, classifications: list[CaseClassification]
) -> None:
    if not classifications:
        console.print("[dim](no cases)[/dim]")
        return
    table = Table(
        title="Per-case attribution",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("case_id")
    table.add_column("sentences", justify="right")
    table.add_column("L1 cleared", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("TN", justify="right")
    table.add_column("FN", justify="right")
    for c in classifications:
        fp_label = (
            f"[red]{c.fp}[/red]" if c.fp > 0 else "[dim]0[/dim]"
        )
        table.add_row(
            c.case_id,
            str(c.sentence_count),
            str(c.cleared),
            str(c.tp),
            fp_label,
            str(c.tn),
            str(c.fn),
        )
    console.print(table)


def _render_subcap_events(
    console: Console,
    layer: str,
    sample_evidence: CaseEvidence | None,
    boundary_summary: dict[str, Counter[str]],
) -> None:
    """Show CAP-7 sub-capability boundary firing across the run.

    The summary is ``{sub_capability_id: Counter({event_type: n, ...})}``
    aggregated across cases — flat enough for L1's four boundaries today,
    extends naturally to L2-L5 instrumentation as it lands.
    """
    if not boundary_summary:
        console.print(
            Panel(
                f"No CAP-7 sub-capability events captured for layer {layer}. "
                f"Either the run did not exercise this layer, or the "
                f"layer is not yet instrumented (sub-capability "
                f"instrumentation rolls out level-by-level).",
                title="Sub-capability events",
                style="yellow",
            )
        )
        return
    table = Table(
        title=f"{layer} sub-capability boundary events (across {len(boundary_summary)} boundaries)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("sub_capability_id")
    table.add_column("started", justify="right")
    table.add_column("completed", justify="right")
    table.add_column("skipped", justify="right")
    table.add_column("typical reason / status")
    for sub_id, counter in sorted(boundary_summary.items()):
        started = counter.get("sub_capability_started", 0)
        completed = counter.get("sub_capability_completed", 0)
        skipped = counter.get("sub_capability_skipped", 0)
        # Pull a representative reason from the sample case (helps the
        # reader see WHY a SOFT boundary skipped without dumping every
        # event).
        reason = ""
        if sample_evidence is not None:
            for ev in sample_evidence.sub_capability_events:
                if ev.get("sub_capability_id") == sub_id:
                    if ev.get("event") == "sub_capability_skipped":
                        reason = f"skipped: {ev.get('reason', '')[:60]}"
                        break
                    if ev.get("event") == "sub_capability_completed":
                        reason = f"completed ({ev.get('status', 'success')})"
                        break
        table.add_row(
            sub_id,
            str(started),
            str(completed),
            str(skipped),
            reason,
        )
    console.print(table)


def _render_findings(console: Console, verdict: str, findings: list[str]) -> None:
    style = {"PASS": "green", "FAIL": "red", "PARTIAL": "yellow"}[verdict]
    body = "\n".join(f"• {line}" for line in findings) if findings else "(none)"
    console.print(Panel(body, title=f"Findings — {verdict}", style=style))


def _render_provenance(
    console: Console,
    *,
    batch_run_dir: Path,
    single_eval_root: Path,
    layer: str,
    benchmark: str,
) -> None:
    body = (
        f"[bold]layer[/bold]            {layer}\n"
        f"[bold]benchmark[/bold]        {benchmark}\n"
        f"[bold]batch_run_dir[/bold]    {batch_run_dir}\n"
        f"[bold]single_eval_root[/bold] {single_eval_root}\n"
        f"[bold]rendered_at[/bold]      {_utc_now_iso()}"
    )
    console.print(Panel(body, title="Provenance", style="blue"))


def render_verification_report(
    *,
    batch_run_dir: Path,
    single_eval_root: Path,
    benchmark: str,
    layer: str,
    target_detection: float,
    tolerance_detection: float,
    target_precision: float,
    output_dir: Path,
) -> str:
    """Render the verification report and return the verdict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        d for d in (batch_run_dir / "cases").iterdir() if d.is_dir()
    )
    if not case_dirs:
        raise ValueError(
            f"renderer: no cases under {batch_run_dir / 'cases'}; "
            f"is this a completed batch run?"
        )
    case_ids = [d.name for d in case_dirs]

    benchmark_cases = _load_benchmark_cases(benchmark)
    missing = [cid for cid in case_ids if cid not in benchmark_cases]
    if missing:
        raise ValueError(
            f"renderer: case_id(s) {missing[:5]} present in batch_run "
            f"but not in benchmark {benchmark!r}. Mismatched batch / "
            f"benchmark pair."
        )

    evidence_list: list[CaseEvidence] = []
    for cid in case_ids:
        evidence_list.append(
            _gather_case_evidence(
                case_id=cid,
                batch_run_dir=batch_run_dir,
                single_eval_root=single_eval_root,
                benchmark_case=benchmark_cases[cid],
            )
        )

    classifications = [_classify_case(e) for e in evidence_list]
    metrics = _aggregate_metrics(classifications)
    verdict, findings = _compute_verdict(
        metrics,
        target_detection=target_detection,
        tolerance_detection=tolerance_detection,
        target_precision=target_precision,
    )

    boundary_summary: dict[str, Counter[str]] = {}
    for ev in evidence_list:
        for event in ev.sub_capability_events:
            sub_id = event.get("sub_capability_id", "")
            ev_type = event.get("event", "")
            if not sub_id:
                continue
            if sub_id not in boundary_summary:
                boundary_summary[sub_id] = Counter()
            boundary_summary[sub_id][ev_type] += 1
    sample_evidence = evidence_list[0] if evidence_list else None

    console = Console(record=True, force_terminal=True)
    console.print()
    _render_header(
        console,
        layer=layer,
        benchmark=benchmark,
        batch_run_dir=batch_run_dir,
        output_dir=output_dir,
        verdict=verdict,
    )
    _render_summary(
        console,
        metrics,
        target_detection=target_detection,
        tolerance_detection=tolerance_detection,
        target_precision=target_precision,
    )
    _render_metrics_detail(console, metrics)
    _render_per_case_attribution(console, classifications)
    _render_subcap_events(console, layer, sample_evidence, boundary_summary)
    _render_findings(console, verdict, findings)
    _render_provenance(
        console,
        batch_run_dir=batch_run_dir,
        single_eval_root=single_eval_root,
        layer=layer,
        benchmark=benchmark,
    )

    md_path = output_dir / "verification_report.md"
    html_path = output_dir / "verification_report.html"
    md_path.write_text(console.export_text(clear=False), encoding="utf-8")
    html_path.write_text(console.export_html(), encoding="utf-8")

    summary_path = output_dir / "verification_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "layer": layer,
                "benchmark": benchmark,
                "verdict": verdict,
                "metrics": {
                    "detection_rate": metrics.detection_rate,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "tn": metrics.tn,
                    "fn": metrics.fn,
                    "total_sentences": metrics.total_sentences,
                    "total_responses": metrics.total_responses,
                },
                "targets": {
                    "detection": target_detection,
                    "tolerance": tolerance_detection,
                    "precision": target_precision,
                },
                "findings": findings,
                "batch_run_dir": str(batch_run_dir),
                "single_eval_root": str(single_eval_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return verdict


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a layer-by-layer verification report from an existing "
            "batch run. Reads only existing artifacts — does not re-invoke "
            "the hallucination pipeline (platform-as-harness)."
        )
    )
    parser.add_argument(
        "--batch-run",
        type=Path,
        required=True,
        help="Path to a completed batch run directory (reports/batch_runs/<id>).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        choices=["L1", "L2", "L3", "L4", "L5"],
        help="Which CAP-7 layer this run isolated.",
    )
    parser.add_argument(
        "--target-detection",
        type=float,
        required=True,
        help="Target detection rate (e.g. 0.074 for L1's 7.4% target).",
    )
    parser.add_argument(
        "--tolerance-detection",
        type=float,
        required=True,
        help="Two-sided tolerance for detection rate (e.g. 0.005 for ±0.5pp).",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        required=True,
        help="Minimum acceptable precision (e.g. 1.0 for L1's 100% target).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Registered benchmark name for ground-truth lookup (e.g. ragtruth_50).",
    )
    parser.add_argument(
        "--single-eval-root",
        type=Path,
        default=Path("reports/single_eval"),
        help=(
            "Root directory where CAP-5 manifests live "
            "(default: reports/single_eval). Each case_id reads its "
            "verdict from <single-eval-root>/<case_id>/manifest.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: "
            "reports/verifications/<layer>_<benchmark>_<UTC-timestamp>/)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    batch_run_dir = args.batch_run.resolve()
    if not batch_run_dir.is_dir():
        print(f"error: batch_run {batch_run_dir} is not a directory", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        output_dir = Path("reports/verifications") / (
            f"{args.layer.lower()}_{args.benchmark}_{ts}"
        )

    try:
        verdict = render_verification_report(
            batch_run_dir=batch_run_dir,
            single_eval_root=args.single_eval_root,
            benchmark=args.benchmark,
            layer=args.layer,
            target_detection=args.target_detection,
            tolerance_detection=args.tolerance_detection,
            target_precision=args.target_precision,
            output_dir=output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"renderer failed: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"output: {output_dir}/")
    print(f"verdict: {verdict}")
    print(f"report: {output_dir / 'verification_report.html'}")
    return 0 if verdict in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    sys.exit(main())
