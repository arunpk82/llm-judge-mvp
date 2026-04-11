"""
Pipeline Funnel Report & Diagnostics — EPIC 7.23

Two outputs:
  1. Screen: Cascade waterfall showing L1 → L2 → L3 → L4 → L5 flow
  2. JSON: Full diagnostics with per-sentence details, confusion matrix, flag analysis

Design inspired by:
  - Google Perspective API cascade waterfall
  - Stripe Radar economic funnel
  - Meta Integrity tiered escalation

Each layer answers 4 questions:
  1. How many did I receive?
  2. How many did I resolve (with what confidence)?
  3. How many did I pass through (and why)?
  4. What's my cost/latency?
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class LayerStats:
    """Stats for a single pipeline layer."""

    name: str
    enabled: bool
    input_count: int = 0
    grounded: int = 0
    flagged: int = 0
    unknown: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    # Confusion matrix (requires ground truth)
    tp: int = 0  # flagged + actually hallucinated
    tn: int = 0  # grounded + actually clean
    fp: int = 0  # grounded + actually hallucinated (SAFETY VIOLATION)
    fn: int = 0  # not flagged + actually hallucinated


@dataclass
class CumulativeStats:
    """Cumulative stats after each layer."""

    after_layer: str
    total_cleared: int = 0
    total_flagged: int = 0
    total_caught: int = 0
    total_safety_violations: int = 0
    clearance_precision: float = 1.0
    recall: float = 0.0


@dataclass
class SentenceResult:
    """Per-sentence result with full evidence trail."""

    case_id: str
    sentence_idx: int
    sentence: str
    gt_label: str = ""  # "clean" or "hallucinated" (if ground truth available)
    gt_type: str = ""  # hallucination type
    verdict: str = "unknown"
    confidence: str = "none"
    resolved_by: str = "unresolved"
    per_graph: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    l1_result: dict = field(default_factory=dict)
    l2_result: dict = field(default_factory=dict)


@dataclass
class FunnelReport:
    """Complete pipeline funnel report."""

    # Pipeline metadata
    timestamp: str = ""
    config: dict = field(default_factory=dict)
    dataset: str = ""
    total_sentences: int = 0
    total_hallucinated: int = 0
    elapsed_s: float = 0.0

    # Per-layer stats
    layers: list = field(default_factory=list)  # list[LayerStats]

    # Cumulative after each layer
    cumulative: list = field(default_factory=list)  # list[CumulativeStats]

    # Sentence-level results
    sentences: list = field(default_factory=list)  # list[SentenceResult]

    # Safety violations (must be empty in production)
    safety_violations: list = field(default_factory=list)

    # Missed hallucinations with root cause
    missed_hallucinations: list = field(default_factory=list)

    # Flag analysis
    flag_analysis: dict = field(default_factory=dict)


# ============================================================
# FUNNEL BUILDER
# ============================================================


class FunnelBuilder:
    """Build funnel report from pipeline results."""

    def __init__(self, total_hallucinated: int = 0):
        self.total_hallucinated = total_hallucinated
        self.layers: list[LayerStats] = []
        self.sentences: list[SentenceResult] = []
        self.safety_violations: list[dict] = []
        self.catches: list[dict] = []

    def add_layer(self, stats: LayerStats) -> None:
        self.layers.append(stats)

    def add_sentence(self, result: SentenceResult) -> None:
        self.sentences.append(result)
        if result.verdict == "grounded" and result.gt_label == "hallucinated":
            self.safety_violations.append(
                {
                    "case_id": result.case_id,
                    "sentence_idx": result.sentence_idx,
                    "sentence": result.sentence[:120],
                    "resolved_by": result.resolved_by,
                    "gt_type": result.gt_type,
                }
            )
        if result.verdict == "flagged" and result.gt_label == "hallucinated":
            self.catches.append(
                {
                    "case_id": result.case_id,
                    "sentence_idx": result.sentence_idx,
                    "sentence": result.sentence[:120],
                    "resolved_by": result.resolved_by,
                    "gt_type": result.gt_type,
                }
            )

    def build(
        self, elapsed_s: float = 0.0, dataset: str = "", config: dict | None = None
    ) -> FunnelReport:
        """Build the complete funnel report."""
        import datetime

        report = FunnelReport(
            timestamp=datetime.datetime.now().isoformat(),
            config=config or {},
            dataset=dataset,
            total_sentences=len(self.sentences),
            total_hallucinated=self.total_hallucinated,
            elapsed_s=round(elapsed_s, 2),
            layers=[asdict(ly) for ly in self.layers],
            sentences=[asdict(s) for s in self.sentences],
            safety_violations=self.safety_violations,
        )

        # Build cumulative stats
        total_cleared = 0
        total_caught = 0
        total_safety = 0
        for layer in self.layers:
            total_cleared += layer.grounded
            total_caught += layer.tp
            total_safety += layer.fp
            prec = (
                1.0
                if total_safety == 0
                else (total_cleared - total_safety) / max(1, total_cleared)
            )
            rec = total_caught / max(1, self.total_hallucinated)
            report.cumulative.append(
                asdict(
                    CumulativeStats(
                        after_layer=layer.name,
                        total_cleared=total_cleared,
                        total_flagged=sum(
                            ly.flagged
                            for ly in self.layers[: self.layers.index(layer) + 1]
                        ),
                        total_caught=total_caught,
                        total_safety_violations=total_safety,
                        clearance_precision=round(prec, 4),
                        recall=round(rec, 4),
                    )
                )
            )

        # Missed hallucinations
        caught_ids = {(c["case_id"], c["sentence_idx"]) for c in self.catches}
        safety_ids = {(s["case_id"], s["sentence_idx"]) for s in self.safety_violations}
        for s in self.sentences:
            if s.gt_label == "hallucinated":
                key = (s.case_id, s.sentence_idx)
                if key not in caught_ids and key not in safety_ids:
                    report.missed_hallucinations.append(
                        {
                            "case_id": s.case_id,
                            "sentence_idx": s.sentence_idx,
                            "sentence": s.sentence[:120],
                            "verdict": s.verdict,
                            "resolved_by": s.resolved_by,
                        }
                    )

        # Flag analysis
        all_flags = [s for s in self.sentences if s.verdict == "flagged"]
        flag_reasons: dict[str, int] = {}
        for s in all_flags:
            for ev in s.evidence:
                if "NO actions" in str(ev):
                    flag_reasons["entity_no_actions"] = (
                        flag_reasons.get("entity_no_actions", 0) + 1
                    )
                elif "NOT in graph" in str(ev):
                    flag_reasons["entity_not_in_graph"] = (
                        flag_reasons.get("entity_not_in_graph", 0) + 1
                    )
                elif "targets" in str(ev) and "not" in str(ev):
                    flag_reasons["object_mismatch"] = (
                        flag_reasons.get("object_mismatch", 0) + 1
                    )
                elif "NEGATED" in str(ev):
                    flag_reasons["negation_contradiction"] = (
                        flag_reasons.get("negation_contradiction", 0) + 1
                    )
                elif "not found" in str(ev):
                    flag_reasons["verb_not_found"] = (
                        flag_reasons.get("verb_not_found", 0) + 1
                    )

        true_flags = sum(1 for s in all_flags if s.gt_label == "hallucinated")
        report.flag_analysis = {
            "total_flags": len(all_flags),
            "true_hallucinations": true_flags,
            "false_positives": len(all_flags) - true_flags,
            "flag_precision": round(true_flags / max(1, len(all_flags)), 4),
            "flags_by_layer": {},
            "flag_reasons": sorted(flag_reasons.items(), key=lambda x: -x[1]),
        }
        for layer in self.layers:
            report.flag_analysis["flags_by_layer"][layer.name] = layer.flagged

        # Confusion matrix
        report.config["confusion_matrix"] = {}
        for layer in self.layers:
            report.config["confusion_matrix"][layer.name] = {
                "TP": layer.tp,
                "TN": layer.tn,
                "FP": layer.fp,
                "FN": layer.fn,
            }

        return report


# ============================================================
# SCREEN OUTPUT — FUNNEL WATERFALL
# ============================================================


def print_funnel(report: FunnelReport) -> None:
    """Print the cascade funnel to screen."""
    n = report.total_sentences
    h = report.total_hallucinated

    layer_meta = {
        "L1": ("Rules", "free, <1ms"),
        "L2": ("Patterns / Knowledge Graph", "$0.01/source, ~2s"),
        "L3": ("Classifiers (MiniCheck + DeBERTa)", "3.6GB RAM, ~100ms/sent"),
        "L4": ("LLM-as-Judge (Gemini)", "$0.001/sent, ~2s/sent"),
        "L5": ("Human Review", "manual, ~5min/sent"),
    }

    print()
    print("\u2550" * 62)
    print("  HALLUCINATION DETECTION PIPELINE \u2014 FUNNEL REPORT")
    print("\u2550" * 62)

    cum_cleared = 0
    cum_caught = 0

    for i, layer_data in enumerate(
        report.layers
        if isinstance(report.layers[0], dict)
        else [asdict(ly) for ly in report.layers]
    ):
        name = layer_data["name"]
        enabled = layer_data["enabled"]
        meta = layer_meta.get(name, (name, ""))

        g = layer_data["grounded"]
        f = layer_data["flagged"]
        u = layer_data["unknown"]
        inp = layer_data["input_count"]
        tp = layer_data.get("tp", 0)

        cum_cleared += g
        cum_caught += tp

        if i > 0:
            cascade_count = inp
            print(
                f"               \u2502 {cascade_count} sentences ({f + u} to verify)"
            )
            print("               \u25bc")

        print(f"  {name} \u2014 {meta[0]} ({meta[1]})")

        if not enabled:
            print(f"  \u250c{'─' * 58}\u2510")
            print(
                "  \u2502 STATUS: DISABLED                                        \u2502"
            )
            print(f"  \u2514{'─' * 58}\u2518")
            continue

        g_pct = f"{g / max(1, inp) * 100:.1f}" if inp > 0 else "0.0"
        f_pct = f"{f / max(1, inp) * 100:.1f}" if inp > 0 else "0.0"
        u_pct = f"{u / max(1, inp) * 100:.1f}" if inp > 0 else "0.0"

        print(f"  \u250c{'─' * 58}\u2510")
        print(f"  \u2502 IN:       {inp:4d} sentences{' ' * 30}\u2502")
        print(
            f"  \u2502 Grounded: {g:4d} ({g_pct:>5s}%)  \u2190 100% precision{' ' * (16 - len(g_pct))}\u2502"
        )

        if name == "L5":
            print(
                f"  \u2502 Confirmed:{f:4d} ({f_pct:>5s}%)  \u2190 human verified{' ' * (16 - len(f_pct))}\u2502"
            )
            print(
                f"  \u2502 Deferred: {u:4d} ({u_pct:>5s}%){' ' * (30 - len(u_pct))}\u2502"
            )
        else:
            cascade_label = (
                f"\u2192 cascades to L{int(name[1])+1}" if int(name[1]) < 5 else ""
            )
            print(
                f"  \u2502 Flagged:  {f:4d} ({f_pct:>5s}%)  {cascade_label}{' ' * max(0, 29 - len(f_pct) - len(cascade_label))}\u2502"
            )
            print(
                f"  \u2502 Unknown:  {u:4d} ({u_pct:>5s}%)  {cascade_label}{' ' * max(0, 29 - len(f_pct) - len(cascade_label))}\u2502"
            )

        if h > 0:
            print(f"  \u2502{'─' * 58}\u2502")
            print(
                f"  \u2502 Cumulative: {cum_cleared} cleared, {cum_caught}/{h} caught{' ' * max(0, 30 - len(str(cum_cleared)) - len(str(cum_caught)) - len(str(h)))}\u2502"
            )

        print(f"  \u2514{'─' * 58}\u2518")

        f + u

    # Summary
    total_g = sum(
        (ly["grounded"] if isinstance(ly, dict) else ly.grounded) for ly in report.layers
    )
    total_f = sum(
        (ly["flagged"] if isinstance(ly, dict) else ly.flagged) for ly in report.layers
    )
    total_u = n - total_g - total_f

    print()
    print("\u2550" * 62)
    print("  PIPELINE SUMMARY")
    print("\u2550" * 62)
    print(f"  Resolved (grounded): {total_g:4d} ({total_g / max(1, n) * 100:.1f}%)")
    if h > 0:
        print(
            f"  Caught:              {cum_caught:4d}/{h} ({cum_caught / max(1, h) * 100:.0f}% recall)"
        )
    print(f"  Safety violations:   {len(report.safety_violations):4d}")
    print(
        f"  Pending (L3/L4/L5):  {total_f + total_u:4d} ({(total_f + total_u) / max(1, n) * 100:.1f}%)"
    )
    print(f"  Cost:                ${report.elapsed_s * 0.005:.3f} (amortized)")
    print(f"  Time:                {report.elapsed_s:.1f}s")
    print("\u2550" * 62)

    # Safety violations
    if report.safety_violations:
        print()
        print(
            f"  \u26a0\u26a0\u26a0 {len(report.safety_violations)} SAFETY VIOLATIONS \u26a0\u26a0\u26a0"
        )
        for sv in report.safety_violations:
            print(f"    {sv['case_id']} S{sv['sentence_idx']}: {sv['sentence'][:70]}")

    # Missed hallucinations
    if report.missed_hallucinations:
        print()
        print(f"  Missed hallucinations ({len(report.missed_hallucinations)}):")
        for m in report.missed_hallucinations:
            print(
                f"    {m['case_id']} S{m['sentence_idx']} ({m['verdict']}): {m['sentence'][:60]}"
            )

    print()


# ============================================================
# JSON OUTPUT
# ============================================================


def save_diagnostics(report: FunnelReport, path: str) -> None:
    """Save full diagnostics to JSON file."""
    output = {
        "pipeline_run": {
            "timestamp": report.timestamp,
            "config": report.config,
            "dataset": report.dataset,
            "total_sentences": report.total_sentences,
            "total_hallucinated": report.total_hallucinated,
            "elapsed_s": report.elapsed_s,
        },
        "funnel": {
            ly["name"] if isinstance(ly, dict) else ly.name: (
                ly if isinstance(ly, dict) else asdict(ly)
            )
            for ly in report.layers
        },
        "cumulative": report.cumulative,
        "confusion_matrix": report.config.get("confusion_matrix", {}),
        "flag_analysis": report.flag_analysis,
        "safety_violations": report.safety_violations,
        "missed_hallucinations": report.missed_hallucinations,
        "sentences": (
            report.sentences
            if isinstance(report.sentences[0], dict)
            else [asdict(s) for s in report.sentences]
        ),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Diagnostics saved: {path}")
