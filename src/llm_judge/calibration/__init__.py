"""
LLM Judge Registry, Calibration Pipeline, and Trust Gate (EPIC 7.1).

L4 capability: No untested LLM judge evaluates production responses.

Components:
  - JudgeMeta: registered judge configuration
  - load_judge_registry(): load configs/judges/registry.yaml
  - CalibrationResult: per-dimension accuracy from calibration run
  - run_calibration(): run judge against golden dataset, compute accuracy
  - check_trust_gate(): verify judge is calibrated before production use
  - CalibratedJudge: JudgeEngine wrapper that enforces trust gate
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_judge.judge_base import JudgeEngine
from llm_judge.paths import config_root, state_root
from llm_judge.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

JUDGE_REGISTRY_PATH = config_root() / "judges" / "registry.yaml"
CALIBRATION_DIR = state_root() / "calibration"

VALID_JUDGE_STATUSES = frozenset(
    {
        "registered",
        "calibrating",
        "calibrated",
        "blocked",
    }
)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# =====================================================================
# Judge Registry
# =====================================================================


@dataclass(frozen=True)
class JudgeCalibrationConfig:
    """Calibration requirements for a judge."""

    min_accuracy: float = 0.70
    min_dimension_accuracy: float = 0.60
    golden_dataset_id: str = "golden"
    golden_dataset_version: str = "v1"


@dataclass(frozen=True)
class JudgeMeta:
    """Registered LLM judge configuration."""

    judge_id: str
    provider: str
    model: str
    prompt_version: str
    domain: str
    status: str
    calibration_config: JudgeCalibrationConfig


@dataclass(frozen=True)
class TrustGateConfig:
    """Trust gate configuration."""

    enforce: bool = True
    allow_deterministic: bool = True
    default_confidence_threshold: float = 0.6


def load_judge_registry(
    path: Path = JUDGE_REGISTRY_PATH,
) -> tuple[dict[str, JudgeMeta], TrustGateConfig]:
    """Load judge registry from YAML config."""
    if not path.exists():
        raise FileNotFoundError(f"Judge registry not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Judge registry must be a YAML mapping: {path}")

    judges: dict[str, JudgeMeta] = {}
    for judge_id, meta in raw.get("judges", {}).items():
        if not isinstance(meta, dict):
            continue

        cal_raw = meta.get("calibration", {})
        if not isinstance(cal_raw, dict):
            cal_raw = {}

        cal_config = JudgeCalibrationConfig(
            min_accuracy=float(cal_raw.get("min_accuracy", 0.70)),
            min_dimension_accuracy=float(cal_raw.get("min_dimension_accuracy", 0.60)),
            golden_dataset_id=str(cal_raw.get("golden_dataset_id", "golden")),
            golden_dataset_version=str(cal_raw.get("golden_dataset_version", "v1")),
        )

        judges[str(judge_id)] = JudgeMeta(
            judge_id=str(judge_id),
            provider=str(meta.get("provider", "unknown")),
            model=str(meta.get("model", judge_id)),
            prompt_version=str(meta.get("prompt_version", "v1")),
            domain=str(meta.get("domain", "general")),
            status=str(meta.get("status", "registered")),
            calibration_config=cal_config,
        )

    tg_raw = raw.get("trust_gate", {})
    if not isinstance(tg_raw, dict):
        tg_raw = {}

    trust_gate = TrustGateConfig(
        enforce=bool(tg_raw.get("enforce", True)),
        allow_deterministic=bool(tg_raw.get("allow_deterministic", True)),
        default_confidence_threshold=float(
            tg_raw.get("default_confidence_threshold", 0.6)
        ),
    )

    return judges, trust_gate


# =====================================================================
# Calibration Pipeline
# =====================================================================


@dataclass
class DimensionAccuracy:
    """Accuracy for a single scoring dimension."""

    dimension: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class CalibrationResult:
    """Result of running a judge against the golden dataset."""

    judge_id: str
    golden_dataset_id: str
    golden_dataset_version: str
    timestamp: str
    cases_evaluated: int = 0
    decision_matches: int = 0
    dimension_accuracies: dict[str, DimensionAccuracy] = field(default_factory=dict)
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return (
            self.decision_matches / self.cases_evaluated
            if self.cases_evaluated > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "golden_dataset_id": self.golden_dataset_id,
            "golden_dataset_version": self.golden_dataset_version,
            "timestamp": self.timestamp,
            "cases_evaluated": self.cases_evaluated,
            "decision_matches": self.decision_matches,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "dimension_accuracies": {
                dim: {
                    "correct": da.correct,
                    "total": da.total,
                    "accuracy": round(da.accuracy, 4),
                }
                for dim, da in self.dimension_accuracies.items()
            },
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


def run_calibration(
    *,
    judge: JudgeEngine,
    judge_meta: JudgeMeta,
    golden_cases: list[dict[str, Any]],
    dimensions: list[str] | None = None,
) -> CalibrationResult:
    """
    Run a judge against golden dataset cases and compute calibration metrics.

    Each golden case must have: conversation, candidate_answer, human_decision,
    and optionally human_scores (dict of dimension → score).
    """
    from llm_judge.schemas import Message

    if dimensions is None:
        dimensions = ["relevance", "clarity", "correctness", "tone"]

    result = CalibrationResult(
        judge_id=judge_meta.judge_id,
        golden_dataset_id=judge_meta.calibration_config.golden_dataset_id,
        golden_dataset_version=judge_meta.calibration_config.golden_dataset_version,
        timestamp=_utc_now_iso(),
    )

    # Initialize dimension tracking
    for dim in dimensions:
        result.dimension_accuracies[dim] = DimensionAccuracy(dimension=dim)

    for case in golden_cases:
        try:
            conv = [Message(**m) for m in case.get("conversation", [])]
            req = PredictRequest(
                conversation=conv,
                candidate_answer=str(case.get("candidate_answer", "")),
                rubric_id=str(case.get("rubric_id", "chat_quality")),
            )
            response = judge.evaluate(req)
        except Exception as e:
            logger.warning(
                "calibration.case_failed",
                extra={"judge_id": judge_meta.judge_id, "error": str(e)},
            )
            continue

        result.cases_evaluated += 1

        # Compare decisions
        human_decision = case.get("human_decision")
        if human_decision and response.decision == human_decision:
            result.decision_matches += 1

        # Compare per-dimension scores
        human_scores = case.get("human_scores")
        if isinstance(human_scores, dict) and isinstance(response.scores, dict):
            for dim in dimensions:
                da = result.dimension_accuracies[dim]
                h_score = human_scores.get(dim)
                j_score = response.scores.get(dim)
                if h_score is not None and j_score is not None:
                    da.total += 1
                    # "Close enough" — within 1 point counts as correct
                    if abs(int(h_score) - int(j_score)) <= 1:
                        da.correct += 1

    # Check against thresholds
    cal = judge_meta.calibration_config

    if result.overall_accuracy < cal.min_accuracy:
        result.failure_reasons.append(
            f"Overall accuracy {result.overall_accuracy:.2%} < {cal.min_accuracy:.0%}"
        )

    for dim, da in result.dimension_accuracies.items():
        if da.total > 0 and da.accuracy < cal.min_dimension_accuracy:
            result.failure_reasons.append(
                f"Dimension '{dim}' accuracy {da.accuracy:.2%} < {cal.min_dimension_accuracy:.0%}"
            )

    result.passed = len(result.failure_reasons) == 0 and result.cases_evaluated > 0

    return result


def save_calibration_result(
    result: CalibrationResult,
    calibration_dir: Path = CALIBRATION_DIR,
) -> Path:
    """Save calibration result to reports/calibration/."""
    calibration_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.judge_id}_{result.timestamp.replace(':', '-')}.json"
    path = calibration_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_latest_calibration(
    judge_id: str,
    calibration_dir: Path = CALIBRATION_DIR,
) -> CalibrationResult | None:
    """Load the most recent calibration result for a judge."""
    if not calibration_dir.exists():
        return None

    matching = sorted(
        calibration_dir.glob(f"{judge_id}_*.json"),
        key=lambda p: p.name,
        reverse=True,
    )

    if not matching:
        return None

    with matching[0].open("r", encoding="utf-8") as f:
        data = json.load(f)

    dim_accuracies = {}
    for dim, da_data in data.get("dimension_accuracies", {}).items():
        da = DimensionAccuracy(dimension=dim)
        da.correct = da_data.get("correct", 0)
        da.total = da_data.get("total", 0)
        dim_accuracies[dim] = da

    result = CalibrationResult(
        judge_id=data["judge_id"],
        golden_dataset_id=data.get("golden_dataset_id", ""),
        golden_dataset_version=data.get("golden_dataset_version", ""),
        timestamp=data.get("timestamp", ""),
        cases_evaluated=data.get("cases_evaluated", 0),
        decision_matches=data.get("decision_matches", 0),
        dimension_accuracies=dim_accuracies,
        passed=data.get("passed", False),
        failure_reasons=data.get("failure_reasons", []),
    )
    return result


# =====================================================================
# Trust Gate
# =====================================================================


def check_trust_gate(
    *,
    judge_id: str,
    engine_choice: str = "deterministic",
    registry_path: Path = JUDGE_REGISTRY_PATH,
    calibration_dir: Path = CALIBRATION_DIR,
) -> tuple[bool, str]:
    """
    Check if a judge is allowed to evaluate production responses.

    Returns (allowed: bool, reason: str).
    """
    # Deterministic judge always allowed
    if engine_choice == "deterministic":
        return True, "Deterministic judge — no calibration required"

    try:
        judges, trust_gate = load_judge_registry(registry_path)
    except FileNotFoundError:
        if engine_choice == "deterministic":
            return True, "No registry — deterministic allowed"
        return False, "Judge registry not found — cannot verify calibration"

    if judge_id not in judges:
        return (
            False,
            f"Judge '{judge_id}' not registered in configs/judges/registry.yaml",
        )

    meta = judges[judge_id]

    if meta.status == "blocked":
        return False, f"Judge '{judge_id}' is blocked"

    if meta.status == "calibrated":
        # Verify calibration result still exists and passed
        cal_result = load_latest_calibration(judge_id, calibration_dir)
        if cal_result and cal_result.passed:
            return (
                True,
                f"Judge '{judge_id}' calibrated (accuracy: {cal_result.overall_accuracy:.0%})",
            )
        # Calibration result missing or failed — revert to uncalibrated
        if not trust_gate.enforce:
            return (
                True,
                f"Judge '{judge_id}' calibration unverified — trust gate not enforced (warning)",
            )
        return (
            False,
            f"Judge '{judge_id}' marked calibrated but no passing calibration result found",
        )

    if meta.status in ("registered", "calibrating"):
        if not trust_gate.enforce:
            return (
                True,
                f"Judge '{judge_id}' not yet calibrated — trust gate not enforced (warning)",
            )
        return (
            False,
            f"Judge '{judge_id}' is {meta.status} — must be calibrated before production use",
        )

    return False, f"Judge '{judge_id}' has unknown status: {meta.status}"


# =====================================================================
# CalibratedJudge — JudgeEngine wrapper with trust gate
# =====================================================================


class CalibratedJudge(JudgeEngine):
    """
    Wraps an LLM judge with trust gate enforcement.

    If the judge isn't calibrated and the trust gate is enforced,
    evaluate() raises RuntimeError instead of producing ungoverned results.
    """

    def __init__(
        self,
        judge: JudgeEngine,
        judge_id: str,
        *,
        registry_path: Path = JUDGE_REGISTRY_PATH,
        calibration_dir: Path = CALIBRATION_DIR,
        confidence_threshold: float = 0.6,
    ) -> None:
        self._judge = judge
        self._judge_id = judge_id
        self._registry_path = registry_path
        self._calibration_dir = calibration_dir
        self._confidence_threshold = confidence_threshold

        # Check trust gate at initialization — fail fast
        allowed, reason = check_trust_gate(
            judge_id=judge_id,
            engine_choice="llm",
            registry_path=registry_path,
            calibration_dir=calibration_dir,
        )
        if not allowed:
            raise RuntimeError(f"Trust gate blocked: {reason}")

        self._gate_reason = reason
        logger.info(
            "judge.trust_gate.passed",
            extra={"judge_id": judge_id, "reason": reason},
        )

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        response = self._judge.evaluate(request)

        # Tag low-confidence responses for potential human adjudication
        if response.confidence < self._confidence_threshold:
            flags = list(response.flags) + [f"low_confidence:{response.confidence:.2f}"]
            response = PredictResponse(
                decision=response.decision,
                overall_score=response.overall_score,
                scores=response.scores,
                confidence=response.confidence,
                flags=flags,
                explanations=response.explanations,
            )

        return response


# =====================================================================
# CLI
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM Judge management for LLM-Judge (L4)."
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List registered judges with calibration status")
    sub.add_parser("status", help="Show trust gate status for all judges")

    args = parser.parse_args()

    if args.cmd == "list":
        try:
            judges, trust_gate = load_judge_registry()
        except FileNotFoundError:
            print("No judge registry found at configs/judges/registry.yaml")
            return 1

        print("")
        print("Registered LLM Judges")
        print("-" * 80)
        print(
            f"{'JUDGE_ID':<18} {'MODEL':<18} {'STATUS':<14} "
            f"{'DOMAIN':<10} {'MIN_ACC':<8} {'CALIBRATED':<10}"
        )
        print("-" * 80)

        for jid, meta in sorted(judges.items()):
            cal = load_latest_calibration(jid)
            cal_str = f"{cal.overall_accuracy:.0%}" if cal and cal.passed else "no"
            print(
                f"{meta.judge_id:<18} {meta.model:<18} {meta.status:<14} "
                f"{meta.domain:<10} {meta.calibration_config.min_accuracy:<8.0%} {cal_str:<10}"
            )

        print(
            f"\nTrust gate: {'ENFORCED' if trust_gate.enforce else 'NOT enforced (warnings only)'}"
        )
        print(f"Confidence threshold: {trust_gate.default_confidence_threshold}")
        return 0

    if args.cmd == "status":
        try:
            judges, _ = load_judge_registry()
        except FileNotFoundError:
            print("No judge registry found.")
            return 1

        print("")
        print("Trust Gate Status")
        print("-" * 60)

        for jid in sorted(judges.keys()):
            allowed, reason = check_trust_gate(judge_id=jid, engine_choice="llm")
            status = "ALLOWED" if allowed else "BLOCKED"
            print(f"  {jid:<20} {status:<10} {reason}")

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
