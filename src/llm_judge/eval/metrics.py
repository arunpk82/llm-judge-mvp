from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

Label = Literal["pass", "fail"]


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    def precision(self) -> float:
        d = self.tp + self.fp
        return 0.0 if d == 0 else self.tp / d

    def recall(self) -> float:
        d = self.tp + self.fn
        return 0.0 if d == 0 else self.tp / d

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _as_label(x: Any) -> Label | None:
    if x == "pass":
        return "pass"
    if x == "fail":
        return "fail"
    return None


def confusion_for_positive(judgments: Iterable[dict[str, Any]], *, positive: Label) -> Confusion:
    tp = fp = tn = fn = 0
    for j in judgments:
        h = _as_label(j.get("human_decision"))
        y = _as_label(j.get("judge_decision"))
        if h is None or y is None:
            continue

        h_pos = h == positive
        y_pos = y == positive

        if h_pos and y_pos:
            tp += 1
        elif (not h_pos) and y_pos:
            fp += 1
        elif (not h_pos) and (not y_pos):
            tn += 1
        else:
            fn += 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def cohen_kappa(judgments: Iterable[dict[str, Any]]) -> tuple[float | None, list[str]]:
    warnings: list[str] = []

    n = 0
    n_pp = n_pf = n_fp = n_ff = 0
    for j in judgments:
        h = _as_label(j.get("human_decision"))
        y = _as_label(j.get("judge_decision"))
        if h is None or y is None:
            continue
        n += 1
        if h == "pass" and y == "pass":
            n_pp += 1
        elif h == "pass" and y == "fail":
            n_pf += 1
        elif h == "fail" and y == "pass":
            n_fp += 1
        else:
            n_ff += 1

    if n == 0:
        return None, ["kappa_no_valid_labels"]

    p0 = (n_pp + n_ff) / n

    p_h_pass = (n_pp + n_pf) / n
    p_h_fail = (n_fp + n_ff) / n
    p_y_pass = (n_pp + n_fp) / n
    p_y_fail = (n_pf + n_ff) / n

    pe = (p_h_pass * p_y_pass) + (p_h_fail * p_y_fail)
    denom = 1.0 - pe
    if denom == 0.0:
        return None, ["kappa_undefined_due_to_label_collapse"]

    return (p0 - pe) / denom, warnings


def compute_metrics(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    c_pass = confusion_for_positive(judgments, positive="pass")
    c_fail = confusion_for_positive(judgments, positive="fail")

    valid_pairs: list[tuple[Label, Label]] = []
    for j in judgments:
        h = _as_label(j.get("human_decision"))
        y = _as_label(j.get("judge_decision"))
        if h is None or y is None:
            continue
        valid_pairs.append((h, y))

    n_valid = len(valid_pairs)
    human_pass_rate = 0.0
    judge_pass_rate = 0.0
    if n_valid:
        human_pass_rate = sum(1 for (h, _) in valid_pairs if h == "pass") / n_valid
        judge_pass_rate = sum(1 for (_, y) in valid_pairs if y == "pass") / n_valid

    kappa, kappa_warnings = cohen_kappa(judgments)

    out: dict[str, Any] = {
        "n_cases_total": len(judgments),
        "n_cases_scored": n_valid,
        "human_pass_rate": human_pass_rate,
        "judge_pass_rate": judge_pass_rate,
        "confusion_pass_positive": {"tp": c_pass.tp, "fp": c_pass.fp, "tn": c_pass.tn, "fn": c_pass.fn},
        "confusion_fail_positive": {"tp": c_fail.tp, "fp": c_fail.fp, "tn": c_fail.tn, "fn": c_fail.fn},
        "precision_pass": c_pass.precision(),
        "recall_pass": c_pass.recall(),
        "f1_pass": c_pass.f1(),
        "precision_fail": c_fail.precision(),
        "recall_fail": c_fail.recall(),
        "f1_fail": c_fail.f1(),
        "cohen_kappa": kappa,
    }

    warnings: list[str] = []
    warnings.extend(kappa_warnings)
    if warnings:
        out["warnings"] = warnings
    return out
