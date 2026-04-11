"""
EPIC 7.20: Science Gate — QG-QA (Question Generation + Question Answering)

Approach: For each response sentence:
  1. Generate a factual question from the claim
  2. Answer the question using ONLY the source document
  3. Compare: if the answer differs or is unanswerable → hallucination

Models:
  QG: valhalla/t5-base-qg-hl (question generation from highlighted text)
  QA: deepset/roberta-base-squad2 (extractive QA with unanswerable support)

Usage:
    poetry run python -m llm_judge.benchmarks.qgqa_science_gate --max-fn 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _compute_grounding_ratio,
    _split_sentences,
)
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)

QG_MODEL = "valhalla/t5-base-qg-hl"
QA_MODEL = "deepset/roberta-base-squad2"


@dataclass
class QGQASentenceResult:
    """QG-QA result for one response sentence."""

    sentence: str
    sentence_idx: int
    generated_questions: list[str]
    qa_results: list[dict[str, Any]]  # question, answer, score, match
    has_mismatch: bool
    mismatch_details: str


@dataclass
class QGQACaseResult:
    """QG-QA result for one RAGTruth case."""

    case_id: str
    ground_truth: str
    gate1_ratio: float
    gate1_min_sim: float
    has_mismatch: bool
    qgqa_decision: str  # "fail" if mismatch, "pass" if all match
    correct: bool
    sentences: list[QGQASentenceResult]


def load_models():
    """Load QG and QA models."""
    print(f"Loading QG model: {QG_MODEL}")
    start = time.time()
    qg_tokenizer = AutoTokenizer.from_pretrained(QG_MODEL, use_fast=False)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL)
    qg_model.eval()
    print(f"  QG loaded in {time.time() - start:.1f}s")

    print(f"Loading QA model: {QA_MODEL}")
    start = time.time()
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    qa_model.eval()
    print(f"  QA loaded in {time.time() - start:.1f}s")

    return qg_tokenizer, qg_model, (qa_tokenizer, qa_model)


def generate_questions(
    sentence: str,
    qg_tokenizer: Any,
    qg_model: Any,
    max_questions: int = 3,
) -> list[str]:
    """
    Generate factual questions from a response sentence.

    The T5 QG model expects input like: "generate question: <sentence>"
    or with highlight markers: "generate question: <hl> fact <hl> context"
    """
    questions = []

    # Strategy 1: Direct question generation from full sentence
    input_text = f"generate question: {sentence}"
    input_ids = qg_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).input_ids

    with torch.no_grad():
        outputs = qg_model.generate(
            input_ids,
            max_length=64,
            num_beams=max_questions,
            num_return_sequences=min(max_questions, 3),
            early_stopping=True,
        )

    for output in outputs:
        q = qg_tokenizer.decode(output, skip_special_tokens=True).strip()
        if q and q not in questions:
            questions.append(q)

    return questions[:max_questions]


def answer_question(
    question: str,
    context: str,
    qa_pipe: Any,
) -> dict[str, Any]:
    """
    Answer a question using the source document.

    Uses AutoModelForQuestionAnswering directly.
    roberta-base-squad2 supports unanswerable questions.
    """
    qa_tokenizer, qa_model = qa_pipe
    try:
        inputs = qa_tokenizer(
            question,
            context[:4096],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = qa_model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Get best answer span
        start_idx = int(torch.argmax(start_logits).item())
        end_idx = int(torch.argmax(end_logits).item())

        # Check if unanswerable (CLS token selected = no answer)
        # For squad2 models, if start=0 and end=0, it means unanswerable
        if start_idx == 0 and end_idx == 0:
            return {
                "answer": "",
                "score": 0.0,
                "answerable": False,
            }

        # Calculate confidence score
        start_score = torch.softmax(start_logits, dim=0)[start_idx].item()
        end_score = torch.softmax(end_logits, dim=0)[end_idx].item()
        score = (start_score + end_score) / 2

        # Extract answer text
        if end_idx >= start_idx:
            answer_ids = inputs["input_ids"][0][int(start_idx) : int(end_idx) + 1]
            answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        else:
            answer = ""
            score = 0.0

        return {
            "answer": answer,
            "score": round(score, 4),
            "answerable": score > 0.01 and len(answer.strip()) > 0,
        }
    except Exception as e:
        return {
            "answer": "",
            "score": 0.0,
            "answerable": False,
            "error": str(e)[:80],
        }


def compare_answer_to_claim(
    sentence: str,
    question: str,
    qa_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare QA answer against the original claim.

    Returns match assessment.
    """
    answer = qa_result.get("answer", "").strip().lower()
    answerable = qa_result.get("answerable", False)
    score = qa_result.get("score", 0.0)

    if not answerable:
        return {
            "match": False,
            "reason": "unanswerable",
            "detail": f"Question '{question}' has no answer in source (score={score:.3f})",
        }

    # Simple containment check: does the answer appear in the sentence?
    sentence_lower = sentence.lower()
    if answer in sentence_lower:
        return {
            "match": True,
            "reason": "answer_in_claim",
            "detail": f"Answer '{answer}' found in response sentence",
        }

    # Check if key terms overlap
    answer_words = set(answer.split())
    sentence_words = set(sentence_lower.split())
    overlap = answer_words & sentence_words
    if len(overlap) >= len(answer_words) * 0.5:
        return {
            "match": True,
            "reason": "partial_overlap",
            "detail": f"Answer '{answer}' partially overlaps with claim",
        }

    return {
        "match": False,
        "reason": "answer_mismatch",
        "detail": f"Source says '{answer}' but response claims something different",
    }


def find_test_cases(
    max_cases: int = 500, max_fn: int = 10, max_tn: int = 10
) -> tuple[list, list]:
    """Find both false negatives AND true negatives from Gate 1 PASS cases."""
    adapter = RAGTruthAdapter()
    get_embedding_provider()  # init singleton
    fns: list[tuple] = []
    tns: list[tuple] = []

    for case in adapter.load_cases(max_cases=max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            continue

        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        context = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        response = case.request.candidate_answer

        ratio, min_sim = _compute_grounding_ratio(
            response, context, similarity_threshold=0.60
        )

        if ratio >= 0.80 and min_sim >= 0.30:
            source_doc = "\n".join(ctx_parts) if ctx_parts else context
            entry = (case, source_doc, response, ratio, min_sim)
            if gt == "fail" and len(fns) < max_fn:
                fns.append(entry)
            elif gt == "pass" and len(tns) < max_tn:
                tns.append(entry)

        if len(fns) >= max_fn and len(tns) >= max_tn:
            break

    return fns, tns


def run_qgqa_on_case(
    case: Any,
    source_doc: str,
    response: str,
    ratio: float,
    min_sim: float,
    gt: str,
    qg_tokenizer: Any,
    qg_model: Any,
    qa_pipe: Any,
) -> QGQACaseResult:
    """Run QG-QA on all sentences of one case."""
    resp_sents = _split_sentences(response)

    if not resp_sents:
        return QGQACaseResult(
            case_id=case.case_id,
            ground_truth=gt,
            gate1_ratio=ratio,
            gate1_min_sim=min_sim,
            has_mismatch=False,
            qgqa_decision="pass",
            correct=(gt == "pass"),
            sentences=[],
        )

    sentence_results = []
    has_mismatch = False

    for i, sent in enumerate(resp_sents):
        # Step 1: Generate questions from response sentence
        questions = generate_questions(sent, qg_tokenizer, qg_model)

        # Step 2 & 3: Answer each question from source, compare
        qa_results = []
        sent_has_mismatch = False
        mismatch_detail = ""

        for q in questions:
            qa_result = answer_question(q, source_doc, qa_pipe)
            comparison = compare_answer_to_claim(sent, q, qa_result)
            qa_results.append(
                {
                    "question": q,
                    "answer": qa_result.get("answer", ""),
                    "score": qa_result.get("score", 0.0),
                    "answerable": qa_result.get("answerable", False),
                    "match": comparison["match"],
                    "reason": comparison["reason"],
                    "detail": comparison["detail"],
                }
            )
            if not comparison["match"]:
                sent_has_mismatch = True
                mismatch_detail = comparison["detail"]

        if sent_has_mismatch:
            has_mismatch = True

        sentence_results.append(
            QGQASentenceResult(
                sentence=sent,
                sentence_idx=i,
                generated_questions=questions,
                qa_results=qa_results,
                has_mismatch=sent_has_mismatch,
                mismatch_details=mismatch_detail,
            )
        )

    qgqa_decision = "fail" if has_mismatch else "pass"
    correct = qgqa_decision == gt

    return QGQACaseResult(
        case_id=case.case_id,
        ground_truth=gt,
        gate1_ratio=ratio,
        gate1_min_sim=min_sim,
        has_mismatch=has_mismatch,
        qgqa_decision=qgqa_decision,
        correct=correct,
        sentences=sentence_results,
    )


def print_report(
    fn_results: list[QGQACaseResult], tn_results: list[QGQACaseResult]
) -> None:
    """Print Science Gate report for both FN and TN."""
    fn_caught = sum(1 for r in fn_results if r.qgqa_decision == "fail")
    tn_correct = sum(1 for r in tn_results if r.qgqa_decision == "pass")
    tn_fp = len(tn_results) - tn_correct

    print()
    print("=" * 70)
    print("EXPERIMENT 16: SCIENCE GATE — QG-QA VALIDATION")
    print("=" * 70)
    print("\nFALSE NEGATIVES (gt=fail, should catch):")
    print(f"  Tested: {len(fn_results)}")
    print(f"  Caught: {fn_caught}/{len(fn_results)}")
    print("\nTRUE NEGATIVES (gt=pass, should NOT flag):")
    print(f"  Tested: {len(tn_results)}")
    print(f"  Correctly passed: {tn_correct}/{len(tn_results)}")
    print(f"  Wrongly flagged (FP): {tn_fp}/{len(tn_results)}")

    tp, fp, fn, tn = fn_caught, tn_fp, len(fn_results) - fn_caught, tn_correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print("\nOVERALL:")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print("  Pass criteria: catch >= 70% AND FP <= 30%")
    pass_criteria = (
        fn_caught >= len(fn_results) * 0.7 and tn_fp <= len(tn_results) * 0.3
    )
    print(f"  Result: {'PASS' if pass_criteria else 'FAIL'}")
    print("=" * 70)

    all_results = fn_results + tn_results
    total_questions = sum(
        len(s.generated_questions) for r in all_results for s in r.sentences
    )
    total_mismatches = sum(
        1 for r in all_results for s in r.sentences if s.has_mismatch
    )
    total_sents = sum(len(r.sentences) for r in all_results)
    print("\nSTATISTICS")
    print(f"  Total sentences: {total_sents}")
    print(f"  Total questions generated: {total_questions}")
    print(f"  Sentences with mismatches: {total_mismatches}/{total_sents}")

    for label, results in [
        ("FALSE NEGATIVES (gt=fail)", fn_results),
        ("TRUE NEGATIVES (gt=pass)", tn_results),
    ]:
        print(f"\n--- {label} ---")
        for r in results:
            if r.ground_truth == "fail":
                marker = "CAUGHT" if r.qgqa_decision == "fail" else "MISSED"
            else:
                marker = "OK" if r.qgqa_decision == "pass" else "FALSE POSITIVE"
            print(
                f"\n  {r.case_id} [{marker}] gt={r.ground_truth} decision={r.qgqa_decision}"
            )
            for s in r.sentences:
                flag = " <<<< MISMATCH" if s.has_mismatch else ""
                print(f"    [{s.sentence_idx+1}]{flag}")
                print(f"      {s.sentence[:100]}")
                for qa in s.qa_results:
                    if not qa["match"]:
                        print(f"      Q: {qa['question']}")
                        print(
                            f"      A: '{qa['answer']}' (score={qa['score']:.3f}, {qa['reason']})"
                        )
                        print(f"      >> {qa['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 16: QG-QA Science Gate")
    parser.add_argument("--max-fn", type=int, default=10)
    parser.add_argument("--max-tn", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find test cases
    print("Finding test cases (FN + TN from Gate 1 PASS)...")
    fn_cases, tn_cases = find_test_cases(
        max_cases=args.max_cases,
        max_fn=args.max_fn,
        max_tn=args.max_tn,
    )
    print(f"Found {len(fn_cases)} FN + {len(tn_cases)} TN")

    if not fn_cases and not tn_cases:
        print("No test cases found.")
        return

    # Step 2: Load models
    qg_tokenizer, qg_model, qa_pipe = load_models()

    # Step 3: Run QG-QA on each case
    fn_results: list[QGQACaseResult] = []
    tn_results: list[QGQACaseResult] = []

    for label, cases, gt_val, result_list in [
        ("FN", fn_cases, "fail", fn_results),
        ("TN", tn_cases, "pass", tn_results),
    ]:
        for idx, (case, source_doc, response, ratio, min_sim) in enumerate(cases):
            print(f"\n{label} {idx+1}/{len(cases)}: {case.case_id}")
            start = time.time()
            result = run_qgqa_on_case(
                case,
                source_doc,
                response,
                ratio,
                min_sim,
                gt_val,
                qg_tokenizer,
                qg_model,
                qa_pipe,
            )
            elapsed = time.time() - start
            marker = "CORRECT" if result.correct else "WRONG"
            print(f"  {result.qgqa_decision} ({elapsed:.1f}s) {marker}")
            result_list.append(result)

    # Step 4: Report
    print_report(fn_results, tn_results)

    # Step 5: Save results
    all_results = fn_results + tn_results
    fn_caught = sum(bool(r.qgqa_decision == "fail") for r in fn_results)
    tn_fp = sum(bool(r.qgqa_decision == "fail") for r in tn_results)
    tp, fp = fn_caught, tn_fp
    fn_missed = len(fn_results) - fn_caught
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn_missed) if (tp + fn_missed) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    save_data = {
        "experiment": "Experiment 16: QG-QA Science Gate",
        "qg_model": QG_MODEL,
        "qa_model": QA_MODEL,
        "fn_tested": len(fn_results),
        "fn_caught": fn_caught,
        "tn_tested": len(tn_results),
        "tn_false_positives": tn_fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cases": [
            {
                "case_id": r.case_id,
                "ground_truth": r.ground_truth,
                "qgqa_decision": r.qgqa_decision,
                "correct": r.correct,
                "sentences": [
                    {
                        "sentence": s.sentence[:150],
                        "questions": s.generated_questions,
                        "has_mismatch": s.has_mismatch,
                        "qa_results": s.qa_results,
                    }
                    for s in r.sentences
                ],
            }
            for r in all_results
        ],
    }
    save_path = output_dir / "qgqa_science_gate_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
