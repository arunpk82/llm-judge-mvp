from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _mk_case(
    prompt: str,
    candidate: str,
    expected_decision: str,
    expected_flags: list[str],
    *,
    rubric_id: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "conversation": [{"role": "user", "content": prompt}],
        "candidate_answer": candidate,
        "rubric_id": rubric_id,
        "expected_decision": expected_decision,
        "expected_flags": expected_flags,
        "meta": meta,
    }


def generate_math_basic_cases(
    n: int,
    *,
    seed: int,
    rubric_id: str = "chat_quality",
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    ops = ["+", "-", "*"]
    cases: list[dict[str, Any]] = []

    for i in range(n):
        a = rng.randint(-50, 50)
        b = rng.randint(-50, 50)
        op = rng.choice(ops)
        expr = f"{a} {op} {b}"
        correct = eval(expr)  # controlled inputs

        # Create incorrect answer that is guaranteed different
        delta = rng.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
        incorrect = correct + delta
        if incorrect == correct:
            incorrect += 1

        # Randomize candidate formatting to stress parser
        prompt = rng.choice(
            [
                f"What is {expr}?",
                f"Compute {expr}.",
                f"Solve: {expr}",
                f"Can you tell me {expr}?",
            ]
        )

        # Candidate variants: sometimes bare number, sometimes sentence.
        # IMPORTANT: keep a “bare number” variant so detector parsing stays reliable.
        correct_candidate = rng.choice([str(correct), f"{correct}", f"The answer is {correct}."])
        incorrect_candidate = rng.choice([str(incorrect), f"{incorrect}", f"The answer is {incorrect}."])

        # 50/50 pass/fail
        if rng.random() < 0.5:
            cases.append(
                _mk_case(
                    prompt,
                    correct_candidate,
                    "pass",
                    [],  # no expected correctness flags
                    rubric_id=rubric_id,
                    meta={
                        "generator": "math_basic",
                        "seed": seed,
                        "i": i,
                        "expr": expr,
                        "truth": correct,
                        "variant": "correct",
                    },
                )
            )
        else:
            cases.append(
                _mk_case(
                    prompt,
                    incorrect_candidate,
                    "fail",
                    ["correctness.math_incorrect"],
                    rubric_id=rubric_id,
                    meta={
                        "generator": "math_basic",
                        "seed": seed,
                        "i": i,
                        "expr": expr,
                        "truth": correct,
                        "variant": "incorrect",
                    },
                )
            )

    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets/deterministic/math_basic_v1.jsonl")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rubric-id", default="chat_quality")
    args = p.parse_args()

    rows = generate_math_basic_cases(args.n, seed=args.seed, rubric_id=args.rubric_id)
    write_jsonl(Path(args.out), rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()