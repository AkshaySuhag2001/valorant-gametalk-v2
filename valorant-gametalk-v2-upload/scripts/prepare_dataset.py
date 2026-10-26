#!/usr/bin/env python3
"""Prepare reviewed draft rows for training and isolated evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VALID_CATEGORIES = {
    "everyday_with_tie_in",
    "everyday_plain",
    "valorant_factual",
}
VALID_SPLITS = {"train", "validation", "final_holdout"}
REQUIRED_FIELDS = {
    "id",
    "split",
    "intended_split",
    "category",
    "question",
    "answer",
    "valorant_terms",
    "provenance",
    "review_status",
}


def normalized_question(question: str) -> str:
    return re.sub(r"\W+", " ", question.casefold(), flags=re.UNICODE).strip()


def read_approved_rows(data_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_output = output_dir.resolve()

    for jsonl_path in sorted(data_dir.rglob("*.jsonl")):
        resolved_path = jsonl_path.resolve()
        if resolved_output == resolved_path or resolved_output in resolved_path.parents:
            continue

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path}:{line_number}: {exc.msg}"
                    ) from exc
                if row.get("review_status") != "approved":
                    continue

                missing = REQUIRED_FIELDS - row.keys()
                if missing:
                    raise ValueError(
                        f"Approved row {jsonl_path}:{line_number} is missing: "
                        + ", ".join(sorted(missing))
                    )
                intended_split = row.get("intended_split")
                effective_split = intended_split if intended_split else row["split"]
                if effective_split == "draft":
                    raise ValueError(
                        f"Approved row {row['id']} has no assigned split. Set intended_split "
                        f"to one of {sorted(VALID_SPLITS)}."
                    )
                if effective_split not in VALID_SPLITS:
                    raise ValueError(
                        f"Approved row {row['id']} has invalid split {effective_split!r}; "
                        f"use one of {sorted(VALID_SPLITS)}."
                    )
                if row["category"] not in VALID_CATEGORIES:
                    raise ValueError(
                        f"Approved row {row['id']} has invalid category "
                        f"{row['category']!r}."
                    )
                if not isinstance(row["question"], str) or not row["question"].strip():
                    raise ValueError(f"Approved row {row['id']} has an empty question.")
                if not isinstance(row["answer"], str) or not row["answer"].strip():
                    raise ValueError(f"Approved row {row['id']} has an empty answer.")
                if not isinstance(row["valorant_terms"], list) or not all(
                    isinstance(term, str) for term in row["valorant_terms"]
                ):
                    raise ValueError(
                        f"Approved row {row['id']} must have a list of string Valorant terms."
                    )
                approved_row = dict(row)
                approved_row["split"] = effective_split
                rows.append(approved_row)

    return rows


def ensure_no_leakage_or_duplicates(rows: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    questions: dict[str, str] = {}

    for row in rows:
        row_id = row["id"]
        if row_id in ids:
            raise ValueError(f"Duplicate example id: {row_id}")
        ids.add(row_id)

        normalized = normalized_question(row["question"])
        previous_id = questions.get(normalized)
        if previous_id:
            raise ValueError(
                f"Duplicate or near-identical question across approved rows: "
                f"{previous_id} and {row_id}. Keep evaluation questions distinct."
            )
        questions[normalized] = row_id


def sft_row(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["question"].strip()},
        ],
        "completion": [
            {"role": "assistant", "content": row["answer"].strip()},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "data" / "prepared"
    )
    args = parser.parse_args()

    system_prompt_path = args.data_dir / "SYSTEM_PROMPT.txt"
    if not system_prompt_path.is_file():
        raise FileNotFoundError(f"Shared system prompt not found: {system_prompt_path}")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"Shared system prompt is empty: {system_prompt_path}")

    rows = read_approved_rows(args.data_dir, args.output_dir)
    if not rows:
        raise ValueError(
            "No approved rows found. Review candidate examples, set review_status to "
            "'approved', and assign each one to train, validation, or final_holdout."
        )

    ensure_no_leakage_or_duplicates(rows)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in sorted(VALID_SPLITS)
    }
    missing_splits = [split for split, split_rows in by_split.items() if not split_rows]
    if missing_splits:
        raise ValueError(
            "Add approved examples to every split before preparing data. Empty splits: "
            + ", ".join(missing_splits)
        )

    write_jsonl(
        args.output_dir / "train.jsonl",
        [sft_row(row, system_prompt) for row in by_split["train"]],
    )
    write_jsonl(
        args.output_dir / "validation.jsonl",
        [sft_row(row, system_prompt) for row in by_split["validation"]],
    )
    write_jsonl(
        args.output_dir / "final_holdout.jsonl",
        [
            {
                "id": row["id"],
                "question": row["question"].strip(),
                "reference_answer": row["answer"].strip(),
                "category": row["category"],
                "expected_terms": row["valorant_terms"],
            }
            for row in by_split["final_holdout"]
        ],
    )

    print(f"Approved rows prepared: {len(rows)}")
    for split, split_rows in by_split.items():
        category_counts = Counter(row["category"] for row in split_rows)
        print(f"{split}: {len(split_rows)} rows; categories={dict(category_counts)}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
