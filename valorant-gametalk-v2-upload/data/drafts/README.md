# Draft examples

The examples in this folder were drafted by the assistant, then reviewed for usefulness, style, duplication, factuality, and split leakage. The accepted rows are marked `review_status: approved`; their `provenance` remains `assistant-draft`. Training rows are in `batch_01_review.jsonl` through `batch_05_review.jsonl`; evaluation-only rows are in `validation_review.jsonl` and `final_holdout_review.jsonl`.

Plain answers are intentional in the `everyday_plain` category. The `intended_split` field is already assigned and the preparation script uses it. Never train on validation or final-holdout answers.

`final_holdout_review.jsonl` is excluded from Git. Keep a private copy in Google Drive and copy it into this folder only for the Colab preparation/evaluation run.
