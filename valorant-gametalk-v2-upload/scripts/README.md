# Data preparation

`prepare_dataset.py` turns reviewed JSONL rows into model-ready files. It includes only rows with `review_status: "approved"`, follows each row's `intended_split`, checks for duplicate questions, and writes the final holdout separately from training and validation.

From the project root, run:

```bash
python scripts/prepare_dataset.py
```

Outputs go to `data/prepared/`. The current examples are approved. Keep `data/drafts/final_holdout_review.jsonl` private; it is ignored by Git. Before running the Colab notebook, upload that file to your Drive project folder so the notebook can copy it into the cloned repository for evaluation.
