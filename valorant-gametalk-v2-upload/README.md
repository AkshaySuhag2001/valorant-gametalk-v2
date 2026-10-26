# Valorant Game-Talk v2

An improved QLoRA model and small web demo based on Qwen3-4B. Everyday answers should remain useful and accurate, with a brief Valorant comparison only when it fits. Direct Valorant questions should be answered factually.

## Project layout

- `data/drafts/` — reviewed train/validation examples plus a private final holdout that Git ignores.
- `scripts/prepare_dataset.py` — checks approved rows and creates isolated trainer/evaluation files.
- `scripts/train_qlora.py` — QLoRA training script for a CUDA GPU such as Colab T4.
- `scripts/evaluate_holdout.py` — generates base, prompted-base, and adapter answers on the final holdout.
- `EVALUATION_RUBRIC.md` — human scoring rubric for correctness, style, and clarity.
- `app.py` — Gradio UI with base, prompted-base, and adapter comparison modes.
- `notebooks/train_colab.ipynb` — step-by-step Colab setup and run instructions.

## Quick path

1. Inspect the reviewed JSONL files if you want to change any examples. The current set is marked `approved` and already has separate train, validation, and final-holdout assignments.
2. In Colab, select a T4 GPU, open `notebooks/train_colab.ipynb`, and run its cells in order.
3. Set `ADAPTER_MODEL_ID` to the saved adapter directory or its Hugging Face repository ID, then run `python app.py`.

The current split contains 200 training rows, 30 validation rows, and 40 final holdout rows. The final holdout file is intentionally excluded from Git; keep it in Drive and use it only for final comparison.

## Run the UI

Install `requirements.txt` in an environment with a CUDA GPU. Then set the adapter location and start the app. In PowerShell:

```powershell
$env:ADAPTER_MODEL_ID = "AkshaySuhag/your-new-adapter-repo"
python app.py
```

In Colab/Linux, use `export ADAPTER_MODEL_ID="AkshaySuhag/your-new-adapter-repo"` before `python app.py`. The demo uses 4-bit loading and FP16 compute for T4 compatibility. It never stores prompts or answers.

## Data and model notes

Do not commit Hugging Face tokens, private evaluation answers, checkpoints, or downloaded model weights. In Colab, put the private `final_holdout_review.jsonl` file in `MyDrive/valorant-takehome-v2/` before running the notebook. Use `DATASET_PLAN.md` for the data design and `EVALUATION_RUBRIC.md` for scoring.
