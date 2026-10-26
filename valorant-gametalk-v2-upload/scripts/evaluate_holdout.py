#!/usr/bin/env python3
"""Generate comparable base, prompted-base, and adapter answers on the holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def args_parser() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=project_dir / "data/prepared/final_holdout.jsonl")
    parser.add_argument("--system-prompt", type=Path, default=project_dir / "data/SYSTEM_PROMPT.txt")
    parser.add_argument("--adapter", required=True, help="Local adapter directory or Hugging Face model ID")
    parser.add_argument("--output", type=Path, default=project_dir / "data/prepared/holdout_outputs.json")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser.parse_args()


def generate(model, tokenizer, question: str, system_prompt: str | None, max_new_tokens: int) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = args_parser()
    if not torch.cuda.is_available():
        raise RuntimeError("Holdout generation requires a CUDA GPU such as Colab T4.")
    holdout_rows = [
        json.loads(line)
        for line in args.holdout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    results = []
    for index, row in enumerate(holdout_rows, start=1):
        with torch.inference_mode(), model.disable_adapter():
            plain = generate(model, tokenizer, row["question"], None, args.max_new_tokens)
            prompted = generate(model, tokenizer, row["question"], system_prompt, args.max_new_tokens)
        with torch.inference_mode():
            finetuned = generate(model, tokenizer, row["question"], system_prompt, args.max_new_tokens)
        results.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "category": row.get("category"),
                "reference_answer": row.get("reference_answer"),
                "expected_terms": row.get("expected_terms", []),
                "base": plain,
                "prompted": prompted,
                "finetuned": finetuned,
            }
        )
        print(f"Generated holdout comparison {index}/{len(holdout_rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(results)} comparisons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
