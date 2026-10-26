#!/usr/bin/env python3
"""Fine-tune Qwen3-4B with QLoRA from prepared chat-format JSONL files."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


class ChatJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows: list[dict[str, list[int]]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt_messages = item.get("prompt")
                completion_messages = item.get("completion")
                if not isinstance(prompt_messages, list) or not isinstance(
                    completion_messages, list
                ):
                    raise ValueError(f"Expected prompt/completion message arrays at {path}:{line_number}")

                prompt_ids = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                full_ids = tokenizer.apply_chat_template(
                    prompt_messages + completion_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise ValueError(
                        f"Chat template prefix mismatch at {path}:{line_number}; "
                        "cannot safely mask the prompt labels."
                    )

                answer_ids = full_ids[len(prompt_ids) :]
                if not answer_ids:
                    raise ValueError(f"Empty completion after tokenization at {path}:{line_number}")
                if len(answer_ids) >= max_length:
                    raise ValueError(
                        f"Completion exceeds max_length={max_length} at {path}:{line_number}"
                    )

                prompt_budget = max_length - len(answer_ids)
                prompt_ids = prompt_ids[-prompt_budget:]
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
                self.rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": labels,
                    }
                )
        if not self.rows:
            raise ValueError(f"No rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU. In Colab, select a T4 GPU.")
    args.output.mkdir(parents=True, exist_ok=True)

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
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatJsonlDataset(args.train, tokenizer, args.max_length)
    validation_dataset = ChatJsonlDataset(args.validation, tokenizer, args.max_length)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": args.learning_rate,
        "warmup_steps": 5,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": 42,
    }
    eval_arg = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    if eval_arg in parameters:
        training_kwargs[eval_arg] = "epoch"
    if "gradient_checkpointing_kwargs" in parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    result = trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    metrics = dict(result.metrics)
    metrics.update(
        {
            "base_model": args.base_model,
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "epochs": args.epochs,
        }
    )
    (args.output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Adapter and tokenizer saved to {args.output.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
