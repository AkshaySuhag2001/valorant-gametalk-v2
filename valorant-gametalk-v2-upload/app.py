"""Small Gradio demo for the Valorant Game-Talk QLoRA adapter."""

from __future__ import annotations

import os
from contextlib import nullcontext

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-4B")
ADAPTER_MODEL_ID = os.getenv("ADAPTER_MODEL_ID", "").strip()
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "180"))

STYLE_INSTRUCTION = (
    "You are a helpful assistant with a natural Valorant-inspired voice. "
    "Answer accurately and directly, usually in 2–4 concise sentences; use a "
    "short list only when steps help. For everyday questions, add at most one "
    "brief Valorant comparison when it genuinely fits and clarifies the advice. "
    "Keep serious or safety-sensitive guidance plain and direct. For direct "
    "Valorant questions, answer factually and avoid patch-sensitive claims "
    "unless asked. Do not force game terms, repeat catchphrases, use emojis, or "
    "pretend real life has game mechanics."
)


def load_models():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This demo is configured for a CUDA GPU such as a Colab T4. "
            "In Colab, choose Runtime → Change runtime type → T4 GPU, then rerun."
        )

    compute_dtype = torch.float16  # T4 supports FP16; use this rather than BF16.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map="auto",
    )
    base_model.config.use_cache = True

    if ADAPTER_MODEL_ID:
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
        model.eval()
        available_modes = [
            "Base model",
            "Base + style prompt",
            "Fine-tuned adapter",
        ]
    else:
        model = base_model
        model.eval()
        available_modes = ["Base model", "Base + style prompt"]

    return model, tokenizer, available_modes


MODEL, TOKENIZER, MODES = load_models()


@torch.inference_mode()
def answer_question(question: str, mode: str, temperature: float) -> str:
    question = (question or "").strip()
    if not question:
        return "Enter a question first."
    if mode == "Fine-tuned adapter" and not ADAPTER_MODEL_ID:
        return "Set ADAPTER_MODEL_ID to your published adapter or local adapter folder, then restart the app."

    system_message = None if mode == "Base model" else STYLE_INSTRUCTION
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": question})

    # Qwen3 supports enable_thinking=False in its chat template. Render to text
    # first, then tokenize separately so the input length is a regular tensor.
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = TOKENIZER(prompt, return_tensors="pt")
    input_device = MODEL.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    use_adapter = mode == "Fine-tuned adapter"
    adapter_context = nullcontext()
    if ADAPTER_MODEL_ID and not use_adapter:
        adapter_context = MODEL.disable_adapter()

    with adapter_context:
        generated = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


with gr.Blocks(title="Valorant Game-Talk v2") as demo:
    gr.Markdown(
        "# Valorant Game-Talk v2\n"
        "Ask an everyday question or a Valorant question. Compare the base model, "
        "prompted base, and adapter when the adapter is configured."
    )
    if not ADAPTER_MODEL_ID:
        gr.Markdown(
            "**Adapter not configured yet.** The base and prompted-base modes work now; "
            "set `ADAPTER_MODEL_ID` after training to enable the fine-tuned mode."
        )

    mode = gr.Radio(MODES, value=MODES[0], label="Model condition")
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g., How can I plan a study week?",
        lines=2,
    )
    temperature = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Creativity")
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6, interactive=False)
    submit.click(answer_question, [question, mode, temperature], answer)
    question.submit(answer_question, [question, mode, temperature], answer)


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(server_name="0.0.0.0", share=share)
