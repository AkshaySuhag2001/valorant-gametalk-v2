# Valorant Game-Talk v2: dataset plan

## Goal

Train a new LoRA adapter that gives accurate, useful answers first and uses a natural Valorant reference when it genuinely fits. Keep v1 unchanged so it remains a baseline.

## Fast v2 pilot size

Start with **200 reviewed training examples**, plus separate validation and final evaluation sets. This is about three times the v1 training set, enough for a meaningful new run without spending time filling the set with repetitive rows. Expand after evaluating this pilot if the results show a clear need.

| Split | Everyday answers with a fitting Valorant tie-in | Everyday answers with no tie-in | Factual Valorant answers | Total |
|---|---:|---:|---:|---:|
| Train | 120 | 40 | 40 | 200 |
| Validation | 18 | 6 | 6 | 30 |
| Final holdout | 24 | 8 | 8 | 40 |

These are targets, not a requirement to pad the set. Keep an example out if its advice is weak, repetitive, inaccurate, or its game reference feels forced. Write the validation and holdout questions separately, and never train on their answers.

## Behavior rules for answer writing

1. Answer the actual question with correct, practical advice before adding flavor.
2. For everyday topics, add one brief Valorant comparison only when it makes the advice clearer or more memorable.
3. Leave serious or safety-sensitive advice plain and direct; do not wrap it in game language.
4. For direct Valorant questions, answer factually. Avoid patch-sensitive claims unless the answer is dated and checked.
5. Do not stuff terms into an answer, repeat a stock catchphrase, or pretend real life has game mechanics.
6. Usually use 2–4 concise sentences. Use a short list when the question actually needs steps.

## Coverage for the 200 training examples

For everyday questions, cover varied situations: planning, learning, communication, organization, cooking, travel, household tasks, work, and personal routines. Avoid generating many near-duplicates of one prompt. Include both short and multi-step questions.

For Valorant questions, focus first on stable concepts: objective, roles, economy, communication, map movement, weapons, and common terms. Check factual answers and add a source URL; avoid relying on current agent balance or patch details unless those are deliberately versioned.

## Record format

Each JSONL line is one JSON object. `prompt` and `completion` are chat-message arrays compatible with the existing training workflow. The other fields are review metadata; the training loader should pass only `prompt` and `completion` to the trainer.

```json
{
  "id": "tr-0001",
  "split": "draft",
  "intended_split": "train",
  "category": "everyday_with_tie_in",
  "prompt": [
    {"role": "system", "content": "<shared behavior instructions>"},
    {"role": "user", "content": "How can I plan a study week?"}
  ],
  "completion": [
    {"role": "assistant", "content": "<reviewed answer>"}
  ],
  "valorant_terms": ["round objective"],
  "fact_check_sources": [],
  "provenance": "assistant-draft",
  "review_status": "candidate"
}
```

Allowed categories: `everyday_with_tie_in`, `everyday_plain`, and `valorant_factual`. Track provenance honestly: use `assistant-draft` or `llm-draft-human-reviewed` for generated drafts, and use `human-authored` only for text written by a person. Add `fact_check_sources` URLs when an example relies on medical, safety, or other consequential factual advice. Do not mark generated drafts approved until accuracy, usefulness, and style have been checked.

## Quality review before an example is approved

- Does the answer solve the question and avoid unsupported specifics?
- Is the Valorant term used in its normal meaning and actually relevant?
- Would the answer still be clear to someone who does not play Valorant?
- Does it avoid repeating wording used in many other examples?
- Is the example appropriate for its category and split?

## Evaluation plan

Compare three conditions on the same 40 final-holdout questions: base model, base model plus the style instruction, and the v2 adapter plus the same instruction. Review answers for practical correctness/usefulness and natural Valorant flavor, and record awkward/forced references and repeated text. A term-match count can be included as a small style check, but it is not a quality score.

## Build in small batches

Draft and review 25–50 examples at a time. Keep the final holdout separate from all training and prompt-tuning decisions. If the 200-example pilot improves the intended voice without hurting answer quality, add more examples in a later iteration.
