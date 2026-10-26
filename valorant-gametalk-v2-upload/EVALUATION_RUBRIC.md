# Holdout evaluation rubric

`scripts/evaluate_holdout.py` produces base, prompted-base, and adapter answers for each of the 40 final-holdout questions. It uses the same decoding settings for all three conditions and does not send the reference answer to the model.

Score each answer before comparing the three conditions:

| Dimension | Score | How to judge |
|---|---:|---|
| Correctness and usefulness | 0–2 | 0 = wrong or unhelpful; 1 = partly useful or missing a key point; 2 = accurate and actionable. |
| Valorant voice | 0–2 | For tie-in questions: 0 = absent or forced; 1 = present but awkward; 2 = one natural, relevant reference. For plain and factual questions, 2 means it correctly avoids forced flavor. |
| Clarity | 0–2 | 0 = confusing or repetitive; 1 = understandable with issues; 2 = concise and easy to follow. |

Record any unsafe advice, fabricated game mechanics, repeated phrases, or broken output separately. Report average scores by category as well as totals; a term-match count alone is not a quality measure. Keep the holdout untouched if you change training examples or prompting after seeing the results. Make those changes first, then reserve a new holdout for a fresh final comparison.
