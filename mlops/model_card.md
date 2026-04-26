# Model Card — AI Career Copilot (RAG)

## Intended use
Personalised, source-cited career advice for employees over the internal job-posting catalogue. Always advisory; never replaces an HR / manager conversation.

## Training data
Synthetic 1,000 postings + 2,000 employee profiles (with eval-only `target_job_id`). Drop-in: any internal ATS / HRIS export with the same shape.

## Model family
- Retriever: TF-IDF over posting text (title + required_skills + description), (1, 2)-grams, sublinear_tf
- Generator: templated text using retrieved chunks + employee profile (no live LLM in the demo)
- Production slot: LangGraph state machine + sentence-transformer / hosted embedding + LLM call with a retrieval-grounding verifier

## Metrics (target)
| Metric | Target |
|---|---|
| Recall@5 | >= 0.55 |
| Grounded-answer rate | >= 90% |
| Latency p95 (CPU) | < 50 ms |

## Limitations
- TF-IDF can't handle synonymy ("AWS" vs "cloud").
- Templated generator doesn't summarise or paraphrase — it renders structured text.
- Production LLM addition needs the verifier to keep grounding intact.

## Ethical considerations
- No PII; surrogate keys.
- Templated answers cannot fabricate jobs.
- Disclaimer surfaced on every response.

## Retraining
- Weekly. Refit retriever when posting churn > 10% / week.
