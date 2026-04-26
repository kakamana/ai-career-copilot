# Evaluation Plan — AI Career Copilot

## 1. Held-out evaluation set
Per employee, we have a synthetic "true next role" `target_job_id` derived from their aspiration + dept; held out from training.

## 2. Primary scorecard
| Model | Recall@5 | Recall@10 | Latency p95 | Grounded rate |
|---|---|---|---|---|
| BM25 (rank-bm25) | – | – | – | – |
| TF-IDF cosine | – | – | – | – |
| TF-IDF + profile blend | – | – | – | – |

## 3. Grounded-answer audit
- Manually inspect 30 responses. Every cited `job_id` must exist in the corpus.
- No phantom titles or skills; templated generator forbids them by construction.

## 4. Latency
- p50 / p95 retrieval latency on 100 random queries (CPU).

## 5. Action diversity
- Avg # of distinct `suggested_next_actions` per response.

## 6. Robustness
- Drop the profile blend → measure Recall@5 decay (confirms personalisation contributes).
- Inject 100 noisy postings → measure top-K stability.

## 7. Deployment readiness checklist
- [ ] Recall@5 >= 0.55 on holdout
- [ ] Grounded rate >= 90%
- [ ] Latency p95 < 50 ms
- [ ] /chat returns answer + sources + suggested_next_actions
- [ ] UI sources panel renders correctly
- [ ] LangGraph slot documented in methodology
