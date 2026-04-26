# Feasibility Study — AI Career Copilot

## 1. Data feasibility
- **Synthetic generator:** 1,000 internal postings (`job_id, title, dept, level, required_skills, description`) + 2,000 employee profiles (`emp_id, current_role, skills, tenure_yrs, aspiration`).
- **External signal drop-in:** BLS Employment Projections + O*NET Skills are open and load into the same skill vocabulary.

## 2. Technical feasibility
- **Retriever:** TF-IDF on posting text (title + required_skills + description) (main); BM25 (alt); sentence-transformer embedding (stretch, GPU-bound).
- **Answer generator (demo):** templated text using the top-K retrieved chunks. No live LLM.
- **Agent stub:** LangGraph-style state machine; described in `docs/03_methodology.md` but not invoked at runtime.

## 3. Economic feasibility
| Line item | Monthly cost |
|---|---|
| 1× small container | ~$8 |
| Storage | ~$1 |
| **Total** | **~$9 / mo** |

**LLM cost note:** the production version with LangGraph + Claude / GPT-4o-mini would be ~$20-50 / mo at moderate query volume. The demo avoids this entirely.

## 4. Operational feasibility
- Refit retriever weekly. Stateless inference: load TF-IDF artefacts, run cosine, render template.
- LangGraph agent loop is the production-time slot; out of scope for the demo.

## 5. Ethical / legal feasibility
- No PII; surrogate keys.
- Templated answers cannot fabricate jobs — only render retrieved chunks.
- Disclaimer surfaced on every response.

## 6. Recommendation
**Go.** Cheap, deterministic, dependency-light. The honest "no live LLM in the demo" framing is itself a portfolio differentiator: most RAG demos quietly need an API key.
