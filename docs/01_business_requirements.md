# Business Requirements — AI Career Copilot

## 1. Problem Statement
Employees ask "what should I do next here?" and get inconsistent answers — depending on which manager they ask, which HRBP picks up the case, or whether they happen to see an internal posting. Internal mobility loses to external offers because we don't surface the right opportunities at the right moment. Leadership wants a **traceable** career copilot grounded in the internal job-postings catalogue, the employee's profile, and external skill signals (BLS / O*NET) — not a generic chatbot.

## 2. Stakeholders
| Role | Interest | Success criterion |
|---|---|---|
| Employee | Honest answer + suggested next step | Answer cites at least one internal posting |
| HRBP | Pre-screen mobility conversations | Top-K postings match the employee's skill profile |
| Talent Acquisition | Internal-first sourcing | Internal-fill rate moves up |
| Audit / Ethics | No hallucinated jobs | Every answer is grounded in retrieved sources |

## 3. Business Objectives
1. Every answer cites **>= 1 retrieved internal job_id** with a similarity score.
2. **Recall@5** on a held-out (employee, target_job) set >= 0.55.
3. Suggest **>= 2 next actions** per response (skill to build, course to take, manager to talk to).

## 4. KPIs
| KPI | Definition | Target |
|---|---|---|
| Recall@5 | Ground-truth job_id appears in top-5 retrieved | >= 0.55 |
| Grounded-answer rate | % responses citing >= 1 source | >= 90% |
| Mean retrieval latency (CPU) | TF-IDF nearest neighbours | < 50 ms |

## 5. Scope
**In scope:** 1,000 internal postings + 2,000 employee profiles + small skill graph; TF-IDF retriever; templated answer generator.
**Out of scope:** live LLM generation; live recruiter chat; ATS write-back.

## 6. Constraints / assumptions
- **No live LLM in the demo** — keeps the repo runnable anywhere; all answers are templated using retrieved chunks. The methodology section names the LangGraph + LLM slot.
- **PII-safe:** profiles are surrogate-keyed; no real names, no real titles.

## 7. Risks
| Risk | Mitigation |
|---|---|
| Hallucinated jobs | Templated answers only render from retrieved sources — no free generation |
| Skill-name drift | Use the H6 controlled vocabulary |
| Over-reliance on the copilot | UI banner: advisory only |
