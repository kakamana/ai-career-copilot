# Data Card — H9 AI Career Copilot

## Dataset composition

| Layer | Source | Shape | Purpose |
|---|---|---|---|
| Synthetic postings | `src/career_copilot/data.py` | 1,000 × {title, dept, level, required_skills, description} | RAG corpus |
| Synthetic profiles | `src/career_copilot/data.py` | 2,000 × {current_role, skills, tenure_yrs, aspiration} | Personalisation |
| Skill graph (small) | reused vocabulary | 40 skills | Skill-gap diff |

## Files
- `data/processed/postings.parquet`
- `data/processed/profiles.parquet`

## Synthetic ground truth
Each profile has a `target_job_id` it would naturally aspire to (function of current role + aspiration + dept). Eval-only column.

## Known biases
- Synthetic generator induces a clear theme→department mapping so retrieval has signal to recover.
- Skill names follow the same controlled vocabulary as H6 / H7.

## PII
None.

## Reproducing
```bash
python -m career_copilot.data
```
Deterministic seed = 42.
