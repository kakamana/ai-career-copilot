# Data Sources — H9 AI Career Copilot

## Primary (synthetic)
| # | Source | Notes |
|---|---|---|
| 1 | `src/career_copilot/data.py` | Deterministic, seed=42 |

## Real-world drop-ins
| Source | URL | Use |
|---|---|---|
| Internal ATS / HRIS | (private) | Real postings + profiles |
| BLS Employment Projections | https://www.bls.gov/emp/ | Demand outlook signal |
| O*NET 28.1 Skills | https://www.onetcenter.org/database.html | Skill taxonomy alignment |

## Schema for drop-in
- `postings.parquet`: `job_id, title, dept, level, required_skills, description`
- `profiles.parquet`: `emp_id, current_role, skills, tenure_yrs, aspiration` (+ optional `target_job_id` for eval)

## Attribution
External sources retain their respective licences. The repo ships only synthetic data.
