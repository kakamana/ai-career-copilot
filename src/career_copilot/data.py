"""Synthetic generator for the AI Career Copilot.

Writes::

    data/processed/postings.parquet      - 1,000 internal job postings
    data/processed/profiles.parquet      - 2,000 employee profiles (+ eval target_job_id)

Run::

    python -m career_copilot.data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED = DATA_DIR / "processed"

DEPTS = ("Engineering", "Data", "Product", "Design", "Sales", "Marketing", "People", "Operations")
LEVELS = (1, 2, 3, 4, 5)

ROLE_TEMPLATES: dict[str, list[str]] = {
    "Engineering": ["Software Engineer", "Backend Engineer", "Frontend Engineer", "Platform Engineer", "Site Reliability Engineer"],
    "Data":        ["Data Engineer", "Data Scientist", "Analytics Engineer", "ML Engineer", "Data Platform Lead"],
    "Product":     ["Product Manager", "Technical Product Manager", "Product Operations Lead"],
    "Design":      ["UX Designer", "Product Designer", "Design Researcher"],
    "Sales":       ["Account Executive", "Sales Engineer", "Customer Success Manager"],
    "Marketing":   ["Growth Marketer", "Content Strategist", "Lifecycle Marketer"],
    "People":      ["HR Business Partner", "L&D Specialist", "Talent Acquisition Lead"],
    "Operations":  ["Operations Lead", "Programme Manager", "Workforce Planner"],
}

THEMES: dict[str, list[str]] = {
    "Software":    ["Python", "JavaScript", "TypeScript", "Java", "Go", "REST APIs", "GraphQL", "Microservices", "Git", "CI-CD"],
    "Data":        ["Python", "SQL", "Pandas", "Spark", "dbt", "Airflow", "Snowflake", "BigQuery", "ETL Design", "Data Modelling"],
    "ML":          ["Python", "Scikit-learn", "PyTorch", "Deep Learning", "NLP", "MLOps", "Model Evaluation", "Prompt Engineering"],
    "Cloud":       ["AWS", "Azure", "GCP", "Terraform", "Kubernetes", "Docker", "Linux", "Observability"],
    "Product":     ["Roadmapping", "User Research", "A/B Testing", "Stakeholder Management", "Prioritisation", "Agile Delivery"],
    "Design":      ["Figma", "Design Systems", "Prototyping", "User Research", "Accessibility", "Visual Design"],
    "GTM":         ["Pipeline Management", "Salesforce", "Outbound Strategy", "Negotiation", "Customer Success", "Demo Skills"],
    "Marketing":   ["SEO", "Content Strategy", "Lifecycle Email", "HubSpot", "Brand Storytelling", "Analytics"],
    "People":      ["HR Business Partnering", "Learning Design", "Compensation Bands", "Coaching", "Employee Relations"],
    "Operations":  ["Workforce Planning", "Process Design", "Programme Management", "OKR Facilitation", "Vendor Management"],
    "Soft":        ["Communication", "Stakeholder Management", "Mentoring", "Decision Making", "Workshop Facilitation"],
}

DEPT_THEMES: dict[str, list[str]] = {
    "Engineering": ["Software", "Cloud", "Soft"],
    "Data":        ["Data", "ML", "Cloud", "Soft"],
    "Product":     ["Product", "Soft"],
    "Design":      ["Design", "Product", "Soft"],
    "Sales":       ["GTM", "Soft"],
    "Marketing":   ["Marketing", "Soft"],
    "People":      ["People", "Soft"],
    "Operations":  ["Operations", "Soft"],
}

ASPIRATIONS: tuple[str, ...] = (
    "move into a leadership track",
    "become a deeper specialist",
    "switch into a data role",
    "switch into product management",
    "switch into machine learning",
    "switch into a cloud-engineering role",
    "stay in current track but widen scope",
    "become a manager of managers",
)


def _pick_skills(rng: np.random.Generator, themes: list[str], k_per_theme: tuple[int, int] = (2, 5)) -> list[str]:
    out: set[str] = set()
    for t in themes:
        pool = THEMES[t]
        k = int(rng.integers(k_per_theme[0], min(k_per_theme[1] + 1, len(pool) + 1)))
        out.update(rng.choice(pool, size=k, replace=False).tolist())
    return sorted(out)


def make_postings(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        dept = str(rng.choice(DEPTS))
        themes = DEPT_THEMES[dept][:]
        # Sprinkle a cross-functional theme
        if rng.random() < 0.35:
            extra = str(rng.choice([t for t in THEMES if t not in themes]))
            themes.append(extra)
        title = f"{rng.choice(ROLE_TEMPLATES[dept])}"
        level = int(rng.choice(LEVELS, p=[0.20, 0.30, 0.25, 0.15, 0.10]))
        skills = _pick_skills(rng, themes)
        # Description = a templated paragraph that references skills + dept
        desc = (
            f"We're hiring a {title} (L{level}) in our {dept} team. "
            f"You'll work on {rng.choice(['greenfield', 'platform', 'customer-facing', 'internal-tools', 'AI-enabled'])} "
            f"projects alongside a senior {dept.lower()} crew. Required: {', '.join(skills[:8])}. "
            f"Nice to have: {', '.join(rng.choice(skills, size=min(3, len(skills)), replace=False).tolist())}."
        )
        rows.append(dict(
            job_id=f"J-{i:05d}",
            title=title,
            dept=dept,
            level=level,
            required_skills=", ".join(skills),
            description=desc,
        ))
    return pd.DataFrame(rows)


def make_profiles(postings: pd.DataFrame, n: int = 2000, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # Build dept-indexed posting view to derive a plausible target_job_id
    dept_to_postings: dict[str, list[str]] = {
        d: postings.loc[postings["dept"] == d, "job_id"].tolist() for d in postings["dept"].unique()
    }

    for i in range(n):
        cur_dept = str(rng.choice(DEPTS))
        themes = DEPT_THEMES[cur_dept][:]
        cur_role = str(rng.choice(ROLE_TEMPLATES[cur_dept]))
        skills = _pick_skills(rng, themes)
        tenure = float(np.clip(rng.gamma(shape=2.0, scale=2.0), 0.1, 25.0))
        aspiration = str(rng.choice(ASPIRATIONS))

        # Heuristic "true next role": same dept (80%) or aspirational pivot (20%)
        if rng.random() < 0.80 and dept_to_postings[cur_dept]:
            target = str(rng.choice(dept_to_postings[cur_dept]))
        else:
            other = str(rng.choice([d for d in DEPTS if d != cur_dept]))
            pool = dept_to_postings.get(other) or dept_to_postings[cur_dept]
            target = str(rng.choice(pool))

        rows.append(dict(
            emp_id=f"E-{i:05d}",
            current_role=cur_role,
            current_dept=cur_dept,
            skills=", ".join(skills),
            tenure_yrs=round(tenure, 2),
            aspiration=aspiration,
            target_job_id=target,
        ))
    return pd.DataFrame(rows)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    postings = pd.read_parquet(PROCESSED / "postings.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles.parquet")
    return postings, profiles


def make_training_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    postings = make_postings()
    profiles = make_profiles(postings)
    postings.to_parquet(PROCESSED / "postings.parquet", index=False)
    profiles.to_parquet(PROCESSED / "profiles.parquet", index=False)
    return postings, profiles


if __name__ == "__main__":
    postings, profiles = make_training_artifacts()
    print(f"postings: {len(postings):,} rows -> data/processed/postings.parquet")
    print(f"profiles: {len(profiles):,} rows -> data/processed/profiles.parquet")
