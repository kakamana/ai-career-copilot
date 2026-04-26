"""Inference: retrieve top-K postings, render templated answer, return sources."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from . import models
from .features import (
    blend_query_with_profile,
    profile_text,
    required_skill_set,
    skill_set,
)

DATA_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


@lru_cache(maxsize=1)
def _load() -> dict:
    art = models.load()
    art["postings"] = pd.read_parquet(DATA_PROC / "postings.parquet").set_index("job_id")
    art["profiles"] = pd.read_parquet(DATA_PROC / "profiles.parquet").set_index("emp_id")
    return art


def _resolve_profile(employee_id: str | None) -> dict:
    if not employee_id:
        return {
            "current_role": "Software Engineer",
            "current_dept": "Engineering",
            "skills": "Python, SQL, REST APIs, Git, Communication",
            "tenure_yrs": 3.0,
            "aspiration": "switch into a data role",
        }
    art = _load()
    if employee_id not in art["profiles"].index:
        raise KeyError(f"unknown employee_id: {employee_id}")
    row = art["profiles"].loc[employee_id]
    return dict(
        current_role=row["current_role"],
        current_dept=row.get("current_dept", ""),
        skills=row["skills"],
        tenure_yrs=float(row["tenure_yrs"]),
        aspiration=row["aspiration"],
    )


def retrieve(query: str, employee_id: str | None, k: int = 5) -> list[dict]:
    art = _load()
    profile = _resolve_profile(employee_id)
    blended = blend_query_with_profile(query, profile)
    qv = art["vec"].transform([blended])
    sims = cosine_similarity(qv, art["X"]).ravel()
    order = np.argsort(-sims)[:k]
    out = []
    for idx in order:
        jid = art["job_ids"][int(idx)]
        post = art["postings"].loc[jid]
        snippet = str(post["description"])[:240] + ("..." if len(str(post["description"])) > 240 else "")
        out.append(dict(
            job_id=jid,
            title=str(post["title"]),
            dept=str(post["dept"]),
            level=int(post["level"]),
            score=float(sims[int(idx)]),
            snippet=snippet,
            required_skills=str(post["required_skills"]),
        ))
    return out


def _gap(profile: dict, posting: dict) -> tuple[list[str], list[str]]:
    have = skill_set(profile.get("skills", ""))
    need = required_skill_set(posting.get("required_skills", ""))
    overlap = sorted(have & need)
    missing = sorted(need - have)
    return overlap, missing


def render_answer(query: str, profile: dict, top: list[dict]) -> str:
    """Templated answer — every line grounded in retrieved chunks + profile."""
    if not top:
        return "I couldn't find a strong internal match for that query yet."
    lines = []
    lines.append(
        f"Based on your profile (current role: {profile['current_role']}, "
        f"top skills: {', '.join(list(skill_set(profile['skills']))[:3]) or '—'}, "
        f"aspiration: {profile['aspiration']}), here are {len(top)} internal opportunities that look like a fit:"
    )
    for rank, post in enumerate(top, 1):
        overlap, missing = _gap(profile, post)
        lines.append(
            f"{rank}. {post['title']} ({post['dept']}, L{post['level']}, {post['job_id']}, "
            f"score {post['score']:.2f}) — overlap on {', '.join(overlap[:5]) or '(none)'}; "
            f"missing {', '.join(missing[:5]) or '(none)'}."
        )
    return "\n".join(lines)


def suggested_next_actions(profile: dict, top: list[dict]) -> list[str]:
    actions: list[str] = []
    if not top:
        return ["Talk to your HRBP to refine your aspiration tags."]
    # Aggregate the most-frequent missing skills across the top-K
    missing_counter: dict[str, int] = {}
    for post in top:
        _, missing = _gap(profile, post)
        for s in missing[:5]:
            missing_counter[s] = missing_counter.get(s, 0) + 1
    top_missing = sorted(missing_counter.items(), key=lambda kv: -kv[1])[:3]
    for skill, _ in top_missing:
        actions.append(f"Build skill: {skill} (try the H7 micro-cert recommender for matching certs)")
    actions.append(f"Reach out to the hiring manager for {top[0]['job_id']} for an informational chat")
    actions.append("Update your aspiration tags in the HRIS so the copilot can refine its suggestions")
    return actions


def chat(query: str, employee_id: str | None, k: int = 5) -> dict:
    profile = _resolve_profile(employee_id)
    top = retrieve(query, employee_id, k=k)
    return dict(
        answer=render_answer(query, profile, top),
        sources=[dict(job_id=t["job_id"], score=t["score"], snippet=t["snippet"]) for t in top],
        suggested_next_actions=suggested_next_actions(profile, top),
        profile_used=profile_text(profile),
    )
