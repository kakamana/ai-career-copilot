"""Text shaping helpers for postings + employee profiles."""
from __future__ import annotations

import pandas as pd


def posting_text(row: pd.Series) -> str:
    """Concatenate the fields a retriever should index for one posting."""
    return " | ".join([
        str(row.get("title", "")),
        str(row.get("dept", "")),
        f"L{row.get('level', '')}",
        f"required: {row.get('required_skills', '')}",
        str(row.get("description", "")),
    ])


def profile_text(profile: dict) -> str:
    """Render a profile in the same vocabulary the postings index uses."""
    return " | ".join([
        f"current role: {profile.get('current_role', '')}",
        f"dept: {profile.get('current_dept', '')}",
        f"skills: {profile.get('skills', '')}",
        f"aspiration: {profile.get('aspiration', '')}",
    ])


def blend_query_with_profile(query: str, profile: dict, weight: float = 1.0) -> str:
    """Glue the user's question to the profile text so retrieval is personalised.

    The `weight` arg is reserved for future use (e.g. repeating the query string
    to up-weight it lexically vs the profile prose).
    """
    base = profile_text(profile)
    if query:
        return f"{query} || {base}"
    return base


def skill_set(profile_skills_text: str) -> set[str]:
    return {s.strip() for s in str(profile_skills_text).split(",") if s.strip()}


def required_skill_set(posting_required_text: str) -> set[str]:
    return {s.strip() for s in str(posting_required_text).split(",") if s.strip()}
