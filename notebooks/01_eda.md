# Notebook 01 — EDA

>>> `from career_copilot.data import load_all; postings, profiles = load_all()`

## 1. Corpus shape
- # postings, # profiles, distribution by dept and level.

## 2. Skill density
- Skills per posting; skills per profile.

## 3. Aspiration distribution
- Most common aspirations; cross-tab with current_role.

## 4. Hypotheses
1. Profile blending lifts Recall@5 by 5-10 pts vs query-only retrieval.
2. The skill-gap step adds high-value next actions even when retrieval is wrong.
