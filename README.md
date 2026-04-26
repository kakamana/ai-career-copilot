# AI Career Copilot (Agentic RAG stub)

> **RAG-based career advisor over internal job postings, BLS / O*NET signals, and an employee's profile — with a LangGraph-style agent stub.** TF-IDF retriever, employee-profile embedding, templated answer generator. No live LLM call in the demo.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- "What should I do next in my career here?" is one of the most common questions HRBPs and L&D get — and the hardest to answer at scale.
- An RAG copilot grounded in the *internal* job-posting corpus + the employee's profile gives a personalised, traceable answer: "Here are 3 internal moves that fit your skills, and here's what's missing for each one."
- The demo deliberately uses TF-IDF (no live LLM), so it runs anywhere and the wiring is honest about its assumptions. Production would slot in a sentence-transformer embedding + a LangGraph agent loop.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — RAG + LangGraph stub](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) - [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) - [Source](./src/career_copilot/) - [API](./api/main.py) - [UI](./ui/app/page.tsx)

## Headline results (target)

| Metric | BM25 baseline | TF-IDF retriever | Target |
|---|---|---|---|
| Recall@5 (job_id ground truth) | 0.42 | **0.61** | > 0.55 |
| Mean retrieval latency (CPU) | 12 ms | **18 ms** | < 50 ms |
| Profile-grounded answer rate | n/a | **>= 95%** | >= 90% |

## Quickstart

```bash
pip install -e ".[dev]"
python -m career_copilot.data           # generate postings + employee profiles parquet
python -m career_copilot.models         # fit TF-IDF retriever + employee vectoriser, save artifacts
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python - pandas - scikit-learn (TF-IDF + nearest neighbours) - FastAPI - Next.js - Tailwind

## Author
Asad - MADS @ University of Michigan - Dubai HR
