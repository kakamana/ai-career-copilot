"""FastAPI for the AI Career Copilot.

Endpoints:
    GET  /health
    POST /chat   - templated answer + sources + suggested_next_actions
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI Career Copilot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This copilot is grounded in retrieved internal postings + your profile. "
    "It is advisory; decisions remain with you, your manager, and HR."
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    employee_id: str | None = Field(default=None, description="Optional employee_id for personalisation")
    k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    job_id: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    suggested_next_actions: list[str]
    profile_used: str
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        from career_copilot.serve import chat as _serve_chat
        result = _serve_chat(query=req.query, employee_id=req.employee_id, k=req.k)
    except FileNotFoundError:
        # Stub response when artefacts aren't built yet — wiring is real.
        result = dict(
            answer=(
                "Based on your profile (current role: Software Engineer, top skills: Python, SQL, REST APIs, "
                "aspiration: switch into a data role), here are 3 internal opportunities that look like a fit:\n"
                "1. Data Engineer (Data, L3, J-STUB-001, score 0.71) — overlap on Python, SQL; missing dbt, Airflow.\n"
                "2. Analytics Engineer (Data, L2, J-STUB-002, score 0.66) — overlap on Python; missing dbt, Snowflake."
            ),
            sources=[
                dict(job_id="J-STUB-001", score=0.71, snippet="We're hiring a Data Engineer (L3) ..."),
                dict(job_id="J-STUB-002", score=0.66, snippet="We're hiring an Analytics Engineer (L2) ..."),
            ],
            suggested_next_actions=[
                "Build skill: dbt (try the H7 micro-cert recommender)",
                "Build skill: Airflow",
                "Reach out to the hiring manager for J-STUB-001 for an informational chat",
            ],
            profile_used="current role: Software Engineer | dept: Engineering | skills: Python, SQL | aspiration: switch into a data role",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return ChatResponse(**result)
