from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_stub_or_real():
    r = client.post(
        "/chat",
        json={
            "query": "I'm a Software Engineer keen to move into a data role.",
            "employee_id": None,
            "k": 5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and len(body["answer"]) > 10
    assert isinstance(body["sources"], list) and len(body["sources"]) >= 1
    for s in body["sources"]:
        assert "job_id" in s and "score" in s and "snippet" in s
    assert isinstance(body["suggested_next_actions"], list)
    assert "disclaimer" in body


def test_data_shape_deterministic():
    from career_copilot.data import make_postings, make_profiles

    p = make_postings(n=200, seed=42)
    pr = make_profiles(p, n=400, seed=43)

    assert {"job_id", "title", "dept", "level", "required_skills", "description"} <= set(p.columns)
    assert {"emp_id", "current_role", "skills", "tenure_yrs", "aspiration", "target_job_id"} <= set(pr.columns)
    assert len(p) == 200
    assert len(pr) == 400

    # determinism
    p2 = make_postings(n=200, seed=42)
    assert (p == p2).all().all()
