import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.admin import jobs as jobs_mod
from app.admin.security import hash_password
from app.db.models import AdminUser, AuditLog, NegativeFeedback, Source, UnansweredQuestion, VerifiedAnswer
from app.db.session import session_scope
from app.main import app

PASSWORD = "Correct-horse-42"


@pytest.fixture(scope="module", autouse=True)
def users():
    with session_scope() as db:
        for name, role in (("boss", "super"), ("editor1", "editor"), ("locky", "editor")):
            if not db.scalar(select(AdminUser).where(AdminUser.username == name)):
                db.add(AdminUser(username=name, role=role, password_hash=hash_password(PASSWORD)))


def client() -> TestClient:
    return TestClient(app, follow_redirects=False)  # no `with`: lifespan (model warm-up, scheduler) not started


def login(c: TestClient, username="boss", password=PASSWORD):
    return c.post("/admin/login", data={"username": username, "password": password, "next": "/admin"})


def csrf(c: TestClient, page="/admin/upload") -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', c.get(page).text).group(1)


# ---------------------------------------------------------------- authentication
def test_pages_require_login():
    r = client().get("/admin/sources")
    assert r.status_code == 303 and r.headers["location"].startswith("/admin/login")


def test_login_sets_secure_cookie_and_dashboard_loads():
    c = client()
    r = login(c)
    assert r.status_code == 303 and r.headers["location"] == "/admin"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/admin" in cookie
    page = c.get("/admin")
    assert page.status_code == 200 and "Dashboard" in page.text


def test_wrong_password_is_generic_and_locks_after_five_tries():
    c = client()
    r = login(c, "locky", "nope")
    assert "Wrong username or password." in r.text
    assert "Wrong username or password." in login(c, "nobody-here", "nope").text  # same message: no user probing
    for _ in range(4):
        login(c, "locky", "nope")
    r = login(c, "locky", PASSWORD)  # even the right password is refused while locked
    assert "Too many failed attempts" in r.text


def test_security_headers_on_admin_pages():
    r = client().get("/admin/login")
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"


def test_state_changes_need_csrf_token():
    c = client()
    login(c)
    assert c.post("/admin/crawl/start", data={}).status_code == 403
    assert c.post("/admin/crawl/start", data={"csrf_token": "forged"}).status_code == 403


def test_editor_cannot_manage_users():
    c = client()
    login(c, "editor1")
    assert c.get("/admin/users").status_code == 403
    assert c.get("/admin/sources").status_code == 200


def test_logout_ends_session():
    c = client()
    login(c)
    token = csrf(c, "/admin")
    c.post("/admin/logout", data={"csrf_token": token})
    assert c.get("/admin").status_code == 303


# ---------------------------------------------------------------- uploads
def test_upload_rejects_disguised_files():
    assert jobs_mod.validate_upload("notice.pdf", b"MZ\x90\x00 this is an exe")[0]
    assert jobs_mod.validate_upload("virus.exe", b"MZ")[0]
    assert jobs_mod.validate_upload("notes.txt", b"%PDF-1.7 fake")[0]
    assert jobs_mod.validate_upload("notes.txt", b"Library timings: 9 to 5")[0] is None
    assert jobs_mod.safe_filename("../../etc/pass wd?.PDF") == "pass wd.pdf"  # no path, no odd characters


def test_upload_is_stored_outside_web_root_and_indexed(monkeypatch):
    monkeypatch.setattr(jobs_mod.indexer, "index_sources", lambda srcs: {s.id: 3 for s in srcs})
    monkeypatch.setattr(jobs_mod.jobs, "start_upload", lambda sid: jobs_mod.process_upload(sid))
    c = client()
    login(c)
    r = c.post("/admin/upload", data={"csrf_token": csrf(c), "title": "Library timings 2026"},
               files={"file": ("timings.txt", b"The central library is open 8:30 am to 6 pm, Monday to Saturday.",
                               "text/plain")})
    assert r.status_code == 303
    sid = int(r.headers["location"].split("/")[3].split("?")[0])
    with session_scope() as db:
        src = db.get(Source, sid)
        assert src.source_type == "upload" and src.index_status == "indexed" and src.chunk_count == 3
        assert "8:30 am" in src.text and src.title == "Library timings 2026"
        path = jobs_mod.upload_path(src)
    assert path.exists() and "static" not in str(path)
    assert "static" not in str(path.parent)
    with session_scope() as db:
        assert db.scalar(select(AuditLog).where(AuditLog.action == "upload", AuditLog.username == "boss"))


# ---------------------------------------------------------------- crawling safety
@pytest.mark.parametrize("url", ["http://localhost:8000/admin", "https://sxca.edu.in.evil.example/", "http://169.254.169.254/",
                                 "https://lms.sxca.edu.in/login", "file:///etc/passwd", "not a url"])
def test_add_url_refuses_non_college_and_internal_addresses(url):
    assert jobs_mod.jobs.start_url(url, "boss") is not None


# ---------------------------------------------------------------- official answers
def test_official_answer_save_removes_unanswered_entry():
    with session_scope() as db:
        q = UnansweredQuestion(question="What are the library timings on Sunday?", reason="no relevant source found")
        db.add(q)
        db.flush()
        qid = q.id
    c = client()
    login(c)
    r = c.post("/admin/answers/save", data={
        "csrf_token": csrf(c), "question": "What are the library timings on Sunday?",
        "alt_questions": "Is the library open on Sunday?\nlibrary sunday timings",
        "answer": "The central library is closed on Sundays.", "active": "1", "from_unanswered": str(qid)})
    assert r.status_code == 303
    with session_scope() as db:
        row = db.scalar(select(VerifiedAnswer).where(VerifiedAnswer.question.like("%Sunday%")))
        assert row and row.active and row.alt_questions.count("\n") == 1 and row.updated_by == "boss"
        assert db.get(UnansweredQuestion, qid) is None


async def test_close_official_answer_is_used_word_for_word(monkeypatch):
    from app.rag import graph as graph_mod
    from app.rag.verified import VerifiedMatch

    monkeypatch.setattr(graph_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(graph_mod.verified_index, "best",
                        lambda e: VerifiedMatch(1, "Library on Sunday?", "The library is closed on Sundays.", None, 0.93))
    monkeypatch.setattr(graph_mod, "record_use", lambda i: None)

    def no_llm():
        raise AssertionError("official answers must not call the LLM")
    monkeypatch.setattr(graph_mod, "get_llm", no_llm)
    text, final = "", {}
    async for mode, chunk in graph_mod.build_graph().astream({"question": "Is the library open on Sunday?", "history": []},
                                                              stream_mode=["custom", "values"]):
        if mode == "custom" and chunk["type"] == "token":
            text += chunk["text"]
        elif mode == "values":
            final = chunk
    assert text == "The library is closed on Sundays." and final["answered"] and final["verified_direct"]


# ---------------------------------------------------------------- feedback API
def test_thumbs_down_keeps_only_question_and_sources():
    c = client()
    r = c.post("/api/feedback", json={"rating": "down", "question": "What is the BCA fee?",
                                      "sources": ["https://sxca.edu.in/admissions/tuition-fees/"]})
    assert r.status_code == 204
    with session_scope() as db:
        row = db.scalar(select(NegativeFeedback).order_by(NegativeFeedback.id.desc()))
        assert row.question == "What is the BCA fee?" and "tuition-fees" in row.sources
        assert {c.name for c in NegativeFeedback.__table__.columns} == {"id", "question", "sources", "created_at"}
    assert c.post("/api/feedback", json={"rating": "meh"}).status_code == 422
