"""Admin panel pages (server-rendered, served under /admin)."""
from __future__ import annotations

import base64
import io
from datetime import date, datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, func, or_, select

from app.admin import security
from app.admin.jobs import jobs, remove_source, save_upload, unblock_source, validate_upload
from app.admin.security import COOKIE_NAME, Admin, audit
from app.config import BASE_DIR, get_settings
from app.crawler.documents import ocr_available
from app.db.models import (AdminUser, AuditLog, CrawlError, CrawlRun, NegativeFeedback, Source, UnansweredQuestion,
                           UsageDay, VerifiedAnswer, as_utc, utcnow)
from app.db.session import session_scope
from app.llm.factory import get_llm
from app.rag import vectorstore

router = APIRouter(prefix="/admin", include_in_schema=False)
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "admin" / "templates"))
PAGE_SIZE = 50
_IST = ZoneInfo("Asia/Kolkata")


def _fmt_dt(value, fmt: str = "%d %b %Y, %H:%M") -> str:
    """UTC datetimes from the database shown in Indian time; dates as-is."""
    if value is None or value == "":
        return "—"
    if isinstance(value, datetime):
        return as_utc(value).astimezone(_IST).strftime(fmt)
    if isinstance(value, date):
        return value.strftime("%d %b %Y")
    return str(value)


templates.env.filters["dt"] = _fmt_dt
templates.env.filters["d"] = lambda v: _fmt_dt(v, "%d %b %Y")


class LoginRequired(Exception):
    def __init__(self, next_url: str = "/admin"):
        self.next_url = next_url


# ---------------------------------------------------------------- helpers
def current_admin(request: Request) -> Admin:
    admin = security.load_session(request.cookies.get(COOKIE_NAME))
    if admin is None:
        raise LoginRequired(request.url.path)
    return admin


def super_admin(admin: Admin = Depends(current_admin)) -> Admin:
    if not admin.is_super:
        raise HTTPException(403, "Only the super admin can do this.")
    return admin


def check_csrf(admin: Admin, token: str) -> None:
    if not security.csrf_ok(admin, token):
        raise HTTPException(403, "The form expired. Please reload the page and try again.")


def render(request: Request, template: str, admin: Admin | None, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, template, {
        "admin": admin, "settings": get_settings(), "ok": request.query_params.get("ok"),
        "err": request.query_params.get("err"), **ctx})


def back(url: str, ok: str | None = None, err: str | None = None) -> RedirectResponse:
    sep = "&" if "?" in url else "?"
    if ok:
        url += f"{sep}ok={quote(ok)}"
    elif err:
        url += f"{sep}err={quote(err)}"
    return RedirectResponse(url, status_code=303)


def set_session_cookie(resp, token: str) -> None:
    s = get_settings()
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict", secure=s.is_production, path="/admin",
                    max_age=s.admin_session_max_hours * 3600)


def safe_next(url: str | None) -> str:
    return url if url and url.startswith("/admin") and not url.startswith("//") else "/admin"


# ---------------------------------------------------------------- login / logout / 2FA
@router.get("/login")
def login_page(request: Request, next: str = "/admin"):
    return render(request, "login.html", None, next=safe_next(next))


@router.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/admin")):
    key = security.client_key(request)
    if security.ip_blocked(key):
        return render(request, "login.html", None, next=safe_next(next),
                      error="Too many login attempts from this network. Please wait 15 minutes.")
    result = security.authenticate(username, password)
    if result.error:
        security.record_ip_failure(key)
        return render(request, "login.html", None, next=safe_next(next), error=result.error, username=username)
    token = security.create_session(result.user_id, pending_2fa=result.needs_2fa)
    resp = RedirectResponse("/admin/2fa?next=" + quote(safe_next(next)) if result.needs_2fa else safe_next(next),
                            status_code=303)
    set_session_cookie(resp, token)
    if not result.needs_2fa:
        audit(username.strip().lower(), "login")
    return resp


@router.get("/2fa")
def twofa_page(request: Request, next: str = "/admin"):
    if security.load_session(request.cookies.get(COOKIE_NAME), allow_pending_2fa=True) is None:
        return RedirectResponse("/admin/login", status_code=303)
    return render(request, "twofa.html", None, next=safe_next(next))


@router.post("/2fa")
def twofa(request: Request, code: str = Form(...), next: str = Form("/admin")):
    token = request.cookies.get(COOKIE_NAME)
    pending = security.load_session(token, allow_pending_2fa=True)
    if pending is None:
        return RedirectResponse("/admin/login", status_code=303)
    key = security.client_key(request)
    if security.ip_blocked(key) or not security.verify_totp(pending.user_id, code):
        security.record_ip_failure(key)
        return render(request, "twofa.html", None, next=safe_next(next), error="That code is not valid.")
    security.complete_2fa(token)
    audit(pending.username, "login", details="with 2FA")
    return RedirectResponse(safe_next(next), status_code=303)


@router.post("/logout")
def logout(request: Request, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    security.destroy_session(request.cookies.get(COOKIE_NAME))
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/admin")
    return resp


# ---------------------------------------------------------------- dashboard
@router.get("")
def dashboard(request: Request, admin: Admin = Depends(current_admin)):
    s = get_settings()
    since = date.today() - timedelta(days=29)
    with session_scope() as db:
        by_type = db.execute(select(Source.source_type, Source.content_type, func.count(), func.sum(Source.chunk_count))
                             .where(Source.index_status == "indexed")
                             .group_by(Source.source_type, Source.content_type)).all()
        pending = db.scalar(select(func.count()).where(Source.index_status.in_(("pending", "processing")))) or 0
        days = db.scalars(select(UsageDay).where(UsageDay.day >= since).order_by(UsageDay.day.desc())).all()
        last_run = db.scalar(select(CrawlRun).where(CrawlRun.trigger != "cli").order_by(CrawlRun.id.desc()).limit(1)) \
            or db.scalar(select(CrawlRun).order_by(CrawlRun.id.desc()).limit(1))
        counts = {
            "unanswered": db.scalar(select(func.count(UnansweredQuestion.id))) or 0,
            "feedback": db.scalar(select(func.count(NegativeFeedback.id))) or 0,
            "verified": db.scalar(select(func.count(VerifiedAnswer.id)).where(VerifiedAnswer.active.is_(True))) or 0,
        }
        days = [{c: getattr(d, c) for c in ("day", "chats", "unanswered", "verified_hits", "thumbs_up", "thumbs_down",
                                            "llm_calls", "input_tokens", "output_tokens", "cost_usd")} for d in days]
        last_run = last_run and {"id": last_run.id, "status": last_run.status, "started_at": last_run.started_at,
                                 "finished_at": last_run.finished_at, "trigger": last_run.trigger}
    tot = {k: sum(d[k] for d in days) for k in ("chats", "unanswered", "verified_hits", "thumbs_up", "thumbs_down",
                                                 "input_tokens", "output_tokens", "cost_usd")}
    rated = tot["thumbs_up"] + tot["thumbs_down"]
    from app.jobs.scheduler import next_crawl_time

    return render(request, "dashboard.html", admin, by_type=by_type, pending=pending, days=days[:14], tot=tot,
                  unanswered_rate=(100 * tot["unanswered"] / tot["chats"]) if tot["chats"] else None,
                  satisfaction=(100 * tot["thumbs_up"] / rated) if rated else None, last_run=last_run,
                  counts=counts, chunks=vectorstore.count(), llm=get_llm().describe(), paid=s.llm_provider != "ollama",
                  ocr=ocr_available(), crawl=jobs.status(), next_crawl=next_crawl_time())


# ---------------------------------------------------------------- sources
@router.get("/sources")
def sources(request: Request, q: str = "", kind: str = "", status: str = "", page: int = 1,
            admin: Admin = Depends(current_admin)):
    page = max(1, page)
    with session_scope() as db:
        stmt = select(Source)
        if q:
            like = f"%{q.strip()}%"
            stmt = stmt.where(or_(Source.title.ilike(like), Source.url.ilike(like), Source.section.ilike(like)))
        if kind:
            stmt = stmt.where(Source.content_type == kind) if kind != "upload" else stmt.where(Source.source_type == "upload")
        if status:
            stmt = stmt.where(Source.index_status == status)
        total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = db.scalars(stmt.order_by(Source.last_changed_at.desc().nullslast(), Source.id.desc())
                          .offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)).all()
        items = [{"id": r.id, "title": r.title, "url": r.url, "type": r.content_type, "source_type": r.source_type,
                  "status": r.index_status, "chunks": r.chunk_count, "date": r.published_at, "year": r.academic_year,
                  "section": r.section, "ocr": r.ocr_used} for r in rows]
    return render(request, "sources.html", admin, items=items, total=total, page=page,
                  pages=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE), q=q, kind=kind, status=status)


@router.get("/sources/{source_id}")
def source_detail(request: Request, source_id: int, admin: Admin = Depends(current_admin)):
    with session_scope() as db:
        src = db.get(Source, source_id)
        if src is None:
            raise HTTPException(404)
        data = {c.name: getattr(src, c.name) for c in Source.__table__.columns if c.name not in ("raw_text",)}
    return render(request, "source_detail.html", admin, src=data, text=(data["text"] or "")[:30000],
                  truncated=len(data["text"] or "") > 30000)


@router.post("/sources/{source_id}/delete")
def source_delete(source_id: int, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    with session_scope() as db:
        src = db.get(Source, source_id)
        target = src.url if src else str(source_id)
    result = remove_source(source_id)
    audit(admin.username, "delete_source" if result == "deleted" else "block_source", target)
    if result == "deleted":
        return back("/admin/sources", ok="Upload deleted and removed from the chatbot.")
    return back(f"/admin/sources/{source_id}", ok="Removed from the chatbot and blocked from future crawls.")


@router.post("/sources/{source_id}/unblock")
def source_unblock(source_id: int, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    unblock_source(source_id)
    with session_scope() as db:
        src = db.get(Source, source_id)
        url = src.url if src else ""
    audit(admin.username, "unblock_source", url)
    if url.startswith("http"):
        jobs.start_url(url, admin.username)
    return back(f"/admin/sources/{source_id}", ok="Unblocked. It is being indexed again.")


@router.post("/sources/{source_id}/recrawl")
def source_recrawl(source_id: int, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    with session_scope() as db:
        src = db.get(Source, source_id)
        url = src.url if src else ""
    err = jobs.start_url(url, admin.username)
    if err:
        return back(f"/admin/sources/{source_id}", err=err)
    audit(admin.username, "recrawl_url", url)
    return back("/admin/crawl", ok="Fetching this page again.")


# ---------------------------------------------------------------- upload
@router.get("/upload")
def upload_page(request: Request, admin: Admin = Depends(current_admin)):
    return render(request, "upload.html", admin)


@router.post("/upload")
async def upload(file: UploadFile = File(...), title: str = Form(""), csrf_token: str = Form(""),
                 admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    s = get_settings()
    data = await file.read(int(s.upload_max_mb * 1024 * 1024) + 1)
    err, kind = validate_upload(file.filename or "", data)
    if err:
        return back("/admin/upload", err=err)
    source_id = save_upload(file.filename or "upload", data, kind, title, admin.username)
    jobs.start_upload(source_id)
    audit(admin.username, "upload", file.filename or "", f"{len(data) // 1024} KB, source #{source_id}")
    return back(f"/admin/sources/{source_id}", ok="Uploaded. The text is being extracted and indexed.")


# ---------------------------------------------------------------- crawling
@router.get("/crawl")
def crawl_page(request: Request, admin: Admin = Depends(current_admin)):
    from app.jobs.scheduler import next_crawl_time

    with session_scope() as db:
        runs = [{"id": r.id, "trigger": r.trigger, "status": r.status, "started_at": r.started_at,
                 "finished_at": r.finished_at, "stats": r.stats or {}}
                for r in db.scalars(select(CrawlRun).order_by(CrawlRun.id.desc()).limit(10))]
        errors = [{"url": e.url, "error": e.error, "at": e.created_at}
                  for e in db.scalars(select(CrawlError).order_by(CrawlError.id.desc()).limit(40))]
    return render(request, "crawl.html", admin, runs=runs, errors=errors, status=jobs.status(),
                  next_crawl=next_crawl_time())


@router.get("/crawl/status.json")
def crawl_status(admin: Admin = Depends(current_admin)):
    return JSONResponse(jobs.status())


@router.post("/crawl/start")
def crawl_start(csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    if not jobs.start_full_crawl(admin.username):
        return back("/admin/crawl", err="A crawl is already running.")
    audit(admin.username, "crawl_start")
    return back("/admin/crawl", ok="Full re-crawl started. Only changed pages are re-indexed.")


@router.post("/crawl/stop")
def crawl_stop(csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    if jobs.stop_crawl():
        audit(admin.username, "crawl_stop")
        return back("/admin/crawl", ok="Stopping after the current page. Progress so far is kept.")
    return back("/admin/crawl", err="No crawl is running.")


@router.post("/crawl/url")
def crawl_url(url: str = Form(...), csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    err = jobs.start_url(url.strip(), admin.username)
    if err:
        return back("/admin/crawl", err=err)
    audit(admin.username, "add_url", url.strip())
    return back("/admin/crawl", ok="Fetching and indexing that page now.")


# ---------------------------------------------------------------- official answers
@router.get("/answers")
def answers(request: Request, test: str = "", admin: Admin = Depends(current_admin)):
    with session_scope() as db:
        rows = [{"id": r.id, "question": r.question, "alts": len([a for a in (r.alt_questions or "").splitlines() if a.strip()]),
                 "answer": r.answer, "active": r.active, "used": r.times_used, "updated": r.updated_at,
                 "by": r.updated_by} for r in db.scalars(select(VerifiedAnswer).order_by(VerifiedAnswer.updated_at.desc()))]
    match = None
    if test.strip():
        from app.rag.embeddings import embed_query
        from app.rag.verified import verified_index

        m = verified_index.best(embed_query(test.strip()))
        s = get_settings()
        if m:
            match = {"id": m.id, "question": m.question, "score": m.score,
                     "mode": "answered word for word" if m.score >= s.verified_direct_threshold
                     else "given to the AI as its main source" if m.score >= s.verified_context_threshold
                     else "not used (too different)"}
    return render(request, "answers.html", admin, rows=rows, test=test, match=match)


@router.get("/answers/new")
def answer_new(request: Request, question: str = "", from_unanswered: int = 0, admin: Admin = Depends(current_admin)):
    return render(request, "answer_edit.html", admin, row={"id": None, "question": question, "alt_questions": "",
                  "answer": "", "source_url": "", "active": True}, from_unanswered=from_unanswered)


@router.get("/answers/{answer_id}")
def answer_edit(request: Request, answer_id: int, admin: Admin = Depends(current_admin)):
    with session_scope() as db:
        r = db.get(VerifiedAnswer, answer_id)
        if r is None:
            raise HTTPException(404)
        row = {c.name: getattr(r, c.name) for c in VerifiedAnswer.__table__.columns}
    return render(request, "answer_edit.html", admin, row=row, from_unanswered=0)


@router.post("/answers/save")
def answer_save(answer_id: str = Form(""), question: str = Form(...), alt_questions: str = Form(""),
                answer: str = Form(...), source_url: str = Form(""), active: str = Form(""),
                from_unanswered: int = Form(0), csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    question, answer = question.strip(), answer.strip()
    if len(question) < 5 or len(answer) < 2:
        return back("/admin/answers/new", err="Please write both the question and the answer.")
    if source_url.strip() and not source_url.strip().startswith(("https://", "http://")):
        return back("/admin/answers/new", err="The link must start with https://")
    with session_scope() as db:
        row = db.get(VerifiedAnswer, int(answer_id)) if answer_id else None
        if row is None:
            row = VerifiedAnswer(question=question, answer=answer)
            db.add(row)
        row.question, row.answer = question[:2000], answer[:8000]
        row.alt_questions = "\n".join(l.strip() for l in alt_questions.splitlines() if l.strip())[:4000]
        row.source_url = source_url.strip()[:2000] or None
        row.active = bool(active)
        row.updated_by, row.updated_at = admin.username, utcnow()
        if from_unanswered:
            db.execute(delete(UnansweredQuestion).where(UnansweredQuestion.id == from_unanswered))
        db.flush()
        rid = row.id
    audit(admin.username, "edit_answer" if answer_id else "add_answer", question[:200], f"answer #{rid}")
    return back("/admin/answers", ok="Official answer saved. The chatbot uses it immediately.")


@router.post("/answers/{answer_id}/delete")
def answer_delete(answer_id: int, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    with session_scope() as db:
        row = db.get(VerifiedAnswer, answer_id)
        q = row.question if row else ""
        if row:
            db.delete(row)
    audit(admin.username, "delete_answer", q[:200])
    return back("/admin/answers", ok="Official answer deleted.")


# ---------------------------------------------------------------- unanswered + feedback
@router.get("/unanswered")
def unanswered(request: Request, admin: Admin = Depends(current_admin)):
    with session_scope() as db:
        rows = [{"id": r.id, "question": r.question, "reason": r.reason, "at": r.created_at}
                for r in db.scalars(select(UnansweredQuestion).order_by(UnansweredQuestion.id.desc()).limit(500))]
    return render(request, "unanswered.html", admin, rows=rows)


@router.post("/unanswered/{row_id}/delete")
def unanswered_delete(row_id: int, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    with session_scope() as db:
        db.execute(delete(UnansweredQuestion).where(UnansweredQuestion.id == row_id))
    return back("/admin/unanswered")


@router.post("/unanswered/clear")
def unanswered_clear(csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    with session_scope() as db:
        n = db.execute(delete(UnansweredQuestion)).rowcount
    audit(admin.username, "clear_unanswered", details=f"{n} questions")
    return back("/admin/unanswered", ok=f"Cleared {n} questions.")


@router.get("/feedback")
def feedback_page(request: Request, admin: Admin = Depends(current_admin)):
    with session_scope() as db:
        rows = [{"id": r.id, "question": r.question, "sources": [u for u in r.sources.splitlines() if u],
                 "at": r.created_at}
                for r in db.scalars(select(NegativeFeedback).order_by(NegativeFeedback.id.desc()).limit(500))]
    return render(request, "feedback.html", admin, rows=rows)


@router.post("/feedback/{row_id}/delete")
def feedback_delete(row_id: int, csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    with session_scope() as db:
        db.execute(delete(NegativeFeedback).where(NegativeFeedback.id == row_id))
    return back("/admin/feedback")


# ---------------------------------------------------------------- users (super admin)
@router.get("/users")
def users(request: Request, admin: Admin = Depends(super_admin)):
    with session_scope() as db:
        rows = [{"id": u.id, "username": u.username, "role": u.role, "active": u.active, "twofa": bool(u.totp_secret),
                 "last_login": u.last_login_at, "locked": u.locked_until}
                for u in db.scalars(select(AdminUser).order_by(AdminUser.username))]
    return render(request, "users.html", admin, rows=rows)


@router.post("/users/add")
def user_add(username: str = Form(...), role: str = Form("editor"), password: str = Form(...),
             csrf_token: str = Form(""), admin: Admin = Depends(super_admin)):
    check_csrf(admin, csrf_token)
    username = username.strip().lower()
    if not username.replace(".", "").replace("_", "").replace("-", "").isalnum() or not 3 <= len(username) <= 64:
        return back("/admin/users", err="Usernames use 3–64 letters, numbers, dots, dashes or underscores.")
    problem = security.password_problem(password)
    if problem:
        return back("/admin/users", err=problem)
    with session_scope() as db:
        if db.scalar(select(AdminUser).where(AdminUser.username == username)):
            return back("/admin/users", err="That username already exists.")
        db.add(AdminUser(username=username, role="super" if role == "super" else "editor",
                         password_hash=security.hash_password(password)))
    audit(admin.username, "add_user", username, role)
    return back("/admin/users", ok=f"Added {username}.")


@router.post("/users/{user_id}/toggle")
def user_toggle(user_id: int, csrf_token: str = Form(""), admin: Admin = Depends(super_admin)):
    check_csrf(admin, csrf_token)
    if user_id == admin.user_id:
        return back("/admin/users", err="You cannot deactivate your own account.")
    with session_scope() as db:
        u = db.get(AdminUser, user_id)
        if u is None:
            raise HTTPException(404)
        u.active = not u.active
        name, active = u.username, u.active
    if not active:
        security.destroy_user_sessions(user_id)
    audit(admin.username, "activate_user" if active else "deactivate_user", name)
    return back("/admin/users", ok=f"{name} is now {'active' if active else 'deactivated'}.")


@router.post("/users/{user_id}/password")
def user_reset_password(user_id: int, password: str = Form(...), csrf_token: str = Form(""),
                        admin: Admin = Depends(super_admin)):
    check_csrf(admin, csrf_token)
    problem = security.password_problem(password)
    if problem:
        return back("/admin/users", err=problem)
    with session_scope() as db:
        u = db.get(AdminUser, user_id)
        if u is None:
            raise HTTPException(404)
        u.password_hash, u.failed_logins, u.locked_until = security.hash_password(password), 0, None
        name = u.username
    security.destroy_user_sessions(user_id)
    audit(admin.username, "reset_password", name)
    return back("/admin/users", ok=f"Password reset for {name}.")


# ---------------------------------------------------------------- own account + 2FA
def _qr_data_uri(uri: str) -> str:
    import qrcode
    import qrcode.image.svg

    buf = io.BytesIO()
    qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=8).save(buf)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


@router.get("/account")
def account(request: Request, admin: Admin = Depends(current_admin)):
    ctx = {}
    if not admin.has_2fa:
        import pyotp

        secret = pyotp.random_base32()
        uri = pyotp.TOTP(secret).provisioning_uri(name=admin.username, issuer_name=get_settings().app_name + " admin")
        ctx = {"secret": secret, "qr": _qr_data_uri(uri)}
    return render(request, "account.html", admin, **ctx)


@router.post("/account/password")
def change_password(request: Request, current: str = Form(...), new: str = Form(...), csrf_token: str = Form(""),
                    admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    if security.authenticate(admin.username, current).user_id is None:
        return back("/admin/account", err="Your current password is not correct.")
    problem = security.password_problem(new)
    if problem:
        return back("/admin/account", err=problem)
    with session_scope() as db:
        db.get(AdminUser, admin.user_id).password_hash = security.hash_password(new)
    security.destroy_user_sessions(admin.user_id, except_token=request.cookies.get(COOKIE_NAME))
    audit(admin.username, "change_password")
    return back("/admin/account", ok="Password changed. Other sessions were signed out.")


@router.post("/account/2fa/enable")
def enable_2fa(secret: str = Form(...), code: str = Form(...), csrf_token: str = Form(""),
               admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    import pyotp

    if not pyotp.TOTP(secret).verify(code.replace(" ", ""), valid_window=1):
        return back("/admin/account", err="That code did not match. Scan the new QR code and try again.")
    with session_scope() as db:
        db.get(AdminUser, admin.user_id).totp_secret = secret
    audit(admin.username, "enable_2fa")
    return back("/admin/account", ok="Two-step login is on. You'll need the app code at every login.")


@router.post("/account/2fa/disable")
def disable_2fa(password: str = Form(...), csrf_token: str = Form(""), admin: Admin = Depends(current_admin)):
    check_csrf(admin, csrf_token)
    if security.authenticate(admin.username, password).user_id is None:
        return back("/admin/account", err="Your password is not correct.")
    with session_scope() as db:
        db.get(AdminUser, admin.user_id).totp_secret = None
    audit(admin.username, "disable_2fa")
    return back("/admin/account", ok="Two-step login is off.")


# ---------------------------------------------------------------- audit log
@router.get("/audit")
def audit_page(request: Request, action: str = "", admin: Admin = Depends(current_admin)):
    with session_scope() as db:
        stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(300)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        rows = [{"at": r.created_at, "user": r.username, "action": r.action, "target": r.target, "details": r.details}
                for r in db.scalars(stmt)]
        actions = [a for (a,) in db.execute(select(AuditLog.action).distinct().order_by(AuditLog.action))]
    return render(request, "audit.html", admin, rows=rows, actions=actions, action=action)
