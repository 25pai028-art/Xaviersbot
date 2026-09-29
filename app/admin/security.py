"""Admin authentication and protection.

- Passwords: Argon2id (argon2-cffi), minimum length, rehash on login when parameters change.
- Sessions: random token in an HttpOnly, SameSite=Strict cookie (Secure in production);
  only its SHA-256 is stored. Idle timeout + absolute lifetime.
- Brute force: per-account lockout after N failures, plus a per-IP limit (in memory).
- CSRF: per-session token required on every state-changing request.
- Optional TOTP two-factor login (authenticator apps).
- Audit log of every change an admin makes.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Request
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.models import AdminSession, AdminUser, AuditLog, as_utc, utcnow
from app.db.session import session_scope

COOKIE_NAME = "sxca_admin"
_hasher = PasswordHasher()
# A real hash so unknown usernames take as long as wrong passwords (no username probing).
_DUMMY_HASH = _hasher.hash("not-a-real-password-" + secrets.token_hex(8))


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    return _hasher.hash(password)


def password_problem(password: str) -> str | None:
    """Why a new password is not acceptable, or None."""
    if len(password) < 10:
        return "Use at least 10 characters."
    if password.isdigit() or password.isalpha():
        return "Mix letters with numbers or symbols."
    if password.lower() in {"password123", "sxca@12345", "admin@12345", "xavier@1234"}:
        return "That password is too common."
    return None


# ---------------------------------------------------------------- per-IP limit (in memory)
_ip_attempts: dict[str, list[float]] = {}
_ip_lock = threading.Lock()
IP_MAX_ATTEMPTS = 20
IP_WINDOW_SECONDS = 15 * 60


def client_key(request: Request) -> str:
    """Anonymised client id for rate limiting (IP address is never stored)."""
    ip = request.client.host if request.client else "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def ip_blocked(key: str) -> bool:
    now = time.time()
    with _ip_lock:
        hits = [t for t in _ip_attempts.get(key, []) if now - t < IP_WINDOW_SECONDS]
        _ip_attempts[key] = hits
        return len(hits) >= IP_MAX_ATTEMPTS


def record_ip_failure(key: str) -> None:
    with _ip_lock:
        _ip_attempts.setdefault(key, []).append(time.time())


# ---------------------------------------------------------------- login
@dataclass
class LoginResult:
    user_id: int | None = None
    needs_2fa: bool = False
    error: str | None = None


GENERIC_LOGIN_ERROR = "Wrong username or password."


def authenticate(username: str, password: str) -> LoginResult:
    s = get_settings()
    with session_scope() as db:
        user = db.scalar(select(AdminUser).where(AdminUser.username == username.strip().lower()))
        if user is None or not user.active:
            try:
                _hasher.verify(_DUMMY_HASH, password)
            except VerificationError:
                pass
            return LoginResult(error=GENERIC_LOGIN_ERROR)
        locked = as_utc(user.locked_until)
        if locked and locked > utcnow():
            minutes = max(1, int((locked - utcnow()).total_seconds() // 60) + 1)
            return LoginResult(error=f"Too many failed attempts. Try again in {minutes} minutes.")
        try:
            _hasher.verify(user.password_hash, password)
        except (VerificationError, InvalidHashError):
            user.failed_logins += 1
            if user.failed_logins >= s.admin_max_failed_logins:
                user.locked_until = utcnow() + timedelta(minutes=s.admin_lockout_minutes)
                user.failed_logins = 0
                db.add(AuditLog(username=user.username, action="account_locked",
                                details=f"{s.admin_max_failed_logins} failed logins"))
            return LoginResult(error=GENERIC_LOGIN_ERROR)
        if _hasher.check_needs_rehash(user.password_hash):
            user.password_hash = _hasher.hash(password)
        user.failed_logins = 0
        user.locked_until = None
        return LoginResult(user_id=user.id, needs_2fa=bool(user.totp_secret))


def verify_totp(user_id: int, code: str) -> bool:
    import pyotp

    with session_scope() as db:
        user = db.get(AdminUser, user_id)
        if not user or not user.totp_secret:
            return False
        return pyotp.TOTP(user.totp_secret).verify(code.replace(" ", ""), valid_window=1)


# ---------------------------------------------------------------- sessions
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int, pending_2fa: bool = False) -> str:
    token = secrets.token_urlsafe(32)
    with session_scope() as db:
        db.add(AdminSession(token_hash=_hash_token(token), user_id=user_id, csrf_token=secrets.token_urlsafe(32),
                            pending_2fa=pending_2fa))
        if not pending_2fa:
            user = db.get(AdminUser, user_id)
            user.last_login_at = utcnow()
    return token


def complete_2fa(token: str) -> None:
    with session_scope() as db:
        sess = db.scalar(select(AdminSession).where(AdminSession.token_hash == _hash_token(token)))
        if sess:
            sess.pending_2fa = False
            db.get(AdminUser, sess.user_id).last_login_at = utcnow()


def destroy_session(token: str | None) -> None:
    if token:
        with session_scope() as db:
            db.execute(delete(AdminSession).where(AdminSession.token_hash == _hash_token(token)))


def destroy_user_sessions(user_id: int, except_token: str | None = None) -> None:
    keep = _hash_token(except_token) if except_token else ""
    with session_scope() as db:
        db.execute(delete(AdminSession).where(AdminSession.user_id == user_id, AdminSession.token_hash != keep))


@dataclass
class Admin:
    user_id: int
    username: str
    role: str
    csrf_token: str
    has_2fa: bool

    @property
    def is_super(self) -> bool:
        return self.role == "super"


def load_session(token: str | None, allow_pending_2fa: bool = False) -> Admin | None:
    if not token:
        return None
    s = get_settings()
    now = utcnow()
    with session_scope() as db:
        sess = db.scalar(select(AdminSession).where(AdminSession.token_hash == _hash_token(token)))
        if sess is None:
            return None
        idle = now - as_utc(sess.last_seen_at) > timedelta(minutes=s.admin_session_idle_minutes)
        too_old = now - as_utc(sess.created_at) > timedelta(hours=s.admin_session_max_hours)
        user = db.get(AdminUser, sess.user_id)
        if idle or too_old or user is None or not user.active:
            db.delete(sess)
            return None
        if sess.pending_2fa and not allow_pending_2fa:
            return None
        sess.last_seen_at = now
        return Admin(user_id=user.id, username=user.username, role=user.role, csrf_token=sess.csrf_token,
                     has_2fa=bool(user.totp_secret))


def csrf_ok(admin: Admin, submitted: str | None) -> bool:
    return bool(submitted) and hmac.compare_digest(admin.csrf_token, submitted)


def cleanup_sessions() -> int:
    s = get_settings()
    cutoff = utcnow() - timedelta(hours=s.admin_session_max_hours)
    with session_scope() as db:
        return db.execute(delete(AdminSession).where(AdminSession.created_at < cutoff)).rowcount or 0


# ---------------------------------------------------------------- audit
def audit(username: str, action: str, target: str = "", details: str = "") -> None:
    with session_scope() as db:
        db.add(AuditLog(username=username, action=action, target=target[:1000], details=details[:4000]))
