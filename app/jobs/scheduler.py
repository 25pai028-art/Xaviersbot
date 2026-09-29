"""Background schedule: weekly incremental re-crawl and nightly privacy cleanup.

Runs inside the web server process (start the server with ONE worker, otherwise
each worker would schedule its own crawl).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import delete

from app.config import get_settings
from app.db.models import NegativeFeedback, UnansweredQuestion, utcnow
from app.db.session import session_scope

log = logging.getLogger(__name__)
TIMEZONE = "Asia/Kolkata"
_scheduler = None


def scheduled_crawl() -> None:
    from app.admin.jobs import jobs

    if jobs.start_full_crawl("scheduler"):
        log.info("Scheduled re-crawl started")
    else:
        log.info("Scheduled re-crawl skipped: a crawl is already running")


def cleanup() -> None:
    """Delete anonymous question logs older than RETENTION_DAYS, and expired admin sessions."""
    from app.admin.security import cleanup_sessions

    cutoff = utcnow() - timedelta(days=get_settings().retention_days)
    with session_scope() as db:
        n1 = db.execute(delete(UnansweredQuestion).where(UnansweredQuestion.created_at < cutoff)).rowcount
        n2 = db.execute(delete(NegativeFeedback).where(NegativeFeedback.created_at < cutoff)).rowcount
    n3 = cleanup_sessions()
    log.info("Cleanup: %s unanswered, %s feedback, %s sessions removed", n1, n2, n3)


def start() -> None:
    global _scheduler
    s = get_settings()
    if not s.scheduler_enabled or _scheduler is not None:
        return
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    _scheduler = BackgroundScheduler(timezone=TIMEZONE)
    _scheduler.add_job(scheduled_crawl, CronTrigger(day_of_week=s.crawl_schedule_day, hour=s.crawl_schedule_hour,
                                                    minute=0, timezone=TIMEZONE), id="crawl", max_instances=1,
                       coalesce=True, misfire_grace_time=3600)
    _scheduler.add_job(cleanup, CronTrigger(hour=3, minute=30, timezone=TIMEZONE), id="cleanup", coalesce=True)
    _scheduler.start()
    log.info("Scheduler started: re-crawl %s at %02d:00, cleanup daily 03:30 (%s)", s.crawl_schedule_day,
             s.crawl_schedule_hour, TIMEZONE)


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def next_crawl_time() -> datetime | None:
    if _scheduler is None:
        return None
    job = _scheduler.get_job("crawl")
    return job.next_run_time if job else None
