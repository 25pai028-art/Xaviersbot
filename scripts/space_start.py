"""Start the chatbot on a Hugging Face Space (Docker).

A Space's disk is wiped on every restart, so:
1. at start, the knowledge base (ChromaDB + SQLite) is downloaded from a private dataset (DATA_REPO);
2. while running, the SQLite database (admin accounts, official answers, feedback, unanswered questions) is
   copied back to that dataset every BACKUP_MINUTES and when the Space stops, so nothing is lost.

Space settings: secrets OPENAI_API_KEY (Groq) and HF_TOKEN (write access to DATA_REPO); variables DATA_REPO,
LLM_PROVIDER, OPENAI_BASE_URL, OPENAI_MODEL, CHECK_MODEL. See scripts/deploy_space.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
import signal
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("space")
DATA_DIR = Path(os.environ.setdefault("DATA_DIR", "/home/user/data"))
DATA_REPO = os.environ.get("DATA_REPO", "")
TOKEN = os.environ.get("HF_TOKEN") or None
BACKUP_MINUTES = float(os.environ.get("BACKUP_MINUTES", "10"))
_last_hash = ""
_backup_lock = threading.Lock()


def download_knowledge_base() -> None:
    from huggingface_hub import snapshot_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading the knowledge base from %s …", DATA_REPO)
    snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=DATA_DIR, token=TOKEN)
    log.info("Knowledge base ready in %s", DATA_DIR)


def backup_database(reason: str) -> None:
    """Copy the live SQLite database to the dataset (only when it changed). Safe while the app is running:
    SQLite's backup API takes a consistent copy."""
    global _last_hash
    db = DATA_DIR / "sxca.db"
    if not (DATA_REPO and TOKEN and db.exists()):
        return
    with _backup_lock:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                copy = Path(tmp) / "sxca.db"
                src, dst = sqlite3.connect(db), sqlite3.connect(copy)
                src.backup(dst)
                dst.close()
                src.close()
                digest = hashlib.sha256(copy.read_bytes()).hexdigest()
                if digest == _last_hash:
                    return
                from huggingface_hub import HfApi

                HfApi(token=TOKEN).upload_file(path_or_fileobj=str(copy), path_in_repo="sxca.db", repo_id=DATA_REPO,
                                               repo_type="dataset", commit_message=f"Database backup ({reason})")
                _last_hash = digest
                log.info("Database backed up to %s (%s)", DATA_REPO, reason)
        except Exception as e:  # a failed backup must never stop the chatbot
            log.warning("Database backup failed: %s", e)


def _backup_loop(stop: threading.Event) -> None:
    while not stop.wait(BACKUP_MINUTES * 60):
        backup_database("every %g min" % BACKUP_MINUTES)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    global _last_hash
    if DATA_REPO:
        download_knowledge_base()
        db = DATA_DIR / "sxca.db"
        _last_hash = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else ""
    else:
        log.warning("DATA_REPO is not set: starting with an empty knowledge base")

    stop = threading.Event()
    threading.Thread(target=_backup_loop, args=(stop,), daemon=True).start()

    def on_stop(signum, _frame):  # the Space is stopping or restarting: save first
        stop.set()
        backup_database("shutdown")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_stop)

    import uvicorn

    # Behind Hugging Face's proxy: take the visitor's address from X-Forwarded-For, so the per-visitor question
    # limit applies to each visitor and not to the proxy (which would make everyone share one limit).
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")),
                proxy_headers=True, forwarded_allow_ips="*", log_level="info")
    stop.set()
    backup_database("exit")


if __name__ == "__main__":
    main()
