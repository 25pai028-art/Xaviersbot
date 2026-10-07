"""Deploy (or update) the chatbot on a Hugging Face Docker Space, for a pilot / testing link.

    python -m scripts.deploy_space                 # knowledge base + code + settings
    python -m scripts.deploy_space --code-only     # code changes only (keeps the live database on the Space)

Reads from .env: HF_TOKEN (write), OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL / CHECK_MODEL (Groq).
Creates, under the token's account:
- a PRIVATE dataset  <account>/xaviers-assistant-data  with the knowledge base (DATA_DIR: sxca.db + chroma/);
- a PUBLIC Docker Space <account>/xaviers-assistant  running scripts/space_start.py.
Keys go into the Space's Secrets (hidden); nothing secret is uploaded with the code.
Uploading the knowledge base replaces the database on the dataset, including feedback and admin accounts
backed up from the Space since the last deploy: use --code-only for code changes.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPACE_README = """---
title: Xavier's Assistant
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Pilot of the St. Xavier's College Ahmedabad chatbot
---

# Xavier's Assistant (pilot)

Answers questions about St. Xavier's College (Autonomous), Ahmedabad from the college website.
This is a test version: please use 👍 / 👎 under an answer to flag anything wrong.
"""


def read_env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-only", action="store_true", help="upload code only; keep the Space's live database")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/colab"))
    args = ap.parse_args()

    from huggingface_hub import HfApi

    env = read_env()
    token = env.get("HF_TOKEN")
    if not token:
        sys.exit("Add HF_TOKEN=<a write token> to .env first (huggingface.co/settings/tokens).")
    api = HfApi(token=token)
    me = api.whoami()
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    if role not in ("write", "fineGrained"):
        sys.exit(f"The HF_TOKEN in .env is a '{role}' token; a 'write' token is needed.")
    account = me["name"]
    data_repo, space = f"{account}/xaviers-assistant-data", f"{account}/xaviers-assistant"
    print(f"Account {account}: dataset {data_repo} (private), Space {space} (public)")

    # 1. Knowledge base -> private dataset
    if not args.code_only:
        data_dir = (ROOT / args.data_dir).resolve()
        api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            src, dst = sqlite3.connect(data_dir / "sxca.db"), sqlite3.connect(stage / "sxca.db")
            src.backup(dst)  # consistent copy even while the local chatbot is running
            dst.close()
            src.close()
            shutil.copytree(data_dir / "chroma", stage / "chroma")
            print("Uploading the knowledge base (a few minutes)…")
            api.upload_folder(folder_path=str(stage), repo_id=data_repo, repo_type="dataset",
                              commit_message="Knowledge base")

    # 2. Space settings: secrets are hidden; variables are visible to the Space owner only in settings
    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(space, "OPENAI_API_KEY", env["OPENAI_API_KEY"])
    api.add_space_secret(space, "HF_TOKEN", token)
    for key, value in {"DATA_REPO": data_repo, "LLM_PROVIDER": env.get("LLM_PROVIDER", "openai"),
                       "OPENAI_BASE_URL": env.get("OPENAI_BASE_URL", ""), "OPENAI_MODEL": env.get("OPENAI_MODEL", ""),
                       "CHECK_MODEL": env.get("CHECK_MODEL", "")}.items():
        api.add_space_variable(space, key, value)

    # 3. Code -> Space (no data, no .env, no tests)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(ROOT / "app", stage / "app", ignore=ignore)
        shutil.copytree(ROOT / "scripts", stage / "scripts", ignore=ignore)
        for f in ("Dockerfile", "requirements.txt"):
            shutil.copy2(ROOT / f, stage / f)
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        print("Uploading the code…")
        api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space", commit_message="Deploy",
                          delete_patterns=["app/**", "scripts/**"])
    print(f"\nDone. Building now (about 10 minutes the first time):\n  https://huggingface.co/spaces/{space}\n"
          f"Link for testers:\n  https://{space.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
