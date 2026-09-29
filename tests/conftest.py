"""Tests use a throw-away data directory so they never touch the real knowledge base."""
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="sxca-test-")
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["ENVIRONMENT"] = "test"
os.environ["HF_HUB_OFFLINE"] = "1"  # never contact Hugging Face from tests (models come from the local cache)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
