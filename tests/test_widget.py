"""The embeddable chat widget: script, config endpoint and demo page."""
from fastapi.testclient import TestClient

from app.main import app


def client() -> TestClient:
    return TestClient(app, follow_redirects=False)  # no `with`: lifespan (model warm-up, scheduler) not started


def test_widget_script_is_served_as_javascript():
    r = client().get("/widget.js")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/javascript")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "attachShadow" in r.text  # styles isolated from the college site
    assert ".innerHTML = text" not in r.text and "innerHTML = m.text" not in r.text  # answers never go in as HTML


def test_widget_config_has_branding_and_chips():
    r = client().get("/api/widget/config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["bot_name"] == "Xavier's Assistant"
    assert cfg["logo_url"] == "/static/widget/crest.png"
    labels = [c["label"] for c in cfg["chips"]]
    for want in ("Admissions", "Courses", "Fees", "Exams", "Hostel", "Contact", "Book a Meeting", "Faculty"):
        assert want in labels
    assert all("question" in c or c.get("action") == "booking" for c in cfg["chips"])
    assert cfg["office"]["url"].startswith("https://sxca.edu.in/")
    assert cfg["languages"] == ["en"]
    assert "Xavier's Assistant" in cfg["welcome"]


def test_crest_and_demo_page():
    c = client()
    assert c.get("/static/widget/crest.png").headers["content-type"] == "image/png"
    demo = c.get("/demo")
    assert demo.status_code == 200 and '<script src="/widget.js" defer></script>' in demo.text
    assert c.get("/").headers["location"] == "/demo"
