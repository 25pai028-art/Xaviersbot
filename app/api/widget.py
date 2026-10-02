"""Settings the embeddable chat widget reads at start-up: `GET /api/widget/config`."""
from __future__ import annotations

from fastapi import APIRouter, Response

from app.config import get_settings
from app.i18n.languages import LANGUAGES
from app.i18n.messages import WELCOME
from app.i18n.translate import TranslationError, translate
from app.i18n.translate import provider as translation_provider
from app.rag.prompts import SMALL_TALK_MESSAGES

router = APIRouter(prefix="/api/widget", tags=["widget"])

# Quick-question chips: the label is shown, the question is what gets asked.
CHIPS = [
    {"label": "Admissions", "question": "How do I apply for admission at St. Xavier's College?"},
    {"label": "Courses", "question": "Which courses does the college offer?"},
    {"label": "Fees", "question": "What is the fee structure?"},
    {"label": "Exams", "question": "Where can I find the examination schedule and results?"},
    {"label": "Hostel", "question": "Does the college have hostel accommodation?"},
    {"label": "Faculty", "question": "How can I find the faculty members of a department and their contact details?"},
    {"label": "Contact", "question": "How can I contact the college office?"},
    {"label": "Book a Meeting", "action": "booking"},
]
# Chip questions are asked in English; the answer comes in the language chosen in the selector.

# Shown for the "Book a Meeting" chip until booking is built (Phase 6).
BOOKING_MESSAGE = (
    "Booking a meeting with a faculty member from this chat is coming soon. For now, you can ask me for a "
    "faculty member's email (for example, \"email of the Head of the Data Science department\") and write to them, "
    "or contact the college office."
)


def _welcome() -> str:
    s = get_settings()
    return s.widget_welcome or SMALL_TALK_MESSAGES["greeting"].format(bot_name=s.app_name)


@router.get("/welcome")
async def welcome(response: Response, lang: str = "en") -> dict:
    """The welcome message in the chosen language (translated once, then cached)."""
    lang = lang if lang in LANGUAGES else "en"
    if not get_settings().widget_welcome and lang in WELCOME:  # the standard greeting: hand-written per language
        text = WELCOME[lang]
    else:
        try:
            text = await translate(_welcome(), "en", lang)
        except TranslationError:
            text, lang = _welcome(), "en"
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"language": lang, "text": text}


@router.get("/config")
def widget_config(response: Response) -> dict:
    s = get_settings()
    response.headers["Cache-Control"] = "public, max-age=300"
    return {
        "bot_name": s.app_name,
        "college_name": "St. Xavier's College (Autonomous), Ahmedabad",
        "welcome": _welcome(),
        # A path is resolved by the widget against this server (correct behind a proxy / HTTPS too).
        "logo_url": s.widget_logo_url or "/static/widget/crest.png",
        "languages": [l for l in s.widget_languages if l in LANGUAGES],
        "translation": translation_provider(),
        "chips": CHIPS,
        "booking_message": BOOKING_MESSAGE,
        "office": {"url": s.college_office_url, "email": s.college_office_email, "phone": s.college_office_phone},
        "max_chars": s.max_message_chars,
    }
